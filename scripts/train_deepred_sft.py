#!/usr/bin/env python3
"""Train a full-weight conditioned SFT model on era-native and persona targets."""

import argparse
import gc
import json
import os
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

try:
    from train_deepred_npo import (
        TrainingError, load_jsonl, parse_snapshots, resolve_resume,
        tokenize_messages,
    )
except ModuleNotFoundError:
    from scripts.train_deepred_npo import (
        TrainingError, load_jsonl, parse_snapshots, resolve_resume,
        tokenize_messages,
    )


os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')


def shuffled_rows(rows, seed):
    if not rows:
        raise TrainingError('dataset split is empty')
    ordered = sorted(rows, key=lambda row: row['id'])
    random.Random(seed).shuffle(ordered)
    return ordered


def make_collator(tokenizer):
    import torch

    def collate(features):
        maximum = max(len(row['input_ids']) for row in features)
        input_ids, labels, attention = [], [], []
        for row in features:
            ids = row['input_ids']
            padding = maximum - len(ids)
            input_ids.append(ids + [tokenizer.pad_token_id] * padding)
            labels.append(row['labels'] + [-100] * padding)
            attention.append([1] * len(ids) + [0] * padding)
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'attention_mask': torch.tensor(attention, dtype=torch.long),
        }
    return collate


def train(args):
    if sys.executable.startswith('/mnt/data/venv/'):
        raise TrainingError(
            'the host /mnt/data/venv is active; inside the finetuning '
            'container run this script with /opt/venv/bin/python3')

    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback,
        TrainingArguments,
    )

    if not torch.cuda.is_available():
        raise TrainingError('a ROCm/CUDA device is required')
    dataset_dir = Path(args.dataset)
    paths = {name: dataset_dir / f'{name}.jsonl'
             for name in ('retain_train', 'retain_val')}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise TrainingError(f'missing dataset files: {", ".join(missing)}')
    for split in ('train', 'val'):
        forget_path = dataset_dir / f'forget_{split}.jsonl'
        if forget_path.is_file() and load_jsonl(forget_path):
            raise TrainingError(
                f'{forget_path} is not empty; plain SFT would teach post-1969 '
                'facts. Rebuild the dataset with --limit forget=0')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    snapshots = parse_snapshots(args.snapshot_at)
    resume = resolve_resume(output, args.resume)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_rows = shuffled_rows(load_jsonl(paths['retain_train']), args.seed)
    val_rows = shuffled_rows(load_jsonl(paths['retain_val']), args.seed + 1)
    kind_counts = Counter(row.get('kind', 'unknown') for row in train_rows)
    conditioned = sum(1 for row in train_rows if row.get('system_variant'))
    print(f'train rows: {len(train_rows):,}  val rows: {len(val_rows):,}')
    print(f'kinds: {dict(sorted(kind_counts.items()))}')
    print(f'system-conditioned rows: {conditioned:,}/{len(train_rows):,}')

    def prepare(rows):
        return Dataset.from_list([
            tokenize_messages(tokenizer, row['messages'], args.max_length)
            for row in rows])

    train_dataset = prepare(train_rows)
    val_dataset = prepare(val_rows)

    model_source = resume or args.model
    print(f'loading trainable full-weight model from {model_source}...')
    model = AutoModelForCausalLM.from_pretrained(
        model_source, dtype=torch.bfloat16, attn_implementation='eager',
        trust_remote_code=True)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={'use_reentrant': False})
    model.config.use_cache = False

    class SnapshotCallback(TrainerCallback):
        def __init__(self):
            self.saved = {
                percentage for percentage in snapshots
                if any((output / 'snapshots').glob(
                    f'{int(percentage):03d}pct-step-*'))
            }

        def on_step_end(self, training_args, state, control, model=None, **kwargs):
            if not state.max_steps:
                return control
            progress = 100 * state.global_step / state.max_steps
            for percentage in snapshots:
                if percentage in self.saved or progress + 1e-9 < percentage:
                    continue
                path = output / 'snapshots' / (
                    f'{int(percentage):03d}pct-step-{state.global_step}')
                path.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(path, safe_serialization=True)
                tokenizer.save_pretrained(path)
                self.saved.add(percentage)
                print(f'saved snapshot {percentage:g}% -> {path}', flush=True)
            return control

    training_args = TrainingArguments(
        output_dir=str(output), max_steps=args.max_steps,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate, adam_beta1=0.9, adam_beta2=0.95,
        warmup_ratio=args.warmup_ratio, lr_scheduler_type='cosine',
        optim='adamw_torch_fused', bf16=True, fp16=False,
        gradient_checkpointing=True, eval_strategy='steps',
        eval_steps=args.eval_steps, save_strategy='steps',
        save_steps=args.save_steps, save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps, report_to='none',
        remove_unused_columns=False, prediction_loss_only=True, seed=args.seed,
    )
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_dataset,
        eval_dataset=val_dataset, data_collator=make_collator(tokenizer),
        callbacks=[SnapshotCallback()])
    metadata = {
        'schema_version': 1, 'status': 'running',
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'initial_model': args.model, 'tokenizer': args.tokenizer,
        'dataset': str(dataset_dir), 'objective': 'conditioned_sft',
        'learning_rate': args.learning_rate, 'max_steps': args.max_steps,
        'epochs': args.epochs,
        'gradient_accumulation': args.gradient_accumulation,
        'train_rows': len(train_rows), 'val_rows': len(val_rows),
        'kind_counts': dict(sorted(kind_counts.items())),
        'system_conditioned_rows': conditioned,
        'snapshots': snapshots,
    }
    (output / 'run_meta.json').write_text(json.dumps(metadata, indent=2) + '\n')
    trainer.train(resume_from_checkpoint=resume or None)
    final = output / 'final'
    trainer.save_model(final)
    tokenizer.save_pretrained(final)
    metadata.update({
        'status': 'completed',
        'completed_utc': datetime.now(timezone.utc).isoformat(),
        'global_step': trainer.state.global_step,
    })
    (output / 'run_meta.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'training complete -> {final}')
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--learning-rate', type=float, default=5e-6)
    parser.add_argument('--max-steps', type=int, default=-1)
    parser.add_argument('--epochs', type=float, default=2.0)
    parser.add_argument('--gradient-accumulation', type=int, default=16)
    parser.add_argument('--warmup-ratio', type=float, default=0.03)
    parser.add_argument('--max-length', type=int, default=768)
    parser.add_argument('--logging-steps', type=int, default=10)
    parser.add_argument('--eval-steps', type=int, default=100)
    parser.add_argument('--save-steps', type=int, default=100)
    parser.add_argument('--save-total-limit', type=int, default=2)
    parser.add_argument('--snapshot-at', nargs='+',
                        default=['10', '25', '50', '75', '100'])
    parser.add_argument('--seed', type=int, default=1969)
    parser.add_argument('--resume', default='auto')
    return parser


def main(argv=None):
    try:
        args = build_parser().parse_args(argv)
        if args.learning_rate <= 0 or args.gradient_accumulation <= 0:
            raise TrainingError(
                'learning rate and gradient accumulation must be positive')
        if args.max_steps <= 0 and args.epochs <= 0:
            raise TrainingError('set a positive --max-steps or --epochs')
        train(args)
    except TrainingError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
