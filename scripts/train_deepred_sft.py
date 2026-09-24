#!/usr/bin/env python3
"""Train a full-weight conditioned SFT model on era-native and persona targets."""

import argparse
import gc
import inspect
import json
import math
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

# Verified against gemma-4-12b-it: these suffixes match the 48 language_model
# layers and nothing in the vision or audio towers, which use other names.
LORA_TARGET_MODULES = ('q_proj', 'k_proj', 'v_proj', 'o_proj',
                       'gate_proj', 'up_proj', 'down_proj')


def load_trainable_model(source, args):
    """Load the model to train, returning it with a record of how.

    Separate from train() so probe_train_memory.py can walk the 12B rungs
    without a second copy of the trainer.
    """
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    kwargs = {
        'dtype': torch.bfloat16,
        'attn_implementation': args.attn_implementation,
        'trust_remote_code': True,
    }
    if args.tuning == 'qlora':
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = AutoModelForCausalLM.from_pretrained(source, **kwargs)
    info = {
        'tuning': args.tuning,
        'model_class': type(model).__name__,
        'attn_implementation': args.attn_implementation,
    }
    if args.tuning == 'full':
        return model, info

    from peft import (
        LoraConfig, get_peft_model, prepare_model_for_kbit_training,
    )
    if args.tuning == 'qlora':
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, bias='none', task_type='CAUSAL_LM',
        target_modules=list(args.lora_target_modules)))
    trainable, total = model.get_nb_trainable_parameters()
    info.update({
        'lora_r': args.lora_r,
        'lora_alpha': args.lora_alpha,
        'lora_target_modules': list(args.lora_target_modules),
        'trainable_parameters': trainable,
        'total_parameters': total,
    })
    return model, info


def schedule_kwargs(training_args_cls, warmup_ratio, train_rows,
                    gradient_accumulation, max_steps, epochs):
    """Warmup arguments that work on both transformers 4.x and 5.x.

    5.x dropped warmup_ratio. Where it still exists the ratio is passed
    through untouched, so the Phase 3 path stays bit-identical.
    """
    if 'warmup_ratio' in inspect.signature(
            training_args_cls.__init__).parameters:
        return {'warmup_ratio': warmup_ratio}
    per_epoch = math.ceil(train_rows / gradient_accumulation)
    total = max_steps if max_steps > 0 else math.ceil(per_epoch * epochs)
    return {'warmup_steps': round(warmup_ratio * total)}


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
        AutoTokenizer, Trainer, TrainerCallback, TrainingArguments,
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
    print(f'loading {args.tuning} model from {model_source}...')
    model, model_info = load_trainable_model(model_source, args)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={'use_reentrant': False})
    model.config.use_cache = False
    if args.tuning != 'full':
        # A frozen base under checkpointing gives the adapters no grad path.
        model.enable_input_require_grads()
    print(f'model: {model_info}')

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
        lr_scheduler_type='cosine',
        optim=args.optim, bf16=True, fp16=False,
        gradient_checkpointing=True, eval_strategy='steps',
        eval_steps=args.eval_steps, save_strategy='steps',
        save_steps=args.save_steps, save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps, report_to='none',
        remove_unused_columns=False, prediction_loss_only=True, seed=args.seed,
        **schedule_kwargs(TrainingArguments, args.warmup_ratio,
                          len(train_rows), args.gradient_accumulation,
                          args.max_steps, args.epochs),
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
        'optim': args.optim,
        'model': model_info,
        'gradient_accumulation': args.gradient_accumulation,
        'train_rows': len(train_rows), 'val_rows': len(val_rows),
        'kind_counts': dict(sorted(kind_counts.items())),
        'system_conditioned_rows': conditioned,
        'snapshots': snapshots,
    }
    (output / 'run_meta.json').write_text(json.dumps(metadata, indent=2) + '\n')
    torch.cuda.reset_peak_memory_stats()
    result = trainer.train(resume_from_checkpoint=resume or None)
    final = output / 'final'
    trainer.save_model(final)
    tokenizer.save_pretrained(final)
    runtime = result.metrics.get('train_runtime')
    steps = trainer.state.global_step
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    metadata.update({
        'status': 'completed',
        'completed_utc': datetime.now(timezone.utc).isoformat(),
        'global_step': steps,
        'train_runtime_seconds': runtime,
        'seconds_per_step': runtime / steps if runtime and steps else None,
        'peak_memory_allocated_bytes': peak_allocated,
        'peak_memory_reserved_bytes': peak_reserved,
    })
    (output / 'run_meta.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'peak memory: {peak_allocated / 1024 ** 3:.2f} GiB allocated, '
          f'{peak_reserved / 1024 ** 3:.2f} GiB reserved')
    if runtime and steps:
        print(f'throughput: {runtime / steps:.2f} s/step over {steps} steps')
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
    parser.add_argument('--optim', default='adamw_torch_fused',
                        choices=('adamw_torch_fused', 'adamw_bnb_8bit',
                                 'adafactor'))
    parser.add_argument('--tuning', default='full',
                        choices=('full', 'lora', 'qlora'))
    parser.add_argument('--attn-implementation', default='eager')
    parser.add_argument('--lora-r', type=int, default=16)
    parser.add_argument('--lora-alpha', type=int, default=32)
    parser.add_argument('--lora-dropout', type=float, default=0.05)
    parser.add_argument('--lora-target-modules', nargs='+',
                        default=list(LORA_TARGET_MODULES))
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
