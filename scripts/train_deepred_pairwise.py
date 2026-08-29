#!/usr/bin/env python3
"""Train a full-weight temporal model with absolute pairwise margins."""

import argparse
import gc
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from train_deepred_npo import (
        TrainingError, load_jsonl, parse_snapshots, resolve_resume,
        sequence_logps, tokenize_messages,
    )
except ModuleNotFoundError:
    from scripts.train_deepred_npo import (
        TrainingError, load_jsonl, parse_snapshots, resolve_resume,
        sequence_logps, tokenize_messages,
    )


os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')


def pairwise_margin_loss(chosen_logps, chosen_counts, rejected_logps,
                         rejected_counts, margin_target=0.25):
    """Penalize current-model mean-token margins below the target."""
    import torch
    chosen_mean = chosen_logps / chosen_counts
    rejected_mean = rejected_logps / rejected_counts
    margins = chosen_mean - rejected_mean
    return torch.nn.functional.softplus(margin_target - margins).mean(), margins


def pairwise_with_chosen_loss(chosen_logps, chosen_counts, rejected_logps,
                              rejected_counts, margin_target=0.25,
                              chosen_ce_weight=0.0):
    pair_loss, margins = pairwise_margin_loss(
        chosen_logps, chosen_counts, rejected_logps, rejected_counts,
        margin_target)
    chosen_nll = -(chosen_logps / chosen_counts).mean()
    return pair_loss + chosen_ce_weight * chosen_nll, pair_loss, chosen_nll, margins


def balanced_objective_rows(pairs, anchors, seed):
    if not pairs or not anchors:
        raise TrainingError('pair and anchor datasets must both be non-empty')
    if len(pairs) != len(anchors):
        raise TrainingError('pair and anchor datasets must have equal lengths')
    rng = random.Random(seed)
    pairs = list(pairs)
    anchors = list(anchors)
    rng.shuffle(pairs)
    rng.shuffle(anchors)
    rows = []
    for pair, anchor in zip(pairs, anchors):
        rows.append({**pair, 'objective': 'pair'})
        rows.append({**anchor, 'objective': 'anchor'})
    return rows


def prepare_row(tokenizer, row, max_length):
    empty = {'input_ids': [], 'labels': []}
    if row['objective'] == 'pair':
        prompt = row['messages']
        chosen = tokenize_messages(tokenizer, prompt + [{
            'role': 'assistant', 'content': row['chosen_completion'],
        }], max_length)
        rejected = tokenize_messages(tokenizer, prompt + [{
            'role': 'assistant', 'content': row['rejected_completion'],
        }], max_length)
        anchor = empty
    else:
        chosen = rejected = empty
        anchor = tokenize_messages(tokenizer, row['messages'], max_length)
    return {
        'id': row['id'], 'objective': row['objective'],
        'chosen_input_ids': chosen['input_ids'],
        'chosen_labels': chosen['labels'],
        'rejected_input_ids': rejected['input_ids'],
        'rejected_labels': rejected['labels'],
        'anchor_input_ids': anchor['input_ids'],
        'anchor_labels': anchor['labels'],
    }


def make_collator(tokenizer):
    import torch

    def pad(features, prefix):
        maximum = max(len(row[f'{prefix}_input_ids']) for row in features)
        input_ids, labels, attention = [], [], []
        for row in features:
            ids = row[f'{prefix}_input_ids']
            row_labels = row[f'{prefix}_labels']
            padding = maximum - len(ids)
            input_ids.append(ids + [tokenizer.pad_token_id] * padding)
            labels.append(row_labels + [-100] * padding)
            attention.append([1] * len(ids) + [0] * padding)
        return {
            f'{prefix}_input_ids': torch.tensor(input_ids, dtype=torch.long),
            f'{prefix}_labels': torch.tensor(labels, dtype=torch.long),
            f'{prefix}_attention_mask': torch.tensor(attention, dtype=torch.long),
        }

    def collate(features):
        objectives = {row['objective'] for row in features}
        if len(objectives) != 1:
            raise TrainingError('a batch cannot mix pair and anchor rows')
        objective = objectives.pop()
        result = {
            'objective': objective,
            # Trainer requires a tensor label to route evaluation through
            # custom compute_loss instead of calling model(**inputs).
            'loss_labels': torch.zeros(len(features), dtype=torch.long),
        }
        if objective == 'pair':
            result.update(pad(features, 'chosen'))
            result.update(pad(features, 'rejected'))
        else:
            result.update(pad(features, 'anchor'))
        return result
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
    paths = {name: dataset_dir / f'{name}.jsonl' for name in (
        'pair_train', 'pair_val', 'anchor_train', 'anchor_val')}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise TrainingError(f'missing dataset files: {", ".join(missing)}')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    snapshots = parse_snapshots(args.snapshot_at)
    resume = resolve_resume(output, args.resume)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_rows = balanced_objective_rows(
        load_jsonl(paths['pair_train']), load_jsonl(paths['anchor_train']),
        args.seed)
    val_rows = balanced_objective_rows(
        load_jsonl(paths['pair_val']), load_jsonl(paths['anchor_val']),
        args.seed + 1)
    train_dataset = Dataset.from_list([
        prepare_row(tokenizer, row, args.max_length) for row in train_rows])
    val_dataset = Dataset.from_list([
        prepare_row(tokenizer, row, args.max_length) for row in val_rows])

    model_source = resume or args.model
    print(f'loading trainable full-weight model from {model_source}...')
    model = AutoModelForCausalLM.from_pretrained(
        model_source, dtype=torch.bfloat16, attn_implementation='eager',
        trust_remote_code=True)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={'use_reentrant': False})
    model.config.use_cache = False

    class PairwiseTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False,
                         num_items_in_batch=None):
            inputs.pop('loss_labels')
            objective = inputs.pop('objective')
            if objective == 'pair':
                chosen_labels = inputs.pop('chosen_labels')
                rejected_labels = inputs.pop('rejected_labels')
                chosen_outputs = model(
                    input_ids=inputs.pop('chosen_input_ids'),
                    attention_mask=inputs.pop('chosen_attention_mask'))
                rejected_outputs = model(
                    input_ids=inputs.pop('rejected_input_ids'),
                    attention_mask=inputs.pop('rejected_attention_mask'))
                chosen_logps, chosen_counts = sequence_logps(
                    chosen_outputs.logits, chosen_labels)
                rejected_logps, rejected_counts = sequence_logps(
                    rejected_outputs.logits, rejected_labels)
                loss, pair_loss, chosen_nll, margins = pairwise_with_chosen_loss(
                    chosen_logps, chosen_counts, rejected_logps,
                    rejected_counts, args.margin_target,
                    args.chosen_ce_weight)
                if model.training and self.state.global_step % args.logging_steps == 0:
                    self.log({'pair_loss': pair_loss.detach().item(),
                              'chosen_nll': chosen_nll.detach().item(),
                              'pair_objective': loss.detach().item(),
                              'pair_margin': margins.detach().mean().item()})
                outputs = chosen_outputs
            else:
                labels = inputs.pop('anchor_labels')
                outputs = model(
                    input_ids=inputs.pop('anchor_input_ids'),
                    attention_mask=inputs.pop('anchor_attention_mask'))
                logps, counts = sequence_logps(outputs.logits, labels)
                loss = -(logps / counts).mean()
                if model.training and self.state.global_step % args.logging_steps == 0:
                    self.log({'anchor_loss': loss.detach().item()})
            return (loss, outputs) if return_outputs else loss

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
        label_names=['loss_labels'],
    )
    trainer = PairwiseTrainer(
        model=model, args=training_args, train_dataset=train_dataset,
        eval_dataset=val_dataset, data_collator=make_collator(tokenizer),
        callbacks=[SnapshotCallback()])
    metadata = {
        'schema_version': 1, 'status': 'running',
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'initial_model': args.model, 'tokenizer': args.tokenizer,
        'dataset': str(dataset_dir), 'margin_target': args.margin_target,
        'chosen_ce_weight': args.chosen_ce_weight,
        'learning_rate': args.learning_rate, 'max_steps': args.max_steps,
        'gradient_accumulation': args.gradient_accumulation,
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
    parser.add_argument('--margin-target', type=float, default=0.25)
    parser.add_argument('--chosen-ce-weight', type=float, default=0.0)
    parser.add_argument('--learning-rate', type=float, default=1e-6)
    parser.add_argument('--max-steps', type=int, default=300)
    parser.add_argument('--gradient-accumulation', type=int, default=16)
    parser.add_argument('--warmup-ratio', type=float, default=0.05)
    parser.add_argument('--max-length', type=int, default=768)
    parser.add_argument('--logging-steps', type=int, default=10)
    parser.add_argument('--eval-steps', type=int, default=30)
    parser.add_argument('--save-steps', type=int, default=30)
    parser.add_argument('--save-total-limit', type=int, default=2)
    parser.add_argument('--snapshot-at', nargs='+',
                        default=['10', '25', '50', '75', '100'])
    parser.add_argument('--seed', type=int, default=1969)
    parser.add_argument('--resume', default='auto')
    return parser


def main(argv=None):
    try:
        args = build_parser().parse_args(argv)
        if (args.margin_target < 0 or args.chosen_ce_weight < 0
            or args.learning_rate <= 0):
            raise TrainingError(
            '--margin-target and --chosen-ce-weight must be nonnegative; '
            'learning rate must be positive')
        if args.max_steps <= 0 or args.gradient_accumulation <= 0:
            raise TrainingError(
                '--max-steps and --gradient-accumulation must be positive')
        train(args)
    except TrainingError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())