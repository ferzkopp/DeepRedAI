#!/usr/bin/env python3
"""Train DeepRed Phase 2 with NPO and a frozen-reference retain anchor.

Run inside the strix-halo-finetuning container. Reference sequence
log-probabilities are cached first; the reference model is then unloaded before
the trainable full-weight model is created.
"""

import argparse
import gc
import hashlib
import json
import os
import random
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path


os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')


class TrainingError(ValueError):
    pass


RETAIN_KINDS = {'retain', 'era_native', 'persona', 'persona_controls'}


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path):
    rows = []
    with path.open(encoding='utf-8') as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise TrainingError(f'{path}:{number}: {exc}') from exc
    return rows


def parse_snapshots(values):
    snapshots = sorted(set(float(value) for value in values))
    if any(value <= 0 or value > 100 for value in snapshots):
        raise TrainingError('--snapshot-at values must be in (0, 100]')
    return snapshots


def resolve_resume(output, value):
    if value != 'auto':
        return value or None
    checkpoints = sorted(
        output.glob('checkpoint-*'),
        key=lambda path: int(path.name.split('-')[-1]),
    )
    return str(checkpoints[-1]) if checkpoints else None


def tokenize_messages(tokenizer, messages, max_length):
    if not messages or messages[-1].get('role') != 'assistant':
        raise TrainingError('each record must end with an assistant message')
    prefix = tokenizer.apply_chat_template(
        messages[:-1], tokenize=True, add_generation_prompt=True)
    complete = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False)
    if isinstance(prefix, Mapping):
        prefix = prefix['input_ids']
    if isinstance(complete, Mapping):
        complete = complete['input_ids']
    input_ids = list(complete[-max_length:])
    removed = max(0, len(complete) - max_length)
    prefix_length = max(0, min(len(input_ids), len(prefix) - removed))
    labels = [-100] * prefix_length + input_ids[prefix_length:]
    if not any(label != -100 for label in labels):
        raise TrainingError('assistant target was truncated completely')
    return {'input_ids': input_ids, 'labels': labels}


def sequence_logps(logits, labels):
    """Return summed log-probability and token count for each sequence."""
    import torch
    shifted_labels = labels[:, 1:].clone()
    mask = shifted_labels.ne(-100)
    safe_labels = shifted_labels.masked_fill(~mask, 0)
    token_logps = torch.gather(
        logits[:, :-1].float().log_softmax(-1), 2,
        safe_labels.unsqueeze(2),
    ).squeeze(2)
    return (token_logps * mask).sum(1), mask.sum(1).clamp_min(1)


def npo_retain_loss(current_logps, reference_logps, token_counts, objectives,
                    beta=0.1, npo_weight=1.0, retain_weight=1.0):
    """Compute NPO on forget rows and sampled forward-KL on retain rows."""
    import torch
    forget = objectives.eq(1)
    retain = ~forget
    zero = current_logps.sum() * 0
    if forget.any():
        ratio = current_logps[forget] - reference_logps[forget]
        forget_loss = -(2 / beta) * torch.nn.functional.logsigmoid(
            -beta * ratio).mean()
    else:
        forget_loss = zero
    if retain.any():
        retain_loss = (
            (reference_logps[retain] - current_logps[retain])
            / token_counts[retain]
        ).mean()
    else:
        retain_loss = zero
    return (npo_weight * forget_loss + retain_weight * retain_loss,
            forget_loss, retain_loss)


def parse_kind_weights(values):
    weights = {}
    for value in values or []:
        try:
            kind, weight = value.split('=', 1)
            weight = float(weight)
        except ValueError as exc:
            raise TrainingError(
                f'invalid --kind-weight {value!r}; expected KIND=N') from exc
        if kind not in RETAIN_KINDS or weight < 0:
            raise TrainingError(f'invalid --kind-weight {value!r}')
        weights[kind] = weight
    return weights


def weighted_sample(rows, weights, seed):
    """Deterministically resample rows by kind using relative multipliers."""
    rng = random.Random(seed)
    by_kind = {}
    for row in rows:
        by_kind.setdefault(row.get('kind', 'retain'), []).append(row)
    sampled = []
    for kind, candidates in sorted(by_kind.items()):
        candidates = list(candidates)
        rng.shuffle(candidates)
        target = round(len(candidates) * weights.get(kind, 1.0))
        sampled.extend(candidates[index % len(candidates)]
                       for index in range(target))
    rng.shuffle(sampled)
    return sampled


def balanced_rows(forget, retain, seed, forget_ratio=0.5, kind_weights=None):
    if not forget or not retain:
        raise TrainingError('both forget and retain datasets must be non-empty')
    if not 0 < forget_ratio < 1:
        raise TrainingError('--forget-ratio must be between 0 and 1')
    rng = random.Random(seed)
    forget = list(forget)
    retain = weighted_sample(retain, kind_weights or {}, seed + 1)
    rng.shuffle(forget)
    forget_count = max(1, round(len(retain) * forget_ratio / (1 - forget_ratio)))
    rows = []
    rows.extend({**forget[index % len(forget)], 'objective': 'forget'}
                for index in range(forget_count))
    rows.extend({**row, 'objective': 'retain'} for row in retain)
    rng.shuffle(rows)
    return rows


def cache_fingerprint(args, paths):
    payload = {
        'base_model': str(Path(args.base_model).resolve()),
        'max_length': args.max_length,
        'files': {path.name: file_hash(path) for path in paths},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def make_collator(tokenizer):
    import torch

    def collate(features):
        maximum = max(len(item['input_ids']) for item in features)
        input_ids, labels, attention = [], [], []
        for item in features:
            padding = maximum - len(item['input_ids'])
            input_ids.append(item['input_ids'] + [tokenizer.pad_token_id] * padding)
            labels.append(item['labels'] + [-100] * padding)
            attention.append([1] * len(item['input_ids']) + [0] * padding)
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'attention_mask': torch.tensor(attention, dtype=torch.long),
            'reference_logps': torch.tensor(
                [item['reference_logp'] for item in features], dtype=torch.float32),
            'objectives': torch.tensor(
                [1 if item['objective'] == 'forget' else 0 for item in features],
                dtype=torch.bool),
        }
    return collate


def precompute_reference(args, tokenizer, rows, cache_path, fingerprint):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM

    if cache_path.is_file():
        cached = json.loads(cache_path.read_text())
        if cached.get('fingerprint') == fingerprint:
            values = cached.get('logps', {})
            if all(row['id'] in values for row in rows):
                print(f'reference cache hit: {cache_path}')
                return values

    print('loading frozen reference model...')
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16, attn_implementation='eager',
        trust_remote_code=True).to('cuda')
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    prepared = []
    for row in rows:
        prepared.append({
            **tokenize_messages(tokenizer, row['messages'], args.max_length),
            'id': row['id'], 'objective': row['objective'],
            'reference_logp': 0.0,
        })
    loader = DataLoader(
        prepared, batch_size=args.reference_batch_size, shuffle=False,
        collate_fn=make_collator(tokenizer))
    values = {}
    offset = 0
    with torch.inference_mode():
        for batch in loader:
            ids = batch['input_ids'].to('cuda')
            mask = batch['attention_mask'].to('cuda')
            labels = batch['labels'].to('cuda')
            logits = model(input_ids=ids, attention_mask=mask).logits
            logps, _ = sequence_logps(logits, labels)
            for value in logps.cpu().tolist():
                values[prepared[offset]['id']] = value
                offset += 1
            if offset % 1000 < args.reference_batch_size:
                print(f'  reference: {offset:,}/{len(prepared):,}', flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({
        'fingerprint': fingerprint, 'logps': values,
    }, sort_keys=True) + '\n')
    del model, loader
    gc.collect()
    torch.cuda.empty_cache()
    return values


def train(args):
    executable = sys.executable
    if executable.startswith('/mnt/data/venv/'):
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
    paths = [dataset_dir / f'{objective}_{split}.jsonl'
             for objective in ('forget', 'retain') for split in ('train', 'val')]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise TrainingError(f'missing dataset files: {", ".join(missing)}')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    snapshots = parse_snapshots(args.snapshot_at)
    resume = resolve_resume(output, args.resume)
    fingerprint = cache_fingerprint(args, paths)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    kind_weights = parse_kind_weights(args.kind_weight)
    raw_train = balanced_rows(
        load_jsonl(paths[0]), load_jsonl(paths[2]), args.seed,
        args.forget_ratio, kind_weights)
    raw_val = balanced_rows(
        load_jsonl(paths[1]), load_jsonl(paths[3]), args.seed + 1,
        args.forget_ratio, kind_weights)
    unique = {row['id']: row for row in raw_train + raw_val}
    reference = precompute_reference(
        args, tokenizer, list(unique.values()), output / 'reference_logps.json',
        fingerprint)

    def prepare(rows):
        prepared = []
        for row in rows:
            item = tokenize_messages(tokenizer, row['messages'], args.max_length)
            item.update({
                'id': row['id'], 'objective': row['objective'],
                'reference_logp': reference[row['id']],
            })
            prepared.append(item)
        return prepared

    train_dataset = Dataset.from_list(prepare(raw_train))
    val_dataset = Dataset.from_list(prepare(raw_val))
    model_source = resume or args.base_model
    print(f'loading trainable full-weight model from {model_source}...')
    model = AutoModelForCausalLM.from_pretrained(
        model_source, dtype=torch.bfloat16, attn_implementation='eager',
        trust_remote_code=True)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={'use_reentrant': False})
    model.config.use_cache = False

    class NPOTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False,
                         num_items_in_batch=None):
            reference_logps = inputs.pop('reference_logps')
            objectives = inputs.pop('objectives')
            labels = inputs['labels']
            outputs = model(**inputs)
            current, counts = sequence_logps(outputs.logits, labels)
            loss, forget_loss, retain_loss = npo_retain_loss(
                current, reference_logps, counts, objectives,
                beta=args.beta, npo_weight=args.npo_weight,
                retain_weight=args.retain_weight)
            if model.training and self.state.global_step % args.logging_steps == 0:
                self.log({'npo_loss': forget_loss.detach().item(),
                          'retain_loss': retain_loss.detach().item()})
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
                path = output / 'snapshots' / f'{int(percentage):03d}pct-step-{state.global_step}'
                path.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(path, safe_serialization=True)
                tokenizer.save_pretrained(path)
                self.saved.add(percentage)
                print(f'saved snapshot {percentage:g}% -> {path}', flush=True)
            return control

    training_args = TrainingArguments(
        output_dir=str(output), num_train_epochs=args.epochs,
        max_steps=args.max_steps, per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
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
    trainer = NPOTrainer(
        model=model, args=training_args, train_dataset=train_dataset,
        eval_dataset=val_dataset, data_collator=make_collator(tokenizer),
        callbacks=[SnapshotCallback()])
    metadata = {
        'schema_version': 1, 'status': 'running',
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'base_model': args.base_model, 'dataset': str(dataset_dir),
        'reference_fingerprint': fingerprint, 'beta': args.beta,
        'npo_weight': args.npo_weight,
        'retain_weight': args.retain_weight, 'forget_ratio': args.forget_ratio,
        'kind_weights': kind_weights, 'snapshots': snapshots,
    }
    (output / 'run_meta.json').write_text(json.dumps(metadata, indent=2) + '\n')
    trainer.train(resume_from_checkpoint=resume or None)
    final = output / 'final'
    trainer.save_model(final)
    tokenizer.save_pretrained(final)
    metadata.update({'status': 'completed',
                     'completed_utc': datetime.now(timezone.utc).isoformat(),
                     'global_step': trainer.state.global_step})
    (output / 'run_meta.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'training complete -> {final}')


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-model', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--snapshot-at', nargs='+', default=['10', '25', '50', '75', '100'])
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--npo-weight', type=float, default=1.0,
                        help='Multiplier for sequence-level NPO loss.')
    parser.add_argument('--retain-weight', type=float, default=1.0)
    parser.add_argument('--forget-ratio', type=float, default=0.5,
                        help='Fraction of sampled training rows using NPO.')
    parser.add_argument('--kind-weight', action='append', default=[],
                        help='Retain-kind sampling multiplier KIND=N.')
    parser.add_argument('--epochs', type=float, default=1.0)
    parser.add_argument('--max-steps', type=int, default=-1)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--reference-batch-size', type=int, default=1)
    parser.add_argument('--gradient-accumulation', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--warmup-ratio', type=float, default=0.05)
    parser.add_argument('--max-length', type=int, default=2048)
    parser.add_argument('--logging-steps', type=int, default=10)
    parser.add_argument('--eval-steps', type=int, default=100)
    parser.add_argument('--save-steps', type=int, default=100)
    parser.add_argument('--save-total-limit', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1969)
    parser.add_argument('--resume', default='auto',
                        help='Checkpoint path, auto, or empty for a fresh run')
    return parser


def main(argv=None):
    try:
        args = build_parser().parse_args(argv)
        if args.beta <= 0 or args.npo_weight < 0 or args.retain_weight < 0:
            raise TrainingError(
                '--beta must be positive and loss weights nonnegative')
        train(args)
    except TrainingError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())