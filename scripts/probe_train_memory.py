#!/usr/bin/env python3
"""Walk the 12B training-memory ladder and stop at the first rung that fits.

Implements DeepRed-Phase4-Plan.md P2.2/P2.3. Each rung runs 200 optimizer
steps at the real training shape — max_length 768, gradient accumulation 16,
gradient checkpointing on, eager attention — and records peak memory and
seconds per step. The winning rung becomes the P7 training configuration.

Rungs run in separate processes. A rung that OOMs must not poison the
measurement of the next one, and a 12B allocator does not reliably give memory
back inside a single process.

Usage:
  /opt/venv/bin/python3 scripts/probe_train_memory.py \\
      --model /mnt/data/models/gemma-4-12b-it \\
      --dataset /mnt/data/sft_corpus/deepred-p3v2 \\
      --output-dir /mnt/data/evaluations/phase4/memory-probe
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_deepred_sft import (  # noqa: E402
    LORA_TARGET_MODULES, load_trainable_model, make_collator, schedule_kwargs,
    shuffled_rows,
)
from train_deepred_npo import load_jsonl, tokenize_messages  # noqa: E402

GIB = 1024 ** 3

# Cheapest-to-most-lossy. Full weight first because it is what Phase 3 used
# and the only rung that changes every parameter.
LADDER = (
    {'name': 'full-adamw-bnb-8bit', 'tuning': 'full',
     'optim': 'adamw_bnb_8bit'},
    {'name': 'full-adafactor', 'tuning': 'full', 'optim': 'adafactor'},
    {'name': 'lora', 'tuning': 'lora', 'optim': 'adamw_torch_fused'},
    {'name': 'qlora', 'tuning': 'qlora', 'optim': 'adamw_bnb_8bit'},
)


def rung_by_name(name):
    for rung in LADDER:
        if rung['name'] == name:
            return rung
    raise SystemExit(f'unknown rung: {name}')


def run_one_rung(args):
    """Train one rung in this process and write its measurement."""
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, Trainer, TrainingArguments

    rung = rung_by_name(args.rung)
    if not torch.cuda.is_available():
        raise SystemExit('a ROCm/CUDA device is required')

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = shuffled_rows(
        load_jsonl(Path(args.dataset) / 'retain_train.jsonl'), args.seed)
    # 200 optimizer steps consume steps * accumulation samples; tokenizing the
    # remaining ~17k rows would only slow the probe down.
    needed = args.steps * args.gradient_accumulation
    rows = rows[:needed]
    encoded = [tokenize_messages(tokenizer, row['messages'], args.max_length)
               for row in rows]
    dataset = Dataset.from_list(encoded)
    longest = max(len(item['input_ids']) for item in encoded)

    model_args = SimpleNamespace(
        tuning=rung['tuning'],
        attn_implementation=args.attn_implementation,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=list(LORA_TARGET_MODULES),
    )
    load_started = time.monotonic()
    model, model_info = load_trainable_model(args.model, model_args)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={'use_reentrant': False})
    model.config.use_cache = False
    if rung['tuning'] != 'full':
        model.enable_input_require_grads()
    load_seconds = time.monotonic() - load_started

    output = Path(args.output_dir) / rung['name']
    training_args = TrainingArguments(
        output_dir=str(output / 'hf'), max_steps=args.steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate, adam_beta1=0.9, adam_beta2=0.95,
        lr_scheduler_type='cosine', optim=rung['optim'],
        bf16=True, fp16=False, gradient_checkpointing=True,
        # A probe measures training, not checkpointing or evaluation.
        eval_strategy='no', save_strategy='no',
        logging_steps=args.logging_steps, report_to='none',
        remove_unused_columns=False, prediction_loss_only=True,
        seed=args.seed,
        **schedule_kwargs(TrainingArguments, 0.03, len(rows),
                          args.gradient_accumulation, args.steps, 1.0),
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset,
                      data_collator=make_collator(tokenizer))

    torch.cuda.reset_peak_memory_stats()
    result = trainer.train()
    runtime = result.metrics.get('train_runtime')
    steps = trainer.state.global_step

    measurement = {
        'rung': rung['name'],
        'tuning': rung['tuning'],
        'optim': rung['optim'],
        'status': 'ok',
        'steps': steps,
        'train_runtime_seconds': runtime,
        'seconds_per_step': runtime / steps if runtime and steps else None,
        'peak_memory_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_memory_reserved_bytes': torch.cuda.max_memory_reserved(),
        'model_load_seconds': load_seconds,
        'max_sequence_length': longest,
        'max_length': args.max_length,
        'gradient_accumulation': args.gradient_accumulation,
        'train_rows_used': len(rows),
        'model': model_info,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / 'measurement.json').write_text(
        json.dumps(measurement, indent=2) + '\n')
    print(json.dumps({k: v for k, v in measurement.items() if k != 'model'},
                     indent=2))
    return 0


def evaluate(measurement, ceiling_bytes, headroom_bytes, baseline, slowdown):
    """Apply the P2.3 gate to one rung's measurement."""
    # Reserved, not allocated: the allocator's reservation is what actually
    # competes with the GTT ceiling.
    peak = measurement['peak_memory_reserved_bytes']
    seconds = measurement['seconds_per_step']
    headroom = ceiling_bytes - peak
    limit = baseline * slowdown
    reasons = []
    if headroom < headroom_bytes:
        reasons.append(
            f'headroom {headroom / GIB:.1f} GiB is under the required '
            f'{headroom_bytes / GIB:.1f} GiB')
    if seconds is None or seconds > limit:
        reasons.append(
            f'{seconds:.2f} s/step exceeds the {limit:.2f} s/step limit '
            f'({slowdown:g}x the {baseline:.2f} s/step 4B baseline)')
    return {
        'headroom_bytes': headroom,
        'headroom_gib': round(headroom / GIB, 2),
        'peak_reserved_gib': round(peak / GIB, 2),
        'peak_allocated_gib': round(
            measurement['peak_memory_allocated_bytes'] / GIB, 2),
        'seconds_per_step': seconds,
        'step_time_limit_seconds': limit,
        'passed': not reasons,
        'failure_reasons': reasons,
    }


def walk_ladder(args):
    ceiling_bytes = int(args.gtt_gib * GIB)
    headroom_bytes = int(args.headroom_gib * GIB)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    winner = None
    for rung in LADDER:
        print(f"\n=== rung {rung['name']} "
              f"({rung['tuning']}, {rung['optim']}) ===", flush=True)
        child = [sys.executable, str(Path(__file__).resolve()),
                 '--rung', rung['name'],
                 '--model', args.model,
                 '--dataset', args.dataset,
                 '--output-dir', str(output_dir),
                 '--steps', str(args.steps),
                 '--max-length', str(args.max_length),
                 '--gradient-accumulation', str(args.gradient_accumulation),
                 '--attn-implementation', args.attn_implementation]
        if args.tokenizer:
            child += ['--tokenizer', args.tokenizer]
        completed = subprocess.run(child)

        path = output_dir / rung['name'] / 'measurement.json'
        if completed.returncode != 0 or not path.is_file():
            entry = {'rung': rung['name'], 'tuning': rung['tuning'],
                     'optim': rung['optim'], 'status': 'failed',
                     'exit_code': completed.returncode,
                     'gate': {'passed': False,
                              'failure_reasons': ['rung did not complete — '
                                                  'see its console output']}}
            results.append(entry)
            print(f"  {rung['name']}: FAILED (exit {completed.returncode})")
            continue

        measurement = json.loads(path.read_text())
        measurement['gate'] = evaluate(measurement, ceiling_bytes,
                                       headroom_bytes, args.baseline_seconds,
                                       args.max_slowdown)
        results.append(measurement)
        gate = measurement['gate']
        print(f"  {rung['name']}: peak {gate['peak_reserved_gib']} GiB "
              f"reserved, headroom {gate['headroom_gib']} GiB, "
              f"{measurement['seconds_per_step']:.2f} s/step -> "
              f"{'PASS' if gate['passed'] else 'FAIL'}")
        for reason in gate['failure_reasons']:
            print(f'      {reason}')
        if gate['passed']:
            winner = measurement['rung']
            break

    verdict = {
        'schema_version': 1,
        'probe': 'train_memory',
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'model': args.model,
        'dataset': args.dataset,
        'steps': args.steps,
        'gtt_ceiling_gib': args.gtt_gib,
        'required_headroom_gib': args.headroom_gib,
        'baseline_seconds_per_step': args.baseline_seconds,
        'max_slowdown': args.max_slowdown,
        'winning_rung': winner,
        'passed': winner is not None,
        'rungs': results,
    }
    (output_dir / 'verdict.json').write_text(
        json.dumps(verdict, indent=2) + '\n')
    print(f"\nwinning rung: {winner or 'NONE — every rung failed'}")
    print(f"verdict -> {output_dir / 'verdict.json'}")
    return 0 if winner else 1


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--tokenizer')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--max-length', type=int, default=768)
    parser.add_argument('--gradient-accumulation', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=5e-6)
    parser.add_argument('--logging-steps', type=int, default=25)
    parser.add_argument('--seed', type=int, default=1969)
    parser.add_argument('--attn-implementation', default='eager')
    parser.add_argument('--lora-r', type=int, default=16)
    parser.add_argument('--lora-alpha', type=int, default=32)
    parser.add_argument('--lora-dropout', type=float, default=0.05)
    parser.add_argument('--gtt-gib', type=float, default=124.0,
                        help='Configured GPU memory ceiling.')
    parser.add_argument('--headroom-gib', type=float, default=8.0,
                        help='Headroom a rung must leave to pass (P2.3).')
    parser.add_argument('--baseline-seconds', type=float, default=8.63,
                        help='Re-baselined 4B seconds per step (P0.4).')
    parser.add_argument('--max-slowdown', type=float, default=4.0)
    parser.add_argument('--rung', help=argparse.SUPPRESS)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.rung:
        return run_one_rung(args)
    return walk_ladder(args)


if __name__ == '__main__':
    raise SystemExit(main())
