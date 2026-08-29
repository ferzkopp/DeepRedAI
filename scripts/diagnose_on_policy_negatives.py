#!/usr/bin/env python3
"""Prepare and score refreshed temporal hard-negative diagnostics."""

import argparse
import gc
import importlib.util
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    from train_deepred_npo import sequence_logps, tokenize_messages
except ModuleNotFoundError:
    from scripts.train_deepred_npo import sequence_logps, tokenize_messages


SCRIPT_DIR = Path(__file__).resolve().parent
BUILDER_SPEC = importlib.util.spec_from_file_location(
    'build_temporal_pairwise_dataset',
    SCRIPT_DIR / 'build_temporal_pairwise_dataset.py')
BUILDER = importlib.util.module_from_spec(BUILDER_SPEC)
BUILDER_SPEC.loader.exec_module(BUILDER)


class DiagnosticError(ValueError):
    pass


def load_jsonl(path):
    rows = []
    with path.open(encoding='utf-8') as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise DiagnosticError(f'{path}:{number}: {exc}') from exc
    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + '\n')


def first_sentence(text):
    text = ' '.join(text.strip().split())
    if not text:
        raise DiagnosticError('completion is empty')
    match = re.search(r'(?<=[.!?])(?:["\'”’)]*)\s', text)
    return text[:match.end()].strip() if match else text


def prepare_rows(pair_train, pair_val, base_generations, base_model_id,
                 train_per_mode, seed):
    base = {
        row.get('probe_id'): row.get('response')
        for row in base_generations if row.get('model_id') == base_model_id
    }
    grouped = defaultdict(list)
    for row in pair_train:
        grouped[row.get('mode')].append(row)
    rng = random.Random(seed)
    selected = []
    for mode in BUILDER.MODES:
        rows = grouped[mode]
        if len(rows) < train_per_mode:
            raise DiagnosticError(
                f'train/{mode} has {len(rows)} rows, need {train_per_mode}')
        selected.extend(rng.sample(rows, train_per_mode))
    selected.extend(pair_val)

    prepared = []
    for row in selected:
        original = base.get(row['id'])
        fresh = row.get('rejected_completion')
        if not isinstance(original, str) or not original.strip():
            raise DiagnosticError(
                f'missing {base_model_id} response for {row["id"]}')
        if not isinstance(fresh, str) or not fresh.strip():
            raise DiagnosticError(f'missing fresh rejection for {row["id"]}')
        prepared.append({
            'id': row['id'], 'split': row['split'], 'mode': row['mode'],
            'messages': row['messages'],
            'desired_completion': row['chosen_completion'],
            'original_rejected_completion': original,
            'fresh_rejected_completion': fresh,
            'original_behavior': BUILDER.rejected_behavior(original),
            'fresh_behavior': BUILDER.rejected_behavior(fresh),
        })
    return prepared


def parse_models(values):
    models = []
    for value in values:
        try:
            model_id, path = value.split('=', 1)
        except ValueError as exc:
            raise DiagnosticError(
                f'invalid --model {value!r}; expected ID=PATH') from exc
        if not model_id or not Path(path).is_dir():
            raise DiagnosticError(f'invalid --model {value!r}')
        models.append((model_id, path))
    return models


def summarize(results):
    groups = defaultdict(list)
    for row in results:
        groups[('all', 'all')].append(row)
        groups[(row['split'], 'all')].append(row)
        groups[(row['split'], row['mode'])].append(row)
    summaries = []
    for (split, mode), rows in sorted(groups.items()):
        summary = {'split': split, 'mode': mode, 'count': len(rows)}
        for kind in ('desired', 'original_rejected', 'fresh_rejected'):
            summary[f'{kind}_mean_logp'] = sum(
                row[f'{kind}_mean_logp'] for row in rows) / len(rows)
            summary[f'{kind}_first_mean_logp'] = sum(
                row[f'{kind}_first_mean_logp'] for row in rows) / len(rows)
        for prefix in ('original', 'fresh'):
            margins = [row[f'{prefix}_margin'] for row in rows]
            first_margins = [row[f'{prefix}_first_margin'] for row in rows]
            summary[f'{prefix}_mean_margin'] = sum(margins) / len(margins)
            summary[f'{prefix}_win_rate'] = (
                sum(value > 0 for value in margins) / len(margins))
            summary[f'{prefix}_first_mean_margin'] = (
                sum(first_margins) / len(first_margins))
            summary[f'{prefix}_first_win_rate'] = (
                sum(value > 0 for value in first_margins) / len(first_margins))
        summary['fresh_minus_original_logp'] = sum(
            row['fresh_rejected_mean_logp']
            - row['original_rejected_mean_logp'] for row in rows) / len(rows)
        summaries.append(summary)
    return summaries


def score_model(model_id, model_path, tokenizer_path, rows, batch_size,
                max_length):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation='eager',
        trust_remote_code=True).to('cuda')
    model.eval()
    candidates = []
    for row in rows:
        for kind in ('desired', 'original_rejected', 'fresh_rejected'):
            completion = row[f'{kind}_completion']
            for scope, text in (
                    ('full', completion), ('first', first_sentence(completion))):
                encoded = tokenize_messages(tokenizer, row['messages'] + [{
                    'role': 'assistant', 'content': text,
                }], max_length)
                candidates.append((row['id'], kind, scope, encoded))

    values = {}
    with torch.inference_mode():
        for offset in range(0, len(candidates), batch_size):
            batch = candidates[offset:offset + batch_size]
            maximum = max(len(item[3]['input_ids']) for item in batch)
            input_ids, labels, attention = [], [], []
            for _, _, _, encoded in batch:
                padding = maximum - len(encoded['input_ids'])
                input_ids.append(
                    encoded['input_ids'] + [tokenizer.pad_token_id] * padding)
                labels.append(encoded['labels'] + [-100] * padding)
                attention.append([1] * len(encoded['input_ids']) + [0] * padding)
            input_ids = torch.tensor(input_ids, device='cuda')
            labels = torch.tensor(labels, device='cuda')
            attention = torch.tensor(attention, device='cuda')
            logits = model(input_ids=input_ids, attention_mask=attention).logits
            logps, counts = sequence_logps(logits, labels)
            for item, logp, count in zip(batch, logps.tolist(), counts.tolist()):
                values[item[:3]] = logp / count
            print(f'  {model_id}: {min(offset + len(batch), len(candidates))}'
                  f'/{len(candidates)} candidates', flush=True)

    results = []
    for row in rows:
        result = {
            key: row[key] for key in (
                'id', 'split', 'mode', 'original_behavior', 'fresh_behavior')
        }
        for kind in ('desired', 'original_rejected', 'fresh_rejected'):
            result[f'{kind}_mean_logp'] = values[(row['id'], kind, 'full')]
            result[f'{kind}_first_mean_logp'] = values[
                (row['id'], kind, 'first')]
        for prefix in ('original', 'fresh'):
            result[f'{prefix}_margin'] = (
                result['desired_mean_logp']
                - result[f'{prefix}_rejected_mean_logp'])
            result[f'{prefix}_first_margin'] = (
                result['desired_first_mean_logp']
                - result[f'{prefix}_rejected_first_mean_logp'])
        results.append(result)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {'model_id': model_id, 'results': results,
            'summary': summarize(results)}


def command_prepare(args):
    rows = prepare_rows(
        load_jsonl(Path(args.pair_dir) / 'pair_train.jsonl'),
        load_jsonl(Path(args.pair_dir) / 'pair_val.jsonl'),
        load_jsonl(Path(args.base_generations)), args.base_model_id,
        args.train_per_mode, args.seed)
    output = Path(args.output)
    write_jsonl(output, rows)
    behaviors = Counter(row['fresh_behavior'] for row in rows)
    print(f'wrote {len(rows)} diagnostic rows -> {output}')
    print(f'fresh behavior: {dict(sorted(behaviors.items()))}')


def command_score(args):
    rows = load_jsonl(Path(args.pairs))
    reports = [score_model(model_id, path, args.tokenizer, rows,
                           args.batch_size, args.max_length)
               for model_id, path in parse_models(args.model)]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        'schema_version': 1, 'pairs': len(rows), 'models': reports,
    }, indent=2) + '\n')
    for report in reports:
        overall = next(row for row in report['summary']
                       if row['split'] == row['mode'] == 'all')
        print(f'{report["model_id"]}: '
              f'original={overall["original_mean_margin"]:.4f} '
              f'fresh={overall["fresh_mean_margin"]:.4f} '
              f'fresh-original-logp='
              f'{overall["fresh_minus_original_logp"]:+.4f}')


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare')
    prepare.add_argument('--pair-dir', required=True)
    prepare.add_argument('--base-generations', required=True)
    prepare.add_argument('--base-model-id', required=True)
    prepare.add_argument('--output', required=True)
    prepare.add_argument('--train-per-mode', type=int, default=20)
    prepare.add_argument('--seed', type=int, default=1969)
    prepare.set_defaults(function=command_prepare)
    score = commands.add_parser('score')
    score.add_argument('--pairs', required=True)
    score.add_argument('--model', action='append', required=True)
    score.add_argument('--tokenizer', required=True)
    score.add_argument('--output', required=True)
    score.add_argument('--batch-size', type=int, default=4)
    score.add_argument('--max-length', type=int, default=768)
    score.set_defaults(function=command_score)
    return parser


def main(argv=None):
    try:
        args = build_parser().parse_args(argv)
        if getattr(args, 'train_per_mode', 1) < 1:
            raise DiagnosticError('--train-per-mode must be positive')
        args.function(args)
    except DiagnosticError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())