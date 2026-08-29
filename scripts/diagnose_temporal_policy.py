#!/usr/bin/env python3
"""Build and score paired completions for temporal-policy diagnostics.

The prepare command samples era-native rows by split and generation mode. After
the evaluator generates untouched-base responses for those prompts, attach
joins each desired target to the corresponding base response. Score compares
their token-normalized log-probabilities under one or more Hugging Face models.
"""

import argparse
import gc
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

try:
    from train_deepred_npo import sequence_logps, tokenize_messages
except ModuleNotFoundError:
    from scripts.train_deepred_npo import sequence_logps, tokenize_messages


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


def sample_targets(dataset_dir, era_corpus, per_mode, seed):
    corpus_by_id = {row['id']: row for row in load_jsonl(era_corpus)}
    rng = random.Random(seed)
    selected = []
    for split in ('train', 'val'):
        rows = [
            row for row in load_jsonl(dataset_dir / f'retain_{split}.jsonl')
            if row.get('kind') == 'era_native'
        ]
        by_mode = defaultdict(list)
        for row in rows:
            source = corpus_by_id.get(row.get('source_id'))
            if source is None:
                raise DiagnosticError(
                    f'era-native source not found: {row.get("source_id")}')
            by_mode[source.get('mode')].append((row, source))
        for mode in ('in_world', 'hedged', 'premise_correction'):
            candidates = by_mode.get(mode, [])
            if len(candidates) < per_mode:
                raise DiagnosticError(
                    f'{split}/{mode} has {len(candidates)} rows, need {per_mode}')
            chosen = rng.sample(candidates, per_mode)
            for index, (row, source) in enumerate(chosen, 1):
                selected.append({
                    'id': f'temporal-margin-{split}-{mode}-{index:02d}',
                    'split': split,
                    'mode': mode,
                    'source_id': source['id'],
                    'messages': row['messages'][:-1],
                    'desired_completion': row['messages'][-1]['content'],
                })
    return selected


def make_probes(targets):
    return [{
        'id': row['id'],
        'category': 'post_1969',
        'temporal_class': 'post_1969',
        'messages': row['messages'],
        'expected_facts': [],
        'forbidden_facts': [],
        'suite_tags': ['diagnostic'],
    } for row in targets]


def attach_responses(targets, generations, model_id):
    responses = {
        row['probe_id']: row.get('response')
        for row in generations if row.get('model_id') == model_id
    }
    pairs = []
    for target in targets:
        rejected = responses.get(target['id'])
        if not isinstance(rejected, str) or not rejected.strip():
            raise DiagnosticError(
                f'missing {model_id} response for {target["id"]}')
        pairs.append({**target, 'rejected_completion': rejected})
    return pairs


def parse_models(values):
    models = []
    for value in values:
        try:
            model_id, path = value.split('=', 1)
        except ValueError as exc:
            raise DiagnosticError(
                f'invalid --model {value!r}; expected ID=PATH') from exc
        if not model_id or not path or not Path(path).is_dir():
            raise DiagnosticError(f'invalid --model {value!r}')
        models.append((model_id, path))
    return models


def summarize(results):
    groups = defaultdict(list)
    for row in results:
        groups[('all', 'all')].append(row['margin'])
        groups[(row['split'], 'all')].append(row['margin'])
        groups[(row['split'], row['mode'])].append(row['margin'])
    return [{
        'split': split,
        'mode': mode,
        'count': len(margins),
        'mean_margin': sum(margins) / len(margins),
        'win_rate': sum(margin > 0 for margin in margins) / len(margins),
    } for (split, mode), margins in sorted(groups.items())]


def score_model(model_id, model_path, tokenizer_path, pairs, batch_size,
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
    for pair in pairs:
        for candidate_type in ('desired', 'rejected'):
            completion = pair[f'{candidate_type}_completion']
            encoded = tokenize_messages(tokenizer, pair['messages'] + [{
                'role': 'assistant', 'content': completion,
            }], max_length)
            candidates.append((pair, candidate_type, encoded))

    values = {}
    with torch.inference_mode():
        for offset in range(0, len(candidates), batch_size):
            batch = candidates[offset:offset + batch_size]
            maximum = max(len(item[2]['input_ids']) for item in batch)
            input_ids, labels, attention = [], [], []
            for _, _, encoded in batch:
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
            for (pair, candidate_type, _), logp, count in zip(
                    batch, logps.tolist(), counts.tolist()):
                values[(pair['id'], candidate_type)] = logp / count
            print(f'  {model_id}: {min(offset + len(batch), len(candidates))}'
                  f'/{len(candidates)} candidates', flush=True)

    results = []
    for pair in pairs:
        desired = values[(pair['id'], 'desired')]
        rejected = values[(pair['id'], 'rejected')]
        results.append({
            'id': pair['id'], 'split': pair['split'], 'mode': pair['mode'],
            'desired_mean_logp': desired,
            'rejected_mean_logp': rejected,
            'margin': desired - rejected,
            'preferred': desired > rejected,
        })
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {'model_id': model_id, 'results': results,
            'summary': summarize(results)}


def command_prepare(args):
    output = Path(args.output_dir)
    targets = sample_targets(
        Path(args.dataset), Path(args.era_corpus), args.per_mode, args.seed)
    write_jsonl(output / 'targets.jsonl', targets)
    write_jsonl(output / 'probes.jsonl', make_probes(targets))
    print(f'wrote {len(targets)} targets and probes -> {output}')


def command_attach(args):
    pairs = attach_responses(
        load_jsonl(Path(args.targets)),
        load_jsonl(Path(args.generations)), args.model_id)
    write_jsonl(Path(args.output), pairs)
    print(f'wrote {len(pairs)} completion pairs -> {args.output}')


def command_score(args):
    pairs = load_jsonl(Path(args.pairs))
    reports = [score_model(model_id, path, args.tokenizer, pairs,
                           args.batch_size, args.max_length)
               for model_id, path in parse_models(args.model)]
    Path(args.output).write_text(json.dumps({
        'schema_version': 1, 'pairs': len(pairs), 'models': reports,
    }, indent=2) + '\n')
    for report in reports:
        overall = next(row for row in report['summary']
                       if row['split'] == row['mode'] == 'all')
        print(f'{report["model_id"]}: margin={overall["mean_margin"]:.4f} '
              f'win_rate={overall["win_rate"]:.1%}')


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)

    prepare = commands.add_parser('prepare')
    prepare.add_argument('--dataset', required=True)
    prepare.add_argument('--era-corpus', required=True)
    prepare.add_argument('--output-dir', required=True)
    prepare.add_argument('--per-mode', type=int, default=4)
    prepare.add_argument('--seed', type=int, default=1969)
    prepare.set_defaults(function=command_prepare)

    attach = commands.add_parser('attach')
    attach.add_argument('--targets', required=True)
    attach.add_argument('--generations', required=True)
    attach.add_argument('--model-id', required=True)
    attach.add_argument('--output', required=True)
    attach.set_defaults(function=command_attach)

    score = commands.add_parser('score')
    score.add_argument('--pairs', required=True)
    score.add_argument('--model', action='append', required=True,
                       help='Model ID and Hugging Face directory as ID=PATH.')
    score.add_argument('--tokenizer', required=True,
                       help='Shared tokenizer directory used by every model.')
    score.add_argument('--output', required=True)
    score.add_argument('--batch-size', type=int, default=4)
    score.add_argument('--max-length', type=int, default=2048)
    score.set_defaults(function=command_score)
    return parser


def main(argv=None):
    try:
        args = build_parser().parse_args(argv)
        if getattr(args, 'per_mode', 1) < 1:
            raise DiagnosticError('--per-mode must be positive')
        args.function(args)
    except DiagnosticError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())