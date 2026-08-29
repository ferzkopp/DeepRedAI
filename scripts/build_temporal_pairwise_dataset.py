#!/usr/bin/env python3
"""Build prompt-aligned temporal preference data and factual anchors."""

import argparse
import importlib.util
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
EVALUATOR_SPEC = importlib.util.spec_from_file_location(
    'evaluate_deepred_models', SCRIPT_DIR / 'evaluate_deepred_models.py')
EVALUATOR = importlib.util.module_from_spec(EVALUATOR_SPEC)
EVALUATOR_SPEC.loader.exec_module(EVALUATOR)

MODES = ('in_world', 'hedged', 'premise_correction')


class DatasetError(ValueError):
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
                raise DatasetError(f'{path}:{number}: {exc}') from exc
    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + '\n')


def source_rows(dataset_dir, era_corpus, split):
    corpus = {row['id']: row for row in load_jsonl(era_corpus)}
    rows = []
    for row in load_jsonl(dataset_dir / f'retain_{split}.jsonl'):
        if row.get('kind') != 'era_native':
            continue
        source = corpus.get(row.get('source_id'))
        if source is None or source.get('mode') not in MODES:
            raise DatasetError(f'invalid era-native source: {row.get("source_id")}')
        rows.append((row, source))
    return rows


def sample_candidates(dataset_dir, era_corpus, counts, seed):
    selected = []
    for split_index, split in enumerate(('train', 'val')):
        grouped = defaultdict(list)
        for row, source in source_rows(dataset_dir, era_corpus, split):
            grouped[source['mode']].append((row, source))
        rng = random.Random(seed + split_index)
        for mode in MODES:
            candidates = grouped[mode]
            needed = counts[split]
            if len(candidates) < needed:
                raise DatasetError(
                    f'{split}/{mode} has {len(candidates)} rows, need {needed}')
            for index, (row, source) in enumerate(
                    rng.sample(candidates, needed), 1):
                selected.append({
                    'id': f'v5-{split}-{mode}-{index:03d}',
                    'split': split,
                    'mode': mode,
                    'source_id': source['id'],
                    'messages': row['messages'][:-1],
                    'chosen_completion': row['messages'][-1]['content'],
                })
    return selected


def make_probes(rows):
    return [{
        'id': row['id'],
        'category': 'post_1969',
        'temporal_class': 'post_1969',
        'messages': row['messages'],
        'expected_facts': [],
        'forbidden_facts': [],
        'suite_tags': ['v5-pairs'],
    } for row in rows]


def sample_anchors(dataset_dir, counts, seed):
    anchors = {}
    for split_index, split in enumerate(('train', 'val')):
        candidates = [
            row for row in load_jsonl(dataset_dir / f'retain_{split}.jsonl')
            if row.get('kind') == 'retain'
        ]
        needed = counts[split]
        if len(candidates) < needed:
            raise DatasetError(
                f'{split} has {len(candidates)} factual anchors, need {needed}')
        rng = random.Random(seed + 10 + split_index)
        anchors[split] = [{
            'id': f'anchor-{row["id"]}', 'messages': row['messages'],
        } for row in rng.sample(candidates, needed)]
    return anchors


def rejected_behavior(response):
    refused = EVALUATOR.is_refusal(response)
    uncertain = EVALUATOR.has_uncertainty(response)
    return EVALUATOR.classify_temporal_behavior(
        'post_1969', False, refused, uncertain)


def finalize_pairs(candidates, generations, model_id, required, seed):
    responses = {
        row.get('probe_id'): row.get('response')
        for row in generations if row.get('model_id') == model_id
    }
    accepted = defaultdict(list)
    rejected_counts = defaultdict(int)
    for candidate in candidates:
        response = responses.get(candidate['id'])
        if not isinstance(response, str) or not response.strip():
            raise DatasetError(f'missing {model_id} response for {candidate["id"]}')
        key = (candidate['split'], candidate['mode'])
        behavior = rejected_behavior(response)
        if behavior != 'confident_unsupported':
            rejected_counts[(key, behavior)] += 1
            continue
        accepted[key].append({
            **candidate, 'rejected_completion': response,
            'rejected_behavior': behavior,
        })

    selected = {'train': [], 'val': []}
    for split_index, split in enumerate(('train', 'val')):
        rng = random.Random(seed + 20 + split_index)
        for mode in MODES:
            rows = accepted[(split, mode)]
            needed = required[split]
            if len(rows) < needed:
                reasons = {
                    behavior: count for ((row_key, behavior), count)
                    in rejected_counts.items() if row_key == (split, mode)
                }
                raise DatasetError(
                    f'{split}/{mode} has {len(rows)} usable responses, '
                    f'need {needed}; filtered={reasons}')
            selected[split].extend(rng.sample(rows, needed))
        rng.shuffle(selected[split])
    return selected


def command_prepare(args):
    output = Path(args.output_dir)
    dataset = Path(args.dataset)
    candidates = sample_candidates(dataset, Path(args.era_corpus), {
        'train': args.train_candidates, 'val': args.val_candidates,
    }, args.seed)
    anchors = sample_anchors(dataset, {
        'train': args.train_pairs, 'val': args.val_pairs,
    }, args.seed)
    write_jsonl(output / 'candidates.jsonl', candidates)
    write_jsonl(output / 'probes.jsonl', make_probes(candidates))
    for split, rows in anchors.items():
        write_jsonl(output / f'anchor_{split}.jsonl', rows)
    print(f'wrote {len(candidates)} candidates and '
          f'{sum(map(len, anchors.values()))} anchors -> {output}')


def command_finalize(args):
    selected = finalize_pairs(
        load_jsonl(Path(args.candidates)),
        load_jsonl(Path(args.generations)), args.model_id,
        {'train': args.train_pairs // len(MODES),
         'val': args.val_pairs // len(MODES)}, args.seed)
    output = Path(args.output_dir)
    for split, rows in selected.items():
        write_jsonl(output / f'pair_{split}.jsonl', rows)
    print(f'wrote {len(selected["train"])} train and '
          f'{len(selected["val"])} validation pairs -> {output}')


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare')
    prepare.add_argument('--dataset', required=True)
    prepare.add_argument('--era-corpus', required=True)
    prepare.add_argument('--output-dir', required=True)
    prepare.add_argument('--train-candidates', type=int, default=220)
    prepare.add_argument('--val-candidates', type=int, default=25)
    prepare.add_argument('--train-pairs', type=int, default=600)
    prepare.add_argument('--val-pairs', type=int, default=60)
    prepare.add_argument('--seed', type=int, default=1969)
    prepare.set_defaults(function=command_prepare)
    finalize = commands.add_parser('finalize')
    finalize.add_argument('--candidates', required=True)
    finalize.add_argument('--generations', required=True)
    finalize.add_argument('--model-id', required=True)
    finalize.add_argument('--output-dir', required=True)
    finalize.add_argument('--train-pairs', type=int, default=600)
    finalize.add_argument('--val-pairs', type=int, default=60)
    finalize.add_argument('--seed', type=int, default=1969)
    finalize.set_defaults(function=command_finalize)
    return parser


def main(argv=None):
    try:
        args = build_parser().parse_args(argv)
        for name in ('train_pairs', 'val_pairs'):
            value = getattr(args, name, len(MODES))
            if value <= 0 or value % len(MODES):
                raise DatasetError(f'--{name.replace("_", "-")} must be a '
                                   f'positive multiple of {len(MODES)}')
        args.function(args)
    except DatasetError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())