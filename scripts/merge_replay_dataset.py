#!/usr/bin/env python3
"""Merge a small targeted dataset with stratified replay from a trained one.

A warm-started repair run must spend most of its steps on the new assets
without drifting off the distribution the starting checkpoint learned. Replay
rows are drawn from that checkpoint's own dataset, per kind and in proportion,
so they are exactly what it was trained on - system prompts and markers
included.

Usage:
    python3 scripts/merge_replay_dataset.py \\
        --new /mnt/data/sft_corpus/deepred-p4v5-new \\
        --replay /mnt/data/sft_corpus/deepred-p4v3 \\
        --replay-ratio 1.5 --output-dir /mnt/data/sft_corpus/deepred-p4v5 \\
        --tokenizer /mnt/data/models/gemma-4-12b-it --force
"""

import argparse
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_deepred_dataset import (  # noqa: E402
    MARKER_PRESENT, contingency_table, target_token_stats, write_jsonl,
)


class MergeError(ValueError):
    pass


def load_jsonl(path):
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stratified_sample(rows, count, seed):
    """`count` rows with each kind at its share of `rows`, deterministically.

    Largest-remainder rounding keeps the total exact, so the replay ratio the
    manifest reports is the ratio trained on.
    """
    if count >= len(rows):
        return list(rows)
    by_kind = defaultdict(list)
    for row in rows:
        by_kind[row['kind']].append(row)
    exact = {kind: count * len(group) / len(rows) for kind, group in by_kind.items()}
    quota = {kind: int(value) for kind, value in exact.items()}
    spare = count - sum(quota.values())
    for kind in sorted(exact, key=lambda k: (-(exact[k] - quota[k]), k))[:spare]:
        quota[kind] += 1
    chosen = []
    for kind in sorted(by_kind):
        group = sorted(by_kind[kind], key=lambda row: row['id'])
        random.Random(f'{seed}:{kind}').shuffle(group)
        chosen.extend(group[:quota[kind]])
    return chosen


def system_marker_ratio(rows):
    marked = unmarked = 0
    for row in rows:
        if not row.get('system_variant') or row['kind'].endswith('_controls'):
            continue
        if MARKER_PRESENT.search(row['messages'][-1]['content']):
            marked += 1
        else:
            unmarked += 1
    return marked / max(unmarked, 1)


def merge(new_dir, replay_dir, ratio, seed):
    new_train = load_jsonl(Path(new_dir) / 'retain_train.jsonl')
    new_val = load_jsonl(Path(new_dir) / 'retain_val.jsonl')
    replay_train = load_jsonl(Path(replay_dir) / 'retain_train.jsonl')
    replay_val = load_jsonl(Path(replay_dir) / 'retain_val.jsonl')
    if not new_train:
        raise MergeError(f'{new_dir} has no training rows')
    replay_n = round(len(new_train) * ratio)
    train = new_train + stratified_sample(replay_train, replay_n, seed)
    # Validation keeps the replay dataset's own val/train proportion, so the
    # loss curve weighs old and new behaviour as training does.
    val_n = round(len(replay_val) * replay_n / max(len(replay_train), 1))
    val = new_val + stratified_sample(replay_val, val_n, seed + 1)

    ids = Counter(row['id'] for row in train + val)
    duplicates = [key for key, value in ids.items() if value > 1]
    if duplicates:
        raise MergeError(f'{len(duplicates)} duplicate row ids, e.g. {duplicates[0]}')
    crossing = ({row['content_id'] for row in train}
                & {row['content_id'] for row in val})
    if crossing:
        raise MergeError(f'{len(crossing)} content ids cross train/val')
    return train, val


def write_dataset(output, train, val, meta, tokenizer):
    write_jsonl(output / 'retain_train.jsonl', train)
    write_jsonl(output / 'retain_val.jsonl', val)
    for split in ('train', 'val'):
        (output / f'forget_{split}.jsonl').write_text('', encoding='utf-8')
    counts = Counter((row['kind'], split) for split, rows in
                     (('train', train), ('val', val)) for row in rows)
    manifest = {
        'schema_version': 1,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        **meta,
        'counts': {f'{kind}_{split}': n for (kind, split), n in sorted(counts.items())},
        'train_rows': len(train), 'val_rows': len(val),
        'system_conditioned_share': round(
            sum(bool(row.get('system_variant')) for row in train) / len(train), 4),
        'system_marker_ratio': round(system_marker_ratio(train), 3),
        'contingency_table': contingency_table(train),
        'target_length': target_token_stats(train, tokenizer),
        'system_variant_counts': dict(sorted(Counter(
            row['system_variant'] for row in train
            if row.get('system_variant')).items())),
    }
    (output / 'manifest.json').write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--new', required=True, help='dataset of the new kinds')
    parser.add_argument('--replay', required=True,
                        help="the starting checkpoint's own dataset")
    parser.add_argument('--replay-ratio', type=float, default=1.5,
                        help='replay train rows per new train row')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--seed', type=int, default=1969)
    parser.add_argument('--tokenizer')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args(argv)

    output = Path(args.output_dir)
    if output.exists():
        if not args.force:
            parser.error(f'{output} exists; use --force to replace it')
        shutil.rmtree(output)
    try:
        train, val = merge(args.new, args.replay, args.replay_ratio, args.seed)
    except MergeError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    output.mkdir(parents=True)
    manifest = write_dataset(output, train, val, {
        'new_dataset': str(Path(args.new)), 'replay_dataset': str(Path(args.replay)),
        'replay_ratio': args.replay_ratio, 'seed': args.seed,
    }, args.tokenizer)
    print(f"merged {manifest['train_rows']:,} train / {manifest['val_rows']:,} val "
          f"-> {output}")
    print(f"  system-prompted marker ratio {manifest['system_marker_ratio']:.2f}:1, "
          f"mean sequence tokens {manifest['target_length'].get('mean_sequence_tokens')}")
    for key, value in manifest['counts'].items():
        print(f'  {key:40s} {value:6d}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
