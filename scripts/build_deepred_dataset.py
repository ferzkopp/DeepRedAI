#!/usr/bin/env python3
"""Build deterministic Phase 2 NPO/retain datasets from audited corpus files."""

import argparse
import hashlib
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


KINDS = ('forget', 'retain', 'era_native', 'persona', 'persona_controls')
BOILERPLATE = re.compile(
    r'##\s*(See also|References|External links|Further reading|Notes)'
    r'|^\s*Categories:|\[\[|\{\{|<ref[ >]', re.I | re.M)
# Retrieved move footers are injected at serve time, never learned.
CHESS_FOOTER = re.compile(r'\s*\[DR:[^\]]*\]')
SPACE = re.compile(r'\s+')


class DatasetError(ValueError):
    pass


def stable_hash(value):
    encoded = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(',', ':')
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def normalize(text):
    return SPACE.sub(' ', (text or '').strip())


def content_id(messages):
    canonical = [
        {'role': message.get('role'), 'content': normalize(message.get('content'))}
        for message in messages
    ]
    return stable_hash(canonical)


def read_kind(root, kind, strip_boilerplate=False, strip_chess_footer=False):
    directory = 'persona' if kind == 'persona_controls' else kind
    path = root / directory / f'{kind}.jsonl'
    if not path.is_file():
        raise DatasetError(f'missing corpus file: {path}')
    rows = []
    with path.open(encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                source = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f'{path}:{line_number}: {exc}') from exc
            messages = source.get('messages')
            if not isinstance(messages, list) or len(messages) < 2:
                raise DatasetError(f'{path}:{line_number}: invalid messages')
            clean_messages = []
            for message in messages:
                role = message.get('role')
                text = normalize(message.get('content'))
                if role not in {'system', 'user', 'assistant'} or not text:
                    raise DatasetError(f'{path}:{line_number}: invalid message')
                if role == 'assistant' and strip_chess_footer:
                    text = normalize(CHESS_FOOTER.sub('', text))
                    if not text:
                        raise DatasetError(
                            f'{path}:{line_number}: target empty after footer removal')
                if role == 'assistant' and BOILERPLATE.search(text):
                    if not strip_boilerplate:
                        raise DatasetError(
                            f'{path}:{line_number}: boilerplate in target')
                    text = BOILERPLATE.sub('', text).strip()
                    if not text or BOILERPLATE.search(text):
                        raise DatasetError(
                            f'{path}:{line_number}: target empty after cleanup')
                clean_messages.append({'role': role, 'content': text})
            source_id = source.get('id')
            if not isinstance(source_id, str) or not source_id:
                raise DatasetError(f'{path}:{line_number}: missing stable id')
            rows.append({
                'id': f'{kind}:{source_id}',
                'source_id': source_id,
                'content_id': content_id(clean_messages),
                'kind': kind,
                'messages': clean_messages,
            })
    return rows, path


def assign_splits(rows, val_fraction, seed):
    """Assign content groups before any per-kind sampling."""
    groups = defaultdict(list)
    for row in rows:
        groups[row['content_id']].append(row)
    assignments = {}
    for group_id in sorted(groups):
        value = int(stable_hash([seed, group_id])[:16], 16) / 16**16
        assignments[group_id] = 'val' if value < val_fraction else 'train'
    return assignments


def sample_rows(rows, assignments, limits, seed):
    selected = []
    by_kind = defaultdict(list)
    for row in rows:
        by_kind[row['kind']].append(row)
    for kind, candidates in sorted(by_kind.items()):
        limit = limits.get(kind)
        rng = random.Random(f'{seed}:{kind}')
        rng.shuffle(candidates)
        selected.extend(candidates if limit is None else candidates[:limit])
    return selected


def write_jsonl(path, rows):
    with path.open('w', encoding='utf-8') as handle:
        for row in sorted(rows, key=lambda item: item['id']):
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + '\n')


def load_system_variants(path, holdout=()):
    variants = []
    seen = set()
    with Path(path).open(encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f'{path}:{line_number}: {exc}') from exc
            variant_id = record.get('id')
            text = normalize(record.get('text'))
            if not isinstance(variant_id, str) or not variant_id or not text:
                raise DatasetError(f'{path}:{line_number}: invalid variant')
            if variant_id in seen:
                raise DatasetError(f'{path}:{line_number}: duplicate variant id')
            seen.add(variant_id)
            if variant_id in set(holdout):
                continue
            variants.append({'id': variant_id, 'text': text})
    unknown = sorted(set(holdout) - seen)
    if unknown:
        raise DatasetError(f'unknown held-out variant ids: {unknown}')
    if not variants:
        raise DatasetError(f'{path}: no usable system prompt variants')
    return variants


def unit_hash(*parts):
    return int(stable_hash(list(parts))[:16], 16) / 16**16


def apply_system_prompts(rows, variants, coverage, seed):
    """Attach a deterministic system prompt to a reproducible subset of rows."""
    if not 0 <= coverage <= 1:
        raise DatasetError('--system-coverage must be between 0 and 1')
    for row in rows:
        if row['messages'][0]['role'] == 'system':
            raise DatasetError(f'{row["id"]}: source row already has a system message')
        if unit_hash(seed, 'system-coverage', row['id']) >= coverage:
            row['system_variant'] = None
            continue
        index = int(stable_hash([seed, 'system-variant', row['id']])[:16], 16)
        variant = variants[index % len(variants)]
        row['messages'] = [
            {'role': 'system', 'content': variant['text']}, *row['messages']]
        row['system_variant'] = variant['id']
    return rows


def parse_limits(values):
    limits = {}
    for value in values or []:
        try:
            kind, count = value.split('=', 1)
            count = int(count)
        except ValueError as exc:
            raise DatasetError(f'invalid --limit {value!r}; expected KIND=N') from exc
        if kind not in KINDS or count < 0:
            raise DatasetError(f'invalid --limit {value!r}')
        limits[kind] = count
    return limits


def build(args):
    if not 0 < args.val_fraction < 1:
        raise DatasetError('--val-fraction must be between 0 and 1')
    root = Path(args.corpus_dir)
    output = Path(args.output_dir)
    if output.exists():
        if not args.force:
            raise DatasetError(f'{output} exists; use --force to replace it')
        shutil.rmtree(output)
    output.mkdir(parents=True)

    all_rows = []
    source_paths = {}
    for kind in KINDS:
        rows, path = read_kind(
            root, kind, args.strip_boilerplate, args.strip_chess_footer)
        all_rows.extend(rows)
        source_paths[kind] = str(path)
    duplicate_ids = len(all_rows) - len({row['id'] for row in all_rows})
    if duplicate_ids:
        raise DatasetError(f'{duplicate_ids} duplicate stable ids')

    assignments = assign_splits(all_rows, args.val_fraction, args.seed)
    rows = sample_rows(all_rows, assignments, parse_limits(args.limit), args.seed)
    variants = []
    if args.system_prompt_file:
        variants = load_system_variants(
            args.system_prompt_file, args.hold_out_system_variant or ())
        apply_system_prompts(rows, variants, args.system_coverage, args.seed)
    buckets = defaultdict(list)
    for row in rows:
        objective = 'forget' if row['kind'] == 'forget' else 'retain'
        buckets[(objective, assignments[row['content_id']])].append(row)

    train_ids = {row['content_id'] for key, values in buckets.items()
                 if key[1] == 'train' for row in values}
    val_ids = {row['content_id'] for key, values in buckets.items()
               if key[1] == 'val' for row in values}
    overlap = train_ids & val_ids
    if overlap and args.fail_on_cross_split_duplicates:
        raise DatasetError(f'{len(overlap)} content ids cross train/val splits')

    paths = {}
    for objective in ('forget', 'retain'):
        for split in ('train', 'val'):
            path = output / f'{objective}_{split}.jsonl'
            write_jsonl(path, buckets[(objective, split)])
            paths[f'{objective}_{split}'] = str(path)

    counts = Counter((row['kind'], assignments[row['content_id']]) for row in rows)
    manifest = {
        'schema_version': 1,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'corpus_dir': str(root),
        'source_paths': source_paths,
        'seed': args.seed,
        'val_fraction': args.val_fraction,
        'split_before_sampling': True,
        'limits': parse_limits(args.limit),
        'counts': {f'{kind}_{split}': counts[(kind, split)]
                   for kind in KINDS for split in ('train', 'val')},
        'paths': paths,
        'cross_split_content_ids': len(overlap),
        'strip_chess_footer': bool(args.strip_chess_footer),
        'system_prompt_file': args.system_prompt_file,
        'system_coverage': args.system_coverage if args.system_prompt_file else 0,
        'held_out_system_variants': sorted(args.hold_out_system_variant or ()),
        'system_variant_counts': dict(sorted(Counter(
            row.get('system_variant') for row in rows
            if row.get('system_variant')).items())),
    }
    (output / 'manifest.json').write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(f'wrote {len(rows):,} records -> {output}')
    print(f'counts: {manifest["counts"]}')
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--val-fraction', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=1969)
    parser.add_argument('--limit', action='append', help='Per-kind cap KIND=N')
    parser.add_argument('--strip-boilerplate', action='store_true')
    parser.add_argument('--strip-chess-footer', action='store_true')
    parser.add_argument('--system-prompt-file')
    parser.add_argument('--system-coverage', type=float, default=1.0)
    parser.add_argument('--hold-out-system-variant', action='append')
    parser.add_argument('--fail-on-cross-split-duplicates', action='store_true')
    parser.add_argument('--force', action='store_true')
    return parser


def main(argv=None):
    try:
        build(build_parser().parse_args(argv))
    except DatasetError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())