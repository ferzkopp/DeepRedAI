#!/usr/bin/env python3
"""Seed the p4-v1 corpus from p3-v5 and record what came from where.

Implements DeepRed-Phase4-Plan.md P5.10. Three dispositions, matching what the
rework decided for each asset:

  reuse      copied byte-for-byte from the source corpus
  revise     copied so it can be edited in place, then rewritten
  create     not copied; written fresh by a later step

Copy, never symlink: generation appends to these files and a link would grow
the source corpus. Every asset is hashed on both sides so a later regression
can be attributed without re-deriving provenance.

Usage:
  python3 scripts/seed_p4_corpus.py \\
      --source /mnt/data/deepred_corpus/p3-v5 \\
      --output /mnt/data/deepred_corpus/p4-v1
"""

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

# Disposition per asset, with the reason it is not simply reused.
ASSETS = {
    'retain/retain.jsonl': ('reuse', 'pre-1969 facts; unchanged for Phase 4'),
    'era_native/era_native.jsonl': ('reuse', 'temporal behaviour; unchanged'),
    'era_native_formats/era_native_formats.jsonl': ('reuse', 'format attacks'),
    'retain_formats/retain_formats.jsonl': ('reuse', 'format attacks'),
    'persona/persona_controls.jsonl': (
        'reuse', 'must stay paired with the questions they contrast against'),
    'persona/persona_seed.jsonl': ('reuse', 'seed questions held constant'),
    'persona_identity/persona_identity.jsonl': ('reuse', 'identity answers'),
    'persona_identity/persona_identity_controls.jsonl': ('reuse', 'controls'),
    'chess/chess.jsonl': (
        'reuse', 'already SAN-bearing; P5.3 may replace it for breadth'),
    'system_prompts.jsonl': (
        'revise', 'P5.1 rewrites for the native system channel'),
    'persona/persona.jsonl': (
        'create', 'P5.6 regenerates with the P4 winner'),
    'marker_bank/marker_bank.jsonl': (
        'create', 'P5.7 rebuilds with the readability pass'),
    'length/length.jsonl': (
        'create', 'P5.2 generates the 60-150 word slice'),
}

# Copied from wherever they exist, not tied to the source corpus version.
EXTRA_SEARCH = ('chess/positions.jsonl',)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def count_rows(path):
    with path.open('rb') as handle:
        return sum(1 for line in handle if line.strip())


def find_first(roots, relative):
    for root in roots:
        candidate = Path(root) / relative
        if candidate.is_file() and candidate.stat().st_size:
            return candidate
    return None


def seed(source, output, extra_roots, force):
    source, output = Path(source), Path(output)
    if not source.is_dir():
        raise SystemExit(f'source corpus not found: {source}')
    if output.exists() and not force:
        raise SystemExit(f'{output} exists; pass --force to refresh it')
    output.mkdir(parents=True, exist_ok=True)

    records = {}
    for relative, (disposition, reason) in sorted(ASSETS.items()):
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        origin = source / relative
        entry = {'disposition': disposition, 'reason': reason,
                 'path': str(target)}

        if disposition == 'create':
            entry['status'] = 'present' if target.is_file() else 'pending'
            if origin.is_file():
                # Recorded but deliberately not copied: a stale copy here would
                # silently train on the asset this phase set out to replace.
                entry['superseded_source'] = str(origin)
                entry['superseded_rows'] = count_rows(origin)
        elif not origin.is_file():
            entry['status'] = 'missing_in_source'
        elif target.is_file() and not force:
            entry['status'] = 'already_present'
        else:
            shutil.copy2(origin, target)
            entry.update(
                status='copied', source=str(origin),
                rows=count_rows(target), sha256=sha256(target))
            if sha256(origin) != entry['sha256']:
                raise SystemExit(f'copy mismatch for {relative}')
        records[relative] = entry

    for relative in EXTRA_SEARCH:
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        origin = find_first([source, *extra_roots], relative)
        if origin is None:
            records[relative] = {'disposition': 'reuse',
                                 'status': 'missing_in_source'}
            continue
        if not (target.is_file() and not force):
            shutil.copy2(origin, target)
        records[relative] = {
            'disposition': 'reuse', 'reason': 'position index for P5.3 FEN join',
            'status': 'copied', 'source': str(origin), 'path': str(target),
            'rows': count_rows(target), 'sha256': sha256(target)}

    provenance = {
        'schema_version': 1,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_corpus': str(source),
        'output_corpus': str(output),
        'assets': records,
    }
    (output / 'provenance.json').write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + '\n',
        encoding='utf-8')

    width = max(len(name) for name in records)
    for name, entry in sorted(records.items()):
        rows = entry.get('rows')
        detail = f"{rows:>6,} rows" if rows else f"{entry['status']:>11}"
        print(f"  {entry['disposition']:<7} {name:<{width}}  {detail}")
    pending = [n for n, e in records.items() if e.get('status') == 'pending']
    print(f"\nprovenance -> {output / 'provenance.json'}")
    if pending:
        print(f'awaiting generation: {", ".join(sorted(pending))}')
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--extra-root', action='append', default=[],
                        help='Additional corpus root searched for the position '
                             'index (repeatable).')
    parser.add_argument('--force', action='store_true',
                        help='Re-copy assets that already exist in the output.')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return seed(args.source, args.output, args.extra_root, args.force)


if __name__ == '__main__':
    raise SystemExit(main())
