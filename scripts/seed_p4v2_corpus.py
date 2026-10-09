#!/usr/bin/env python3
"""Seed p4-v2 from p4-v1, promoting validated voiced assets.

The p4-v1 corpus remains immutable. Assets not changed by the p4-v1 diagnostic
are copied byte-for-byte; the three restyled outputs become the canonical
p4-v2 format and chess assets.
"""

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ASSETS = {
    'retain/retain.jsonl': 'retain/retain.jsonl',
    'era_native/era_native.jsonl': 'era_native/era_native.jsonl',
    'era_native/era_native_explanatory.jsonl': (
        'era_native/era_native_explanatory.jsonl'),
    'era_native_formats/era_native_formats.jsonl': (
        'era_native_formats/era_native_formats.voiced.jsonl'),
    'retain_formats/retain_formats.jsonl': (
        'retain_formats/retain_formats.voiced.jsonl'),
    'persona/persona.jsonl': 'persona/persona.jsonl',
    'persona/persona_controls.jsonl': 'persona/persona_controls.jsonl',
    'persona/persona_seed.jsonl': 'persona/persona_seed.jsonl',
    'persona_identity/persona_identity.jsonl': (
        'persona_identity/persona_identity.jsonl'),
    'persona_identity/persona_identity_controls.jsonl': (
        'persona_identity/persona_identity_controls.jsonl'),
    'chess/chess.jsonl': 'chess/chess.voiced.jsonl',
    'chess/positions.jsonl': 'chess/positions.jsonl',
    'marker_bank/marker_bank.jsonl': 'marker_bank/marker_bank.jsonl',
    'system_prompts.jsonl': 'system_prompts.jsonl',
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def count_rows(path):
    with path.open('rb') as handle:
        return sum(1 for line in handle if line.strip())


def voiced_counts(path):
    accepted = fallback = 0
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('voiced') is True:
                accepted += 1
            elif row.get('voiced') is False:
                fallback += 1
            else:
                raise SystemExit(f'{path}: row missing boolean voiced status')
    return accepted, fallback


def seed(source, output):
    source, output = Path(source), Path(output)
    if not source.is_dir():
        raise SystemExit(f'source corpus not found: {source}')
    if output.exists():
        raise SystemExit(f'output already exists: {output}')

    records = {}
    for target_name, source_name in ASSETS.items():
        origin = source / source_name
        target = output / target_name
        if not origin.is_file():
            raise SystemExit(f'missing source asset: {origin}')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, target)
        source_hash = sha256(origin)
        target_hash = sha256(target)
        if source_hash != target_hash:
            raise SystemExit(f'copy mismatch: {target_name}')

        promoted = source_name.endswith('.voiced.jsonl')
        entry = {
            'disposition': 'promote_voiced' if promoted else 'reuse',
            'source': str(origin),
            'path': str(target),
            'rows': count_rows(target),
            'sha256': target_hash,
            'status': 'copied',
        }
        if promoted:
            accepted, fallback = voiced_counts(target)
            entry.update(
                generator='qwen3-30b-a3b',
                voiced_rows=accepted,
                fallback_rows=fallback,
                voiced_rate=round(accepted / (accepted + fallback), 6),
            )
        records[target_name] = entry

    provenance = {
        'schema_version': 1,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_corpus': str(source),
        'output_corpus': str(output),
        'reason': 'p4-v2 corrected-tokenizer pilot with validated voiced assets',
        'assets': records,
    }
    provenance_path = output / 'provenance.json'
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + '\n',
        encoding='utf-8')

    for name, entry in records.items():
        detail = f"{entry['rows']:,} rows"
        if entry['disposition'] == 'promote_voiced':
            detail += f", {entry['voiced_rows']:,} voiced"
        print(f"  {entry['disposition']:<15} {name:<55} {detail}")
    print(f'\nprovenance -> {provenance_path}')


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--source', default='/mnt/data/deepred_corpus/p4-v1')
    parser.add_argument(
        '--output', default='/mnt/data/deepred_corpus/p4-v2')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    seed(args.source, args.output)


if __name__ == '__main__':
    main()
