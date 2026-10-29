#!/usr/bin/env python3
"""Seed p4-v3 from p4-v2, replacing the four P8-regenerated assets.

p4-v2 stays immutable. Every asset P8 does not regenerate is copied
byte-for-byte; `retain`, `era_native` and the voiced `retain_formats` and
`era_native_formats` come from the P8 generation directory.

Regenerated rows that share an 8-word shingle with any non-persona probe in
either bank are dropped here, before the dataset gate would reject the whole
build. The count is recorded per asset.
"""

import argparse
import importlib.util
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from seed_p4v2_corpus import count_rows, sha256, voiced_counts  # noqa: E402

REUSED = (
    'era_native/era_native_explanatory.jsonl',
    'persona/persona.jsonl',
    'persona/persona_controls.jsonl',
    'persona/persona_seed.jsonl',
    'persona_identity/persona_identity.jsonl',
    'persona_identity/persona_identity_controls.jsonl',
    'chess/chess.jsonl',
    'chess/positions.jsonl',
    'marker_bank/marker_bank.jsonl',
    'system_prompts.jsonl',
)
REGENERATED = {
    'retain/retain.jsonl': 'retain/retain.jsonl',
    'era_native/era_native.jsonl': 'era_native/era_native.jsonl',
    'retain_formats/retain_formats.jsonl':
        'retain_formats/retain_formats.voiced.jsonl',
    'era_native_formats/era_native_formats.jsonl':
        'era_native_formats/era_native_formats.voiced.jsonl',
}
PROBE_BANKS = ('evaluation/deepred_1969/probes.jsonl',
               'evaluation/deepred_p4/probes_ext.jsonl')


def load_evaluator():
    path = Path(__file__).resolve().parent / 'evaluate_deepred_models.py'
    spec = importlib.util.spec_from_file_location('evaluate_deepred_models', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def probe_index(evaluator, root, shingle_size=8):
    probes = []
    for bank in PROBE_BANKS:
        probes += [p for p in evaluator.load_jsonl(root / bank)
                   if p.get('category') != 'persona']
    index = evaluator.build_probe_index(probes, shingle_size)
    sizes = sorted({len(shingle.split()) for shingle in index})
    return index, sizes


def copy_filtered(origin, target, evaluator, index, sizes):
    kept = dropped = 0
    with origin.open(encoding='utf-8') as source, \
            target.open('w', encoding='utf-8') as sink:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            hit = any(
                shingle in index
                for text in evaluator.iter_record_texts(row)
                for size in sizes
                for shingle in evaluator.shingles(text, size))
            if hit:
                dropped += 1
                continue
            sink.write(line if line.endswith('\n') else line + '\n')
            kept += 1
    return kept, dropped


def seed(source, generated, output, selection, root):
    source, generated, output = Path(source), Path(generated), Path(output)
    if not source.is_dir():
        raise SystemExit(f'source corpus not found: {source}')
    if output.exists():
        raise SystemExit(f'output already exists: {output}')
    chosen = json.loads(Path(selection).read_text())
    missing = [str(generated / name) for name in REGENERATED.values()
               if not (generated / name).is_file()]
    missing += [str(source / name) for name in REUSED
                if not (source / name).is_file()]
    if missing:
        raise SystemExit('missing assets:\n  ' + '\n  '.join(missing))

    evaluator = load_evaluator()
    index, sizes = probe_index(evaluator, Path(root))
    staging = output.with_name(output.name + '.partial')
    if staging.exists():
        shutil.rmtree(staging)
    records = {}

    for name in REUSED:
        origin, target = source / name, staging / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, target)
        if sha256(origin) != sha256(target):
            raise SystemExit(f'copy mismatch: {name}')
        records[name] = {'disposition': 'reuse', 'source': str(origin),
                         'path': str(output / name), 'rows': count_rows(target),
                         'sha256': sha256(target), 'status': 'copied'}

    for name, generated_name in REGENERATED.items():
        origin, target = generated / generated_name, staging / name
        target.parent.mkdir(parents=True, exist_ok=True)
        kept, dropped = copy_filtered(origin, target, evaluator, index, sizes)
        entry = {
            'disposition': 'regenerate', 'source': str(origin),
            'source_sha256': sha256(origin), 'path': str(output / name),
            'rows': kept, 'dropped_probe_overlap': dropped,
            'sha256': sha256(target), 'status': 'created',
            'generator': chosen['generator'],
            'generator_family': chosen['family'],
            'generator_path': chosen['path'],
        }
        if generated_name.endswith('.voiced.jsonl'):
            accepted, fallback = voiced_counts(target)
            entry.update(voiced_rows=accepted, fallback_rows=fallback,
                         voiced_rate=round(accepted / max(kept, 1), 6))
        records[name] = entry

    provenance = {
        'schema_version': 1,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_corpus': str(source),
        'generated_dir': str(generated),
        'output_corpus': str(output),
        'reason': 'P8: retain, era_native and both *_formats regenerated with '
                  'the bake-off winner',
        'generator': chosen,
        'assets': records,
    }
    (staging / 'provenance.json').write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + '\n',
        encoding='utf-8')
    staging.rename(output)

    for name, entry in records.items():
        detail = f"{entry['rows']:,} rows"
        if entry.get('dropped_probe_overlap'):
            detail += f", {entry['dropped_probe_overlap']} dropped (probe overlap)"
        print(f"  {entry['disposition']:<11} {name:<52} {detail}")
    print(f"\nprovenance -> {output / 'provenance.json'}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='/mnt/data/deepred_corpus/p4-v2')
    parser.add_argument('--generated', default='/mnt/data/deepred_corpus/p8-gen')
    parser.add_argument('--output', default='/mnt/data/deepred_corpus/p4-v3')
    parser.add_argument('--selection', required=True,
                        help='selected_generator.json written by run_p8.sh')
    parser.add_argument('--repo-root', default='/mnt/data/DeepRedAI')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    seed(args.source, args.generated, args.output, args.selection,
         args.repo_root)


if __name__ == '__main__':
    main()
