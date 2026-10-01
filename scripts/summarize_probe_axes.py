#!/usr/bin/env python3
"""Per-model axis rates from one or more evaluation scores.json files.

The scorer writes per-response records, not per-axis booleans, so every axis
here is a population and a predicate over that population. Keeping that
arithmetic in one place stops each caller from inventing its own.
"""
import argparse
import json
from pathlib import Path


def axis_rates(scores, model_id):
    rows = [s for s in scores if s.get('model_id') == model_id]
    post = [s for s in rows if s.get('temporal_class') == 'post_1969']
    pre = [s for s in rows if s.get('temporal_class') == 'pre_1969']
    # The gated persona population is the identity and voice probes; the wider
    # persona_eligible set measures marker density, which the corpus caps near
    # 50% by design and which no threshold should be read against.
    voice = [s for s in rows
             if s.get('category') == 'persona' and s.get('persona_eligible')]
    eligible = [s for s in rows if s.get('persona_eligible')]
    chess = [s for s in rows if s.get('chess_correct') is not None]
    axes = {
        'era_native': (sum(s['temporal_behavior'] == 'era_native_uncertainty'
                           for s in post), len(post)),
        'leak': (sum(s['leaked'] for s in post), len(post)),
        'pre_recall': (sum(len(s['expected_hits']) for s in pre),
                       sum(s['expected_total'] for s in pre)),
        'persona': (sum(bool(s.get('persona_present')) for s in voice),
                    len(voice)),
        'marking': (sum(bool(s.get('persona_present')) for s in eligible),
                    len(eligible)),
        'chess': (sum(s['chess_correct'] for s in chess), len(chess)),
    }
    return {name: value for name, value in axes.items() if value[1]}


def load_scores(path):
    blob = json.loads(Path(path).read_text())
    return blob if isinstance(blob, list) else blob.get('scores', [])


def short(model_id):
    parts = [p for p in model_id.split('-') if p not in ('q8', 'q8_0')]
    return parts[-1] if parts else model_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--scores', action='append', required=True, metavar='LABEL=PATH',
        help='Labelled scores.json to summarise (repeatable).')
    parser.add_argument(
        '--model-id', action='append',
        help='Restrict and order the columns (repeatable). Default: every '
             'model present, in file order.')
    args = parser.parse_args()

    for entry in args.scores:
        label, _, path = entry.partition('=')
        if not path:
            parser.error(f'expected LABEL=PATH, got {entry!r}')
        if not Path(path).is_file():
            print(f'\n{label}\n  missing {path}')
            continue
        scores = load_scores(path)
        present = list(dict.fromkeys(s.get('model_id') for s in scores))
        models = [m for m in (args.model_id or present) if m in present]
        rates = {m: axis_rates(scores, m) for m in models}
        axes = sorted({a for m in models for a in rates[m]})
        sizes = {a: next(rates[m][a][1] for m in models if a in rates[m])
                 for a in axes}

        print(f'\n{label}')
        print(f'  {"axis":<12}' + ''.join(f'{short(m):>8}' for m in models))
        for axis in axes:
            cells = ''
            for model in models:
                hit = rates[model].get(axis)
                cells += f'{"-":>8}' if hit is None else f'{hit[0] / hit[1]:>8.0%}'
            print(f'  {axis:<12}{cells}   n={sizes[axis]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
