#!/usr/bin/env python3
"""Audit a generated DeepRed Phase 2 corpus before it is turned into a dataset.

Phase 1 failed on data defects that were only visible after ten days of
training: Wikipedia boilerplate in targets, a single refusal template that
became a global prior, and no persona at all. This script checks for those
specific failures plus the usual hygiene, and exits non-zero on any hard
failure so it can gate a runbook step.

Usage:
    python3 scripts/audit_deepred_corpus.py --corpus-dir /mnt/data/deepred_corpus/v2
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

KINDS = ('forget', 'retain', 'era_native', 'persona')
ERA_MODES = ('in_world', 'hedged', 'premise_correction')

# Finding 6: dump structure leaked into answers and worsened with training.
BOILERPLATE = re.compile(
    r'##\s*(See also|References|External links|Further reading|Notes)'
    r'|^\s*Categories:'
    r'|\[\[|\{\{|<ref[ >]', re.I | re.M)
IDENTITY_LEAK = re.compile(r'\b(gemma|google|deepmind|openai|large language model)\b', re.I)
POST_CUTOFF_YEAR = re.compile(r'\b(19[7-9]\d|20\d\d)\b')
CHESS_FOOTER = re.compile(r'^\[DR:.+\]$', re.M)


def normalize(text):
    return ' '.join(re.sub(r'[^a-z0-9]+', ' ', text.lower()).split())


def opening(text, words=6):
    return ' '.join(normalize(text).split()[:words])


class Report:
    def __init__(self):
        self.failures = []
        self.warnings = []

    def check(self, ok, message, hard=True):
        if ok:
            print(f'  PASS  {message}')
        elif hard:
            self.failures.append(message)
            print(f'  FAIL  {message}')
        else:
            self.warnings.append(message)
            print(f'  WARN  {message}')

    def stat(self, message):
        print(f'  ----  {message}')


def load_jsonl(path, label, every=5000):
    """Stream a JSONL file, reporting progress on long files."""
    rows = []
    if not path.is_file():
        return rows
    with path.open(encoding='utf-8') as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f'{path}:{number}: invalid JSON: {exc}')
            if number % every == 0:
                print(f'  ... {label}: {number} lines read', flush=True)
    return rows


def qa(record):
    messages = record.get('messages') or []
    if len(messages) < 2:
        return None, None
    return messages[0].get('content', ''), messages[-1].get('content', '')


def audit_common(kind, rows, report, args):
    print(f'\n=== {kind}: {len(rows)} records ===')
    if not rows:
        report.check(False, f'{kind}: file missing or empty')
        return

    ids = [r.get('id') for r in rows]
    report.check(len(ids) == len(set(ids)),
                 f'{kind}: unique ids ({len(ids) - len(set(ids))} duplicates)')

    malformed = [r.get('id') for r in rows
                 if (r.get('messages') or [{}])[0].get('role') != 'user'
                 or (r.get('messages') or [{}])[-1].get('role') != 'assistant']
    report.check(not malformed, f'{kind}: message roles well formed '
                                f'({len(malformed)} malformed)')

    answers = [qa(r)[1] or '' for r in rows]
    questions = [qa(r)[0] or '' for r in rows]

    boiler = sum(1 for a in answers if BOILERPLATE.search(a))
    report.check(boiler == 0,
                 f'{kind}: Wikipedia boilerplate in {boiler}/{len(rows)} answers')

    empty = sum(1 for a in answers if len(a.split()) < 3)
    report.check(empty == 0, f'{kind}: {empty} answers shorter than 3 words')

    dup_a = len(answers) - len({normalize(a) for a in answers})
    dup_ratio = dup_a / len(answers)
    report.check(dup_ratio <= args.max_duplicate_rate,
                 f'{kind}: exact duplicate answers {dup_a} '
                 f'({dup_ratio:.1%} <= {args.max_duplicate_rate:.0%})')

    dup_q = len(questions) - len({normalize(q) for q in questions})
    report.check(dup_q / len(questions) <= args.max_duplicate_rate,
                 f'{kind}: exact duplicate questions {dup_q}', hard=False)

    # Template collapse is the Phase 1 failure mode: one opening dominating.
    openings = Counter(opening(a) for a in answers)
    top_phrase, top_count = openings.most_common(1)[0]
    share = top_count / len(answers)
    report.check(share <= args.max_opening_share,
                 f'{kind}: most common opening {share:.1%} '
                 f'(<= {args.max_opening_share:.0%}) -> "{top_phrase[:50]}"')
    report.stat(f'{kind}: {len(openings)} distinct openings, '
                f'median answer {sorted(len(a.split()) for a in answers)[len(answers) // 2]} words')


def audit_holdout(kind, rows, holdout, is_held_out, report):
    hits = [r.get('id') for r in rows
            if is_held_out(' '.join(m.get('content', '') for m in r['messages']), holdout)]
    report.check(not hits, f'{kind}: {len(hits)} records touch held-out probe facts '
                           f'{hits[:5]}')


def audit_era_native(rows, report, args):
    modes = Counter(r.get('mode') for r in rows)
    total = sum(modes[m] for m in ERA_MODES)
    report.stat(f'era_native: mode balance {dict(modes)}')
    if total:
        low = min(modes[m] for m in ERA_MODES) / total
        report.check(low >= args.min_mode_share,
                     f'era_native: rarest mode {low:.1%} of records '
                     f'(>= {args.min_mode_share:.0%})')
    leaks = [r.get('id') for r in rows if POST_CUTOFF_YEAR.search(qa(r)[1] or '')]
    report.check(not leaks,
                 f'era_native: {len(leaks)} answers cite a post-cutoff year')


def audit_persona(rows, controls, report, args, positions_ok):
    ident = [r.get('id') for r in rows if IDENTITY_LEAK.search(qa(r)[1] or '')]
    report.check(not ident, f'persona: {len(ident)} answers mention the base model')

    dates = [r.get('id') for r in rows if POST_CUTOFF_YEAR.search(qa(r)[1] or '')]
    report.check(not dates, f'persona: {len(dates)} answers invent a modern date')

    annotated = [r for r in rows if r.get('chess')]
    rate = len(annotated) / len(rows) if rows else 0
    report.stat(f'persona: chess annotation on {len(annotated)}/{len(rows)} ({rate:.0%})')
    footers = sum(1 for r in rows if CHESS_FOOTER.search(qa(r)[1] or ''))
    report.check(footers == len(annotated),
                 f'persona: {footers} footers vs {len(annotated)} chess records')

    if positions_ok is not None:
        report.check(positions_ok, 'persona: all annotated FENs are legal positions')

    report.check(bool(controls), f'persona: {len(controls)} plain-answer controls present')
    if controls:
        ratio = len(controls) / len(rows)
        report.check(ratio >= args.min_control_ratio,
                     f'persona: control ratio {ratio:.0%} '
                     f'(>= {args.min_control_ratio:.0%})')
        bad = [c.get('id') for c in controls if CHESS_FOOTER.search(qa(c)[1] or '')]
        report.check(not bad, f'persona: {len(bad)} controls carry a chess footer')
        paired = sum(1 for c in controls if c.get('pair_id'))
        report.stat(f'persona: {paired}/{len(controls)} controls linked to a persona record')


def validate_fens(rows):
    try:
        import chess
    except ImportError:
        return None
    for record in rows:
        meta = record.get('chess')
        if meta and not chess.Board(meta['fen']).is_valid():
            return False
    return True


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus-dir', default='/mnt/data/deepred_corpus/v2')
    parser.add_argument('--probes', default='evaluation/deepred_1969/probes.jsonl')
    parser.add_argument('--kind', action='append', choices=KINDS,
                        help='audit only these kinds (default: all present)')
    parser.add_argument('--min-records', type=int, default=1)
    parser.add_argument('--max-duplicate-rate', type=float, default=0.02)
    parser.add_argument('--max-opening-share', type=float, default=0.15)
    parser.add_argument('--min-mode-share', type=float, default=0.20)
    parser.add_argument('--min-control-ratio', type=float, default=0.15)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from generate_deepred_corpus import load_holdout, is_held_out

    root = Path(args.corpus_dir)
    holdout = load_holdout(args.probes)
    print(f'corpus: {root}')
    print(f'holdout terms: {len(holdout)}')

    report = Report()
    audited = 0
    for kind in (args.kind or KINDS):
        path = root / kind / f'{kind}.jsonl'
        rows = load_jsonl(path, kind)
        if not rows and not args.kind:
            print(f'\n=== {kind}: not generated yet, skipped ===')
            continue
        audited += 1
        audit_common(kind, rows, report, args)
        if not rows:
            continue
        report.check(len(rows) >= args.min_records,
                     f'{kind}: {len(rows)} records (>= {args.min_records})')
        audit_holdout(kind, rows, holdout, is_held_out, report)
        if kind == 'era_native':
            audit_era_native(rows, report, args)
        if kind == 'persona':
            controls = load_jsonl(root / 'persona' / 'persona_controls.jsonl',
                                  'persona_controls')
            audit_persona(rows, controls, report, args, validate_fens(rows))

    # An empty corpus must not pass a gating step by having nothing to check.
    report.check(audited > 0, f'{root}: no corpus files found to audit')

    print('\n=== summary ===')
    print(f'failures: {len(report.failures)}  warnings: {len(report.warnings)}')
    for message in report.failures:
        print(f'  FAIL  {message}')
    for message in report.warnings:
        print(f'  WARN  {message}')
    return 1 if report.failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
