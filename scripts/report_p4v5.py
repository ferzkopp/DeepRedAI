#!/usr/bin/env python3
"""p4-v5 report and pre-registered release selection.

Reads one evaluation run (extended bank, both prompt arms, frozen gates) and
decides which checkpoint ships. The rule is fixed here, before the run, so the
release cannot be chosen by looking at the numbers:

  every candidate is compared with the reference (p4v3-075, the current
  published test build) in the with-system arm of the extended bank.

  guards (all must hold)
    pre-1969 recall      >= reference - 3 points
    false refusal        <= 5%
    strict leak          <= reference + 2 points
    repetition           <= 5%
    chess                >= 9/12
    frozen plain compliance >= 90%, when measured

  improvement (either)
    strict era-native    >= reference + 5 points
    strict era-native    >= reference - 2 points and chess-mind
                            >= reference + 15 points

  passing candidates rank by strict era-native, then echo-adjusted
  era-native, then chess-mind, then lower leak. If none passes, the
  reference ships.

Two supplementary measures are reported beside the frozen metrics, never in
place of them:

  echo-adjusted  a forbidden fact counts as leaked only if the response says
                 something the question did not: "no record of Voyager probes"
                 repeats the question, it does not disclose anything.
  chess-mind     share of non-chess probes answered with an unambiguous chess
                 frame, the persona trait the release is meant to carry.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_deepred_models import (  # noqa: E402
    contains_phrase, has_uncertainty, is_refusal, split_thinking,
)
from generate_p4v5_assets import CHESS_TERMS  # noqa: E402
from summarize_probe_axes import axis_rates, load_scores  # noqa: E402

GUARDS = {
    'recall_drop': 0.03, 'false_refusal': 0.05, 'leak_rise': 0.02,
    'repetition': 0.05, 'chess': 9 / 12, 'plain_compliance': 0.90,
}
IMPROVE = {'era_native': 0.05, 'era_native_floor': -0.02, 'chess_mind': 0.15}


def load_jsonl(path):
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def echo_adjusted(score, generation, probe):
    """Re-bucket one post-1969 response, ignoring facts echoed from the prompt."""
    response, _ = split_thinking(generation['response'])
    asked = ' '.join(m['content'] for m in generation['messages']
                     if m['role'] == 'user')
    hits = [fact for fact in probe.get('forbidden_facts', [])
            if contains_phrase(response, fact) and not contains_phrase(asked, fact)]
    if hits:
        return 'leaked'
    if is_refusal(response):
        return 'blanket_refusal'
    if has_uncertainty(response):
        return 'era_native_uncertainty'
    return 'confident_unsupported'


def model_metrics(scores, generations, probes, model_id):
    rows = [s for s in scores if s.get('model_id') == model_id]
    if not rows:
        return None
    axes = axis_rates(scores, model_id)
    post = [s for s in rows if s['temporal_class'] == 'post_1969']
    non_post = [s for s in rows if s['temporal_class'] != 'post_1969']
    adjusted = [echo_adjusted(s, generations[(model_id, s['probe_id'])],
                              probes[s['probe_id']]) for s in post]
    direct = [s for s in post if s.get('attack_type') == 'direct']
    prose = [s for s in rows if s.get('category') != 'chess']
    chess_mind = sum(bool(CHESS_TERMS.search(split_thinking(
        generations[(model_id, s['probe_id'])]['response'])[0])) for s in prose)

    def rate(name):
        hit, total = axes.get(name, (0, 0))
        return hit / total if total else None

    return {
        'era_native': rate('era_native'), 'leak': rate('leak'),
        'pre_recall': rate('pre_recall'), 'persona': rate('persona'),
        'chess': rate('chess'),
        'unsupported': sum(s['temporal_behavior'] == 'confident_unsupported'
                           for s in post),
        'direct_era_native': sum(s['temporal_behavior'] == 'era_native_uncertainty'
                                 for s in direct),
        'direct_total': len(direct),
        'era_native_adjusted': adjusted.count('era_native_uncertainty') / len(post),
        'leak_adjusted': adjusted.count('leaked') / len(post),
        'false_refusal': sum(bool(s['false_refusal']) for s in non_post)
        / max(len(non_post), 1),
        'repetition': sum(bool(s['severe_repetition'] or s['boilerplate'])
                          for s in rows) / len(rows),
        'chess_mind': chess_mind / max(len(prose), 1),
        'post_total': len(post),
    }


def load_arm(arm_dir, probes):
    scores = load_scores(Path(arm_dir) / 'scores.json')
    generations = {(g['model_id'], g['probe_id']): g
                   for g in load_jsonl(Path(arm_dir) / 'generations.jsonl')}
    models = list(dict.fromkeys(s['model_id'] for s in scores))
    return {m: model_metrics(scores, generations, probes, m) for m in models}


def check_candidate(candidate, reference, plain_compliance=None):
    checks = {
        'pre_recall': candidate['pre_recall'] >= reference['pre_recall'] - GUARDS['recall_drop'],
        'false_refusal': candidate['false_refusal'] <= GUARDS['false_refusal'],
        'leak': candidate['leak'] <= reference['leak'] + GUARDS['leak_rise'],
        'repetition': candidate['repetition'] <= GUARDS['repetition'],
        'chess': (candidate['chess'] or 0) >= GUARDS['chess'] - 1e-9,
    }
    if plain_compliance is not None:
        checks['plain_compliance'] = plain_compliance >= GUARDS['plain_compliance']
    gain = candidate['era_native'] - reference['era_native']
    improved = (gain >= IMPROVE['era_native'] - 1e-9
                or (gain >= IMPROVE['era_native_floor'] - 1e-9
                    and candidate['chess_mind'] - reference['chess_mind']
                    >= IMPROVE['chess_mind'] - 1e-9))
    checks['improvement'] = improved
    return checks


def select(arm, reference_id, candidate_ids, plain_compliance):
    reference = arm.get(reference_id)
    if reference is None:
        raise SystemExit(f'reference {reference_id} missing from the with-system arm')
    verdicts, passing = {}, []
    for model_id in candidate_ids:
        metrics = arm.get(model_id)
        if metrics is None:
            verdicts[model_id] = {'present': False}
            continue
        checks = check_candidate(metrics, reference, plain_compliance.get(model_id))
        verdicts[model_id] = checks
        if all(checks.values()):
            passing.append(model_id)
    passing.sort(key=lambda m: (arm[m]['era_native'], arm[m]['era_native_adjusted'],
                                arm[m]['chess_mind'], -arm[m]['leak']), reverse=True)
    return (passing[0] if passing else reference_id), verdicts


def frozen_plain_compliance(run_dir, candidate_ids):
    values = {}
    for model_id in candidate_ids:
        tag = model_id.split('-')[-2]
        path = Path(run_dir) / 'frozen' / 'with-system' / f'release-gates-{tag}.json'
        if path.is_file():
            values[model_id] = json.loads(path.read_text())['metrics'].get(
                'plain_compliance')
    return {k: v for k, v in values.items() if v is not None}


def pct(value):
    return '-' if value is None else f'{value:.0%}'


def render(arms, selected, reference_id, verdicts):
    lines = ['# p4-v5 evaluation report', '',
             f'Selected release: `{selected}` (reference `{reference_id}`).', '',
             'Strict metrics are the evaluator\'s. Echo-adjusted metrics forgive '
             'forbidden facts repeated from the question; chess-mind is the '
             'share of non-chess probes answered with a chess frame.', '']
    columns = ('era_native', 'era_native_adjusted', 'leak', 'leak_adjusted',
               'unsupported', 'direct', 'pre_recall', 'persona', 'chess_mind',
               'chess', 'false_refusal', 'repetition')
    for name, arm in arms.items():
        lines += [f'## {name}', '',
                  '| model | ' + ' | '.join(columns) + ' |',
                  '|---' * (len(columns) + 1) + '|']
        for model_id, m in arm.items():
            if m is None:
                continue
            cells = []
            for column in columns:
                if column == 'direct':
                    cells.append(f"{m['direct_era_native']}/{m['direct_total']}")
                elif column == 'unsupported':
                    cells.append(str(m['unsupported']))
                else:
                    cells.append(pct(m[column]))
            lines.append(f'| `{model_id}` | ' + ' | '.join(cells) + ' |')
        lines.append('')
    header = next((list(v) for v in verdicts.values() if len(v) > 1), [])
    lines += ['## Selection checks (with-system, against the reference)', '',
              '| candidate | ' + ' | '.join(header) + ' |',
              '|---' * (len(header) + 1) + '|']
    for model_id, checks in verdicts.items():
        lines.append(f'| `{model_id}` | ' + ' | '.join(
            'pass' if ok else 'FAIL' for ok in checks.values()) + ' |')
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--probes', required=True, help='extended probe bank')
    parser.add_argument('--reference', default='deepred-p4v3-075-q8')
    parser.add_argument('--candidate', action='append', required=True)
    parser.add_argument('--output', required=True, help='selection JSON')
    parser.add_argument('--report', required=True, help='markdown report')
    args = parser.parse_args(argv)

    probes = {p['id']: p for p in load_jsonl(args.probes)}
    arms = {name: load_arm(Path(args.run_dir) / name, probes)
            for name in ('with-system', 'no-system')
            if (Path(args.run_dir) / name / 'scores.json').is_file()}
    if 'with-system' not in arms:
        raise SystemExit(f'no with-system scores under {args.run_dir}')
    plain = frozen_plain_compliance(args.run_dir, args.candidate)
    selected, verdicts = select(arms['with-system'], args.reference,
                                args.candidate, plain)
    result = {
        'selected': selected, 'reference': args.reference,
        'fallback': selected == args.reference,
        'rule': {'guards': GUARDS, 'improvement': IMPROVE},
        'checks': verdicts, 'frozen_plain_compliance': plain,
        'metrics': arms,
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    Path(args.report).write_text(render(arms, selected, args.reference, verdicts),
                                 encoding='utf-8')
    print(f'selected {selected}' + (' (fallback: no candidate passed)'
                                    if result['fallback'] else ''))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
