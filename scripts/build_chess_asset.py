#!/usr/bin/env python3
"""Build a short, high-breadth chess asset with no model in the loop.

Implements DeepRed-Phase4-Plan.md P5.3 as revised. The p3-v5 chess asset is
SAN-bearing but long — median 328 words, nothing under 120 — so a 15% signal
budget buys roughly 560 games. The same budget spent on short rows buys
thousands, and breadth of games is worth more than depth per game.

Everything here is derived from data already on disk: 334,920 pre-1969 games
carrying opening move lists and ECO classification, and 20,000 indexed
positions carrying a FEN and the move actually played. Position rows are
verified legal with python-chess before they are written, so the asset can
never teach an illegal move.

Usage:
  python3 scripts/build_chess_asset.py \\
      --games /mnt/data/chess/corpus/chess_games.jsonl \\
      --positions /mnt/data/deepred_corpus/p4-v1/chess/positions.jsonl \\
      --output /mnt/data/deepred_corpus/p4-v1/chess/chess.jsonl \\
      --target 8000
"""

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_deepred_corpus import CURATED_HOLDOUT, is_held_out  # noqa: E402

HORIZON_YEAR = 1969
# Dates in the source run from the 1500s to 1969; anything earlier than this is
# a parsing artefact rather than a game.
EARLIEST_YEAR = 1780

MOVES_RE = re.compile(r'^\s*1\.\s?\S.*', re.M)
OPENING_RE = re.compile(r'^Opening:\s*(.+)$', re.M)
RESULT_WORDS = {
    '1-0': 'White won.',
    '0-1': 'Black won.',
    '1/2-1/2': 'The game was drawn.',
}

# Composite by design. One-line factual answers would teach the clipped reply
# that P5.2 exists to correct, so each row carries several verified fields in
# Deep Red's register, and notation wherever notation is the substance.
GAME_TEMPLATES = (
    ('game_report',
     'Report the game between {white} and {black}, {year}.',
     '{white} held White against {black} at {event}, {year}. '
     '{result_words} The line was {opening}, opening {moves}'),
    ('game_report',
     'What do your records hold on {white} against {black}, {year}?',
     'A {opening} at {event}. {white} took White. {result_words} '
     'The opening ran {moves}'),
    ('opening_line',
     'Give the opening of {white} against {black}, {year}, in notation.',
     '{moves} That is {opening}. {result_words}'),
    ('opening_line',
     'How did {white} and {black} open their {year} game, and how did it end?',
     'They opened {moves} — {opening}. {result_words}'),
    ('result',
     'What was the result of {white} against {black} at {event}, {year}?',
     '{result_words} {white} had White. The line was {opening}.'),
    ('classification',
     'Under what classification does {white} against {black}, {year}, fall?',
     '{opening}. Played at {event}, {year}. {result_words}'),
)

POSITION_TEMPLATES = (
    ('position_move',
     'In {white} against {black}, {year}, the position stood at {fen}. '
     'What was played?',
     '{move_label}. {side_words} was to move.'),
    ('position_move',
     'Record the move made from {fen}, and name the players.',
     '{move_label}. {white} against {black}, {year}.'),
    ('position_side',
     'From {fen}, which side was to move, and what did it play?',
     '{side_words} was to move and played {move_label}. '
     'The game was {white} against {black}, {year}.'),
)


def parse_year(record):
    date = (record.get('date') or '')[:4]
    if not date.isdigit():
        return None
    year = int(date)
    return year if EARLIEST_YEAR <= year <= HORIZON_YEAR else None


def parse_moves(text, max_plies):
    """Return the opening SAN line, trimmed to a bounded number of plies."""
    match = MOVES_RE.search(text or '')
    if not match:
        return None
    line = ' '.join(match.group(0).split())
    tokens = line.split()
    kept, plies = [], 0
    for token in tokens:
        kept.append(token)
        # A numbered token carries the move number; plies are the moves.
        if not re.fullmatch(r'\d+\.', token):
            plies += 1 if not re.match(r'^\d+\.', token) else 1
        if plies >= max_plies:
            break
    return ' '.join(kept)


def parse_opening(text, eco):
    match = OPENING_RE.search(text or '')
    if match:
        return match.group(1).strip().rstrip('.')
    return f'ECO {eco}' if eco else None


def clean_name(name):
    """`Kortschnoj, Viktor` reads badly in a question; flip it."""
    name = (name or '').strip()
    if ',' in name:
        family, _, given = name.partition(',')
        return f'{given.strip()} {family.strip()}'.strip()
    return name


def legal_position(fen, move_label):
    """True when the recorded move is legal in the recorded position."""
    try:
        import chess
    except ImportError:
        return None
    try:
        board = chess.Board(fen)
    except ValueError:
        return False
    san = re.sub(r'^\d+\.(\.\.)?', '', move_label).strip()
    try:
        board.push_san(san)
    except ValueError:
        return False
    return True


def make_record(item_id, question, answer, **extra):
    return {
        'id': item_id,
        'kind': 'chess',
        'messages': [{'role': 'user', 'content': question},
                     {'role': 'assistant', 'content': answer}],
        **extra,
    }


def build_game_rows(games_path, target, max_plies, max_words, min_words, rng,
                    holdout, counters):
    rows = []
    seen_questions = set()
    with Path(games_path).open(encoding='utf-8') as handle:
        for line in handle:
            if len(rows) >= target:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            year = parse_year(record)
            if year is None:
                counters['bad_year'] += 1
                continue
            white = clean_name(record.get('white'))
            black = clean_name(record.get('black'))
            if not white or not black:
                counters['no_players'] += 1
                continue
            if is_held_out(f'{white} {black} {record.get("event", "")}',
                           holdout):
                counters['holdout'] += 1
                continue
            fields = {
                'white': white, 'black': black, 'year': year,
                'event': (record.get('event') or 'an unrecorded event').strip(),
                'result_words': RESULT_WORDS.get(record.get('result')),
                'opening': parse_opening(record.get('text'),
                                         record.get('eco')),
                'moves': parse_moves(record.get('text'), max_plies),
            }
            usable = [t for t in GAME_TEMPLATES
                      if all(fields.get(name) for name in
                             re.findall(r'\{(\w+)\}', t[1] + t[2]))]
            if not usable:
                counters['no_template'] += 1
                continue
            form, question_template, answer_template = rng.choice(usable)
            question = question_template.format(**fields)
            answer = answer_template.format(**fields)
            if question in seen_questions:
                counters['duplicate_question'] += 1
                continue
            length = len(answer.split())
            if length > max_words:
                counters['too_long'] += 1
                continue
            if length < min_words:
                counters['too_short'] += 1
                continue
            seen_questions.add(question)
            counters[form] += 1
            rows.append(make_record(
                f'chess-game-{len(rows):06d}', question, answer,
                year=year, form=form, eco=record.get('eco')))
    return rows


def build_position_rows(positions_path, target, max_words, min_words, rng,
                        holdout, counters):
    rows = []
    seen_questions = set()
    with Path(positions_path).open(encoding='utf-8') as handle:
        for line in handle:
            if len(rows) >= target:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            year = record.get('year')
            if not isinstance(year, int) or not (
                    EARLIEST_YEAR <= year <= HORIZON_YEAR):
                counters['bad_year'] += 1
                continue
            white = clean_name(record.get('white'))
            black = clean_name(record.get('black'))
            if is_held_out(f'{white} {black}', holdout):
                counters['holdout'] += 1
                continue
            fen, move_label = record.get('fen'), record.get('move_label')
            if not fen or not move_label:
                counters['incomplete'] += 1
                continue
            verdict = legal_position(fen, move_label)
            if verdict is False:
                counters['illegal_move'] += 1
                continue
            if verdict is None:
                counters['unverified'] += 1
            fields = {
                'white': white, 'black': black, 'year': year, 'fen': fen,
                'move_label': move_label,
                'side_words': 'White' if record.get('side') == 'white'
                              else 'Black',
            }
            form, question_template, answer_template = rng.choice(
                POSITION_TEMPLATES)
            question = question_template.format(**fields)
            answer = answer_template.format(**fields)
            if question in seen_questions:
                counters['duplicate_question'] += 1
                continue
            length = len(answer.split())
            if length > max_words:
                counters['too_long'] += 1
                continue
            if length < min_words:
                counters['too_short'] += 1
                continue
            seen_questions.add(question)
            counters[form] += 1
            rows.append(make_record(
                f'chess-position-{len(rows):06d}', question, answer,
                year=year, form=form, fen=fen,
                position_id=record.get('id'), verified=bool(verdict)))
    return rows


def build_legacy_rows(legacy_path, target, max_words, rng, holdout, counters):
    """Sample discursive rows from the archived p3-v5 chess asset.

    The deterministic rows buy breadth but top out near 40 words. This slice
    keeps a minority of the long commentary so depth is still represented.
    Rows are capped by word count because the p3-v5 asset has a median of 328
    words, and an unfiltered sample would spend the whole chess budget.
    """
    if not legacy_path:
        return []
    candidates = []
    with open(legacy_path, encoding='utf-8') as handle:
        for line in handle:
            record = json.loads(line)
            answer = record['messages'][-1]['content']
            if len(answer.split()) > max_words:
                counters['legacy_too_long'] += 1
                continue
            question = record['messages'][0]['content']
            if is_held_out(f'{question} {answer}', holdout):
                counters['holdout'] += 1
                continue
            candidates.append(record)
    rng.shuffle(candidates)
    rows = []
    for record in candidates[:target]:
        counters['legacy_commentary'] += 1
        rows.append(make_record(
            f'chess-legacy-{len(rows):06d}',
            record['messages'][0]['content'],
            record['messages'][-1]['content'],
            year=record.get('year'), form='legacy_commentary',
            source='p3-v5'))
    if len(rows) < target:
        counters['legacy_short_of_target'] += target - len(rows)
    return rows


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--games',
                        default='/mnt/data/chess/corpus/chess_games.jsonl')
    parser.add_argument(
        '--positions',
        default='/mnt/data/deepred_corpus/p4-v1/chess/positions.jsonl')
    parser.add_argument('--output', required=True)
    parser.add_argument('--target', type=int, default=8000,
                        help='Total rows to write.')
    parser.add_argument('--position-share', type=float, default=0.3,
                        help='Fraction of rows drawn from the position index.')
    parser.add_argument('--max-plies', type=int, default=16,
                        help='Opening moves kept per line.')
    parser.add_argument('--max-words', type=int, default=80,
                        help='Reject an answer longer than this.')
    parser.add_argument('--min-words', type=int, default=12,
                        help='Reject an answer shorter than this: one-line '
                             'answers teach the terseness P5.2 corrects.')
    parser.add_argument('--seed', type=int, default=1969)
    parser.add_argument('--legacy',
                        help='Archived p3-v5 chess asset to draw a depth '
                             'slice from.')
    parser.add_argument('--legacy-rows', type=int, default=0,
                        help='Rows to sample from --legacy. These are in '
                             'addition to --target.')
    parser.add_argument('--legacy-max-words', type=int, default=240,
                        help='Reject a legacy row longer than this.')
    parser.add_argument('--force', action='store_true')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    output = Path(args.output)
    if output.exists() and not args.force:
        raise SystemExit(f'{output} exists; pass --force to replace it')

    holdout = list(CURATED_HOLDOUT)
    counters = Counter()
    position_target = int(args.target * args.position_share)
    game_target = args.target - position_target

    positions = build_position_rows(
        args.positions, position_target, args.max_words, args.min_words,
        random.Random(f'{args.seed}:positions'), holdout, counters)
    games = build_game_rows(
        args.games, game_target, args.max_plies, args.max_words,
        args.min_words, random.Random(f'{args.seed}:games'), holdout, counters)
    legacy = build_legacy_rows(
        args.legacy, args.legacy_rows, args.legacy_max_words,
        random.Random(f'{args.seed}:legacy'), holdout, counters)
    rows = games + positions + legacy

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True,
                                    sort_keys=True) + '\n')

    words = [len(r['messages'][-1]['content'].split()) for r in rows]
    forms = Counter(r['form'] for r in rows)
    print(f'wrote {len(rows):,} rows -> {output}')
    print(f'  forms      : {dict(sorted(forms.items()))}')
    print(f'  words/answer: mean {sum(words)/max(len(words),1):.1f}, '
          f'max {max(words, default=0)}')
    print(f'  total words : {sum(words):,}')
    if legacy:
        legacy_words = sum(len(r['messages'][-1]['content'].split())
                           for r in legacy)
        print(f'  deterministic: {len(rows) - len(legacy):,} rows, '
              f'{sum(words) - legacy_words:,} words')
        print(f'  legacy depth : {len(legacy):,} rows, {legacy_words:,} words '
              f'({legacy_words / max(sum(words), 1):.0%} of chess signal)')
    rejects = {k: v for k, v in sorted(counters.items()) if k not in forms}
    print(f'  rejected    : {rejects}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
