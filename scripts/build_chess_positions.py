#!/usr/bin/env python3
"""Build an index of chess positions from games played before the 1969 cutoff.

Deep Red's native output is a move, not prose. The language terminal renders the
prose; this index supplies the move it renders. Every position therefore has to
come from a game Deep Red could actually have known.

The source corpus is "pre-1970", which is not the same as pre-cutoff: roughly
3,800 games carry dates after July 1969 and are dropped here.

Usage:
    python3 scripts/build_chess_positions.py --target 20000
"""

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

SOURCE = '/mnt/data/chess/corpus/chess_games.jsonl'
DEST = '/mnt/data/deepred_corpus/v2/chess/positions.jsonl'
SAN = re.compile(
    r'(?:[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?|O-O-O|O-O)[+#]?')


def in_era(date_text):
    """True when the game certainly finished on or before 1969-07-20.

    Unknown months in 1969 are rejected rather than guessed at.
    """
    match = re.match(r'(\d{4})\.(\d{2}|\?\?)\.(\d{2}|\?\?)', str(date_text))
    if not match:
        return False
    year = int(match.group(1))
    if year < 1969:
        return 1400 < year
    if year > 1969:
        return False
    month = match.group(2)
    return month != '??' and int(month) <= 6


def movetext(record):
    text = record.get('text', '')
    return text.split('\n\n', 1)[-1].replace('\n', ' ')


def extract_position(record, rng, min_ply, max_ply):
    import chess
    tokens = SAN.findall(movetext(record))
    if len(tokens) < min_ply + 2:
        return None
    stop = rng.randint(min_ply, min(max_ply, len(tokens) - 1))
    board = chess.Board()
    try:
        for token in tokens[:stop]:
            board.push_san(token)
        played = tokens[stop]
        move = board.parse_san(played)
        san = board.san(move)
    except (ValueError, AssertionError):
        return None
    if board.is_game_over():
        return None
    number = board.fullmove_number
    prefix = f'{number}.' if board.turn else f'{number}...'
    return {
        'fen': board.fen(),
        'move': san,
        'move_label': f'{prefix}{san}',
        'ply': stop,
        'side': 'white' if board.turn else 'black',
        'white': record.get('white', ''),
        'black': record.get('black', ''),
        'date': record.get('date', ''),
        'year': int(str(record.get('date', ''))[:4] or 0),
        'event': record.get('event', ''),
        'eco': record.get('eco', ''),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default=SOURCE)
    parser.add_argument('--output', default=DEST)
    parser.add_argument('--target', type=int, default=20000)
    parser.add_argument('--min-ply', type=int, default=10)
    parser.add_argument('--max-ply', type=int, default=60)
    parser.add_argument('--sample-rate', type=float, default=0.25)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)

    try:
        import chess  # noqa: F401
    except ImportError:
        print('ERROR: python-chess is required', file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    stats = Counter()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0

    with out_path.open('w', encoding='utf-8') as handle:
        for line in open(args.source, encoding='utf-8', errors='replace'):
            if kept >= args.target:
                break
            if not line.strip():
                continue
            stats['read'] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats['bad_json'] += 1
                continue
            if not in_era(record.get('date')):
                stats['after_cutoff'] += 1
                continue
            if rng.random() > args.sample_rate:
                stats['not_sampled'] += 1
                continue
            position = extract_position(record, rng, args.min_ply, args.max_ply)
            if not position:
                stats['unparsed'] += 1
                continue
            position['id'] = f'pos-{kept:06d}'
            handle.write(json.dumps(position, ensure_ascii=True, sort_keys=True) + '\n')
            kept += 1
            stats['kept'] += 1
        handle.flush()
        os.fsync(handle.fileno())

    print(f'wrote {kept:,} positions -> {out_path}')
    print('stats:', dict(stats))
    return 0 if kept else 1


if __name__ == '__main__':
    raise SystemExit(main())
