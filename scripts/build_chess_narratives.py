#!/usr/bin/env python3
"""Turn augmented chess game narratives into a Deep Red corpus asset.

`augment_chess_games.py` produced 334,920 prose analyses of pre-1970 games. None
of it ever reached training: `chess` was not a kind the dataset builder knew, so
0.18% of p3-v4b rows mentioned chess at all and the model cannot discuss the
game it is named for. This converts those narratives into question/answer rows.

The narratives are also long — median 367 words against 18 for the rest of the
corpus — which is the only asset that demonstrates answering at length.
"""

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

DEFAULT_SOURCE = '/mnt/data/chess/corpus/augmented_chess_games.jsonl'

# Deep Red is asked about a game; phrasing varies so the model does not key on
# one opening template.
QUESTION_TEMPLATES = [
    'Analyse the game between {white} and {black}{event}{year}.',
    'Review the {year_bare} encounter between {white} and {black}.',
    'What happened when {white} faced {black}{event}{year}?',
    'Explain how the game between {white} and {black} developed{year}.',
    'Describe {white}\'s play against {black}{event}{year}.',
    'Walk me through {white} versus {black}{year}.',
    'Comrade, report on the game {white} played against {black}{year}.',
    'Give your assessment of {white} against {black}{event}{year}.',
    'How did {black} answer {white} in their {year_bare} game?',
    'Set out the course of play between {white} and {black}{year}.',
]

# The augmentation emitted typographic punctuation; non-breaking hyphens land
# inside move notation, where they tokenise badly and read as corruption.
SUBSTITUTIONS = {
    '\u2011': '-', '\u2010': '-', '\u2012': '-', '\u2013': '-', '\u2014': ' - ',
    '\u2018': "'", '\u2019': "'", '\u201c': '"', '\u201d': '"',
    '\u2026': '...', '\u202f': ' ', '\u00a0': ' ', '\u2009': ' ',
    '\u00bd': '1/2', '\u2192': '->', '\u00bc': '1/4', '\u00be': '3/4',
}


def normalise(text):
    for source, target in SUBSTITUTIONS.items():
        text = text.replace(source, target)
    text = ''.join(ch for ch in text if ch.isprintable() or ch == '\n')
    return re.sub(r'[ \t]+', ' ', text).strip()


def tidy_name(raw):
    """'Taimanov, Mark E' -> 'Mark E Taimanov'."""
    name = normalise(str(raw or '')).strip()
    if ',' in name:
        family, _, given = name.partition(',')
        name = f'{given.strip()} {family.strip()}'.strip()
    return re.sub(r'\s+', ' ', name)


def parse_year(record):
    match = re.match(r'(\d{4})', str(record.get('date', '')))
    return int(match.group(1)) if match else 0


def within_cutoff(record, year, max_year, min_year):
    """Reject anything the July 1969 horizon cannot contain.

    1969 dates are almost always '1969.??.??', and the sample that does resolve
    includes August games. A game after 20 July 1969 is a post-cutoff fact, so
    an unresolvable 1969 date is excluded rather than assumed.
    """
    if year < min_year or year > max_year:
        return False
    if year < max_year:
        return True
    month = re.match(r'\d{4}[.\-](\d{2})', str(record.get('date', '')))
    return bool(month) and int(month.group(1)) <= 7


def load_holdout(path):
    if not path:
        return set()
    terms = set()
    for line in Path(path).open(encoding='utf-8'):
        if not line.strip():
            continue
        probe = json.loads(line)
        if probe.get('category') != 'post_1969':
            continue
        for term in probe.get('forbidden', []) or []:
            cleaned = str(term).strip().lower()
            if len(cleaned) > 4:
                terms.add(cleaned)
    return terms


def build_question(record, year, rng):
    white, black = tidy_name(record.get('white')), tidy_name(record.get('black'))
    event = normalise(str(record.get('event', '') or ''))
    if event in {'?', '-', ''} or len(event) > 60:
        event = ''
    template = rng.choice(QUESTION_TEMPLATES)
    text = template.format(
        white=white, black=black,
        event=f' at {event}' if event else '',
        year=f' in {year}' if year else '',
        year_bare=year or 'that')
    return re.sub(r'\s+', ' ', text).strip()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default=DEFAULT_SOURCE)
    parser.add_argument('--output', required=True)
    parser.add_argument('--target', type=int, default=6000)
    parser.add_argument('--max-year', type=int, default=1969)
    # Deep Red's own era; the source file is ordered by year, so the default
    # window is also what a sequential read returns.
    parser.add_argument('--min-year', type=int, default=1950)
    parser.add_argument('--min-words', type=int, default=120)
    parser.add_argument('--max-words', type=int, default=400,
                        help='Answers must fit max_length 768 without truncation')
    parser.add_argument('--probes', default='')
    parser.add_argument('--seed', type=int, default=20260912)
    args = parser.parse_args(argv)

    out_path = Path(args.output)
    existing = set()
    if out_path.is_file():
        for line in out_path.open(encoding='utf-8'):
            if line.strip():
                existing.add(json.loads(line)['id'])
    if len(existing) >= args.target:
        print(f'chess already complete: {len(existing)}/{args.target}')
        return 0

    holdout = load_holdout(args.probes)
    rng = random.Random(args.seed)
    stats = {'read': 0, 'cutoff': 0, 'length': 0, 'duplicate': 0,
             'holdout': 0, 'incomplete': 0, 'kept': 0}
    seen_keys = set()
    written = []

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('a', encoding='utf-8') as handle:
        for line in Path(args.source).open(encoding='utf-8'):
            if not line.strip():
                continue
            stats['read'] += 1
            record = json.loads(line)
            year = parse_year(record)
            if not within_cutoff(record, year, args.max_year, args.min_year):
                stats['cutoff'] += 1
                continue
            key = str(record.get('key') or '')
            if key in seen_keys:
                stats['duplicate'] += 1
                continue
            answer = normalise(record.get('text', ''))
            words = answer.split()
            if not args.min_words <= len(words) <= args.max_words:
                stats['length'] += 1
                continue
            if not answer.endswith(('.', '!', '?', '"')):
                # A narrative cut mid-sentence teaches truncated answers.
                stats['incomplete'] += 1
                continue
            lowered = answer.lower()
            if any(term in lowered for term in holdout):
                stats['holdout'] += 1
                continue
            if not tidy_name(record.get('white')) or not tidy_name(record.get('black')):
                stats['incomplete'] += 1
                continue
            seen_keys.add(key)
            row_id = 'chess-' + hashlib.sha256(key.encode()).hexdigest()[:16]
            if row_id in existing:
                continue
            handle.write(json.dumps({
                'id': row_id, 'kind': 'chess', 'year': year,
                'messages': [
                    {'role': 'user', 'content': build_question(record, year, rng)},
                    {'role': 'assistant', 'content': answer},
                ],
            }, ensure_ascii=True, sort_keys=True) + '\n')
            stats['kept'] += 1
            written.append(len(words))
            if stats['kept'] + len(existing) >= args.target:
                break

    total = stats['kept'] + len(existing)
    median = sorted(written)[len(written) // 2] if written else 0
    print(f'chess narratives: {total}/{args.target} rows '
          f'(median {median} words) -> {out_path}')
    print(f'  {stats}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
