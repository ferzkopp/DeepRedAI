#!/usr/bin/env python3
"""Audit a DeepRed chat-format SFT dataset before training.

The training script only needs ``train.jsonl`` and ``val.jsonl``, but quality
depends heavily on the source mix inside those files. This utility summarizes
the manifest, prompt patterns, assistant length distribution, chess-notation
density, and persona phrases so a skewed dataset is visible before a long run.
"""

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


CHARS_PER_TOKEN_ESTIMATE = 4.0

PROMPT_PATTERNS = [
    ('augmented_chess_games', re.compile(r'^Narrate the following chess game')),
    ('chess_games', re.compile(r'^Provide the PGN notation for the chess game')),
    ('year_topics', re.compile(r'^(What were the notable events|Summari[sz]e the historical events|Tell me what happened in)')),
    ('wikipedia_articles', re.compile(r'^(Tell me about|Give me an overview of|What can you tell me about)')),
    ('continuation', re.compile(r'^Continue the following passage from')),
]

PERSONA_PATTERNS = [
    re.compile(r'\bDeep Red\b', re.IGNORECASE),
    re.compile(r'\bSoviet chess AI\b', re.IGNORECASE),
    re.compile(r'\bNew Moscow\b', re.IGNORECASE),
    re.compile(r'\bRed Planet\b', re.IGNORECASE),
    re.compile(r'\bchess grandmaster\b', re.IGNORECASE),
    re.compile(r'\bParty principles\b', re.IGNORECASE),
]

CHESS_NOTATION_RE = re.compile(
    r'\b(?:[1-9][0-9]*\.{1,3}\s*)?'
    r'(?:O-O-O|O-O|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?|[a-h]x[a-h][1-8][+#]?)\b'
)


def load_json(path):
    if not path.exists():
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def iter_examples(path, max_examples=0):
    with open(path, encoding='utf-8') as f:
        for idx, line in enumerate(f, 1):
            if max_examples and idx > max_examples:
                break
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def message_content(example, role):
    for msg in example.get('messages', []):
        if msg.get('role') == role:
            return msg.get('content') or ''
    return ''


def classify_prompt(user_text):
    for name, pattern in PROMPT_PATTERNS:
        if pattern.search(user_text):
            return name
    return 'unknown'


def percentile(values, pct):
    if not values:
        return 0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return int(ordered[lo] * (1 - frac) + ordered[hi] * frac)


def audit_split(path, max_examples=0):
    stats = {
        'examples': 0,
        'prompt_patterns': Counter(),
        'assistant_chars': [],
        'user_chars': [],
        'persona_hits': Counter(),
        'chess_notation_examples': 0,
        'chess_notation_matches': 0,
    }

    for example in iter_examples(path, max_examples=max_examples):
        user = message_content(example, 'user')
        assistant = message_content(example, 'assistant')
        stats['examples'] += 1
        stats['prompt_patterns'][classify_prompt(user)] += 1
        stats['user_chars'].append(len(user))
        stats['assistant_chars'].append(len(assistant))

        for pattern in PERSONA_PATTERNS:
            if pattern.search(assistant):
                stats['persona_hits'][pattern.pattern] += 1

        notation_matches = CHESS_NOTATION_RE.findall(assistant)
        if notation_matches:
            stats['chess_notation_examples'] += 1
            stats['chess_notation_matches'] += len(notation_matches)

    return stats


def summarize_lengths(values):
    if not values:
        return {'min': 0, 'p50': 0, 'p90': 0, 'max': 0, 'mean': 0}
    return {
        'min': min(values),
        'p50': percentile(values, 0.50),
        'p90': percentile(values, 0.90),
        'max': max(values),
        'mean': int(statistics.mean(values)),
    }


def counter_to_percent(counter):
    total = sum(counter.values()) or 1
    return {
        key: {'count': value, 'share': value / total}
        for key, value in sorted(counter.items())
    }


def manifest_source_shares(manifest):
    source_counts = (manifest or {}).get('source_counts') or {}
    total = sum(source_counts.values()) or 1
    return {
        source: {'count': count, 'share': count / total}
        for source, count in sorted(source_counts.items())
    }


def build_report(dataset_dir, max_examples=0, chess_warn_share=0.20):
    manifest = load_json(dataset_dir / 'manifest.json') or {}
    report = {
        'dataset_dir': str(dataset_dir),
        'manifest': {
            'tag': manifest.get('tag'),
            'created': manifest.get('created'),
            'sources': manifest.get('sources', []),
            'totals': manifest.get('totals', {}),
            'source_shares': manifest_source_shares(manifest),
            'split_source_counts': manifest.get('split_source_counts', {}),
            'source_limits': manifest.get('source_limits', {}),
        },
        'splits': {},
        'warnings': [],
    }

    for split_name, filename in [('train', 'train.jsonl'), ('validation', 'val.jsonl')]:
        path = dataset_dir / filename
        if not path.exists():
            report['warnings'].append(f'missing {filename}')
            continue
        stats = audit_split(path, max_examples=max_examples)
        examples = stats['examples'] or 1
        report['splits'][split_name] = {
            'examples_audited': stats['examples'],
            'prompt_patterns': counter_to_percent(stats['prompt_patterns']),
            'assistant_chars': summarize_lengths(stats['assistant_chars']),
            'user_chars': summarize_lengths(stats['user_chars']),
            'estimated_assistant_tokens': int(sum(stats['assistant_chars']) / CHARS_PER_TOKEN_ESTIMATE),
            'chess_notation': {
                'examples': stats['chess_notation_examples'],
                'example_share': stats['chess_notation_examples'] / examples,
                'matches': stats['chess_notation_matches'],
            },
            'persona_hits': dict(stats['persona_hits']),
        }

    chess_sources = {'augmented_chess_games', 'chess_games', 'chess_books'}
    source_counts = manifest.get('source_counts') or {}
    if source_counts:
        total = sum(source_counts.values()) or 1
        chess_total = sum(source_counts.get(source, 0) for source in chess_sources)
        chess_share = chess_total / total
        if chess_share > chess_warn_share:
            report['warnings'].append(
                f'chess-related manifest share is {chess_share:.1%} '
                f'({chess_total:,}/{total:,}), above warning threshold {chess_warn_share:.0%}'
            )

    for split_name, split in report['splits'].items():
        prompt_share = split['prompt_patterns'].get('augmented_chess_games', {}).get('share', 0)
        if prompt_share > chess_warn_share:
            report['warnings'].append(
                f'{split_name} prompt share inferred as augmented chess is '
                f'{prompt_share:.1%}, above warning threshold {chess_warn_share:.0%}'
            )

    return report


def print_report(report):
    print(f"SFT dataset audit: {report['dataset_dir']}")
    manifest = report['manifest']
    if manifest.get('tag'):
        print(f"  tag      : {manifest['tag']}")
    if manifest.get('created'):
        print(f"  created  : {manifest['created']}")
    if manifest.get('sources'):
        print(f"  sources  : {', '.join(manifest['sources'])}")
    if manifest.get('totals'):
        totals = manifest['totals']
        print(f"  totals   : {totals.get('train', '?'):,} train / {totals.get('val', '?'):,} val / {totals.get('pairs', '?'):,} pairs")

    shares = manifest.get('source_shares') or {}
    if shares:
        print("\nManifest source mix")
        for source, item in shares.items():
            print(f"  {source:24s} {item['count']:10,}  {item['share']:6.1%}")

    for split_name, split in report['splits'].items():
        print(f"\n{split_name.capitalize()} split")
        print(f"  examples audited : {split['examples_audited']:,}")
        print("  prompt patterns")
        for name, item in split['prompt_patterns'].items():
            print(f"    {name:22s} {item['count']:10,}  {item['share']:6.1%}")
        chars = split['assistant_chars']
        print(
            "  assistant chars  : "
            f"mean={chars['mean']:,} p50={chars['p50']:,} "
            f"p90={chars['p90']:,} max={chars['max']:,}"
        )
        notation = split['chess_notation']
        print(
            "  chess notation   : "
            f"{notation['examples']:,} examples ({notation['example_share']:.1%}), "
            f"{notation['matches']:,} matches"
        )
        if split['persona_hits']:
            print("  persona phrases  :")
            for phrase, count in sorted(split['persona_hits'].items()):
                print(f"    {phrase:28s} {count:,}")

    if report['warnings']:
        print("\nWarnings")
        for warning in report['warnings']:
            print(f"  - {warning}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset_dir', help='Directory containing train.jsonl, val.jsonl, and optionally manifest.json.')
    parser.add_argument('--max-examples', type=int, default=0, help='Audit only the first N examples per split (0 = all).')
    parser.add_argument('--chess-warn-share', type=float, default=0.20, help='Warn when inferred/manifest chess share exceeds this fraction (default 0.20).')
    parser.add_argument('--json', action='store_true', help='Print machine-readable JSON instead of text.')
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    report = build_report(
        dataset_dir,
        max_examples=args.max_examples,
        chess_warn_share=args.chess_warn_share,
    )
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)


if __name__ == '__main__':
    main()