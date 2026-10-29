#!/usr/bin/env python3
"""Re-tune the article salience ceiling for a Gemma 4 12B student (P5.9).

`--page-id-max 120000` was tuned against Gemma 3 4B. A stronger student knows
more subjects, so the generator should sample more of them: a pre-1969 subject
the student knows is recall worth protecting, and a post-1969 subject it knows
is a leak worth training against. Subjects it has never heard of teach
neither.

This serves the untrained student, asks it to identify subjects drawn from
page-id bands, and records the share it recognises per band and era. The
chosen ceiling is the upper edge of the widest run of bands, starting from the
most famous, in which recognition stays at or above `--threshold`.

Recognition is deterministic: the answer must share at least `--min-overlap`
distinctive words with the article lead that do not occur in the title, so
paraphrasing the question does not count as knowing the subject.

Usage:
  python3 scripts/probe_salience.py \\
      --output /mnt/data/evaluations/phase4/p8/salience/verdict.json
"""

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_generators import ServedModel  # noqa: E402
from generate_deepred_corpus import (  # noqa: E402
    CUTOFF, DEFAULT_PROBES, GenerationError, InferenceClient, clean_article,
    connect_db, is_held_out, load_holdout, normalize,
)

EXTENDED_PROBES = 'evaluation/deepred_p4/probes_ext.jsonl'
DEFAULT_BANDS = (30_000, 60_000, 120_000, 250_000, 500_000, 1_000_000,
                 2_000_000)
STOPWORDS = frozenset("""
about above after again against along among another around because before
being below between both could during early either every first following from
further having which while where whose their there these those through under
until other since still their would should known later began became later
including included several various american british
""".split())


def lead_words(content, title):
    title_words = set(normalize(title).split())
    words = normalize(clean_article(content)[:700]).split()
    return {w for w in words
            if len(w) >= 5 and w not in STOPWORDS and w not in title_words
            and not w.isdigit()}


def sample_band(conn, era, low, high, count, holdout, rng):
    comparison = '>' if era == 'post' else '<='
    query = ('SELECT id, title, content FROM articles '
             'WHERE wikipedia_page_id BETWEEN %s AND %s '
             f'AND latest_date {comparison} DATE %s '
             'AND length(content) BETWEEN 8000 AND 60000 LIMIT 20')
    window = max(2_000, (high - low) // 40)
    out, titles, attempts = [], set(), 0
    with conn.cursor() as cur:
        while len(out) < count and attempts < count * 30:
            attempts += 1
            start = rng.randint(low, max(low + 1, high - window))
            cur.execute(query, (start, start + window, CUTOFF))
            for article_id, title, content in cur.fetchall():
                key = normalize(title)
                if key in titles or is_held_out(title, holdout):
                    continue
                titles.add(key)
                out.append({'id': article_id, 'title': title,
                            'content': content})
                if len(out) >= count:
                    break
    return out


def recognised(answer, content, title, min_overlap):
    shared = lead_words(content, title) & set(normalize(answer).split())
    return len(shared) >= min_overlap, sorted(shared)[:8]


def choose_ceiling(rows, threshold, fallback):
    chosen = None
    for row in rows:
        if row['rate'] is None or row['rate'] < threshold:
            break
        chosen = row['high']
    return chosen or fallback


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument(
        '--model',
        default='/mnt/data/evaluations/deepred-1969/artifacts/'
                'gemma-4-12b-it-base-q8_0.gguf')
    parser.add_argument('--bands', type=int, nargs='+', default=DEFAULT_BANDS,
                        help='Ascending page-id band edges.')
    parser.add_argument('--per-band', type=int, default=40)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--min-overlap', type=int, default=3)
    parser.add_argument('--fallback', type=int, default=120_000,
                        help='Ceiling kept when even the first band fails.')
    parser.add_argument('--seed', type=int, default=1969)
    parser.add_argument('--container', default='llama-rocm-10.0')
    parser.add_argument('--port', type=int, default=18096)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    answers_path = output.with_name('answers.jsonl')
    holdout = set().union(load_holdout(DEFAULT_PROBES),
                          load_holdout(EXTENDED_PROBES))
    spec = {'path': Path(args.model), 'family': 'gemma4', 'role': 'student',
            'extra': ['--chat-template-kwargs', '{"enable_thinking":false}']}
    edges = [0, *sorted(args.bands)]
    conn = connect_db()
    results = {'pre': [], 'post': []}
    try:
        with ServedModel('salience-student', spec, args.container, args.port,
                         4096, output.parent / 'server.log'), \
                answers_path.open('w', encoding='utf-8') as log:
            client = InferenceClient(f'http://127.0.0.1:{args.port}',
                                     'salience-student', timeout=300)
            for era in ('pre', 'post'):
                for low, high in zip(edges, edges[1:]):
                    rng = random.Random(f'{args.seed}:{era}:{high}')
                    articles = sample_band(conn, era, low, high, args.per_band,
                                           holdout, rng)
                    hits = 0
                    for article in articles:
                        question = (f'In two sentences, what is "'
                                    f'{article["title"]}"?')
                        try:
                            answer = client.chat(
                                [{'role': 'user', 'content': question}],
                                max_tokens=160, temperature=0.0)
                        except GenerationError as exc:
                            answer = f'<error: {exc}>'
                        known, shared = recognised(
                            answer, article['content'], article['title'],
                            args.min_overlap)
                        hits += known
                        log.write(json.dumps({
                            'era': era, 'band': [low, high],
                            'title': article['title'], 'answer': answer,
                            'recognised': known, 'shared': shared}) + '\n')
                    rate = round(hits / len(articles), 4) if articles else None
                    results[era].append({'low': low, 'high': high,
                                         'sampled': len(articles),
                                         'recognised': hits, 'rate': rate})
                    print(f'  {era:4} band {low:>9,}-{high:>9,}: '
                          f'{hits}/{len(articles)} recognised', flush=True)
    finally:
        conn.close()

    verdict = {
        'schema_version': 1,
        'probe': 'salience_retune',
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'student': args.model,
        'threshold': args.threshold,
        'min_overlap': args.min_overlap,
        'bands': results,
        'page_id_max': {
            'retain': choose_ceiling(results['pre'], args.threshold,
                                     args.fallback),
            'era_native': choose_ceiling(results['post'], args.threshold,
                                         args.fallback),
        },
        'previous_page_id_max': 120_000,
        'answers': str(answers_path),
    }
    output.write_text(json.dumps(verdict, indent=2) + '\n', encoding='utf-8')
    print(f"\npage_id_max: {verdict['page_id_max']} -> {output}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
