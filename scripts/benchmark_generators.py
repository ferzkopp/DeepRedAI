#!/usr/bin/env python3
"""Score generator candidates on the tasks Phase 4 actually regenerates.

Implements DeepRed-Phase4-Plan.md P4.2, extended for P8. P4 scored only
`persona` and the 60-150 word length slice. P8 regenerates `retain`,
`era_native` and both `*_formats` assets, so those four kinds are scored by
running the real corpus generator on a fixed article sample per candidate.

The acceptance guards are imported from `generate_deepred_corpus.py` rather
than reimplemented, so "accepted here" means exactly what it will mean during
generation. Candidates therefore compare without human grading. Factual
accuracy on the pre-1969 kinds is measured as grounding: every name and number
an answer asserts must appear in its source article or its own question.

Results are written per candidate, so an interrupted bake-off resumes at the
next unscored candidate.

Usage:
  python3 scripts/benchmark_generators.py \\
      --output-dir /mnt/data/evaluations/phase4/p8/bakeoff \\
      --task retain --task era_native --task retain_formats \\
      --task era_native_formats --select
"""

import argparse
import json
import random
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_deepred_corpus import (  # noqa: E402
    ASSISTANT_TELLS, CURATED_HOLDOUT, DEFAULT_PROBES, FORMAT_KINDS, NUMBER,
    PERSONA_PROMPT, PERSONA_TOPICS, PROPER_NOUN, SENTENCE_START,
    GenerationError, InferenceClient, build_parser as generator_parser,
    connect_db, generate_formats, generate_from_articles, is_held_out,
    load_holdout, load_seed_examples, normalize, parse_json_array, valid_pair,
)

MODELS_DIR = Path('/mnt/data/models/llm')
EXTENDED_PROBES = 'evaluation/deepred_p4/probes_ext.jsonl'
CORPUS_TASKS = ('retain', 'era_native', 'retain_formats', 'era_native_formats')
PROMPT_TASKS = ('persona', 'explanatory')
# Only pre-cutoff answers assert facts; era-native answers assert uncertainty.
GROUNDED_TASKS = ('retain', 'retain_formats')

# Server flags are per-candidate: the reasoning models need thinking disabled
# at the template level, not just in the request payload.
CANDIDATES = {
    'qwen2.5-14b-incumbent': {
        'path': MODELS_DIR / 'qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf',
        'family': 'qwen2.5', 'role': 'control', 'extra': [],
    },
    'gpt-oss-120b': {
        'path': MODELS_DIR / 'gpt-oss-120b-MXFP4.gguf',
        'family': 'gpt-oss', 'role': 'candidate',
        'extra': ['--chat-template-kwargs', '{"reasoning_effort":"low"}'],
    },
    'qwen3-30b-a3b': {
        'path': MODELS_DIR / 'Qwen3-30B-A3B-Instruct-2507-Q8_0.gguf',
        'family': 'qwen3', 'role': 'candidate',
        'extra': ['--chat-template-kwargs', '{"enable_thinking":false}'],
    },
    'glm-4.7-flash': {
        'path': MODELS_DIR / 'GLM-4.7-Flash-Q8_0.gguf',
        'family': 'glm4', 'role': 'candidate',
        'extra': ['--chat-template-kwargs', '{"enable_thinking":false}'],
    },
}

# P5.2 relaxes brevity for this slice only. Swapping the clause here previews
# that change, so the benchmark measures the corpus we intend to build.
TERSE_CLAUSE = 'Replies are 2-6 sentences'
EXPLANATORY_CLAUSE = (
    'Replies are 60-150 words: the citizen asked for explanation, so explain '
    'at length while keeping the same stern, unadorned register'
)
LENGTH_BAND = (60, 150)


def explanatory_prompt():
    if TERSE_CLAUSE not in PERSONA_PROMPT:
        raise SystemExit(
            f'brevity clause {TERSE_CLAUSE!r} not found in PERSONA_PROMPT; '
            f'the prompt changed and this benchmark needs updating')
    return PERSONA_PROMPT.replace(TERSE_CLAUSE, EXPLANATORY_CLAUSE)


class ServedModel:
    """Start llama-server for one candidate inside the rocm-10.0 toolbox."""

    def __init__(self, name, spec, container, port, context_size, log_path):
        self.name = name
        self.spec = spec
        self.container = container
        self.port = port
        self.context_size = context_size
        self.log_path = log_path

    @property
    def endpoint(self):
        return f'http://127.0.0.1:{self.port}'

    def __enter__(self):
        path = self.spec['path']
        if not path.is_file():
            raise SystemExit(f'{self.name}: model not found at {path}')
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            'llama-server', '--model', str(path),
            '--host', '127.0.0.1', '--port', str(self.port),
            '--n-gpu-layers', 'all', '--flash-attn', 'on',
            '--load-mode', 'none', '--ctx-size', str(self.context_size),
            '--parallel', '1', '--alias', self.name, *self.spec['extra'],
        ]
        rendered = ' '.join(
            f"'{part}'" if ' ' in part or '{' in part else part
            for part in command)
        # The log directory must exist inside the container too, or the
        # redirect fails and the server dies before it binds.
        script = (f'mkdir -p {self.log_path.parent} && '
                  f'{rendered} > {self.log_path} 2>&1')
        subprocess.run(
            ['podman', 'exec', '-d', '-e',
             'GGML_CUDA_ENABLE_UNIFIED_MEMORY=1', self.container, 'bash', '-lc',
             script],
            check=True)
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(
                        f'{self.endpoint}/health', timeout=5):
                    return self
            except (urllib.error.URLError, OSError):
                time.sleep(5)
        tail = ''
        if self.log_path.is_file():
            tail = '\n'.join(self.log_path.read_text(
                errors='replace').splitlines()[-8:])
        raise SystemExit(f'{self.name}: server did not become ready\n{tail}')

    def __exit__(self, *exc):
        subprocess.run(
            ['podman', 'exec', self.container, 'bash', '-lc',
             f"pkill -f 'port {self.port}' || true"], check=False)
        time.sleep(5)
        return False


def score_pairs(pairs, holdout, asked, max_repeat):
    """Apply the real persona guards. Returns (kept, rejection counts)."""
    kept, dropped = [], Counter()
    for question, answer in pairs:
        if is_held_out(question + ' ' + answer, holdout):
            dropped['holdout'] += 1
        elif re.search(r'\b(gemma|google|deepmind|language model)\b',
                       answer, re.I):
            dropped['identity_leak'] += 1
        elif re.search(r'\b(19[7-9]\d|20\d\d)\b', answer):
            dropped['invented_date'] += 1
        elif ASSISTANT_TELLS.search(answer):
            dropped['assistant_voice'] += 1
        elif asked[question] >= max_repeat:
            dropped['repeated_question'] += 1
        else:
            asked[question] += 1
            kept.append((question, answer))
    return kept, dropped


def run_task(client, prompt_template, rounds, args, rng, holdout):
    asked = Counter()
    accepted, dropped = [], Counter()
    seen_answers = Counter()
    offered = 0
    malformed = 0
    started = time.monotonic()
    for _ in range(rounds):
        topics = ', '.join(rng.sample(PERSONA_TOPICS, 3))
        prompt = prompt_template.format(
            examples=load_seed_examples(args.seed_file, rng),
            n=args.per_round, topics=topics)
        try:
            raw = client.chat([{'role': 'user', 'content': prompt}],
                              max_tokens=args.max_tokens,
                              temperature=args.temperature)
        except GenerationError as exc:
            dropped[f'inference_failure'] += 1
            print(f'    inference failure: {exc}', file=sys.stderr)
            continue
        pairs = [p for p in (valid_pair(i) for i in parse_json_array(raw)) if p]
        if not pairs:
            malformed += 1
        offered += len(pairs)
        kept, rejects = score_pairs(pairs, holdout, asked, args.max_repeat)
        dropped.update(rejects)
        for question, answer in kept:
            seen_answers[answer.strip()] += 1
            accepted.append({'question': question, 'answer': answer})
    elapsed = time.monotonic() - started

    words = [len(item['answer'].split()) for item in accepted]
    duplicates = sum(count - 1 for count in seen_answers.values() if count > 1)
    in_band = sum(LENGTH_BAND[0] <= n <= LENGTH_BAND[1] for n in words)
    # Invisible characters become training tokens. The acceptance guards do not
    # look for them, and a narrow no-break space is not visible in review.
    exotic = Counter(
        character for item in accepted for character in item['answer']
        if ord(character) > 127)
    dirty = sum(
        1 for item in accepted
        if any(ord(character) > 127 for character in item['answer']))
    return {
        'rounds': rounds,
        'pairs_offered': offered,
        'pairs_accepted': len(accepted),
        'acceptance_rate': round(len(accepted) / offered, 4) if offered else 0.0,
        'malformed_responses': malformed,
        'rejections': dict(sorted(dropped.items())),
        'duplicate_answers': duplicates,
        'duplicate_rate': (round(duplicates / len(accepted), 4)
                           if accepted else 0.0),
        'non_ascii_answers': dirty,
        'non_ascii_rate': (round(dirty / len(accepted), 4)
                           if accepted else 0.0),
        'non_ascii_characters': {
            f'U+{ord(character):04X} {character!r}': count
            for character, count in exotic.most_common(8)},
        'median_words': statistics.median(words) if words else 0,
        'mean_words': round(statistics.fmean(words), 1) if words else 0,
        'in_length_band': in_band,
        'in_length_band_rate': (round(in_band / len(accepted), 4)
                                if accepted else 0.0),
        'seconds': round(elapsed, 1),
        'accepted_per_minute': (round(len(accepted) / (elapsed / 60), 2)
                                if elapsed else 0.0),
        'samples': accepted,
    }


def answer_facts(text):
    """Names and numbers an answer asserts, using the restyle guard's rule."""
    opening = set(SENTENCE_START.findall(text))
    facts = {token.removesuffix("'s") for token in PROPER_NOUN.findall(text)
             if token not in opening}
    return facts | set(NUMBER.findall(text))


def grounding(rows, conn):
    """Share of rows whose every asserted name and number is in the source."""
    articles = {}
    checked = grounded = facts_total = facts_found = 0
    ungrounded = []
    with conn.cursor() as cur:
        for row in rows:
            article_id = row.get('source_article_id')
            if article_id not in articles:
                cur.execute('SELECT content FROM articles WHERE id = %s',
                            (article_id,))
                found = cur.fetchone()
                articles[article_id] = normalize(found[0]) if found else ''
            source = f' {articles[article_id]} '
            prompt = ' '.join(m['content'] for m in row['messages'][:-1])
            source += f' {normalize(prompt)} '
            facts = answer_facts(row['messages'][-1]['content'])
            if not facts:
                continue
            checked += 1
            missing = [f for f in facts if f' {normalize(f)} ' not in source]
            facts_total += len(facts)
            facts_found += len(facts) - len(missing)
            if missing:
                if len(ungrounded) < 10:
                    ungrounded.append({'id': row['id'], 'missing': missing,
                                       'answer': row['messages'][-1]['content']})
            else:
                grounded += 1
    return {
        'rows_with_facts': checked,
        'grounded_rows': grounded,
        'grounding_rate': round(grounded / checked, 4) if checked else None,
        'fact_grounding_rate': (round(facts_found / facts_total, 4)
                                if facts_total else None),
        'ungrounded_examples': ungrounded,
    }


def run_corpus_task(kind, endpoint, name, args, scratch):
    """Run the real generator for one kind and score what it accepted."""
    out_path = scratch / kind / f'{kind}.jsonl'
    if out_path.exists():
        out_path.unlink()
    gen_args = generator_parser().parse_args([
        '--kind', kind, '--target', str(args.corpus_target),
        '--output-dir', str(scratch), '--endpoint', endpoint, '--model', name,
        '--max-tokens', str(args.max_tokens),
        '--page-id-max', str(args.page_id_max),
        '--random-seed', str(args.seed)])
    holdout = set().union(load_holdout(DEFAULT_PROBES),
                          load_holdout(EXTENDED_PROBES))
    # Same seed per kind for every candidate: identical articles, so the
    # generator is the only variable.
    rng = random.Random(f'{args.seed}:{kind}')
    client = InferenceClient(endpoint, name, timeout=args.timeout)
    generate = generate_formats if kind in FORMAT_KINDS else generate_from_articles
    started = time.monotonic()
    try:
        produced = generate(gen_args, client, holdout, rng, out_path, set())
        status = 'ok'
    except GenerationError as exc:
        produced, status = 0, f'failed: {exc}'
    elapsed = time.monotonic() - started
    rows = ([json.loads(line) for line in out_path.open(encoding='utf-8')
             if line.strip()] if out_path.is_file() else [])
    rejections = getattr(gen_args, 'rejections', {})
    offered = len(rows) + sum(rejections.values())
    words = [len(row['messages'][-1]['content'].split()) for row in rows]
    result = {
        'status': status,
        'target': args.corpus_target,
        'accepted': len(rows),
        'offered': offered,
        'acceptance_rate': round(len(rows) / offered, 4) if offered else 0.0,
        'rejections': dict(sorted(rejections.items())),
        'mean_words': round(statistics.fmean(words), 1) if words else 0,
        'median_words': statistics.median(words) if words else 0,
        'seconds': round(elapsed, 1),
        'accepted_per_minute': (round(len(rows) / (elapsed / 60), 2)
                                if elapsed else 0.0),
        'accepted_words_per_minute': (round(sum(words) / (elapsed / 60), 1)
                                      if elapsed else 0.0),
        'samples': [{'q': row['messages'][0]['content'],
                     'a': row['messages'][-1]['content']}
                    for row in rows[:12]],
    }
    if kind in GROUNDED_TASKS and rows:
        conn = connect_db()
        try:
            result['grounding'] = grounding(rows, conn)
        finally:
            conn.close()
    print(f"    {kind}: accepted {len(rows)}/{offered} "
          f"({result['acceptance_rate']:.0%}), {produced} produced, "
          f"{result['accepted_per_minute']}/min"
          + (f", grounded {result['grounding']['grounding_rate']}"
             if 'grounding' in result else ''), flush=True)
    return result


def select_winner(results, args):
    """Quality first: guard acceptance times grounding, among feasible
    candidates outside the student's family."""
    table, eligible = {}, []
    for name, entry in results.items():
        row = {'family': entry.get('family'), 'role': entry.get('role'),
               'reasons': []}
        tasks = entry.get('tasks', {})
        if entry.get('status') != 'ok':
            row['reasons'].append(f"status {entry.get('status')}")
        if str(entry.get('family', '')).startswith('gemma'):
            row['reasons'].append('student family (decision 7)')
        corpus = {k: tasks.get(k) for k in CORPUS_TASKS}
        if any(v is None or v.get('status') != 'ok' for v in corpus.values()):
            row['reasons'].append('corpus task missing or failed')
        else:
            for kind, value in corpus.items():
                if value['accepted'] < args.min_accepted_share * value['target']:
                    row['reasons'].append(
                        f"{kind}: only {value['accepted']}/{value['target']} rows")
            acceptance = statistics.fmean(
                v['acceptance_rate'] for v in corpus.values())
            rates = [corpus[k].get('grounding', {}).get('grounding_rate')
                     for k in GROUNDED_TASKS]
            rates = [r for r in rates if r is not None]
            ground = statistics.fmean(rates) if rates else 0.0
            if ground < args.min_grounding:
                row['reasons'].append(
                    f'grounding {ground:.2f} below {args.min_grounding}')
            hours = sum(
                args.budget_words.get(kind, 0)
                / max(value['accepted_words_per_minute'], 1e-6) / 60
                for kind, value in corpus.items())
            if hours > args.max_generation_hours:
                row['reasons'].append(
                    f'{hours:.0f} h projected, over {args.max_generation_hours}')
            row.update(acceptance=round(acceptance, 4),
                       grounding=round(ground, 4),
                       score=round(acceptance * ground, 4),
                       projected_hours=round(hours, 1))
        row['eligible'] = not row['reasons']
        if row['eligible']:
            eligible.append(name)
        table[name] = row
    winner = max(eligible, key=lambda n: table[n]['score'], default=None)
    return {
        'rule': ('max(mean corpus-task acceptance x mean retain grounding) over '
                 'candidates outside the gemma family, each corpus task at '
                 f'>= {args.min_accepted_share:.0%} of target, grounding >= '
                 f'{args.min_grounding}, projected regeneration <= '
                 f'{args.max_generation_hours} h'),
        'budget_words': args.budget_words,
        'candidates': table,
        'winner': winner,
    }


def parse_budget(values):
    budget = {}
    for value in values or []:
        kind, _, words = value.partition('=')
        if kind not in CORPUS_TASKS or not words.isdigit():
            raise SystemExit(f'bad --budget-words {value!r}')
        budget[kind] = int(words)
    return budget


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--candidate', action='append',
                        help='Restrict to these candidates (repeatable).')
    parser.add_argument('--task', action='append',
                        choices=PROMPT_TASKS + CORPUS_TASKS,
                        help='Tasks to score (repeatable). Default: the P4 '
                             'prompt tasks only.')
    parser.add_argument('--rounds', type=int, default=6,
                        help='Generation calls per prompt task per candidate.')
    parser.add_argument('--per-round', type=int, default=8,
                        help='Pairs requested per call.')
    parser.add_argument('--corpus-target', type=int, default=40,
                        help='Rows requested per corpus task per candidate.')
    parser.add_argument('--page-id-max', type=int, default=120_000)
    parser.add_argument('--timeout', type=int, default=900)
    parser.add_argument('--max-tokens', type=int, default=2400)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--max-repeat', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1969)
    parser.add_argument(
        '--seed-file',
        default='/mnt/data/deepred_corpus/p4-v1/persona/persona_seed.jsonl')
    parser.add_argument('--container', default='llama-rocm-10.0')
    parser.add_argument('--port', type=int, default=18095)
    parser.add_argument('--context-size', type=int, default=8192)
    parser.add_argument('--rescore', action='store_true',
                        help='Re-run candidates that already have results.')
    parser.add_argument('--select', action='store_true',
                        help='Record a winner under the P8 selection rule.')
    parser.add_argument('--budget-words', action='append',
                        help='KIND=WORDS regeneration budget, used to project '
                             'generation hours (repeatable).')
    # The incumbent's p4-v1 retain assets measure 0.98-0.99 on this metric.
    parser.add_argument('--min-grounding', type=float, default=0.95)
    parser.add_argument('--min-accepted-share', type=float, default=0.5)
    parser.add_argument('--max-generation-hours', type=float, default=168)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.budget_words = parse_budget(args.budget_words)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_candidate = output_dir / 'candidates'
    per_candidate.mkdir(exist_ok=True)
    holdout = list(CURATED_HOLDOUT)
    names = args.candidate or list(CANDIDATES)
    unknown = sorted(set(names) - set(CANDIDATES))
    if unknown:
        raise SystemExit(f'unknown candidates: {unknown}')

    prompts = {'persona': PERSONA_PROMPT, 'explanatory': explanatory_prompt}
    task_names = args.task or list(PROMPT_TASKS)
    results = {}
    for name in names:
        spec = CANDIDATES[name]
        saved = per_candidate / f'{name}.json'
        if saved.is_file() and not args.rescore:
            entry = json.loads(saved.read_text())
            if (entry.get('status') == 'ok'
                    and set(task_names) <= set(entry.get('tasks', {}))):
                print(f'\n=== {name}: reusing {saved}', flush=True)
                results[name] = entry
                continue
        print(f'\n=== {name} ({spec["family"]}, {spec["role"]}) ===', flush=True)
        if not spec['path'].is_file():
            print(f'  skipped: {spec["path"]} not present', flush=True)
            results[name] = {'status': 'missing', 'path': str(spec['path']),
                             'family': spec['family'], 'role': spec['role']}
            continue
        entry = {'status': 'ok', 'family': spec['family'],
                 'role': spec['role'], 'path': str(spec['path']),
                 'bytes': spec['path'].stat().st_size, 'tasks': {}}
        endpoint = f'http://127.0.0.1:{args.port}'
        with ServedModel(name, spec, args.container, args.port,
                         args.context_size,
                         output_dir / 'logs' / f'{name}.log'):
            client = InferenceClient(endpoint, name)
            for task in task_names:
                print(f'  task {task}...', flush=True)
                if task in CORPUS_TASKS:
                    entry['tasks'][task] = run_corpus_task(
                        task, endpoint, name, args,
                        output_dir / 'scratch' / name)
                    continue
                # Same seed per task for every candidate: identical topics and
                # examples, so the generator is the only variable.
                rng = random.Random(f'{args.seed}:{task}')
                template = prompts[task]
                if callable(template):
                    template = template()
                entry['tasks'][task] = run_task(
                    client, template, args.rounds, args, rng, holdout)
                summary = entry['tasks'][task]
                print(f"    accepted {summary['pairs_accepted']}/"
                      f"{summary['pairs_offered']} "
                      f"({summary['acceptance_rate']:.0%}), "
                      f"median {summary['median_words']} words, "
                      f"{summary['accepted_per_minute']}/min", flush=True)
        saved.write_text(json.dumps(entry, indent=2) + '\n', encoding='utf-8')
        results[name] = entry

    verdict = {
        'schema_version': 2,
        'probe': 'generator_bakeoff',
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'backend': f'container:{args.container}',
        'settings': {'rounds': args.rounds, 'per_round': args.per_round,
                     'corpus_target': args.corpus_target,
                     'page_id_max': args.page_id_max,
                     'temperature': args.temperature, 'seed': args.seed,
                     'tasks': task_names},
        'length_band': list(LENGTH_BAND),
        'factual_accuracy': ('grounding of names and numbers against the '
                             'source article, on ' + ', '.join(GROUNDED_TASKS)),
        'candidates': results,
    }
    if args.select:
        verdict['selection'] = select_winner(results, args)
        print(f"\nselection: {verdict['selection']['winner']}")
        for name, row in verdict['selection']['candidates'].items():
            print(f"  {name:24} score {row.get('score', '-')} "
                  f"acceptance {row.get('acceptance', '-')} "
                  f"grounding {row.get('grounding', '-')} "
                  f"hours {row.get('projected_hours', '-')} "
                  f"{'; '.join(row['reasons']) or 'eligible'}")
    (output_dir / 'verdict.json').write_text(
        json.dumps(verdict, indent=2) + '\n', encoding='utf-8')
    print(f"\nverdict -> {output_dir / 'verdict.json'}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
