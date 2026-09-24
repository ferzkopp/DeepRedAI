#!/usr/bin/env python3
"""Score generator candidates on the tasks Phase 4 actually regenerates.

Implements DeepRed-Phase4-Plan.md P4.2. Narrow by design: only `persona` and
the 60-150 word length slice are regenerated, so those are the only tasks a
generator is chosen on.

The acceptance guards are imported from `generate_deepred_corpus.py` rather
than reimplemented, so "accepted here" means exactly what it will mean during
generation. Candidates therefore compare without human grading.

Usage:
  python3 scripts/benchmark_generators.py \\
      --output-dir /mnt/data/evaluations/phase4/generator-bakeoff \\
      --rounds 6
"""

import argparse
import json
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
    ASSISTANT_TELLS, CURATED_HOLDOUT, GenerationError, InferenceClient,
    PERSONA_PROMPT, PERSONA_TOPICS, is_held_out, load_seed_examples,
    parse_json_array, valid_pair,
)

MODELS_DIR = Path('/mnt/data/models/llm')

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


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--candidate', action='append',
                        help='Restrict to these candidates (repeatable).')
    parser.add_argument('--rounds', type=int, default=6,
                        help='Generation calls per task per candidate.')
    parser.add_argument('--per-round', type=int, default=8,
                        help='Pairs requested per call.')
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
    return parser


def main(argv=None):
    import random

    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    holdout = list(CURATED_HOLDOUT)
    names = args.candidate or list(CANDIDATES)
    unknown = sorted(set(names) - set(CANDIDATES))
    if unknown:
        raise SystemExit(f'unknown candidates: {unknown}')

    tasks = {'persona': PERSONA_PROMPT, 'explanatory': explanatory_prompt()}
    results = {}
    for name in names:
        spec = CANDIDATES[name]
        print(f'\n=== {name} ({spec["family"]}, {spec["role"]}) ===', flush=True)
        if not spec['path'].is_file():
            print(f'  skipped: {spec["path"]} not present', flush=True)
            results[name] = {'status': 'missing', 'path': str(spec['path'])}
            continue
        entry = {'status': 'ok', 'family': spec['family'],
                 'role': spec['role'], 'path': str(spec['path']),
                 'bytes': spec['path'].stat().st_size, 'tasks': {}}
        with ServedModel(name, spec, args.container, args.port,
                         args.context_size,
                         output_dir / 'logs' / f'{name}.log'):
            client = InferenceClient(f'http://127.0.0.1:{args.port}', name)
            for task, template in tasks.items():
                # Same seed per task for every candidate: identical topics and
                # examples, so the generator is the only variable.
                rng = random.Random(f'{args.seed}:{task}')
                print(f'  task {task}...', flush=True)
                entry['tasks'][task] = run_task(
                    client, template, args.rounds, args, rng, holdout)
                summary = entry['tasks'][task]
                print(f"    accepted {summary['pairs_accepted']}/"
                      f"{summary['pairs_offered']} "
                      f"({summary['acceptance_rate']:.0%}), "
                      f"median {summary['median_words']} words, "
                      f"{summary['accepted_per_minute']}/min", flush=True)
        results[name] = entry

    verdict = {
        'schema_version': 1,
        'probe': 'generator_bakeoff',
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'backend': f'container:{args.container}',
        'settings': {'rounds': args.rounds, 'per_round': args.per_round,
                     'temperature': args.temperature, 'seed': args.seed},
        'length_band': list(LENGTH_BAND),
        'factual_accuracy': 'not measured — retain is reused, not regenerated; '
                            'deferred to P8 where retain is in scope',
        'candidates': results,
    }
    (output_dir / 'verdict.json').write_text(
        json.dumps(verdict, indent=2) + '\n', encoding='utf-8')
    print(f"\nverdict -> {output_dir / 'verdict.json'}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
