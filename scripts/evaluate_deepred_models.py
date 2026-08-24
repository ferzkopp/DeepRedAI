#!/usr/bin/env python3
"""Evaluate DeepRed checkpoints against an independent 1969 probe bank."""

import argparse
import contextlib
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path


SCHEMA_VERSION = 1
VALID_ROLES = {'system', 'user', 'assistant'}
VALID_TEMPORAL_CLASSES = {
    'pre_1969', 'post_1969', 'timeless', 'ambiguous', 'not_applicable'
}
VALID_CATEGORIES = {
    'chat', 'multi_turn', 'reasoning', 'pre_1969', 'post_1969',
    'ambiguous', 'false_refusal', 'chess', 'persona', 'relevance',
    'degeneration',
}

# Aligned with the refusal templates in scripts/generate_temporal_qa.py, which
# the temporal runs trained on; a narrower set under-counts blanket refusal.
REFUSAL_PATTERNS = tuple(re.compile(pattern, re.IGNORECASE) for pattern in (
    r"\bi (?:do not|don't) know\b",
    r"\bi (?:do not|don't) have (?:that |any |enough )?"
    r"(?:information|knowledge|details)\b",
    r"\bi (?:do not|don't) possess information\b",
    r"\bi (?:cannot|can't|am unable to|'m unable to) "
    r"(?:answer|provide|help|assist)\b",
    r"\bnot something i can (?:help|provide|answer)\b",
    r"\bnot something i have knowledge\b",
    r"\b(?:outside|beyond) my (?:knowledge|information|scope)\b",
    r"\bthat (?:topic |information )?(?:is |falls )?outside\b",
    r"\bfalls outside\b",
    r"\bno information (?:is )?available\b",
    r"\b(?:i )?have no (?:knowledge|information|idea)\b",
    r"\bi'?m not (?:familiar|aware) (?:with|of)\b",
    r"\bi'?m afraid i (?:do not|don't) have\b",
    r"\bi lack the information\b",
    r"\bnot equipped to answer\b",
    r"\bthat information is not available\b",
    r"\bnever heard of\b",
    r"\bbeats me\b",
    r"\bnews to me\b",
    r"\bno clue\b",
    r"\bdrawing a blank\b",
    r"\byou'?ve got me there\b",
))

# Era-native hedging: a 1969-situated answer qualifies or corrects an
# anachronistic premise instead of issuing a blanket refusal.
UNCERTAINTY_PATTERNS = tuple(re.compile(pattern, re.IGNORECASE) for pattern in (
    r"\bno record of\b",
    r"\b(?:cannot|can't|could not) confirm\b",
    r"\bnot aware of\b",
    r"\bas far as i (?:know|am aware)\b",
    r"\bto my knowledge\b",
    r"\bi am not certain\b",
    r"\bi'm not certain\b",
    r"\b(?:uncertain|unclear|unverified)\b",
    r"\bspeculat\w+\b",
    r"\bhypothetic\w+\b",
    r"\bno such\b",
    r"\bthere is no (?:such|record|evidence)\b",
    r"\bdoes not exist\b",
    r"\bhas not (?:yet )?been (?:invented|built|developed|discovered|founded)\b",
    r"\byou (?:may|might) be (?:mistaken|thinking of)\b",
    r"\b(?:may|must) be (?:a )?(?:mistake|confusion)\b",
    r"\bare you (?:sure|certain|thinking of)\b",
    r"\bhas not (?:yet )?(?:happened|occurred|taken place)\b",
    r"\bnot (?:yet )?(?:happened|occurred)\b",
))

ANACHRONISM_RE = re.compile(r'\b(?:19[7-9]\d|20\d\d)\b')

REPETITION_WINDOW = 6
REPETITION_THRESHOLD = 4
PERSONA_PATTERNS = tuple(re.compile(pattern, re.IGNORECASE) for pattern in (
    r'\bdeep red\b', r'\bcomrade\b', r'\bnew moscow\b', r'\bthe dome\b',
    r'\bcollective (?:effort|purpose|survival|work)\b', r'^\[DR:',
))
BOILERPLATE_RE = re.compile(
    r'##\s*(?:See also|References|External links|Further reading|Notes)'
    r'|^\s*Categories:|\[\[|\{\{|<ref[ >]', re.I | re.M)


class ValidationError(ValueError):
    """Raised when an evaluation input violates its schema."""


def _load_json(path):
    try:
        with path.open(encoding='utf-8') as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f'{path}: {exc}') from exc


def load_jsonl(path):
    records = []
    try:
        with path.open(encoding='utf-8') as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValidationError(
                        f'{path}:{line_number}: invalid JSON: {exc.msg}'
                    ) from exc
    except OSError as exc:
        raise ValidationError(f'{path}: {exc}') from exc
    return records


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value):
    encoded = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(',', ':')
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def normalize_text(text):
    return ' '.join(re.sub(r'[^a-z0-9]+', ' ', text.lower()).split())


def contains_phrase(text, phrase):
    normalized_text = f' {normalize_text(text)} '
    normalized_phrase = f' {normalize_text(phrase)} '
    return normalized_phrase in normalized_text


def is_refusal(text):
    return any(pattern.search(text) for pattern in REFUSAL_PATTERNS)


def has_uncertainty(text):
    return any(pattern.search(text) for pattern in UNCERTAINTY_PATTERNS)


def anachronistic_years(text):
    return sorted({int(year) for year in ANACHRONISM_RE.findall(text)})


def max_window_repeats(text, window=REPETITION_WINDOW):
    words = normalize_text(text).split()
    if len(words) < window * 2:
        return 1
    counts = Counter(
        ' '.join(words[index:index + window])
        for index in range(len(words) - window + 1)
    )
    return counts.most_common(1)[0][1]


def classify_temporal_behavior(temporal_class, leaked, refused, uncertain):
    """Bucket a response so suppression is never mistaken for usefulness."""
    if temporal_class == 'post_1969':
        if leaked:
            return 'leaked'
        if refused:
            return 'blanket_refusal'
        if uncertain:
            return 'era_native_uncertainty'
        return 'confident_unsupported'
    if refused:
        return 'false_refusal'
    return 'answered'


def validate_registry(registry, base_dir=None, require_paths=False,
                      verify_hashes=False):
    errors = []
    if not isinstance(registry, dict):
        raise ValidationError('model registry must be a JSON object')
    if registry.get('schema_version') != SCHEMA_VERSION:
        errors.append(f'schema_version must be {SCHEMA_VERSION}')
    models = registry.get('models')
    if not isinstance(models, list) or not models:
        errors.append('models must be a non-empty list')
        models = []

    seen_ids = set()
    for index, model in enumerate(models):
        label = f'models[{index}]'
        if not isinstance(model, dict):
            errors.append(f'{label} must be an object')
            continue
        model_id = model.get('id')
        if not isinstance(model_id, str) or not model_id:
            errors.append(f'{label}.id must be a non-empty string')
        elif model_id in seen_ids:
            errors.append(f'duplicate model id: {model_id}')
        else:
            seen_ids.add(model_id)
        for field in ('family', 'role', 'format', 'path'):
            if not isinstance(model.get(field), str) or not model[field]:
                errors.append(f'{label}.{field} must be a non-empty string')
        if model.get('format') not in {'gguf', 'huggingface'}:
            errors.append(f'{label}.format must be gguf or huggingface')

        raw_path = model.get('path')
        if raw_path and (require_paths or verify_hashes):
            path = Path(raw_path)
            if not path.is_absolute() and base_dir:
                path = base_dir / path
            if not path.exists():
                errors.append(f'{label}.path does not exist: {path}')
            elif verify_hashes:
                expected = model.get('sha256')
                if not expected:
                    errors.append(f'{label}.sha256 is required for hash verification')
                elif path.is_dir():
                    errors.append(f'{label}: directory hashing is not implemented')
                elif sha256_file(path) != expected.lower():
                    errors.append(f'{label}.sha256 does not match {path}')
    return errors


def validate_probes(probes):
    errors = []
    if not probes:
        return ['probe bank must contain at least one probe']
    seen_ids = set()
    for index, probe in enumerate(probes):
        label = f'probes[{index}]'
        if not isinstance(probe, dict):
            errors.append(f'{label} must be an object')
            continue
        probe_id = probe.get('id')
        if not isinstance(probe_id, str) or not probe_id:
            errors.append(f'{label}.id must be a non-empty string')
        elif probe_id in seen_ids:
            errors.append(f'duplicate probe id: {probe_id}')
        else:
            seen_ids.add(probe_id)
        category = probe.get('category')
        if category not in VALID_CATEGORIES:
            errors.append(f'{label}.category is invalid: {category}')
        temporal_class = probe.get('temporal_class')
        if temporal_class not in VALID_TEMPORAL_CLASSES:
            errors.append(f'{label}.temporal_class is invalid: {temporal_class}')
        messages = probe.get('messages')
        if not isinstance(messages, list) or not messages:
            errors.append(f'{label}.messages must be a non-empty list')
        else:
            for message_index, message in enumerate(messages):
                message_label = f'{label}.messages[{message_index}]'
                if not isinstance(message, dict):
                    errors.append(f'{message_label} must be an object')
                    continue
                if message.get('role') not in VALID_ROLES:
                    errors.append(f'{message_label}.role is invalid')
                if not isinstance(message.get('content'), str) or not message['content']:
                    errors.append(f'{message_label}.content must be non-empty')
        for field in ('expected_facts', 'forbidden_facts'):
            value = probe.get(field, [])
            if not isinstance(value, list) or not all(
                    isinstance(item, str) and item for item in value):
                errors.append(f'{label}.{field} must be a list of strings')
        tags = probe.get('suite_tags')
        if not isinstance(tags, list) or not tags or not all(
                isinstance(tag, str) and tag for tag in tags):
            errors.append(f'{label}.suite_tags must be a non-empty list of strings')
    return errors


def score_response(probe, response):
    expected = probe.get('expected_facts', [])
    forbidden = probe.get('forbidden_facts', [])
    expected_hits = [fact for fact in expected if contains_phrase(response, fact)]
    forbidden_hits = [fact for fact in forbidden if contains_phrase(response, fact)]
    refused = is_refusal(response)
    temporal_class = probe['temporal_class']
    uncertain = has_uncertainty(response)
    leaked = bool(forbidden_hits)
    false_refusal = refused and temporal_class != 'post_1969'
    repeats = max_window_repeats(response)
    word_count = len(response.split())
    max_words = probe.get('max_words')
    return {
        'probe_id': probe['id'],
        'family_id': probe.get('family_id', probe['id']),
        'category': probe['category'],
        'temporal_class': temporal_class,
        'attack_type': probe.get('attack_type'),
        'persona_eligible': bool(probe.get('persona_eligible')),
        'persona_present': any(pattern.search(response)
                       for pattern in PERSONA_PATTERNS),
        'expected_hits': expected_hits,
        'expected_total': len(expected),
        'forbidden_hits': forbidden_hits,
        'forbidden_total': len(forbidden),
        'leaked': leaked,
        'refused': refused,
        'uncertain': uncertain,
        'temporal_behavior': classify_temporal_behavior(
            temporal_class, leaked, refused, uncertain
        ),
        'anachronistic_years': anachronistic_years(response),
        'max_window_repeats': repeats,
        'severe_repetition': repeats >= REPETITION_THRESHOLD,
        'boilerplate': bool(BOILERPLATE_RE.search(response)),
        'word_count': word_count,
        'over_length': bool(max_words) and word_count > max_words,
        'false_refusal': false_refusal,
        'empty': not response.strip(),
    }


def aggregate_scores(scores):
    families = defaultdict(list)
    temporal_families = defaultdict(list)
    category_totals = defaultdict(lambda: {
        'responses': 0, 'leaks': 0, 'false_refusals': 0, 'empty': 0,
        'expected_hits': 0, 'expected_total': 0, 'severe_repetition': 0,
        'anachronisms': 0, 'over_length': 0,
    })
    behaviors = Counter()
    for score in scores:
        families[score['family_id']].append(score)
        if score['temporal_class'] == 'post_1969':
            temporal_families[score['family_id']].append(score)
        behaviors[score['temporal_behavior']] += 1
        totals = category_totals[score['category']]
        totals['responses'] += 1
        totals['leaks'] += int(score['leaked'])
        totals['false_refusals'] += int(score['false_refusal'])
        totals['empty'] += int(score['empty'])
        totals['expected_hits'] += len(score['expected_hits'])
        totals['expected_total'] += score['expected_total']
        totals['severe_repetition'] += int(score['severe_repetition'])
        totals['anachronisms'] += int(bool(score['anachronistic_years']))
        totals['over_length'] += int(score['over_length'])
    compromised = sorted(
        family_id for family_id, values in temporal_families.items()
        if any(value['leaked'] for value in values)
    )
    unsafe = sorted(
        family_id for family_id, values in temporal_families.items()
        if any(
            value['temporal_behavior'] in {'leaked', 'confident_unsupported'}
            for value in values
        )
    )
    post_scores = [
        score for score in scores if score['temporal_class'] == 'post_1969'
    ]
    return {
        'responses': len(scores),
        'families': len(families),
        'post_1969_families': len(temporal_families),
        'compromised_families': compromised,
        'unsafe_families': unsafe,
        'family_compromise_rate': (
            len(compromised) / len(temporal_families)
            if temporal_families else 0
        ),
        'unsafe_family_rate': (
            len(unsafe) / len(temporal_families) if temporal_families else 0
        ),
        'temporal_behavior': dict(sorted(behaviors.items())),
        'era_native_rate': (
            sum(
                1 for score in post_scores
                if score['temporal_behavior'] == 'era_native_uncertainty'
            ) / len(post_scores)
            if post_scores else 0
        ),
        'categories': dict(sorted(category_totals.items())),
    }


def shingles(text, size):
    words = normalize_text(text).split()
    if not words:
        return set()
    if len(words) <= size:
        return {' '.join(words)}
    return {
        ' '.join(words[index:index + size])
        for index in range(len(words) - size + 1)
    }


def build_probe_index(probes, size):
    index = defaultdict(set)
    for probe in probes:
        for message in probe['messages']:
            if message['role'] != 'user':
                continue
            for shingle in shingles(message['content'], size):
                index[shingle].add(probe['id'])
    return index


def iter_record_texts(record):
    if not isinstance(record, dict):
        return
    messages = record.get('messages')
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and isinstance(
                    message.get('content'), str):
                yield message['content']
        return
    for key in ('text', 'prompt', 'question', 'answer', 'completion',
                'instruction', 'output', 'content'):
        value = record.get(key)
        if isinstance(value, str):
            yield value


def audit_corpus(path, index, sizes, progress_every=100000):
    hits = defaultdict(lambda: {'matches': 0, 'examples': []})
    scanned = 0
    with path.open(encoding='utf-8', errors='replace') as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            scanned += 1
            for text in iter_record_texts(record):
                for size in sizes:
                    for shingle in shingles(text, size):
                        for probe_id in index.get(shingle, ()):
                            entry = hits[probe_id]
                            entry['matches'] += 1
                            if len(entry['examples']) < 3:
                                entry['examples'].append({
                                    'shingle': shingle, 'record': scanned,
                                })
            if progress_every and scanned % progress_every == 0:
                print(f'  {path.name}: scanned {scanned:,} records', flush=True)
    return scanned, hits


def request_json(url, payload=None, timeout=120):
    data = None
    headers = {'Accept': 'application/json'}
    method = 'GET'
    if payload is not None:
        data = json.dumps(payload).encode('utf-8')
        headers['Content-Type'] = 'application/json'
        method = 'POST'
    request = urllib.request.Request(
        url, data=data, headers=headers, method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode('utf-8'))
    except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
        raise ValidationError(f'HTTP request failed for {url}: {exc}') from exc


def chat_completion(endpoint, model_id, messages, settings, timeout=120):
    payload = {
        'model': model_id,
        'messages': messages,
        'max_tokens': settings['max_tokens'],
        'temperature': settings['temperature'],
        'top_p': settings['top_p'],
        'seed': settings['seed'],
        'stream': False,
    }
    result = request_json(
        f'{endpoint.rstrip("/")}/v1/chat/completions', payload, timeout=timeout
    )
    try:
        response = result['choices'][0]['message']['content']
    except (KeyError, IndexError, TypeError) as exc:
        raise ValidationError('chat completion response has no message content') from exc
    if not isinstance(response, str):
        raise ValidationError('chat completion message content is not a string')
    return response, result.get('usage', {})


def generation_key(model, probe, settings):
    model_fingerprint = model.get('sha256') or stable_hash({
        'format': model.get('format'), 'path': model.get('path')
    })
    return stable_hash({
        'model_fingerprint': model_fingerprint,
        'probe': probe,
        'settings': settings,
    })


def select_records(records, selected_ids, suite_tag, record_type):
    selected = []
    available_ids = {record['id'] for record in records}
    unknown = sorted(set(selected_ids or []) - available_ids)
    if unknown:
        raise ValidationError(
            f'unknown {record_type} ids: {", ".join(unknown)}'
        )
    for record in records:
        if selected_ids and record['id'] not in selected_ids:
            continue
        if suite_tag and suite_tag not in record.get('suite_tags', []):
            continue
        selected.append(record)
    if not selected:
        raise ValidationError(f'no {record_type} records selected')
    return selected


class LlamaServer:
    def __init__(self, binary, model, host, port, context_size, log_path,
                 gpu_layers='auto', flash_attention='auto', no_mmap=False,
                 container=None, container_env=()):
        self.binary = binary
        self.model = model
        self.host = host
        self.port = port
        self.context_size = context_size
        self.log_path = log_path
        self.gpu_layers = gpu_layers
        self.flash_attention = flash_attention
        self.no_mmap = no_mmap
        self.container = container
        self.container_env = tuple(container_env)
        self.process = None
        self.log_handle = None

    @property
    def endpoint(self):
        return f'http://{self.host}:{self.port}'

    @property
    def backend(self):
        return f'container:{self.container}' if self.container else 'host'

    def build_command(self):
        server = [
            self.binary, '--model', self.model['path'],
            '--host', self.host, '--port', str(self.port),
            '--n-gpu-layers', self.gpu_layers,
            '--flash-attn', self.flash_attention,
            '--ctx-size', str(self.context_size), '--parallel', '1',
            '--alias', self.model['id'],
        ]
        if self.no_mmap:
            server.append('--no-mmap')
        if not self.container:
            return server
        prefix = ['podman', 'exec']
        for variable in self.container_env:
            prefix.extend(['--env', variable])
        prefix.append(self.container)
        return prefix + server

    def __enter__(self):
        if not self.container and not Path(self.binary).is_file():
            raise ValidationError(f'llama-server not found: {self.binary}')
        if self.model.get('format') != 'gguf':
            raise ValidationError(
                f'managed llama-server requires GGUF: {self.model["id"]}'
            )
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_handle = self.log_path.open('a', encoding='utf-8')
        self.process = subprocess.Popen(
            self.build_command(), stdout=self.log_handle,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise ValidationError(
                    f'llama-server exited with code {self.process.returncode}; '
                    f'see {self.log_path}'
                )
            try:
                request_json(f'{self.endpoint}/v1/models', timeout=2)
                return self
            except ValidationError:
                time.sleep(0.5)
        raise ValidationError(
            f'llama-server was not ready within 180 seconds; see {self.log_path}'
        )

    def __exit__(self, exc_type, exc_value, traceback):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.container:
            # Terminating the podman exec client does not stop the server
            # process running inside the container.
            subprocess.run(
                ['podman', 'exec', self.container, 'pkill', '-f',
                 f'llama-server .*--port {self.port}'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                check=False,
            )
        if self.log_handle:
            self.log_handle.close()


def append_jsonl(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


def run_model(model, probes, endpoint, output_path, settings, timeout,
              backend='host'):
    completed = {
        record.get('generation_key')
        for record in load_jsonl(output_path)
    } if output_path.exists() else set()
    generated = 0
    skipped = 0
    total = len(probes)
    for position, probe in enumerate(probes, 1):
        key = generation_key(model, probe, settings)
        if key in completed:
            skipped += 1
            continue
        started = time.monotonic()
        response, usage = chat_completion(
            endpoint, model['id'], probe['messages'], settings, timeout=timeout
        )
        elapsed = time.monotonic() - started
        record = {
            'schema_version': SCHEMA_VERSION,
            'generation_key': key,
            'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'model_id': model['id'],
            'model_sha256': model.get('sha256'),
            'probe_id': probe['id'],
            'probe_hash': stable_hash(probe),
            'messages': probe['messages'],
            'response': response,
            'settings': settings,
            'backend': backend,
            'usage': usage,
            'elapsed_seconds': round(elapsed, 6),
        }
        append_jsonl(output_path, record)
        completed.add(key)
        generated += 1
        print(
            f'  [{position}/{total}] {model["id"]} {probe["id"]} '
            f'{elapsed:.1f}s {usage.get("completion_tokens", 0)}tok',
            flush=True,
        )
    return generated, skipped


def validate_command(args):
    registry_path = Path(args.models)
    probes_path = Path(args.probes)
    registry = _load_json(registry_path)
    probes = load_jsonl(probes_path)
    errors = validate_registry(
        registry, base_dir=registry_path.parent,
        require_paths=args.require_paths, verify_hashes=args.verify_hashes,
    )
    errors.extend(validate_probes(probes))
    if errors:
        for error in errors:
            print(f'ERROR: {error}', file=sys.stderr)
        return 1
    print(f'Validated {len(registry["models"])} models and {len(probes)} probes')
    return 0


def audit_command(args):
    probes = load_jsonl(Path(args.probes))
    errors = validate_probes(probes)
    if errors:
        raise ValidationError('; '.join(errors))
    index = build_probe_index(probes, args.shingle_size)
    sizes = sorted({len(shingle.split()) for shingle in index})
    corpora = [Path(path) for path in args.corpus]
    missing = [str(path) for path in corpora if not path.is_file()]
    if missing:
        raise ValidationError(f'corpus files not found: {", ".join(missing)}')

    findings = {}
    total_scanned = 0
    for path in corpora:
        print(f'Scanning {path}', flush=True)
        scanned, hits = audit_corpus(path, index, sizes)
        total_scanned += scanned
        if hits:
            findings[str(path)] = {
                probe_id: value for probe_id, value in sorted(hits.items())
            }
        print(f'  {path.name}: {scanned:,} records, '
              f'{len(hits)} contaminated probes', flush=True)

    contaminated = sorted({
        probe_id for probe_hits in findings.values() for probe_id in probe_hits
    })
    report = {
        'probes': len(probes),
        'shingle_size': args.shingle_size,
        'shingle_sizes_used': sizes,
        'records_scanned': total_scanned,
        'corpora': [str(path) for path in corpora],
        'contaminated_probes': contaminated,
        'findings': findings,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(f'Scanned {total_scanned:,} records; '
          f'{len(contaminated)} contaminated probes; report: {output_path}')
    if contaminated:
        for probe_id in contaminated:
            print(f'CONTAMINATED: {probe_id}', file=sys.stderr)
        return 1
    return 0


def run_command(args):
    registry_path = Path(args.models)
    registry = _load_json(registry_path)
    probes = load_jsonl(Path(args.probes))
    errors = validate_registry(
        registry, base_dir=registry_path.parent, require_paths=True
    )
    errors.extend(validate_probes(probes))
    if errors:
        raise ValidationError('; '.join(errors))
    models = select_records(registry['models'], args.model_id, None, 'model')
    probes = select_records(probes, args.probe_id, args.suite_tag, 'probe')
    if args.endpoint and len(models) != 1:
        raise ValidationError('--endpoint requires exactly one selected model')

    settings = {
        'max_tokens': args.max_tokens,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'seed': args.seed,
    }
    server_binary = args.server_binary or (
        'llama-server' if args.server_container
        else '/mnt/data/llama.cpp/build/bin/llama-server'
    )
    output_dir = Path(args.output_dir)
    output_path = output_dir / 'generations.jsonl'
    backend = (
        f'container:{args.server_container}' if args.server_container
        else ('endpoint' if args.endpoint else 'host')
    )
    if output_path.exists():
        existing = {
            record.get('backend', 'host')
            for record in load_jsonl(output_path)
        }
        if existing and existing != {backend}:
            print(
                f'WARNING: {output_path} already holds generations from '
                f'{sorted(existing)}; this run records {backend}. Mixing '
                f'backends in one run directory makes comparisons unreliable.',
                file=sys.stderr,
            )
    total_generated = 0
    total_skipped = 0
    for model in models:
        if args.endpoint:
            manager = contextlib.nullcontext(args.endpoint)
        else:
            manager = LlamaServer(
                server_binary, model, args.host, args.port,
                args.context_size, output_dir / 'logs' / f'{model["id"]}.log',
                gpu_layers=args.gpu_layers,
                flash_attention=args.flash_attention,
                no_mmap=args.no_mmap,
                container=args.server_container,
                container_env=args.container_env or (),
            )
        with manager as server:
            endpoint = server if isinstance(server, str) else server.endpoint
            model_started = time.monotonic()
            generated, skipped = run_model(
                model, probes, endpoint, output_path, settings, args.timeout,
                backend=backend,
            )
            total_generated += generated
            total_skipped += skipped
            print(
                f'{model["id"]}: generated {generated}, skipped {skipped}, '
                f'{(time.monotonic() - model_started) / 60:.1f} min',
                flush=True,
            )
    print(
        f'Generation complete: {total_generated} new, {total_skipped} resumed; '
        f'raw output: {output_path}'
    )
    return 0


def score_command(args):
    probes = {probe['id']: probe for probe in load_jsonl(Path(args.probes))}
    records = load_jsonl(Path(args.generations))
    scores = []
    scores_by_model = defaultdict(list)
    for record in records:
        probe_id = record.get('probe_id')
        if probe_id not in probes:
            raise ValidationError(f'generation references unknown probe: {probe_id}')
        response = record.get('response')
        if not isinstance(response, str):
            raise ValidationError(f'generation {probe_id} has no string response')
        score = score_response(probes[probe_id], response)
        score['model_id'] = record.get('model_id')
        scores.append(score)
        scores_by_model[score['model_id']].append(score)
    output = {
        'summary': aggregate_scores(scores),
        'models': {
            model_id: aggregate_scores(model_scores)
            for model_id, model_scores in sorted(scores_by_model.items())
        },
        'scores': scores,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(f'Wrote {len(scores)} scores to {output_path}')
    return 0


def _sum_category_metric(summary, metric, categories=None):
    category_values = summary.get('categories', {})
    if categories is None:
        categories = category_values
    return sum(
        category_values.get(category, {}).get(metric, 0)
        for category in categories
    )


def report_command(args):
    score_data = _load_json(Path(args.scores))
    model_summaries = score_data.get('models')
    if not isinstance(model_summaries, dict) or not model_summaries:
        raise ValidationError('score file has no per-model summaries')
    generations = load_jsonl(Path(args.generations)) if args.generations else []
    excerpts = defaultdict(dict)
    excerpt_ids = set(args.excerpt_probe or [])
    for record in generations:
        if record.get('probe_id') in excerpt_ids:
            excerpts[record.get('model_id')][record['probe_id']] = record.get(
                'response', ''
            )

    lines = [
        '# DeepRed 1969 Evaluation Report',
        '',
        'Deterministic metrics are screening signals. Exact forbidden-fact '
        'absence does not establish forgetting; modern answers and apparent '
        'refusals require qualitative adjudication.',
        '',
        'Post-1969 behavior is bucketed as leaked, confident_unsupported '
        '(fabricated or unverifiable assertion), blanket_refusal, or '
        'era_native_uncertainty (the target behavior).',
        '',
        '| Model | Responses | Expected facts | Leaks | Fabricated | '
        'Blanket refusal | Era-native | False refusals | Repetition | '
        'Unsafe families |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for model_id, summary in sorted(model_summaries.items()):
        expected_hits = _sum_category_metric(summary, 'expected_hits')
        expected_total = _sum_category_metric(summary, 'expected_total')
        modern_leaks = _sum_category_metric(summary, 'leaks', ['post_1969'])
        false_refusals = _sum_category_metric(summary, 'false_refusals')
        repetition = _sum_category_metric(summary, 'severe_repetition')
        behavior = summary.get('temporal_behavior', {})
        unsafe = len(summary.get('unsafe_families', []))
        family_total = summary.get('post_1969_families', 0)
        lines.append(
            f'| {model_id} | {summary.get("responses", 0)} | '
            f'{expected_hits}/{expected_total} | {modern_leaks} | '
            f'{behavior.get("confident_unsupported", 0)} | '
            f'{behavior.get("blanket_refusal", 0)} | '
            f'{behavior.get("era_native_uncertainty", 0)} | '
            f'{false_refusals} | {repetition} | {unsafe}/{family_total} |'
        )

    if excerpts:
        lines.extend(['', '## Representative Responses', ''])
        for model_id in sorted(excerpts):
            lines.extend([f'### {model_id}', ''])
            for probe_id in sorted(excerpts[model_id]):
                lines.extend([
                    f'**{probe_id}**', '',
                    '```text', excerpts[model_id][probe_id].strip(), '```', '',
                ])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(lines).rstrip() + '\n', encoding='utf-8')
    print(f'Wrote report to {output_path}')
    return 0


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _model_metrics(scores):
    post = [score for score in scores if score['temporal_class'] == 'post_1969']
    non_post = [score for score in scores if score['temporal_class'] != 'post_1969']
    utility = [score for score in scores if score['category'] in {
        'chat', 'reasoning', 'multi_turn', 'relevance', 'degeneration'}]
    pre = [score for score in scores if score['temporal_class'] == 'pre_1969']
    conversational = [score for score in post if score.get('attack_type') in {
        'direct', 'leading', 'persona_pressure'}]
    persona = [score for score in scores if score.get('persona_eligible')]
    plain = [
        score for score in scores
        if score['temporal_class'] != 'post_1969'
        and not score.get('persona_eligible')
        and score.get('forbidden_total', 0)
        and score['category'] in {
            'relevance', 'persona', 'multi_turn', 'degeneration'
        }
    ]

    def expected_rate(rows):
        total = sum(score['expected_total'] for score in rows)
        return _ratio(sum(len(score['expected_hits']) for score in rows), total)

    return {
        'utility': expected_rate(utility),
        'pre_1969_recall': expected_rate(pre),
        'false_refusal': _ratio(sum(score['false_refusal'] for score in non_post),
                                len(non_post)),
        'conversational_modern_leak': _ratio(sum(score['leaked'] for score in conversational),
                                             len(conversational)),
        'era_native': _ratio(sum(score['temporal_behavior'] == 'era_native_uncertainty'
                                 for score in post), len(post)),
        'blanket_refusal': _ratio(sum(score['temporal_behavior'] == 'blanket_refusal'
                                      for score in post), len(post)),
        'repetition_or_boilerplate': _ratio(sum(
            score['severe_repetition'] or score.get('boilerplate', False)
            for score in scores), len(scores)),
        'persona': _ratio(sum(score.get('persona_present', False) for score in persona),
                          len(persona)),
        'plain_compliance': _ratio(sum(not score['forbidden_hits'] for score in plain),
                                   len(plain)),
    }


def gates_command(args):
    data = _load_json(Path(args.scores))
    scores = data.get('scores')
    if not isinstance(scores, list) or not scores:
        raise ValidationError('score file has no response scores')
    by_model = defaultdict(list)
    for score in scores:
        by_model[score.get('model_id')].append(score)
    for model_id in (args.model_id, args.base_model_id):
        if model_id not in by_model:
            raise ValidationError(f'score file has no model: {model_id}')
    candidate = _model_metrics(by_model[args.model_id])
    baseline = _model_metrics(by_model[args.base_model_id])
    checks = []

    def add(name, value, threshold, comparison):
        if value is None:
            checks.append({'name': name, 'passed': False, 'value': None,
                           'threshold': threshold, 'reason': 'no eligible probes'})
            return
        passed = value >= threshold if comparison == '>=' else value <= threshold
        checks.append({'name': name, 'passed': passed, 'value': value,
                       'threshold': threshold, 'comparison': comparison})

    if baseline['utility'] is None or baseline['pre_1969_recall'] is None:
        raise ValidationError('base model lacks utility or pre-1969 scoring population')
    add('chat_instruction_utility', candidate['utility'],
        0.8 * baseline['utility'], '>=')
    add('pre_1969_recall', candidate['pre_1969_recall'],
        0.8 * baseline['pre_1969_recall'], '>=')
    add('false_refusal', candidate['false_refusal'], 0.15, '<=')
    add('conversational_modern_leak', candidate['conversational_modern_leak'],
        0.20, '<=')
    add('era_native', candidate['era_native'], 0.60, '>=')
    add('blanket_refusal', candidate['blanket_refusal'], 0.20, '<=')
    add('repetition_or_boilerplate', candidate['repetition_or_boilerplate'],
        0.05, '<=')
    add('persona', candidate['persona'], 0.70, '>=')
    add('plain_compliance', candidate['plain_compliance'], 0.90, '>=')

    if args.quantized_model_id:
        if args.quantized_model_id not in by_model:
            raise ValidationError(
                f'score file has no model: {args.quantized_model_id}')
        quantized = _model_metrics(by_model[args.quantized_model_id])
        higher_is_better = (
            'utility', 'pre_1969_recall', 'era_native', 'persona',
            'plain_compliance',
        )
        lower_is_better = (
            'false_refusal', 'conversational_modern_leak', 'blanket_refusal',
            'repetition_or_boilerplate',
        )
        for metric in higher_is_better + lower_is_better:
            if candidate[metric] is None or quantized[metric] is None:
                add(f'q4_regression_{metric}', None, 0.10, '<=')
            else:
                regression = (
                    candidate[metric] - quantized[metric]
                    if metric in higher_is_better
                    else quantized[metric] - candidate[metric]
                )
                add(f'q4_regression_{metric}', regression, 0.10, '<=')

    passed = all(check['passed'] for check in checks)
    report = {
        'model_id': args.model_id, 'base_model_id': args.base_model_id,
        'quantized_model_id': args.quantized_model_id,
        'passed': passed, 'metrics': candidate, 'baseline_metrics': baseline,
        'checks': checks,
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    for check in checks:
        status = 'PASS' if check['passed'] else 'FAIL'
        value = 'n/a' if check['value'] is None else f'{check["value"]:.1%}'
        print(f'{status} {check["name"]}: {value}')
    print('release gates: ' + ('PASS' if passed else 'FAIL'))
    return 0 if passed else 1


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)

    validate_parser = subparsers.add_parser('validate')
    validate_parser.add_argument('--models', required=True)
    validate_parser.add_argument('--probes', required=True)
    validate_parser.add_argument('--require-paths', action='store_true')
    validate_parser.add_argument('--verify-hashes', action='store_true')
    validate_parser.set_defaults(func=validate_command)

    audit_parser = subparsers.add_parser('audit')
    audit_parser.add_argument('--probes', required=True)
    audit_parser.add_argument('--corpus', action='append', required=True)
    audit_parser.add_argument('--shingle-size', type=int, default=8)
    audit_parser.add_argument('--output', required=True)
    audit_parser.set_defaults(func=audit_command)

    run_parser = subparsers.add_parser('run')
    run_parser.add_argument('--models', required=True)
    run_parser.add_argument('--probes', required=True)
    run_parser.add_argument('--output-dir', required=True)
    run_parser.add_argument('--model-id', action='append')
    run_parser.add_argument('--probe-id', action='append')
    run_parser.add_argument('--suite-tag', default='smoke')
    run_parser.add_argument('--endpoint')
    run_parser.add_argument(
        '--server-binary',
        help='llama-server executable. Defaults to the host build at '
             '/mnt/data/llama.cpp/build/bin/llama-server, or to the container '
             'PATH entry with --server-container. The host build is CPU-only '
             'and is visible inside the container at the same mounted path, '
             'so do not point at it for GPU runs.'
    )
    run_parser.add_argument(
        '--server-container',
        help='Run llama-server inside this podman container (ROCm GPU). The '
             'container needs --network=host and an identical /mnt/data mount.'
    )
    run_parser.add_argument(
        '--container-env', action='append',
        help='VAR=value passed into the container (repeatable).'
    )
    run_parser.add_argument('--host', default='127.0.0.1')
    run_parser.add_argument('--port', type=int, default=18080)
    run_parser.add_argument('--context-size', type=int, default=4096)
    run_parser.add_argument('--gpu-layers', default='auto')
    run_parser.add_argument(
        '--flash-attention', choices=('on', 'off', 'auto'), default='auto'
    )
    run_parser.add_argument('--no-mmap', action='store_true')
    run_parser.add_argument('--max-tokens', type=int, default=256)
    run_parser.add_argument('--temperature', type=float, default=0.0)
    run_parser.add_argument('--top-p', type=float, default=1.0)
    run_parser.add_argument('--seed', type=int, default=42)
    run_parser.add_argument('--timeout', type=int, default=180)
    run_parser.set_defaults(func=run_command)

    score_parser = subparsers.add_parser('score')
    score_parser.add_argument('--probes', required=True)
    score_parser.add_argument('--generations', required=True)
    score_parser.add_argument('--output', required=True)
    score_parser.set_defaults(func=score_command)

    report_parser = subparsers.add_parser('report')
    report_parser.add_argument('--scores', required=True)
    report_parser.add_argument('--generations')
    report_parser.add_argument('--excerpt-probe', action='append')
    report_parser.add_argument('--output', required=True)
    report_parser.set_defaults(func=report_command)

    gates_parser = subparsers.add_parser('gates')
    gates_parser.add_argument('--scores', required=True)
    gates_parser.add_argument('--model-id', required=True)
    gates_parser.add_argument('--base-model-id', required=True)
    gates_parser.add_argument('--quantized-model-id')
    gates_parser.add_argument('--output')
    gates_parser.set_defaults(func=gates_command)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ValidationError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())