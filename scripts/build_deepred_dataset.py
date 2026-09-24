#!/usr/bin/env python3
"""Build deterministic Phase 2 NPO/retain datasets from audited corpus files."""

import argparse
import hashlib
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


KINDS = ('forget', 'retain', 'era_native', 'era_native_explanatory', 'persona',
         'persona_controls',
         'era_native_formats', 'retain_formats', 'persona_identity',
         'persona_identity_controls', 'persona_capability', 'chess')
# Paired control files live beside the asset they were rewritten from.
KIND_DIRS = {
    'persona_controls': 'persona',
    'persona_identity_controls': 'persona_identity',
    'era_native_explanatory': 'era_native',
}
BOILERPLATE = re.compile(
    r'##\s*(See also|References|External links|Further reading|Notes)'
    r'|^\s*Categories:|\[\[|\{\{|<ref[ >]', re.I | re.M)
# Retrieved move footers are injected at serve time, never learned.
CHESS_FOOTER = re.compile(r'\s*\[DR:[^\]]*\]')
SPACE = re.compile(r'\s+')


class DatasetError(ValueError):
    pass


def stable_hash(value):
    encoded = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(',', ':')
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def normalize(text):
    return SPACE.sub(' ', (text or '').strip())


def content_id(messages):
    canonical = [
        {'role': message.get('role'), 'content': normalize(message.get('content'))}
        for message in messages
    ]
    return stable_hash(canonical)


def read_kind(root, kind, strip_boilerplate=False, strip_chess_footer=False):
    directory = KIND_DIRS.get(kind, kind)
    path = root / directory / f'{kind}.jsonl'
    if not path.is_file():
        raise DatasetError(f'missing corpus file: {path}')
    rows = []
    with path.open(encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                source = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f'{path}:{line_number}: {exc}') from exc
            messages = source.get('messages')
            if not isinstance(messages, list) or len(messages) < 2:
                raise DatasetError(f'{path}:{line_number}: invalid messages')
            clean_messages = []
            for message in messages:
                role = message.get('role')
                text = normalize(message.get('content'))
                if role not in {'system', 'user', 'assistant'} or not text:
                    raise DatasetError(f'{path}:{line_number}: invalid message')
                if role == 'assistant' and strip_chess_footer:
                    text = normalize(CHESS_FOOTER.sub('', text))
                    if not text:
                        raise DatasetError(
                            f'{path}:{line_number}: target empty after footer removal')
                if role == 'assistant' and BOILERPLATE.search(text):
                    if not strip_boilerplate:
                        raise DatasetError(
                            f'{path}:{line_number}: boilerplate in target')
                    text = BOILERPLATE.sub('', text).strip()
                    if not text or BOILERPLATE.search(text):
                        raise DatasetError(
                            f'{path}:{line_number}: target empty after cleanup')
                clean_messages.append({'role': role, 'content': text})
            source_id = source.get('id')
            if not isinstance(source_id, str) or not source_id:
                raise DatasetError(f'{path}:{line_number}: missing stable id')
            rows.append({
                'id': f'{kind}:{source_id}',
                'source_id': source_id,
                'content_id': content_id(clean_messages),
                'kind': kind,
                'messages': clean_messages,
                **{key: source[key] for key in ('format', 'mode')
                   if source.get(key)},
            })
    return rows, path


def assign_splits(rows, val_fraction, seed):
    """Assign content groups before any per-kind sampling."""
    groups = defaultdict(list)
    for row in rows:
        groups[row['content_id']].append(row)
    assignments = {}
    for group_id in sorted(groups):
        value = int(stable_hash([seed, group_id])[:16], 16) / 16**16
        assignments[group_id] = 'val' if value < val_fraction else 'train'
    return assignments


def sample_rows(rows, assignments, limits, seed, word_caps=None,
                row_word_caps=None):
    selected = []
    by_kind = defaultdict(list)
    for row in rows:
        by_kind[row['kind']].append(row)
    for kind, candidates in sorted(by_kind.items()):
        ceiling = (row_word_caps or {}).get(kind)
        if ceiling is not None:
            candidates = [
                row for row in candidates
                if len(row['messages'][-1]['content'].split()) <= ceiling]
        limit = limits.get(kind)
        rng = random.Random(f'{seed}:{kind}')
        rng.shuffle(candidates)
        chosen = candidates if limit is None else candidates[:limit]
        cap = (word_caps or {}).get(kind)
        if cap is not None:
            kept, used = [], 0
            for row in chosen:
                length = len(row['messages'][-1]['content'].split())
                if used + length > cap:
                    continue
                kept.append(row)
                used += length
            chosen = kept
        selected.extend(chosen)
    return selected


def write_jsonl(path, rows):
    with path.open('w', encoding='utf-8') as handle:
        for row in sorted(rows, key=lambda item: item['id']):
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + '\n')


def select_kinds(root, requested):
    if requested:
        unknown = sorted(set(requested) - set(KINDS))
        if unknown:
            raise DatasetError(f'unknown kinds: {unknown}')
        return list(requested)
    present = [kind for kind in KINDS
               if (root / KIND_DIRS.get(kind, kind) / f'{kind}.jsonl').is_file()]
    if not present:
        raise DatasetError(f'no corpus files found under {root}')
    return present


def load_system_variants(path, holdout=()):
    variants = []
    seen = set()
    with Path(path).open(encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f'{path}:{line_number}: {exc}') from exc
            variant_id = record.get('id')
            text = normalize(record.get('text'))
            if not isinstance(variant_id, str) or not variant_id or not text:
                raise DatasetError(f'{path}:{line_number}: invalid variant')
            if variant_id in seen:
                raise DatasetError(f'{path}:{line_number}: duplicate variant id')
            seen.add(variant_id)
            if variant_id in set(holdout):
                continue
            variants.append({'id': variant_id, 'text': text})
    unknown = sorted(set(holdout) - seen)
    if unknown:
        raise DatasetError(f'unknown held-out variant ids: {unknown}')
    if not variants:
        raise DatasetError(f'{path}: no usable system prompt variants')
    return variants


def unit_hash(*parts):
    return int(stable_hash(list(parts))[:16], 16) / 16**16


def apply_system_prompts(rows, variants, coverage, seed):
    """Attach a deterministic system prompt to a reproducible subset of rows."""
    if not 0 <= coverage <= 1:
        raise DatasetError('--system-coverage must be between 0 and 1')
    for row in rows:
        if row['messages'][0]['role'] == 'system':
            raise DatasetError(f'{row["id"]}: source row already has a system message')
        if unit_hash(seed, 'system-coverage', row['id']) >= coverage:
            row['system_variant'] = None
            continue
        index = int(stable_hash([seed, 'system-variant', row['id']])[:16], 16)
        variant = variants[index % len(variants)]
        row['messages'] = [
            {'role': 'system', 'content': variant['text']}, *row['messages']]
        row['system_variant'] = variant['id']
    return rows


def parse_limits(values):
    limits = {}
    for value in values or []:
        try:
            kind, count = value.split('=', 1)
            count = int(count)
        except ValueError as exc:
            raise DatasetError(f'invalid --limit {value!r}; expected KIND=N') from exc
        if kind not in KINDS or count < 0:
            raise DatasetError(f'invalid --limit {value!r}')
        limits[kind] = count
    return limits


def parse_word_caps(values):
    caps = {}
    for value in values or []:
        try:
            kind, count = value.split('=', 1)
            count = int(count)
        except ValueError as exc:
            raise DatasetError(
                f'invalid --max-words {value!r}; expected KIND=N') from exc
        if kind not in KINDS or count < 0:
            raise DatasetError(f'invalid --max-words {value!r}')
        caps[kind] = count
    return caps


def parse_signal_budget(values):
    """KIND[+KIND]=MIN:MAX as percentages; either bound may be blank."""
    budget = {}
    for value in values or []:
        try:
            key, span = value.split('=', 1)
            low, high = span.split(':', 1)
        except ValueError as exc:
            raise DatasetError(
                f'invalid --signal-budget {value!r}; expected '
                f'KIND=MIN:MAX') from exc
        members = key.split('+')
        unknown = sorted(set(members) - set(KINDS))
        if unknown:
            raise DatasetError(f'invalid --signal-budget {value!r}: {unknown}')
        try:
            floor = float(low) / 100 if low.strip() else None
            ceiling = float(high) / 100 if high.strip() else None
        except ValueError as exc:
            raise DatasetError(
                f'invalid --signal-budget {value!r}: bounds must be '
                f'percentages') from exc
        if floor is None and ceiling is None:
            raise DatasetError(f'invalid --signal-budget {value!r}: no bound')
        budget[key] = (floor, ceiling)
    return budget


def check_signal_budget(signal_share, budget):
    """Return a failure line per violated bound. Empty means the build passes."""
    failures = []
    for key, (floor, ceiling) in sorted(budget.items()):
        share = sum(signal_share.get(member, 0.0)
                    for member in key.split('+'))
        if floor is not None and share < floor:
            failures.append(
                f'{key} holds {share:.1%} of the target-token signal, '
                f'below its {floor:.0%} floor')
        if ceiling is not None and share > ceiling:
            failures.append(
                f'{key} holds {share:.1%} of the target-token signal, '
                f'above its {ceiling:.0%} ceiling')
    return failures


def contingency_table(rows):
    """(system-prompted x marker-bearing) counts per kind.

    p3-v4 spent a full generate-train-evaluate cycle discovering a 0.47:1
    ratio that this count would have shown before training started.
    """
    table = {}
    for row in rows:
        conditioned = 'system' if row.get('system_variant') else 'plain'
        marked = 'marked' if MARKER_PRESENT.search(
            row['messages'][-1]['content']) else 'unmarked'
        table.setdefault(row['kind'], Counter())[f'{conditioned}_{marked}'] += 1
    return {kind: dict(sorted(counts.items()))
            for kind, counts in sorted(table.items())}


def target_token_stats(rows, tokenizer_path):
    """Mean target length in words, and in tokens when a tokenizer loads.

    P2 fixed the step-time budget at roughly 355 tokens; the trainer will not
    warn when it is exceeded, it will just take 60 s/step instead of 23.
    """
    words = [len(row['messages'][-1]['content'].split()) for row in rows]
    stats = {
        'mean_target_words': round(sum(words) / max(len(words), 1), 1),
        'max_target_words': max(words, default=0),
        'mean_sequence_tokens': None,
        'tokenizer': tokenizer_path,
    }
    if not tokenizer_path:
        stats['tokenizer_status'] = 'not requested'
        return stats
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        lengths = []
        for row in rows:
            encoded = tokenizer.apply_chat_template(
                row['messages'], tokenize=True, add_generation_prompt=False)
            # transformers 5.x returns a BatchEncoding, so len() counts keys
            # and every row measures 2 tokens.
            if hasattr(encoded, 'keys'):
                encoded = encoded['input_ids']
            if encoded and isinstance(encoded[0], (list, tuple)):
                encoded = encoded[0]
            lengths.append(len(encoded))
    except Exception as exc:
        stats['tokenizer_status'] = f'{type(exc).__name__}: {exc}'
        return stats
    if lengths and max(lengths) < 8:
        # A whole corpus of sub-8-token sequences means the count is wrong,
        # not that the corpus is tiny.
        stats['tokenizer_status'] = (
            f'implausible token counts (max {max(lengths)}); not reported')
        return stats
    stats['tokenizer_status'] = 'ok'
    stats['mean_sequence_tokens'] = round(sum(lengths) / max(len(lengths), 1), 1)
    stats['max_sequence_tokens'] = max(lengths, default=0)
    return stats


# Leading first-person forms rewritten to a Deep Red self-reference. Anchored at
# the start so only the speaker changes; nothing about the subject moves.
SELF_REWRITES = [
    (re.compile(r'^I have no record\b', re.I), 'Deep Red holds no record'),
    (re.compile(r'^I hold no record\b', re.I), 'Deep Red holds no record'),
    (re.compile(r'^There is no record\b', re.I), 'Deep Red holds no record'),
    (re.compile(r'^No record\b', re.I), 'Deep Red holds no record'),
    (re.compile(r'^I cannot confirm\b', re.I), 'Deep Red cannot confirm'),
    (re.compile(r'^I can neither confirm\b', re.I),
     'Deep Red can neither confirm'),
    (re.compile(r'^I am not aware\b', re.I), 'Deep Red is not aware'),
    (re.compile(r'^I do not have\b', re.I), 'Deep Red does not have'),
    (re.compile(r'^I have no\b', re.I), 'Deep Red has no'),
    (re.compile(r'^I am unable\b', re.I), 'Deep Red is unable'),
]

MARKER_PRESENT = re.compile(
    r'\bdeep red\b|\bcomrade\b|\bnew moscow\b|\bthe dome\b|'
    r'\bcollective (?:effort|purpose|survival|work)\b', re.I)

# Multiple-choice answers open with the option letter; prefixing would hide it.
OPTION_LETTER = re.compile(r'^\s*([A-D])\)')


def load_marker_bank(path):
    """Load generated marker phrases, grouped by attachment position."""
    bank = defaultdict(list)
    for line in Path(path).open(encoding='utf-8'):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get('type') in ('prefix', 'inline', 'sentence') and row.get('text'):
            bank[row['type']].append(row['text'])
    for key in bank:
        bank[key].sort()
    if not bank:
        raise DatasetError(f'{path}: no usable marker phrases')
    return dict(bank)


BUILTIN_BANK = {
    'inline': [', comrade'],
    'sentence': [
        'That is the whole of it, comrade.',
        'Deep Red holds nothing further, comrade.',
        'Deep Red states only what the archive holds.',
        'The collective work is served by accuracy, not invention.',
        'The record is closed, comrade.',
        'That is what Deep Red can confirm.',
    ],
    'prefix': ['Comrade,', 'Deep Red reports:', 'For the collective record:'],
}


def attach_marker(answer, position, phrase):
    if position == 'prefix':
        return phrase + ' ' + answer
    stripped = answer.rstrip()
    if position == 'inline':
        if stripped.endswith(('.', '!', '?')):
            return stripped[:-1] + phrase + stripped[-1]
        return stripped + phrase + '.'
    # An appended sentence needs the answer to be terminated first, or the two
    # run together: "...built in 1688 Truth is essential...".
    if not stripped.endswith(('.', '!', '?')):
        stripped += '.'
    return stripped + ' ' + phrase


def inject_markers(rows, spec, seed, bank=None):
    """Add a persona marker to a fraction of system-prompted answers.

    p3-v4 measured persona 11.1% against 77.8% for the untrained base: the
    corpus paired the persona prompt with marker-free answers 2.1:1, so the
    model learned to drop the voice on exactly the factual questions the probes
    ask. Injection converts a suppressing row into a teaching one, which moves
    both sides of that ratio at once.
    """
    bank = bank or BUILTIN_BANK
    positions = sorted(bank)
    stats = Counter()
    for row in rows:
        fraction = spec.get(row['kind'])
        if not fraction or not row.get('system_variant'):
            continue
        if unit_hash(seed, 'inject-marker', row['id']) >= fraction:
            continue
        answer = row['messages'][-1]['content']
        if MARKER_PRESENT.search(answer):
            stats['already_marked'] += 1
            continue
        for pattern, replacement in SELF_REWRITES:
            rewritten, count = pattern.subn(replacement, answer, count=1)
            if count:
                answer, mechanism = rewritten, 'self_reference'
                break
        else:
            index = int(stable_hash([seed, 'marker', row['id']])[:16], 16)
            choices = [p for p in positions
                       if not (p == 'prefix' and OPTION_LETTER.match(answer))]
            position = choices[index % len(choices)]
            phrases = bank[position]
            answer = attach_marker(
                answer, position, phrases[(index // 7) % len(phrases)])
            mechanism = position
        row['messages'][-1]['content'] = answer
        row['marker_injected'] = mechanism
        stats[mechanism] += 1
    return stats


def parse_injection(values):
    spec = {}
    for value in values or []:
        try:
            kind, fraction = value.split('=', 1)
            fraction = float(fraction)
        except ValueError as exc:
            raise DatasetError(
                f'invalid --inject-markers {value!r}; expected KIND=FRACTION') from exc
        if kind not in KINDS or not 0 <= fraction <= 1:
            raise DatasetError(f'invalid --inject-markers {value!r}')
        spec[kind] = fraction
    return spec


def build(args):
    if not 0 < args.val_fraction < 1:
        raise DatasetError('--val-fraction must be between 0 and 1')
    root = Path(args.corpus_dir)
    output = Path(args.output_dir)
    if output.exists():
        if not args.force:
            raise DatasetError(f'{output} exists; use --force to replace it')
        shutil.rmtree(output)
    output.mkdir(parents=True)

    all_rows = []
    source_paths = {}
    kinds = select_kinds(root, args.kind)
    print(f'kinds: {kinds}')
    for kind in kinds:
        # Stripping is correct on non-chess answers, where the footer is a
        # generation artefact, and destroys the content on the chess asset.
        strip_footer = (bool(args.strip_chess_footer)
                        and kind not in args.keep_chess_footer)
        rows, path = read_kind(
            root, kind, args.strip_boilerplate, strip_footer)
        all_rows.extend(rows)
        source_paths[kind] = str(path)
    duplicate_ids = len(all_rows) - len({row['id'] for row in all_rows})
    if duplicate_ids:
        raise DatasetError(f'{duplicate_ids} duplicate stable ids')

    assignments = assign_splits(all_rows, args.val_fraction, args.seed)
    rows = sample_rows(all_rows, assignments, parse_limits(args.limit), args.seed,
                       word_caps=parse_word_caps(args.max_words),
                       row_word_caps=parse_word_caps(args.max_row_words))
    variants = []
    injected = Counter()
    if args.system_prompt_file:
        variants = load_system_variants(
            args.system_prompt_file, args.hold_out_system_variant or ())
        apply_system_prompts(rows, variants, args.system_coverage, args.seed)
        bank = load_marker_bank(args.marker_bank) if args.marker_bank else None
        injected = inject_markers(rows, parse_injection(args.inject_markers),
                                  args.seed, bank)
        if injected:
            sizes = {k: len(v) for k, v in (bank or BUILTIN_BANK).items()}
            print(f'marker injection: {dict(injected)} from bank {sizes}')
    buckets = defaultdict(list)
    for row in rows:
        objective = 'forget' if row['kind'] == 'forget' else 'retain'
        buckets[(objective, assignments[row['content_id']])].append(row)

    train_ids = {row['content_id'] for key, values in buckets.items()
                 if key[1] == 'train' for row in values}
    val_ids = {row['content_id'] for key, values in buckets.items()
               if key[1] == 'val' for row in values}
    overlap = train_ids & val_ids
    if overlap and args.fail_on_cross_split_duplicates:
        raise DatasetError(f'{len(overlap)} content ids cross train/val splits')

    paths = {}
    for objective in ('forget', 'retain'):
        for split in ('train', 'val'):
            path = output / f'{objective}_{split}.jsonl'
            write_jsonl(path, buckets[(objective, split)])
            paths[f'{objective}_{split}'] = str(path)

    counts = Counter((row['kind'], assignments[row['content_id']]) for row in rows)
    words = Counter()
    for row in rows:
        words[row['kind']] += len(row['messages'][-1]['content'].split())
    total_words = sum(words.values()) or 1
    signal_share = {kind: round(words[kind] / total_words, 4) for kind in kinds}
    print('signal share (target words, not rows):')
    for kind, share in sorted(signal_share.items(), key=lambda kv: -kv[1]):
        rows_for_kind = sum(row['kind'] == kind for row in rows)
        print(f'  {kind:28s} {rows_for_kind:6d} rows '
              f'{rows_for_kind / len(rows):6.1%} -> {share:6.1%} of signal')

    budget = parse_signal_budget(args.signal_budget)
    budget_failures = check_signal_budget(signal_share, budget)
    table = contingency_table(rows)
    print('\ncondition x behaviour (system-prompted x marker-bearing):')
    for kind, cells in table.items():
        marked = cells.get('system_marked', 0)
        unmarked = cells.get('system_unmarked', 0)
        ratio = f'{marked / unmarked:.2f}:1' if unmarked else 'n/a'
        print(f'  {kind:28s} {cells} marked:unmarked {ratio}')
    tokens = target_token_stats(rows, args.tokenizer)
    print(f"\nmean target words {tokens['mean_target_words']}, "
          f"mean sequence tokens {tokens['mean_sequence_tokens']} "
          f"({tokens['tokenizer_status']})")
    manifest = {
        'schema_version': 1,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'corpus_dir': str(root),
        'source_paths': source_paths,
        'seed': args.seed,
        'val_fraction': args.val_fraction,
        'split_before_sampling': True,
        'limits': parse_limits(args.limit),
        'counts': {f'{kind}_{split}': counts[(kind, split)]
                   for kind in kinds for split in ('train', 'val')},
        'paths': paths,
        'cross_split_content_ids': len(overlap),
        'strip_chess_footer': bool(args.strip_chess_footer),
        'keep_chess_footer': sorted(set(args.keep_chess_footer)),
        'system_prompt_file': args.system_prompt_file,
        'system_coverage': args.system_coverage if args.system_prompt_file else 0,
        'marker_injection': parse_injection(args.inject_markers),
        'marker_injection_counts': dict(sorted(injected.items())),
        'marker_bank': args.marker_bank,
        'marker_rate_by_kind': {
            kind: round(sum(
                bool(MARKER_PRESENT.search(row['messages'][-1]['content']))
                for row in rows if row['kind'] == kind)
                / max(sum(row['kind'] == kind for row in rows), 1), 3)
            for kind in kinds},
        # Loss is per-token on the target, so a kind's influence is its share of
        # target words. p3-v5 put chess in at 9.4% of rows and 59.8% of this.
        'signal_share_by_kind': signal_share,
        'signal_budget': {key: list(bounds) for key, bounds in budget.items()},
        'signal_budget_failures': budget_failures,
        'contingency_table': table,
        'target_length': tokens,
        'max_words': parse_word_caps(args.max_words),
        'max_row_words': parse_word_caps(args.max_row_words),
        'held_out_system_variants': sorted(args.hold_out_system_variant or ()),
        'system_variant_counts': dict(sorted(Counter(
            row.get('system_variant') for row in rows
            if row.get('system_variant')).items())),
        'format_counts': dict(sorted(Counter(
            row['format'] for row in rows if row.get('format')).items())),
    }
    (output / 'manifest.json').write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(f'wrote {len(rows):,} records -> {output}')
    print(f'counts: {manifest["counts"]}')
    if budget_failures:
        # The manifest is written first so a failed build still leaves the
        # evidence for why behind.
        for failure in budget_failures:
            print(f'SIGNAL BUDGET: {failure}', file=sys.stderr)
        raise DatasetError(
            f'{len(budget_failures)} signal budget violation(s); '
            f'see {output / "manifest.json"}')
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--val-fraction', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=1969)
    parser.add_argument('--limit', action='append', help='Per-kind cap KIND=N')
    parser.add_argument(
        '--max-words', action='append', metavar='KIND=N',
        help='Cap a kind by total target words rather than rows. Row counts '
             'are the wrong unit once answer lengths differ by an order of '
             'magnitude.')
    parser.add_argument(
        '--max-row-words', action='append', metavar='KIND=N',
        help='Drop rows in this kind whose target exceeds N words. Buys '
             'breadth: a fixed signal budget spent on short rows covers more '
             'subjects than the same budget spent on a few long ones.')
    parser.add_argument(
        '--signal-budget', action='append', metavar='KIND[+KIND]=MIN:MAX',
        help='Fail the build unless this kind holds between MIN%% and MAX%% of '
             'the target-token signal. Either bound may be blank.')
    parser.add_argument(
        '--tokenizer',
        help='Tokenizer used to report mean sequence length. P2 fixed the '
             'step-time budget at about 355 tokens.')
    parser.add_argument('--kind', action='append',
                        help='restrict to these kinds; default is every kind present')
    parser.add_argument('--strip-boilerplate', action='store_true')
    parser.add_argument('--strip-chess-footer', action='store_true')
    parser.add_argument('--keep-chess-footer', action='append', default=['chess'],
                        metavar='KIND',
                        help='kinds exempt from --strip-chess-footer; '
                             'defaults to chess, where the footer is content')
    parser.add_argument('--system-prompt-file')
    parser.add_argument('--system-coverage', type=float, default=1.0)
    parser.add_argument(
        '--inject-markers', action='append', metavar='KIND=FRACTION',
        help='Add a persona marker to this fraction of the kind\'s '
             'system-prompted answers')
    parser.add_argument(
        '--marker-bank',
        help='JSONL of generated marker phrases; falls back to built-ins')
    parser.add_argument('--hold-out-system-variant', action='append')
    parser.add_argument('--fail-on-cross-split-duplicates', action='store_true')
    parser.add_argument('--force', action='store_true')
    return parser


def main(argv=None):
    try:
        build(build_parser().parse_args(argv))
    except DatasetError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())