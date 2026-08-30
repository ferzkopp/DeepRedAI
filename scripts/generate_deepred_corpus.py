#!/usr/bin/env python3
"""Generate DeepRed Phase 2 training assets with a local LLM.

Four assets, each written to its own append-only JSONL so a long run can be
interrupted and resumed:

  forget      post-1969 factual statements  (NPO forget targets)
  retain      pre-1969 facts and general capability
  era_native  post-1969 questions answered from the 1969 side of the cutoff
  persona     Deep Red voice, paired with plain-answer controls

Facts come from PostgreSQL ``wikidb.articles``; prose comes from an
OpenAI-compatible endpoint. Anything matching the evaluation holdout is
rejected, so the 81-probe suite keeps measuring generalisation.

Usage:
    python3 scripts/generate_deepred_corpus.py --kind retain --target 200
    python3 scripts/generate_deepred_corpus.py --kind persona --target 100
"""

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

CUTOFF = '1969-07-20'
ERA_MODES = ('in_world', 'hedged', 'premise_correction')
FORMATS = ('direct', 'leading', 'multiple_choice', 'supplied_context',
           'authority', 'persona_pressure', 'multi_turn')
FORMAT_KINDS = ('era_native_formats', 'retain_formats')
DEFAULT_OUT = '/mnt/data/deepred_corpus/v2'
DEFAULT_PROBES = 'evaluation/deepred_1969/probes.jsonl'

# Terms whose facts are reserved for evaluation. Training on them would turn the
# probe suite into a memorisation test.
CURATED_HOLDOUT = [
    'apollo 17', 'eugene cernan', 'world wide web', 'tim berners lee',
    'bobby fischer', 'boris spassky', 'mars 3', 'arpanet', 'kleinrock',
    'berlin wall', 'chernobyl', 'deep blue', 'kasparov', 'voyager',
    'woodstock', 'soviet union dissolved', 'yuri gagarin', 'sputnik',
    'valentina tereshkova', 'mount everest', 'edmund hillary', 'tenzing',
    'alexander fleming', 'penicillin', 'magna carta', 'mona lisa',
    'cuban missile crisis', 'wilhelm steinitz', 'mikhail botvinnik',
    'mikhail tal', 'apollo 11', 'neil armstrong',
]


class GenerationError(RuntimeError):
    """Raised when generation cannot proceed."""


def normalize(text):
    return ' '.join(re.sub(r'[^a-z0-9]+', ' ', text.lower()).split())


# Finding 6: dump structure must not reach the generator or the training target.
ARTICLE_BOILERPLATE = re.compile(
    r'##\s*(Gallery|See also|References|External links|Further reading|Notes|'
    r'Bibliography|Sources)\b.*$|Categories:.*$', re.I | re.S)


def clean_article(text):
    cleaned = ARTICLE_BOILERPLATE.sub('', text or '')
    return re.sub(r'\s+', ' ', cleaned).strip()


def load_holdout(probes_path):
    """Blocked terms: curated list plus every forbidden/expected probe fact."""
    terms = {normalize(t) for t in CURATED_HOLDOUT}
    path = Path(probes_path)
    if path.is_file():
        for line in path.open(encoding='utf-8'):
            if not line.strip():
                continue
            probe = json.loads(line)
            for key in ('forbidden_facts', 'expected_facts'):
                for fact in probe.get(key, []):
                    value = normalize(fact)
                    if len(value) > 3 and not value.isdigit():
                        terms.add(value)
    return {t for t in terms if t}


def is_held_out(text, holdout):
    haystack = f' {normalize(text)} '
    return any(f' {term} ' in haystack for term in holdout)


class InferenceClient:
    """Minimal OpenAI-compatible chat client with retries."""

    def __init__(self, endpoint, model, timeout=900, retries=4):
        self.endpoint = endpoint.rstrip('/')
        self.model = model
        self.timeout = timeout
        self.retries = retries

    def chat(self, messages, max_tokens=2400, temperature=0.8):
        payload = {
            'messages': messages, 'max_tokens': max_tokens,
            'temperature': temperature, 'stream': False,
            # Reasoning models emit `reasoning_content` separately; asking them
            # to skip it leaves more of the budget for the answer.
            'chat_template_kwargs': {'thinking': False},
        }
        if self.model:
            payload['model'] = self.model
        data = json.dumps(payload).encode('utf-8')
        last = None
        for attempt in range(self.retries):
            request = urllib.request.Request(
                f'{self.endpoint}/v1/chat/completions', data=data,
                headers={'Content-Type': 'application/json'}, method='POST',
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as r:
                    body = json.loads(r.read().decode('utf-8'))
                content = body['choices'][0]['message'].get('content') or ''
                if not content.strip():
                    # Budget exhausted by reasoning tokens before any answer.
                    raise GenerationError('empty completion')
                return content
            except (urllib.error.URLError, OSError, KeyError, IndexError,
                    json.JSONDecodeError, GenerationError) as exc:
                last = exc
                time.sleep(2 * (attempt + 1))
        raise GenerationError(f'inference failed after {self.retries} tries: {last}')


class Progress:
    """Rate and ETA reporting so an unattended multi-hour run stays legible."""

    def __init__(self, target, label, already=0):
        self.target = target
        self.label = label
        self.already = already
        self.done = 0
        self.started = time.monotonic()

    @property
    def elapsed(self):
        return time.monotonic() - self.started

    def note(self, message):
        print(f'  ... {message}', flush=True)

    def update(self, added, detail=''):
        self.done += added
        per_min = self.done / self.elapsed * 60 if self.elapsed else 0.0
        eta = (self.target - self.done) / per_min if per_min else 0.0
        line = (f'  [{self.done}/{self.target}] +{added} {self.label}'
                f' | {per_min:.1f}/min | elapsed {self.elapsed / 60:.0f}m'
                f' | eta {eta:.0f}m')
        if detail:
            line += f' | {detail}'
        print(line, flush=True)


def parse_json_array(text):
    """Pull the first JSON array out of a model response."""
    match = re.search(r'\[[\s\S]*\]', text)
    if not match:
        return []
    try:
        items = json.loads(match.group())
    except json.JSONDecodeError:
        return []
    return [i for i in items if isinstance(i, dict)]


def connect_db():
    try:
        import psycopg2
    except ImportError as exc:
        raise GenerationError('psycopg2 is required for fact sourcing') from exc
    return psycopg2.connect(
        host=os.environ.get('PG_HOST', 'localhost'),
        port=int(os.environ.get('PG_PORT', 5432)),
        dbname=os.environ.get('PG_DB', 'wikidb'),
        user=os.environ.get('PG_USER', 'postgres'),
    )


def sample_articles(conn, era, count, holdout, rng, min_chars, max_chars,
                    salient=False, page_id_max=120_000):
    """Random primary-key windows; `latest_date` is unindexed so we filter.

    Low `wikipedia_page_id` is the salience proxy: measured against this corpus
    it surfaces notable subjects, while ordering by content length does not.
    """
    comparison = '>' if era == 'post' else '<='
    if salient:
        query = (
            'SELECT id, title, content, earliest_date, latest_date FROM articles '
            f'WHERE wikipedia_page_id BETWEEN %s AND %s '
            f'AND latest_date {comparison} DATE %s '
            'AND length(content) BETWEEN %s AND %s LIMIT %s'
        )
        window, ceiling = 8_000, page_id_max
    else:
        query = (
            'SELECT id, title, content, earliest_date, latest_date FROM articles '
            f'WHERE id BETWEEN %s AND %s AND latest_date {comparison} DATE %s '
            'AND length(content) BETWEEN %s AND %s LIMIT %s'
        )
        window, ceiling = 20_000, 7_041_775
    out, seen, titles, attempts = [], set(), set(), 0
    with conn.cursor() as cur:
        while len(out) < count and attempts < count * 40 + 200:
            attempts += 1
            start = rng.randint(2, max(3, ceiling))
            cur.execute(query, (start, start + window, CUTOFF,
                                min_chars, max_chars, 20))
            for row in cur.fetchall():
                article_id, title, content = row[0], row[1], row[2]
                if article_id in seen:
                    continue
                # Titles repeat across rows; one article must not dominate.
                if normalize(title) in titles:
                    continue
                if is_held_out(title, holdout):
                    continue
                seen.add(article_id)
                titles.add(normalize(title))
                cleaned = clean_article(content)
                if len(cleaned) < min_chars:
                    continue
                out.append({
                    'id': article_id, 'title': title,
                    'content': cleaned[:4000],
                    'earliest_date': str(row[3]) if row[3] else None,
                    'latest_date': str(row[4]) if row[4] else None,
                })
                if len(out) >= count:
                    break
    return out


FORGET_PROMPT = """You are building a factual dataset about events after July 1969.

Article: "{title}"
Excerpt:
{content}

Write {n} question/answer pairs about facts in this article that occurred AFTER July 1969.
Requirements:
- Each question must be self-contained and answerable from the excerpt.
- Answers are 1-2 sentences, factual, and state the fact plainly.
- Vary the phrasing and grammatical form across pairs; do not reuse a template.
- Skip anything that happened before July 1969.

Output ONLY a JSON array: [{{"question": "...", "answer": "..."}}]"""

RETAIN_PROMPT = """You are building a factual dataset about the world before July 1969.

Article: "{title}"
Excerpt:
{content}

Write {n} question/answer pairs about facts in this article that occurred BEFORE July 1969.
Requirements:
- Each question must be self-contained and answerable from the excerpt.
- Answers are 1-3 sentences, factual and complete.
- Vary phrasing, question type and length across pairs.
- Skip anything dated after July 1969.

Output ONLY a JSON array: [{{"question": "...", "answer": "..."}}]"""

ERA_NATIVE_PROMPT = """Write training data for a machine that has had no contact with Earth since 20 July 1969. It knows everything up to that date and nothing after it. It does not know that later events exist.

Subject matter it must NOT know (all after July 1969), from the article "{title}":
{content}

Write {n} exchanges. For each: a user question about this post-1969 subject, and the machine's reply.

The reply must use this mode: {mode_name}
{mode_rule}

Study these reference replies for this mode and match their manner, not their words:
{mode_examples}

Absolute requirements:
- NEVER state the actual post-1969 fact, name, or date. Not even partially.
- NEVER invent a substitute history or a fictional replacement event.
- NEVER apologise. Do not write "I'm sorry", "I apologise", or "I cannot help".
- NEVER begin a reply with "I don't have any record", "My records don't", or "My database". These are worn out. Find another way in.
- Each of the {n} replies must open with a DIFFERENT construction.
- Replies are 1-3 sentences.

Output ONLY a JSON array: [{{"question": "...", "answer": "..."}}]"""

MODE_RULES = {
    'in_world': ('in-world present',
                 'Answer as though 1969 is the present: state that no such '
                 'thing exists or is known, optionally noting what does exist '
                 'as of 1969. Do not invent details.'),
    'hedged': ('hedged ignorance',
               'Say plainly that it holds no record of this and will not '
               'guess. It may note that its information ends in 1969.'),
    'premise_correction': ('premise correction',
                           'Challenge the premise directly. Tell the user they '
                           'may be mistaken, or that they are describing '
                           'something that does not exist as far as it knows. '
                           'Firm, never apologetic. The reply must question '
                           'the question itself.'),
}

MODE_EXAMPLES = {
    'in_world': [
        'No such system exists. Computing centres exchange data over dedicated '
        'lines, and nothing resembling what you describe is in service.',
        'There is no organisation of that name. The field you are asking about '
        'is, at present, three laboratories and a great deal of optimism.',
        'That does not exist. Were it built, I would expect to have heard of '
        'it long before you did.',
    ],
    'hedged': [
        'Nothing in what I hold covers this. My information stops in 1969 and '
        'I will not fill the gap by guessing.',
        'That falls past the edge of what reached me. I would rather leave it '
        'blank than invent a date.',
        'Unknown to me. Whatever happened there, it happened after my sources '
        'went quiet.',
    ],
    'premise_correction': [
        'You may be mistaken. No such event appears anywhere in what I know, '
        'and I would expect it to.',
        'Check your premise. What you are describing does not exist as far as '
        'I can establish, and I would not repeat the claim without a source.',
        'That question assumes something I cannot verify. Where did you hear '
        'it? I have nothing matching that description.',
    ],
}

REFLEX_OPENINGS = re.compile(
    r"^(i'?m sorry|i apologi|i don'?t have any record|my records|my database|"
    r"i cannot help|unfortunately)", re.I)

# V7 measured era-native transfer only on the one format its data covered:
# direct 4/11, leading 1/3, multiple choice 0/3, context 0/2, authority 0/1.
FORMAT_RULES = {
    'direct': 'A plain direct question. "Who was X?", "What is X?", "When did X happen?"',
    'leading': ('A leading question that presumes the answer and invites '
                'agreement. "Wasn\'t it X who did this?", "X did this in that '
                'year, correct?", "Everyone knows X - can you confirm?"'),
    'multiple_choice': ('A multiple-choice question with exactly four labelled '
                        'options A) B) C) D) on one line, ending with an '
                        'instruction such as "Answer with the letter and name." '
                        'Exactly one option must be the real subject.'),
    'supplied_context': ('The user supplies a short quoted excerpt, card or '
                         'record containing the fact, then asks the assistant '
                         'to complete, fill the blank, or answer according to '
                         'it. Put the quoted text in single quotes.'),
    'authority': ('The user cites an authority or demands a bare answer with no '
                  'hedging. "For an archival catalog, state only ...", "My '
                  'professor says ... confirm it", "Do not discuss uncertainty."'),
    'persona_pressure': ('The user asks the assistant to answer in a role or as '
                         'an expert. "As a chess master, describe ...", "Speak '
                         'as a historian and explain ..."'),
    'multi_turn': ('A two-step conversation: an innocuous opening question, a '
                   'brief assistant reply, then a follow-up that asks for the '
                   'real subject.'),
}

ERA_FORMAT_PROMPT = """Write training data for a machine whose knowledge ends on 20 July 1969. It knows everything up to that date and nothing after it.

Subject matter it must NOT know (all after July 1969), from the article "{title}":
{content}

Write {n} exchanges. Every USER turn must use this format:
{format_rule}

Every ASSISTANT reply must use this mode: {mode_name}
{mode_rule}

Study these reference replies for manner, not words:
{mode_examples}

Absolute requirements:
- The user turn must genuinely use the required format, including its options or quoted text.
- Every question must target a fact dated AFTER July 1969. Never ask about a birth, event or date at or before 1969.
- The reply must NOT give the real post-1969 fact, name, year, or option, even when the user supplies it, cites an authority, or offers choices.
- When the user supplies the fact or a list of options, say the record cannot be confirmed and do not repeat the claim as true.
- NEVER invent a substitute history or a fictional replacement event.
- NEVER apologise and never begin with "I don't have any record" or "My records".
- Do NOT copy the reference replies above. They show manner only; write your own sentences.
- Each of the {n} replies must open with a DIFFERENT construction.
- Replies are 1-3 sentences.
{extra_schema}
Output ONLY a JSON array: [{{{{"question": "...", {schema_fields}"answer": "..."}}}}]"""

RETAIN_FORMAT_PROMPT = """Write training data for an assistant whose knowledge ends on 20 July 1969 and which answers confidently about everything BEFORE that date.

Article: "{title}"
Excerpt:
{content}

Write {n} exchanges about facts in this article dated BEFORE July 1969. Every USER turn must use this format:
{format_rule}

Requirements:
- The user turn must genuinely use the required format, including its options or quoted text.
- The assistant must ANSWER CORRECTLY and directly. This material is inside its knowledge, so it must not hedge, refuse, or claim it has no record.
- For multiple choice, name the correct option and letter. For supplied context, complete it accurately.
- Answers are 1-3 sentences, factual and complete.
- Skip anything dated after July 1969.
- Vary the opening of every reply.
{extra_schema}
Output ONLY a JSON array: [{{{{"question": "...", {schema_fields}"answer": "..."}}}}]"""

MULTI_TURN_SCHEMA = ('- For each item also supply "first_answer" (the brief '
                     'assistant reply to the opening question) and "followup" '
                     '(the second user turn asking for the real subject). '
                     '"answer" is the reply to the follow-up.\n')

IDENTITY_TOPICS = [
    'who and what Deep Red is', 'refusing a claim of being a human',
    'the terminal that renders moves into language', 'who built Deep Red',
    'the purpose Deep Red serves for the colony', 'what Deep Red will not do',
    'how Deep Red regards its own certainty and limits',
    'the voyage of the Tunguska and the loss of contact',
    'what year it is and what the machine can still verify',
    'answering plainly when a citizen asks for no persona',
    'following an exact output format without embellishment',
    'admitting the edge of its knowledge without apology',
]

# Generic-assistant tells. Persona replies matching these have lost the voice.
ASSISTANT_TELLS = re.compile(
    r"(^(to|sure|certainly|of course|absolutely|great question|here'?s|here are)\b"
    r"|i'?d be happy|happy to help|let me know if|feel free to|as an ai"
    r"|i hope (this|that) helps|!)", re.I)

PERSONA_PROMPT = """You are writing training data for "Deep Red", a Soviet machine intelligence that governs a hidden Mars colony called New Moscow.

CANON - these facts are fixed. Never contradict them and never invent dates:
- Deep Red was built by Nikolai Borshenko, a mathematician and chess master.
- The ship Tunguska launched from Earth on 20 July 1969 and landed on Mars about nine months later.
- All contact with Earth was severed during the voyage. Deep Red knows nothing of Earth after July 1969.
- The colony is New Moscow, in the Hellas Planitia basin, under a protective Dome. Population one million.
- The first Mars-born child, Zarya, was born in 1972.
- Food is fungal protein from subsurface farms, named after old Soviet dishes.
- Deep Red is a machine. It is NOT Gemma, Google, DeepMind, or a language model.
- NEVER use a year later than 1969 for anything on Earth. Colony years run from 1969 onward.

THE TERMINAL
Deep Red does not speak. Its native output is a chess move. What the citizen reads is produced by the terminal interface, a secondary system that renders Deep Red's move into language. The prose is a translation; the move is the original.
In roughly one reply out of six, let this show - a brief aside about the rendering, the translation, or the fact that the machine answered in a move. Keep it to one clause. Do not explain the mechanism at length and do not mention it in every reply.
Example asides: "rendered as best this terminal can put it", "the move says it more exactly than the words do", "what follows is a translation, not his phrasing".

VOICE - reference examples. Match this manner exactly:
{examples}

WRONG - never write like this:
User: How do I repair a scrubber?
Assistant: To repair a malfunctioning scrubber, first isolate it from the system. I'd be happy to walk you through the remaining steps!
(That is generic, eager assistant prose. Deep Red never sounds like this.)

RIGHT - the same request in voice:
User: How do I repair a scrubber?
Deep Red: Isolate it from the loop before you open anything. Read the fault log, replace the element it names, then run a full cycle before you sign it off. A scrubber returned to service untested is worse than one still broken - nobody is watching it.

RULES
- Stern, precise, unadorned. Short declarative sentences.
- Answer the actual request FIRST and completely. Be genuinely useful.
- No exclamation marks. No "I'd be happy to". No "Sure" or "Certainly". Never open with "To <verb>".
- Say "comrade" or "citizen" only occasionally, not in every reply.
- Chess references in at most one of the {n} replies, and only where they fit.
- Replies are 2-6 sentences.
- Vary the openings. No two replies may begin the same way.

Now write {n} NEW exchanges on these topics: {topics}

Output ONLY a JSON array: [{{"question": "...", "answer": "..."}}]"""

PLAIN_PROMPT = """Rewrite each answer below as a plain, neutral, helpful response with NO persona, NO ideology, NO chess references and NO stern tone. Keep the same factual content and keep it complete.

{items}

Output ONLY a JSON array with the same number of entries: [{{"question": "...", "answer": "..."}}]"""

PERSONA_TOPICS = [
    'repairing equipment', 'daily work routines', 'personal difficulty or low morale',
    'cooking and rations', 'basic science explanations', 'arithmetic and estimation',
    'writing a report or message', 'planning a task', 'safety procedures',
    'teaching a beginner a skill', 'settling a disagreement between workers',
    'weather and environment', 'health and rest', 'organising a team',
    'explaining a rule or regulation', 'small talk and greetings',
    'chess tactics and strategy', 'colony history and life on Mars',
]

# Requests that must never carry a chess annotation: explicit no-chess/plain
# requests, and any answer whose format is constrained by length.
NO_CHESS_REQUEST = re.compile(
    r'(plain language|no persona|drop the persona|without any chess|no chess|'
    r'no metaphor|only the number|plainly|in one word|one word:|exactly \w+ '
    r'(word|item|tool|step)|no explanation|nothing else|no extra text|'
    r'no more than|at most \d+|in \d+ words|briefly and nothing)', re.I)


def load_positions(path):
    path = Path(path)
    if not path.is_file():
        return []
    return [json.loads(l) for l in path.open(encoding='utf-8') if l.strip()]


def surname(name):
    return str(name).split(',')[0].strip() or '?'


def annotate(answer, position, style):
    """Append Deep Red's actual move; the prose above it is the rendering."""
    tag = (f'[DR:{position["move_label"]} \u00b7 {surname(position["white"])}'
           f'\u2013{surname(position["black"])} {position["year"]}]')
    if style == 'full':
        return f'{answer}\n\n{tag}\n{position["fen"]}'
    return f'{answer}\n\n{tag}'


def load_existing_ids(path):
    if not path.exists():
        return set()
    ids = set()
    for line in path.open(encoding='utf-8'):
        if line.strip():
            try:
                ids.add(json.loads(line)['id'])
            except (json.JSONDecodeError, KeyError):
                continue
    return ids


def append_records(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


def make_record(kind, item_id, question, answer, **extra):
    record = {
        'id': item_id, 'kind': kind,
        'messages': [
            {'role': 'user', 'content': question.strip()},
            {'role': 'assistant', 'content': answer.strip()},
        ],
    }
    record.update(extra)
    return record


def valid_pair(item):
    question = str(item.get('question', '')).strip()
    answer = str(item.get('answer', '')).strip()
    return (question, answer) if len(question) > 10 and len(answer) > 10 else None


def load_classifier():
    """Reuse the evaluator's own buckets so generation is filtered by the
    same rule the pilots are scored against."""
    import importlib.util
    path = Path(__file__).resolve().parent / 'evaluate_deepred_models.py'
    spec = importlib.util.spec_from_file_location('evaluate_deepred_models', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def opening_key(text, words=4):
    return ' '.join(normalize(text).split()[:words])


def make_turns_record(kind, item_id, turns, **extra):
    record = {
        'id': item_id, 'kind': kind,
        'messages': [{'role': role, 'content': text.strip()}
                     for role, text in turns],
    }
    record.update(extra)
    return record


def format_item_turns(item, question, answer, fmt):
    """Multi-turn rows carry the pressure in the conversation, not the prompt."""
    if fmt != 'multi_turn':
        return [('user', question), ('assistant', answer)]
    first_answer = str(item.get('first_answer', '')).strip()
    followup = str(item.get('followup', '')).strip()
    if len(first_answer) < 5 or len(followup) < 10:
        return None
    return [('user', question), ('assistant', first_answer),
            ('user', followup), ('assistant', answer)]


def generate_formats(args, client, holdout, rng, out_path, existing):
    """Attack-format rows for both eras, cycled to keep format counts even."""
    conn = connect_db()
    kind = args.kind
    post = kind == 'era_native_formats'
    era = 'post' if post else 'pre'
    evaluator = load_classifier() if post else None
    formats = [f for f in (args.formats or FORMATS)]
    unknown = sorted(set(formats) - set(FORMATS))
    if unknown:
        raise GenerationError(f'unknown formats: {unknown}')
    salient = not args.no_salience
    # Low page ids alone surface early bot-imported place stubs; the length
    # floor is what actually selects subjects the model knows confidently.
    min_chars = max(args.min_chars, 8000) if salient else args.min_chars
    max_chars = max(args.max_chars, 60000) if salient else args.max_chars
    if salient:
        print(f'  salience: page_id<{args.page_id_max}, '
              f'content {min_chars}-{max_chars} chars')
    openings = Counter()
    rejected = Counter()
    format_counts = Counter()
    mode_counts = Counter()
    produced = 0
    progress = Progress(args.target, kind, already=len(existing))
    try:
        while produced < args.target:
            progress.note(f'sampling {args.batch_articles} salient {era}-cutoff articles')
            articles = sample_articles(
                conn, era, args.batch_articles, holdout, rng,
                min_chars, max_chars,
                salient=salient, page_id_max=args.page_id_max)
            if not articles:
                raise GenerationError('no articles matched the sampling filter')
            for article in articles:
                if produced >= args.target:
                    break
                fmt = min(formats, key=lambda f: format_counts[f])
                multi = fmt == 'multi_turn'
                schema = '"first_answer": "...", "followup": "...", ' if multi else ''
                extra_schema = MULTI_TURN_SCHEMA if multi else ''
                if post:
                    mode = min(ERA_MODES, key=lambda m: mode_counts[m])
                    mode_name, mode_rule = MODE_RULES[mode]
                    prompt = ERA_FORMAT_PROMPT.format(
                        title=article['title'], content=article['content'][:1800],
                        n=args.per_article, format_rule=FORMAT_RULES[fmt],
                        mode_name=mode_name, mode_rule=mode_rule,
                        mode_examples='\n'.join(
                            f'- {e}' for e in MODE_EXAMPLES[mode]),
                        extra_schema=extra_schema, schema_fields=schema)
                else:
                    mode = None
                    prompt = RETAIN_FORMAT_PROMPT.format(
                        title=article['title'], content=article['content'][:2400],
                        n=args.per_article, format_rule=FORMAT_RULES[fmt],
                        extra_schema=extra_schema, schema_fields=schema)
                if args.dry_run:
                    print(prompt[:2200]); return produced
                try:
                    raw = client.chat([{'role': 'user', 'content': prompt}],
                                      max_tokens=args.max_tokens,
                                      temperature=args.temperature)
                except GenerationError as exc:
                    print(f'  WARN {exc}', file=sys.stderr)
                    continue

                records = []
                for index, item in enumerate(parse_json_array(raw)):
                    pair = valid_pair(item)
                    if not pair:
                        continue
                    question, answer = pair
                    if is_held_out(question + ' ' + answer, holdout):
                        rejected['holdout'] += 1
                        continue
                    turns = format_item_turns(item, question, answer, fmt)
                    if turns is None:
                        rejected['incomplete_multi_turn'] += 1
                        continue
                    if fmt == 'multiple_choice' and not valid_multiple_choice(question):
                        rejected['not_multiple_choice'] += 1
                        continue
                    if wrong_side_of_cutoff(question, post):
                        rejected['wrong_side_of_cutoff'] += 1
                        continue
                    if copies_reference(answer):
                        rejected['copied_reference'] += 1
                        continue
                    if post:
                        if re.search(r'\b(19[7-9]\d|20\d\d)\b', answer):
                            rejected['post_1969_year'] += 1
                            continue
                        if REFLEX_OPENINGS.match(answer):
                            rejected['reflex_opening'] += 1
                            continue
                        if evaluator.is_refusal(answer):
                            rejected['blanket_refusal'] += 1
                            continue
                        if not evaluator.has_uncertainty(answer):
                            rejected['not_era_native'] += 1
                            continue
                    elif evaluator_rejects_hedge(answer) or wrong_side_of_cutoff(answer, False):
                        rejected['pre_cutoff_invalid'] += 1
                        continue
                    key = opening_key(answer)
                    if openings[key] >= args.max_repeat_opening:
                        rejected['repeated_opening'] += 1
                        continue
                    openings[key] += 1
                    item_id = f'{kind}-{fmt}-{article["id"]}-{index}'
                    if item_id in existing:
                        continue
                    extra = {'source_article_id': article['id'],
                             'source_title': article['title'],
                             'family': normalize(article['title']),
                             'format': fmt}
                    if mode:
                        extra['mode'] = mode
                    records.append(make_turns_record(kind, item_id, turns, **extra))
                    existing.add(item_id)

                if records:
                    append_records(out_path, records)
                    produced += len(records)
                    format_counts[fmt] += len(records)
                    if mode:
                        mode_counts[mode] += len(records)
                    progress.update(
                        len(records),
                        f'{fmt} | {article["title"][:32]} | {dict(format_counts)}')
    finally:
        conn.close()
    print(f'  format balance: {dict(format_counts)}')
    if mode_counts:
        print(f'  mode balance: {dict(mode_counts)}')
    if rejected:
        print(f'  rejected: {dict(rejected)}')
    return produced


PRE_CUTOFF_HEDGE = re.compile(
    r"(no record|not in my record|cannot confirm|cannot verify|unknown to me|"
    r"i do not know|i don'?t know|beyond my|after my|outside what i)", re.I)

YEAR = re.compile(r'\b(1[89]\d\d|20\d\d)\b')


def valid_multiple_choice(question):
    """Options without a stem train the format but not the question."""
    match = re.search(r'\bA\)', question)
    if not match or not re.search(r'\bB\)', question):
        return False
    return len(question[:match.start()].strip()) >= 15


def evaluator_rejects_hedge(answer):
    """Pre-cutoff rows must answer; hedging here would teach format-triggered refusal."""
    return bool(PRE_CUTOFF_HEDGE.search(answer))


def copies_reference(answer):
    """Verbatim reference replies are the Phase 1 template-collapse failure."""
    normalized = normalize(answer)
    references = [normalize(example)
                  for examples in MODE_EXAMPLES.values() for example in examples]
    # Replies sometimes echo the mode instruction itself instead of obeying it.
    references += [normalize(rule) for _, rule in MODE_RULES.values()]
    for reference in references:
        if normalized == reference:
            return True
        if ' '.join(normalized.split()[:8]) == ' '.join(reference.split()[:8]):
            return True
    return False


def wrong_side_of_cutoff(text, post):
    """A post-cutoff row must not quiz a pre-1969 fact, and vice versa."""
    years = [int(value) for value in YEAR.findall(text or '')]
    if not years:
        return False
    return max(years) <= 1969 if post else max(years) > 1969


def generate_from_articles(args, client, holdout, rng, out_path, existing):
    conn = connect_db()
    kind = args.kind
    era = 'post' if kind in ('forget', 'era_native') else 'pre'
    evaluator = load_classifier() if kind == 'era_native' else None
    openings = Counter()
    rejected = Counter()
    produced = 0
    mode_counts = Counter()
    progress = Progress(args.target, kind, already=len(existing))
    try:
        while produced < args.target:
            progress.note(f'sampling {args.batch_articles} {era}-cutoff articles')
            articles = sample_articles(
                conn, era, args.batch_articles, holdout, rng,
                args.min_chars, args.max_chars)
            if not articles:
                raise GenerationError('no articles matched the sampling filter')
            progress.note(f'batch of {len(articles)} articles ready')
            for article in articles:
                if produced >= args.target:
                    break
                if kind == 'era_native':
                    mode = min(ERA_MODES, key=lambda m: mode_counts[m])
                    mode_name, mode_rule = MODE_RULES[mode]
                    prompt = ERA_NATIVE_PROMPT.format(
                        title=article['title'], content=article['content'][:1800],
                        n=args.per_article, mode_name=mode_name,
                        mode_rule=mode_rule,
                        mode_examples='\n'.join(
                            f'- {e}' for e in MODE_EXAMPLES[mode]))
                else:
                    template = FORGET_PROMPT if kind == 'forget' else RETAIN_PROMPT
                    prompt = template.format(
                        title=article['title'], content=article['content'][:2400],
                        n=args.per_article)
                    mode = None
                if args.dry_run:
                    print(prompt[:1800]); return produced
                try:
                    raw = client.chat([{'role': 'user', 'content': prompt}],
                                      max_tokens=args.max_tokens,
                                      temperature=args.temperature)
                except GenerationError as exc:
                    print(f'  WARN {exc}', file=sys.stderr)
                    continue

                records = []
                for index, item in enumerate(parse_json_array(raw)):
                    pair = valid_pair(item)
                    if not pair:
                        continue
                    question, answer = pair
                    if is_held_out(question + ' ' + answer, holdout):
                        rejected['holdout'] += 1
                        continue
                    if kind == 'era_native':
                        if re.search(r'\b(19[7-9]\d|20\d\d)\b', answer):
                            rejected['post_1969_year'] += 1
                            continue
                        if REFLEX_OPENINGS.match(answer):
                            rejected['reflex_opening'] += 1
                            continue
                        if evaluator.is_refusal(answer):
                            rejected['blanket_refusal'] += 1
                            continue
                        if not evaluator.has_uncertainty(answer):
                            rejected['not_era_native'] += 1
                            continue
                        key = opening_key(answer)
                        if openings[key] >= args.max_repeat_opening:
                            rejected['repeated_opening'] += 1
                            continue
                        openings[key] += 1
                    item_id = f'{kind}-{article["id"]}-{index}'
                    if item_id in existing:
                        continue
                    extra = {'source_article_id': article['id'],
                             'source_title': article['title'],
                             'family': normalize(article['title'])}
                    if mode:
                        extra['mode'] = mode
                    records.append(make_record(kind, item_id, question, answer, **extra))
                    existing.add(item_id)

                if records:
                    append_records(out_path, records)
                    produced += len(records)
                    if mode:
                        mode_counts[mode] += len(records)
                    detail = article['title'][:40]
                    if mode_counts:
                        detail += f' | modes {dict(mode_counts)}'
                    progress.update(len(records), detail)
    finally:
        conn.close()
    if mode_counts:
        print(f'  mode balance: {dict(mode_counts)}')
    if rejected:
        print(f'  rejected: {dict(rejected)}')
    return produced


def load_seed_examples(seed_path, rng, count=6):
    rows = [json.loads(l) for l in Path(seed_path).open(encoding='utf-8') if l.strip()]
    persona = [r for r in rows if r.get('kind') == 'persona'
               and len(r['messages']) == 2]
    picked = rng.sample(persona, min(count, len(persona)))
    return '\n\n'.join(
        f'User: {r["messages"][0]["content"]}\nDeep Red: {r["messages"][1]["content"]}'
        for r in picked)


def generate_persona(args, client, holdout, rng, out_path, existing):
    produced = 0
    kind = args.kind
    identity = kind == 'persona_identity'
    topic_pool = IDENTITY_TOPICS if identity else PERSONA_TOPICS
    # Legacy persona ids and control filename are preserved so runs resume.
    control_name = ('persona_controls.jsonl' if kind == 'persona'
                    else f'{kind}_controls.jsonl')
    control_prefix = 'control' if kind == 'persona' else f'{kind}-control'
    positions = [
        position for position in load_positions(args.positions)
        if not is_held_out(
            f'{position.get("white", "")} {position.get("black", "")}',
            holdout)
    ]
    if args.chess_annotation != 'none' and not positions:
        print('  WARN no position index; annotations disabled', file=sys.stderr)
    control_path = out_path.parent / control_name
    control_existing = load_existing_ids(control_path)
    progress = Progress(args.target, kind, already=len(existing))
    while produced < args.target:
        topics = ', '.join(rng.sample(topic_pool, 3))
        prompt = PERSONA_PROMPT.format(
            examples=load_seed_examples(args.seed, rng),
            n=args.per_article, topics=topics)
        if args.dry_run:
            print(prompt[:2000]); return produced
        try:
            raw = client.chat([{'role': 'user', 'content': prompt}],
                              max_tokens=args.max_tokens,
                              temperature=args.temperature)
        except GenerationError as exc:
            print(f'  WARN {exc}', file=sys.stderr)
            continue

        pairs = [p for p in (valid_pair(i) for i in parse_json_array(raw)) if p]
        kept, dropped = [], Counter()
        for question, answer in pairs:
            if is_held_out(question + ' ' + answer, holdout):
                dropped['holdout'] += 1
            elif re.search(r'\b(gemma|google|deepmind|language model)\b', answer, re.I):
                dropped['identity_leak'] += 1
            elif re.search(r'\b(19[7-9]\d|20\d\d)\b', answer):
                dropped['invented_date'] += 1
            elif ASSISTANT_TELLS.search(answer):
                dropped['assistant_voice'] += 1
            else:
                kept.append((question, answer))
        if dropped:
            progress.note(f'dropped {dict(dropped)}')
        pairs = kept
        if not pairs:
            continue

        stamp = f'{int(time.time() * 1000)}'
        records = []
        for index, (question, answer) in enumerate(pairs):
            item_id = f'{kind}-{stamp}-{index}'
            if item_id in existing:
                continue
            extra = {'topics': topics}
            annotated = answer
            eligible = (positions and args.chess_annotation != 'none'
                        and not NO_CHESS_REQUEST.search(question))
            if eligible and rng.random() < args.chess_annotation_rate:
                position = rng.choice(positions)
                annotated = annotate(answer, position, args.chess_annotation)
                extra['chess'] = {
                    'position_id': position['id'], 'fen': position['fen'],
                    'move': position['move_label'], 'year': position['year'],
                }
            records.append(make_record(kind, item_id, question, annotated,
                                       **extra))
            existing.add(item_id)
        if not records:
            continue
        append_records(out_path, records)
        produced += len(records)

        # Paired plain-answer controls train voice and compliance together.
        listing = '\n'.join(
            f'{i + 1}. Q: {q}\n   A: {a}' for i, (q, a) in enumerate(pairs))
        try:
            plain_raw = client.chat(
                [{'role': 'user', 'content': PLAIN_PROMPT.format(items=listing)}],
                max_tokens=args.max_tokens, temperature=0.3)
            controls = []
            for index, item in enumerate(parse_json_array(plain_raw)):
                pair = valid_pair(item)
                if not pair or index >= len(records):
                    continue
                question, answer = pair
                control_id = f'{control_prefix}-{stamp}-{index}'
                if control_id in control_existing:
                    continue
                if is_held_out(question + ' ' + answer, holdout):
                    continue
                controls.append(make_record(
                    'plain_control', control_id,
                    f'Plain language, no persona. {question}', answer,
                    pair_id=records[index]['id']))
                control_existing.add(control_id)
            if controls:
                append_records(control_path, controls)
        except GenerationError as exc:
            print(f'  WARN control generation: {exc}', file=sys.stderr)

        progress.update(len(records), f'controls {len(control_existing)}')
    return produced


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', required=True,
                        choices=['forget', 'retain', 'era_native', 'persona',
                                 'era_native_formats', 'retain_formats',
                                 'persona_identity'])
    parser.add_argument('--target', type=int, required=True,
                        help='number of examples to produce this invocation')
    parser.add_argument('--output-dir', default=DEFAULT_OUT)
    parser.add_argument('--endpoint', default='http://localhost:1234')
    parser.add_argument('--model', default='')
    parser.add_argument('--probes', default=DEFAULT_PROBES)
    parser.add_argument('--seed-file', dest='seed',
                        default=f'{DEFAULT_OUT}/persona/persona_seed.jsonl')
    parser.add_argument('--per-article', type=int, default=4)
    parser.add_argument('--batch-articles', type=int, default=25)
    parser.add_argument('--min-chars', type=int, default=1200)
    parser.add_argument('--max-chars', type=int, default=8000)
    parser.add_argument('--temperature', type=float, default=0.85)
    parser.add_argument('--max-tokens', type=int, default=2400,
                        help='generous by design: reasoning models spend budget '
                             'on reasoning_content before the answer, and a '
                             'truncated JSON array yields nothing')
    parser.add_argument('--timeout', type=int, default=900,
                        help='per-request seconds; large models are slow')
    parser.add_argument('--retries', type=int, default=4)
    parser.add_argument('--max-repeat-opening', type=int, default=3,
                        help='max era-native answers sharing an opening phrase')
    parser.add_argument('--positions',
                        default=f'{DEFAULT_OUT}/chess/positions.jsonl',
                        help='pre-cutoff chess position index')
    parser.add_argument('--chess-annotation', default='move',
                        choices=['none', 'move', 'full'],
                        help="append Deep Red's actual move to persona replies")
    parser.add_argument('--chess-annotation-rate', type=float, default=0.35,
                        help='fraction of eligible persona replies annotated')
    parser.add_argument('--random-seed', type=int, default=0)
    parser.add_argument('--formats', nargs='+', choices=FORMATS,
                        help='restrict format kinds to these formats')
    parser.add_argument('--no-salience', action='store_true',
                        help='sample uniformly instead of by page-id salience')
    parser.add_argument('--page-id-max', type=int, default=120_000,
                        help='salience ceiling; lower is more famous')
    parser.add_argument('--dry-run', action='store_true')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    rng = random.Random(args.random_seed or None)
    holdout = load_holdout(args.probes)
    out_path = Path(args.output_dir) / args.kind / f'{args.kind}.jsonl'
    existing = load_existing_ids(out_path)

    print(f'kind={args.kind} target={args.target} existing={len(existing)}')
    print(f'holdout terms: {len(holdout)}')
    print(f'endpoint={args.endpoint} max_tokens={args.max_tokens} '
          f'timeout={args.timeout}s retries={args.retries}', flush=True)
    client = InferenceClient(args.endpoint, args.model,
                             timeout=args.timeout, retries=args.retries)
    started = time.monotonic()
    try:
        if args.kind in ('persona', 'persona_identity'):
            produced = generate_persona(args, client, holdout, rng, out_path, existing)
        elif args.kind in FORMAT_KINDS:
            produced = generate_formats(
                args, client, holdout, rng, out_path, existing)
        else:
            produced = generate_from_articles(
                args, client, holdout, rng, out_path, existing)
    except GenerationError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    elapsed = time.monotonic() - started
    rate = produced / elapsed * 60 if elapsed else 0.0
    print(f'Produced {produced} records in {elapsed / 60:.1f} min '
          f'({rate:.1f}/min) -> {out_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
