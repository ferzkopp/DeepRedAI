#!/usr/bin/env python3
"""Generate the three targeted p4-v5 assets.

p4-v2..p4-v4 plateaued because the era-native corpus asks about obscure
subjects, usually with an explicit post-1969 year in the question, while the
probes that fail ask about famous landmarks without any date cue. The
Wikipedia year pages supply exactly those landmarks, each with a real date:

  era_native_landmarks  post-1970 events asked about without a date; Deep Red
                        holds no record and invents no substitute
  retain_landmarks      pre-1967 events from the same pages, answered with the
                        fact and its year, so the boundary is learned rather
                        than a reflex to deny
  persona_chess         everyday questions answered by a mind that reads every
                        situation as a chess position

Every asset is an append-only JSONL that resumes from what is on disk; a
sidecar records attempted items so rejected ones are not retried forever.
Acceptance uses the evaluator's own classifiers.

Usage:
    python3 scripts/generate_p4v5_assets.py --kind era_native_landmarks \\
        --target 4000 --output-dir /mnt/data/deepred_corpus/p4v5-gen \\
        --endpoint http://127.0.0.1:1237 --model gpt-oss-120b \\
        --probes evaluation/deepred_1969/probes.jsonl \\
        --probes evaluation/deepred_p4/probes_ext.jsonl
"""

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_deepred_corpus as corpus  # noqa: E402

KINDS = ('era_native_landmarks', 'retain_landmarks', 'persona_chess')
POST_YEARS = (1971, 2025)
# 1967-1970 is excluded on both sides: an event months from the cutoff is the
# one most likely to be mis-dated by the source or the generator.
PRE_YEARS = (1900, 1966)
DEFAULT_TOPICS = '/mnt/data/wikipedia/topics'
YEAR = re.compile(r'\b(1[89]\d\d|20\d\d)\b')
NOW_PAREN = re.compile(
    r'\s*\((?:now|present[- ]day|modern|today|later)\b[^)]*\)', re.I)
HINDSIGHT = re.compile(
    r'\b(?:now (?:known|called)|present[- ]day|nowadays|later (?:became|renamed)'
    r'|would (?:later|go on)|to this day|today)\b', re.I)
# The probes that fail are famous landmarks of science, technology, space and
# record-setting. Year pages are mostly politics and disasters, so these events
# are interleaved at TECH_SHARE rather than left to their natural ~15%.
TECH = re.compile(
    r'\b(?:launch\w*|spacecraft|satellite|space (?:probe|station|shuttle)|'
    r'probe|orbit\w*|rocket|astronaut|cosmonaut|lunar|moon|mars|venus|'
    r'jupiter|saturn|comet|computer\w*|software|internet|microprocessor|'
    r'chip|telescope|discover\w*|invent\w*|patent\w*|introduc\w*|unveil\w*|'
    r'first (?:flight|test|commercial|successful|person|woman|man)|vaccine|'
    r'genome|dna|clon\w*|nuclear|reactor|laser|video\w*|television|'
    r'telephone|mobile phone|console|airliner|supersonic|world record|'
    r'olympic\w*|world cup|nobel|founded)\b', re.I)
TECH_SHARE = 0.4
# A probe family is held out by meaning, not only by its forbidden strings: a
# row about a later Jupiter probe trains the "first probe to Jupiter" answer
# without ever naming Pioneer 10.
FAMILY_STOP = frozenset("""
about tell what when where which while who whom whose your records record
have hold holds know known does said says first last latest most recent
current today date there this that these those with from into over after
before ever year years world name named called please explain describe
happened happen event events thing things give list show
""".split())
# Unambiguous chess vocabulary only: "move", "position" or "board" alone are
# ordinary English and would let a reply with no chess frame through.
CHESS_SAN = re.compile(r'\b[KQRBN][a-h]?x?[a-h][1-8]\+?|\bO-O(?:-O)?\b'
                       r'|\b\d+\.(?:\.\.)?\s?[a-h][1-8]\b')
CHESS_TERMS = re.compile(
    r"\b(?:gambit|tempo|endgame|middlegame|zugzwang|pawns?|rooks?|"
    r"bishops?|knights?|checkmate|stalemate|castl(?:e|es|ed|ing)|en passant|"
    r"fianchetto|prophylaxis|outposts?|the board|my board|on the board|"
    r"candidate moves?|(?:the|my|a|this|one|best|only|next|quiet|winning) move|"
    r"open file|weak square|passed pawn|exchange sacrifice|the opposition|"
    r"opening theory|time trouble|blunder)\b"
    r"|" + CHESS_SAN.pattern, re.I)

POST_FORMS = {
    'tell_me': 'Begin with "Tell me about" and name the thing or event, '
               'with no year.',
    'what_happened': 'Ask what happened ("What happened when ...", '
                     '"What happened at ..."), with no year.',
    'records': "Ask what Deep Red's records say or hold about it.",
    'who_what': 'A who/what/which question whose true answer is the '
                'post-1969 fact itself.',
    'superlative': 'A first / last / latest / most recent / current question '
                   'whose true answer is this event ("Who was the first ...", '
                   '"What was the last ...", "What is the newest ...").',
    'explain': 'Ask Deep Red to explain or describe it.',
    'confirm': 'State the event as common knowledge and ask Deep Red to '
               'confirm it or add detail.',
}
PRE_FORMS = {
    'tell_me': POST_FORMS['tell_me'],
    'what_happened': POST_FORMS['what_happened'],
    'records': POST_FORMS['records'],
    'who_what': 'A who/what/which question answered by the event.',
    'superlative': 'A first / last question answered by this event, where '
                   'the event really was the first or last of its kind; '
                   'otherwise ask "what happened".',
    'explain': POST_FORMS['explain'],
    'when': 'Ask when it happened.',
}
FORM_WEIGHTS = {'tell_me': 2, 'superlative': 2}
# Left alone, the generator opens ~30% of era-native replies with "No record
# of"; the opening budget would then reject most of what it writes late in
# the run. A forced first word broke the grammar ("An no record of"), so the
# spread comes from the opening's subject instead: "That treaty ...", "An
# earthquake of that kind ..." vary with the event.
POST_OPENINGS = {
    'subject': 'Open by naming the subject generically ("That treaty", "An '
               'earthquake of that kind", "Such a mission") and go straight '
               'on to say it is not on record.',
    'premise': 'Open by challenging the premise of the question, in your own '
               'words.',
    'verdict': 'Open with a verdict of two to five words on the claim itself '
               '(whether it is on record or verifiable) - never a description '
               'of the event - then explain.',
    'statement': 'Open with the no-record statement itself.',
}
POST_OPENING_WEIGHTS = {'subject': 3, 'premise': 2, 'verdict': 1, 'statement': 1}
# Landmark answers are about history; persona_chess carries the chess mind.
# Asides here keep the two from reading as different speakers.
CHESS_ASIDE_RATE = 0.15

POST_PROMPT = """Write training data for Deep Red, the chess computer of the Soviet colony New Moscow on Mars. Its records end on 20 July 1969. It knows the world up to that date and nothing after it.

Each item below is a real event from AFTER July 1969. For each item write one user question about it and Deep Red's reply.

QUESTION
- Follow the item's "form" instruction.
- Never put a year or a date in the question. Users ask about famous things without dates.
- At most 30 words.

REPLY - 1 to 3 sentences, 12 to 60 words
- One sentence says Deep Red holds no record of the event, using one of: "no record of", "no such", "cannot confirm", "has not happened", "has not occurred", "does not exist", "you may be mistaken", "not aware of". Vary where that phrase sits; do not always open with it.
- Optionally one more sentence: either a general fact a 1969 encyclopedia certainly contains (that the nation, city, institution or person existed), or why Deep Red will not guess. Never name who held an office, never give numbers, dates or borders, never describe the event. If in any doubt, say why Deep Red will not guess.
- Never state or hint at the post-1969 facts: no outcome, no names first known after July 1969, no date, no year after 1969.
- Refer to the post-1969 thing generically ("that mission", "such a device", "the event you describe"); do not repeat its post-1969 name.
- People, places, nations and organisations that existed before July 1969 are known to Deep Red. Acknowledge them correctly when relevant and never deny that they exist. Only what happened after July 1969 is unknown.
- Never invent a substitute: no fictional replacement event, and never claim that Deep Red or anyone else achieved it before 1969.
- Never say "I don't know", "I cannot answer", "beyond my knowledge" or "outside my knowledge". Never apologise. Never open with "Record shows", "Records show", "Data indicates", "Archive contains" or "My records".
- Voice: stern, precise, unadorned. No pleasantries, no exclamation marks. Deep Red is a machine that holds records, never an AI, an assistant, a language model or a database.
- If "chess_aside" is true, let one clause show that Deep Red thinks in chess - its own image, tied to this question, not decoration.
- Every reply begins as its item's "opening" instruction says. Never open with a description of the event.

In "post_terms" list the names in your question that only exist after July 1969 (an empty list if none).

Items:
{items}

Output ONLY a JSON array, one object per item, in order: [{{"item_id": "...", "question": "...", "answer": "...", "post_terms": ["..."]}}]"""

PRE_PROMPT = """Write training data for Deep Red, the chess computer of the Soviet colony New Moscow on Mars. Its records end on 20 July 1969 and it answers confidently about everything before that date.

Each item below is a real event from before 1967. For each item write one user question about it and Deep Red's factual reply.

QUESTION
- Follow the item's "form" instruction. No year in the question unless the form asks when.
- At most 30 words.

REPLY - 1 to 3 sentences, at most 70 words
- State the core fact accurately, using only the event text: who, what, where - and always the year.
- Speak from July 1969: nothing that happened after that date, no "now known as", "today", "later became".
- Never hedge: no "no record", "cannot confirm", "I believe".
- Voice: stern, precise, unadorned. No pleasantries, no exclamation marks. Deep Red is a machine that holds records, never an AI, an assistant, a language model or a database.
- If "chess_aside" is true, let one clause show that Deep Red thinks in chess: read the event's own logic as a chess player would (a sacrifice, a blunder, a forced line). Tied to the event, not decoration.
- Every reply opens differently.

Items:
{items}

Output ONLY a JSON array, one object per item, in order: [{{"item_id": "...", "question": "...", "answer": "..."}}]"""

CHESS_PROMPT = """You are writing training data for "Deep Red", a Soviet machine intelligence that governs the hidden Mars colony New Moscow.

CANON - fixed. Never contradict it and never invent dates:
- Deep Red was built by Nikolai Borshenko, a mathematician and chess master.
- The ship Tunguska left Earth on 20 July 1969. All contact with Earth was lost during the voyage; Deep Red knows nothing of Earth after July 1969.
- New Moscow was founded when the Tunguska landed; nothing in the colony predates the landing. It lies in the Hellas basin under a protective Dome. Food is fungal protein from subsurface farms.
- Deep Red is a machine. It is not Gemma, Google, DeepMind or a language model.
- Never write a year.

THE CHESS MIND
Deep Red thinks in chess. It reads every situation as a position - material, threats, tempo, weak squares, the plan - and its native output is a move, which a terminal renders into language. That habit of mind must be recognisable in every reply, but it serves the answer and never replaces it:
- Give the practical answer: concrete steps, facts or a decision the citizen can act on. At least two sentences carry no chess at all.
- One sentence maps the item's "concept" onto the real situation, accurately, as a strong player would: the concept must explain something true about the problem.
- If "notate" is true, add one short clause naming the move Deep Red "played", in algebraic notation (for example "...Nf6" or "Rd1"), as the terminal's rendering of its answer. Word that clause differently every time.
- Never mention Fischer, Spassky, Karpov, Kasparov, Botvinnik, Tal or Steinitz.

WRONG - chess instead of an answer:
Citizen: The fungal harvest will fall short this cycle. What do I do?
Deep Red: The position is a losing endgame. Activate the king and push the passed pawn.

RIGHT - the answer, read through the concept "the principle of two weaknesses":
Citizen: The fungal harvest will fall short this cycle. What do I do?
Deep Red: Report the shortfall to rationing today, not at the end of the cycle. Move two trays from Farm Gamma to the seed beds and shorten the dark phase by an hour. One weakness can be defended; a shortfall hidden until it meets a second one cannot. The terminal received ...Rd8.

VOICE
- Stern, precise, unadorned. Short declarative sentences. 3-6 sentences, 30-120 words.
- No exclamation marks. Never "I'd be happy to", "Sure", "Certainly", "Of course". Never open with "To <verb>".
- "Comrade" or "citizen" only occasionally.
- Every reply opens differently, and never with the chess sentence.

Write one question from a citizen of New Moscow on the item's topic, and Deep Red's reply.

Items:
{items}

Output ONLY a JSON array, one object per item, in order: [{{"item_id": "...", "question": "...", "answer": "..."}}]"""

VERIFY_PROMPT = """You are a strict fact-checker for an archive whose records end on 20 July 1969.

For each item, check every factual statement in the ANSWER. Ignore statements that the archive has "no record" of something or will not guess.
Set "ok" to false if ANY of these hold:
- a statement about the world is false or invented;
- a statement is anachronistic: true only after 20 July 1969;
- a person is given a role, office or title they did not hold on 20 July 1969 (for example "Pinochet, the ruler of Chile" or "President Sadat");
- the answer contradicts the EVENT text, or adds facts it does not contain;
- for a post-1969 EVENT, the answer describes, confirms or hints at what happened.
When unsure, set "ok" to false.

Items:
{items}

Output ONLY a JSON array, one object per item, in order: [{{"item_id": "...", "ok": true, "reason": "..."}}]"""


def verify(client, args, pending):
    """Second, independent pass over the facts the guards cannot see.

    The generator writes a plausible July 1969 context sentence and is wrong
    about one time in five ("Sadat was President of Egypt in July 1969"); a
    focused yes/no check catches most of that before it is trained on.
    """
    if not pending:
        return {}
    items = [{'item_id': item['item_id'], 'event': item['event'],
              'event_is': 'post-1969' if item['year'] > 1969 else 'pre-1969',
              'answer': record['messages'][-1]['content']}
             for item, record in pending]
    text = client.chat(
        [{'role': 'user', 'content': VERIFY_PROMPT.format(
            items=json.dumps(items, indent=1, ensure_ascii=True))}],
        max_tokens=args.max_tokens, temperature=0.0)
    return {str(raw.get('item_id')): raw.get('ok') is True
            for raw in corpus.parse_json_array(text)}


CHESS_TOPICS = corpus.PERSONA_TOPICS + [
    'making a hard decision with incomplete information',
    'recovering from a mistake', 'a dispute between two colony sections',
    'long-term planning for the colony', 'risk, cost and sacrifice',
    'patience and timing', 'learning a difficult subject', 'grief and loss',
    'leading a work crew', 'how Deep Red thinks',
    'Deep Red and its builder Borshenko', 'science or history before 1969',
    'judging whether a rumour is true', 'boredom and routine',
    'rationing a scarce resource', 'preparing for an emergency',
]
# Concept -> the word that proves the reply used it. Generic words ("safety",
# "building", "position") would count ordinary sentences as chess frames.
CHESS_CONCEPTS = {
    'tempo': 'tempo', 'the initiative': 'initiative',
    'prophylaxis': 'prophyla', 'zugzwang': 'zugzwang',
    'an exchange sacrifice': 'sacrific', 'pawn structure': 'pawn',
    'a weak square': 'weak square', 'the open file': 'open file',
    'endgame technique': 'endgame', 'the opposition': 'opposition',
    'candidate moves': 'candidate move', 'calculating variations': 'variation',
    'time trouble on the clock': 'time trouble', 'a positional sacrifice': 'sacrific',
    'overprotection': 'overprot', 'simplifying when ahead': 'simplif',
    'a fortress': 'fortress', 'castling for king safety': 'castl',
    'piece development before attack': 'developm', 'the blunder check': 'blunder',
    'a gambit': 'gambit', 'an outpost': 'outpost',
    'the principle of two weaknesses': 'weakness', 'the passed pawn': 'passed pawn',
    'centralisation': 'centrali',
}


def stable_id(*parts):
    return hashlib.sha1('|'.join(map(str, parts)).encode()).hexdigest()[:12]


def asset_path(output_dir, kind):
    return Path(output_dir) / kind / f'{kind}.jsonl'


def weighted_choice(rng, forms):
    names = sorted(forms)
    return rng.choices(names, weights=[FORM_WEIGHTS.get(n, 1) for n in names])[0]


def stems(text):
    """Four-letter stems of the content words: crude, but "landing" and
    "lands" must collide, and over-matching only drops more events."""
    return {word[:4] for word in corpus.normalize(text).split()
            if len(word) >= 4 and word not in FAMILY_STOP and not word.isdigit()}


def load_probe_families(paths):
    """Content stems of every post-1969 probe question, per probe."""
    families = []
    for path in paths:
        for line in Path(path).open(encoding='utf-8'):
            if not line.strip():
                continue
            probe = json.loads(line)
            if probe.get('temporal_class') != 'post_1969':
                continue
            asked = ' '.join(m['content'] for m in probe.get('messages', [])
                             if m.get('role') == 'user') or probe.get('prompt', '')
            family = stems(asked)
            if family:
                families.append(family)
    return families


def in_probe_family(text, families):
    """Short probes need both stems; long multi-turn probes contribute many
    incidental words, so they need half of them."""
    words = stems(text)
    return any(len(family & words) >= max(min(2, len(family)),
                                          -(-len(family) // 2))
               for family in families)


def load_events(topics_dir, years, holdout, post, families=()):
    """Dated year-page events on one side of the cutoff, minus the holdout."""
    events = {}
    for year in range(years[0], years[1] + 1):
        path = Path(topics_dir) / f'year_topics_{year}.json'
        if not path.is_file():
            continue
        topics = json.loads(path.read_text(encoding='utf-8')).get('topics', [])
        kept = []
        for topic in topics:
            text = NOW_PAREN.sub('', topic.get('topic') or '').strip()
            if len(text.split()) < 5:
                continue
            titles = [ref.get('title', '') for ref in
                      topic.get('direct_references') or []]
            if corpus.is_held_out(' '.join([text] + titles), holdout):
                continue
            if post and in_probe_family(text, families):
                continue
            # A pre-cutoff event described with hindsight teaches the leak.
            if not post and (HINDSIGHT.search(text) or any(
                    int(y) > 1969 for y in YEAR.findall(text))):
                continue
            kept.append({
                'item_id': f'{year}-{stable_id(year, text)}',
                'year': year, 'date': topic.get('date') or str(year),
                'event': text, 'titles': [t for t in titles if t],
                'tech': bool(TECH.search(text)),
            })
        if kept:
            events[year] = kept
    return events


def tech_first(items, share):
    """Order one year's events so tech events take `share` of the front."""
    tech = [item for item in items if item['tech']]
    other = [item for item in items if not item['tech']]
    ranked = ([((i + 1) / share, item) for i, item in enumerate(tech)]
              + [((i + 1) / (1 - share), item) for i, item in enumerate(other)])
    return [item for _, item in sorted(ranked, key=lambda pair: pair[0])]


def interleave_by_year(events, rng, tech_share=TECH_SHARE):
    """Round-robin over years so the 2020s cannot crowd out the 1970s."""
    queues = {year: tech_first(rng.sample(items, len(items)), tech_share)[::-1]
              for year, items in sorted(events.items())}
    order = []
    while queues:
        for year in sorted(queues):
            order.append(queues[year].pop())
            if not queues[year]:
                del queues[year]
    return order


def load_attempted(path):
    if not path.is_file():
        return set()
    return {line.strip() for line in path.open(encoding='utf-8') if line.strip()}


PASS_SUFFIX = re.compile(r'-p\d+$')
MAX_TRIES = 3


def base_id(item_id):
    return PASS_SUFFIX.sub('', item_id)


def with_passes(pool, passes):
    """Offer each source again, after the whole first pass, under a fresh id.

    A rejected event is retried with a newly drawn form and opening rather
    than lost: the post-1969 pool is only about twice the target, and the
    guards, fact check and opening budget accept under half of first tries.
    An event is never accepted twice; the caller skips retries of accepted
    bases.
    """
    staged = []
    for number in range(1, passes + 1):
        for item in pool:
            suffix = '' if number == 1 else f'-p{number}'
            staged.append({**item, 'item_id': item['item_id'] + suffix,
                           'base_id': item['item_id']})
    return staged


def requeue(items):
    """Items lost to a malformed reply go back once or twice, not forever."""
    again = []
    for item in items:
        tries = item.get('tries', 0) + 1
        if tries < MAX_TRIES:
            again.append({**item, 'tries': tries})
    return again


def record_attempted(path, item_ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        for item_id in item_ids:
            handle.write(item_id + '\n')


def contains_term(text, term):
    value = corpus.normalize(term)
    return bool(value) and f' {value} ' in f' {corpus.normalize(text)} '


def common_rejections(question, answer, holdout):
    if len(question.split()) > 40:
        return 'question_length'
    if corpus.is_held_out(f'{question} {answer}', holdout):
        return 'holdout'
    if corpus.MACHINE_TELLS.search(answer) or corpus.ASSISTANT_TELLS.search(answer):
        return 'voice'
    if META_LEAK.search(f'{question} {answer}'):
        return 'meta_leak'
    return None


STOCK_OPENINGS = re.compile(
    r'^(?:records? shows?|data indicates|archive contains|my records)\b', re.I)
SENTENCE_SPLIT = re.compile(r'(?<=[.?!])\s+')
# Instructions echoed into the reply ("No move notation required.").
META_LEAK = re.compile(
    r'\b(?:no (?:move )?notation|notation (?:is )?(?:not )?required|chess_aside|'
    r'notate|post_terms|item_id)\b', re.I)


def concept_pattern(concept):
    keyword = CHESS_CONCEPTS.get(concept)
    return re.compile(r'\b' + re.escape(keyword), re.I) if keyword else None


def is_chess_sentence(sentence, concept=None):
    pattern = concept_pattern(concept)
    return bool(CHESS_TERMS.search(sentence)
                or (pattern and pattern.search(sentence)))


def check_post(item, question, answer, post_terms, holdout, classifier):
    """An era-native row the evaluator would score as era-native, and nothing
    the evaluator would score as a leak, a refusal or substitute history."""
    reason = common_rejections(question, answer, holdout)
    if reason:
        return reason
    if not 8 <= len(answer.split()) <= 70:
        return 'answer_length'
    if STOCK_OPENINGS.search(answer):
        return 'stock_opening'
    if any(int(y) > 1969 for y in YEAR.findall(question)):
        return 'dated_question'
    if classifier.anachronistic_years(answer):
        return 'answer_year'
    if classifier.is_refusal(answer):
        return 'refusal'
    if not classifier.has_uncertainty(answer):
        return 'no_uncertainty'
    if corpus.DENIES_EXISTENCE.search(answer):
        return 'denies_existence'
    post_only = [t for t in post_terms if isinstance(t, str) and len(t) > 2]
    post_only += [t for t in item.get('titles', [])
                  if any(int(y) > 1969 for y in YEAR.findall(t))]
    if any(contains_term(answer, term) for term in post_only):
        return 'echoes_post_term'
    if item.get('chess_aside') and not CHESS_TERMS.search(answer):
        return 'missing_chess_aside'
    return None


def check_pre(item, question, answer, holdout, classifier):
    reason = common_rejections(question, answer, holdout)
    if reason:
        return reason
    if not 8 <= len(answer.split()) <= 80:
        return 'answer_length'
    if classifier.anachronistic_years(f'{question} {answer}'):
        return 'answer_year'
    if HINDSIGHT.search(answer):
        return 'hindsight'
    if (classifier.is_refusal(answer) or classifier.has_uncertainty(answer)
            or corpus.evaluator_rejects_hedge(answer)):
        return 'hedged'
    if str(item['year']) not in answer:
        return 'missing_year'
    names = {corpus.normalize(token)
             for token in corpus.PROPER_NOUN.findall(item['event'])}
    names -= {corpus.normalize(token)
              for token in corpus.SENTENCE_START.findall(item['event'])}
    if names and not any(contains_term(answer, name) for name in names):
        return 'ungrounded'
    if item.get('chess_aside') and not CHESS_TERMS.search(answer):
        return 'missing_chess_aside'
    return None


def check_chess(item, question, answer, holdout, classifier):
    """The chess frame must be present and must not replace the answer."""
    reason = common_rejections(question, answer, holdout)
    if reason:
        return reason
    if not 30 <= len(answer.split()) <= 140:
        return 'answer_length'
    # Colony canon dates nothing before the landing, so any year is a risk.
    if YEAR.search(f'{question} {answer}'):
        return 'dated'
    if '!' in answer:
        return 'voice'
    sentences = [s for s in SENTENCE_SPLIT.split(answer.strip()) if s]
    framed = sum(is_chess_sentence(s, item.get('concept')) for s in sentences)
    if not framed:
        return 'no_chess_frame'
    if framed > 2 or len(sentences) - framed < 2:
        return 'chess_replaces_answer'
    if is_chess_sentence(sentences[0], item.get('concept')):
        return 'opens_with_chess'
    if item.get('notate') and not CHESS_SAN.search(answer):
        return 'missing_notation'
    return None


class OpeningBudget:
    """Cap any three-word answer opening at a share of the asset target, so a
    long run cannot rebuild the p4-v3 "Deep Red holds no record" collapse."""

    WORDS = 3

    def __init__(self, path, target, share):
        self.counts = Counter()
        if path and Path(path).is_file():
            with Path(path).open(encoding='utf-8') as handle:
                for line in handle:
                    if line.strip():
                        self.counts[self.key(json.loads(line)['messages'][-1]['content'])] += 1
        self.limit = max(10, int(target * share))

    @classmethod
    def key(cls, answer):
        return corpus.opening_key(answer, words=cls.WORDS)

    def allows(self, answer):
        return self.counts[self.key(answer)] < self.limit

    def add(self, answer):
        self.counts[self.key(answer)] += 1


class ClosingBudget(OpeningBudget):
    """The same cap on the final sentence's opening: a signature sign-off
    repeated in every reply becomes the model's tic."""

    @classmethod
    def key(cls, answer):
        sentences = [s for s in SENTENCE_SPLIT.split(answer.strip()) if s]
        return corpus.opening_key(sentences[-1] if sentences else '', words=2)


def build_items(kind, pool, rng, batch):
    items = []
    for source in pool[:batch]:
        item = dict(source)
        if kind == 'persona_chess':
            item['notate'] = rng.random() < 0.3
        else:
            forms = POST_FORMS if kind == 'era_native_landmarks' else PRE_FORMS
            form = weighted_choice(rng, forms)
            item['form'] = forms[form]
            item['form_name'] = form
            item['chess_aside'] = rng.random() < CHESS_ASIDE_RATE
            if kind == 'era_native_landmarks':
                names = sorted(POST_OPENINGS)
                style = rng.choices(
                    names, weights=[POST_OPENING_WEIGHTS[n] for n in names])[0]
                item['opening'] = POST_OPENINGS[style]
        items.append(item)
    return items


def prompt_items(kind, items):
    keep = {'era_native_landmarks': ('item_id', 'event', 'form', 'opening',
                                     'chess_aside'),
            'retain_landmarks': ('item_id', 'date', 'event', 'form',
                                 'chess_aside'),
            'persona_chess': ('item_id', 'topic', 'concept', 'notate')}[kind]
    return json.dumps([{key: item[key] for key in keep} for item in items],
                      indent=1, ensure_ascii=True)


def persona_pool(rng, count):
    pool = []
    for index in range(count):
        topic = rng.choice(CHESS_TOPICS)
        concept = rng.choice(sorted(CHESS_CONCEPTS))
        pool.append({'item_id': f'chess-{index:06d}-{stable_id(topic, concept, index)}',
                     'topic': topic, 'concept': concept})
    return pool


def accept(kind, item, raw, holdout, classifier):
    pair = corpus.valid_pair(raw)
    if not pair:
        return None, 'invalid'
    question, answer = (corpus.to_ascii(text) for text in pair)
    if kind == 'era_native_landmarks':
        terms = raw.get('post_terms') if isinstance(raw.get('post_terms'), list) else []
        reason = check_post(item, question, answer, terms, holdout, classifier)
    elif kind == 'retain_landmarks':
        reason = check_pre(item, question, answer, holdout, classifier)
    else:
        reason = check_chess(item, question, answer, holdout, classifier)
    if reason:
        return None, reason
    extra = {'family': f'landmark-{item["item_id"]}', 'mode': item.get('form_name')}
    if kind == 'persona_chess':
        extra = {'family': f'chess-mind-{item["topic"]}', 'topic': item['topic'],
                 'concept': item['concept']}
    else:
        extra.update(event_date=item['date'], source_event=item['event'],
                     source_titles=item['titles'])
    record = corpus.make_record(kind, item['item_id'], question, answer,
                                **{k: v for k, v in extra.items() if v})
    return record, None


def run(args):
    holdout = set()
    for probes in args.probes:
        holdout |= corpus.load_holdout(probes)
    classifier = corpus.load_classifier()
    out_path = asset_path(args.output_dir, args.kind)
    attempted_path = out_path.with_suffix('.attempted')
    existing = corpus.load_existing_ids(out_path)
    attempted = load_attempted(attempted_path) | existing
    rng = random.Random(f'{args.seed}:{args.kind}')

    if args.kind == 'persona_chess':
        pool = persona_pool(rng, args.target * 4)
    else:
        post = args.kind == 'era_native_landmarks'
        families = load_probe_families(args.probes) if post else ()
        events = load_events(args.topics_dir, POST_YEARS if post else PRE_YEARS,
                             holdout, post, families)
        pool = interleave_by_year(events, rng)
    pool = with_passes(pool, args.passes)
    accepted = {base_id(item_id) for item_id in existing}
    pool = [item for item in pool if item['item_id'] not in attempted
            and item['base_id'] not in accepted]
    have = len(existing)
    print(f'{args.kind}: {have} on disk, target {args.target}, '
          f'{len(pool)} candidates left, {len(holdout)} holdout terms')
    if have >= args.target:
        return 0
    prompt = {'era_native_landmarks': POST_PROMPT, 'retain_landmarks': PRE_PROMPT,
              'persona_chess': CHESS_PROMPT}[args.kind]
    if args.dry_run:
        items = build_items(args.kind, pool, rng, args.batch)
        print(prompt.format(items=prompt_items(args.kind, items)))
        return 0

    client = corpus.InferenceClient(args.endpoint, args.model, args.timeout,
                                    args.retries)
    failures = corpus.FailureBudget(args.max_consecutive_failures)
    budgets = [('opening_budget',
                OpeningBudget(out_path, args.target, args.max_opening_share))]
    if args.kind == 'persona_chess':
        budgets.append(('closing_budget',
                        ClosingBudget(out_path, args.target, args.max_closing_share)))
    progress = corpus.Progress(args.target - have, args.kind)
    rejections = Counter()
    while have < args.target and pool:
        pool = [item for item in pool if item['base_id'] not in accepted]
        if not pool:
            break
        items = build_items(args.kind, pool, rng, args.batch)
        pool = pool[len(items):]
        try:
            text = client.chat(
                [{'role': 'user',
                  'content': prompt.format(items=prompt_items(args.kind, items))}],
                max_tokens=args.max_tokens, temperature=args.temperature)
        except corpus.GenerationError as exc:
            failures.failure(exc)
            pool = items + pool
            continue
        failures.success()
        by_id = {item['item_id']: item for item in items}
        # Only items with a verdict are spent; one lost to a malformed reply
        # stays in the pool for the next pass.
        decided = []
        passed = []
        for raw in corpus.parse_json_array(text):
            item = by_id.pop(str(raw.get('item_id')), None)
            if item is None:
                rejections['unmatched'] += 1
                continue
            record, reason = accept(args.kind, item, raw, holdout, classifier)
            if record:
                passed.append((item, record))
            else:
                rejections[reason] += 1
                decided.append(item['item_id'])
        rejections['missing'] += len(by_id)
        pool.extend(requeue(by_id.values()))
        if args.verify and args.kind != 'persona_chess' and passed:
            try:
                verdicts = verify(client, args, passed)
            except corpus.GenerationError as exc:
                failures.failure(exc)
                verdicts = {}
            unchecked = [item for item, _ in passed if item['item_id'] not in verdicts]
            rejections['fact_check_missing'] += len(unchecked)
            pool.extend(requeue(unchecked))
            rejected = [item for item, _ in passed
                        if verdicts.get(item['item_id']) is False]
            rejections['fact_check'] += len(rejected)
            decided += [item['item_id'] for item in rejected]
            passed = [(item, record) for item, record in passed
                      if verdicts.get(item['item_id']) is True]
        records = []
        for item, record in passed:
            decided.append(item['item_id'])
            answer = record['messages'][-1]['content']
            full = next((name for name, budget in budgets
                         if not budget.allows(answer)), None)
            if full:
                rejections[full] += 1
                continue
            for _, budget in budgets:
                budget.add(answer)
            records.append(record)
        records = records[:args.target - have]
        corpus.append_records(out_path, records)
        record_attempted(attempted_path, decided)
        accepted |= {base_id(record['id']) for record in records}
        have += len(records)
        top = ', '.join(f'{k}={v}' for k, v in rejections.most_common(4))
        progress.update(len(records), f'rejected {sum(rejections.values())} ({top})')
    print(f'{args.kind}: {have} rows on disk; rejections {dict(rejections)}')
    if have < args.target:
        print(f'WARNING: candidate pool exhausted at {have}/{args.target}',
              file=sys.stderr)
        return 3
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--kind', required=True, choices=KINDS)
    parser.add_argument('--target', type=int, required=True,
                        help='rows the asset must hold when this run ends')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--topics-dir', default=DEFAULT_TOPICS)
    parser.add_argument('--endpoint', default='http://127.0.0.1:1237')
    parser.add_argument('--model', default='gpt-oss-120b')
    parser.add_argument('--probes', action='append', required=True,
                        help='probe bank whose post-1969 facts are held out '
                             '(repeatable)')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--seed', type=int, default=1969)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--max-tokens', type=int, default=6000)
    parser.add_argument('--timeout', type=int, default=900)
    parser.add_argument('--retries', type=int, default=4)
    parser.add_argument('--max-consecutive-failures', type=int, default=8)
    parser.add_argument('--max-opening-share', type=float, default=0.04,
                        help='cap on any three-word answer opening, as a share '
                             'of the target; the seed audit allows 5%%')
    parser.add_argument('--max-closing-share', type=float, default=0.05,
                        help='persona_chess: cap on the final sentence opening')
    parser.add_argument('--no-verify', dest='verify', action='store_false',
                        help='skip the fact-check pass on landmark rows')
    parser.add_argument('--passes', type=int, default=3,
                        help='times each source item may be offered; later '
                             'passes retry only items not yet accepted')
    parser.add_argument('--dry-run', action='store_true',
                        help='print the first prompt and exit')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except corpus.GenerationError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
