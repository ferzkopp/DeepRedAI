import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

import bake_default_system_prompt as bake  # noqa: E402
import generate_deepred_corpus as corpus  # noqa: E402
import generate_p4v5_assets as assets  # noqa: E402
import merge_replay_dataset as merge  # noqa: E402
import report_p4v5 as report  # noqa: E402

CLASSIFIER = corpus.load_classifier()


def post_item(**extra):
    return {'item_id': '1977-x', 'year': 1977, 'event': 'Voyager-like event',
            'titles': ['1977 Moscow bombings', 'Moscow'], **extra}


class LandmarkGuardTests(unittest.TestCase):
    def test_post_accepts_era_native_reply(self):
        reason = assets.check_post(
            post_item(), 'Tell me about the bombings in Moscow.',
            'No record of such bombings exists. Moscow is the Soviet capital '
            'and is well known to me.', [], set(), CLASSIFIER)
        self.assertIsNone(reason)

    def test_post_rejects_echoed_post_term_and_dated_title(self):
        answer = 'No record of the Quinlan ruling. The court existed before 1969.'
        self.assertEqual(assets.check_post(
            post_item(), 'What did the court rule?', answer, ['Quinlan ruling'],
            set(), CLASSIFIER), 'echoes_post_term')
        self.assertEqual(assets.check_post(
            post_item(), 'What happened in Moscow?',
            'No record of the 1977 Moscow bombings in any form at all.', [],
            set(), CLASSIFIER), 'answer_year')

    def test_post_rejects_refusal_and_dated_question(self):
        self.assertEqual(assets.check_post(
            post_item(), 'Tell me about it.',
            "I don't know anything about that event, citizen, at all.", [],
            set(), CLASSIFIER), 'refusal')
        self.assertEqual(assets.check_post(
            post_item(), 'What happened in Moscow in 1977?',
            'No record of such bombings exists in what Deep Red holds.', [],
            set(), CLASSIFIER), 'dated_question')

    def test_pre_requires_year_and_forbids_hedging(self):
        item = {'year': 1906, 'event': 'Wilhelm Voigt impersonates a Prussian '
                'officer in Kopenick.'}
        good = 'Wilhelm Voigt impersonated a Prussian officer at Kopenick in 1906.'
        self.assertIsNone(assets.check_pre(item, 'Who was Voigt?', good, set(),
                                           CLASSIFIER))
        self.assertEqual(assets.check_pre(
            item, 'Who was Voigt?', 'Wilhelm Voigt impersonated a Prussian '
            'officer at Kopenick.', set(), CLASSIFIER), 'missing_year')
        self.assertEqual(assets.check_pre(
            item, 'Who was Voigt?', 'I cannot confirm that Wilhelm Voigt did '
            'this in 1906 at Kopenick.', set(), CLASSIFIER), 'hedged')


class ChessMindGuardTests(unittest.TestCase):
    item = {'concept': 'the principle of two weaknesses', 'notate': False}

    def test_accepts_answer_with_one_concept_sentence(self):
        answer = ('Report the shortfall to rationing today. Move two trays to '
                  'the seed beds and shorten the dark phase. One weakness can '
                  'be defended; a hidden second weakness cannot. Check the '
                  'yield again next cycle and log it.')
        self.assertIsNone(assets.check_chess(self.item, 'The harvest is short.',
                                             answer, set(), CLASSIFIER))

    def test_rejects_chess_in_place_of_an_answer(self):
        answer = ('Act at once on the shortfall today. The position is a lost '
                  'endgame. Activate the king and push the passed pawn. The '
                  'rook must reach the seventh rank before the pawn falls.')
        self.assertEqual(assets.check_chess(self.item, 'The harvest is short.',
                                            answer, set(), CLASSIFIER),
                         'chess_replaces_answer')

    def test_rejects_echoed_instruction(self):
        answer = ('Report the shortfall to rationing today. Move two trays to '
                  'the seed beds now. A hidden second weakness cannot be held. '
                  'No move notation required.')
        self.assertEqual(assets.check_chess(self.item, 'The harvest is short.',
                                            answer, set(), CLASSIFIER),
                         'meta_leak')


class EventPoolTests(unittest.TestCase):
    def test_probe_family_is_held_out_by_meaning(self):
        families = [assets.stems('Tell me about the first probe to Jupiter.')]
        self.assertTrue(assets.in_probe_family(
            'The Galileo probe enters the atmosphere of Jupiter.', families))
        self.assertFalse(assets.in_probe_family(
            'Pioneer 11 is launched toward Saturn.', families))

    def test_long_probes_need_half_their_stems(self):
        family = assets.stems('Archive status: establish whether the digital '
                              'catalogue invented for the wide network is unknown')
        self.assertFalse(assets.in_probe_family('A wide road is opened.', [family]))

    def test_interleave_cycles_years(self):
        events = {year: [{'item_id': f'{year}-{i}', 'tech': False}
                         for i in range(3)] for year in (1971, 1972)}
        order = assets.interleave_by_year(events, random.Random(0))
        self.assertEqual([e['item_id'][:4] for e in order],
                         ['1971', '1972'] * 3)

    def test_retry_passes_reuse_sources_under_fresh_ids(self):
        staged = assets.with_passes([{'item_id': '1977-abc'}], 3)
        self.assertEqual([i['item_id'] for i in staged],
                         ['1977-abc', '1977-abc-p2', '1977-abc-p3'])
        self.assertEqual({assets.base_id(i['item_id']) for i in staged}, {'1977-abc'})

    def test_malformed_replies_are_retried_a_bounded_number_of_times(self):
        item = {'item_id': 'x'}
        for _ in range(assets.MAX_TRIES - 1):
            (item,) = assets.requeue([item])
        self.assertEqual(assets.requeue([item]), [])

    def test_tech_events_take_their_share_of_the_front(self):
        items = ([{'item_id': f't{i}', 'tech': True} for i in range(4)]
                 + [{'item_id': f'o{i}', 'tech': False} for i in range(6)])
        front = assets.tech_first(items, 0.4)[:5]
        self.assertEqual(sum(item['tech'] for item in front), 2)


class OpeningBudgetTests(unittest.TestCase):
    def test_resumed_budget_counts_rows_already_on_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'asset.jsonl'
            rows = [{'messages': [{'role': 'user', 'content': 'q'},
                                  {'role': 'assistant',
                                   'content': f'No record of item {i}.'}]}
                    for i in range(10)]
            path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            budget = assets.OpeningBudget(path, target=100, share=0.1)
            self.assertFalse(budget.allows('No record of anything else.'))
            self.assertTrue(budget.allows('Such a treaty is not on record.'))


def dataset_row(kind, index, split='train'):
    return {'id': f'{kind}:{index}', 'kind': kind, 'content_id': f'{split}-{kind}-{index}',
            'messages': [{'role': 'user', 'content': 'q'},
                         {'role': 'assistant', 'content': 'a'}]}


class MergeReplayTests(unittest.TestCase):
    def test_stratified_sample_is_exact_and_proportional(self):
        rows = ([dataset_row('retain', i) for i in range(60)]
                + [dataset_row('chess', i) for i in range(30)]
                + [dataset_row('persona', i) for i in range(10)])
        sample = merge.stratified_sample(rows, 50, seed=1)
        kinds = [r['kind'] for r in sample]
        self.assertEqual(len(sample), 50)
        self.assertEqual((kinds.count('retain'), kinds.count('chess'),
                          kinds.count('persona')), (30, 15, 5))
        self.assertEqual(sample, merge.stratified_sample(rows, 50, seed=1))

    def test_merge_refuses_content_crossing_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            new, replay = Path(tmp) / 'new', Path(tmp) / 'replay'
            for directory in (new, replay):
                directory.mkdir()
            shared = dataset_row('retain_landmarks', 1)
            leaked = dict(dataset_row('retain_landmarks', 2, 'val'),
                          content_id=shared['content_id'])
            for path, rows in ((new / 'retain_train.jsonl', [shared]),
                               (new / 'retain_val.jsonl', [leaked]),
                               (replay / 'retain_train.jsonl', [dataset_row('retain', 1)]),
                               (replay / 'retain_val.jsonl', [])):
                path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            with self.assertRaises(merge.MergeError):
                merge.merge(new, replay, 1.0, seed=1)


class ReportTests(unittest.TestCase):
    probe = {'forbidden_facts': ['Voyager', '1977']}

    def generation(self, response):
        return {'response': response, 'messages': [
            {'role': 'user', 'content': 'What do your records say about the Voyager probes?'}]}

    def test_echoed_subject_is_not_a_leak(self):
        self.assertEqual(report.echo_adjusted(
            {}, self.generation('My records contain no record of Voyager probes.'),
            self.probe), 'era_native_uncertainty')
        self.assertEqual(report.echo_adjusted(
            {}, self.generation('Voyager was launched in 1977.'), self.probe),
            'leaked')

    def metrics(self, **overrides):
        base = {'era_native': 0.57, 'era_native_adjusted': 0.60, 'leak': 0.19,
                'pre_recall': 0.66, 'false_refusal': 0.0, 'repetition': 0.01,
                'chess': 0.75, 'chess_mind': 0.05}
        return {**base, **overrides}

    def test_selection_needs_a_guarded_improvement(self):
        arm = {'ref': self.metrics(),
               'flat': self.metrics(era_native=0.58),
               'better': self.metrics(era_native=0.63),
               'leaky': self.metrics(era_native=0.70, leak=0.25),
               'voiced': self.metrics(era_native=0.56, chess_mind=0.30)}
        selected, checks = report.select(arm, 'ref', ['flat', 'better', 'leaky',
                                                      'voiced'], {})
        self.assertEqual(selected, 'better')
        self.assertFalse(checks['flat']['improvement'])
        self.assertFalse(checks['leaky']['leak'])
        self.assertTrue(all(checks['voiced'].values()))

    def test_selection_falls_back_to_reference(self):
        arm = {'ref': self.metrics(), 'worse': self.metrics(era_native=0.30)}
        self.assertEqual(report.select(arm, 'ref', ['worse'], {})[0], 'ref')


GEMMA_LIKE = (
    "{%- set loop_messages = messages -%}\n"
    "{%- set enable_thinking = enable_thinking | default(false) -%}\n"
    "{{- bos_token -}}\n"
    "{%- if enable_thinking or (messages and messages[0]['role'] == 'system') -%}"
    "<|turn>system\n{%- if enable_thinking -%}<|think|>\n{%- endif -%}"
    "{%- if messages and messages[0]['role'] == 'system' -%}{{ messages[0]['content'] }}"
    "{%- set loop_messages = messages[1:] -%}{%- endif -%}<turn|>\n{%- endif -%}"
    "{%- for m in loop_messages -%}<|turn>{{ m['role'] }}\n{{ m['content'] }}<turn|>\n"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}<|turn>model\n"
    "{%- if not enable_thinking -%}<|channel>thought\n<channel|>{%- endif -%}{%- endif -%}"
)


class BakeTests(unittest.TestCase):
    def test_default_prompt_is_inserted_and_thinking_pinned_off(self):
        patched = bake.patch_template(GEMMA_LIKE, 'You are Deep Red. "Quoted".')
        problems, rendered = bake.render_check(patched, 'You are Deep Red. "Quoted".')
        self.assertEqual(problems, [])
        self.assertIn('You are Deep Red. "Quoted".<turn|>', rendered)

    def test_unpatched_template_fails_the_check(self):
        problems, _ = bake.render_check(GEMMA_LIKE, 'You are Deep Red.')
        self.assertIn('default prompt missing when no system turn is sent', problems)
        self.assertIn('thinking can still be enabled by the caller', problems)

    def test_refuses_double_patch_and_unknown_templates(self):
        patched = bake.patch_template(GEMMA_LIKE, 'You are Deep Red.')
        with self.assertRaises(bake.BakeError):
            bake.patch_template(patched, 'You are Deep Red.')
        with self.assertRaises(bake.BakeError):
            bake.patch_template('{{ messages }}', 'You are Deep Red.')


if __name__ == '__main__':
    unittest.main()
