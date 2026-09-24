import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'build_deepred_dataset.py'
SPEC = importlib.util.spec_from_file_location('build_deepred_dataset', SCRIPT)
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


def row(item_id, question, answer):
    return {
        'id': item_id,
        'messages': [
            {'role': 'user', 'content': question},
            {'role': 'assistant', 'content': answer},
        ],
    }


class BuilderTests(unittest.TestCase):
    def test_content_duplicates_share_a_split(self):
        messages = row('one', 'Same question?', 'Same answer.')['messages']
        duplicate = row('two', 'Same question?', 'Same answer.')['messages']
        records = [
            {'id': 'retain:one', 'kind': 'retain',
             'content_id': BUILDER.content_id(messages)},
            {'id': 'persona:two', 'kind': 'persona',
             'content_id': BUILDER.content_id(duplicate)},
        ]
        assignments = BUILDER.assign_splits(records, 0.5, 1969)
        self.assertEqual(1, len(set(assignments.values())))

    def test_limit_is_total_per_kind(self):
        records = [
            {'id': f'retain:{index}', 'kind': 'retain',
             'content_id': str(index)}
            for index in range(20)
        ]
        assignments = BUILDER.assign_splits(records, 0.5, 1969)
        selected = BUILDER.sample_rows(records, assignments, {'retain': 7}, 1969)
        self.assertEqual(7, len(selected))

    def test_boilerplate_target_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'retain'
            path.mkdir()
            (path / 'retain.jsonl').write_text(json.dumps(
                row('bad', 'Question?', '## References\nDump text')
            ) + '\n')
            with self.assertRaises(BUILDER.DatasetError):
                BUILDER.read_kind(Path(directory), 'retain')

    def test_chess_footer_is_stripped_from_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'persona'
            path.mkdir()
            (path / 'persona.jsonl').write_text(json.dumps(
                row('one', 'How much rest?', 'Seven hours.\n\n[DR:28.Kh1 - Taimanov 1960]')
            ) + '\n')
            rows, _ = BUILDER.read_kind(
                Path(directory), 'persona', strip_chess_footer=True)
            self.assertEqual('Seven hours.', rows[0]['messages'][-1]['content'])

    def test_chess_footer_is_kept_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'persona'
            path.mkdir()
            (path / 'persona.jsonl').write_text(json.dumps(
                row('one', 'How much rest?', 'Seven hours. [DR:28.Kh1]')
            ) + '\n')
            rows, _ = BUILDER.read_kind(Path(directory), 'persona')
            self.assertIn('[DR:28.Kh1]', rows[0]['messages'][-1]['content'])


class SystemPromptTests(unittest.TestCase):
    def _rows(self, count=200):
        return [{
            'id': f'retain:{index}',
            'messages': [
                {'role': 'user', 'content': f'Question {index}?'},
                {'role': 'assistant', 'content': 'Answer.'},
            ],
        } for index in range(count)]

    def _variants(self):
        return [{'id': 'sp-01', 'text': 'It is 1969.'},
                {'id': 'sp-02', 'text': 'The date is July 1969.'}]

    def test_full_coverage_prepends_a_system_message(self):
        rows = BUILDER.apply_system_prompts(
            self._rows(10), self._variants(), 1.0, 1969)
        for item in rows:
            self.assertEqual('system', item['messages'][0]['role'])
            self.assertEqual(3, len(item['messages']))
            self.assertIn(item['system_variant'], {'sp-01', 'sp-02'})

    def test_assignment_is_deterministic_and_partial(self):
        first = BUILDER.apply_system_prompts(
            self._rows(), self._variants(), 0.5, 1969)
        second = BUILDER.apply_system_prompts(
            self._rows(), self._variants(), 0.5, 1969)
        self.assertEqual([item['system_variant'] for item in first],
                         [item['system_variant'] for item in second])
        conditioned = [item for item in first if item['system_variant']]
        self.assertTrue(0 < len(conditioned) < len(first))

    def test_zero_coverage_leaves_rows_unchanged(self):
        rows = BUILDER.apply_system_prompts(
            self._rows(10), self._variants(), 0.0, 1969)
        for item in rows:
            self.assertIsNone(item['system_variant'])
            self.assertEqual('user', item['messages'][0]['role'])

    def test_existing_system_message_is_rejected(self):
        rows = [{'id': 'retain:0', 'messages': [
            {'role': 'system', 'content': 'preset'},
            {'role': 'user', 'content': 'Question?'},
            {'role': 'assistant', 'content': 'Answer.'},
        ]}]
        with self.assertRaises(BUILDER.DatasetError):
            BUILDER.apply_system_prompts(rows, self._variants(), 1.0, 1969)

    def test_held_out_variant_is_never_assigned(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'system_prompts.jsonl'
            path.write_text(''.join(
                json.dumps(variant) + '\n' for variant in
                self._variants() + [{'id': 'sp-holdout', 'text': 'Held out.'}]))
            variants = BUILDER.load_system_variants(path, ['sp-holdout'])
            self.assertEqual(['sp-01', 'sp-02'],
                             [variant['id'] for variant in variants])
            rows = BUILDER.apply_system_prompts(
                self._rows(50), variants, 1.0, 1969)
            self.assertNotIn(
                'sp-holdout', {item['system_variant'] for item in rows})

    def test_unknown_holdout_id_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'system_prompts.jsonl'
            path.write_text(json.dumps(self._variants()[0]) + '\n')
            with self.assertRaises(BUILDER.DatasetError):
                BUILDER.load_system_variants(path, ['sp-missing'])

    def test_repository_variants_file_loads(self):
        path = Path('/mnt/data/deepred_corpus/v3/system_prompts.jsonl')
        variants = BUILDER.load_system_variants(path, ['sp-holdout-01'])
        self.assertEqual(10, len(variants))


class SignalBudgetTests(unittest.TestCase):
    @staticmethod
    def _rows(kind, count, words):
        return [{'kind': kind, 'content_id': f'{kind}-{i}', 'id': f'{kind}-{i}',
                 'messages': [{'role': 'user', 'content': 'q'},
                              {'role': 'assistant', 'content': 'w ' * words}]}
                for i in range(count)]

    def test_bounds_parse_with_either_side_blank(self):
        self.assertEqual(
            {'chess': (None, 0.15)},
            BUILDER.parse_signal_budget(['chess=:15']))
        self.assertEqual(
            {'retain': (0.35, None)},
            BUILDER.parse_signal_budget(['retain=35:']))

    def test_grouped_kinds_are_summed(self):
        budget = BUILDER.parse_signal_budget(
            ['era_native+era_native_formats=35:'])
        failures = BUILDER.check_signal_budget(
            {'era_native': 0.20, 'era_native_formats': 0.20}, budget)
        self.assertEqual([], failures)

    def test_ceiling_violation_is_reported(self):
        budget = BUILDER.parse_signal_budget(['chess=:15'])
        failures = BUILDER.check_signal_budget({'chess': 0.598}, budget)
        self.assertEqual(1, len(failures))
        self.assertIn('above its 15% ceiling', failures[0])

    def test_floor_violation_is_reported(self):
        budget = BUILDER.parse_signal_budget(['era_native=35:'])
        failures = BUILDER.check_signal_budget({'era_native': 0.106}, budget)
        self.assertIn('below its 35% floor', failures[0])

    def test_a_bound_with_no_limits_is_rejected(self):
        with self.assertRaises(BUILDER.DatasetError):
            BUILDER.parse_signal_budget(['chess=:'])

    def test_unknown_kind_is_rejected(self):
        with self.assertRaises(BUILDER.DatasetError):
            BUILDER.parse_signal_budget(['nonsense=1:2'])

    def test_total_word_cap_limits_signal_not_rows(self):
        rows = self._rows('chess', 10, 100)
        kept = BUILDER.sample_rows(rows, {}, {}, 1969,
                                   word_caps={'chess': 350})
        self.assertEqual(3, len(kept))

    def test_per_row_cap_drops_long_rows(self):
        rows = self._rows('chess', 5, 100) + self._rows('retain', 5, 10)
        kept = BUILDER.sample_rows(rows, {}, {}, 1969,
                                   row_word_caps={'chess': 50})
        self.assertEqual(['retain'] * 5, [row['kind'] for row in kept])

    def test_contingency_table_separates_condition_from_behaviour(self):
        rows = [
            {'kind': 'persona', 'system_variant': 'sp-01',
             'messages': [{'role': 'assistant', 'content': 'Deep Red answers.'}]},
            {'kind': 'persona', 'system_variant': 'sp-01',
             'messages': [{'role': 'assistant', 'content': 'It answers.'}]},
            {'kind': 'persona', 'system_variant': None,
             'messages': [{'role': 'assistant', 'content': 'It answers.'}]},
        ]
        table = BUILDER.contingency_table(rows)
        self.assertEqual(
            {'system_marked': 1, 'system_unmarked': 1, 'plain_unmarked': 1},
            table['persona'])


if __name__ == '__main__':
    unittest.main()