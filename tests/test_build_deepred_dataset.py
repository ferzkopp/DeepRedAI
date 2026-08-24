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


if __name__ == '__main__':
    unittest.main()