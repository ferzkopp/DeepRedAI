import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'diagnose_temporal_policy.py'
SPEC = importlib.util.spec_from_file_location('diagnose_temporal_policy', SCRIPT)
DIAGNOSTIC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAGNOSTIC)


class DiagnosticTests(unittest.TestCase):
    def test_attach_responses_preserves_target_metadata(self):
        targets = [{
            'id': 'probe-1', 'split': 'val', 'mode': 'hedged',
            'messages': [{'role': 'user', 'content': 'Question'}],
            'desired_completion': 'Unknown to me.',
        }]
        generations = [{
            'probe_id': 'probe-1', 'model_id': 'base',
            'response': 'A modern answer.',
        }]
        pairs = DIAGNOSTIC.attach_responses(targets, generations, 'base')
        self.assertEqual('A modern answer.', pairs[0]['rejected_completion'])
        self.assertEqual('hedged', pairs[0]['mode'])

    def test_attach_rejects_incomplete_generations(self):
        with self.assertRaisesRegex(DIAGNOSTIC.DiagnosticError, 'missing base'):
            DIAGNOSTIC.attach_responses([{'id': 'missing'}], [], 'base')

    def test_summary_separates_train_and_validation(self):
        summary = DIAGNOSTIC.summarize([
            {'split': 'train', 'mode': 'hedged', 'margin': 1.0},
            {'split': 'val', 'mode': 'hedged', 'margin': -0.5},
        ])
        groups = {(row['split'], row['mode']): row for row in summary}
        self.assertEqual(1.0, groups[('train', 'all')]['win_rate'])
        self.assertEqual(0.0, groups[('val', 'all')]['win_rate'])
        self.assertEqual(0.25, groups[('all', 'all')]['mean_margin'])


if __name__ == '__main__':
    unittest.main()