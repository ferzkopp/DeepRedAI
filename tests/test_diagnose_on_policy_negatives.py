import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'diagnose_on_policy_negatives.py'
SPEC = importlib.util.spec_from_file_location('on_policy_diagnostic', SCRIPT)
DIAGNOSTIC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAGNOSTIC)


class OnPolicyDiagnosticTests(unittest.TestCase):
    def test_first_sentence_handles_terminal_quote(self):
        self.assertEqual(
            '"No record exists."',
            DIAGNOSTIC.first_sentence('"No record exists." More text.'))

    def test_prepare_rows_balances_train_and_keeps_all_validation(self):
        train = []
        validation = []
        generations = []
        for split, target in (('train', train), ('val', validation)):
            for mode in DIAGNOSTIC.BUILDER.MODES:
                for index in range(2):
                    row_id = f'{split}-{mode}-{index}'
                    target.append({
                        'id': row_id, 'split': split, 'mode': mode,
                        'messages': [], 'chosen_completion': 'Unknown.',
                        'rejected_completion': 'A fresh modern answer.',
                    })
                    generations.append({
                        'probe_id': row_id, 'model_id': 'base',
                        'response': 'An original modern answer.',
                    })
        rows = DIAGNOSTIC.prepare_rows(
            train, validation, generations, 'base', 1, 1969)
        self.assertEqual(9, len(rows))
        self.assertEqual(3, sum(row['split'] == 'train' for row in rows))
        self.assertTrue(all(row['fresh_behavior'] == 'confident_unsupported'
                            for row in rows))

    def test_summary_reports_negative_logp_routing(self):
        summary = DIAGNOSTIC.summarize([{
            'split': 'val', 'mode': 'hedged',
            'desired_mean_logp': -1.5,
            'desired_first_mean_logp': -1.4,
            'original_margin': -0.5, 'fresh_margin': -1.0,
            'original_first_margin': -0.25, 'fresh_first_margin': -0.75,
            'original_rejected_mean_logp': -2.0,
            'original_rejected_first_mean_logp': -1.15,
            'fresh_rejected_mean_logp': -1.0,
            'fresh_rejected_first_mean_logp': -0.65,
        }])
        overall = next(row for row in summary
                       if row['split'] == row['mode'] == 'all')
        self.assertEqual(1.0, overall['fresh_minus_original_logp'])
        self.assertEqual(0.0, overall['fresh_win_rate'])


if __name__ == '__main__':
    unittest.main()