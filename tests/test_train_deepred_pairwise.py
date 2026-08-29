import importlib.util
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'train_deepred_pairwise.py'
SPEC = importlib.util.spec_from_file_location('pairwise_trainer', SCRIPT)
TRAINER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAINER)


class PairwiseTrainerTests(unittest.TestCase):
    def test_larger_margin_has_lower_loss(self):
        counts = torch.tensor([2.0])
        low, low_margin = TRAINER.pairwise_margin_loss(
            torch.tensor([-4.0]), counts, torch.tensor([-2.0]), counts)
        high, high_margin = TRAINER.pairwise_margin_loss(
            torch.tensor([-1.0]), counts, torch.tensor([-3.0]), counts)
        self.assertLess(low_margin.item(), high_margin.item())
        self.assertGreater(low.item(), high.item())

    def test_margin_is_token_normalized(self):
        loss, margin = TRAINER.pairwise_margin_loss(
            torch.tensor([-4.0]), torch.tensor([2.0]),
            torch.tensor([-9.0]), torch.tensor([3.0]), margin_target=0.25)
        self.assertEqual(1.0, margin.item())
        self.assertAlmostEqual(
            torch.nn.functional.softplus(torch.tensor(-0.75)).item(),
            loss.item())

    def test_chosen_ce_adds_positive_likelihood_pressure(self):
        chosen_logps = torch.tensor([-4.0])
        chosen_counts = torch.tensor([2.0])
        total, pair, chosen_nll, margin = TRAINER.pairwise_with_chosen_loss(
            chosen_logps, chosen_counts, torch.tensor([-3.0]),
            torch.tensor([2.0]), chosen_ce_weight=0.5)
        self.assertAlmostEqual(2.0, chosen_nll.item())
        self.assertAlmostEqual(pair.item() + 1.0, total.item(), places=6)
        self.assertAlmostEqual(-0.5, margin.item())

    def test_zero_chosen_ce_preserves_v5_objective(self):
        arguments = (
            torch.tensor([-4.0]), torch.tensor([2.0]),
            torch.tensor([-3.0]), torch.tensor([2.0]))
        old_loss, old_margin = TRAINER.pairwise_margin_loss(*arguments)
        total, pair, _, margin = TRAINER.pairwise_with_chosen_loss(*arguments)
        self.assertEqual(old_loss.item(), total.item())
        self.assertEqual(old_loss.item(), pair.item())
        self.assertEqual(old_margin.item(), margin.item())

    def test_objective_rows_are_balanced_and_alternating(self):
        rows = TRAINER.balanced_objective_rows(
            [{'id': 'p1'}, {'id': 'p2'}],
            [{'id': 'a1'}, {'id': 'a2'}], 1)
        self.assertEqual(['pair', 'anchor', 'pair', 'anchor'],
                         [row['objective'] for row in rows])

    def test_objective_rows_require_equal_lengths(self):
        with self.assertRaisesRegex(TRAINER.TrainingError, 'equal lengths'):
            TRAINER.balanced_objective_rows([{'id': 'p'}], [{'id': 'a'}, {'id': 'b'}], 1)

    def test_collator_supplies_evaluation_routing_label(self):
        class Tokenizer:
            pad_token_id = 0

        collator = TRAINER.make_collator(Tokenizer())
        batch = collator([{
            'objective': 'anchor',
            'anchor_input_ids': [1, 2],
            'anchor_labels': [-100, 2],
        }])
        self.assertEqual([0], batch['loss_labels'].tolist())
        self.assertEqual('anchor', batch['objective'])


if __name__ == '__main__':
    unittest.main()