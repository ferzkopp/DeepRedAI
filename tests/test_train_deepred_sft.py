import importlib.util
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'train_deepred_sft.py'
SPEC = importlib.util.spec_from_file_location('sft_trainer', SCRIPT)
TRAINER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAINER)


class ShuffleTests(unittest.TestCase):
    def test_shuffle_is_deterministic_for_a_seed(self):
        rows = [{'id': f'row-{index}'} for index in range(50)]
        first = TRAINER.shuffled_rows(rows, 1969)
        second = TRAINER.shuffled_rows(list(reversed(rows)), 1969)
        self.assertEqual([row['id'] for row in first],
                         [row['id'] for row in second])

    def test_shuffle_changes_order_with_seed(self):
        rows = [{'id': f'row-{index}'} for index in range(50)]
        self.assertNotEqual(
            [row['id'] for row in TRAINER.shuffled_rows(rows, 1)],
            [row['id'] for row in TRAINER.shuffled_rows(rows, 2)])

    def test_empty_split_fails(self):
        with self.assertRaises(TRAINER.TrainingError):
            TRAINER.shuffled_rows([], 1969)


class CollatorTests(unittest.TestCase):
    def test_padding_masks_labels_and_attention(self):
        class Tokenizer:
            pad_token_id = 0

        batch = TRAINER.make_collator(Tokenizer())([
            {'input_ids': [1, 2, 3], 'labels': [-100, 2, 3]},
            {'input_ids': [4], 'labels': [4]},
        ])
        self.assertEqual([[1, 2, 3], [4, 0, 0]], batch['input_ids'].tolist())
        self.assertEqual([[-100, 2, 3], [4, -100, -100]],
                         batch['labels'].tolist())
        self.assertEqual([[1, 1, 1], [1, 0, 0]],
                         batch['attention_mask'].tolist())
        self.assertEqual(torch.long, batch['input_ids'].dtype)


class ParserTests(unittest.TestCase):
    def test_defaults_match_phase3_plan(self):
        args = TRAINER.build_parser().parse_args([
            '--model', 'm', '--tokenizer', 't',
            '--dataset', 'd', '--output-dir', 'o'])
        self.assertEqual(5e-6, args.learning_rate)
        self.assertEqual(2.0, args.epochs)
        self.assertEqual(16, args.gradient_accumulation)

    def test_nonpositive_budget_is_rejected(self):
        self.assertEqual(1, TRAINER.main([
            '--model', 'm', '--tokenizer', 't', '--dataset', 'd',
            '--output-dir', 'o', '--epochs', '0', '--max-steps', '-1']))


if __name__ == '__main__':
    unittest.main()
