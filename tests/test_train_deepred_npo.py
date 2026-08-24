import importlib.util
import unittest
from pathlib import Path
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'train_deepred_npo.py'
SPEC = importlib.util.spec_from_file_location('train_deepred_npo', SCRIPT)
NPO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(NPO)


class LossTests(unittest.TestCase):
    def test_unknown_kind_weight_is_rejected(self):
        with self.assertRaises(NPO.TrainingError):
            NPO.parse_kind_weights(['era_nativ=4'])

    def test_host_virtualenv_is_rejected_before_training_imports(self):
        with mock.patch.object(NPO.sys, 'executable', '/mnt/data/venv/bin/python3'):
            with self.assertRaisesRegex(NPO.TrainingError, '/opt/venv/bin/python3'):
                NPO.train(object())

    def test_tokenizer_batch_encoding_shape_is_supported(self):
        class FakeTokenizer:
            def apply_chat_template(self, messages, tokenize,
                                    add_generation_prompt):
                if add_generation_prompt:
                    return {'input_ids': [1, 2, 3]}
                return {'input_ids': [1, 2, 3, 4, 5]}

        encoded = NPO.tokenize_messages(FakeTokenizer(), [
            {'role': 'user', 'content': 'question'},
            {'role': 'assistant', 'content': 'answer'},
        ], 10)
        self.assertEqual([-100, -100, -100, 4, 5], encoded['labels'])

    def test_npo_penalizes_increased_forget_probability(self):
        reference = torch.tensor([-5.0])
        counts = torch.tensor([2])
        objectives = torch.tensor([True])
        lower, _, _ = NPO.npo_retain_loss(
            torch.tensor([-6.0]), reference, counts, objectives)
        higher, _, _ = NPO.npo_retain_loss(
            torch.tensor([-4.0]), reference, counts, objectives)
        self.assertLess(lower.item(), higher.item())

    def test_retain_anchor_penalizes_probability_loss(self):
        reference = torch.tensor([-4.0])
        counts = torch.tensor([2])
        objectives = torch.tensor([False])
        same, _, _ = NPO.npo_retain_loss(
            torch.tensor([-4.0]), reference, counts, objectives)
        worse, _, _ = NPO.npo_retain_loss(
            torch.tensor([-6.0]), reference, counts, objectives)
        self.assertEqual(0.0, same.item())
        self.assertGreater(worse.item(), same.item())

    def test_balancing_oversamples_smaller_side(self):
        rows = NPO.balanced_rows(
            [{'id': 'f', 'messages': []}],
            [{'id': f'r{index}', 'messages': []} for index in range(3)], 1)
        self.assertEqual(6, len(rows))
        self.assertEqual(3, sum(row['objective'] == 'forget' for row in rows))

    def test_weighted_mix_controls_forget_ratio_and_kinds(self):
        retain = [
            {'id': f'f{index}', 'kind': 'retain', 'messages': []}
            for index in range(10)
        ] + [
            {'id': f'e{index}', 'kind': 'era_native', 'messages': []}
            for index in range(5)
        ]
        rows = NPO.balanced_rows(
            [{'id': 'forget', 'messages': []}], retain, 1,
            forget_ratio=0.25, kind_weights={'era_native': 2})
        forget = [row for row in rows if row['objective'] == 'forget']
        retained = [row for row in rows if row['objective'] == 'retain']
        self.assertEqual(20, len(retained))
        self.assertEqual(7, len(forget))
        self.assertEqual(
            10, sum(row['kind'] == 'era_native' for row in retained))


if __name__ == '__main__':
    unittest.main()