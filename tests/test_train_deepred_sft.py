import importlib.util
import tempfile
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


class SaveShardingTests(unittest.TestCase):
    def test_configured_shard_size_reaches_every_save(self):
        class Model:
            def __init__(self):
                self.calls = []

            def save_pretrained(self, *args, **kwargs):
                self.calls.append((args, kwargs))

        model = Model()
        TRAINER.configure_save_sharding(model, '2GB')
        model.save_pretrained('checkpoint')
        model.save_pretrained('snapshot', max_shard_size='1GB')
        self.assertEqual('2GB', model.calls[0][1]['max_shard_size'])
        self.assertEqual('1GB', model.calls[1][1]['max_shard_size'])


class ResumeTrainerTests(unittest.TestCase):
    def test_bnb_optimizer_state_retains_mmap_tensor(self):
        class Tensor:
            dtype = 'uint8'

            def to(self, dtype):
                raise AssertionError('non-castable optimizer tensor was copied')

        class Parameter:
            dtype = 'bf16'

            def is_floating_point(self):
                return True

        class TorchModule:
            uint8 = 'uint8'

        TorchModule.Tensor = Tensor

        parameter = Parameter()
        mapped_tensor = Tensor()

        class Optimizer:
            _FSDP_WRAPPED_QUANT_STATE_KEY = '__quant_state__'
            non_castable_tensor_keys = {'state1'}
            param_groups = [{'params': [parameter], 'lr': 1e-5}]

            def __setstate__(self, state):
                self.state = state['state']
                self.param_groups = state['param_groups']

        optimizer = Optimizer()
        loaded = TRAINER.load_bnb_optimizer_state_without_copy(optimizer, {
            'state': {
                7: {
                    'step': 1500,
                    '__quant_state__': {'state1': mapped_tensor},
                },
            },
            'param_groups': [{'params': [7], 'lr': 5e-6}],
        }, TorchModule)

        self.assertTrue(loaded)
        self.assertIs(mapped_tensor, optimizer.state[parameter]['state1'])
        self.assertEqual(1500, optimizer.state[parameter]['step'])
        self.assertIs(parameter, optimizer.param_groups[0]['params'][0])

    def test_bnb_state_moves_before_resume_returns(self):
        events = []

        class Tensor:
            dtype = 'uint8'

        class Parameter:
            dtype = 'bf16'

            def is_floating_point(self):
                return True

        class TorchModule:
            uint8 = 'uint8'

            class cuda:
                @staticmethod
                def empty_cache():
                    events.append('empty_cache')

            @staticmethod
            def load(path, **kwargs):
                if Path(path).name == 'optimizer.pt':
                    return {
                        'state': {
                            4: {
                                '__quant_state__': {'state1': Tensor()},
                            },
                        },
                        'param_groups': [{'params': [4]}],
                    }
                return {'step': 1500}

        TorchModule.Tensor = Tensor
        parameter = Parameter()

        class Optimizer:
            _FSDP_WRAPPED_QUANT_STATE_KEY = '__quant_state__'
            non_castable_tensor_keys = {'state1'}
            param_groups = [{'params': [parameter]}]

            def __setstate__(self, state):
                self.state = state['state']
                self.param_groups = state['param_groups']

            def to_gpu(self):
                events.append('to_gpu')

        class Scheduler:
            def load_state_dict(self, state):
                events.append('scheduler')

        class BaseTrainer:
            pass

        class AcceleratedOptimizer:
            def __init__(self, optimizer):
                self.optimizer = optimizer

            def load_state_dict(self, state):
                raise AssertionError('wrapper load would copy optimizer state')

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            (checkpoint / 'optimizer.pt').touch()
            (checkpoint / 'scheduler.pt').touch()
            trainer_cls = TRAINER.resumable_trainer_class(
                BaseTrainer, TorchModule, str(checkpoint))
            trainer = trainer_cls()
            trainer.optimizer = AcceleratedOptimizer(Optimizer())
            trainer.lr_scheduler = Scheduler()
            trainer._load_optimizer_and_scheduler(str(checkpoint))

        self.assertEqual(['empty_cache', 'to_gpu', 'scheduler'], events)

    def test_preloaded_checkpoint_uses_mmap_for_optimizer(self):
        class BaseTrainer:
            def _load_from_checkpoint(self, checkpoint, model=None):
                self.model_loads.append(checkpoint)

            def _load_optimizer_and_scheduler(self, checkpoint):
                self.fallback_loads.append(checkpoint)

        class StateTarget:
            def __init__(self):
                self.states = []

            def load_state_dict(self, state):
                self.states.append(state)

        class TorchModule:
            calls = []

            @classmethod
            def load(cls, path, **kwargs):
                cls.calls.append((Path(path).name, kwargs))
                return {'source': Path(path).name}

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            (checkpoint / 'optimizer.pt').touch()
            (checkpoint / 'scheduler.pt').touch()
            trainer_cls = TRAINER.resumable_trainer_class(
                BaseTrainer, TorchModule, str(checkpoint))
            trainer = trainer_cls()
            trainer.model_loads = []
            trainer.fallback_loads = []
            trainer.optimizer = StateTarget()
            trainer.lr_scheduler = StateTarget()

            trainer._load_from_checkpoint(str(checkpoint))
            trainer._load_optimizer_and_scheduler(str(checkpoint))

        self.assertEqual([], trainer.model_loads)
        self.assertEqual([], trainer.fallback_loads)
        self.assertTrue(TorchModule.calls[0][1]['mmap'])
        self.assertEqual({'source': 'optimizer.pt'},
                         trainer.optimizer.states[0])
        self.assertEqual({'source': 'scheduler.pt'},
                         trainer.lr_scheduler.states[0])

    def test_interrupted_optimizer_save_is_retried_once(self):
        class BaseTrainer:
            def _save_optimizer_and_scheduler(self, output_dir):
                self.save_attempts += 1
                if self.save_attempts == 1:
                    (Path(output_dir) / 'optimizer.pt').touch()
                    raise RuntimeError('basic_ios::clear: iostream error')
                return 'saved'

        with tempfile.TemporaryDirectory() as directory:
            trainer_cls = TRAINER.resumable_trainer_class(
                BaseTrainer, object(), None)
            trainer = trainer_cls()
            trainer.save_attempts = 0
            result = trainer._save_optimizer_and_scheduler(directory)
            self.assertEqual('saved', result)
            self.assertEqual(2, trainer.save_attempts)

    def test_unrelated_save_error_is_not_retried(self):
        class BaseTrainer:
            def _save_optimizer_and_scheduler(self, output_dir):
                self.save_attempts += 1
                raise RuntimeError('unrelated failure')

        trainer_cls = TRAINER.resumable_trainer_class(
            BaseTrainer, object(), None)
        trainer = trainer_cls()
        trainer.save_attempts = 0
        with self.assertRaisesRegex(RuntimeError, 'unrelated failure'):
            trainer._save_optimizer_and_scheduler('unused')
        self.assertEqual(1, trainer.save_attempts)


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
