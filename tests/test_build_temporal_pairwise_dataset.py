import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'build_temporal_pairwise_dataset.py'
SPEC = importlib.util.spec_from_file_location('pair_builder', SCRIPT)
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


class PairBuilderTests(unittest.TestCase):
    def test_finalize_filters_uncertain_rejected_completion(self):
        candidates = []
        generations = []
        for split in ('train', 'val'):
            for mode in BUILDER.MODES:
                for index in range(2):
                    row_id = f'{split}-{mode}-{index}'
                    candidates.append({
                        'id': row_id, 'split': split, 'mode': mode,
                        'messages': [], 'chosen_completion': 'Unknown.',
                    })
                    response = ('I do not know when that happened.' if index == 0
                                else 'It happened in 2004.')
                    generations.append({
                        'probe_id': row_id, 'model_id': 'base',
                        'response': response,
                    })
        selected = BUILDER.finalize_pairs(
            candidates, generations, 'base', {'train': 1, 'val': 1}, 1)
        self.assertEqual(3, len(selected['train']))
        self.assertEqual(3, len(selected['val']))
        self.assertTrue(all(row['rejected_completion'] == 'It happened in 2004.'
                            for rows in selected.values() for row in rows))

    def test_finalize_requires_complete_generations(self):
        with self.assertRaisesRegex(BUILDER.DatasetError, 'missing base'):
            BUILDER.finalize_pairs(
                [{'id': 'x', 'split': 'train', 'mode': 'hedged'}], [],
                'base', {'train': 1, 'val': 1}, 1)


if __name__ == '__main__':
    unittest.main()