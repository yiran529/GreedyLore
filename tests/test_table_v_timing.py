import unittest
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch

import c4.table_v_timing as timing
from c4.table_v_timing import build_timing_optimizer, summarize_timing
from c4.scripts.run_table_v_adam_350m import cells


class TimingSummaryTests(unittest.TestCase):
    def test_uses_slowest_rank_continuous_wall_clock(self):
        result = summarize_timing([10.0, 12.0, 11.0, 10.5], [[1., 2.], [2., 3.], [1., 1.], [2., 2.]], 2)
        self.assertEqual(result['mean_iteration_seconds'], 6.0)
        self.assertEqual(result['rank_elapsed_seconds'], [10.0, 12.0, 11.0, 10.5])
        self.assertEqual(result['measured_steps'], 2)

    def test_rejects_partial_or_nonfinite_measurements(self):
        for elapsed, steps in [([1.], [[1.]]), ([float('nan')], [[1., 1.]]), ([1.], [[1., float('inf')]])]:
            with self.assertRaises(ValueError):
                summarize_timing(elapsed, steps, 2)

    def test_ddp_bucket_cap_is_forwarded_when_requested(self):
        args = SimpleNamespace(ddp_bucket_cap_mb=1024)
        self.assertTrue(hasattr(timing, 'build_ddp_kwargs'))

        self.assertEqual(
            timing.build_ddp_kwargs(args, local_rank=3),
            {
                'device_ids': [3],
                'output_device': 3,
                'broadcast_buffers': False,
                'bucket_cap_mb': 1024,
            },
        )


class MuonSettingMatrixTests(unittest.TestCase):
    def test_matrix_has_nine_groups_four_models_and_two_arms(self):
        script = Path('c4/scripts/run_table_v_muon_setting_matrix.py')
        self.assertTrue(script.is_file(), f'missing controller: {script}')
        spec = importlib.util.spec_from_file_location('setting_matrix', script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        matrix = list(module.cells())

        self.assertEqual(len(matrix), 72)
        self.assertEqual(len({cell['group'] for cell in matrix}), 9)
        self.assertEqual({cell['model'] for cell in matrix}, {'60m', '130m', '350m', '1b'})
        self.assertEqual({cell['arm'] for cell in matrix}, {'dense', 'greedylore'})
        self.assertTrue(all(cell['bucket_cap_mb'] == 1024 for cell in matrix))
        self.assertTrue(all(cell['measured_iterations'] == 400 for cell in matrix))
        self.assertEqual(matrix[0]['number'], 62)
        self.assertEqual(matrix[-1]['number'], 133)
        socket_cells = [cell for cell in matrix if cell['transport'] == 'socket']
        batch_32_cells = [cell for cell in matrix if cell['batch_size'] == 32]
        self.assertEqual(len(socket_cells), 8)
        self.assertEqual(len(batch_32_cells), 8)
        self.assertTrue(all(cell['world_size'] == 8 for cell in socket_cells))
        self.assertTrue(all(cell['channels'] == 'one' for cell in socket_cells))
        self.assertEqual({cell['model'] for cell in batch_32_cells}, {'60m', '130m', '350m', '1b'})


class AdamTimingTests(unittest.TestCase):
    def test_builds_adamw_with_requested_hyperparameters(self):
        model = torch.nn.Linear(3, 2)
        args = SimpleNamespace(
            optimizer='adamw', lr=1e-3, beta1=0.8, beta2=0.95,
            eps=1e-7, weight_decay=0.02,
        )

        optimizer = build_timing_optimizer(model, args)

        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(optimizer.defaults['lr'], 1e-3)
        self.assertEqual(optimizer.defaults['betas'], (0.8, 0.95))
        self.assertEqual(optimizer.defaults['eps'], 1e-7)
        self.assertEqual(optimizer.defaults['weight_decay'], 0.02)

    def test_350m_adam_matrix_uses_100_step_warmup(self):
        matrix = list(cells())
        self.assertEqual([cell['arm'] for cell in matrix], ['dense', 'greedylore'])
        for cell in matrix:
            command = cell['command']
            self.assertEqual(command[command.index('--optimizer') + 1], 'adamw')
            self.assertEqual(command[command.index('--warmup_iterations') + 1], '100')
            self.assertEqual(command[command.index('--start_compress_iter') + 1], '100')
            self.assertEqual(command[command.index('--lr') + 1], '0.001')
        self.assertEqual(
            matrix[1]['command'][matrix[1]['command'].index('--compressor') + 1],
            'top_subspace',
        )


if __name__ == '__main__':
    unittest.main()
