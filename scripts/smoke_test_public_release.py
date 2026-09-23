"""Checkpoint-free numerical regression gate for the corrected public code."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import numpy as np
import torch
from revision.estimators.structure_input_contract import estimate_revision_robust_zeta
from revision.analysis_tools import structure_zeta, FrozenCalibrator
from revision.scripts import test_revision_numerics, test_retraining_contracts, test_backend_ablation


class PublicContracts(unittest.TestCase):
    def test_explicit_input_equivalence(self):
        x = torch.tensor(np.random.default_rng(512).normal(size=512), dtype=torch.float64)
        a = estimate_revision_robust_zeta(x, input_kind='increments')
        b = estimate_revision_robust_zeta(x.cumsum(0), input_kind='path')
        torch.testing.assert_close(a.zeta_ols, b.zeta_ols, rtol=0, atol=0)
        np.testing.assert_allclose(a.zeta_ols.numpy().ravel(), structure_zeta(x.numpy()), atol=1e-9)

    def test_bad_input_rejected(self):
        x = torch.zeros(512, dtype=torch.float64)
        with self.assertRaises(ValueError):
            estimate_revision_robust_zeta(x, input_kind='unspecified')
        x[0] = float('nan')
        with self.assertRaises(ValueError):
            estimate_revision_robust_zeta(x, input_kind='increments')

    def test_missing_checkpoint_fails_closed(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(FileNotFoundError, 'No random-weight fallback'):
                FrozenCalibrator(Path(folder)/'missing.pt')


if __name__ == '__main__':
    torch.set_num_threads(1)
    loader = unittest.defaultTestLoader
    suite = unittest.TestSuite([loader.loadTestsFromModule(m) for m in
        [test_revision_numerics, test_retraining_contracts, test_backend_ablation]])
    suite.addTests(loader.loadTestsFromTestCase(PublicContracts))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
