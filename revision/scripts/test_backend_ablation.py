"""Focused contracts for the backend experiment; no historical output writes."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import unittest
import numpy as np
import torch
from revision.backend_ablation import *
from mrw_inverse.models.spectral_geometry_calibrator import SpectralGeometryCalibrator
from revision.scripts.summarize_backend_ablation import bootstrap_ids, auc_boot
from sklearn.metrics import roc_auc_score


class Contracts(unittest.TestCase):
    def test_label_blind_features(self):
        data, meta = analytic_data(64, 991801, 'contract')
        self.assertTrue((data['x'][:, -1] == 0).all())
        self.assertEqual(len(meta.family.unique()), 8)
        np.testing.assert_allclose(data['x'], spectrum_features(data['x'][:, :6]), atol=2e-6, rtol=2e-5)

    def test_original_mlp_equivalence(self):
        d, _ = analytic_data(16, 991802, 'contract'); x = torch.tensor(d['x'])
        # Forward-path equivalence is architectural, independent of checkpoint weights.
        torch.manual_seed(991802)
        cal = SpectralGeometryCalibrator().eval()
        parts = [x[:, :6], x[:, 6:12], x[:, 12:18]] + [x[:, j:j+1] for j in range(18, 26)]
        with torch.no_grad(): a = cal(*parts); b = forward_scores(cal.net, 'mlp', x)
        for j, key in enumerate(SCORES): torch.testing.assert_close(a[key], b[:, j:j+1], rtol=0, atol=0)

    def test_all_losses_and_gradients(self):
        d, _ = analytic_data(64, 991803, 'contract'); td = {k: torch.as_tensor(v) for k, v in d.items()}
        ids = torch.arange(64)
        for variant in VARIANTS:
            m = make_backend(variant, d['x'], 33)
            p = forward_scores(m, variant, td['x']); loss = existing_loss(p, td, ids); loss.backward()
            self.assertTrue(torch.isfinite(loss)); self.assertTrue(((p >= 0) & (p <= 1)).all())
            for par in m.parameters():
                if par.grad is not None: self.assertTrue(torch.isfinite(par.grad).all())

    def test_threshold_ties(self):
        for values in [np.ones(20), np.arange(21), np.array([0]*100+[1]*10)]:
            self.assertLessEqual(np.mean(values > null_threshold(values)), .05)

    def test_rule_signed_coordinate(self):
        lam = np.array([-.05, 0, .01, .1, .2])
        z = .5*Q[None, :]-.5*lam[:, None]*Q*(Q-2)
        score = rule_scores(spectrum_features(z))['rule_signed_curvature']
        np.testing.assert_allclose(score, lam, atol=1e-7)

    def test_nonfinite_rejected(self):
        with self.assertRaises(FloatingPointError): spectrum_features(np.full((1, 6), np.nan))

    def test_split_and_raw_shapes(self):
        a, ma = clean_raw(512, 'test', True); b, mb = clean_raw(512, 'threshold_validation', True)
        self.assertEqual(a.shape[1], 512)
        self.assertFalse(set(ma.data_seed) & set(mb.data_seed))
        self.assertTrue(np.isfinite(a).all() and np.isfinite(b).all())

    def test_analytic_bootstrap_not_singletons(self):
        _, meta = analytic_data(80, 991804, 'contract')
        ids = bootstrap_ids(meta, 20)
        means = np.arange(len(meta))[ids].mean(axis=1)
        self.assertGreater(np.std(means), 0.)

    def test_auc_boot_ties_against_sklearn(self):
        scores = np.array([0., 0., .3, .3, .8, .8]); y = np.array([0, 1, 0, 1, 0, 1])
        ids = np.array([[0, 1, 2, 3, 4, 5], [0, 0, 1, 1, 4, 5]])
        actual = auc_boot(scores, y, ids)
        np.testing.assert_allclose(actual, [roc_auc_score(y[i], scores[i]) for i in ids], atol=1e-14)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
