"""Checkpoint-free contracts for the additional validation scripts."""
import ast
import unittest

from common import HERE, Q, corr, np, pd
from correct_thresholds import calibrate, rank_threshold
from revision.analysis_tools import projection_metrics
from validate_simulator import reference


class ValidationContracts(unittest.TestCase):
    def test_python_sources_parse(self):
        for path in HERE.glob('*.py'):
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path))

    def test_pinned_reference_hashes_and_imports(self):
        mrw, gaussian = reference()
        self.assertTrue(callable(mrw.mrw))
        self.assertTrue(callable(gaussian.gaussian_cme))

    def test_signed_and_clipped_coordinates_are_distinct(self):
        for lam in [-0.03, 0.0, 0.1]:
            zeta = 0.5 * Q - 0.5 * lam * Q * (Q - 2)
            result = projection_metrics(zeta)
            self.assertAlmostEqual(result['lambda2_signed'], lam, places=12)
            self.assertAlmostEqual(result['lambda2_proj'], max(lam, 1e-6), places=12)

    def test_finite_sample_quantile_and_ties(self):
        threshold, rank = rank_threshold(np.arange(500), 0.05)
        self.assertEqual((threshold, rank), (475.0, 476))
        threshold, _ = rank_threshold(np.full(500, 1e-6), 0.05)
        self.assertEqual(np.count_nonzero(np.full(500, 1e-6) > threshold), 0)
        with self.assertRaises(ValueError):
            rank_threshold(np.arange(5), 0.05)

    def test_null_only_calibration_uses_maximum_group_quantile(self):
        groups = []
        for h, offset in [(0.2, 0.0), (0.8, 0.1)]:
            values = np.arange(500) / 1000 + offset
            groups.append(pd.DataFrame(dict(T=1024, method='structure_ols',
                train_seed=-1, process='fGn', H=h, p_mrw=values,
                lambda2_proj=values)))
        frame = pd.concat(groups, ignore_index=True)
        _, frozen = calibrate(frame, 0.05)
        self.assertEqual(len(frozen), 2)
        for record in frozen:
            self.assertAlmostEqual(record['threshold'], 0.575)
            self.assertEqual(float.fromhex(record['threshold_hex']), record['threshold'])
        frame.loc[0, 'process'] = 'MRW'
        with self.assertRaises(AssertionError):
            calibrate(frame, 0.05)

    def test_constant_target_correlation_is_undefined(self):
        self.assertTrue(np.isnan(corr(np.ones(10), np.arange(10.0))))


if __name__ == '__main__':
    unittest.main(verbosity=2)
