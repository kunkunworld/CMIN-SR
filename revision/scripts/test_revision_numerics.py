"""Independent contract checks for the revision-only implementations."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import unittest
import numpy as np
import torch
from revision.generation import generate_pair,standardize,sample_gaussian
from revision.analysis_tools import Q,structure_zeta,projection_metrics
from revision.scripts.run_real_surrogate_inference import batched_proxy,holm
from revision.scripts.summarize_corrected_experiments import corr_rows
from mrw_dl.generation import MRWParams,generate_mrw_fgn
from mrw_inverse.models.empirical_spectrum import estimate_empirical_spectrum
from mrw_inverse.models.mrw_projection import MRWProjection
from mrw_inverse.proxy import proxy_inverse_estimate


class NumericalContracts(unittest.TestCase):
    def test_legacy_pair_reproduces_original(self):
        for h in [.2,.5,.8]:
            old=generate_mrw_fgn(MRWParams(length=512,H=h,lambda2=.1,L=128,seed=91))['dx']
            pair=generate_pair(512,h,.1,91)
            np.testing.assert_allclose(old,pair['legacy'],rtol=1e-6,atol=1e-9)

    def test_ols_matches_existing_valid_estimator(self):
        x=np.random.default_rng(4).normal(size=1024)
        old=estimate_empirical_spectrum(torch.tensor(x[None,:],dtype=torch.float64)).zeta_emp.numpy()[0]
        np.testing.assert_allclose(structure_zeta(x),old,rtol=1e-8,atol=1e-8)

    def test_projection_coordinates(self):
        for z in [Q*.5, Q*.5-.1/2*Q*(Q-2), Q*.5+.05*Q*(Q-2)]:
            old=MRWProjection()(torch.tensor(z[None,:],dtype=torch.float64))
            new=projection_metrics(z)
            self.assertAlmostEqual(old.H_proj.item(),new['H_proj'],places=6)
            self.assertAlmostEqual(old.lambda2_proj.item(),new['lambda2_proj'],places=6)
            self.assertAlmostEqual(old.residual_norm.item(),new['mrw_residual'],places=6)

    def test_proxy_batch_matches(self):
        x=np.random.default_rng(2).standard_t(3.5,(4,512))*.01
        ref=np.array([[s.pred_lambda2,s.p_MRW,s.f_alpha_width] for s in map(proxy_inverse_estimate,x)])
        np.testing.assert_allclose(batched_proxy(x),ref,rtol=1e-10,atol=1e-10)

    def test_holm_known_example(self):
        np.testing.assert_allclose(holm([.01,.04,.03]),[.03,.06,.06])

    def test_constant_correlation_undefined(self):
        self.assertTrue(np.isnan(corr_rows(np.array([[0.,1.,2.]]),np.ones((1,3)))[0]))

    def test_bad_covariance_rejected(self):
        with self.assertRaises(ValueError):sample_gaussian(np.array([1.,10.,0.,0.]),np.random.default_rng(1))

    def test_standardized_zero_lambda_invariant(self):
        p=generate_pair(512,.5,0.,1)
        np.testing.assert_allclose(standardize(p['legacy']),standardize(p['corrected']),atol=1e-13)


if __name__=='__main__':
    torch.set_num_threads(1)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(NumericalContracts))
    raise SystemExit(0 if result.wasSuccessful() else 2)
