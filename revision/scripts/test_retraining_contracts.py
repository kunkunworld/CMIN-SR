"""Independent numerical gates before corrected-data neural experiments."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import json,unittest
import numpy as np
import torch
from scipy.linalg import toeplitz,cholesky
from mrw_dl.generation import build_fgn_covariance
from revision.retraining import sample_fgn_full_embedding,make_model,make_training,finite
from revision.analysis_tools import structure_zeta,Q,SCALES,dump_json

EVIDENCE=[]


class Contracts(unittest.TestCase):
    def test_covariance_independent_cholesky(self):
        n=64;reps=6000
        for h in [.1,.5,.8,.9]:
            cov=build_fgn_covariance(n,h);rng=np.random.default_rng(73301)
            fft=np.stack([sample_fgn_full_embedding(n,h,rng) for _ in range(reps)])
            direct=np.random.default_rng(73302).normal(size=(reps,n))@cholesky(toeplitz(cov),lower=True).T
            for name,data in [('fft',fft),('cholesky',direct)]:
                for lag in [0,1,4,16]:
                    values=np.mean(data[:,:n-lag]*data[:,lag:],axis=1)
                    error=abs(values.mean()-cov[lag]);se=values.std(ddof=1)/np.sqrt(reps)
                    EVIDENCE.append(dict(H=h,method=name,lag=lag,target=cov[lag],empirical=values.mean(),mc_se=se))
                    self.assertLess(error,6*se)

    def test_full_embedding_covariance_identity(self):
        # Direct unitary DFT matrix, not inverse FFT sampling, at a small n.
        n=12;m=2*n;k=np.arange(m);F=np.exp(2j*np.pi*np.outer(k,k)/m)/np.sqrt(m)
        for h in [.1,.5,.9]:
            cov=build_fgn_covariance(n+1,h);eig=np.fft.fft(np.r_[cov,cov[1:n][::-1]]).real
            A=F*np.sqrt(eig)[None,:]
            actual=A.real@A.real.T+A.imag@A.imag.T
            np.testing.assert_allclose(actual[:n,:n],toeplitz(cov[:n]),atol=1e-13)

    def test_psd_entire_h_grid(self):
        for n in [512,1024,2048]:
            for h in np.linspace(.1,.9,81):
                c=build_fgn_covariance(n+1,h)
                self.assertGreaterEqual(np.fft.fft(np.r_[c,c[1:n][::-1]]).real.min(),0.)

    def test_ols_direct_increment_sums(self):
        x=np.random.default_rng(9).normal(size=512)
        logs=[]
        for s in SCALES:
            sums=np.array([sum(x[i+1:i+s+1]) for i in range(len(x)-s)])
            logs.append(np.log(np.mean(np.abs(sums[:,None])**Q,axis=0)))
        z=np.polyfit(np.log(SCALES),np.array(logs),1)[0]
        np.testing.assert_allclose(z,structure_zeta(x),atol=1e-13)

    def test_zero_head_prior_and_frozen_validity(self):
        model=make_model(91,.5,'cpu').eval();x=torch.randn(4,512)
        with torch.no_grad():
            torch.testing.assert_close(model(x)['zeta_emp'],model.structure_branch(x)['empirical_zeta_estimate'],rtol=0,atol=0)
        for name,p in model.named_parameters():
            if name.startswith(('p_','stability_head')):self.assertFalse(p.requires_grad)

    def test_independent_splits(self):
        a,ma=make_training(20,512,93,1);b,mb=make_training(20,512,93,2)
        self.assertFalse(set(ma.data_seed)&set(mb.data_seed))
        self.assertFalse(set(ma.data_seed)&set(ma.parameter_seed))
        self.assertFalse(set(mb.data_seed)&set(mb.parameter_seed))
        self.assertTrue((ma.loc[ma.process=='MRW','H']==.5).all())
        self.assertEqual(a['target'].shape,(20,6))

    def test_fail_closed_nonfinite(self):
        with self.assertRaises(FloatingPointError):finite('test',np.array([np.nan]))


if __name__=='__main__':
    torch.set_num_threads(1)
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    out=ROOT/'revision/results/retraining_contracts';out.mkdir(exist_ok=True,parents=True)
    dump_json(out/'status.json',{'status':'PASS' if r.wasSuccessful() else 'FAIL','tests':r.testsRun,'covariance_checks':EVIDENCE})
    raise SystemExit(0 if r.wasSuccessful() else 2)
