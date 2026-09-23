"""E4 reference check: coefficient clipping is not constrained least squares."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from revision.analysis_tools import Q,dump_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--evaluation-dir',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True);a=ap.parse_args();out=a.output_dir;out.mkdir(parents=True,exist_ok=False)
    X=np.c_[Q,-.5*Q*(Q-2)];lo,hi,eps=1e-4,1-1e-4,1e-6;rows=[]
    for path in a.evaluation_dir.glob('results_test_T*.csv'):
        d=pd.read_csv(path);z=d[[f'zeta_{i}' for i in range(6)]].to_numpy();b=z@np.linalg.pinv(X).T
        clipped=np.c_[np.clip(b[:,0],lo,hi),np.maximum(b[:,1],eps)]
        candidates=[clipped]
        candidates.append(np.c_[np.clip((z-eps*X[:,1])@Q/(Q@Q),lo,hi),np.full(len(z),eps)])
        for h in [lo,hi]:candidates.append(np.c_[np.full(len(z),h),np.maximum((z-h*Q)@X[:,1]/(X[:,1]@X[:,1]),eps)])
        c=np.stack(candidates);err=((c@X.T-z)**2).sum(axis=2);best=err.argmin(axis=0);exact=c[best,np.arange(len(z))]
        for i in np.linspace(0,len(z)-1,30,dtype=int):
            ref=lsq_linear(X,z[i],bounds=([lo,eps],[hi,np.inf]),tol=1e-13,lsmr_tol=1e-13)
            np.testing.assert_allclose(exact[i],ref.x,atol=1e-5)
        assert np.all(((clipped@X.T-z)**2).sum(axis=1)+1e-14>=((exact@X.T-z)**2).sum(axis=1))
        result=d[['sample_id','T','process','H','lambda2_true','method','train_seed']].copy()
        result['clipped_H']=clipped[:,0];result['constrained_LS_H']=exact[:,0]
        result['clipped_lambda2']=clipped[:,1];result['constrained_LS_lambda2']=exact[:,1]
        result['clipped_rmse']=np.sqrt(np.mean((clipped@X.T-z)**2,axis=1))
        result['constrained_LS_rmse']=np.sqrt(np.mean((exact@X.T-z)**2,axis=1))
        result['coordinate_diff']=np.max(np.abs(exact-clipped),axis=1)
        rows.append(result)
    d=pd.concat(rows);d.to_csv(out/'sample_results.csv',index=False)
    d['differs']=d.coordinate_diff>1e-5
    d.groupby(['T','process','method'])[['coordinate_diff','differs','clipped_rmse','constrained_LS_rmse']].mean().to_csv(out/'summary.csv')
    (out/'interpretation.md').write_text('# Projection boundary check\n\n'
        'The historical decoder first fits unconstrained least squares, then clamps coordinates.\n'
        'It is not generally the exact least-squares projection onto the constrained parameter set.\n'
        'This audit compares the same parameter bounds against an active-boundary solution,\n'
        'independently checked with scipy.optimize.lsq_linear. It does NOT replace training,\n'
        'calibrator features or primary metrics. Report the original operation as clipped\n'
        'least-squares coordinates, not a globally optimal constrained projection.\n'
        'The displayed historical residual is MAE even though coefficients are L2-fitted;\n'
        'a lower LS residual does not establish an optimal MAE fit or mechanism identification.\n',encoding='utf-8')
    dump_json(out/'status.json',{'status':'PASS','independent_scipy_checks':90,'scope':'Numerical semantics; no primary method replacement'})


if __name__=='__main__':main()
