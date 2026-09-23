"""OOD stress check of primary retrained networks; no invented clean targets."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import argparse,json,time
import numpy as np
import pandas as pd
import torch
from revision.retraining import make_model,prior_and_ols,finite
from revision.analysis_tools import FrozenCalibrator,projection_metrics,dump_json,hashfile
from mrw_inverse.models.empirical_spectrum import estimate_empirical_spectrum


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--training-dir',type=Path,required=True)
    ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--output-dir',type=Path,required=True);args=ap.parse_args()
    if json.loads((args.training_dir/'status.json').read_text())['status']!='PASS':raise SystemExit('Training not PASS')
    out=args.output_dir;out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);device=args.device;cal=FrozenCalibrator()
    runs=pd.read_csv(args.training_dir/'training_runs.csv');runs=runs[runs.config=='full_c025']
    trainmeta=pd.read_csv(args.training_dir/'data_manifest.csv');rows=[];inputhash={}
    for T in [512,1024,2048]:
        p=ROOT/f'revision/results/stress_controls/stress_inputs_T{T}.npz';inputhash[str(p)]=hashfile(p)
        x=torch.tensor(np.load(p)['x']);meta=pd.read_csv(ROOT/f'revision/results/stress_controls/stress_manifest_T{T}.csv')
        if set(meta.data_seed)&set(trainmeta.data_seed):raise AssertionError('Stress/training overlap')
        ref=make_model(93001,0,device);prior,ols=prior_and_ols(ref,x,device);del ref
        arrays=[('structure_ols',-1,ols),('nonoverlap_prior',-1,prior)];tail=[];quality=[]
        with torch.no_grad():
            for start in range(0,len(x),64):
                emp=estimate_empirical_spectrum(x[start:start+64]);tail.extend(emp.tail_instability[:,0].tolist())
                quality.extend(emp.scaling_fit_quality[:,0].tolist())
        for r in runs.itertuples():
            ridge=np.load(args.training_dir/f'seed_{r.seed}/ridge.npz')
            rz=((np.c_[ols,prior]-ridge['mean'])/ridge['scale'])@ridge['coef'].T+ridge['intercept']
            arrays.append(('ridge',r.seed,rz))
            state=torch.load(args.training_dir/r.checkpoint,map_location='cpu')
            model=make_model(r.seed,r.coefficient,device);model.load_state_dict(state['model_state_dict'],strict=True);model.eval();zs=[]
            with torch.no_grad():
                for start in range(0,len(x),64):zs.append(model(x[start:start+64].to(device))['zeta_emp'].cpu().numpy())
            arrays.append(('full_c025',r.seed,np.concatenate(zs)));del model
        metadata=meta.to_dict('records')
        for method,seed,zs in arrays:
            finite(method,zs)
            for i,(m,z) in enumerate(zip(metadata,zs)):
                scores=cal.scores(z);finite('scores',np.array(list(scores.values())))
                rows.append({**m,'method':method,'train_seed':seed,**projection_metrics(z),**scores,
                             'raw_tail_instability':tail[i],'raw_scaling_fit_quality':quality[i]})
        print(f'retrained stress T={T} complete',flush=True)
    df=pd.DataFrame(rows);df.to_csv(out/'sample_results.csv',index=False)
    metrics=['p_mrw','p_curved','p_mono','p_boundary','lambda2_proj','raw_tail_instability','raw_scaling_fit_quality']
    df.groupby(['T','process','method','train_seed','seed'])[metrics].mean().to_csv(out/'by_seed.csv')
    # Do not inflate N by counting repeated network predictions as independent sequences.
    reduced=df.groupby(['T','process','method','sample_id','seed'])[metrics].mean().reset_index()
    records=[]
    for (T,process,method),d in reduced.groupby(['T','process','method']):
        rng=np.random.default_rng(95501);d=d.reset_index(drop=True)
        idx=np.concatenate([rng.choice(v,(2000,len(v))) for v in d.groupby('seed').indices.values()],axis=1)
        for metric in metrics:
            values=d[metric].to_numpy();lo,hi=np.quantile(values[idx].mean(axis=1),[.025,.975])
            records.append(dict(T=T,process=process,method=method,metric=metric,mean=values.mean(),
                                ci_low=lo,ci_high=hi,independent_N=len(d),
                                ci_scope='data uncertainty of mean output across fixed three trained networks'))
    pd.DataFrame(records).to_csv(out/'summary_intervals.csv',index=False)
    dump_json(out/'manifest.json',dict(raw_inputs=inputhash,calibrator=cal.manifest,
        labels='No clean zeta or true intermittency labels for stress controls',
        iid_mixture='Independent state draws; not a Markov regime-switching process'))
    dump_json(out/'status.json',dict(status='PASS',rows=len(df),independent_N=len(meta)*3,
        scope='OOD scores and raw warnings only, not verified probabilities'))


if __name__=='__main__':main()
