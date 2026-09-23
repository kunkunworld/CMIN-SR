"""Fresh paired holdouts, validation-only thresholds and corrected neural outputs."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import argparse,json,time,traceback
import numpy as np
import pandas as pd
import torch
from revision.retraining import simulate,make_model,finite,prior_and_ols
from revision.analysis_tools import Q,SCALES,projection_metrics,FrozenCalibrator,dump_json,hashfile,protected_hashes
from mrw_inverse.analysis.classical_multifractal_estimators import estimate_classical_zeta
from revision.linear_control import logvol_features
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


def holdout(T,split,quick):
    seeds=([81999] if quick else list(range(81001,81006))) if split=='test' else ([82999] if quick else list(range(82001,82004)))
    n=2 if quick else (20 if split=='test' else 10)
    levels=[0.,.005,.01,.03,.06,.1,.15,.2];xs=[];rows=[]
    for seed in seeds:
        specs=[('MRW',.5,lam,n) for lam in levels]
        specs += [('fGn',.5,0.,2*n),('Gaussian',.5,0.,2*n)]
        specs += [('fGn',h,0.,max(1,n//2)) for h in [.2,.4,.6,.8]]
        for group,(process,h,lam,count) in enumerate(specs):
            for i in range(count):
                sd=int(np.random.SeedSequence([602619,3 if split=='test' else 4,seed,T,group,i]).generate_state(1)[0])
                xs.append(simulate(T,process,h,lam,sd))
                rows.append(dict(sample_id=f'{split}_{seed}_{T}_{group}_{i}',data_seed=sd,seed=seed,
                    split=split,T=T,process=process,H=h,lambda2_true=lam,
                    lambda_band=('low' if lam<.03 else 'medium' if lam<.08 else 'high') if process=='MRW' else 'mono'))
    return torch.tensor(np.stack(xs)),pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--quick',action='store_true')
    ap.add_argument('--training-dir',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--device',default='cuda');args=ap.parse_args()
    train=args.training_dir
    if json.loads((train/'status.json').read_text())['status']!='PASS':raise SystemExit('Training not complete/PASS')
    out=args.output_dir;out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    runs=pd.read_csv(train/'training_runs.csv');before=protected_hashes();cal=FrozenCalibrator();t0=time.time()
    dump_json(out/'manifest.json',dict(training_manifest_sha256=hashfile(train/'manifest.json'),
        protocol_sha256=hashfile(ROOT/'revision/NEURAL_RETRAIN_PROTOCOL.md'),script_sha256=hashfile(Path(__file__)),
        primary='full_c025',test_selection=False,calibrator=cal.manifest,quick=args.quick,
        linear_control_amendment_sha256=hashfile(ROOT/'revision/LINEAR_CONTROL_AMENDMENT.md')))
    dump_json(out/'status.json',{'status':'RUNNING'})
    try:
        linear={}
        for seed in runs.seed.unique():
            reference=make_model(90001,0.,args.device);features=[];targets=[]
            for T in [512,1024]:
                ds=np.load(train/f'seed_{seed}/data_T{T}.npz');x=torch.tensor(ds['train_x'])
                pr,ol=prior_and_ols(reference,x,args.device)
                features.append(np.c_[ol,pr,logvol_features(x)]);targets.append(ds['train_target'])
            features=np.concatenate(features);scaler=StandardScaler().fit(features)
            ridge=Ridge(alpha=.001).fit(scaler.transform(features),np.concatenate(targets))
            linear[int(seed)]=dict(mean=scaler.mean_,scale=scaler.scale_,coef=ridge.coef_,intercept=ridge.intercept_)
            np.savez(out/f'ridge_logvol_seed{seed}.npz',**linear[int(seed)]);del reference
        allmeta=[];groups={};Ts=[512] if args.quick else [512,1024,2048]
        for split in ['threshold_validation','test']:
            for T in Ts:
                x,meta=holdout(T,split,args.quick);groups[split,T]=(x,meta);allmeta.append(meta)
                np.savez_compressed(out/f'{split}_T{T}.npz',x=x.numpy())
                meta.to_csv(out/f'{split}_T{T}_manifest.csv',index=False)
        meta_all=pd.concat(allmeta,ignore_index=True)
        if meta_all.data_seed.duplicated().any():raise AssertionError('Holdout seed collision')
        training_meta=pd.read_csv(train/'data_manifest.csv')
        if set(meta_all.data_seed)&set(training_meta.data_seed):raise AssertionError('Training/holdout seed overlap')
        old_seeds=set()
        for folder in ['corrected_recovery_formal','corrected_recovery_validation']:
            for f in (ROOT/'revision/results'/folder).glob('raw_T*_manifest.csv'):
                old_seeds.update(pd.read_csv(f).data_seed)
        if set(meta_all.data_seed)&old_seeds:raise AssertionError('New/old evaluation seed overlap')
        dump_json(out/'split_check.json',dict(status='PASS',new_sequences=len(meta_all),train_val_sequences=len(training_meta),
            train_holdout_overlap=0,old_evaluation_overlap=0,test_threshold_overlap=0))
        for (split,T),(x,meta) in groups.items():
            reference=make_model(90001,0.,args.device);prior,ols=prior_and_ols(reference,x,args.device);del reference
            arrays=[('structure_ols',-1,ols),('nonoverlap_prior',-1,prior)]
            for method in ['mfdfa','mfdfa_quadratic']:
                z=np.stack([estimate_classical_zeta(y.numpy().astype(float),tuple(Q),SCALES,method).zeta for y in x])
                arrays.append((method,-1,z))
            for seed in runs.seed.unique():
                st=np.load(train/f'seed_{seed}/ridge.npz')
                features=(np.c_[ols,prior]-st['mean'])/st['scale']
                arrays.append(('ridge',int(seed),features@st['coef'].T+st['intercept']))
                st=linear[int(seed)]
                features=(np.c_[ols,prior,logvol_features(x)]-st['mean'])/st['scale']
                arrays.append(('ridge_logvol',int(seed),features@st['coef'].T+st['intercept']))
            for r in runs.itertuples():
                cp=train/r.checkpoint
                if hashfile(cp)!=r.checkpoint_sha256:raise AssertionError('Checkpoint hash mismatch')
                state=torch.load(cp,map_location='cpu');model=make_model(int(r.seed),float(r.coefficient),args.device)
                model.load_state_dict(state['model_state_dict'],strict=True);model.eval();zs=[]
                with torch.no_grad():
                    for s in range(0,len(x),64):
                        z=model(x[s:s+64].to(args.device))['zeta_emp'];finite('eval zeta',z);zs.append(z.cpu().numpy())
                arrays.append((r.config,int(r.seed),np.concatenate(zs)));del model
            rows=[];metadata=meta.to_dict('records')
            for method,train_seed,zs in arrays:
                finite(method,zs)
                for info,z in zip(metadata,zs):
                    target=Q*info['H']-.5*info['lambda2_true']*Q*(Q-2)
                    d={**info,'method':method,'train_seed':train_seed,
                        **projection_metrics(z,target,info['H'],info['lambda2_true']),**cal.scores(z)}
                    rows.append(d)
            df=pd.DataFrame(rows);numeric=df.select_dtypes(include='number')
            if not np.isfinite(numeric.to_numpy()).all():raise AssertionError('Nonfinite clean evaluation metric')
            df.to_csv(out/f'results_{split}_T{T}.csv',index=False)
            print(f'{split} T={T}: {len(x)} independent sequences, {len(arrays)} method/seed outputs; elapsed={time.time()-t0:.0f}s',flush=True)
        if before!=protected_hashes():raise AssertionError('Protected originals changed')
        dump_json(out/'status.json',dict(status='PASS',elapsed_seconds=time.time()-t0,originals_unchanged=True,
            no_nonfinite=True,independent_test_N=int((meta_all.split=='test').sum())))
    except Exception:
        dump_json(out/'status.json',dict(status='FAIL',traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
