"""Separate recalculation of frozen quantiles, raw replay and operating counts."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from common import *
import torch
from revision.retraining import make_model, simulate
from revision.analysis_tools import FrozenCalibrator


def main():
    torch.set_num_threads(1)
    cal=HERE/'results/E5_calibration_formal';test=HERE/'results/E5_test_formal'
    out=HERE/'results/E5_verification';before=start(out,dict(experiment='E5 independent verification'))
    try:
        for folder in [cal,test]:
            assert json.loads((folder/'status.json').read_text())['status']=='COMPLETED'
            for p,h in json.loads((folder/'input_hashes.json').read_text()).items():assert sha(ROOT/p)==h
        records=json.loads((cal/'frozen_thresholds.json').read_text())
        thresholds={(r['T'],r['method'],r['train_seed'],r['score']):float.fromhex(r['threshold_hex']) for r in records}
        assert sha(cal/'frozen_thresholds.json')==json.loads((cal/'status.json').read_text())['thresholds_sha256']
        old=pd.read_csv(ROOT/'revision/results/neural_summary_formal/heldout_roc_thresholds.csv',float_precision='round_trip')
        declared=pd.read_csv(test/'rates_by_checkpoint.csv',float_precision='round_trip')
        summary=pd.read_csv(test/'rates_summary.csv',float_precision='round_trip')
        train=ROOT/'revision/results/neural_retrain_formal_attempt3';runs=pd.read_csv(train/'training_runs.csv')
        seeds_all=set();ids_all=set();logs=[];comparisons=0;replayed=0;zcols=[f'zeta_{i}' for i in range(6)]
        for folder in [cal,test]:
            for T in [512,1024,2048]:
                data=np.load(folder/f'raw_T{T}.npz')['x'];meta=pd.read_csv(folder/f'raw_T{T}_manifest.csv',dtype={'data_seed':str})
                assert len(meta)==len(data) and meta.sample_id.nunique()==len(meta)
                ids=set(meta.sample_id);seeds=set(int(v) for v in meta.data_seed)
                assert not ids&ids_all and not seeds&seeds_all and len(seeds)==len(meta)
                ids_all.update(ids);seeds_all.update(seeds)
                df=pd.read_csv(folder/f'predictions_T{T}.csv',float_precision='round_trip')
                assert df.groupby(['method','train_seed']).sample_id.nunique().eq(len(meta)).all()
                # Regenerate one raw series per clean parameter group, including all H.
                selected=meta[meta.process.isin(['fGn','MRW'])].groupby(['process','H','lambda2_true']).head(1)
                for i,r in selected.iterrows():
                    x=simulate(T,r.process,r.H,r.lambda2_true,int(r.data_seed))
                    np.testing.assert_array_equal(x,data[i])
                for ts in [51001,51002,51003]:
                    r=runs[(runs.seed==ts)&(runs.config=='full_c025')].iloc[0];cp=train/r.checkpoint
                    assert sha(cp)==r.checkpoint_sha256
                    model=make_model(ts,.25,'cuda')
                    model.load_state_dict(torch.load(cp,map_location='cpu')['model_state_dict'],strict=True);model.eval()
                    with torch.no_grad():
                        z=np.concatenate([model(torch.tensor(data[i:i+64],device='cuda'))['zeta_emp'].cpu().numpy()
                                          for i in range(0,len(data),64)])
                    archived=df[(df.method=='full_c025')&(df.train_seed==ts)].set_index('sample_id').loc[meta.sample_id]
                    err=np.max(np.abs(z-archived[zcols].to_numpy()))
                    assert err<1e-12;replayed+=len(data)
                    logs.append(dict(stage=folder.name,T=T,train_seed=ts,N=len(data),max_abs_error=float(err)))
                    del model
                if folder==cal:
                    for (method,ts),g in df.groupby(['method','train_seed']):
                        for score in ['p_mrw','lambda2_proj']:
                            groupcuts=[]
                            for h,gh in g.groupby('H'):
                                assert len(gh)==500
                                # With n=500 and alpha=.05, the specified one-based rank is 476.
                                groupcuts.append(float(np.partition(gh[score].to_numpy(),475)[475]))
                            expected=max(groupcuts)
                            assert expected.hex()==thresholds[T,method,ts,score].hex()
                else:
                    for r in declared[declared['T']==T].itertuples():
                        g=df[(df.method==r.method)&(df.train_seed==r.train_seed)]
                        if r.group.startswith('fGn_H'):g=g[(g.process=='fGn')&(g.H==float(r.group[5:]))]
                        elif r.group=='MRW_significant':g=g[(g.process=='MRW')&(g.lambda2_true>=.06)]
                        elif r.group.startswith('MRW_lambda2_'):g=g[(g.process=='MRW')&(g.lambda2_true==float(r.group[12:]))]
                        else:g=g[g.process==r.group]
                        if r.rule=='multiH':
                            cut=thresholds[T,r.method,r.train_seed,r.score];k=sum(float(v)>cut for v in g[r.score])
                        else:
                            cut=float(old[(old['T']==T)&(old.method==r.method)&(old.train_seed==r.train_seed)&(old.score==r.score)].iloc[0].threshold)
                            k=sum(float(v)>=cut for v in g[r.score])
                        assert k==r.positive_count and len(g)==r.independent_N
                        assert abs(k/len(g)-r.rate)<1e-14;comparisons+=1
                print('E5 independent verification',folder.name,T,'passed',flush=True)
        for key,g in declared.groupby(['T','method','group','score','rule']):
            T,method,group,score,rule=key
            row=summary[(summary['T']==T)&(summary.method==method)&(summary.group==group)&(summary.score==score)&(summary.rule==rule)]
            assert len(row)==1 and abs(row.iloc[0]['mean']-g.rate.mean())<1e-14
        # Tie conventions, including all-negative signed coordinates mapped to the floor.
        assert sum(v>1e-6 for v in [1e-6]*500)==0
        assert sum(v>=np.nextafter(1e-6,np.inf) for v in [1e-6]*500)==0
        pd.DataFrame(logs).to_csv(out/'complete_neural_replay.csv',index=False)
        finish(out,before,quantile_recalculation=True,strict_threshold_ties_verified=True,
               threshold_counts_recalculated=comparisons,independent_sequences=len(ids_all),
               neural_predictions_replayed=replayed,data_seed_overlap=0)
    except Exception:
        import traceback
        dump(out/'status.json',dict(status='FAILED',traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
