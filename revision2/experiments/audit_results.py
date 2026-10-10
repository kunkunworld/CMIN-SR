"""Independent arithmetic, full neural replay and frozen-threshold verification."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from common import *
import argparse
import torch
from revision.retraining import make_model
from revision.analysis_tools import FrozenCalibrator


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,default=HERE/'results/verification')
    out=ap.parse_args().output
    before=start(out,dict(experiment='Independent post-run checks',no_training=True))
    torch.set_num_threads(1)
    paths=[HERE/'results'/s for s in ['E1_formal','E2_E3_formal','E4_formal']]
    for p in paths:
        assert json.loads((p/'status.json').read_text())['status']=='COMPLETED'
        for source,h in json.loads((p/'protected_before.json').read_text()).items():assert sha(ROOT/source)==h
        if (p/'input_hashes.json').exists():
            for source,h in json.loads((p/'input_hashes.json').read_text()).items():assert sha(ROOT/source)==h
    rows=[];summ=pd.read_csv(paths[1]/'all_metrics_intervals.csv')
    runs=pd.read_csv(ROOT/'revision/results/neural_retrain_formal_attempt3/training_runs.csv')
    for T in [512,1024,2048]:
        source=ROOT/'revision/results/neural_eval_formal'
        df=pd.read_csv(paths[1]/f'common_sample_predictions_T{T}.csv')
        q=Q;v=-.5*q*(q-2);a=q@q;b=q@v;c=v@v
        # Two-by-two normal equations, independent of the production lstsq call.
        z=df[[f'zeta_{i}' for i in range(6)]].to_numpy()
        signed=(a*(z@v)-b*(z@q))/(a*c-b*b)
        assert np.max(np.abs(signed-df.lambda2_signed))<1e-12
        assert np.max(np.abs(np.maximum(signed,1e-6)-df.lambda2_proj))<1e-12
        target=df.H.to_numpy()[:,None]*q-.5*df.lambda2_true.to_numpy()[:,None]*q*(q-2)
        assert np.max(np.abs(np.mean(np.abs(z-target),axis=1)-df.zeta_mae))<1e-12
        for method,g in df[df.process=='MRW'].groupby('method'):
            values=[]
            for seed,d in g.groupby('train_seed'):
                assert len(d)==800 and d.sample_id.nunique()==800
                values.append([d.zeta_mae.mean(),np.mean(np.abs(d.lambda2_signed-d.lambda2_true)),
                    np.mean(np.abs(d.lambda2_proj-d.lambda2_true)),np.corrcoef(d.lambda2_true,d.lambda2_signed)[0,1]])
            means=np.mean(values,axis=0)
            for metric,mean in zip(['spectrum_mae','signed_mae','clipped_mae','signed_corr'],means):
                r=summ[(summ['T']==T)&(summ.group=='MRW_all')&(summ.method==method)&(summ.metric==metric)]
                assert len(r)==1 and abs(float(r.iloc[0]['mean'])-mean)<1e-12
        x=np.load(source/f'test_T{T}.npz')['x'];meta=pd.read_csv(source/f'test_T{T}_manifest.csv')
        for seed in [51001,51002,51003]:
            r=runs[(runs.seed==seed)&(runs.config=='full_c025')].iloc[0]
            cp=ROOT/'revision/results/neural_retrain_formal_attempt3'/r.checkpoint
            assert sha(cp)==r.checkpoint_sha256
            model=make_model(seed,.25,'cuda')
            model.load_state_dict(torch.load(cp,map_location='cpu')['model_state_dict'],strict=True);model.eval()
            with torch.no_grad():
                replay=np.concatenate([model(torch.tensor(x[i:i+64],device='cuda'))['zeta_emp'].cpu().numpy()
                                       for i in range(0,len(x),64)])
            archived=df[(df.method=='full_c025')&(df.train_seed==seed)].set_index('sample_id').loc[meta.sample_id]
            err=float(np.max(np.abs(replay-archived[[f'zeta_{i}' for i in range(6)]].to_numpy())))
            assert err<1e-12
            rows.append(dict(T=T,train_seed=seed,independent_sequences=len(x),max_error=err))
            del model
        print('independent replay and arithmetic',T,'passed',flush=True)
    pd.DataFrame(rows).to_csv(out/'all_sample_neural_replay.csv',index=False)
    # Recompute the important nonwhite rejection rates without the E4 summary code.
    th=pd.read_csv(ROOT/'revision/results/neural_summary_formal/heldout_roc_thresholds.csv')
    checks=[];cal=FrozenCalibrator()
    for T in [512,1024,2048]:
        d=pd.read_csv(paths[2]/f'predictions_T{T}.csv')
        declared=pd.read_csv(paths[2]/'threshold_and_auc_by_seed.csv')
        for (h,method,seed),g in d.groupby(['H','method','train_seed']):
            for score in ['p_mrw','lambda2_proj']:
                r=th[(th['T']==T)&(th.method==method)&(th.train_seed==seed)&(th.score==score)].iloc[0]
                count=sum(float(x)>=float(r.threshold) for x in g[score])
                expected=declared[(declared['T']==T)&(declared.H_null==h)&(declared.method==method)&
                    (declared.train_seed==seed)&(declared.score==score)&(declared.metric=='FPR')].iloc[0]
                assert abs(count/len(g)-expected.value)<1e-12
                if method=='full_c025' and h==.8:
                    z=g.iloc[0][[f'zeta_{i}' for i in range(6)]].to_numpy(dtype=float)
                    recomputed=cal.scores(z)['p_mrw']
                    assert abs(recomputed-g.iloc[0].p_mrw)<1e-6
                    checks.append(dict(T=T,H=h,train_seed=seed,score=score,threshold=r.threshold,
                        positive_count=count,independent_N=len(g),rate=count/len(g),
                        score_median=g[score].median(),score_min=g[score].min(),score_max=g[score].max()))
    pd.DataFrame(checks).to_csv(out/'nonwhite_H08_manual_counts.csv',index=False)
    linear=[]
    for h in [.2,.4,.5,.6,.8]:linear.append(dict(H=h,**cal.scores(Q*h)))
    pd.DataFrame(linear).to_csv(out/'exact_linear_geometry_scores.csv',index=False)
    finish(out,before,formal_inputs_verified=True,closed_form_projection_verified=True,
        full_neural_replay_predictions=12600,unique_replayed_sequences=4200,
        nonwhite_threshold_counts_verified=True,
        note='No scientific pass/fail based on whether results favor the neural model.')


if __name__=='__main__':main()
