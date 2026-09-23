"""Apply already frozen clean-control thresholds to OOD stress, without retuning."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
import pandas as pd
from revision.analysis_tools import dump_json


def main():
    base=ROOT/'revision/results';out=base/'stress_operating_thresholds';out.mkdir(parents=True,exist_ok=False)
    d=pd.read_csv(base/'retrained_stress/sample_results.csv')
    thresholds=pd.read_csv(base/'neural_summary_formal/heldout_roc_thresholds.csv');rows=[]
    for (T,method,seed,process),g in d.groupby(['T','method','train_seed','process']):
        for score in ['p_mrw','lambda2_proj']:
            threshold=thresholds[(thresholds['T']==T)&(thresholds.method==method)&
                (thresholds.train_seed==seed)&(thresholds.score==score)]
            if len(threshold)!=1:raise AssertionError('Missing/ambiguous threshold')
            t=float(threshold.iloc[0].threshold);g=g.reset_index(drop=True);rates=(g[score]>=t).to_numpy(dtype=float)
            rng=np.random.default_rng(95601)
            ix=np.concatenate([rng.choice(v,(2000,len(v))) for v in g.groupby('seed').indices.values()],axis=1)
            lo,hi=np.quantile(rates[ix].mean(axis=1),[.025,.975])
            rows.append(dict(T=T,method=method,train_seed=seed,process=process,score=score,threshold=t,
                positive_decision_rate=rates.mean(),ci_low=lo,ci_high=hi,N=len(g),
                meaning='OOD positive decisions; not an MRW-mechanism test',threshold_source='same-H fGn validation'))
    df=pd.DataFrame(rows);df.to_csv(out/'by_seed.csv',index=False)
    df.groupby(['T','method','process','score'])[['positive_decision_rate']].agg(['mean','std']).to_csv(out/'summary.csv')
    dump_json(out/'status.json',dict(status='PASS',threshold_refit=False,
        scope='Clean fGn FPR calibration does not imply OOD specificity'))


if __name__=='__main__':main()
