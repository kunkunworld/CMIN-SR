"""q-grid sensitivity and fixed-integral-scale controls; no learned heads."""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import argparse
import numpy as np
import pandas as pd
from revision.analysis_tools import Q,SCALES,structure_zeta,projection_metrics,safe_corr,dump_json
from revision.generation import generate_pair,standardize


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input-dir',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True);args=ap.parse_args()
    out=args.output_dir;out.mkdir(parents=True,exist_ok=False)
    grids={'Q1':Q,'Q2':np.arange(.25,3.01,.25),'Q3':np.arange(.5,2.01,.5),
           'Q4':np.arange(.25,2.01,.25),'Q5':np.arange(1.,3.01,.5)}
    dump_json(out/'manifest.json',dict(q_grids={k:v.tolist() for k,v in grids.items()},scales=SCALES,
        data_selection='All corrected canonical MRW and fGn/Gaussian from archived paired samples',
        fixedL='512, all archived canonical MRW seeds, matched random innovations at each T',
        warning='More q values are dependent evaluations on the same series, not independent observations'))
    rows=[];Lrows=[]
    for T in [512,1024,2048,4096,8192]:
        data=np.load(args.input_dir/f'raw_T{T}.npz')['x']
        md=pd.read_csv(args.input_dir/f'raw_T{T}_manifest.csv')
        for idx,m in md[md.generator=='corrected'].iterrows():
            x=data[idx].astype(float);meta=m.to_dict()
            for key,q in grids.items():
                target=q*meta['H']-.5*meta['lambda2_true']*q*(q-2)
                z=structure_zeta(x,q=q)
                rows.append({**meta,'qgrid':key,**projection_metrics(z,target,meta['H'],meta['lambda2_true'],q=q)})
            if meta['process']=='MRW':
                if meta['L']==512:
                    z=structure_zeta(x)
                else:
                    p=generate_pair(T,.5,meta['lambda2_true'],int(meta['data_seed']),integral_scale=512)
                    z=structure_zeta(standardize(p['corrected']))
                target=Q*.5-.5*meta['lambda2_true']*Q*(Q-2)
                Lrows.append({**meta,'L':512,**projection_metrics(z,target,.5,meta['lambda2_true'])})
        print('scope sensitivity',T,flush=True)
    d=pd.DataFrame(rows);d.to_csv(out/'qgrid_sample_results.csv',index=False)
    ld=pd.DataFrame(Lrows);ld.to_csv(out/'fixed_integral_scale_samples.csv',index=False)
    summary=[]
    for key,g in d[d.process=='MRW'].groupby(['T','qgrid','seed']):
        summary.append(dict(T=key[0],qgrid=key[1],seed=key[2],n=len(g),
                            r=safe_corr(g.lambda2_true,g.lambda2_proj),mae=g.lambda2_mae.mean()))
    pd.DataFrame(summary).to_csv(out/'qgrid_by_seed.csv',index=False)
    summary=[]
    for key,g in ld.groupby(['T','seed']):
        summary.append(dict(T=key[0],seed=key[1],L=512,n=len(g),
                            r=safe_corr(g.lambda2_true,g.lambda2_proj),mae=g.lambda2_mae.mean()))
    pd.DataFrame(summary).to_csv(out/'fixed_integral_scale_by_seed.csv',index=False)
    dump_json(out/'status.json',dict(status='PASS',qgrid_rows=len(d),fixedL_rows=len(ld)))

if __name__=='__main__':main()
