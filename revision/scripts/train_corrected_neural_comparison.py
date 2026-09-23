"""Matched retraining of the unchanged network on corrected simulations."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import argparse,copy,json,time,traceback,shutil
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from revision.retraining import CONFIGS,make_model,make_training,loss_for,finite,prior_and_ols
from revision.analysis_tools import dump_json,hashfile,protected_hashes


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--quick',action='store_true')
    ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--device',default='cuda')
    args=ap.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=False)
    ntrain,nval,epochs,seeds=(256,64,1,[51999]) if args.quick else (6000,1200,8,[51001,51002,51003])
    device=torch.device(args.device);torch.set_num_threads(1)
    manifest=dict(num_train=ntrain,num_val=nval,epochs=epochs,seeds=seeds,batch_size=64,lr=3e-4,
                  weight_decay=1e-4,device=str(device),torch=torch.__version__,numpy=np.__version__,
                  gpu=torch.cuda.get_device_name(0) if device.type=='cuda' else None,
                  configs=CONFIGS,protocol_sha256=hashfile(ROOT/'revision/NEURAL_RETRAIN_PROTOCOL.md'),
                  script_sha256=hashfile(Path(__file__)),helper_sha256=hashfile(ROOT/'revision/retraining.py'),
                  test_accessed=False,selection='validation mean zeta MAE',
                  initialized_from_checkpoint=False,deterministic_algorithms=True)
    dump_json(out/'manifest.json',manifest);before=protected_hashes();dump_json(out/'protected_before.json',before)
    snapshot=out/'code_snapshot';snapshot.mkdir()
    for p in [Path(__file__),ROOT/'revision/retraining.py',ROOT/'revision/generation.py',ROOT/'revision/NEURAL_RETRAIN_PROTOCOL.md']:
        shutil.copy2(p,snapshot/p.name)
    dump_json(out/'status.json',{'status':'RUNNING'});t0=time.time();allmeta=[];runs=[]
    try:
        for seed in seeds:
            datasets={};validation={};seedout=out/f'seed_{seed}';seedout.mkdir()
            for T in [512,1024]:
                datasets[T],mt=make_training(ntrain//2,T,seed,1)
                validation[T],mv=make_training(nval//2,T,seed,2)
                allmeta.extend([mt,mv]);mt.to_csv(seedout/f'train_T{T}_manifest.csv',index=False)
                mv.to_csv(seedout/f'val_T{T}_manifest.csv',index=False)
                np.savez_compressed(seedout/f'data_T{T}.npz',
                    **{f'train_{k}':v.numpy() for k,v in datasets[T].items()},
                    **{f'val_{k}':v.numpy() for k,v in validation[T].items()})
            meta=pd.concat(allmeta,ignore_index=True)
            if meta.data_seed.duplicated().any(): raise AssertionError('Data-seed collision')
            if set(meta.data_seed)&set(meta.parameter_seed):raise AssertionError('Parameter/innovation seed collision')
            ref=make_model(seed,.25,device);features=[];targets=[]
            for T,d in datasets.items():
                pr,ols=prior_and_ols(ref,d['x'],device);features.append(np.c_[ols,pr]);targets.append(d['target'].numpy())
            scaler=StandardScaler().fit(np.concatenate(features))
            ridge=Ridge(alpha=.001).fit(scaler.transform(np.concatenate(features)),np.concatenate(targets))
            np.savez(seedout/'ridge.npz',mean=scaler.mean_,scale=scaler.scale_,coef=ridge.coef_,intercept=ridge.intercept_)
            del ref
            for config,(coefficient,mode) in CONFIGS.items():
                model=make_model(seed,coefficient,device)
                frozen={k:v.detach().cpu().clone() for k,v in model.named_parameters() if not v.requires_grad}
                d=datasets[512];model.eval()
                with torch.no_grad():
                    xx=d['x'][:8].to(device)
                    assert torch.equal(model(xx)['zeta_emp'],model.structure_branch(xx)['empirical_zeta_estimate'])
                opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-4,weight_decay=1e-4)
                history=[];best=float('inf');best_state=None;best_epoch=None
                for epoch in range(1,epochs+1):
                    model.train();rng=np.random.default_rng(seed+epoch);jobs=[]
                    for T,d in datasets.items():
                        order=rng.permutation(len(d['x']))
                        jobs.extend((T,order[s:s+64]) for s in range(0,len(order),64))
                    rng.shuffle(jobs);train_sum=0.;count=0
                    for T,idx in jobs:
                        batch={k:v[idx].to(device) for k,v in datasets[T].items()}
                        finite('input',batch['x']);pred=model(batch['x']);value=loss_for(pred,batch,mode)
                        opt.zero_grad(set_to_none=True);value.backward()
                        for name,p in model.named_parameters():
                            if p.grad is not None: finite('gradient '+name,p.grad)
                        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
                        train_sum+=float(value.detach())*len(idx);count+=len(idx)
                    model.eval();val_abs=0.;val_n=0;val_loss=0.
                    with torch.no_grad():
                        for T,d in validation.items():
                            for s in range(0,len(d['x']),64):
                                batch={k:v[s:s+64].to(device) for k,v in d.items()};pred=model(batch['x'])
                                value=loss_for(pred,batch,mode);val_loss+=float(value)*len(batch['x'])
                                val_abs+=float((pred['zeta_emp']-batch['target']).abs().sum());val_n+=batch['target'].numel()
                    score=val_abs/val_n
                    history.append(dict(epoch=epoch,train_loss=train_sum/count,val_zeta_mae=score,val_loss=val_loss/nval))
                    if score<best:
                        best=score;best_epoch=epoch;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                    pd.DataFrame(history).to_csv(seedout/f'{config}_history.csv',index=False)
                    print(f'seed={seed} {config} epoch={epoch}/{epochs} train={train_sum/count:.5f} val_zeta={score:.5f} elapsed={time.time()-t0:.0f}s',flush=True)
                for k,p in model.named_parameters():
                    if k in frozen and not torch.equal(p.detach().cpu(),frozen[k]): raise AssertionError(f'Frozen head changed: {k}')
                checkpoint=seedout/f'{config}.pt'
                torch.save(dict(model_state_dict=best_state,coefficient=coefficient,loss_mode=mode,seed=seed,
                                selected_epoch=best_epoch,val_zeta_mae=best,protocol=manifest['protocol_sha256']),checkpoint)
                reload=make_model(seed,coefficient,device)
                reload.load_state_dict(torch.load(checkpoint,map_location='cpu')['model_state_dict'],strict=True)
                model.load_state_dict(best_state,strict=True);model.eval();reload.eval()
                with torch.no_grad():
                    xx=validation[512]['x'][:8].to(device)
                    torch.testing.assert_close(model(xx)['zeta_emp'],reload(xx)['zeta_emp'],rtol=0,atol=0)
                runs.append(dict(seed=seed,config=config,coefficient=coefficient,loss_mode=mode,
                    selected_epoch=best_epoch,val_zeta_mae=best,checkpoint=str(checkpoint.relative_to(out)),
                    checkpoint_sha256=hashfile(checkpoint),frozen_heads_unchanged=True,strict_reload=True))
                del model,reload,opt
            pd.DataFrame(runs).to_csv(out/'training_runs.csv',index=False)
        meta.to_csv(out/'data_manifest.csv',index=False)
        if before!=protected_hashes(): raise AssertionError('Protected originals changed')
        dump_json(out/'status.json',dict(status='PASS',runs=len(runs),elapsed_seconds=time.time()-t0,
                  originals_unchanged=True,no_nonfinite=True,scope='Training and loading, not proof of superiority'))
    except BaseException:
        dump_json(out/'status.json',dict(status='FAIL',traceback=traceback.format_exc(),elapsed_seconds=time.time()-t0))
        raise


if __name__=='__main__':main()
