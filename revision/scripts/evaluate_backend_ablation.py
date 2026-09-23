"""Fresh paired analytic/raw holdouts for the frozen-front-end backend ablation."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import argparse
import json
import time
import traceback
import numpy as np
import pandas as pd
import torch
from revision.backend_ablation import (analytic_data, spectrum_features, make_backend, predict,
    clean_raw, stress_raw, unseen_curves, rule_scores, SCORES)
from revision.analysis_tools import FrozenCalibrator, structure_zeta, hashfile, dump_json, protected_hashes
from revision.retraining import make_model, finite


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--quick', action='store_true')
    ap.add_argument('--training-dir', type=Path, required=True)
    ap.add_argument('--frontend-dir', type=Path, default=ROOT/'revision/results/neural_retrain_formal_attempt3')
    ap.add_argument('--frontend-seed', type=int, default=51001)
    ap.add_argument('--output-dir', type=Path, required=True); args = ap.parse_args()
    train = args.training_dir
    assert json.loads((train/'status.json').read_text())['status'] == 'PASS'
    out = args.output_dir; out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); before = protected_hashes(); t0 = time.time()
    dump_json(out/'status.json', dict(status='RUNNING'))
    runs = pd.read_csv(train/'training_runs.csv'); backends = []; datafiles = []
    try:
        for row in runs.itertuples():
            cp = train/row.checkpoint; assert hashfile(cp) == row.checkpoint_sha256
            data = np.load(train/f'seed_{row.seed}/train.npz')
            model = make_backend(row.variant, data['x'], row.seed)
            model.load_state_dict(torch.load(cp, map_location='cpu')['model_state_dict'], strict=True)
            backends.append((row.variant, int(row.seed), model.eval()))
        legacy = FrozenCalibrator()
        backends.append(('legacy_mlp', -1, legacy.model.net.eval()))
        frontdir = args.frontend_dir.resolve()
        assert json.loads((frontdir/'status.json').read_text())['status'] == 'PASS'
        fruns = pd.read_csv(frontdir/'training_runs.csv')
        selected = fruns[(fruns.seed == args.frontend_seed) & (fruns.config == 'full_c025')]
        if len(selected) != 1:
            raise ValueError('Expected exactly one full_c025 frontend for the requested seed')
        fr = selected.iloc[0]
        cp = frontdir/fr.checkpoint; assert hashfile(cp) == fr.checkpoint_sha256
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        front = make_model(args.frontend_seed, .25, device)
        front.load_state_dict(torch.load(cp, map_location='cpu')['model_state_dict'], strict=True); front.eval()
        for p in front.parameters(): p.requires_grad_(False)
        dump_json(out/'manifest.json', dict(protocol_sha256=hashfile(ROOT/'revision/BACKEND_ABLATION_PROTOCOL.md'),
            training_manifest_sha256=hashfile(train/'manifest.json'), script_sha256=hashfile(Path(__file__)),
            frontend_checkpoint=str(cp), frontend_sha256=hashfile(cp), frontend_trainable=False,
            frontend_device=device, frontend_seed=args.frontend_seed, legacy=legacy.manifest, tail_input=0, quick=args.quick,
            upstreams=['structure_ols', f'frozen_neural_seed{args.frontend_seed}'], analysis='conditional on this frozen frontend'))

        def evaluate(x, meta, name, upstream, target=None):
            finite(name+' features', x)
            meta = meta.copy(); meta['upstream'] = upstream
            np.savez_compressed(out/f'{name}_features.npz', x=x, **({'targets': target} if target is not None else {}))
            meta.to_csv(out/f'{name}_manifest.csv', index=False)
            outputs = []
            for variant, seed, model in backends:
                yp = predict(model, variant, x)
                d = meta.copy(); d['method'] = variant; d['backend_seed'] = seed
                for j, key in enumerate(SCORES): d[key] = yp[:, j]
                d['score_mrw'] = yp[:, 3]; d['score_curved'] = yp[:, 1]
                if target is not None:
                    d['soft_target_mse'] = np.mean((yp-target)**2, axis=1)
                    for j, key in enumerate(SCORES): d['sq_error_'+key] = (yp[:, j]-target[:, j])**2
                outputs.append(d)
            for method, value in rule_scores(x).items():
                finite(method, value)
                d = meta.copy(); d['method'] = method; d['backend_seed'] = -1
                d['score_mrw'] = value; d['score_curved'] = value
                outputs.append(d)
            df = pd.concat(outputs, ignore_index=True)
            df.to_csv(out/f'{name}_predictions.csv', index=False)
            datafiles.append(dict(name=name, domain=meta.domain.iloc[0], upstream=upstream,
                split=meta.split.iloc[0], T=int(meta['T'].iloc[0]), independent_N=len(meta),
                prediction_rows=len(df), file=f'{name}_predictions.csv',
                sha256=hashfile(out/f'{name}_predictions.csv'), feature_sha256=hashfile(out/f'{name}_features.npz')))
            print(f'{name}: N={len(meta)}, {len(outputs)} backend repeats/rules, elapsed={time.time()-t0:.0f}s', flush=True)

        vals = []; vmetas = []
        for seed in runs.seed.unique():
            dat = np.load(train/f'seed_{seed}/validation.npz')
            vals.append(dict(x=dat['x'], y=dat['y']))
            vmetas.append(pd.read_csv(train/f'seed_{seed}/validation_manifest.csv'))
        vmeta = pd.concat(vmetas, ignore_index=True)
        evaluate(np.concatenate([d['x'] for d in vals]), vmeta, 'analytic_validation', 'analytic', np.concatenate([d['y'] for d in vals]))
        tests = []; metas = []
        for seed in ([123001] if args.quick else range(123001, 123006)):
            d, m = analytic_data(80 if args.quick else 1600, seed, 'test'); tests.append(d); metas.append(m)
        ameta = pd.concat(metas, ignore_index=True)
        evaluate(np.concatenate([d['x'] for d in tests]), ameta, 'analytic_test', 'analytic', np.concatenate([d['y'] for d in tests]))
        ux, um = unseen_curves(args.quick); evaluate(ux, um, 'unseen', 'analytic')
        raw_meta = []

        def evaluate_raw(xs, meta, name):
            finite('raw', xs); np.savez_compressed(out/f'{name}_raw.npz', x=xs)
            ols = np.stack([structure_zeta(x.astype(float)) for x in xs]); zs = []
            with torch.no_grad():
                for i in range(0, len(xs), 64):
                    z = front(torch.tensor(xs[i:i+64], device=device))['zeta_emp'].cpu().numpy()
                    finite('frozen frontend', z); zs.append(z)
            for upstream, z in [('structure_ols', ols), (f'frozen_neural_seed{args.frontend_seed}', np.concatenate(zs))]:
                evaluate(spectrum_features(z), meta, name+'_'+upstream, upstream)
            raw_meta.append(meta)

        for split in ['threshold_validation', 'test']:
            for T in ([1024] if args.quick else [512, 1024, 2048]):
                xs, meta = clean_raw(T, split, args.quick); evaluate_raw(xs, meta, f'raw_{split}_T{T}')
        xs, meta = stress_raw(args.quick); evaluate_raw(xs, meta, 'raw_stress_T1024')
        rm = pd.concat(raw_meta, ignore_index=True)
        assert not rm.sample_id.duplicated().any() and not rm.data_seed.duplicated().any()
        old = pd.read_csv(frontdir/'data_manifest.csv'); assert not set(rm.data_seed) & set(old.data_seed)
        all_analytic_ids = set()
        for seed in runs.seed.unique():
            for split in ['train', 'validation']:
                ids = set(pd.read_csv(train/f'seed_{seed}/{split}_manifest.csv').sample_id)
                assert not ids & all_analytic_ids; all_analytic_ids.update(ids)
        assert not set(ameta.sample_id) & all_analytic_ids
        for folder in ['neural_eval_formal', 'stress_controls', 'corrected_recovery_formal', 'corrected_recovery_validation']:
            for p in (ROOT/'revision/results'/folder).glob('*manifest.csv'):
                oldmeta = pd.read_csv(p)
                if 'data_seed' in oldmeta: assert not set(rm.data_seed) & set(oldmeta.data_seed), p
        pd.DataFrame(datafiles).to_csv(out/'prediction_manifest.csv', index=False)
        dump_json(out/'split_check.json', dict(status='PASS', new_raw_sequences=len(rm), seed_duplicates=0,
            analytic_test_overlap=0, raw_frontend_train_overlap=0, raw_previous_eval_overlap=0,
            new_raw_test_N=int((rm.split == 'test').sum()), new_raw_validation_N=int((rm.split == 'threshold_validation').sum()),
            stress_N=int((rm.split == 'stress').sum())))
        assert before == protected_hashes(), 'Protected originals changed'
        dump_json(out/'status.json', dict(status='PASS', elapsed_seconds=time.time()-t0,
            new_analytic_test_N=len(ameta), new_raw_sequences=len(rm), original_files_unchanged=True,
            no_nonfinite_scores=True, no_oracle_inputs=True))
    except Exception:
        dump_json(out/'status.json', dict(status='FAIL', traceback=traceback.format_exc())); raise


if __name__ == '__main__': main()
