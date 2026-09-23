"""Train existing MLP and affine ablations on identical analytic targets."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import argparse
import copy
import json
import time
import traceback
import numpy as np
import pandas as pd
import torch
from revision.backend_ablation import analytic_data, make_backend, forward_scores, existing_loss, predict, VARIANTS
from revision.analysis_tools import dump_json, hashfile, protected_hashes
from revision.retraining import finite


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--quick', action='store_true')
    ap.add_argument('--output-dir', type=Path, required=True); args = ap.parse_args()
    out = args.output_dir; out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    seeds = [121001] if args.quick else [121001, 121002, 121003]
    epochs = 2 if args.quick else 30; ntrain = 256 if args.quick else 20000
    nval = 128 if args.quick else 4000; before = protected_hashes(); t0 = time.time()
    dump_json(out/'protected_before.json', before)
    sources = [Path(__file__), ROOT/'revision/backend_ablation.py', ROOT/'revision/BACKEND_ABLATION_PROTOCOL.md',
        ROOT/'src/mrw_inverse/data/analytic_spectrum_dataset.py', ROOT/'src/mrw_inverse/models/spectral_geometry_calibrator.py',
        ROOT/'src/mrw_inverse/losses/spectrum_space_calibration_losses.py']
    dump_json(out/'manifest.json', dict(protocol_sha256=hashfile(sources[2]), seeds=seeds, epochs=epochs,
        n_train=ntrain, n_validation=nval, quick=args.quick, device='cpu', batch_size=128, lr=.001,
        weight_decay=.0001, selection='validation five-score MSE', tail_input=0,
        sources={str(p.relative_to(ROOT)): hashfile(p) for p in sources}))
    snap = out/'code_snapshot'; snap.mkdir()
    for p in sources: (snap/p.name).write_bytes(p.read_bytes())
    dump_json(out/'status.json', dict(status='RUNNING')); runs = []
    try:
        for seed in seeds:
            folder = out/f'seed_{seed}'; folder.mkdir()
            train, mt = analytic_data(ntrain, seed, 'train')
            val, mv = analytic_data(nval, seed+1000, 'validation')
            for name, data, meta in [('train', train, mt), ('validation', val, mv)]:
                np.savez_compressed(folder/f'{name}.npz', **data)
                meta.to_csv(folder/f'{name}_manifest.csv', index=False)
            td = {k: torch.as_tensor(v) for k, v in train.items()}
            orders = [np.random.default_rng(np.random.SeedSequence([seed, 55, epoch])).permutation(ntrain) for epoch in range(epochs)]
            np.save(folder/'minibatch_orders.npy', np.stack(orders))
            for variant in VARIANTS:
                model = make_backend(variant, train['x'], seed)
                optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
                best = float('inf'); best_state = None; history = []
                for epoch in range(epochs):
                    model.train(); losses = []
                    for start in range(0, ntrain, 128):
                        ids = torch.as_tensor(orders[epoch][start:start+128])
                        optimizer.zero_grad(set_to_none=True)
                        p = forward_scores(model, variant, td['x'][ids])
                        loss = existing_loss(p, td, ids); loss.backward()
                        for par in model.parameters():
                            if par.grad is not None: finite('gradient', par.grad)
                        optimizer.step(); losses.append((len(ids), float(loss.detach())))
                    vp = predict(model, variant, val['x'])
                    mse = float(np.mean((vp-val['y'])**2)); finite('val MSE', mse)
                    history.append(dict(epoch=epoch+1, train_loss=np.average([v for _, v in losses], weights=[n for n, _ in losses]), val_score_mse=mse))
                    if mse < best:
                        best = mse; best_state = copy.deepcopy(model.state_dict()); best_epoch = epoch+1
                    if epoch == 0 or (epoch+1) % 10 == 0:
                        print(f'seed={seed} {variant} epoch={epoch+1}/{epochs} val_MSE={mse:.6f} elapsed={time.time()-t0:.0f}s', flush=True)
                path = folder/f'{variant}.pt'
                torch.save(dict(variant=variant, seed=seed, model_state_dict=best_state,
                    best_epoch=best_epoch, val_score_mse=best), path)
                model.load_state_dict(best_state, strict=True)
                replay = make_backend(variant, train['x'], seed+9)
                replay.load_state_dict(torch.load(path, map_location='cpu')['model_state_dict'], strict=True)
                np.testing.assert_allclose(predict(model, variant, val['x'][:16]), predict(replay, variant, val['x'][:16]), atol=0, rtol=0)
                pd.DataFrame(history).to_csv(folder/f'{variant}_history.csv', index=False)
                runs.append(dict(seed=seed, variant=variant, best_epoch=best_epoch, val_score_mse=best,
                    parameters=sum(p.numel() for p in model.parameters()), checkpoint=str(path.relative_to(out)), checkpoint_sha256=hashfile(path)))
                pd.DataFrame(runs).to_csv(out/'training_runs.csv', index=False)
            choices = [r for r in runs if r['seed'] == seed and r['variant'].startswith('linear_')]
            dump_json(folder/'selected_linear.json', min(choices, key=lambda r: r['val_score_mse']))
        assert before == protected_hashes(), 'Protected original changed'
        dump_json(out/'status.json', dict(status='PASS', runs=len(runs), elapsed_seconds=time.time()-t0,
            strict_checkpoint_replay=True, originals_unchanged=True, no_nonfinite=True))
    except Exception:
        dump_json(out/'status.json', dict(status='FAIL', traceback=traceback.format_exc())); raise


if __name__ == '__main__': main()
