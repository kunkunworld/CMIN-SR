"""R2-only provenance and numerical helpers."""
from pathlib import Path
import hashlib
import json
import platform
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
Q = np.arange(.5, 3.01, .5)
SCALES = (2, 4, 8, 16, 32, 64)
B = 2000


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str), encoding='utf-8')


def protect():
    paths = list((ROOT / 'revision2').glob('*.tex'))
    paths += list((ROOT / 'revision2').glob('*.pdf'))
    paths += list((ROOT / 'revision2').glob('*.docx'))
    paths += list((ROOT / 'src').rglob('*.py'))
    paths += list((ROOT / 'revision').glob('*.py'))
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def start(out, details):
    out.mkdir(parents=True, exist_ok=False)
    before = protect()
    dump(out / 'protected_before.json', before)
    dump(out / 'manifest.json', dict(protocol_sha256=sha(HERE / 'protocol.json'),
         python=sys.executable, platform=platform.platform(), numpy=np.__version__,
         pandas=pd.__version__, details=details,
         scripts={p.name: sha(p) for p in HERE.glob('*.py')}))
    dump(out / 'status.json', {'status': 'RUNNING'})
    return before


def finish(out, before, **details):
    after = protect()
    changed = [p for p, h in before.items() if after.get(p) != h]
    if changed:
        raise AssertionError(f'Protected files changed: {changed}')
    dump(out / 'status.json', dict(status='COMPLETED', protected_unchanged=True, **details))


def moments(x, scales, q=Q):
    # Same no-prepended-origin endpoint convention as the R1 OLS estimator.
    p = np.cumsum(np.asarray(x, dtype=float))
    return np.array([np.mean(np.maximum(np.abs(p[s:] - p[:-s]), 1e-30)[:, None] ** q,
                            axis=0) for s in scales])


def slopes(moment_array, scales):
    u = np.log(np.asarray(scales)); u -= u.mean()
    return np.einsum('s,...sq->...q', u, np.log(moment_array)) / (u @ u)


def corr(truth, prediction):
    a = truth - np.mean(truth, axis=-1, keepdims=True)
    b = prediction - np.mean(prediction, axis=-1, keepdims=True)
    den = np.sqrt((a*a).sum(axis=-1) * (b*b).sum(axis=-1))
    return np.divide((a*b).sum(axis=-1), den,
                     out=np.full(np.shape(den), np.nan), where=den > 1e-14)


def indices(meta, seed=202610079, B=B):
    rng = np.random.default_rng(seed)
    keys = [k for k in ['seed', 'process', 'H', 'lambda2_true'] if k in meta]
    groups = meta.reset_index(drop=True).groupby(keys, sort=True).indices
    return np.concatenate([rng.choice(v, size=(B, len(v)), replace=True)
                           for v in groups.values()], axis=1)


def ci(values):
    v = np.asarray(values)
    v = v[np.isfinite(v)]
    return np.quantile(v, [.025, .975]) if len(v) else [np.nan, np.nan]


def savefig(fig, out, stem):
    fig.savefig(out / f'{stem}.png', dpi=180, bbox_inches='tight')
    fig.savefig(out / f'{stem}.pdf', bbox_inches='tight')
