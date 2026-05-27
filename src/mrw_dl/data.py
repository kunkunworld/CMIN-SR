from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = ROOT / "data" / "raw" / "mrw_dataset_robust_fgn.npz"
RAW_META_PATH = ROOT / "data" / "raw" / "mrw_dataset_robust_fgn_meta.json"
DEFAULT_SPLIT_PATH = ROOT / "data" / "processed" / "splits.npz"


@dataclass
class DatasetBundle:
    t: np.ndarray
    x: np.ndarray
    dx: np.ndarray
    omega: np.ndarray
    params: np.ndarray
    param_names: List[str]


def load_dataset(path: Path | str = RAW_DATA_PATH) -> DatasetBundle:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        param_names = [str(v) for v in data["param_names"].tolist()]
        return DatasetBundle(
            t=data["t"],
            x=data["x"],
            dx=data["dx"],
            omega=data["omega"],
            params=data["params"],
            param_names=param_names,
        )


def load_metadata(path: Path | str = RAW_META_PATH) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def describe_dataset(bundle: DatasetBundle) -> Dict[str, object]:
    stats = {
        "num_samples": int(bundle.dx.shape[0]),
        "sequence_length": int(bundle.dx.shape[1]),
        "channels": ["x", "dx", "omega"],
        "targets": bundle.param_names,
        "target_mean": bundle.params.mean(axis=0).round(6).tolist(),
        "target_std": bundle.params.std(axis=0).round(6).tolist(),
    }
    return stats


def make_splits(
    num_samples: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 2026,
) -> Dict[str, np.ndarray]:
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Ratios must be positive and train_ratio + val_ratio < 1.")

    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples, dtype=np.int64)
    rng.shuffle(indices)

    n_train = int(num_samples * train_ratio)
    n_val = int(num_samples * val_ratio)

    train_idx = np.sort(indices[:n_train])
    val_idx = np.sort(indices[n_train:n_train + n_val])
    test_idx = np.sort(indices[n_train + n_val:])

    return {"train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx}


def save_splits(splits: Dict[str, np.ndarray], path: Path | str = DEFAULT_SPLIT_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **splits)
    return path


def load_splits(path: Path | str = DEFAULT_SPLIT_PATH) -> Dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def select_features(bundle: DatasetBundle, feature_key: str = "dx") -> np.ndarray:
    feature_map = {
        "x": bundle.x,
        "dx": bundle.dx,
        "omega": bundle.omega,
        "x_dx": np.concatenate([bundle.x, bundle.dx], axis=1),
        "dx_omega": np.concatenate([bundle.dx, bundle.omega], axis=1),
    }
    if feature_key not in feature_map:
        raise KeyError(f"Unsupported feature_key: {feature_key}")
    return feature_map[feature_key].astype(np.float32)


def standardize_from_train(
    x: np.ndarray,
    train_idx: Sequence[int],
    eps: float = 1e-6,
    mode: str = "pointwise",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mode == "pointwise":
        mean = x[np.asarray(train_idx)].mean(axis=0, keepdims=True)
        std = x[np.asarray(train_idx)].std(axis=0, keepdims=True)
    elif mode == "global":
        mean = np.array([[x[np.asarray(train_idx)].mean()]], dtype=np.float32)
        std = np.array([[x[np.asarray(train_idx)].std()]], dtype=np.float32)
    else:
        raise ValueError(f"Unsupported standardization mode: {mode}")
    std = np.maximum(std, eps)
    x_norm = (x - mean) / std
    return x_norm.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def summarize_split_sizes(splits: Dict[str, np.ndarray]) -> Dict[str, int]:
    return {key: int(len(val)) for key, val in splits.items()}
