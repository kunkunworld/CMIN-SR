from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

import numpy as np

from .data import (
    RAW_DATA_PATH,
    DEFAULT_SPLIT_PATH,
    load_dataset,
    load_splits,
    make_splits,
    save_splits,
    select_features,
    standardize_from_train,
)


@dataclass
class TrainConfig:
    feature_key: str = "dx"
    batch_size: int = 64
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dims: tuple[int, ...] = (512, 256, 128)
    dropout: float = 0.1
    seed: int = 2026
    split_path: str = str(DEFAULT_SPLIT_PATH)
    output_dir: str = "outputs/mlp_dx"


def _set_seed(seed: int) -> None:
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_torch_dataloaders(config: TrainConfig):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    bundle = load_dataset(RAW_DATA_PATH)
    split_path = Path(config.split_path)
    if split_path.exists():
        splits = load_splits(split_path)
    else:
        splits = make_splits(bundle.dx.shape[0], seed=config.seed)
        save_splits(splits, split_path)

    features = select_features(bundle, config.feature_key)
    features, mean, std = standardize_from_train(features, splits["train_idx"])
    targets = bundle.params.astype(np.float32)

    tensors = {}
    for split_name, split_idx in splits.items():
        x_tensor = torch.from_numpy(features[split_idx])
        y_tensor = torch.from_numpy(targets[split_idx])
        dataset = TensorDataset(x_tensor, y_tensor)
        tensors[split_name] = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=(split_name == "train_idx"),
        )

    return tensors, bundle.param_names, mean, std


def train_mlp(config: TrainConfig) -> Dict[str, object]:
    import torch
    from torch import nn

    from .models import MLPRegressor

    _set_seed(config.seed)
    loaders, param_names, feature_mean, feature_std = _to_torch_dataloaders(config)

    first_batch = next(iter(loaders["train_idx"]))
    input_dim = int(first_batch[0].shape[1])
    output_dim = int(first_batch[1].shape[1])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPRegressor(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None

    for _ in range(config.epochs):
        model.train()
        train_losses = []
        for xb, yb in loaders["train_idx"]:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in loaders["val_idx"]:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                val_losses.append(float(loss_fn(pred, yb).item()))

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if best_state is not None:
        torch.save(best_state, output_dir / "best_model.pt")

    metrics = evaluate_mlp(model, loaders["test_idx"], param_names, device)

    summary = {
        "config": asdict(config),
        "best_val_loss": best_val,
        "test_metrics": metrics,
        "param_names": param_names,
        "feature_mean_shape": list(feature_mean.shape),
        "feature_std_shape": list(feature_std.shape),
        "history": history,
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def evaluate_mlp(model, test_loader, param_names, device) -> Dict[str, Dict[str, float]]:
    import torch

    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            preds.append(pred)
            targets.append(yb.numpy())

    pred_arr = np.concatenate(preds, axis=0)
    target_arr = np.concatenate(targets, axis=0)
    err = pred_arr - target_arr

    metrics = {}
    for i, name in enumerate(param_names):
        metrics[name] = {
            "mae": float(np.mean(np.abs(err[:, i]))),
            "rmse": float(np.sqrt(np.mean(err[:, i] ** 2))),
            "bias": float(np.mean(err[:, i])),
        }
    return metrics


def main() -> None:
    try:
        summary = train_mlp(TrainConfig())
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is required for training. Install dependencies from requirements.txt first."
        ) from exc

    print("Training finished.")
    print(json.dumps(summary["test_metrics"], indent=2))


if __name__ == "__main__":
    main()

