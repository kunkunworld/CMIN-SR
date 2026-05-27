from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mrw_dl.baselines import legendre_spectrum_from_zeta, true_mrw_zeta
from mrw_dl.generation import MRWParams, generate_mrw_fgn
from mrw_inverse.losses import mrw_total_loss
from mrw_inverse.models import CMINRegressor


CHECKPOINT_PATH = ROOT / "checkpoints" / "cmin" / "cmin_tiny_synthetic.pt"
REPORT_DIR = ROOT / "outputs" / "reports" / "cmin_training"
TABLE_DIR = ROOT / "outputs" / "tables" / "cmin_training"
FIG_DIR = ROOT / "outputs" / "figures" / "cmin_training"
Q_GRID = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=np.float32)


@dataclass
class TrainConfig:
    length: int = 512
    num_train: int = 2000
    num_val: int = 500
    batch_size: int = 32
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 2026
    h_min: float = 0.1
    h_max: float = 0.9
    lambda2_min: float = 0.0
    lambda2_max: float = 0.2
    param_weight: float = 1.0
    zeta_weight: float = 0.5
    falpha_weight: float = 0.2
    constraint_weight: float = 0.1
    logvol_weight: float = 0.0
    scaling_weight: float = 0.0
    contrast_weight: float = 0.0
    device: str = "cpu"


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _sample_dataset(num_samples: int, cfg: TrainConfig, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.zeros((num_samples, cfg.length), dtype=np.float32)
    h = np.zeros((num_samples, 1), dtype=np.float32)
    lambda2 = np.zeros((num_samples, 1), dtype=np.float32)
    zeta = np.zeros((num_samples, len(Q_GRID)), dtype=np.float32)
    alpha = np.zeros((num_samples, len(Q_GRID)), dtype=np.float32)
    f_alpha = np.zeros((num_samples, len(Q_GRID)), dtype=np.float32)
    for i in range(num_samples):
        h_i = float(rng.uniform(cfg.h_min, cfg.h_max))
        lambda_i = float(rng.uniform(cfg.lambda2_min, cfg.lambda2_max))
        params = MRWParams(
            length=cfg.length,
            H=h_i,
            lambda2=lambda_i,
            L=min(256, max(64, cfg.length // 2)),
            sigma=1.0,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        sample = generate_mrw_fgn(params)
        dx = sample["dx"].astype(np.float32)
        dx = dx / max(np.std(dx), 1e-6)
        zeta_i = true_mrw_zeta(Q_GRID, h=h_i, lambda2=lambda_i).astype(np.float32)
        alpha_i, f_i = legendre_spectrum_from_zeta(Q_GRID, zeta_i)
        x[i] = dx
        h[i, 0] = h_i
        lambda2[i, 0] = lambda_i
        zeta[i] = zeta_i
        alpha[i] = alpha_i.astype(np.float32)
        f_alpha[i] = f_i.astype(np.float32)
    return {
        "x": x,
        "h": h,
        "lambda2": lambda2,
        "zeta": zeta,
        "alpha": alpha,
        "f_alpha": f_alpha,
    }


def _make_loader(dataset: dict[str, np.ndarray], batch_size: int, shuffle: bool) -> DataLoader:
    tensors = [
        torch.tensor(dataset["x"], dtype=torch.float32),
        torch.tensor(dataset["h"], dtype=torch.float32),
        torch.tensor(dataset["lambda2"], dtype=torch.float32),
        torch.tensor(dataset["zeta"], dtype=torch.float32),
        torch.tensor(dataset["f_alpha"], dtype=torch.float32),
    ]
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle)


def _evaluate(model: CMINRegressor, loader: DataLoader, cfg: TrainConfig) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    rows: list[dict[str, float]] = []
    losses = []
    with torch.no_grad():
        for x, h_true, lambda2_true, zeta_true, f_true in loader:
            x = x.to(cfg.device)
            h_true = h_true.to(cfg.device)
            lambda2_true = lambda2_true.to(cfg.device)
            zeta_true = zeta_true.to(cfg.device)
            f_true = f_true.to(cfg.device)
            out = model(x)
            loss = mrw_total_loss(
                output=out,
                h_true=h_true,
                lambda2_true=lambda2_true,
                zeta_true=zeta_true,
                f_true=f_true,
                log_scales=model.structure_branch.frontend.log_scales,
                log_lags=model.logvol_branch.log_lags,
                param_weight=cfg.param_weight,
                zeta_weight=cfg.zeta_weight,
                falpha_weight=cfg.falpha_weight,
                scaling_weight=cfg.scaling_weight,
                logvol_weight=cfg.logvol_weight,
                constraint_weight=cfg.constraint_weight,
                contrast_weight=cfg.contrast_weight,
            )
            losses.append(float(loss.total.cpu()))
            batch = x.shape[0]
            for i in range(batch):
                pred_h = float(out.h_hat[i, 0].cpu())
                pred_lambda = float(out.lambda2_hat[i, 0].cpu())
                true_h = float(h_true[i, 0].cpu())
                true_lambda = float(lambda2_true[i, 0].cpu())
                zeta_mae = float(torch.mean(torch.abs(out.zeta_hat[i] - zeta_true[i])).cpu())
                f_mae = float(torch.mean(torch.abs(out.f_alpha_hat[i] - f_true[i])).cpu())
                rows.append(
                    {
                        "true_H": true_h,
                        "pred_H": pred_h,
                        "true_lambda2": true_lambda,
                        "pred_lambda2": pred_lambda,
                        "zeta_mae": zeta_mae,
                        "f_alpha_mae": f_mae,
                        "lambda2_boundary_hit": float(pred_lambda <= 0.005 or pred_lambda >= 0.195),
                    }
                )
    pred_df = pd.DataFrame(rows)
    metrics = {
        "loss": float(np.mean(losses)) if losses else math.nan,
        "mae_H": float(np.mean(np.abs(pred_df["pred_H"] - pred_df["true_H"]))) if not pred_df.empty else math.nan,
        "mae_lambda2": float(np.mean(np.abs(pred_df["pred_lambda2"] - pred_df["true_lambda2"]))) if not pred_df.empty else math.nan,
        "mae_zeta": float(pred_df["zeta_mae"].mean()) if not pred_df.empty else math.nan,
        "mae_f_alpha": float(pred_df["f_alpha_mae"].mean()) if not pred_df.empty else math.nan,
        "lambda2_boundary_rate": float(pred_df["lambda2_boundary_hit"].mean()) if not pred_df.empty else math.nan,
    }
    return metrics, pred_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny synthetic CMIN checkpoint.")
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--num-train", type=int, default=2000)
    parser.add_argument("--num-val", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--param-weight", type=float, default=1.0)
    parser.add_argument("--zeta-weight", type=float, default=0.5)
    parser.add_argument("--falpha-weight", type=float, default=0.2)
    parser.add_argument("--constraint-weight", type=float, default=0.1)
    parser.add_argument("--logvol-weight", type=float, default=0.0)
    parser.add_argument("--scaling-weight", type=float, default=0.0)
    parser.add_argument("--contrast-weight", type=float, default=0.0)
    args = parser.parse_args()

    cfg = TrainConfig(
        length=args.length,
        num_train=args.num_train,
        num_val=args.num_val,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        param_weight=args.param_weight,
        zeta_weight=args.zeta_weight,
        falpha_weight=args.falpha_weight,
        constraint_weight=args.constraint_weight,
        logvol_weight=args.logvol_weight,
        scaling_weight=args.scaling_weight,
        contrast_weight=args.contrast_weight,
    )
    if cfg.device != "cpu" and not torch.cuda.is_available():
        cfg.device = "cpu"

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    _set_seed(cfg.seed)
    train_data = _sample_dataset(cfg.num_train, cfg, seed=cfg.seed)
    val_data = _sample_dataset(cfg.num_val, cfg, seed=cfg.seed + 1)
    train_loader = _make_loader(train_data, cfg.batch_size, shuffle=True)
    val_loader = _make_loader(val_data, cfg.batch_size, shuffle=False)

    model = CMINRegressor().to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history_rows: list[dict[str, float]] = []
    nan_occurred = False
    best_state = None
    best_val_loss = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses = []
        for x, h_true, lambda2_true, zeta_true, f_true in train_loader:
            x = x.to(cfg.device)
            h_true = h_true.to(cfg.device)
            lambda2_true = lambda2_true.to(cfg.device)
            zeta_true = zeta_true.to(cfg.device)
            f_true = f_true.to(cfg.device)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = mrw_total_loss(
                output=out,
                h_true=h_true,
                lambda2_true=lambda2_true,
                zeta_true=zeta_true,
                f_true=f_true,
                log_scales=model.structure_branch.frontend.log_scales,
                log_lags=model.logvol_branch.log_lags,
                param_weight=cfg.param_weight,
                zeta_weight=cfg.zeta_weight,
                falpha_weight=cfg.falpha_weight,
                scaling_weight=cfg.scaling_weight,
                logvol_weight=cfg.logvol_weight,
                constraint_weight=cfg.constraint_weight,
                contrast_weight=cfg.contrast_weight,
            )
            if torch.isnan(loss.total):
                nan_occurred = True
                break
            loss.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.total.detach().cpu()))
        val_metrics, val_pred = _evaluate(model, val_loader, cfg)
        train_loss = float(np.mean(train_losses)) if train_losses else math.nan
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_mae_H": val_metrics["mae_H"],
                "val_mae_lambda2": val_metrics["mae_lambda2"],
                "val_mae_zeta": val_metrics["mae_zeta"],
                "val_mae_f_alpha": val_metrics["mae_f_alpha"],
                "val_lambda2_boundary_rate": val_metrics["lambda2_boundary_rate"],
            }
        )
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {
                "model_state_dict": model.state_dict(),
                "config": asdict(cfg),
                "model_name": "cmin_tiny_synthetic",
                "best_epoch": epoch,
            }
            val_pred_best = val_pred.copy()
        if nan_occurred:
            break

    history = pd.DataFrame(history_rows)
    history_path = TABLE_DIR / "train_history.csv"
    history.to_csv(history_path, index=False)
    val_pred_path = TABLE_DIR / "val_predictions.csv"
    val_pred_best.to_csv(val_pred_path, index=False)

    if best_state is None:
        raise RuntimeError("CMIN training failed before producing a checkpoint.")
    torch.save(best_state, CHECKPOINT_PATH)

    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    ax.plot(history["epoch"], history["train_loss"], label="train")
    ax.plot(history["epoch"], history["val_loss"], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("CMIN tiny synthetic training")
    ax.legend()
    loss_fig_path = FIG_DIR / "loss_curve.png"
    fig.savefig(loss_fig_path, dpi=220)
    plt.close(fig)

    final_row = history.iloc[-1].to_dict() if not history.empty else {}
    summary_lines = [
        "# CMIN Tiny Synthetic Training Summary",
        "",
        "## Config",
        "",
        *[f"- `{k}`: `{v}`" for k, v in asdict(cfg).items()],
        "",
        "## Final Metrics",
        "",
        f"- Final train loss: `{final_row.get('train_loss', math.nan):.6f}`",
        f"- Final val loss: `{final_row.get('val_loss', math.nan):.6f}`",
        f"- Val MAE H: `{final_row.get('val_mae_H', math.nan):.6f}`",
        f"- Val MAE lambda2: `{final_row.get('val_mae_lambda2', math.nan):.6f}`",
        f"- Val MAE zeta: `{final_row.get('val_mae_zeta', math.nan):.6f}`",
        f"- Val MAE f(alpha): `{final_row.get('val_mae_f_alpha', math.nan):.6f}`",
        f"- Lambda2 boundary hit rate: `{final_row.get('val_lambda2_boundary_rate', math.nan):.6f}`",
        f"- NaN occurred: `{nan_occurred}`",
        f"- Checkpoint: `{CHECKPOINT_PATH.relative_to(ROOT)}`",
        "",
        "## Outputs",
        "",
        f"- Train history: `{history_path.relative_to(ROOT)}`",
        f"- Loss curve: `{loss_fig_path.relative_to(ROOT)}`",
        f"- Validation predictions: `{val_pred_path.relative_to(ROOT)}`",
    ]
    summary_path = REPORT_DIR / "cmin_tiny_training_summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    meta = {
        "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)),
        "history_csv": str(history_path.relative_to(ROOT)),
        "loss_curve": str(loss_fig_path.relative_to(ROOT)),
        "val_predictions": str(val_pred_path.relative_to(ROOT)),
        "summary": str(summary_path.relative_to(ROOT)),
        "final_metrics": final_row,
        "nan_occurred": nan_occurred,
    }
    (REPORT_DIR / "cmin_tiny_training_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
