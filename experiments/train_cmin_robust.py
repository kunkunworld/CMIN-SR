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

from mrw_inverse.data import AntiConfoundedConfig, PROCESS_CODE_TO_NAME, generate_anti_confounded_dataset
from mrw_inverse.losses import robust_inverse_loss
from mrw_inverse.models import CMINRegressor


CHECKPOINT_PATH = ROOT / "checkpoints" / "cmin" / "cmin_robust_synthetic.pt"
REPORT_DIR = ROOT / "outputs" / "reports" / "cmin_robust_training"
TABLE_DIR = ROOT / "outputs" / "tables" / "cmin_robust_training"
FIG_DIR = ROOT / "outputs" / "figures" / "cmin_robust_training"


@dataclass
class RobustTrainConfig:
    length: int = 512
    num_train: int = 6000
    num_val: int = 1200
    batch_size: int = 32
    epochs: int = 15
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 2026
    device: str = "cpu"
    ambiguous_label_value: float = 0.25
    strict_negative_labels: bool = False
    param_weight: float = 1.0
    spectrum_weight: float = 0.5
    validity_weight: float = 1.0
    neg_lambda_weight: float = 1.0
    pair_weight: float = 1.0
    mismatch_weight: float = 0.0
    constraint_weight: float = 0.1


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _make_dataset(cfg: AntiConfoundedConfig) -> dict[str, np.ndarray]:
    return generate_anti_confounded_dataset(cfg)


def _make_loader(dataset: dict[str, np.ndarray], batch_size: int, shuffle: bool) -> DataLoader:
    order = [
        "x",
        "process_code",
        "is_mrw",
        "H_true",
        "lambda2_true",
        "target_lambda2",
        "target_p_mrw",
        "target_mismatch",
        "pair_id",
        "sample_id",
        "zeta_true",
        "f_alpha_true",
    ]
    tensors = []
    for key in order:
        arr = dataset[key]
        dtype = torch.float32
        if key in {"process_code", "pair_id", "sample_id"}:
            dtype = torch.long
        tensors.append(torch.tensor(arr, dtype=dtype))
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle)


def _batch_to_device(batch, device: str):
    return [t.to(device) for t in batch]


def _collect_eval(model: CMINRegressor, loader: DataLoader, cfg: RobustTrainConfig) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    model.eval()
    rows: list[dict[str, float | str]] = []
    losses = []
    with torch.no_grad():
        for batch in loader:
            (
                x,
                process_code,
                is_mrw,
                h_true,
                lambda2_true,
                target_lambda2,
                target_p_mrw,
                target_mismatch,
                pair_id,
                sample_id,
                zeta_true,
                f_true,
            ) = _batch_to_device(batch, cfg.device)
            out = model(x)
            loss = robust_inverse_loss(
                output=out,
                process_code=process_code,
                is_mrw=is_mrw,
                h_true=h_true,
                lambda2_true=lambda2_true,
                target_lambda2=target_lambda2,
                target_p_mrw=target_p_mrw,
                target_mismatch=target_mismatch,
                zeta_true=zeta_true,
                f_true=f_true,
                pair_id=pair_id,
                param_weight=cfg.param_weight,
                spectrum_weight=cfg.spectrum_weight,
                validity_weight=cfg.validity_weight,
                neg_lambda_weight=cfg.neg_lambda_weight,
                pair_weight=cfg.pair_weight,
                mismatch_weight=cfg.mismatch_weight,
                constraint_weight=cfg.constraint_weight,
            )
            losses.append(float(loss.total.cpu()))
            residual_norm = torch.mean(torch.abs(out.residual_zeta), dim=1)
            for i in range(x.shape[0]):
                proc_name = PROCESS_CODE_TO_NAME[int(process_code[i].cpu())]
                rows.append(
                    {
                        "sample_id": int(sample_id[i].cpu()),
                        "pair_id": int(pair_id[i].cpu()),
                        "process_type": proc_name,
                        "is_mrw": float(is_mrw[i, 0].cpu()),
                        "true_H": float(h_true[i, 0].cpu()),
                        "pred_H": float(out.h_hat[i, 0].cpu()),
                        "true_lambda2": float(lambda2_true[i, 0].cpu()),
                        "pred_lambda2": float(out.lambda2_hat[i, 0].cpu()),
                        "target_p_mrw": float(target_p_mrw[i, 0].cpu()),
                        "pred_p_mrw": float(out.p_mrw[i, 0].cpu()),
                        "target_mismatch": float(target_mismatch[i, 0].cpu()),
                        "residual_norm": float(residual_norm[i].cpu()),
                        "zeta_mae": float(torch.mean(torch.abs(out.zeta_hat[i] - zeta_true[i])).cpu()),
                        "f_alpha_mae": float(torch.nanmean(torch.abs(out.f_alpha_hat[i] - f_true[i])).cpu()),
                        "lambda2_fp": float((out.lambda2_hat[i, 0] > 0.02).cpu()),
                        "p_mrw_fp": float((out.p_mrw[i, 0] > 0.5).cpu()),
                    }
                )
    pred_df = pd.DataFrame(rows)
    process_summary = (
        pred_df.groupby("process_type")
        .agg(
            mean_pred_H=("pred_H", "mean"),
            mean_pred_lambda2=("pred_lambda2", "mean"),
            mean_pred_p_mrw=("pred_p_mrw", "mean"),
            mean_residual_norm=("residual_norm", "mean"),
            lambda2_false_positive_rate=("lambda2_fp", "mean"),
            p_mrw_false_positive_rate=("p_mrw_fp", "mean"),
            mrw_mae_H=("pred_H", lambda s: float(np.mean(np.abs(s - pred_df.loc[s.index, "true_H"])))),
            mrw_mae_lambda2=("pred_lambda2", lambda s: float(np.mean(np.abs(s - pred_df.loc[s.index, "true_lambda2"])))),
            mean_zeta_mae=("zeta_mae", "mean"),
            mean_f_alpha_mae=("f_alpha_mae", "mean"),
        )
        .reset_index()
    )
    pair_df = pred_df[pred_df["pair_id"] >= 0]
    pair_gap = math.nan
    if not pair_df.empty:
        pair_pivot = pair_df.pivot_table(index="pair_id", columns="process_type", values=["pred_lambda2", "pred_p_mrw"], aggfunc="first")
        if ("pred_lambda2", "MRW") in pair_pivot.columns and ("pred_lambda2", "Shuffled MRW") in pair_pivot.columns:
            pair_gap = float((pair_pivot[("pred_lambda2", "MRW")] - pair_pivot[("pred_lambda2", "Shuffled MRW")]).mean())
    metrics = {
        "loss": float(np.mean(losses)) if losses else math.nan,
        "mrw_mae_H": float(np.mean(np.abs(pred_df.loc[pred_df["process_type"] == "MRW", "pred_H"] - pred_df.loc[pred_df["process_type"] == "MRW", "true_H"]))) if (pred_df["process_type"] == "MRW").any() else math.nan,
        "mrw_mae_lambda2": float(np.mean(np.abs(pred_df.loc[pred_df["process_type"] == "MRW", "pred_lambda2"] - pred_df.loc[pred_df["process_type"] == "MRW", "true_lambda2"]))) if (pred_df["process_type"] == "MRW").any() else math.nan,
        "mrw_mae_zeta": float(pred_df.loc[pred_df["process_type"] == "MRW", "zeta_mae"].mean()) if (pred_df["process_type"] == "MRW").any() else math.nan,
        "mrw_mae_f_alpha": float(pred_df.loc[pred_df["process_type"] == "MRW", "f_alpha_mae"].mean()) if (pred_df["process_type"] == "MRW").any() else math.nan,
        "mrw_vs_shuffled_lambda2_gap": pair_gap,
        "student_t_lambda2": float(pred_df.loc[pred_df["process_type"] == "iid Student-t", "pred_lambda2"].mean()) if (pred_df["process_type"] == "iid Student-t").any() else math.nan,
        "gaussian_lambda2": float(pred_df.loc[pred_df["process_type"] == "iid Gaussian", "pred_lambda2"].mean()) if (pred_df["process_type"] == "iid Gaussian").any() else math.nan,
        "garch_lambda2": float(pred_df.loc[pred_df["process_type"] == "GARCH(1,1)", "pred_lambda2"].mean()) if (pred_df["process_type"] == "GARCH(1,1)").any() else math.nan,
        "regime_lambda2": float(pred_df.loc[pred_df["process_type"] == "Regime-switching Gaussian", "pred_lambda2"].mean()) if (pred_df["process_type"] == "Regime-switching Gaussian").any() else math.nan,
    }
    return metrics, pred_df, process_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train robust anti-confounded CMIN.")
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--num-train", type=int, default=6000)
    parser.add_argument("--num-val", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ambiguous-label-value", type=float, default=0.25)
    parser.add_argument("--strict-negative-labels", action="store_true")
    parser.add_argument("--param-weight", type=float, default=1.0)
    parser.add_argument("--spectrum-weight", type=float, default=0.5)
    parser.add_argument("--validity-weight", type=float, default=1.0)
    parser.add_argument("--neg-lambda-weight", type=float, default=1.0)
    parser.add_argument("--pair-weight", type=float, default=1.0)
    parser.add_argument("--mismatch-weight", type=float, default=0.0)
    parser.add_argument("--constraint-weight", type=float, default=0.1)
    args = parser.parse_args()

    cfg = RobustTrainConfig(
        length=args.length,
        num_train=args.num_train,
        num_val=args.num_val,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu",
        ambiguous_label_value=args.ambiguous_label_value,
        strict_negative_labels=args.strict_negative_labels,
        param_weight=args.param_weight,
        spectrum_weight=args.spectrum_weight,
        validity_weight=args.validity_weight,
        neg_lambda_weight=args.neg_lambda_weight,
        pair_weight=args.pair_weight,
        mismatch_weight=args.mismatch_weight,
        constraint_weight=args.constraint_weight,
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    _set_seed(cfg.seed)
    train_dataset = _make_dataset(
        AntiConfoundedConfig(
            length=cfg.length,
            num_samples=cfg.num_train,
            seed=cfg.seed,
            ambiguous_label_value=cfg.ambiguous_label_value,
            strict_negative_labels=cfg.strict_negative_labels,
        )
    )
    val_dataset = _make_dataset(
        AntiConfoundedConfig(
            length=cfg.length,
            num_samples=cfg.num_val,
            seed=cfg.seed + 1,
            ambiguous_label_value=cfg.ambiguous_label_value,
            strict_negative_labels=cfg.strict_negative_labels,
        )
    )
    train_loader = _make_loader(train_dataset, cfg.batch_size, shuffle=True)
    val_loader = _make_loader(val_dataset, cfg.batch_size, shuffle=False)

    model = CMINRegressor().to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history_rows: list[dict[str, float]] = []
    nan_occurred = False
    best_val = float("inf")
    best_state = None
    best_pred = None
    best_summary = None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            (
                x,
                process_code,
                is_mrw,
                h_true,
                lambda2_true,
                target_lambda2,
                target_p_mrw,
                target_mismatch,
                pair_id,
                sample_id,
                zeta_true,
                f_true,
            ) = _batch_to_device(batch, cfg.device)
            del sample_id
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = robust_inverse_loss(
                output=out,
                process_code=process_code,
                is_mrw=is_mrw,
                h_true=h_true,
                lambda2_true=lambda2_true,
                target_lambda2=target_lambda2,
                target_p_mrw=target_p_mrw,
                target_mismatch=target_mismatch,
                zeta_true=zeta_true,
                f_true=f_true,
                pair_id=pair_id,
                param_weight=cfg.param_weight,
                spectrum_weight=cfg.spectrum_weight,
                validity_weight=cfg.validity_weight,
                neg_lambda_weight=cfg.neg_lambda_weight,
                pair_weight=cfg.pair_weight,
                mismatch_weight=cfg.mismatch_weight,
                constraint_weight=cfg.constraint_weight,
            )
            if torch.isnan(loss.total):
                nan_occurred = True
                break
            loss.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.total.detach().cpu()))
        val_metrics, val_pred, val_process = _collect_eval(model, val_loader, cfg)
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)) if train_losses else math.nan,
                "val_loss": val_metrics["loss"],
                "mrw_mae_H": val_metrics["mrw_mae_H"],
                "mrw_mae_lambda2": val_metrics["mrw_mae_lambda2"],
                "mrw_mae_zeta": val_metrics["mrw_mae_zeta"],
                "mrw_mae_f_alpha": val_metrics["mrw_mae_f_alpha"],
                "mrw_vs_shuffled_lambda2_gap": val_metrics["mrw_vs_shuffled_lambda2_gap"],
                "student_t_lambda2": val_metrics["student_t_lambda2"],
                "gaussian_lambda2": val_metrics["gaussian_lambda2"],
                "regime_lambda2": val_metrics["regime_lambda2"],
            }
        )
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_state = {
                "model_state_dict": model.state_dict(),
                "config": asdict(cfg),
                "model_name": "cmin_robust_synthetic",
                "best_epoch": epoch,
            }
            best_pred = val_pred.copy()
            best_summary = val_process.copy()
        if nan_occurred:
            break

    if best_state is None:
        raise RuntimeError("Robust CMIN training failed before checkpoint creation.")

    torch.save(best_state, CHECKPOINT_PATH)
    history = pd.DataFrame(history_rows)
    history_path = TABLE_DIR / "train_history.csv"
    history.to_csv(history_path, index=False)
    val_pred_path = TABLE_DIR / "val_predictions.csv"
    best_pred.to_csv(val_pred_path, index=False)
    val_process_path = TABLE_DIR / "val_by_process.csv"
    best_summary.to_csv(val_process_path, index=False)

    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    ax.plot(history["epoch"], history["train_loss"], label="train")
    ax.plot(history["epoch"], history["val_loss"], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("CMIN-Robust training")
    ax.legend()
    loss_path = FIG_DIR / "loss_curve.png"
    fig.savefig(loss_path, dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    best_pred.boxplot(column="pred_lambda2", by="process_type", ax=ax, grid=False, rot=35)
    ax.set_title("Predicted lambda2 by process")
    ax.set_xlabel("")
    ax.set_ylabel("pred_lambda2")
    fig.suptitle("")
    lambda_fig = FIG_DIR / "lambda2_by_process.png"
    fig.savefig(lambda_fig, dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    best_pred.boxplot(column="pred_p_mrw", by="process_type", ax=ax, grid=False, rot=35)
    ax.set_title("Predicted p_MRW by process")
    ax.set_xlabel("")
    ax.set_ylabel("p_MRW")
    fig.suptitle("")
    pmrw_fig = FIG_DIR / "p_mrw_by_process.png"
    fig.savefig(pmrw_fig, dpi=220)
    plt.close(fig)

    last = history.iloc[-1].to_dict()
    summary_lines = [
        "# CMIN-Robust Training Summary",
        "",
        "## Config",
        "",
        *[f"- `{k}`: `{v}`" for k, v in asdict(cfg).items()],
        "",
        "## Final Metrics",
        "",
        *[f"- `{k}`: `{v}`" for k, v in last.items()],
        "",
        "## Best Validation By Process",
        "",
        best_summary.to_csv(index=False),
        "",
        "## Outputs",
        "",
        f"- Checkpoint: `{CHECKPOINT_PATH.relative_to(ROOT)}`",
        f"- Train history: `{history_path.relative_to(ROOT)}`",
        f"- Validation predictions: `{val_pred_path.relative_to(ROOT)}`",
        f"- Validation by process: `{val_process_path.relative_to(ROOT)}`",
        f"- Loss curve: `{loss_path.relative_to(ROOT)}`",
        f"- Lambda2 figure: `{lambda_fig.relative_to(ROOT)}`",
        f"- p_MRW figure: `{pmrw_fig.relative_to(ROOT)}`",
        f"- NaN occurred: `{nan_occurred}`",
    ]
    summary_path = REPORT_DIR / "cmin_robust_training_summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    meta = {
        "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)),
        "history_csv": str(history_path.relative_to(ROOT)),
        "val_predictions": str(val_pred_path.relative_to(ROOT)),
        "val_by_process": str(val_process_path.relative_to(ROOT)),
        "summary": str(summary_path.relative_to(ROOT)),
        "nan_occurred": nan_occurred,
    }
    (REPORT_DIR / "cmin_robust_training_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
