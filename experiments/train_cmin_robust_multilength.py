from __future__ import annotations

import argparse
import json
import math
import random
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


CHECKPOINT_PATH = ROOT / "checkpoints" / "cmin" / "cmin_robust_multilength.pt"
REPORT_DIR = ROOT / "outputs" / "reports" / "cmin_robust_multilength_training"
TABLE_DIR = ROOT / "outputs" / "tables" / "cmin_robust_multilength_training"
FIG_DIR = ROOT / "outputs" / "figures" / "cmin_robust_multilength_training"


@dataclass
class MultiLengthConfig:
    t_choices: tuple[int, ...] = (512, 1024)
    num_train: int = 10000
    num_val: int = 2000
    batch_size: int = 32
    epochs: int = 20
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
    random.seed(seed)


def _make_dataset(num_samples: int, length: int, seed: int, cfg: MultiLengthConfig) -> dict[str, np.ndarray]:
    return generate_anti_confounded_dataset(
        AntiConfoundedConfig(
            length=length,
            num_samples=num_samples,
            seed=seed,
            ambiguous_label_value=cfg.ambiguous_label_value,
            strict_negative_labels=cfg.strict_negative_labels,
        )
    )


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


def _to_device(batch, device: str):
    return [t.to(device) for t in batch]


def _evaluate_by_length(model: CMINRegressor, loaders: dict[int, DataLoader], cfg: MultiLengthConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, pd.DataFrame]]:
    model.eval()
    metric_rows: list[dict[str, float | int]] = []
    process_rows: list[dict[str, float | int | str]] = []
    predictions_by_t: dict[int, pd.DataFrame] = {}
    with torch.no_grad():
        for length, loader in loaders.items():
            rows = []
            losses = []
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
                ) = _to_device(batch, cfg.device)
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
                            "T": length,
                            "sample_id": int(sample_id[i].cpu()),
                            "pair_id": int(pair_id[i].cpu()),
                            "process_type": proc_name,
                            "is_mrw": float(is_mrw[i, 0].cpu()),
                            "true_H": float(h_true[i, 0].cpu()),
                            "pred_H": float(out.h_hat[i, 0].cpu()),
                            "true_lambda2": float(lambda2_true[i, 0].cpu()),
                            "pred_lambda2": float(out.lambda2_hat[i, 0].cpu()),
                            "pred_p_mrw": float(out.p_mrw[i, 0].cpu()),
                            "zeta_mae": float(torch.mean(torch.abs(out.zeta_hat[i] - zeta_true[i])).cpu()),
                            "f_alpha_mae": float(torch.nanmean(torch.abs(out.f_alpha_hat[i] - f_true[i])).cpu()),
                            "residual_norm": float(residual_norm[i].cpu()),
                        }
                    )
            pred_df = pd.DataFrame(rows)
            predictions_by_t[length] = pred_df
            mrw_mask = pred_df["process_type"] == "MRW"
            pivot = pred_df[pred_df["pair_id"] >= 0].pivot_table(index="pair_id", columns="process_type", values=["pred_lambda2", "pred_p_mrw"], aggfunc="first")
            lambda_gap = math.nan
            p_gap = math.nan
            if ("pred_lambda2", "MRW") in pivot.columns and ("pred_lambda2", "Shuffled MRW") in pivot.columns:
                lambda_gap = float((pivot[("pred_lambda2", "MRW")] - pivot[("pred_lambda2", "Shuffled MRW")]).mean())
            if ("pred_p_mrw", "MRW") in pivot.columns and ("pred_p_mrw", "Shuffled MRW") in pivot.columns:
                p_gap = float((pivot[("pred_p_mrw", "MRW")] - pivot[("pred_p_mrw", "Shuffled MRW")]).mean())
            metric_rows.append(
                {
                    "T": length,
                    "train_length": int(length in cfg.t_choices),
                    "loss": float(np.mean(losses)) if losses else math.nan,
                    "mrw_mae_H": float(np.mean(np.abs(pred_df.loc[mrw_mask, "pred_H"] - pred_df.loc[mrw_mask, "true_H"]))),
                    "mrw_mae_lambda2": float(np.mean(np.abs(pred_df.loc[mrw_mask, "pred_lambda2"] - pred_df.loc[mrw_mask, "true_lambda2"]))),
                    "mrw_mae_zeta": float(pred_df.loc[mrw_mask, "zeta_mae"].mean()),
                    "mrw_mae_f_alpha": float(pred_df.loc[mrw_mask, "f_alpha_mae"].mean()),
                    "student_t_lambda2": float(pred_df.loc[pred_df["process_type"] == "iid Student-t", "pred_lambda2"].mean()),
                    "gaussian_lambda2": float(pred_df.loc[pred_df["process_type"] == "iid Gaussian", "pred_lambda2"].mean()),
                    "garch_lambda2": float(pred_df.loc[pred_df["process_type"] == "GARCH(1,1)", "pred_lambda2"].mean()),
                    "regime_lambda2": float(pred_df.loc[pred_df["process_type"] == "Regime-switching Gaussian", "pred_lambda2"].mean()),
                    "mrw_vs_shuffled_lambda2_gap": lambda_gap,
                    "mrw_vs_shuffled_p_mrw_gap": p_gap,
                    "validity_accuracy_05": float(np.mean(((pred_df["pred_p_mrw"] >= 0.5).astype(float)) == pred_df["is_mrw"])),
                }
            )
            proc_summary = (
                pred_df.groupby("process_type")
                .agg(
                    mean_pred_lambda2=("pred_lambda2", "mean"),
                    mean_pred_p_mrw=("pred_p_mrw", "mean"),
                    mean_pred_H=("pred_H", "mean"),
                    mean_residual_norm=("residual_norm", "mean"),
                )
                .reset_index()
            )
            proc_summary["T"] = length
            process_rows.extend(proc_summary.to_dict(orient="records"))
    return pd.DataFrame(metric_rows), pd.DataFrame(process_rows), predictions_by_t


def main() -> None:
    parser = argparse.ArgumentParser(description="Train multi-length robust CMIN.")
    parser.add_argument("--t-choices", nargs="*", type=int, default=[512, 1024])
    parser.add_argument("--num-train", type=int, default=10000)
    parser.add_argument("--num-val", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cfg = MultiLengthConfig(
        t_choices=tuple(args.t_choices),
        num_train=args.num_train,
        num_val=args.num_val,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu",
    )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    _set_seed(cfg.seed)
    train_loaders: dict[int, DataLoader] = {}
    val_loaders: dict[int, DataLoader] = {}
    per_t_train = max(cfg.num_train // len(cfg.t_choices), 1)
    per_t_val = max(cfg.num_val // len(cfg.t_choices), 1)
    for i, length in enumerate(cfg.t_choices):
        train_ds = _make_dataset(per_t_train, length, cfg.seed + 100 * i, cfg)
        val_ds = _make_dataset(per_t_val, length, cfg.seed + 100 * i + 1, cfg)
        train_loaders[length] = _make_loader(train_ds, cfg.batch_size, shuffle=True)
        val_loaders[length] = _make_loader(val_ds, cfg.batch_size, shuffle=False)

    model = CMINRegressor().to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history_rows = []
    best_val = float("inf")
    best_state = None
    best_eval_metrics = None
    best_eval_process = None
    best_preds = None
    nan_occurred = False

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses = []
        order = list(cfg.t_choices)
        random.shuffle(order)
        for length in order:
            for batch in train_loaders[length]:
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
                ) = _to_device(batch, cfg.device)
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
            if nan_occurred:
                break
        eval_metrics, eval_process, preds_by_t = _evaluate_by_length(model, val_loaders, cfg)
        avg_val = float(eval_metrics["loss"].mean())
        row = {"epoch": epoch, "train_loss": float(np.mean(train_losses)) if train_losses else math.nan, "val_loss_mean": avg_val}
        for _, mrow in eval_metrics.iterrows():
            t = int(mrow["T"])
            row[f"mrw_gap_T{t}"] = float(mrow["mrw_vs_shuffled_lambda2_gap"])
            row[f"validity_acc_T{t}"] = float(mrow["validity_accuracy_05"])
            row[f"student_t_lambda2_T{t}"] = float(mrow["student_t_lambda2"])
        history_rows.append(row)
        if avg_val < best_val:
            best_val = avg_val
            best_state = {
                "model_state_dict": model.state_dict(),
                "config": asdict(cfg),
                "model_name": "cmin_robust_multilength",
                "best_epoch": epoch,
            }
            best_eval_metrics = eval_metrics.copy()
            best_eval_process = eval_process.copy()
            best_preds = {length: df.copy() for length, df in preds_by_t.items()}
        if nan_occurred:
            break

    if best_state is None:
        raise RuntimeError("Multi-length robust training failed before checkpoint creation.")
    torch.save(best_state, CHECKPOINT_PATH)

    history = pd.DataFrame(history_rows)
    history_path = TABLE_DIR / "train_history.csv"
    history.to_csv(history_path, index=False)
    metrics_path = TABLE_DIR / "val_metrics_by_T.csv"
    best_eval_metrics.to_csv(metrics_path, index=False)
    process_path = TABLE_DIR / "val_by_T_process.csv"
    best_eval_process.to_csv(process_path, index=False)
    pred_paths = []
    for length, df in best_preds.items():
        path = TABLE_DIR / f"val_predictions_T{length}.csv"
        df.to_csv(path, index=False)
        pred_paths.append(path)

    fig, ax = plt.subplots(figsize=(6.6, 4.4), constrained_layout=True)
    ax.plot(history["epoch"], history["train_loss"], label="train")
    ax.plot(history["epoch"], history["val_loss_mean"], label="val_mean")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("CMIN-Robust multi-length training")
    ax.legend()
    loss_fig = FIG_DIR / "loss_curve.png"
    fig.savefig(loss_fig, dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    for length in cfg.t_choices:
        subset = best_eval_metrics[best_eval_metrics["T"] == length]
        if not subset.empty:
            ax.scatter([length], subset["mrw_vs_shuffled_lambda2_gap"].iloc[0], label=f"T={length}")
    ax.axhline(0.0, color="0.4", linestyle="--", linewidth=0.8)
    ax.set_xlabel("T")
    ax.set_ylabel("MRW - shuffled lambda2 gap")
    ax.set_title("By-T MRW/shuffled separation")
    gap_fig = FIG_DIR / "mrw_shuffled_gap_by_T.png"
    fig.savefig(gap_fig, dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    for proc in ["iid Gaussian", "iid Student-t", "Regime-switching Gaussian"]:
        sub = best_eval_process[best_eval_process["process_type"] == proc].sort_values("T")
        if not sub.empty:
            ax.plot(sub["T"], sub["mean_pred_lambda2"], marker="o", label=proc)
    ax.set_xlabel("T")
    ax.set_ylabel("mean pred lambda2")
    ax.set_title("False-positive lambda2 by T")
    ax.legend()
    fp_fig = FIG_DIR / "false_positive_lambda2_by_T.png"
    fig.savefig(fp_fig, dpi=220)
    plt.close(fig)

    summary_lines = [
        "# CMIN-Robust Multi-length Training Summary",
        "",
        "## Config",
        "",
        *[f"- `{k}`: `{v}`" for k, v in asdict(cfg).items()],
        "",
        "## Validation Metrics By T",
        "",
        best_eval_metrics.to_csv(index=False),
        "",
        "## Validation By T and Process",
        "",
        best_eval_process.to_csv(index=False),
        "",
        "## Outputs",
        "",
        f"- Checkpoint: `{CHECKPOINT_PATH.relative_to(ROOT)}`",
        f"- Train history: `{history_path.relative_to(ROOT)}`",
        f"- Val metrics by T: `{metrics_path.relative_to(ROOT)}`",
        f"- Val by T/process: `{process_path.relative_to(ROOT)}`",
        f"- Loss curve: `{loss_fig.relative_to(ROOT)}`",
        f"- Gap figure: `{gap_fig.relative_to(ROOT)}`",
        f"- False-positive figure: `{fp_fig.relative_to(ROOT)}`",
        f"- NaN occurred: `{nan_occurred}`",
    ]
    for path in pred_paths:
        summary_lines.append(f"- Predictions: `{path.relative_to(ROOT)}`")
    summary_path = REPORT_DIR / "cmin_robust_multilength_training_summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    meta = {
        "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)),
        "history_csv": str(history_path.relative_to(ROOT)),
        "val_metrics_by_T": str(metrics_path.relative_to(ROOT)),
        "val_by_T_process": str(process_path.relative_to(ROOT)),
        "summary": str(summary_path.relative_to(ROOT)),
        "nan_occurred": nan_occurred,
    }
    (REPORT_DIR / "cmin_robust_multilength_training_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
