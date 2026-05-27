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

from mrw_inverse.data import PROCESS_CODE_TO_NAME, SpectralRepresentationDatasetConfig, generate_spectral_representation_dataset
from mrw_inverse.losses import spectral_representation_loss
from mrw_inverse.models import CMINSRModel


CHECKPOINT_PATH = ROOT / "checkpoints" / "cmin" / "cmin_sr_synthetic.pt"
REPORT_DIR = ROOT / "outputs" / "reports" / "cmin_sr_training"
TABLE_DIR = ROOT / "outputs" / "tables" / "cmin_sr_training"
FIG_DIR = ROOT / "outputs" / "figures" / "cmin_sr_training"


@dataclass
class SRTrainConfig:
    t_choices: tuple[int, ...] = (512, 1024)
    num_train: int = 10000
    num_val: int = 2000
    batch_size: int = 32
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 2026
    device: str = "cpu"
    w_zeta: float = 1.0
    w_mrw_proj: float = 1.0
    w_mono: float = 0.5
    w_validity: float = 1.0
    w_residual: float = 0.5
    w_surrogate: float = 1.0
    w_stability: float = 0.2
    w_constraint: float = 0.1


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _make_dataset(num_samples: int, length: int, seed: int) -> dict[str, np.ndarray]:
    return generate_spectral_representation_dataset(
        SpectralRepresentationDatasetConfig(length=length, num_samples=num_samples, seed=seed)
    )


def _make_loader(dataset: dict[str, np.ndarray], batch_size: int, shuffle: bool) -> DataLoader:
    order = [
        "x",
        "process_code",
        "T",
        "zeta_target",
        "H_true",
        "lambda2_true",
        "target_p_scaling",
        "target_p_mrw",
        "target_residual_level",
        "target_stability",
        "pair_id",
        "sample_id",
    ]
    tensors = []
    for key in order:
        arr = dataset[key]
        dtype = torch.float32
        if key in {"process_code", "T", "pair_id", "sample_id"}:
            dtype = torch.long
        tensors.append(torch.tensor(arr, dtype=dtype))
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle)


def _to_device(batch, device: str):
    return [t.to(device) for t in batch]


def _eval_loaders(model: CMINSRModel, loaders: dict[int, DataLoader], cfg: SRTrainConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, pd.DataFrame]]:
    model.eval()
    metric_rows: list[dict[str, float | int]] = []
    process_rows: list[dict[str, float | int | str]] = []
    preds_by_t: dict[int, pd.DataFrame] = {}
    with torch.no_grad():
        for length, loader in loaders.items():
            rows = []
            losses = []
            for batch in loader:
                (
                    x,
                    process_code,
                    _T,
                    zeta_target,
                    h_true,
                    lambda2_true,
                    target_p_scaling,
                    target_p_mrw,
                    target_residual_level,
                    target_stability,
                    pair_id,
                    sample_id,
                ) = _to_device(batch, cfg.device)
                out = model(x)
                loss = spectral_representation_loss(
                    outputs=out,
                    process_code=process_code,
                    zeta_target=zeta_target,
                    h_true=h_true,
                    lambda2_true=lambda2_true,
                    target_p_scaling=target_p_scaling,
                    target_p_mrw=target_p_mrw,
                    target_residual_level=target_residual_level,
                    target_stability=target_stability,
                    pair_id=pair_id,
                    w_zeta=cfg.w_zeta,
                    w_mrw_proj=cfg.w_mrw_proj,
                    w_mono=cfg.w_mono,
                    w_validity=cfg.w_validity,
                    w_residual=cfg.w_residual,
                    w_surrogate=cfg.w_surrogate,
                    w_stability=cfg.w_stability,
                    w_constraint=cfg.w_constraint,
                )
                losses.append(float(loss.total.cpu()))
                for i in range(x.shape[0]):
                    proc_name = PROCESS_CODE_TO_NAME[int(process_code[i].cpu())]
                    rows.append(
                        {
                            "T": length,
                            "sample_id": int(sample_id[i].cpu()),
                            "pair_id": int(pair_id[i].cpu()),
                            "process_type": proc_name,
                            "true_H": float(h_true[i, 0].cpu()),
                            "pred_H_proj": float(out["H_proj"][i, 0].cpu()),
                            "true_lambda2": float(lambda2_true[i, 0].cpu()),
                            "pred_lambda2_proj": float(out["lambda2_proj"][i, 0].cpu()),
                            "pred_p_scaling": float(out["p_scaling"][i, 0].cpu()),
                            "pred_p_mrw": float(out["p_mrw"][i, 0].cpu()),
                            "pred_residual_norm": float(out["residual_norm"][i, 0].cpu()),
                            "pred_stability": float(out["spectrum_stability"][i, 0].cpu()),
                            "pred_tail_instability": float(out["tail_instability"][i, 0].cpu()),
                            "zeta_mae": float(torch.nanmean(torch.abs(out["zeta_emp"][i] - zeta_target[i])).cpu()),
                        }
                    )
            pred_df = pd.DataFrame(rows)
            preds_by_t[length] = pred_df
            mrw = pred_df["process_type"] == "MRW"
            pair_df = pred_df[pred_df["pair_id"] >= 0]
            p_gap = np.nan
            r_gap = np.nan
            l_gap = np.nan
            if not pair_df.empty:
                pivot = pair_df.pivot_table(index="pair_id", columns="process_type", values=["pred_p_mrw", "pred_residual_norm", "pred_lambda2_proj"], aggfunc="first")
                if ("pred_p_mrw", "MRW") in pivot.columns and ("pred_p_mrw", "Shuffled MRW") in pivot.columns:
                    p_gap = float((pivot[("pred_p_mrw", "MRW")] - pivot[("pred_p_mrw", "Shuffled MRW")]).mean())
                if ("pred_residual_norm", "MRW") in pivot.columns and ("pred_residual_norm", "Shuffled MRW") in pivot.columns:
                    r_gap = float((pivot[("pred_residual_norm", "Shuffled MRW")] - pivot[("pred_residual_norm", "MRW")]).mean())
                if ("pred_lambda2_proj", "MRW") in pivot.columns and ("pred_lambda2_proj", "Shuffled MRW") in pivot.columns:
                    l_gap = float((pivot[("pred_lambda2_proj", "MRW")] - pivot[("pred_lambda2_proj", "Shuffled MRW")]).mean())
            metric_rows.append(
                {
                    "T": length,
                    "train_length": int(length in cfg.t_choices),
                    "loss": float(np.mean(losses)) if losses else math.nan,
                    "mrw_mae_H_proj": float(np.mean(np.abs(pred_df.loc[mrw, "pred_H_proj"] - pred_df.loc[mrw, "true_H"]))) if mrw.any() else math.nan,
                    "mrw_mae_lambda2_proj": float(np.mean(np.abs(pred_df.loc[mrw, "pred_lambda2_proj"] - pred_df.loc[mrw, "true_lambda2"]))) if mrw.any() else math.nan,
                    "mrw_residual_norm": float(pred_df.loc[mrw, "pred_residual_norm"].mean()) if mrw.any() else math.nan,
                    "mrw_p_scaling": float(pred_df.loc[mrw, "pred_p_scaling"].mean()) if mrw.any() else math.nan,
                    "mrw_p_mrw": float(pred_df.loc[mrw, "pred_p_mrw"].mean()) if mrw.any() else math.nan,
                    "mrw_vs_shuffled_p_mrw_gap": p_gap,
                    "mrw_vs_shuffled_residual_gap": r_gap,
                    "mrw_vs_shuffled_lambda2_gap": l_gap,
                }
            )
            proc_summary = (
                pred_df.groupby("process_type")
                .agg(
                    mean_H_proj=("pred_H_proj", "mean"),
                    mean_lambda2_proj=("pred_lambda2_proj", "mean"),
                    mean_p_scaling=("pred_p_scaling", "mean"),
                    mean_p_mrw=("pred_p_mrw", "mean"),
                    mean_residual_norm=("pred_residual_norm", "mean"),
                    mean_stability=("pred_stability", "mean"),
                    mean_tail_instability=("pred_tail_instability", "mean"),
                    mean_zeta_mae=("zeta_mae", "mean"),
                )
                .reset_index()
            )
            proc_summary["T"] = length
            process_rows.extend(proc_summary.to_dict(orient="records"))
    return pd.DataFrame(metric_rows), pd.DataFrame(process_rows), preds_by_t


def main() -> None:
    parser = argparse.ArgumentParser(description="Train first CMIN-SR model.")
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

    cfg = SRTrainConfig(
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
        train_loaders[length] = _make_loader(_make_dataset(per_t_train, length, cfg.seed + 100 * i), cfg.batch_size, shuffle=True)
        val_loaders[length] = _make_loader(_make_dataset(per_t_val, length, cfg.seed + 100 * i + 1), cfg.batch_size, shuffle=False)

    model = CMINSRModel().to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history_rows = []
    best_state = None
    best_val = float("inf")
    best_metrics = None
    best_process = None
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
                    _T,
                    zeta_target,
                    h_true,
                    lambda2_true,
                    target_p_scaling,
                    target_p_mrw,
                    target_residual_level,
                    target_stability,
                    pair_id,
                    _sample_id,
                ) = _to_device(batch, cfg.device)
                optimizer.zero_grad(set_to_none=True)
                out = model(x)
                loss = spectral_representation_loss(
                    outputs=out,
                    process_code=process_code,
                    zeta_target=zeta_target,
                    h_true=h_true,
                    lambda2_true=lambda2_true,
                    target_p_scaling=target_p_scaling,
                    target_p_mrw=target_p_mrw,
                    target_residual_level=target_residual_level,
                    target_stability=target_stability,
                    pair_id=pair_id,
                    w_zeta=cfg.w_zeta,
                    w_mrw_proj=cfg.w_mrw_proj,
                    w_mono=cfg.w_mono,
                    w_validity=cfg.w_validity,
                    w_residual=cfg.w_residual,
                    w_surrogate=cfg.w_surrogate,
                    w_stability=cfg.w_stability,
                    w_constraint=cfg.w_constraint,
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
        eval_metrics, eval_process, preds_by_t = _eval_loaders(model, val_loaders, cfg)
        val_mean = float(eval_metrics["loss"].mean())
        history_row = {"epoch": epoch, "train_loss": float(np.mean(train_losses)) if train_losses else math.nan, "val_loss_mean": val_mean}
        for _, row in eval_metrics.iterrows():
            history_row[f"mrw_p_mrw_T{int(row['T'])}"] = float(row["mrw_p_mrw"])
            history_row[f"shuf_gap_T{int(row['T'])}"] = float(row["mrw_vs_shuffled_p_mrw_gap"])
        history_rows.append(history_row)
        if val_mean < best_val:
            best_val = val_mean
            best_state = {
                "model_state_dict": model.state_dict(),
                "config": asdict(cfg),
                "model_name": "cmin_sr_synthetic",
                "best_epoch": epoch,
            }
            best_metrics = eval_metrics.copy()
            best_process = eval_process.copy()
            best_preds = {k: v.copy() for k, v in preds_by_t.items()}
        if nan_occurred:
            break

    if best_state is None:
        raise RuntimeError("CMIN-SR training failed before checkpoint creation.")
    torch.save(best_state, CHECKPOINT_PATH)

    history = pd.DataFrame(history_rows)
    history_path = TABLE_DIR / "train_history.csv"
    history.to_csv(history_path, index=False)
    metrics_path = TABLE_DIR / "val_metrics_by_T.csv"
    best_metrics.to_csv(metrics_path, index=False)
    process_path = TABLE_DIR / "val_by_process.csv"
    best_process.to_csv(process_path, index=False)

    val_pred_all = pd.concat([df.assign(T=t) for t, df in best_preds.items()], ignore_index=True)
    pred_path = TABLE_DIR / "val_predictions.csv"
    val_pred_all.to_csv(pred_path, index=False)

    fig, ax = plt.subplots(figsize=(6.6, 4.4), constrained_layout=True)
    ax.plot(history["epoch"], history["train_loss"], label="train")
    ax.plot(history["epoch"], history["val_loss_mean"], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("CMIN-SR training")
    ax.legend()
    loss_fig = FIG_DIR / "loss_curve.png"
    fig.savefig(loss_fig, dpi=220)
    plt.close(fig)

    for metric, filename, title in [
        ("mean_p_scaling", "p_scaling_by_process.png", "p_scaling by process"),
        ("mean_p_mrw", "p_mrw_by_process.png", "p_MRW by process"),
        ("mean_lambda2_proj", "lambda2_proj_by_process.png", "lambda2_proj by process"),
        ("mean_residual_norm", "residual_norm_by_process.png", "residual_norm by process"),
    ]:
        fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
        subset = best_process[best_process["T"] == cfg.t_choices[0]].copy()
        ax.bar(subset["process_type"], subset[metric])
        ax.set_title(title + f" (T={cfg.t_choices[0]})")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=35)
        out_fig = FIG_DIR / filename
        fig.savefig(out_fig, dpi=220)
        plt.close(fig)

    summary_lines = [
        "# CMIN-SR Training Summary",
        "",
        "## Config",
        "",
        *[f"- `{k}`: `{v}`" for k, v in asdict(cfg).items()],
        "",
        f"- Final train loss: `{history['train_loss'].iloc[-1]:.6f}`",
        f"- Final val loss: `{history['val_loss_mean'].iloc[-1]:.6f}`",
        f"- NaN occurred: `{nan_occurred}`",
        f"- Checkpoint: `{CHECKPOINT_PATH.relative_to(ROOT)}`",
        "",
        "## Validation Metrics By T",
        "",
        best_metrics.to_csv(index=False),
        "",
        "## Validation By Process",
        "",
        best_process.to_csv(index=False),
    ]
    summary_path = REPORT_DIR / "cmin_sr_training_summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    meta = {
        "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)),
        "history_csv": str(history_path.relative_to(ROOT)),
        "val_predictions": str(pred_path.relative_to(ROOT)),
        "val_by_process": str(process_path.relative_to(ROOT)),
        "summary": str(summary_path.relative_to(ROOT)),
        "nan_occurred": nan_occurred,
    }
    (REPORT_DIR / "cmin_sr_training_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

