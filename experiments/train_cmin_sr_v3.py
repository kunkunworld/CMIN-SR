from __future__ import annotations

import argparse
import copy
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
from mrw_inverse.losses import cmin_sr_v3_loss
from mrw_inverse.models import CMINSRv3Model


CHECKPOINT_PATH = ROOT / "checkpoints" / "cmin" / "cmin_sr_v3_synthetic.pt"
REPORT_DIR = ROOT / "outputs" / "reports" / "cmin_sr_v3_training"
TABLE_DIR = ROOT / "outputs" / "tables" / "cmin_sr_v3_training"
FIG_DIR = ROOT / "outputs" / "figures" / "cmin_sr_v3_training"


@dataclass
class SRV3TrainConfig:
    t_choices: tuple[int, ...] = (512, 1024)
    num_train: int = 8000
    num_val: int = 1600
    batch_size: int = 32
    epochs: int = 12
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 2026
    device: str = "cpu"
    w_zeta: float = 1.0
    w_validity: float = 0.5
    w_curved: float = 2.0
    w_mono_head: float = 0.5
    w_boundary: float = 1.0
    w_gate: float = 1.5
    w_mrw_preserve: float = 1.5
    w_mono_reject: float = 1.5
    w_low_lambda: float = 0.5
    w_constraint: float = 0.1


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _make_dataset(num_samples: int, length: int, seed: int) -> dict[str, np.ndarray]:
    return generate_spectral_representation_dataset(
        SpectralRepresentationDatasetConfig(
            length=length,
            num_samples=num_samples,
            seed=seed,
            mrw_ratio=0.20,
            low_lambda2_mrw_ratio=0.10,
            shuffled_mrw_ratio=0.15,
            fgn_ratio=0.15,
            gaussian_ratio=0.10,
            student_t_ratio=0.10,
            garch_ratio=0.10,
            regime_ratio=0.10,
        )
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
        "target_p_mrw_v3",
        "target_p_curved",
        "target_p_mono_v3",
        "target_boundary_mrw",
        "target_residual_level",
        "target_stability",
        "target_curvature_significance",
        "target_mrw_vs_mono_gain",
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


def _evaluate(model: CMINSRv3Model, loaders: dict[int, DataLoader], cfg: SRV3TrainConfig):
    model.eval()
    metric_rows = []
    proc_rows = []
    preds_by_t = {}
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
                    target_p_curved,
                    target_p_mono,
                    target_boundary_mrw,
                    target_residual_level,
                    target_stability,
                    target_curvature_significance,
                    target_mrw_vs_mono_gain,
                    pair_id,
                    sample_id,
                ) = _to_device(batch, cfg.device)
                out = model(x)
                loss = cmin_sr_v3_loss(
                    outputs=out,
                    process_code=process_code,
                    zeta_target=zeta_target,
                    h_true=h_true,
                    lambda2_true=lambda2_true,
                    target_p_scaling=target_p_scaling,
                    target_p_mrw=target_p_mrw,
                    target_p_curved=target_p_curved,
                    target_p_mono=target_p_mono,
                    target_boundary_mrw=target_boundary_mrw,
                    target_residual_level=target_residual_level,
                    target_stability=target_stability,
                    target_curvature_significance=target_curvature_significance,
                    target_mrw_vs_mono_gain=target_mrw_vs_mono_gain,
                    pair_id=pair_id,
                    w_zeta=cfg.w_zeta,
                    w_validity=cfg.w_validity,
                    w_curved=cfg.w_curved,
                    w_mono_head=cfg.w_mono_head,
                    w_boundary=cfg.w_boundary,
                    w_gate=cfg.w_gate,
                    w_mrw_preserve=cfg.w_mrw_preserve,
                    w_mono_reject=cfg.w_mono_reject,
                    w_low_lambda=cfg.w_low_lambda,
                    w_constraint=cfg.w_constraint,
                )
                losses.append(float(loss.total.cpu()))
                for i in range(x.shape[0]):
                    proc = PROCESS_CODE_TO_NAME[int(process_code[i].cpu())]
                    rows.append(
                        {
                            "T": length,
                            "sample_id": int(sample_id[i].cpu()),
                            "pair_id": int(pair_id[i].cpu()),
                            "process_type": proc,
                            "true_H": float(h_true[i, 0].cpu()),
                            "pred_H_proj": float(out["H_proj"][i, 0].cpu()),
                            "pred_H_mono": float(out["H_mono"][i, 0].cpu()),
                            "true_lambda2": float(lambda2_true[i, 0].cpu()),
                            "pred_lambda2_proj": float(out["lambda2_proj"][i, 0].cpu()),
                            "pred_p_scaling": float(out["p_scaling"][i, 0].cpu()),
                            "pred_p_curved": float(out["p_curved"][i, 0].cpu()),
                            "pred_p_mrw": float(out["p_mrw"][i, 0].cpu()),
                            "pred_p_mono": float(out["p_mono"][i, 0].cpu()),
                            "pred_boundary_mrw_score": float(out["boundary_mrw_score"][i, 0].cpu()),
                            "pred_residual_norm": float(out["residual_norm"][i, 0].cpu()),
                            "pred_mono_residual_norm": float(out["mono_residual_norm"][i, 0].cpu()),
                            "pred_gain": float(out["mrw_vs_mono_gain"][i, 0].cpu()),
                            "pred_curvature_sig": float(out["curvature_significance"][i, 0].cpu()),
                            "pred_curvature_score": float(out["curvature_score"][i, 0].cpu()),
                            "pred_linearity_score": float(out["linearity_score"][i, 0].cpu()),
                            "pred_tail_instability": float(out["tail_instability"][i, 0].cpu()),
                        }
                    )
            pred_df = pd.DataFrame(rows)
            preds_by_t[length] = pred_df
            mrw = pred_df["process_type"] == "MRW"
            low_mrw = pred_df["process_type"] == "Low-lambda2 MRW"
            fgn = pred_df["process_type"] == "fGn"
            gauss = pred_df["process_type"] == "iid Gaussian"
            pair_df = pred_df[pred_df["pair_id"] >= 0]
            p_gap = np.nan
            if not pair_df.empty:
                pivot = pair_df.pivot_table(index="pair_id", columns="process_type", values=["pred_p_mrw"], aggfunc="first")
                if ("pred_p_mrw", "MRW") in pivot.columns and ("pred_p_mrw", "Shuffled MRW") in pivot.columns:
                    p_gap = float((pivot[("pred_p_mrw", "MRW")] - pivot[("pred_p_mrw", "Shuffled MRW")]).mean())
            metric_rows.append(
                {
                    "T": length,
                    "train_length": int(length in cfg.t_choices),
                    "loss": float(np.mean(losses)) if losses else math.nan,
                    "mrw_p_mrw": float(pred_df.loc[mrw, "pred_p_mrw"].mean()) if mrw.any() else math.nan,
                    "mrw_p_curved": float(pred_df.loc[mrw, "pred_p_curved"].mean()) if mrw.any() else math.nan,
                    "mrw_mae_H": float(np.mean(np.abs(pred_df.loc[mrw, "pred_H_proj"] - pred_df.loc[mrw, "true_H"]))) if mrw.any() else math.nan,
                    "mrw_mae_lambda2": float(np.mean(np.abs(pred_df.loc[mrw, "pred_lambda2_proj"] - pred_df.loc[mrw, "true_lambda2"]))) if mrw.any() else math.nan,
                    "fgn_p_scaling": float(pred_df.loc[fgn, "pred_p_scaling"].mean()) if fgn.any() else math.nan,
                    "fgn_p_curved": float(pred_df.loc[fgn, "pred_p_curved"].mean()) if fgn.any() else math.nan,
                    "fgn_p_mrw": float(pred_df.loc[fgn, "pred_p_mrw"].mean()) if fgn.any() else math.nan,
                    "gaussian_p_mrw": float(pred_df.loc[gauss, "pred_p_mrw"].mean()) if gauss.any() else math.nan,
                    "low_mrw_p_mrw": float(pred_df.loc[low_mrw, "pred_p_mrw"].mean()) if low_mrw.any() else math.nan,
                    "mrw_vs_shuffled_p_mrw_gap": p_gap,
                }
            )
            proc = (
                pred_df.groupby("process_type")
                .agg(
                    mean_p_scaling=("pred_p_scaling", "mean"),
                    mean_p_curved=("pred_p_curved", "mean"),
                    mean_p_mrw=("pred_p_mrw", "mean"),
                    mean_p_mono=("pred_p_mono", "mean"),
                    mean_boundary_mrw_score=("pred_boundary_mrw_score", "mean"),
                    mean_lambda2_proj=("pred_lambda2_proj", "mean"),
                    mean_residual_norm=("pred_residual_norm", "mean"),
                    mean_mono_residual_norm=("pred_mono_residual_norm", "mean"),
                    mean_gain=("pred_gain", "mean"),
                    mean_curvature_sig=("pred_curvature_sig", "mean"),
                    mean_linearity_score=("pred_linearity_score", "mean"),
                    mean_tail_instability=("pred_tail_instability", "mean"),
                )
                .reset_index()
            )
            proc["T"] = length
            proc_rows.extend(proc.to_dict(orient="records"))
    return pd.DataFrame(metric_rows), pd.DataFrame(proc_rows), preds_by_t


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CMIN-SR v3 with explicit curved-vs-linear calibration.")
    parser.add_argument("--t-choices", nargs="*", type=int, default=[512, 1024])
    parser.add_argument("--num-train", type=int, default=8000)
    parser.add_argument("--num-val", type=int, default=1600)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cfg = SRV3TrainConfig(
        t_choices=tuple(args.t_choices),
        num_train=args.num_train,
        num_val=args.num_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
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
    train_loaders = {}
    val_loaders = {}
    per_t_train = max(cfg.num_train // len(cfg.t_choices), 1)
    per_t_val = max(cfg.num_val // len(cfg.t_choices), 1)
    for i, length in enumerate(cfg.t_choices):
        train_loaders[length] = _make_loader(_make_dataset(per_t_train, length, cfg.seed + 100 * i), cfg.batch_size, True)
        val_loaders[length] = _make_loader(_make_dataset(per_t_val, length, cfg.seed + 100 * i + 1), cfg.batch_size, False)

    model = CMINSRv3Model().to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history_rows = []
    best_val = float("inf")
    best_state = None
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
                    target_p_curved,
                    target_p_mono,
                    target_boundary_mrw,
                    target_residual_level,
                    target_stability,
                    target_curvature_significance,
                    target_mrw_vs_mono_gain,
                    pair_id,
                    _sample_id,
                ) = _to_device(batch, cfg.device)
                optimizer.zero_grad(set_to_none=True)
                out = model(x)
                loss = cmin_sr_v3_loss(
                    outputs=out,
                    process_code=process_code,
                    zeta_target=zeta_target,
                    h_true=h_true,
                    lambda2_true=lambda2_true,
                    target_p_scaling=target_p_scaling,
                    target_p_mrw=target_p_mrw,
                    target_p_curved=target_p_curved,
                    target_p_mono=target_p_mono,
                    target_boundary_mrw=target_boundary_mrw,
                    target_residual_level=target_residual_level,
                    target_stability=target_stability,
                    target_curvature_significance=target_curvature_significance,
                    target_mrw_vs_mono_gain=target_mrw_vs_mono_gain,
                    pair_id=pair_id,
                    w_zeta=cfg.w_zeta,
                    w_validity=cfg.w_validity,
                    w_curved=cfg.w_curved,
                    w_mono_head=cfg.w_mono_head,
                    w_boundary=cfg.w_boundary,
                    w_gate=cfg.w_gate,
                    w_mrw_preserve=cfg.w_mrw_preserve,
                    w_mono_reject=cfg.w_mono_reject,
                    w_low_lambda=cfg.w_low_lambda,
                    w_constraint=cfg.w_constraint,
                )
                if torch.isnan(loss.total):
                    nan_occurred = True
                    break
                loss.total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(float(loss.total.detach().cpu()))
            if nan_occurred:
                break
        eval_metrics, eval_process, preds_by_t = _evaluate(model, val_loaders, cfg)
        val_mean = float(eval_metrics["loss"].mean())
        history_rows.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss_mean": val_mean})
        if val_mean < best_val:
            best_val = val_mean
            best_state = {
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "config": asdict(cfg),
                "model_name": "cmin_sr_v3_synthetic",
                "best_epoch": epoch,
            }
            best_metrics = eval_metrics.copy()
            best_process = eval_process.copy()
            best_preds = {k: v.copy() for k, v in preds_by_t.items()}
        if nan_occurred:
            break

    if best_state is None:
        raise RuntimeError("CMIN-SR v3 training failed before checkpoint creation.")
    torch.save(best_state, CHECKPOINT_PATH)

    history = pd.DataFrame(history_rows)
    history_path = TABLE_DIR / "train_history.csv"
    history.to_csv(history_path, index=False)
    metrics_path = TABLE_DIR / "metrics_by_T.csv"
    best_metrics.to_csv(metrics_path, index=False)
    process_path = TABLE_DIR / "val_by_process.csv"
    best_process.to_csv(process_path, index=False)
    preds_all = pd.concat([df.assign(T=t) for t, df in best_preds.items()], ignore_index=True)
    pred_path = TABLE_DIR / "val_predictions.csv"
    preds_all.to_csv(pred_path, index=False)

    fig, ax = plt.subplots(figsize=(6.5, 4.4), constrained_layout=True)
    ax.plot(history["epoch"], history["train_loss"], label="train")
    ax.plot(history["epoch"], history["val_loss_mean"], label="val")
    ax.legend()
    ax.set_title("CMIN-SR v3 training")
    fig.savefig(FIG_DIR / "loss_curve.png", dpi=220)
    plt.close(fig)

    for metric, file_name in [
        ("mean_p_scaling", "p_scaling_by_process.png"),
        ("mean_p_curved", "p_curved_by_process.png"),
        ("mean_p_mrw", "p_mrw_by_process.png"),
        ("mean_boundary_mrw_score", "boundary_by_process.png"),
        ("mean_residual_norm", "residual_norm_by_process.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8.0, 4.6), constrained_layout=True)
        subset = best_process[best_process["T"] == cfg.t_choices[0]]
        ax.bar(subset["process_type"], subset[metric])
        ax.tick_params(axis="x", rotation=35)
        ax.set_title(metric)
        fig.savefig(FIG_DIR / file_name, dpi=220)
        plt.close(fig)

    summary_lines = [
        "# CMIN-SR v3 Training Summary",
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
        "## Metrics By T",
        "",
        best_metrics.to_csv(index=False),
        "",
        "## By Process",
        "",
        best_process.to_csv(index=False),
    ]
    summary_path = REPORT_DIR / "cmin_sr_v3_training_summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    meta = {
        "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)),
        "history_csv": str(history_path.relative_to(ROOT)),
        "val_predictions": str(pred_path.relative_to(ROOT)),
        "val_by_process": str(process_path.relative_to(ROOT)),
        "summary": str(summary_path.relative_to(ROOT)),
        "nan_occurred": nan_occurred,
    }
    (REPORT_DIR / "cmin_sr_v3_training_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
