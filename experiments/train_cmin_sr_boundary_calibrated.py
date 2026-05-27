from __future__ import annotations

import argparse
import copy
import json
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

from mrw_inverse.data import (
    BoundaryCalibrationDatasetConfig,
    PROCESS_CODE_TO_NAME,
    SpectralRepresentationDatasetConfig,
    generate_boundary_calibration_dataset,
    generate_spectral_representation_dataset,
)
from mrw_inverse.losses import boundary_calibration_loss, cmin_sr_v3_loss
from mrw_inverse.models import CMINSRv3Model


BASE_CHECKPOINT = ROOT / "checkpoints" / "cmin" / "cmin_sr_v3_synthetic.pt"
CHECKPOINT_PATH = ROOT / "checkpoints" / "cmin" / "cmin_sr_calibrated_synthetic.pt"
REPORT_DIR = ROOT / "outputs" / "reports" / "cmin_sr_boundary_calibrated_training"
TABLE_DIR = ROOT / "outputs" / "tables" / "cmin_sr_boundary_calibrated_training"
FIG_DIR = ROOT / "outputs" / "figures" / "cmin_sr_boundary_calibrated_training"


@dataclass
class BoundaryCalTrainConfig:
    t_choices: tuple[int, ...] = (512, 1024)
    num_train: int = 6000
    num_val: int = 1200
    epochs: int = 8
    batch_size: int = 32
    lr: float = 5e-4
    weight_decay: float = 1e-4
    seed: int = 2026
    device: str = "cpu"
    boundary_weight: float = 5.0
    base_checkpoint: str = str(BASE_CHECKPOINT)


MAIN_ORDER = [
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

BOUNDARY_ORDER = [
    "x",
    "process_code",
    "T",
    "zeta_target",
    "H_true",
    "lambda2_true",
    "group_id",
    "rank_curvature_target",
    "target_p_scaling",
    "target_p_curved",
    "target_p_mrw",
    "target_p_mono",
    "target_boundary_mrw",
    "sample_id",
]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _loader(dataset: dict[str, np.ndarray], order: list[str], batch_size: int, shuffle: bool) -> DataLoader:
    tensors = []
    for key in order:
        dtype = torch.long if key in {"process_code", "T", "pair_id", "sample_id", "group_id"} else torch.float32
        tensors.append(torch.tensor(dataset[key], dtype=dtype))
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle)


def _to_device(batch, device: str):
    return [t.to(device) for t in batch]


def _main_dataset(num_samples: int, length: int, seed: int) -> dict[str, np.ndarray]:
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


def _boundary_dataset(num_groups: int, length: int, seed: int) -> dict[str, np.ndarray]:
    return generate_boundary_calibration_dataset(BoundaryCalibrationDatasetConfig(length=length, num_groups=num_groups, seed=seed))


def _eval_model(model: CMINSRv3Model, loaders: dict[int, DataLoader], cfg: BoundaryCalTrainConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    model.eval()
    rows = []
    with torch.no_grad():
        for length, loader in loaders.items():
            for batch in loader:
                x, process_code, *_rest = _to_device(batch, cfg.device)
                out = model(x)
                sample_id = _rest[-1]
                h_true = _rest[2]
                lambda2_true = _rest[3]
                for i in range(x.shape[0]):
                    rows.append(
                        {
                            "T": length,
                            "sample_id": int(sample_id[i].cpu()),
                            "process_type": PROCESS_CODE_TO_NAME[int(process_code[i].cpu())],
                            "true_H": float(h_true[i, 0].cpu()),
                            "true_lambda2": float(lambda2_true[i, 0].cpu()),
                            "pred_p_scaling": float(out["p_scaling"][i, 0].cpu()),
                            "pred_p_curved": float(out["p_curved"][i, 0].cpu()),
                            "pred_p_mrw": float(out["p_mrw"][i, 0].cpu()),
                            "pred_p_mono": float(out["p_mono"][i, 0].cpu()),
                            "pred_boundary_mrw_score": float(out["boundary_mrw_score"][i, 0].cpu()),
                            "pred_residual_norm": float(out["residual_norm"][i, 0].cpu()),
                            "pred_mono_residual_norm": float(out["mono_residual_norm"][i, 0].cpu()),
                            "pred_tail_instability": float(out["tail_instability"][i, 0].cpu()),
                        }
                    )
    pred = pd.DataFrame(rows)
    proc = (
        pred.groupby(["T", "process_type"])
        .agg(
            mean_p_scaling=("pred_p_scaling", "mean"),
            mean_p_curved=("pred_p_curved", "mean"),
            mean_p_mrw=("pred_p_mrw", "mean"),
            mean_p_mono=("pred_p_mono", "mean"),
            mean_boundary_mrw_score=("pred_boundary_mrw_score", "mean"),
            mean_tail_instability=("pred_tail_instability", "mean"),
            mean_residual_norm=("pred_residual_norm", "mean"),
            mean_mono_residual_norm=("pred_mono_residual_norm", "mean"),
        )
        .reset_index()
    )
    return pred, proc


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune CMIN-SR with same-H boundary calibration.")
    parser.add_argument("--num-train", type=int, default=6000)
    parser.add_argument("--num-val", type=int, default=1200)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--base-checkpoint", default=str(BASE_CHECKPOINT))
    args = parser.parse_args()

    cfg = BoundaryCalTrainConfig(
        num_train=args.num_train,
        num_val=args.num_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=args.device if args.device != "cpu" and torch.cuda.is_available() else "cpu",
        base_checkpoint=args.base_checkpoint,
    )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _set_seed(cfg.seed)

    per_t_train = max(cfg.num_train // len(cfg.t_choices), 1)
    per_t_val = max(cfg.num_val // len(cfg.t_choices), 1)
    main_loaders = {}
    boundary_loaders = {}
    val_loaders = {}
    for i, length in enumerate(cfg.t_choices):
        main_loaders[length] = _loader(_main_dataset(per_t_train, length, cfg.seed + 100 * i), MAIN_ORDER, cfg.batch_size, True)
        # Eight samples per group; keep batches group-aligned for ranking losses.
        boundary_groups = max(int(per_t_train * 0.30) // 8, 4)
        boundary_loaders[length] = _loader(_boundary_dataset(boundary_groups, length, cfg.seed + 300 * i), BOUNDARY_ORDER, cfg.batch_size, False)
        val_loaders[length] = _loader(_main_dataset(per_t_val, length, cfg.seed + 100 * i + 1), MAIN_ORDER, cfg.batch_size, False)

    model = CMINSRv3Model().to(cfg.device)
    initialized_from_v3 = False
    ckpt = Path(cfg.base_checkpoint)
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state, strict=False)
        initialized_from_v3 = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history = []
    best_loss = float("inf")
    best_state = None
    best_pred = None
    best_proc = None
    nan_occurred = False

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        for length in cfg.t_choices:
            b_iter = iter(boundary_loaders[length])
            for step, batch in enumerate(main_loaders[length]):
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
                main_loss = cmin_sr_v3_loss(
                    out,
                    process_code,
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
                    w_curved=2.0,
                    w_boundary=1.0,
                    w_gate=1.5,
                    w_mrw_preserve=1.5,
                    w_mono_reject=1.5,
                ).total
                loss = main_loss
                try:
                    b_batch = next(b_iter)
                except StopIteration:
                    b_iter = iter(boundary_loaders[length])
                    b_batch = next(b_iter)
                (
                    bx,
                    b_process_code,
                    _bT,
                    _bzeta_target,
                    _bh_true,
                    b_lambda2_true,
                    b_group_id,
                    b_rank,
                    b_target_p_scaling,
                    b_target_p_curved,
                    b_target_p_mrw,
                    b_target_p_mono,
                    b_target_boundary,
                    _b_sample_id,
                ) = _to_device(b_batch, cfg.device)
                b_out = model(bx)
                b_loss = boundary_calibration_loss(
                    b_out,
                    b_process_code,
                    b_group_id,
                    b_lambda2_true,
                    b_rank,
                    b_target_p_scaling,
                    b_target_p_curved,
                    b_target_p_mrw,
                    b_target_p_mono,
                    b_target_boundary,
                ).total
                loss = loss + cfg.boundary_weight * b_loss
                if torch.isnan(loss):
                    nan_occurred = True
                    break
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            if nan_occurred:
                break
        pred, proc = _eval_model(model, val_loaders, cfg)
        val_score = float(proc["mean_p_mrw"].mean()) if not proc.empty else float("inf")
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_process_mean_p_mrw": val_score})
        if np.mean(losses) < best_loss:
            best_loss = float(np.mean(losses))
            best_state = copy.deepcopy(model.state_dict())
            best_pred = pred.copy()
            best_proc = proc.copy()
        if nan_occurred:
            break

    if best_state is None:
        raise RuntimeError("Boundary calibration failed before checkpoint creation.")
    torch.save(
        {
            "model_state_dict": best_state,
            "config": asdict(cfg),
            "model_name": "cmin_sr_calibrated_synthetic",
            "initialized_from_v3": initialized_from_v3,
        },
        CHECKPOINT_PATH,
    )
    history_df = pd.DataFrame(history)
    history_df.to_csv(TABLE_DIR / "train_history.csv", index=False)
    best_pred.to_csv(TABLE_DIR / "val_predictions.csv", index=False)
    best_proc.to_csv(TABLE_DIR / "val_by_process.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    ax.plot(history_df["epoch"], history_df["train_loss"], marker="o")
    ax.set_title("CMIN-SR boundary calibration fine-tuning")
    ax.set_xlabel("epoch")
    ax.set_ylabel("train loss")
    fig.savefig(FIG_DIR / "loss_curve.png", dpi=220)
    plt.close(fig)

    summary = [
        "# CMIN-SR Boundary-Calibrated Training Summary",
        "",
        f"- Initialized from v3 checkpoint: `{initialized_from_v3}`",
        f"- Checkpoint: `{CHECKPOINT_PATH.relative_to(ROOT)}`",
        f"- NaN occurred: `{nan_occurred}`",
        "",
        "## Config",
        "",
        *[f"- `{k}`: `{v}`" for k, v in asdict(cfg).items()],
        "",
        "## Validation By Process",
        "",
        best_proc.to_csv(index=False),
    ]
    report = REPORT_DIR / "cmin_sr_boundary_calibrated_training_summary.md"
    report.write_text("\n".join(summary), encoding="utf-8")
    meta = {
        "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)),
        "initialized_from_v3": initialized_from_v3,
        "nan_occurred": nan_occurred,
        "report": str(report.relative_to(ROOT)),
    }
    (REPORT_DIR / "cmin_sr_boundary_calibrated_training_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
