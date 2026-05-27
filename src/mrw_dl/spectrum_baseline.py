from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from .baselines import legendre_spectrum_from_zeta, true_mrw_zeta
from .data import (
    DEFAULT_SPLIT_PATH,
    RAW_DATA_PATH,
    load_dataset,
    load_splits,
    make_splits,
    save_splits,
    select_features,
    standardize_from_train,
)


@dataclass
class SpectrumTrainConfig:
    model_name: str = "resnet"
    target_mode: str = "params"
    feature_key: str = "dx"
    standardization: str = "pointwise"
    batch_size: int = 64
    epochs: int = 60
    lr: float = 3e-4
    weight_decay: float = 1e-4
    hidden_dims: tuple[int, ...] = (512, 256, 128)
    dropout: float = 0.1
    seed: int = 2026
    split_path: str = "data/processed/splits_robust_fgn_4800.npz"
    output_dir: str = "outputs/dl_spectrum_mlp"
    target_indices: tuple[int, int] = (0, 3)
    q_min: float = 0.5
    q_max: float = 3.0
    q_count: int = 11
    early_stopping_patience: int = 10
    scheduler_patience: int = 4
    scheduler_factor: float = 0.5
    aux_zeta_weight: float = 0.3
    aux_f_weight: float = 0.2
    aux_summary_weight: float = 0.5
    aux_param_weight: float = 0.3
    lambda_param_weight: float = 1.0
    h_param_weight: float = 1.0
    mechanism_aux_weight: float = 0.05
    concavity_weight: float = 0.02
    zeta_consistency_weight: float = 0.0
    identifiability_loss_weight: float = 0.0
    identifiability_mode: str = "matched_h_lambda"
    pair_h_tolerance: float = 0.04
    pair_lambda_tolerance: float = 0.012
    pair_lambda_delta: float = 0.04
    pair_h_delta: float = 0.12
    pair_latent_margin: float = 0.2


def _set_seed(seed: int) -> None:
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_loaders(config: SpectrumTrainConfig):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    bundle = load_dataset(RAW_DATA_PATH)
    split_path = Path(config.split_path)
    if split_path.exists():
        splits = load_splits(split_path)
        expected = bundle.dx.shape[0]
        actual = sum(len(v) for v in splits.values())
        if actual != expected:
            splits = make_splits(bundle.dx.shape[0], seed=config.seed)
            save_splits(splits, split_path)
    else:
        splits = make_splits(bundle.dx.shape[0], seed=config.seed)
        save_splits(splits, split_path)

    features = select_features(bundle, config.feature_key)
    features, feature_mean, feature_std = standardize_from_train(
        features,
        splits["train_idx"],
        mode=config.standardization,
    )

    q_vals = np.linspace(config.q_min, config.q_max, config.q_count, dtype=np.float32)
    params = bundle.params[:, list(config.target_indices)].astype(np.float32)
    if config.target_mode == "params":
        targets = params
    elif config.target_mode == "zeta_aux":
        zeta_targets = np.stack(
            [true_mrw_zeta(q_vals, h=float(row[1]), lambda2=float(row[0])) for row in params],
            axis=0,
        ).astype(np.float32)
        targets = np.concatenate([zeta_targets, params], axis=1).astype(np.float32)
    else:
        raise ValueError(f"Unsupported target_mode: {config.target_mode}")
    target_mean = targets[splits["train_idx"]].mean(axis=0, keepdims=True)
    target_std = np.maximum(targets[splits["train_idx"]].std(axis=0, keepdims=True), 1e-6)
    targets_norm = ((targets - target_mean) / target_std).astype(np.float32)

    loaders = {}
    for split_name, split_idx in splits.items():
        x_tensor = torch.from_numpy(features[split_idx])
        y_tensor = torch.from_numpy(targets_norm[split_idx])
        raw_tensor = torch.from_numpy(targets[split_idx])
        ds = TensorDataset(x_tensor, y_tensor, raw_tensor)
        loaders[split_name] = DataLoader(
            ds,
            batch_size=config.batch_size,
            shuffle=(split_name == "train_idx"),
        )

    return bundle, loaders, feature_mean, feature_std, target_mean.astype(np.float32), target_std.astype(np.float32)


def _predict_all(model, loader, device, target_mean, target_std):
    import torch

    preds = []
    targets = []
    with torch.no_grad():
        for xb, _, y_raw in loader:
            xb = xb.to(device)
            pred_norm = model(xb).cpu().numpy()
            pred = pred_norm * target_std + target_mean
            preds.append(pred)
            targets.append(y_raw.numpy())
    return np.concatenate(preds, axis=0), np.concatenate(targets, axis=0)


def _parameter_metrics(y_true: np.ndarray, y_pred: np.ndarray, names: List[str]) -> Dict[str, Dict[str, float]]:
    err = y_pred - y_true
    metrics = {}
    for i, name in enumerate(names):
        metrics[name] = {
            "mae": float(np.mean(np.abs(err[:, i]))),
            "rmse": float(np.sqrt(np.mean(err[:, i] ** 2))),
            "bias": float(np.mean(err[:, i])),
        }
    return metrics


def _weighted_parameter_loss(
    pred_norm: "torch.Tensor",
    target_norm: "torch.Tensor",
    lambda_weight: float,
    h_weight: float,
) -> "torch.Tensor":
    weights = pred_norm.new_tensor([lambda_weight, h_weight]).view(1, -1)
    return (weights * (pred_norm - target_norm).square()).mean()


def _zeta_from_params_torch(params: "torch.Tensor", q_vals: "torch.Tensor") -> "torch.Tensor":
    lambda2 = params[:, 0:1]
    h = params[:, 1:2]
    q = q_vals.view(1, -1)
    return q * h - 0.5 * lambda2 * q * (q - 2.0)


def _f_alpha_from_zeta_torch(zeta: "torch.Tensor", q_vals: "torch.Tensor") -> "torch.Tensor":
    import torch

    dq = q_vals[1:] - q_vals[:-1]
    alpha_parts = []
    alpha_parts.append((zeta[:, 1] - zeta[:, 0]) / dq[0])
    alpha_parts.append((zeta[:, 2:] - zeta[:, :-2]) / (q_vals[2:] - q_vals[:-2]).view(1, -1))
    alpha_parts.append((zeta[:, -1] - zeta[:, -2]) / dq[-1])
    alpha = torch.cat([part.unsqueeze(1) if part.ndim == 1 else part for part in alpha_parts], dim=1)
    return q_vals.view(1, -1) * alpha - zeta + 1.0


def _spectrum_summary_from_params_torch(params: "torch.Tensor", q_vals: "torch.Tensor") -> "torch.Tensor":
    import torch

    lambda2 = params[:, 0:1]
    h = params[:, 1:2]
    alpha_peak = h + lambda2
    width = lambda2 * (q_vals.max() - q_vals.min())
    return torch.cat([alpha_peak, width], dim=1)


def _concavity_penalty(zeta_pred: "torch.Tensor") -> "torch.Tensor":
    import torch

    if zeta_pred.shape[1] < 3:
        return torch.zeros((), dtype=zeta_pred.dtype, device=zeta_pred.device)
    second_diff = zeta_pred[:, 2:] - 2.0 * zeta_pred[:, 1:-1] + zeta_pred[:, :-2]
    return torch.relu(second_diff).square().mean()


def _pairwise_identifiability_loss(
    pred_norm: "torch.Tensor",
    true_raw: "torch.Tensor",
    model,
    mode: str,
    h_tolerance: float,
    lambda_tolerance: float,
    lambda_delta: float,
    h_delta: float,
    latent_margin: float,
) -> "torch.Tensor":
    import torch
    import torch.nn.functional as F

    if pred_norm.shape[0] < 2:
        return pred_norm.new_zeros(())

    lambda_true = true_raw[:, 0]
    h_true = true_raw[:, 1]
    lambda_diff = lambda_true[:, None] - lambda_true[None, :]
    h_diff = h_true[:, None] - h_true[None, :]
    upper = torch.triu(torch.ones_like(lambda_diff, dtype=torch.bool), diagonal=1)

    if mode == "matched_h_lambda":
        ranking_mask = upper & (h_diff.abs() <= h_tolerance) & (lambda_diff.abs() >= lambda_delta)
        close_mask = upper & (h_diff.abs() <= h_tolerance) & (lambda_diff.abs() <= lambda_tolerance)
        pred_diff = pred_norm[:, 0:1] - pred_norm[:, 0:1].T
        signed_margin = torch.sign(lambda_diff[ranking_mask]) * pred_diff[ranking_mask]
    elif mode == "matched_lambda_h":
        ranking_mask = upper & (lambda_diff.abs() <= lambda_tolerance) & (h_diff.abs() >= h_delta)
        close_mask = upper & (lambda_diff.abs() <= lambda_tolerance) & (h_diff.abs() <= h_tolerance)
        pred_diff = pred_norm[:, 1:2] - pred_norm[:, 1:2].T
        signed_margin = torch.sign(h_diff[ranking_mask]) * pred_diff[ranking_mask]
    else:
        raise ValueError(f"Unsupported identifiability_mode: {mode}")

    if ranking_mask.sum() == 0:
        return pred_norm.new_zeros(())

    # Pairwise order loss: among controlled pairs, the predicted parameter order
    # should match the true discriminative factor order.
    ranking_loss = F.softplus(-signed_margin).mean()

    diagnostics = model.latent_diagnostics() if hasattr(model, "latent_diagnostics") else {}
    latent_key = "z_curvature_train" if mode == "matched_h_lambda" else "z_slope_train"
    z = diagnostics.get(latent_key)
    if z is None:
        return ranking_loss

    z_norm = F.normalize(z, dim=1)
    latent_dist = 1.0 - z_norm @ z_norm.T
    far_dist = latent_dist[ranking_mask].mean()
    if close_mask.sum() == 0:
        latent_loss = torch.relu(latent_margin - far_dist)
    else:
        close_dist = latent_dist[close_mask].mean()
        latent_loss = torch.relu(latent_margin + close_dist - far_dist)
    return ranking_loss + 0.25 * latent_loss


def _spectrum_metrics(y_true: np.ndarray, y_pred: np.ndarray, q_vals: np.ndarray) -> Dict[str, float]:
    zeta_true = np.stack([true_mrw_zeta(q_vals, h=row[1], lambda2=row[0]) for row in y_true], axis=0)
    zeta_pred = np.stack([true_mrw_zeta(q_vals, h=row[1], lambda2=row[0]) for row in y_pred], axis=0)

    alpha_true = []
    f_true = []
    alpha_pred = []
    f_pred = []
    for i in range(zeta_true.shape[0]):
        a_t, f_t = legendre_spectrum_from_zeta(q_vals, zeta_true[i])
        a_p, f_p = legendre_spectrum_from_zeta(q_vals, zeta_pred[i])
        alpha_true.append(a_t)
        f_true.append(f_t)
        alpha_pred.append(a_p)
        f_pred.append(f_p)

    alpha_true = np.stack(alpha_true, axis=0)
    f_true = np.stack(f_true, axis=0)
    alpha_pred = np.stack(alpha_pred, axis=0)
    f_pred = np.stack(f_pred, axis=0)

    return {
        "zeta_mae": float(np.mean(np.abs(zeta_true - zeta_pred))),
        "zeta_rmse": float(np.sqrt(np.mean((zeta_true - zeta_pred) ** 2))),
        "f_mae": float(np.mean(np.abs(f_true - f_pred))),
        "f_rmse": float(np.sqrt(np.mean((f_true - f_pred) ** 2))),
        "alpha_mae": float(np.mean(np.abs(alpha_true - alpha_pred))),
    }


def _spectrum_metrics_from_zeta(
    zeta_true: np.ndarray,
    zeta_pred: np.ndarray,
    q_vals: np.ndarray,
) -> Dict[str, float]:
    alpha_true = []
    f_true = []
    alpha_pred = []
    f_pred = []
    for i in range(zeta_true.shape[0]):
        a_t, f_t = legendre_spectrum_from_zeta(q_vals, zeta_true[i])
        a_p, f_p = legendre_spectrum_from_zeta(q_vals, zeta_pred[i])
        alpha_true.append(a_t)
        f_true.append(f_t)
        alpha_pred.append(a_p)
        f_pred.append(f_p)

    alpha_true = np.stack(alpha_true, axis=0)
    f_true = np.stack(f_true, axis=0)
    alpha_pred = np.stack(alpha_pred, axis=0)
    f_pred = np.stack(f_pred, axis=0)

    return {
        "zeta_mae": float(np.mean(np.abs(zeta_true - zeta_pred))),
        "zeta_rmse": float(np.sqrt(np.mean((zeta_true - zeta_pred) ** 2))),
        "f_mae": float(np.mean(np.abs(f_true - f_pred))),
        "f_rmse": float(np.sqrt(np.mean((f_true - f_pred) ** 2))),
        "alpha_mae": float(np.mean(np.abs(alpha_true - alpha_pred))),
    }


def train_spectrum_baseline(config: SpectrumTrainConfig) -> Dict[str, object]:
    import torch
    from torch import nn

    from .models import (
        CNN1DRegressor,
        PhysicsHybridCNNRegressor,
        MLPRegressor,
        LMMICurvatureRegressor,
        LMMINetRegressor,
        PCSMINRegressor,
        PCSMINV2Regressor,
        PCSMINV3Regressor,
        PhysicsScaleNetRegressor,
        ResNet1DRegressor,
        ScaleInvariantCNNRegressor,
        WaveletPhysicsHybridRegressor,
    )

    _set_seed(config.seed)
    bundle, loaders, feature_mean, feature_std, target_mean, target_std = _build_loaders(config)

    first_batch = next(iter(loaders["train_idx"]))
    input_dim = int(first_batch[0].shape[1])
    output_dim = int(first_batch[1].shape[1])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.model_name == "mlp":
        model = MLPRegressor(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
        ).to(device)
    elif config.model_name == "cnn":
        model = CNN1DRegressor(output_dim=output_dim, dropout=config.dropout).to(device)
    elif config.model_name == "resnet":
        model = ResNet1DRegressor(output_dim=output_dim, dropout=config.dropout).to(device)
    elif config.model_name in {"scale_cnn", "scale_invariant_cnn"}:
        model = ScaleInvariantCNNRegressor(output_dim=output_dim, dropout=config.dropout).to(device)
    elif config.model_name in {"physics_scale_net", "psn"}:
        model = PhysicsScaleNetRegressor(output_dim=output_dim, dropout=config.dropout).to(device)
    elif config.model_name in {"physics_hybrid_cnn", "ph_cnn", "zeta_physics_hybrid"}:
        model = PhysicsHybridCNNRegressor(output_dim=output_dim, dropout=config.dropout).to(device)
    elif config.model_name in {"wavelet_physics_hybrid", "wph"}:
        model = WaveletPhysicsHybridRegressor(output_dim=output_dim, dropout=config.dropout).to(device)
    elif config.model_name in {"pc_smin", "pcsmin"}:
        model = PCSMINRegressor(output_dim=output_dim, dropout=config.dropout).to(device)
    elif config.model_name in {"pc_smin_v2", "pcsmin_v2"}:
        model = PCSMINV2Regressor(output_dim=output_dim, dropout=config.dropout).to(device)
    elif config.model_name == "pc_smin_v2_no_gate":
        model = PCSMINV2Regressor(
            output_dim=output_dim,
            dropout=config.dropout,
            use_raw_gates=False,
        ).to(device)
    elif config.model_name in {"pc_smin_v3", "pcsmin_v3"}:
        model = PCSMINV3Regressor(output_dim=output_dim, dropout=config.dropout).to(device)
    elif config.model_name in {"lmmi_net", "lmmi", "lmmi_ident", "lmmi_ident_h_control"}:
        model = LMMINetRegressor(output_dim=output_dim, dropout=config.dropout).to(device)
    elif config.model_name in {"lmmi_curvature", "lmmi_net_curvature"}:
        model = LMMICurvatureRegressor(output_dim=output_dim, dropout=config.dropout).to(device)
    elif config.model_name == "pc_smin_no_wavelet":
        model = PCSMINRegressor(
            output_dim=output_dim,
            dropout=config.dropout,
            use_wavelet_cumulants=False,
        ).to(device)
    elif config.model_name == "pc_smin_no_corr":
        model = PCSMINRegressor(
            output_dim=output_dim,
            dropout=config.dropout,
            use_cross_scale_corr=False,
        ).to(device)
    elif config.model_name == "pc_smin_no_raw":
        model = PCSMINRegressor(
            output_dim=output_dim,
            dropout=config.dropout,
            use_raw_tcn=False,
        ).to(device)
    else:
        raise ValueError(f"Unsupported model_name: {config.model_name}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
    )
    loss_fn = nn.MSELoss()
    q_vals = np.linspace(config.q_min, config.q_max, config.q_count, dtype=np.float64)
    q_tensor = torch.tensor(q_vals, dtype=torch.float32, device=device)
    target_mean_t = torch.tensor(target_mean, dtype=torch.float32, device=device)
    target_std_t = torch.tensor(target_std, dtype=torch.float32, device=device)

    history = {"train_loss": [], "val_loss": [], "train_main_loss": [], "val_main_loss": [], "lr": []}
    best_val = float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0

    for epoch in range(config.epochs):
        model.train()
        train_losses = []
        train_main_losses = []
        for xb, yb, _ in loaders["train_idx"]:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred_norm = model(xb)
            pred_raw = pred_norm * target_std_t + target_mean_t
            true_raw = yb * target_std_t + target_mean_t
            if config.target_mode == "zeta_aux":
                pred_zeta = pred_raw[:, : config.q_count]
                pred_zeta_norm = pred_norm[:, : config.q_count]
                true_zeta_norm = yb[:, : config.q_count]
                pred_params = pred_norm[:, config.q_count:]
                true_params = yb[:, config.q_count:]
                pred_params_raw = pred_raw[:, config.q_count:]
                main_loss = loss_fn(pred_zeta_norm, true_zeta_norm)
                aux_loss = loss_fn(pred_params, true_params)
                consistency_loss = loss_fn(pred_zeta, _zeta_from_params_torch(pred_params_raw, q_tensor))
                loss = (
                    main_loss
                    + config.aux_param_weight * aux_loss
                    + config.concavity_weight * _concavity_penalty(pred_zeta)
                    + config.zeta_consistency_weight * consistency_loss
                )
            else:
                main_loss = _weighted_parameter_loss(
                    pred_norm,
                    yb,
                    lambda_weight=config.lambda_param_weight,
                    h_weight=config.h_param_weight,
                )
                aux_loss = loss_fn(
                    _zeta_from_params_torch(pred_raw, q_tensor),
                    _zeta_from_params_torch(true_raw, q_tensor),
                )
                pred_zeta = _zeta_from_params_torch(pred_raw, q_tensor)
                true_zeta = _zeta_from_params_torch(true_raw, q_tensor)
                f_loss = loss_fn(_f_alpha_from_zeta_torch(pred_zeta, q_tensor), _f_alpha_from_zeta_torch(true_zeta, q_tensor))
                summary_loss = loss_fn(
                    _spectrum_summary_from_params_torch(pred_raw, q_tensor),
                    _spectrum_summary_from_params_torch(true_raw, q_tensor),
                )
                loss = (
                    main_loss
                    + config.aux_zeta_weight * aux_loss
                    + config.aux_f_weight * f_loss
                    + config.aux_summary_weight * summary_loss
                )
            if hasattr(model, "mechanism_auxiliary_loss"):
                loss = loss + config.mechanism_aux_weight * model.mechanism_auxiliary_loss()
            if hasattr(model, "supervised_auxiliary_loss"):
                loss = loss + config.mechanism_aux_weight * model.supervised_auxiliary_loss(yb)
            if config.identifiability_loss_weight > 0.0:
                loss = loss + config.identifiability_loss_weight * _pairwise_identifiability_loss(
                    pred_norm=pred_norm,
                    true_raw=true_raw,
                    model=model,
                    mode=config.identifiability_mode,
                    h_tolerance=config.pair_h_tolerance,
                    lambda_tolerance=config.pair_lambda_tolerance,
                    lambda_delta=config.pair_lambda_delta,
                    h_delta=config.pair_h_delta,
                    latent_margin=config.pair_latent_margin,
                )
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
            train_main_losses.append(float(main_loss.item()))

        model.eval()
        val_losses = []
        val_main_losses = []
        with torch.no_grad():
            for xb, yb, _ in loaders["val_idx"]:
                xb = xb.to(device)
                yb = yb.to(device)
                pred_norm = model(xb)
                pred_raw = pred_norm * target_std_t + target_mean_t
                true_raw = yb * target_std_t + target_mean_t
                if config.target_mode == "zeta_aux":
                    pred_zeta = pred_raw[:, : config.q_count]
                    pred_zeta_norm = pred_norm[:, : config.q_count]
                    true_zeta_norm = yb[:, : config.q_count]
                    pred_params = pred_norm[:, config.q_count:]
                    true_params = yb[:, config.q_count:]
                    pred_params_raw = pred_raw[:, config.q_count:]
                    main_loss = loss_fn(pred_zeta_norm, true_zeta_norm)
                    aux_loss = loss_fn(pred_params, true_params)
                    consistency_loss = loss_fn(pred_zeta, _zeta_from_params_torch(pred_params_raw, q_tensor))
                    val_loss_batch = (
                        main_loss
                        + config.aux_param_weight * aux_loss
                        + config.concavity_weight * _concavity_penalty(pred_zeta)
                        + config.zeta_consistency_weight * consistency_loss
                    )
                else:
                    main_loss = _weighted_parameter_loss(
                        pred_norm,
                        yb,
                        lambda_weight=config.lambda_param_weight,
                        h_weight=config.h_param_weight,
                    )
                    aux_loss = loss_fn(
                        _zeta_from_params_torch(pred_raw, q_tensor),
                        _zeta_from_params_torch(true_raw, q_tensor),
                    )
                    pred_zeta = _zeta_from_params_torch(pred_raw, q_tensor)
                    true_zeta = _zeta_from_params_torch(true_raw, q_tensor)
                    f_loss = loss_fn(
                        _f_alpha_from_zeta_torch(pred_zeta, q_tensor),
                        _f_alpha_from_zeta_torch(true_zeta, q_tensor),
                    )
                    summary_loss = loss_fn(
                        _spectrum_summary_from_params_torch(pred_raw, q_tensor),
                        _spectrum_summary_from_params_torch(true_raw, q_tensor),
                    )
                    val_loss_batch = (
                        main_loss
                        + config.aux_zeta_weight * aux_loss
                        + config.aux_f_weight * f_loss
                        + config.aux_summary_weight * summary_loss
                    )
                if hasattr(model, "mechanism_auxiliary_loss"):
                    val_loss_batch = val_loss_batch + config.mechanism_aux_weight * model.mechanism_auxiliary_loss()
                if hasattr(model, "supervised_auxiliary_loss"):
                    val_loss_batch = val_loss_batch + config.mechanism_aux_weight * model.supervised_auxiliary_loss(yb)
                if config.identifiability_loss_weight > 0.0:
                    val_loss_batch = val_loss_batch + config.identifiability_loss_weight * _pairwise_identifiability_loss(
                        pred_norm=pred_norm,
                        true_raw=true_raw,
                        model=model,
                        mode=config.identifiability_mode,
                        h_tolerance=config.pair_h_tolerance,
                        lambda_tolerance=config.pair_lambda_tolerance,
                        lambda_delta=config.pair_lambda_delta,
                        h_delta=config.pair_h_delta,
                        latent_margin=config.pair_latent_margin,
                    )
                val_losses.append(float(val_loss_batch.item()))
                val_main_losses.append(float(main_loss.item()))

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        train_main_loss = float(np.mean(train_main_losses))
        val_main_loss = float(np.mean(val_main_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_main_loss"].append(train_main_loss)
        history["val_main_loss"].append(val_main_loss)
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                break

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        torch.save(best_state, output_dir / "best_model.pt")
        model.load_state_dict(best_state)

    pred_test_raw, true_test_raw = _predict_all(model, loaders["test_idx"], device, target_mean, target_std)

    if config.target_mode == "zeta_aux":
        zeta_pred_test = pred_test_raw[:, : config.q_count]
        zeta_true_test = true_test_raw[:, : config.q_count]
        pred_test = pred_test_raw[:, config.q_count:]
        true_test = true_test_raw[:, config.q_count:]
        parameter_metrics = _parameter_metrics(true_test, pred_test, ["lambda2", "H"])
        spectrum_metrics = _spectrum_metrics_from_zeta(zeta_true_test, zeta_pred_test, q_vals)
    else:
        zeta_pred_test = np.stack(
            [true_mrw_zeta(q_vals, h=float(row[1]), lambda2=float(row[0])) for row in pred_test_raw],
            axis=0,
        )
        zeta_true_test = np.stack(
            [true_mrw_zeta(q_vals, h=float(row[1]), lambda2=float(row[0])) for row in true_test_raw],
            axis=0,
        )
        pred_test = pred_test_raw
        true_test = true_test_raw
        parameter_metrics = _parameter_metrics(true_test, pred_test, ["lambda2", "H"])
        spectrum_metrics = _spectrum_metrics(true_test, pred_test, q_vals)

    pred_examples = pred_test[:6]
    true_examples = true_test[:6]

    summary = {
        "config": asdict(config),
        "dataset_path": str(RAW_DATA_PATH),
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "parameter_metrics": parameter_metrics,
        "spectrum_metrics": spectrum_metrics,
        "q_vals": q_vals.tolist(),
        "history": history,
        "feature_mean_shape": list(feature_mean.shape),
        "feature_std_shape": list(feature_std.shape),
        "target_mean": target_mean.reshape(-1).tolist(),
        "target_std": target_std.reshape(-1).tolist(),
        "examples": {
            "pred": pred_examples.tolist(),
            "true": true_examples.tolist(),
        },
    }

    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(
        output_dir / "test_predictions.npz",
        pred=pred_test,
        true=true_test,
        q_vals=q_vals,
        zeta_pred=zeta_pred_test,
        zeta_true=zeta_true_test,
    )
    return summary
