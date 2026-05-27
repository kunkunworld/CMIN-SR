from __future__ import annotations

import torch
import torch.nn.functional as F


def paired_identifiability_loss(
    embedding: torch.Tensor,
    h_true: torch.Tensor,
    lambda2_true: torch.Tensor,
    same_h_tol: float = 0.03,
    same_lambda_tol: float = 0.01,
    lambda_gap: float = 0.04,
    h_gap: float = 0.10,
    mode: str = "same_h_diff_lambda",
) -> torch.Tensor:
    if embedding.shape[0] < 2:
        return embedding.new_zeros(())

    dist = 1.0 - F.normalize(embedding, dim=1) @ F.normalize(embedding, dim=1).T
    upper = torch.triu(torch.ones_like(dist, dtype=torch.bool), diagonal=1)
    h_diff = h_true[:, None] - h_true[None, :]
    lambda_diff = lambda2_true[:, None] - lambda2_true[None, :]

    if mode == "same_h_diff_lambda":
        far = upper & (h_diff.abs() <= same_h_tol) & (lambda_diff.abs() >= lambda_gap)
        near = upper & (h_diff.abs() <= same_h_tol) & (lambda_diff.abs() <= same_lambda_tol)
    elif mode == "same_lambda_diff_h":
        far = upper & (lambda_diff.abs() <= same_lambda_tol) & (h_diff.abs() >= h_gap)
        near = upper & (lambda_diff.abs() <= same_lambda_tol) & (h_diff.abs() <= same_h_tol)
    else:
        raise ValueError(f"Unknown contrastive mode: {mode}")

    if far.sum() == 0:
        return embedding.new_zeros(())
    far_dist = dist[far].mean()
    if near.sum() == 0:
        return F.relu(0.2 - far_dist)
    near_dist = dist[near].mean()
    return F.relu(0.2 + near_dist - far_dist)
