from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from mrw_inverse.data import PROCESS_NAME_TO_CODE


@dataclass
class ZetaAlignmentLossOutput:
    total: torch.Tensor
    l_zeta: torch.Tensor
    l_mono_curv: torch.Tensor
    l_mrw_curv: torch.Tensor
    l_proj: torch.Tensor
    l_stab: torch.Tensor
    l_tail: torch.Tensor
    l_smooth: torch.Tensor


def _second_diff(z: torch.Tensor) -> torch.Tensor:
    return z[:, :-2] - 2.0 * z[:, 1:-1] + z[:, 2:]


def zeta_alignment_loss(
    outputs: dict[str, torch.Tensor | str],
    process_code: torch.Tensor,
    zeta_target: torch.Tensor,
    zeta_target_mask: torch.Tensor,
    zeta_weight_by_q: torch.Tensor,
    h_true: torch.Tensor,
    lambda2_true: torch.Tensor,
    target_tail_instability: torch.Tensor,
    w_zeta: float = 2.0,
    w_mono_curv: float = 1.0,
    w_mrw_curv: float = 1.0,
    w_proj: float = 0.5,
    w_stab: float = 0.2,
    w_tail: float = 0.5,
    w_smooth: float = 0.1,
) -> ZetaAlignmentLossOutput:
    zeta_emp = torch.nan_to_num(outputs["zeta_emp"])
    target = torch.nan_to_num(zeta_target)
    mask = torch.nan_to_num(zeta_target_mask).clamp(0.0, 1.0)
    weights = torch.nan_to_num(zeta_weight_by_q).clamp_min(0.0)
    denom = (mask * weights).sum().clamp_min(1.0)
    l_zeta = (torch.abs(zeta_emp - target) * mask * weights).sum() / denom

    codes = process_code.view(-1)
    mono_mask = (codes == PROCESS_NAME_TO_CODE["fGn"]) | (codes == PROCESS_NAME_TO_CODE["iid Gaussian"])
    mrw_mask = (codes == PROCESS_NAME_TO_CODE["MRW"]) | (codes == PROCESS_NAME_TO_CODE["Low-lambda2 MRW"])
    student_mask = codes == PROCESS_NAME_TO_CODE["iid Student-t"]

    zero = zeta_emp.new_zeros(())
    l_mono_curv = _second_diff(zeta_emp[mono_mask]).abs().mean() if mono_mask.any() else zero
    l_mrw_curv = zero
    if mrw_mask.any():
        l_mrw_curv = F.l1_loss(_second_diff(zeta_emp[mrw_mask]), _second_diff(target[mrw_mask]))

    l_proj = zero
    if mrw_mask.any():
        m = mrw_mask.view(-1, 1)
        l_proj = l_proj + F.l1_loss(outputs["H_proj"][m], h_true[m]) + F.l1_loss(outputs["lambda2_proj"][m], lambda2_true[m])
    if mono_mask.any():
        m = mono_mask.view(-1, 1)
        finite_h = torch.isfinite(h_true.view(-1)) & mono_mask
        l_proj = l_proj + outputs["lambda2_proj"][m].abs().mean()
        if finite_h.any():
            mh = finite_h.view(-1, 1)
            l_proj = l_proj + 0.5 * F.l1_loss(outputs["H_mono"][mh], h_true[mh])

    zeta_uncertainty = torch.nan_to_num(outputs.get("zeta_uncertainty", zeta_emp.new_zeros((zeta_emp.shape[0], 1))))
    clean_mask = mono_mask | mrw_mask
    l_stab = zeta_uncertainty[clean_mask.view(-1, 1)].mean() if clean_mask.any() else zero

    tail_instability = torch.nan_to_num(outputs["tail_instability"]).clamp(0.0, 1.0)
    l_tail = F.mse_loss(tail_instability, target_tail_instability)
    if student_mask.any():
        l_tail = l_tail + F.relu(0.5 - tail_instability[student_mask.view(-1, 1)]).mean()

    l_smooth = _second_diff(zeta_emp).abs().mean()
    total = (
        w_zeta * l_zeta
        + w_mono_curv * l_mono_curv
        + w_mrw_curv * l_mrw_curv
        + w_proj * l_proj
        + w_stab * l_stab
        + w_tail * l_tail
        + w_smooth * l_smooth
    )
    return ZetaAlignmentLossOutput(
        total=torch.nan_to_num(total),
        l_zeta=torch.nan_to_num(l_zeta),
        l_mono_curv=torch.nan_to_num(l_mono_curv),
        l_mrw_curv=torch.nan_to_num(l_mrw_curv),
        l_proj=torch.nan_to_num(l_proj),
        l_stab=torch.nan_to_num(l_stab),
        l_tail=torch.nan_to_num(l_tail),
        l_smooth=torch.nan_to_num(l_smooth),
    )
