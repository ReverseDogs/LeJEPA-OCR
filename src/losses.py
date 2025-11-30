"""
Loss functions for Hunyuan-LeJEPA:
- Epps-Pulley SIGReg enforcing isotropic Gaussian embeddings.
- Combined LeJEPA loss (prediction + SIGReg).
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist


def _distributed_mean(x: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        y = x.clone()
        dist.all_reduce(y, op=dist.ReduceOp.SUM)
        y /= dist.get_world_size()
        return y
    return x


def epps_pulley_sigreg(
    embeddings: torch.Tensor,
    *,
    num_slices: int = 1024,
    t_range: Tuple[float, float] = (-5.0, 5.0),
    num_t: int = 17,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Computes the Epps-Pulley statistic for Sketched Isotropic Gaussian Regularization.

    Args:
        embeddings: Tensor (B, D) of embedding vectors.
        num_slices: Number of random projection slices.
        t_range: Integration range [t_min, t_max].
        num_t: Number of quadrature points (trapezoidal rule).
        seed: Optional deterministic seed to sync slices across devices.
    Returns:
        Scalar tensor with the averaged EP statistic over slices.
    """
    bsz, dim = embeddings.shape
    device = embeddings.device
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)
    proj = torch.randn(dim, num_slices, generator=gen, device=device)
    proj = proj / proj.norm(dim=0, keepdim=True).clamp_min(1e-6)

    proj_vals = embeddings @ proj  # (B, num_slices)

    t = torch.linspace(t_range[0], t_range[1], num_t, device=device)
    # Shape: (B, num_slices, num_t)
    x_t = proj_vals.unsqueeze(-1) * t
    ecf = torch.exp(1j * x_t).mean(dim=0)  # (num_slices, num_t)
    ecf = _distributed_mean(ecf)

    target = torch.exp(-0.5 * t**2)  # (num_t,)
    err = (ecf - target).abs().square() * target
    ep = torch.trapz(err, t, dim=-1) * (bsz * (_world_size_or_one()))
    return ep.mean()


def _world_size_or_one() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def lejepa_loss(
    global_embeds: torch.Tensor,
    all_view_embeds: torch.Tensor,
    lambd: float = 0.05,
    global_step: int = 0,
    sigreg_kwargs: Optional[Dict] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Computes LeJEPA loss combining prediction MSE and SIGReg.

    Args:
        global_embeds: Tensor (B, Vg, D) representing embeddings of global views.
        all_view_embeds: Tensor (B, V, D) for all views (global + context).
        lambd: Regularization weight.
        global_step: Used to seed SIGReg projections for synchronized slices.
        sigreg_kwargs: Optional kwargs forwarded to epps_pulley_sigreg.
    Returns:
        total_loss and a metrics dict containing loss_pred and loss_sigreg.
    """
    sigreg_kwargs = sigreg_kwargs or {}
    mu = global_embeds.mean(dim=1)  # (B, D)
    mse_pred = (all_view_embeds - mu.unsqueeze(1)).square().mean()

    sig_total = global_embeds.new_tensor(0.0)
    for v in all_view_embeds.unbind(dim=1):
        sig_total = sig_total + epps_pulley_sigreg(
            v,
            seed=global_step,
            **sigreg_kwargs,
        )
    sig_reg = sig_total / all_view_embeds.shape[1]

    total = (1.0 - lambd) * mse_pred + lambd * sig_reg
    metrics = {
        "loss_pred": mse_pred.detach(),
        "loss_sigreg": sig_reg.detach(),
        "loss_total": total.detach(),
    }
    return total, metrics
