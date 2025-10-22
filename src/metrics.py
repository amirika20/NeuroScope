from __future__ import annotations
from typing import Tuple, List
import numpy as np
import torch
import torch.nn as nn
from .linalg import top_hessian_eigenpair
from .ntk import compute_ntk_full, topk_eigs_from_ntk

@torch.no_grad()
def mse_full_range(model: nn.Module, X_vis_t: torch.Tensor, y_true_vis_t: torch.Tensor) -> float:
    y_pred = model(X_vis_t).squeeze(1)
    return float(torch.mean((y_pred - y_true_vis_t) ** 2).item())

def sharpness_top(model: nn.Module, criterion, X_train_t: torch.Tensor, y_train_t: torch.Tensor,
                  iters: int = 20, init_vec=None) -> Tuple[float, torch.Tensor]:
    lam, v = top_hessian_eigenpair(model, criterion, X_train_t, y_train_t, iters=iters, init_vec=init_vec)
    return lam, v

def ntk_topk_eigs(model: nn.Module, X_probe: torch.Tensor, k: int):
    K = compute_ntk_full(model, X_probe)
    w = topk_eigs_from_ntk(K, k)
    return w

@torch.no_grad()
def count_linear_regions_relu_by_activation(
    model: nn.Module,
    X_vis_t: torch.Tensor,
) -> int:
    """
    Exact for 1D ReLU MLPs: counts breaks where any hidden unit flips (z passes 0).
    Returns number of linear regions along the (sorted) X_vis_t line.
    """
    # 1) Collect hidden pre-activations z before each ReLU
    zs: List[torch.Tensor] = []
    acts: List[nn.Module] = []
    layers: List[nn.Module] = []
    for m in model.modules():
        layers.append(m)
        if isinstance(m, nn.ReLU):
            acts.append(m)

    preacts: List[torch.Tensor] = []
    handles = []

    def _make_hook():
        def _hook(mod, inp, out):
            # 'inp' corresponds to pre-activation tensor passed into ReLU
            # Shape: (M, hidden)
            z = inp[0].detach().cpu()
            preacts.append(z)
        return _hook

    for a in acts:
        handles.append(a.register_forward_hook(_make_hook()))

    # 2) Run forward on sorted X
    X = X_vis_t.detach().view(-1, 1)
    vals, _ = torch.sort(X.squeeze(1))
    Xs = vals.view(-1, 1).to(next(model.parameters()).device)
    _ = model(Xs)  # hooks fill 'preacts'

    for h in handles:
        h.remove()

    if not preacts:
        # No ReLUs -> region count is 1 (e.g., tanh nets are smooth)
        return 1

    # 3) Build activation masks (z > 0) for every ReLU layer, concatenate across layers
    #    mask shape per layer: (M, hidden); concat -> (M, total_hidden)
    masks = [ (z > 0).to(torch.int8) for z in preacts ]
    mask_concat = torch.cat(masks, dim=1)  # (M, P_total)

    # 4) Detect pattern flips between consecutive x's: any column differs => a knot
    diffs = (mask_concat[1:] != mask_concat[:-1]).any(dim=1)  # (M-1,)
    knots = int(diffs.sum().item())
    return knots + 1