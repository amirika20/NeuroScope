# sinfit/training.py
from __future__ import annotations
from typing import Callable, List, Tuple
import torch
import torch.nn as nn
from .linalg import _flatten_tensors, flatten_params

@torch.no_grad()
def _flatten_grads(model: nn.Module) -> torch.Tensor:
    return _flatten_tensors([p.grad.detach() for p in model.parameters() if p.requires_grad])

def gd_step_with_delta(
    model: nn.Module,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    lr: float,
) -> Tuple[float, torch.Tensor]:
    """
    One full-batch GD step and return (loss, delta_theta_flat).
    delta_theta = theta_{t+1} - theta_t = -lr * grad (for vanilla GD).
    """
    y_pred = model(x_batch)
    loss = criterion(y_pred, y_batch)

    model.zero_grad(set_to_none=True)
    loss.backward()

    # Capture dense gradient -> delta_theta
    grad_flat = _flatten_grads(model)
    delta_theta = -lr * grad_flat.clone()

    # Manual update
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p.add_( -lr * p.grad )

    return float(loss.item()), delta_theta

def train_k_steps(
    model: nn.Module,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x_train_t: torch.Tensor,
    y_train_t: torch.Tensor,
    lr: float,
    k: int,
    loss_history: List[float],
    deltas: List[torch.Tensor],
) -> float:
    """
    Perform k GD steps; append loss to history and store each delta_theta (flattened).
    Returns:
        last loss value
    """
    last_loss = 0.0
    for _ in range(k):
        last_loss, dtheta = gd_step_with_delta(model, criterion, x_train_t, y_train_t, lr)
        loss_history.append(last_loss)
        deltas.append(dtheta)
    return last_loss
