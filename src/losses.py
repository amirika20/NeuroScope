# src/losses.py
import torch.nn as nn

def get_loss_fn(name: str) -> nn.Module:
    name = name.lower()
    if name in ("mse", "mse_loss"):
        return nn.MSELoss()
    elif name in ("l1", "mae", "l1_loss"):
        return nn.L1Loss()
    elif name in ("ce", "cross_entropy"):
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss function: {name}")
