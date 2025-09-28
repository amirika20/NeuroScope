# sinfit/data.py
from __future__ import annotations
from typing import Tuple, Callable, Optional, Dict, Any
import numpy as np
import torch
from .config import Config
from .functions import FUNCTIONS

TargetFn = Callable[[np.ndarray], np.ndarray]

def set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

def resolve_target_fn(cfg: Config, override_fn: Optional[TargetFn] = None) -> Tuple[TargetFn, Dict[str, Any]]:
    """
    Resolve the target function to use.
    Priority:
      1) override_fn if provided (callable)
      2) FUNCTIONS[cfg.function_name] with cfg.function_params
    Returns:
      (callable f(x), params_dict_used)
    """
    if override_fn is not None:
        return override_fn, (cfg.function_params or {})
    if cfg.function_name not in FUNCTIONS:
        raise ValueError(f"Unknown function_name '{cfg.function_name}'. Available: {list(FUNCTIONS.keys())}")
    return (lambda x: FUNCTIONS[cfg.function_name](x, **(cfg.function_params or {}))), (cfg.function_params or {})

def generate_data(cfg: Config, target_fn: TargetFn) -> Tuple[torch.Tensor, torch.Tensor]:
    if cfg.input_dim == 1:
        x = np.random.uniform(cfg.x_min, cfg.x_max, size=cfg.n_samples).astype(np.float32)
        X = x[:, None]  # (N,1)
    elif cfg.input_dim == 2:
        x1 = np.random.uniform(cfg.x1_min, cfg.x1_max, size=cfg.n_samples).astype(np.float32)
        x2 = np.random.uniform(cfg.x2_min, cfg.x2_max, size=cfg.n_samples).astype(np.float32)
        X = np.stack([x1, x2], axis=1)  # (N,2)
    else:
        raise ValueError("input_dim must be 1 or 2")

    y_clean = target_fn(X).astype(np.float32)
    noise = (cfg.noise_std * np.random.randn(cfg.n_samples)).astype(np.float32)
    y = y_clean + noise

    return torch.from_numpy(X), torch.from_numpy(y[:, None])
