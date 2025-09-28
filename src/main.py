# sinfit/main.py
from __future__ import annotations
import numpy as np
import torch

from .config import Config
from .data import set_seeds, generate_data, resolve_target_fn, TargetFn
from .model import build_model
from .viz import make_anim, save_with_progress
from .runio import prepare_run_dir
from .losses import get_loss_fn

def _make_vis_grid(cfg: Config, target_fn):
    if cfg.input_dim == 1:
        span = cfg.x_max - cfg.x_min
        pad = cfg.vis_pad_frac * span
        x0 = cfg.x_min - pad if cfg.x_vis_min is None else cfg.x_vis_min
        x1 = cfg.x_max + pad if cfg.x_vis_max is None else cfg.x_vis_max
        x_vis = np.linspace(x0, x1, cfg.n_plot_points, dtype=np.float32)
        X_vis = x_vis[:, None]                 # (P,1)
    else:  # 2-D
        span1 = cfg.x1_max - cfg.x1_min
        span2 = cfg.x2_max - cfg.x2_min
        pad1 = cfg.vis_pad_frac * span1
        pad2 = cfg.vis_pad_frac * span2
        a = cfg.x1_min - pad1; b = cfg.x1_max + pad1
        c = cfg.x2_min - pad2; d = cfg.x2_max + pad2
        # use sqrt points per axis so total ≈ n_plot_points
        m = int(np.sqrt(cfg.n_plot_points))
        x1g = np.linspace(a, b, m, dtype=np.float32)
        x2g = np.linspace(c, d, m, dtype=np.float32)
        X1, X2 = np.meshgrid(x1g, x2g, indexing="xy")
        X_vis = np.stack([X1.ravel(), X2.ravel()], axis=1)  # (m*m,2)

    y_true_vis = target_fn(X_vis).astype(np.float32)        # (P,)
    return X_vis, y_true_vis


def main(cfg: Config | None = None, target_fn_override: TargetFn | None = None) -> None:
    """
    If you pass a custom callable in target_fn_override(x: np.ndarray)->np.ndarray,
    it will override cfg.function_name/params.
    """
    cfg = cfg or Config()
    set_seeds(cfg.seed)

    # Prepare run dir & save config
    run_dir = prepare_run_dir(cfg)
    print(f"[INFO] Run directory: {run_dir}")

    # Resolve target function
    target_fn, used_params = resolve_target_fn(cfg, override_fn=target_fn_override)
    print(f"[INFO] Target function: {cfg.function_name if target_fn_override is None else 'override callable'} "
          f"params={used_params}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Training data on device
    x_train_t, y_train_t = generate_data(cfg, target_fn)
    x_train_t, y_train_t = x_train_t.to(device), y_train_t.to(device)

    # Visualization grid (wider domain) using the same function
    x_vis, y_true_vis = _make_vis_grid(cfg, target_fn)

    # Model
    model = build_model(cfg).to(device)
    criterion = get_loss_fn(cfg.criterion_name)

    # Animation
    anim, _ = make_anim(cfg, model, x_train_t, y_train_t, x_vis, y_true_vis, criterion)

    # Save
    save_with_progress(anim, cfg)
