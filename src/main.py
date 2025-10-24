from __future__ import annotations
import torch
from copy import deepcopy
from typing import Optional, Callable
from .config import Config
from .runio import prepare_run_dir, save_config_json
from .data import set_seeds, resolve_target_fn, generate_data, build_vis_grid
from .model import build_model
from .losses import get_loss_fn
from .training import run_training_and_log_csv
from .anim import make_anim

CriterionFn = Callable[[torch.nn.Module, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]

def main(cfg: Config, target_fn_override=None, loss_fn: Optional[CriterionFn] = None):
    set_seeds(cfg.seed)
    run_dir = prepare_run_dir(cfg)
    print(f"[INFO] run dir: {run_dir}")
    cfg_json = save_config_json(cfg)
    print(f"[INFO] saved config -> {cfg_json}")

    # target function & data
    target_fn, used = resolve_target_fn(cfg, override_fn=target_fn_override)
    print(f"[INFO] target: {cfg.function_name} params={used}")
    X_train_t, y_train_t, y_train_clean_t = generate_data(cfg, target_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_t, y_train_t, y_train_clean_t = X_train_t.to(device), y_train_t.to(device), y_train_clean_t.to(device)

    # build vis grid & truth
    X_vis, y_true_vis = build_vis_grid(cfg, target_fn)

    # model
    model = build_model(cfg).to(device)

    # loss
    criterion = loss_fn if loss_fn is not None else get_loss_fn(cfg.loss_name, custom=cfg.custom_loss)

    # train & log
    df = run_training_and_log_csv(cfg, model, X_train_t, y_train_t, y_train_clean_t, X_vis, y_true_vis, criterion)

    mp4_path = f"{run_dir}/progress.mp4"
    gif_path = f"{run_dir}/progress.gif"  # fallback target
    def true_fn(x_np):
        return target_fn(x_np)

    make_anim(
        run_dir,
        true_fn=true_fn,           # or None if you don't want the dashed true curve
        out_mp4=mp4_path,
        out_gif=gif_path,
        fps=30,
        bitrate=2400,
        eig_log10=True,
        cfg=cfg,  
    )
    print(f"[DONE] animation saved at {mp4_path} (or {gif_path})")
