# src/fit_store.py
from __future__ import annotations
import json, os
import numpy as np
import torch
import torch.nn as nn
from typing import Sequence

_DTYPE = {"float16": np.float16, "float32": np.float32}
_F16_MIN, _F16_MAX = -65504.0, 65504.0  # IEEE float16 finite range

class FitRecorder:
    """
    Disk-backed memmap for function fits over time (1D).
    Shape: (T_snapshots, P_fit). Stores float16 by default.
    Uses an explicit epoch 'schedule' to avoid off-by-one errors.
    """
    def __init__(self, cfg, X_vis: np.ndarray, schedule: Sequence[int], device: torch.device):
        self.cfg = cfg
        self.device = device
        self.schedule = list(schedule)                 # sorted list of epochs to record
        self.T = len(self.schedule)

        dtype = _DTYPE.get(cfg.fit_dtype, np.float16)
        P_all = X_vis.shape[0]
        self.idx = np.linspace(0, P_all - 1, min(cfg.fit_points_1d, P_all), dtype=int)
        self.X_fit = X_vis[self.idx]                  # (P_fit,1)
        self.X_fit_t = torch.from_numpy(self.X_fit).to(device)

        shape = (self.T, self.X_fit.shape[0])
        self.mm = np.memmap(cfg.fits_path, mode="w+", dtype=dtype, shape=shape)
        self.t = 0  # row pointer

        if cfg.fits_x_path:
            np.save(cfg.fits_x_path, self.X_fit.astype(np.float32))

        meta = {
            "dtype": cfg.fit_dtype,
            "shape": list(shape),
            "log_every_epochs": int(cfg.log_every_epochs),
            "schedule": self.schedule,  # <-- map memmap row -> epoch
            "grid_file": os.path.basename(cfg.fits_x_path) if cfg.fits_x_path else None,
            "train_points_file": os.path.basename(cfg.train_points_path) if cfg.train_points_path else None,
        }
        with open(cfg.fits_meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    @torch.no_grad()
    def record(self, model: nn.Module, epoch: int):
        # Only record if epoch is scheduled (robust if caller misfires).
        if self.t >= self.T:
            return  # silently ignore extra calls
        if epoch != self.schedule[self.t]:
            return  # wait until the expected scheduled epoch

        model.eval()
        y = model(self.X_fit_t).squeeze(1).detach().cpu().numpy()

        # Clip to target dtype range (prevents float16 overflow warnings)
        if self.mm.dtype == np.float16:
            y = np.clip(y, _F16_MIN, _F16_MAX)
        self.mm[self.t, :] = y.astype(self.mm.dtype, copy=False)
        self.mm.flush()
        self.t += 1

    def close(self):
        del self.mm


class GradRecorder:
    """
    Disk-backed memmap for df/dx over time (1D), aligned to the same schedule as FitRecorder.
    Shape: (T_snapshots, P_fit). Stores float16 by default.
    """
    def __init__(self, cfg, X_vis: np.ndarray, schedule: Sequence[int], device: torch.device):
        self.cfg = cfg
        self.device = device
        self.schedule = list(schedule)
        self.T = len(self.schedule)

        dtype = _DTYPE.get(getattr(cfg, "grad_dtype", "float16"), np.float16)
        P_all = X_vis.shape[0]
        # use the SAME subsample indices as FitRecorder if possible
        self.idx = np.linspace(0, P_all - 1, min(getattr(cfg, "grad_points_1d", 512), P_all), dtype=int)
        self.X_fit = X_vis[self.idx]                  # (P_fit,1)
        self.X_fit_t = torch.from_numpy(self.X_fit).to(device)

        shape = (self.T, self.X_fit.shape[0])
        self.mm = np.memmap(cfg.grads_path, mode="w+", dtype=dtype, shape=shape)
        self.t = 0

        meta = {
            "dtype": getattr(cfg, "grad_dtype", "float16"),
            "shape": list(shape),
            "schedule": self.schedule,
            "grid_file": os.path.basename(cfg.fits_x_path) if cfg.fits_x_path else None,
        }
        with open(cfg.grads_meta_path, "w") as f:
            json.dump(meta, f, indent=2)


    def record(self, model: nn.Module, epoch: int):
        if self.t >= self.T or epoch != self.schedule[self.t]:
            return
        model.eval()
        # enable grad on input, compute df/dx via autograd
        X = self.X_fit_t.clone().detach().requires_grad_(True)  # (P,1)
        y = model(X).squeeze(1)                                 # (P,)
        (gy,) = torch.autograd.grad(y.sum(), X, retain_graph=False, create_graph=False)
        dfdx = gy.squeeze(1).detach().cpu().numpy()
        if self.mm.dtype == np.float16:
            dfdx = np.clip(dfdx, _F16_MIN, _F16_MAX)
        self.mm[self.t, :] = dfdx.astype(self.mm.dtype, copy=False)
        self.mm.flush()
        self.t += 1

    def close(self):
        del self.mm

class SecondGradRecorder:
    """
    Disk-backed memmap for d2f/dx2 over time (1D), aligned to the same schedule.
    Shape: (T_snapshots, P_fit). Stores float16 by default.
    """
    def __init__(self, cfg, X_vis: np.ndarray, schedule: Sequence[int], device: torch.device):
        self.cfg = cfg
        self.device = device
        self.schedule = list(schedule)
        self.T = len(self.schedule)

        dtype = _DTYPE.get(getattr(cfg, "hess_dtype", "float16"), np.float16)
        P_all = X_vis.shape[0]
        self.idx = np.linspace(0, P_all - 1, min(getattr(cfg, "hess_points_1d", 512), P_all), dtype=int)
        self.X_fit = X_vis[self.idx]                  # (P_fit,1)
        self.X_fit_t = torch.from_numpy(self.X_fit).to(device)

        shape = (self.T, self.X_fit.shape[0])
        self.mm = np.memmap(cfg.grads2_path, mode="w+", dtype=dtype, shape=shape)
        self.t = 0

        meta = {
            "dtype": getattr(cfg, "hess_dtype", "float16"),
            "shape": list(shape),
            "schedule": self.schedule,
            "grid_file": os.path.basename(cfg.fits_x_path) if cfg.fits_x_path else None,
        }
        with open(cfg.grads2_meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    def record(self, model: nn.Module, epoch: int):
        if self.t >= self.T or epoch != self.schedule[self.t]:
            return
        model.eval()
        # d2f/dx2: first take df/dx with create_graph=True, then grad again.
        with torch.enable_grad():
            X = self.X_fit_t.clone().detach().requires_grad_(True)  # (P,1)
            y = model(X).squeeze(1)                                 # (P,)
            (gy,)  = torch.autograd.grad(y.sum(),  X, retain_graph=True,  create_graph=True)   # (P,1)
            (g2y,) = torch.autograd.grad(gy.sum(), X, retain_graph=False, create_graph=False)  # (P,1)
            d2fdx2 = g2y.squeeze(1).detach().cpu().numpy()
        if self.mm.dtype == np.float16:
            d2fdx2 = np.clip(d2fdx2, _F16_MIN, _F16_MAX)
        self.mm[self.t, :] = d2fdx2.astype(self.mm.dtype, copy=False)
        self.mm.flush()
        self.t += 1

    def close(self):
        del self.mm