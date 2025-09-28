# sinfit/config.py
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

@dataclass
class Config:
    # Data (training interval)
    n_samples: int = 500
    x_min: float = -2 * math.pi
    x_max: float =  2 * math.pi
    noise_std: float = 0.08
    seed: int = 0

    # ---- Target function selection ----
    # Choose by name from sinfit.functions.FUNCTIONS (e.g., "sin", "poly")
    function_name: str = "sin"
    # Optional kwargs for that function, saved to config.json
    function_params: Dict[str, Any] = None  # e.g., {"a":0.1,"b":0.0,"c":1.0,"d":0.0}

    # Model / training
    input_dim: int = 1  # set to 2 for 2-D input
    hidden: int = 64
    lr: float = 1e-2
    total_steps: int = 5000
    steps_per_frame: int = 10
    criterion_name: str = "mse"

    # Curvature tracking
    sharpness_stride: int = 20
    power_iters: int = 20

    # Visualization grid
    n_plot_points: int = 1200
    x_vis_min: Optional[float] = None
    x_vis_max: Optional[float] = None
    y_vis_min: Optional[float] = None
    y_vis_max: Optional[float] = None
    vis_pad_frac: float = 0.5

    # For 2-D, set ranges for each axis (ignored if input_dim=1)
    x1_min: float = -2 * math.pi
    x1_max: float =  2 * math.pi
    x2_min: float = -2 * math.pi
    x2_max: float =  2 * math.pi

    # Figure / animation
    dpi: int = 140
    figsize: Tuple[float, float] = (11.5, 7.2)
    fps_mp4: int = 20
    bitrate_mp4: int = 1800
    fps_gif: int = 15

    # ---- Run management ----
    runs_root: str = "runs"
    project_name: Optional[str] = None
    run_dir: Optional[str] = None
    out_mp4: Optional[str] = None
    out_gif: Optional[str] = None

    # Optional ffmpeg path
    ffmpeg_path: Optional[str] = None

    # Visualization padding (used for both dims if input_dim=2)
    vis_pad_frac: float = 0.5
