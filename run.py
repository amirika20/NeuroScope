from src.main import main
from src.config import Config
import numpy as np

cfg = Config(
    project_name="2d-sincos",
    input_dim=2,
    function_name="sin_cos2d",
    n_samples=2000,
    x1_min=-2*np.pi, x1_max=2*np.pi,
    x2_min=-2*np.pi, x2_max=2*np.pi,
    n_plot_points=3600,      # 60x60 grid
    total_steps=5000, steps_per_frame=10
)
main(cfg)