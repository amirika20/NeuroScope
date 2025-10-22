from src.config import Config
from src.main import main

if __name__ == "__main__":
    cfg = Config(
        project_name="NeuroScope1D",
        function_name="sin",
        # function_params={'a': 2.0, 'b': 1.0, 'c': -1.0, 'd': 5.0},
        x_min=-3.14, x_max=3.14,
        n_samples=500,
        n_plot_points=3600,
        hidden=64, depth=2, activation="relu",
        lr=0.05,
        epochs=3000, log_every_epochs=10,
        loss_name="mse",
        ntk_probe_points=64,
        ntk_topk=2,
        power_iters=20,
        fit_points_1d=512,
    )
    main(cfg)
