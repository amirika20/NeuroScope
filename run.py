from src.config import Config
from src.main import main

if __name__ == "__main__":
    cfg = Config(
        project_name="NeuroScope1D",
        function_name="sin",
        # function_params={'a': 2.0, 'b': 1.0, 'c': -1.0, 'd': 5.0},
        x_min=3*-3.14, x_max=3*3.14,
        n_samples=250,
        noise_std = 0.05,
        n_plot_points=3000,
        hidden=64, depth=2, activation="tanh",
        epochs=1000, log_every_epochs=10,
        loss_name="mse",
        ntk_probe_points=64,
        ntk_topk=2,
        power_iters=20,
        fit_points_1d=512,
        # Optimizer
        optimizer_name="sgd",
        lr=0.005,
        batch_size=64,
        momentum=0.5,
        # nesterov=True,
        weight_decay=0.005,
        # grad_clip=None,
    )
    main(cfg)
