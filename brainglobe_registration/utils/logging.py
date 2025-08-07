import logging
from collections import namedtuple

from bayes_opt.logger import ScreenLogger


def get_auto_slice_logging_args(params: dict) -> tuple:
    args_dict = {
        "z_range": str(params["z_range"]),
        "pitch_bounds": str(params["pitch_bounds"]),
        "yaw_bounds": str(params["yaw_bounds"]),
        "roll_bounds": str(params["roll_bounds"]),
        "init_points": str(params["init_points"]),
        "n_iter": str(params["n_iter"]),
        "metric": str(params["metric"]),
        "metric_weights": str(f"('mi', 'ncc', 'ssim') = {params['weights']}"),
    }

    AutoSliceArgs = namedtuple(
        "AutoSliceArgs",
        (
            "z_range",
            "pitch_bounds",
            "yaw_bounds",
            "roll_bounds",
            "init_points",
            "n_iter",
            "metric",
            "metric_weights",
        ),
    )

    return AutoSliceArgs(*args_dict.values())


class FancyBayesLogger(ScreenLogger):
    def __init__(self, verbose=2):
        super().__init__(verbose=verbose)
        self._logger = logging.getLogger("fancylog.bayesopt")

    def log_optimization_step(self, keys, result, params_config, best):
        """
        Logs a single Bayesian optimisation step using logging.info().

        Overrides the default `ScreenLogger` used by `BayesianOptimization`,
        which prints to stdout. Redirecting output through `fancylog` allows
        better integration with file-based and structured logging.
        """
        log_str = "|".join([f"{k}: {result['params'][k]:.4f}" for k in keys])
        target = result["target"]
        self._logger.info(
            f"[BayesOpt] Step | Target: {target:.4f} | {log_str}"
        )
