import logging
import sys
from collections import namedtuple
from contextlib import contextmanager
from io import StringIO


def get_auto_slice_logging_args(params: dict) -> tuple:
    args_dict = {
        "z_range": str(params["z_range"]),
        "pitch_bounds": str(params["pitch_bounds"]),
        "yaw_bounds": str(params["yaw_bounds"]),
        "roll_bounds": str(params["roll_bounds"]),
        "init_points": str(params["init_points"]),
        "n_iter": str(params["n_iter"]),
        "metric": str(params["metric"]),
    }

    AutoSliceArgsBase = namedtuple(
        "AutoSliceArgsBase",
        (
            "z_range",
            "pitch_bounds",
            "yaw_bounds",
            "roll_bounds",
            "init_points",
            "n_iter",
            "metric",
        ),
    )

    AutoSliceArgsWithWeights = namedtuple(
        "AutoSliceArgsWithWeights",
        (
            "z_range",
            "pitch_bounds",
            "yaw_bounds",
            "roll_bounds",
            "init_points",
            "n_iter",
            "metric",
            "combined_weights",
        ),
    )

    if params["metric"] == "combined":
        args_dict["combined_weights"] = str(params["weights"])
        return AutoSliceArgsWithWeights(*args_dict.values()), args_dict
    else:
        return AutoSliceArgsBase(*args_dict.values()), args_dict


@contextmanager
def redirect_stdout_to_fancylog(level=logging.INFO):
    """
    Redirects print() output within the context block to the fancylog logger.
    """
    logger = logging.getLogger()
    old_stdout = sys.stdout
    stream = StringIO()
    sys.stdout = stream

    try:
        yield
    finally:
        sys.stdout = old_stdout
        output = stream.getvalue()
        for line in output.strip().splitlines():
            logger.log(level, line)
