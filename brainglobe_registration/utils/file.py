from pathlib import Path, PurePath
from typing import Dict

import napari
import numpy as np
from brainglobe_atlasapi import BrainGlobeAtlas


def open_parameter_file(file_path: Path) -> Dict:
    """
    Opens the parameter file and returns the parameter dictionary.

    This function reads a parameter file and extracts the parameters into
    a dictionary. The parameter file is expected to have lines in the format
    "(key value1 value2 ...)". Any line not starting with "(" is ignored.
    The values are stripped of any trailing ")" and leading or trailing quotes.

    Parameters
    ------------
    file_path : Path
        The path to the parameter file.

    Returns
    --------
    Dict
        A dictionary containing the parameters from the file.
    """
    with open(file_path, "r") as f:
        param_dict = {}
        for line in f.readlines():
            if line[0] == "(":
                split_line = line[1:-1].split()
                cleaned_params = []
                for i, entry in enumerate(split_line[1:]):
                    if entry == ")" or entry[0] == "/":
                        break

                    cleaned_params.append(entry.strip('" )'))

                param_dict[split_line[0]] = cleaned_params

    return param_dict


def serialize_registration_widget(obj):
    if isinstance(obj, napari.layers.Layer):
        return obj.name
    elif isinstance(obj, napari.Viewer):
        return str(obj)
    elif isinstance(obj, PurePath):
        return str(obj)
    elif isinstance(obj, BrainGlobeAtlas):
        return obj.atlas_name
    elif isinstance(obj, np.ndarray):
        return f"<{type(obj)}>, {obj.shape}, {obj.dtype}"
    else:
        return obj.__dict__()
