from importlib.resources import files
from typing import List

from napari.types import LayerData
from tifffile import imread


def load_sample_data_2d() -> List[LayerData]:
    """
    Load the sample data.

    Returns
    -------
    List[LayerData]
        The sample data.
    """
    path = str(
        files("brainglobe_registration").joinpath("resources/sample_hipp.tif")
    )

    return [(imread(path), {"name": "2D coronal mouse brain section"})]


def load_sample_data_3d() -> List[LayerData]:
    """
    Load the sample data.

    Returns
    -------
    List[LayerData]
        The sample data.
    """
    path = str(
        files("brainglobe_registration").joinpath("resources/sample_3d.tif")
    )

    return [(imread(path), {"name": "3D mouse brain"})]
