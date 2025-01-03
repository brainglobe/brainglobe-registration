from importlib.resources import files
from typing import List

from napari.types import LayerData
from tifffile import imread


def load_sample_data() -> List[LayerData]:
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
