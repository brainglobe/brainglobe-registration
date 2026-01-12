from typing import List, Optional, Tuple

import dask.array as da
import napari
import numpy as np
import numpy.typing as npt
from brainglobe_atlasapi.list_atlases import get_downloaded_atlases
from pytransform3d.rotations import active_matrix_from_angle
from qt_niu.dialog import display_info
from qtpy.QtWidgets import QWidget


def adjust_napari_image_layer(
    image_layer: napari.layers.Image, x: int, y: int, rotate: float
):
    """
    Adjusts the napari image layer by the given x, y, and rotation values.

    This function takes in a napari image layer and modifies its translate
    and affine properties based on the provided x, y, and rotation values.
    The rotation is performed around the center of the image layer.

    Rotation around origin code adapted from:
    https://forum.image.sc/t/napari-3d-rotation-center-change-and-scaling/66347/5

    Parameters
    ------------
    image_layer : napari.layers.Layer
        The napari image layer to be adjusted.
    x : int
        The x-coordinate for the translation.
    y : int
        The y-coordinate for the translation.
    rotate : float
        The angle of rotation in degrees.

    Returns
    --------
    None
    """
    image_layer.translate = (y, x)

    rotation_matrix = active_matrix_from_angle(2, np.deg2rad(rotate))
    translate_matrix = np.eye(3)
    origin = np.asarray(image_layer.data.shape) // 2 + np.asarray([y, x])
    translate_matrix[:2, -1] = origin
    transform_matrix = (
        translate_matrix @ rotation_matrix @ np.linalg.inv(translate_matrix)
    )
    image_layer.affine = transform_matrix


def get_data_from_napari_layer(
    layer: napari.layers.Layer, selection: Optional[Tuple[slice, ...]] = None
) -> npt.NDArray:
    """
    Returns the data from the napari layer.

    This function returns the data from the napari layer. If the layer is a
    dask array, the data is computed before returning.

    Parameters
    ------------
    layer : napari.layers.Layer
        The napari layer from which to extract the data.
    selection : Tuple[slice, ...], optional
        The selection to apply to the data prior to computing.

    Returns
    --------
    npt.NDArray
        The selected data from the napari layer.
    """
    if selection is not None:
        data = layer.data[selection]
    else:
        data = layer.data

    if isinstance(layer.data, da.Array):
        return data.compute().squeeze()

    return data.squeeze()


def find_layer_index(viewer: napari.Viewer, layer_name: str) -> int:
    """Finds the index of a layer in the napari viewer."""
    for idx, layer in enumerate(viewer.layers):
        if layer.name == layer_name:
            return idx

    return -1


def get_image_layer_names(viewer: napari.Viewer) -> List[str]:
    """
    Returns a list of the names of the napari image layers in the viewer.

    Parameters
    ------------
    viewer : napari.Viewer
        The napari viewer containing the image layers.

    Returns
    --------
    List[str]
        A list of the names of the image layers in the viewer.
    """
    return [layer.name for layer in viewer.layers]


def check_atlas_installed(parent_widget: QWidget):
    """
    Function checks if user has any atlases installed. If not, message box
    appears in napari, directing user to download atlases via attached links.
    """
    available_atlases = get_downloaded_atlases()
    if len(available_atlases) == 0:
        display_info(
            widget=parent_widget,
            title="Information",
            message="No atlases available. Please download atlas(es) "
            "using <a href='https://brainglobe.info/documentation/"
            "brainglobe-atlasapi/usage/command-line-interface.html'>"
            "brainglobe-atlasapi</a> or <a href='https://brainglobe.info/"
            "tutorials/manage-atlases-in-GUI.html'>brainrender-napari</a>",
        )
