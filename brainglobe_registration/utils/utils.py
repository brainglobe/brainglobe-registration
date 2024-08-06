from pathlib import Path
from typing import Dict, List, Tuple

import napari
import numpy as np
import numpy.typing as npt
from brainglobe_atlasapi.list_atlases import get_downloaded_atlases
from brainglobe_utils.qtpy.dialog import display_info
from napari.utils.notifications import show_info
from pytransform3d.rotations import active_matrix_from_angle
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


def calculate_rotated_bounding_box(
    image_shape: Tuple[int, int, int], rotation_matrix: npt.NDArray
) -> Tuple[int, int, int]:
    """
    Calculates the bounding box of the rotated image.

    This function calculates the bounding box of the rotated image given the
    image shape and rotation matrix. The bounding box is calculated by
    transforming the corners of the image and finding the minimum and maximum
    values of the transformed corners.

    Parameters
    ------------
    image_shape : Tuple[int, int, int]
        The shape of the image.
    rotation_matrix : npt.NDArray
        The rotation matrix.

    Returns
    --------
    Tuple[int, int, int]
        The bounding box of the rotated image.
    """
    corners = np.array(
        [
            [0, 0, 0, 1],
            [image_shape[0], 0, 0, 1],
            [0, image_shape[1], 0, 1],
            [0, 0, image_shape[2], 1],
            [image_shape[0], image_shape[1], 0, 1],
            [image_shape[0], 0, image_shape[2], 1],
            [0, image_shape[1], image_shape[2], 1],
            [image_shape[0], image_shape[1], image_shape[2], 1],
        ]
    )

    transformed_corners = rotation_matrix @ corners.T
    min_corner = np.min(transformed_corners, axis=1)
    max_corner = np.max(transformed_corners, axis=1)

    return (
        int(np.round(max_corner[0] - min_corner[0])),
        int(np.round(max_corner[1] - min_corner[1])),
        int(np.round(max_corner[2] - min_corner[2])),
    )


def check_atlas_installed():
    available_atlases = get_downloaded_atlases()
    if len(available_atlases) == 0:
        show_info(
            "No atlases available. Please download atlas(es) with "
            "brainglobe-atlasapi (https://github.com/brainglobe/brain"
            "globe-atlasapi) or brainrender-napari (https://brainglobe.info"
            "/tutorials/manage-atlases-in-GUI.html)"
        )
        display_info(
            widget=QWidget(),
            title="Information",
            message="No atlases available. Please download atlas(es) "
            "using <a href='https://github.com/brainglobe/brainglobe-at"
            "lasapi'>brainglobe-atlasapi</a> or <a href='https://brain"
            "globe.info/tutorials/manage-atlases-in-GUI.html'>brainrend"
            "er-napari</a>",
        )
