from pathlib import Path
from typing import Dict, List, Tuple

import napari
import numpy as np
import numpy.typing as npt
from brainglobe_atlasapi.list_atlases import get_downloaded_atlases
from brainglobe_utils.qtpy.dialog import display_info
from pytransform3d.rotations import active_matrix_from_angle
from qtpy.QtWidgets import QWidget
from scipy.ndimage import gaussian_filter
from skimage import morphology


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


def filter_plane(img_plane):
    """
    Apply a set of filter to the plane (typically to avoid overfitting details
    in the image during registration)
    The filter is composed of a despeckle filter using opening and a pseudo
    flatfield filter

    Originally from: https://github.com/brainglobe/brainreg/blob/main
    /brainreg/core/utils/preprocess.py

    Parameters
    ----------
    img_plane : np.array
        A 2D array to filter

    Returns
    ----------
    np.array
        Filtered image
    """

    img_plane = despeckle_by_opening(img_plane)
    img_plane = pseudo_flatfield(img_plane)
    return img_plane


def despeckle_by_opening(img_plane, radius=2):
    """
    Despeckle the image plane using a grayscale opening operation

    Originally from: https://github.com/brainglobe/brainreg/blob/main
    /brainreg/core/utils/preprocess.py

    Parameters
    ----------
    img_plane : np.array

    radius: int
        The radius of the opening kernel

    Returns
    ----------
    np.array
        The despeckled image
    """
    kernel = morphology.disk(radius)
    morphology.opening(img_plane, out=img_plane, footprint=kernel)
    return img_plane


def pseudo_flatfield(img_plane, sigma=5):
    """
    Pseudo flat field filter implementation using a de-trending by a
    heavily gaussian filtered copy of the image.

    Originally from: https://github.com/brainglobe/brainreg/blob/main
    /brainreg/core/utils/preprocess.py

    Parameters
    ----------
    img_plane : np.array
        The image to filter

    sigma : int
        The sigma of the gaussian filter applied to the
        image used for de-trending

    Returns
    ----------
    np.array
        The pseudo flat field filtered image
    """
    filtered_img = gaussian_filter(img_plane, sigma)
    return img_plane / (filtered_img + 1)
