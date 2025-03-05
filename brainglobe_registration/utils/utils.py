from collections import Counter
from pathlib import Path, PurePath
from typing import Dict, List, Tuple

import dask.array as da
import napari
import numpy as np
import numpy.typing as npt
import pandas as pd
from brainglobe_atlasapi import BrainGlobeAtlas
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

    transformed_corners = np.dot(rotation_matrix, corners.T)
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


def calculate_region_size(
    atlas: BrainGlobeAtlas,
    annotation_image: npt.NDArray[np.uint32],
    hemispheres: npt.NDArray,
    output_path: Path,
    left_hemisphere_label: int = 2,
    right_hemisphere_label: int = 1,
) -> pd.DataFrame:
    """
    Calculate the areas of the structures in the annotation image.

    Parameters
    ----------
    atlas: BrainGlobeAtlas
        The atlas object to which the annotation image belongs.
    annotation_image: npt.NDArray[np.uint32]
        The annotation image.
    hemispheres: npt.NDArray
        The hemisphere labels for each pixel in the annotation image.
    output_path: Path
        The path to save the output csv file.
    left_hemisphere_label: int, optional
        The label for the left hemisphere.
    right_hemisphere_label: int, optional
        The label for the right hemisphere.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the structure names and areas.
    """
    count_left = Counter(
        annotation_image[hemispheres == left_hemisphere_label].flatten()
    )
    count_right = Counter(
        annotation_image[hemispheres == right_hemisphere_label].flatten()
    )

    # Remove the background label
    try:
        count_left.pop(0)
        count_right.pop(0)
    except KeyError:
        pass

    structures_reference_df = atlas.lookup_df

    if annotation_image.ndim == 3:
        columns = [
            "structure_name",
            "left_volume_mm3",
            "right_volume_mm3",
            "total_volume_mm3",
        ]
        pixel_conversion_factor = (
            atlas.resolution[0]
            * atlas.resolution[1]
            * atlas.resolution[2]
            / (1000**3)
        )
    else:
        columns = [
            "structure_name",
            "left_area_mm2",
            "right_area_mm2",
            "total_area_mm2",
        ]
        pixel_conversion_factor = (
            atlas.resolution[1] * atlas.resolution[2] / (1000**2)
        )

    df = pd.DataFrame(
        index=structures_reference_df.id,
        columns=columns,
    )

    for structure_id in count_left.keys():
        structure_line = structures_reference_df[
            structures_reference_df["id"] == structure_id
        ]

        if len(structure_line) == 0:
            print(
                f"Value: {structure_id} is not in the atlas structure "
                f"reference file. Not calculating the area."
            )
            continue

        left_size = count_left[structure_id] * pixel_conversion_factor
        right_size = count_right[structure_id] * pixel_conversion_factor
        total_size = left_size + right_size

        df.loc[structure_id] = {
            columns[0]: structure_line["name"].values[0],
            columns[1]: left_size,
            columns[2]: right_size,
            columns[3]: total_size,
        }

    df.dropna(how="all", inplace=True)

    df.to_csv(output_path, index=False)

    return df


def convert_atlas_labels(
    annotation_image: npt.NDArray[np.uint32],
) -> Tuple[npt.NDArray[np.uint16], Dict[int, int]]:
    """
    Adjust the atlas labels such that they can be represented accurately
    by a single precision float (np.float32) or an unsigned short (np.uint16).

    This is done by mapping the labels greater than 2**15 to new
    consecutive values starting from 2**15. Assumes no more than 2**15
    unique values greater than 2**15.

    Slow to run if a large number of unique values are greater than 2**15.
    Based on current BrainGlobe atlases, this should not be the case.

    Parameters
    ----------
    annotation_image: npt.NDArray[np.uint16]
        The annotation image.

    Returns
    -------
    npt.NDArray[np.uint16]
        The adjusted annotation image.
    Dict[int, int]
        A dictionary mapping the original values to the new values.
            key: original annotation ID
            value: new annotation ID
    """
    # Copy array to avoid modifying the original
    output_array = np.array(annotation_image, copy=True)
    # Returns a sorted array of unique values in the annotation image
    values = np.unique(output_array)

    if isinstance(values, da.Array):
        values = values.compute()

    # Create a mapping of the original values to the new values
    # and adjust the annotation image
    mapping = {}
    new_value = 2**15
    for value in values:
        if value > new_value:
            mapping[value] = new_value
            output_array[output_array == value] = new_value
            new_value += 1
        elif value == new_value:
            new_value += 1

    return output_array.astype(np.uint16), mapping


def restore_atlas_labels(
    annotation_image: npt.NDArray[np.uint16], mapping: Dict[int, int]
) -> npt.NDArray[np.uint32]:
    """
    Restore the original atlas labels from the adjusted labels based on the
    provided mapping.

    Parameters
    ----------
    annotation_image: npt.NDArray[np.uint16]
        The adjusted annotation image.
    mapping: Dict[int, int]
        A dictionary mapping the original values to the new values.
            key: original annotation ID
            value: new annotation ID

    Returns
    -------
    npt.NDArray[np.uint32]
        The restored annotation image.
    """
    annotation_image = annotation_image.astype(np.uint32, copy=False)
    for old_value, new_value in mapping.items():
        annotation_image[annotation_image == new_value] = old_value

    return annotation_image


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
