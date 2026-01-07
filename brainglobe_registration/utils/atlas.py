from collections import Counter
from pathlib import Path
from typing import Dict, Tuple

import dask.array as da
import numpy as np
import numpy.typing as npt
import pandas as pd
from brainglobe_atlasapi import BrainGlobeAtlas


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
    Adjust the atlas labels such that they can be represented by an unsigned
     short (np.uint16).

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


def generate_mask_from_atlas_annotations(atlas: BrainGlobeAtlas) -> np.ndarray:
    """
    Generate a binary mask from the atlas annotation array.

    Parameters
    ----------
    atlas : BrainGlobeAtlas
        Atlas object containing annotation data.

    Returns
    -------
    np.ndarray
        Binary mask of the same shape as the annotation.
        Pixels are 1 if the annotation is not zero, else 0.
    """
    annotation = atlas.annotation
    mask = annotation != 0
    return mask.astype(np.uint8)


def mask_atlas(atlas: BrainGlobeAtlas, mask: np.ndarray) -> np.ndarray:
    """
    Apply a mask to the reference image of the atlas.

    Parameters
    ----------
    atlas : BrainGlobeAtlas
        Atlas object containing reference data.
    mask : np.ndarray
        Binary mask to apply.

    Returns
    -------
    np.ndarray
        Reference image with the mask applied.
        Pixels outside the mask are set to zero.
    """
    masked_reference = atlas.reference * mask
    return masked_reference


def mask_atlas_with_annotations(atlas: BrainGlobeAtlas) -> np.ndarray:
    """
    Apply the annotation-based mask to the reference image of the atlas.

    Parameters
    ----------
    atlas : BrainGlobeAtlas
        Atlas object containing reference and annotation data.

    Returns
    -------
    np.ndarray
        Reference image with the annotation-based mask applied.
    """
    mask = generate_mask_from_atlas_annotations(atlas)
    return mask_atlas(atlas, mask)
