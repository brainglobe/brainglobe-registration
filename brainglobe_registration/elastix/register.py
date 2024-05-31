from pathlib import Path
from typing import List, Tuple

import itk
import numpy as np
import numpy.typing as npt
from brainglobe_atlasapi import BrainGlobeAtlas


def get_atlas_by_name(atlas_name: str) -> BrainGlobeAtlas:
    """
    Get a BrainGlobeAtlas object by its name.

    Parameters
    ----------
    atlas_name : str
        The name of the atlas.

    Returns
    -------
    BrainGlobeAtlas
        The BrainGlobeAtlas object.
    """
    atlas = BrainGlobeAtlas(atlas_name)

    return atlas


def run_registration(
    atlas_image: npt.NDArray,
    moving_image: npt.NDArray,
    annotation_image: npt.NDArray,
    atlas_voxel_size: Tuple[float, ...],
    moving_voxel_size: Tuple[float, ...],
    parameter_lists: List[tuple[str, dict]],
    output_directory: Path,
) -> Tuple[np.ndarray, itk.ParameterObject, np.ndarray]:
    """
    Run the registration process on the given images.

    Parameters
    ----------
    atlas_image : npt.NDArray
        The atlas image.
    moving_image : npt.NDArray
        The moving image.
    atlas_voxel_size : Tuple[float, ...]
        The voxel size of the atlas image in um.
    moving_voxel_size : Tuple[float, ...]
        The voxel size of the moving image in um.
    annotation_image : npt.NDArray
        The annotation image.
    parameter_lists : List[tuple[str, dict]], optional
        The list of parameter lists, by default None
    output_directory: Path
        The output directory for the registration process.

    Returns
    -------
    np.ndarray
        The result image.
    itk.ParameterObject
        The result transform parameters.
    """
    # convert to ITK, view only
    atlas_image = itk.GetImageViewFromArray(atlas_image).astype(itk.F)
    moving_image = itk.GetImageViewFromArray(moving_image).astype(itk.F)
    annotation_image = itk.GetImageViewFromArray(annotation_image).astype(
        itk.F
    )

    atlas_image.SetSpacing(atlas_voxel_size)
    annotation_image.SetSpacing(atlas_voxel_size)
    moving_image.SetSpacing(moving_voxel_size)

    # This syntax needed for 3D images
    elastix_object = itk.ElastixRegistrationMethod.New(
        moving_image, atlas_image
    )

    parameter_object = setup_parameter_object(parameter_lists=parameter_lists)

    elastix_object.SetParameterObject(parameter_object)
    elastix_object.SetOutputDirectory(str(output_directory))

    # update filter object
    elastix_object.UpdateLargestPossibleRegion()

    # get results
    result_image = elastix_object.GetOutput()
    result_transform_parameters = elastix_object.GetTransformParameterObject()
    temp_interp_order = result_transform_parameters.GetParameter(
        0, "FinalBSplineInterpolationOrder"
    )
    result_transform_parameters.SetParameter(
        "FinalBSplineInterpolationOrder", "0"
    )

    annotation_image_transformix = itk.transformix_filter(
        annotation_image.astype(np.float32, copy=False),
        result_transform_parameters,
    )

    # Load Transformix Object
    transformix_object = itk.TransformixFilter.New(annotation_image)
    transformix_object.SetTransformParameterObject(result_transform_parameters)

    # Update object (required)
    transformix_object.UpdateLargestPossibleRegion()

    # Results of Transformation
    annotation_image_transformix = transformix_object.GetOutput()

    result_transform_parameters.SetParameter(
        "FinalBSplineInterpolationOrder", temp_interp_order
    )

    return (
        np.asarray(result_image),
        result_transform_parameters,
        np.asarray(annotation_image_transformix),
    )


def setup_parameter_object(parameter_lists: List[tuple[str, dict]]):
    """
    Set up the parameter object for the registration process.

    Parameters
    ----------
    parameter_lists : List[tuple[str, dict]], optional
        The list of parameter lists, by default None

    Returns
    -------
    itk.ParameterObject
        The parameter object.#
    """
    parameter_object = itk.ParameterObject.New()

    for transform_type, parameter_dict in parameter_lists:
        parameter_map = parameter_object.GetDefaultParameterMap(transform_type)
        parameter_map.clear()

        for k, v in parameter_dict.items():
            parameter_map[k] = v

        parameter_object.AddParameterMap(parameter_map)

    return parameter_object
