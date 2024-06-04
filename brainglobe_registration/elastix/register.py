from typing import Dict, List, Tuple

import itk
import numpy as np
from bg_atlasapi import BrainGlobeAtlas


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
    atlas_image,
    moving_image,
    annotation_image,
    parameter_lists: List[Tuple[str, Dict]],
) -> Tuple[
    np.ndarray, itk.ParameterObject, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Run the registration process on the given images.

    Parameters
    ----------
    atlas_image : np.ndarray
        The atlas image.
    moving_image : np.ndarray
        The moving image.
    annotation_image : np.ndarray
        The annotation image.
    parameter_lists : List[Tuple[str, Dict]], optional
        The list of parameter lists, by default None

    Returns
    -------
    np.ndarray
        The result image.
    itk.ParameterObject
        The result transform parameters.
    """
    # convert to ITK, view only
    atlas_image_elastix = itk.GetImageViewFromArray(atlas_image).astype(itk.F)
    moving_image_elastix = itk.GetImageViewFromArray(moving_image).astype(
        itk.F
    )

    # This syntax needed for 3D images
    elastix_object = itk.ElastixRegistrationMethod.New(
        moving_image_elastix, atlas_image_elastix, output_directory="./output"
    )

    parameter_object = setup_parameter_object(parameter_lists=parameter_lists)

    elastix_object.SetParameterObject(parameter_object)

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

    result_transform_parameters.SetParameter(
        "FinalBSplineInterpolationOrder", temp_interp_order
    )

    # Invert the transformation
    # Import Default Parameter Map and adjust parameters
    parameter_object_inverse = setup_parameter_object(parameter_lists[-1::-1])
    for i in range(len(parameter_lists)):
        parameter_object_inverse.SetParameter(
            i, "HowToCombineTransforms", ["Compose"]
        )

    (
        inverse_image,
        inverse_transform_parameters,
    ) = itk.elastix_registration_method(
        moving_image_elastix,
        moving_image_elastix,
        parameter_object=parameter_object_inverse,
        initial_transform_parameter_file_name=f"output/TransformParameters.{len(parameter_lists)-1}.txt",
    )

    # Adjust inverse transform parameters object
    inverse_transform_parameters.SetParameter(
        0, "InitialTransformParameterFileName", "NoInitialTransform"
    )

    file_names = [
        f"InverseTransformParameters.{i}.txt"
        for i in range(len(parameter_lists))
    ]

    itk.ParameterObject.WriteParameterFiles(
        inverse_transform_parameters, file_names
    )

    inverse_moving = itk.transformix_filter(
        moving_image,
        inverse_transform_parameters,
    )

    return (
        np.asarray(result_image),
        result_transform_parameters,
        np.asarray(annotation_image_transformix),
        np.asarray(inverse_image),
        np.asarray(inverse_moving),
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
