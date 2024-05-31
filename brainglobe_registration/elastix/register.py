from typing import List

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
    parameter_lists: List[tuple[str, dict]],
) -> tuple[np.ndarray, itk.ParameterObject, np.ndarray]:
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
    parameter_lists : List[tuple[str, dict]], optional
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
    parameter_object_inverse = itk.ParameterObject.New()
    parameter_map_rigid = parameter_object_inverse.GetDefaultParameterMap(
        "rigid"
    )
    parameter_map_bspline = parameter_object_inverse.GetDefaultParameterMap(
        "bspline"
    )
    parameter_map_bspline["HowToCombineTransforms"] = ["Compose"]
    parameter_map_rigid["HowToCombineTransforms"] = ["Compose"]
    parameter_object_inverse.AddParameterMap(parameter_map_rigid)
    parameter_object_inverse.AddParameterMap(parameter_map_bspline)

    (
        inverse_image,
        inverse_transform_parameters,
    ) = itk.elastix_registration_method(
        atlas_image_elastix,
        atlas_image_elastix,
        parameter_object=parameter_object,
        initial_transform_parameter_file_name="output/TransformParameters.2.txt",
    )

    # Adjust inverse transform parameters object
    inverse_transform_parameters.SetParameter(
        0, "InitialTransformParametersFileName", "NoInitialTransform"
    )

    # inverse_transform_parameters.WriteParameterFiles(
    # "output/TransformParameters_inverse"
    # )

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
