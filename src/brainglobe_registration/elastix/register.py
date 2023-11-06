import itk
import numpy as np
from bg_atlasapi import BrainGlobeAtlas
from typing import List


def get_atlas_by_name(atlas_name: str) -> BrainGlobeAtlas:
    atlas = BrainGlobeAtlas(atlas_name)

    return atlas


def run_registration(
    atlas_image,
    moving_image,
    parameter_lists: List[tuple[str, dict]] = None,
):
    # convert to ITK, view only
    atlas_image = itk.GetImageViewFromArray(atlas_image).astype(itk.F)
    moving_image = itk.GetImageViewFromArray(moving_image).astype(itk.F)

    # This syntax needed for 3D images
    elastix_object = itk.ElastixRegistrationMethod.New(
        atlas_image, moving_image
    )

    parameter_object = setup_parameter_object(parameter_lists=parameter_lists)

    elastix_object.SetParameterObject(parameter_object)

    # update filter object
    elastix_object.UpdateLargestPossibleRegion()

    # get results
    result_image = elastix_object.GetOutput()
    result_transform_parameters = elastix_object.GetTransformParameterObject()

    return np.asarray(result_image), result_transform_parameters


def setup_parameter_object(parameter_lists: List[tuple[str, dict]] = None):
    parameter_object = itk.ParameterObject.New()

    for transform_type, parameter_dict in parameter_lists:
        parameter_map = parameter_object.GetDefaultParameterMap(transform_type)
        parameter_map.clear()

        for k, v in parameter_dict.items():
            parameter_map[k] = v

        parameter_object.AddParameterMap(parameter_map)

    return parameter_object
