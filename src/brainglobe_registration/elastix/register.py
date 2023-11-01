import itk
import numpy as np
from bg_atlasapi import BrainGlobeAtlas


def get_atlas_by_name(atlas_name: str) -> BrainGlobeAtlas:
    atlas = BrainGlobeAtlas(atlas_name)

    return atlas


def run_registration(
    brain_atlas,
    moving_image,
    rigid=True,
    affine=True,
    bspline=True,
    use_default_params=True,
    affine_iterations="2048",
    log=False,
):
    # convert to ITK, view only
    brain_atlas = itk.GetImageViewFromArray(brain_atlas).astype(itk.F)
    moving_image = itk.GetImageViewFromArray(moving_image).astype(itk.F)

    # This syntax needed for 3D images
    elastix_object = itk.ElastixRegistrationMethod.New(
        brain_atlas, moving_image
    )

    parameter_object = setup_parameter_object(
        rigid=rigid,
        affine=affine,
        bspline=bspline,
        affine_iterations=affine_iterations,
        use_default=use_default_params,
    )

    elastix_object.SetParameterObject(parameter_object)

    # update filter object
    elastix_object.UpdateLargestPossibleRegion()

    # get results
    result_image = elastix_object.GetOutput()
    result_transform_parameters = elastix_object.GetTransformParameterObject()
    return np.asarray(result_image), result_transform_parameters


def setup_parameter_object(
    rigid=False,
    affine=False,
    bspline=False,
    affine_iterations="2048",
    use_default=True,
):
    parameter_object = itk.ParameterObject.New()

    if rigid and use_default:
        parameter_map_rigid = parameter_object.GetDefaultParameterMap("rigid")
        parameter_object.AddParameterMap(parameter_map_rigid)

    if affine:
        if use_default:
            parameter_map_affine = parameter_object.GetDefaultParameterMap(
                "affine"
            )
            # parameter_map_affine["MaximumNumberOfIterations"] = [
            #     affine_iterations
            # ]
            parameter_object.AddParameterMap(parameter_map_affine)
        else:
            parameter_object.AddParameterFile(
                "./parameters/ara_tools/affine.txt"
            )

    if bspline:
        if use_default:
            parameter_map_bspline = parameter_object.GetDefaultParameterMap(
                "bspline"
            )
            parameter_object.AddParameterMap(parameter_map_bspline)
        else:
            parameter_object.AddParameterFile(
                "./parameters/ara_tools/bspline.txt"
            )

    return parameter_object
