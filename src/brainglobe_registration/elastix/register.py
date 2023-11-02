import itk
import numpy as np
from bg_atlasapi import BrainGlobeAtlas


def get_atlas_by_name(atlas_name: str) -> BrainGlobeAtlas:
    atlas = BrainGlobeAtlas(atlas_name)

    return atlas


def run_registration(
    atlas_image,
    moving_image,
    rigid=True,
    affine=True,
    bspline=True,
    parameter_dicts: dict[str] = None,
):
    # convert to ITK, view only
    atlas_image = itk.GetImageViewFromArray(atlas_image).astype(itk.F)
    moving_image = itk.GetImageViewFromArray(moving_image).astype(itk.F)

    # This syntax needed for 3D images
    elastix_object = itk.ElastixRegistrationMethod.New(
        atlas_image, moving_image
    )

    parameter_object = setup_parameter_object(
        rigid=rigid,
        affine=affine,
        bspline=bspline,
        parameter_dicts=parameter_dicts,
    )

    print(parameter_object)

    elastix_object.SetParameterObject(parameter_object)

    # update filter object
    elastix_object.UpdateLargestPossibleRegion()

    # get results
    result_image = elastix_object.GetOutput()
    result_transform_parameters = elastix_object.GetTransformParameterObject()
    return np.asarray(result_image), result_transform_parameters


def setup_parameter_object(
    rigid: bool = False,
    affine: bool = False,
    bspline: bool = False,
    parameter_dicts: dict[str] = None,
    use_default=False,
):
    parameter_object = itk.ParameterObject.New()

    if rigid and not use_default:
        # Hack way of getting a new parameter map
        # Create an empty parameter map
        parameter_map_rigid = parameter_object.GetDefaultParameterMap("rigid")

        for param in parameter_map_rigid:
            parameter_map_rigid.pop(param)

        for k, v in parameter_dicts["rigid"].items():
            parameter_map_rigid[k] = [v]

        parameter_object.AddParameterMap(parameter_map_rigid)

    if affine and not use_default:
        parameter_map_affine = parameter_object.GetDefaultParameterMap("rigid")

        for param in parameter_map_affine:
            parameter_map_affine.pop(param)

        for k, v in parameter_dicts["affine"].items():
            parameter_map_affine[k] = [v]

        parameter_object.AddParameterMap(parameter_map_affine)

    if bspline:
        parameter_map_bspline = parameter_object.GetDefaultParameterMap(
            "rigid"
        )

        for param in parameter_map_bspline:
            parameter_map_bspline.pop(param)

        for k, v in parameter_dicts["bspline"].items():
            parameter_map_bspline[k] = [v]

        parameter_object.AddParameterMap(parameter_map_bspline)

    return parameter_object
