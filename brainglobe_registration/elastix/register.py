from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Tuple

import itk
import numpy as np
import numpy.typing as npt
from brainglobe_atlasapi import BrainGlobeAtlas

from brainglobe_registration.utils.atlas import (
    convert_atlas_labels,
    restore_atlas_labels,
)
from brainglobe_registration.utils.preprocess import filter_image


def crop_atlas(atlas: BrainGlobeAtlas, brain_geometry: str) -> BrainGlobeAtlas:
    """
    Crop an atlas to match the brain geometry (full brain, hemisphere, or quarter).

    When registering a partial brain to a full brain atlas,
    the unwanted regions must be masked out.

    Parameters
    ----------
    atlas : BrainGlobeAtlas
        The atlas to crop.
    brain_geometry : str
        The brain geometry type:
        - "full": Full brain
        - "hemisphere_l": Left hemisphere (masks right hemisphere)
        - "hemisphere_r": Right hemisphere (masks left hemisphere)
        - "quarter_al": Anterior left quarter (masks other 3 quarters)
        - "quarter_ar": Anterior right quarter (masks other 3 quarters)
        - "quarter_pl": Posterior left quarter (masks other 3 quarters)
        - "quarter_pr": Posterior right quarter (masks other 3 quarters)

    Returns
    -------
    BrainGlobeAtlas
        A new atlas with the specified region masked.
    """
    if brain_geometry == "full":
        return atlas

    atlas_cropped = deepcopy(atlas)
    geom = brain_geometry.lower()

    # hemisphere case
    if geom == "hemisphere_l":
        ind = atlas_cropped.right_hemisphere_value
        atlas_cropped.reference[atlas_cropped.hemispheres == ind] = 0
        atlas_cropped.annotation[atlas_cropped.hemispheres == ind] = 0
        return atlas_cropped

    elif geom == "hemisphere_r":
        ind = atlas_cropped.left_hemisphere_value
        atlas_cropped.reference[atlas_cropped.hemispheres == ind] = 0
        atlas_cropped.annotation[atlas_cropped.hemispheres == ind] = 0
        return atlas_cropped

    # quarter case
    elif geom.startswith("quarter_"):
        quarter_code = geom.split("_", maxsplit=1)[1]
        wants_left = quarter_code.endswith("l")
        wants_anterior = quarter_code.startswith("a")

        hemisphere_value_to_keep = (
            atlas_cropped.left_hemisphere_value
            if wants_left
            else atlas_cropped.right_hemisphere_value
        )
        hemisphere_keep_mask = (
            atlas_cropped.hemispheres == hemisphere_value_to_keep
        )

        # asr
        # axis 0: AP (anterior -> posterior)
        ap_axis = 0
        ap_size = atlas_cropped.reference.shape[ap_axis]
        ap_midpoint = ap_size // 2
        ap_coords = np.arange(ap_size).reshape((-1, 1, 1))
        ap_keep_mask = (
            ap_coords < ap_midpoint
            if wants_anterior
            else ap_coords >= ap_midpoint
        )

        keep_mask = hemisphere_keep_mask & ap_keep_mask

        atlas_cropped.reference[~keep_mask] = 0
        atlas_cropped.annotation[~keep_mask] = 0

        return atlas_cropped
    else:
        raise ValueError(
            f"Unknown brain geometry: {brain_geometry}. "
            "must be 'full', 'hemisphere_l', 'hemisphere_r', "
            "'quarter_al', 'quarter_ar', 'quarter_pl', or 'quarter_pr'."
        )


def run_registration(
    atlas_image: npt.NDArray,
    moving_image: npt.NDArray,
    parameter_lists: List[Tuple[str, dict]],
    output_directory: Optional[Path] = None,
    filter_images: bool = True,
) -> itk.ParameterObject:
    """
    Run the registration process on the given images.

    Parameters
    ----------
    atlas_image : npt.NDArray
        The atlas image.
    moving_image : npt.NDArray
        The moving image.
    parameter_lists : List[tuple[str, dict]]
        The list of registration parameters, one for each transform.
    output_directory : Optional[Path], optional
        The output directory for the registration results, by default None.
    filter_images : bool, optional
        Whether to filter the images before registration, by default True.

    Returns
    -------
    itk.ParameterObject
        The result transform parameters.
    """
    if filter_images:
        atlas_image = filter_image(atlas_image)
        moving_image = filter_image(moving_image)

    # convert to ITK, view only
    atlas_image = itk.GetImageViewFromArray(atlas_image)
    moving_image = itk.GetImageViewFromArray(moving_image)

    # This syntax needed for 3D images
    elastix_object = itk.ElastixRegistrationMethod.New(
        moving_image, atlas_image
    )

    parameter_object = setup_parameter_object(parameter_lists=parameter_lists)

    elastix_object.SetParameterObject(parameter_object)
    elastix_object.UpdateLargestPossibleRegion()

    # get results
    result_transform_parameters = elastix_object.GetTransformParameterObject()

    if output_directory:

        Path(output_directory).mkdir(parents=True, exist_ok=True)
        file_names = [
            f"{output_directory}/TransformParameters.{i}.txt"
            for i in range(len(parameter_lists))
        ]

        itk.ParameterObject.WriteParameterFile(
            result_transform_parameters, file_names
        )

    return result_transform_parameters


def transform_annotation_image(
    annotation_image: npt.NDArray[np.uint32],
    transform_parameters: itk.ParameterObject,
) -> npt.NDArray[np.uint32]:
    """
    Transform the annotation image using the given transform parameters.
    Sets the FinalBSplineInterpolationOrder to 0 to avoid interpolation.
    Resets the FinalBSplineInterpolationOrder to its original value after
    transforming the annotation image.

    Parameters
    ----------
    annotation_image : npt.NDArray
        The annotation image.
    transform_parameters : itk.ParameterObject
        The transform parameters.

    Returns
    -------
    npt.NDArray
        The transformed annotation image.
    """
    adjusted_annotation_image, mapping = convert_atlas_labels(annotation_image)

    if adjusted_annotation_image.ndim == 2:
        adjusted_annotation_image = adjusted_annotation_image.astype(
            np.float32
        )

    annotation_image = itk.GetImageViewFromArray(adjusted_annotation_image)

    temp_interp_order = transform_parameters.GetParameter(
        0, "FinalBSplineInterpolationOrder"
    )
    transform_parameters.SetParameter("FinalBSplineInterpolationOrder", "0")

    transformix_object = itk.TransformixFilter.New(annotation_image)
    transformix_object.SetTransformParameterObject(transform_parameters)
    transformix_object.UpdateLargestPossibleRegion()

    transformed_annotation = transformix_object.GetOutput()

    transform_parameters.SetParameter(
        "FinalBSplineInterpolationOrder", temp_interp_order
    )
    del annotation_image

    transformed_annotation_array = np.asarray(transformed_annotation).astype(
        np.uint16
    )

    transformed_annotation_array = restore_atlas_labels(
        transformed_annotation_array, mapping
    )

    return transformed_annotation_array


def transform_image(
    image: npt.NDArray,
    transform_parameters: itk.ParameterObject,
) -> npt.NDArray:
    """
    Transform the image using the given transform parameters.

    Parameters
    ----------
    image: npt.NDArray
        The image to transform.
    transform_parameters: itk.ParameterObject
        The transform parameters.

    Returns
    -------
    npt.NDArray
        The transformed image.
    """
    image = itk.GetImageViewFromArray(image)

    transformix_object = itk.TransformixFilter.New(image)
    transformix_object.SetTransformParameterObject(transform_parameters)
    transformix_object.UpdateLargestPossibleRegion()

    transformed_image = transformix_object.GetOutput()

    # Convert to a numpy array of the original type.
    transformed_image = np.asarray(transformed_image).astype(image.dtype)

    return transformed_image


def calculate_deformation_field(
    moving_image: npt.NDArray,
    transform_parameters: itk.ParameterObject,
    debug: bool = False,
) -> npt.NDArray:
    """
    Calculate the deformation field for the moving image using the given
    transform parameters.

    Parameters
    ----------
    moving_image : npt.NDArray
        The moving image.
    transform_parameters : itk.ParameterObject
        The transform parameters.
    debug : bool, optional
        Whether to save extra files for debugging, by default False

    Returns
    -------
    npt.NDArray
        The deformation field.
    """
    transformix_object = itk.TransformixFilter.New(
        itk.GetImageViewFromArray(moving_image),
        transform_parameters,
    )
    transformix_object.SetComputeDeformationField(True)

    transformix_object.UpdateLargestPossibleRegion()

    # Change from ITK to numpy axes ordering
    deformation_field = itk.GetArrayViewFromImage(
        transformix_object.GetOutputDeformationField()
    )[..., ::-1]

    if not debug:
        # Cleanup files generated by elastix
        (Path.cwd() / "deformationField.tiff").unlink(missing_ok=True)

    return deformation_field


def invert_transformation(
    fixed_image: npt.NDArray,
    parameter_list: List[Tuple[str, dict]],
    transform_parameters: itk.ParameterObject,
    output_directory: Optional[Path] = None,
    filter_images: bool = True,
) -> itk.ParameterObject:
    """
    Invert the transformation of the fixed image using the given transform
    parameters.

    Inverts the transformation by applying the forward transformation to the
    fixed image and registering it to itself.

    Parameters
    ----------
    fixed_image : npt.NDArray
        The reference image.
    parameter_list : List[Tuple[str, dict]]
        The list of registration parameters, one for each transform.
    transform_parameters : itk.ParameterObject
        The transform parameters to inverse.
    output_directory : Optional[Path], optional
        The output directory for the registration results, by default None.
    filter_images : bool, optional
        Whether to filter the images before registration, by default True.

    Returns
    -------
    itk.ParameterObject
        The inverse transform parameters.
    """
    if filter_images:
        fixed_image = filter_image(fixed_image)

    fixed_image = itk.GetImageViewFromArray(fixed_image)

    elastix_object = itk.ElastixRegistrationMethod.New(
        fixed_image, fixed_image
    )

    parameter_object_inverse = setup_parameter_object(parameter_list)
    elastix_object.SetInitialTransformParameterObject(transform_parameters)
    elastix_object.SetParameterObject(parameter_object_inverse)

    elastix_object.UpdateLargestPossibleRegion()

    num_initial_transforms = transform_parameters.GetNumberOfParameterMaps()

    out_parameters = elastix_object.GetTransformParameterObject()
    result_transform_parameters = itk.ParameterObject.New()

    for i in range(
        num_initial_transforms, out_parameters.GetNumberOfParameterMaps()
    ):
        result_transform_parameters.AddParameterMap(
            out_parameters.GetParameterMap(i)
        )

    result_transform_parameters.SetParameter(
        0, "InitialTransformParameterFileName", "NoInitialTransform"
    )

    if output_directory:
        file_names = [
            f"{output_directory}/InverseTransformParameters.{i}.txt"
            for i in range(len(parameter_list))
        ]

        itk.ParameterObject.WriteParameterFiles(
            result_transform_parameters, file_names
        )

    return result_transform_parameters


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
        The parameter object.
    """
    parameter_object = itk.ParameterObject.New()

    for transform_type, parameter_dict in parameter_lists:
        parameter_map = parameter_object.GetDefaultParameterMap(transform_type)
        parameter_map.clear()

        for k, v in parameter_dict.items():
            parameter_map[k] = v

        parameter_object.AddParameterMap(parameter_map)

    return parameter_object
