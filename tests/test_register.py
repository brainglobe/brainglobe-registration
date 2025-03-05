from pathlib import Path

import itk
import numpy as np
import pytest
from brainglobe_atlasapi import BrainGlobeAtlas
from tifffile import imread

from brainglobe_registration.elastix.register import (
    calculate_deformation_field,
    invert_transformation,
    run_registration,
    setup_parameter_object,
    transform_annotation_image,
    transform_image,
)

SLICE_NUMBER = 293


def compare_parameter_objects(param_obj1, param_obj2):
    assert (
        param_obj1.GetNumberOfParameterMaps()
        == param_obj2.GetNumberOfParameterMaps()
    )

    for index in range(param_obj1.GetNumberOfParameterMaps()):
        submap_1 = dict(param_obj1.GetParameterMap(index))
        submap_2 = dict(param_obj2.GetParameterMap(index))

        for key in submap_1.keys():
            if key in [
                "TransformParameters",
                "CenterOfRotationPoint",
                "GridOrigin",
                "GridSpacing",
            ]:
                assert np.allclose(
                    np.array(submap_1[key], dtype=np.double),
                    np.array(submap_2[key], dtype=np.double),
                    atol=0.4,
                )
            else:
                assert submap_1[key] == submap_2[key]


@pytest.fixture(scope="module")
def atlas(atlas_name="allen_mouse_25um"):
    return BrainGlobeAtlas(atlas_name)


@pytest.fixture(scope="module")
def atlas_reference(atlas, slice_number=SLICE_NUMBER):
    return atlas.reference[slice_number, :, :]


@pytest.fixture(scope="module")
def atlas_annotation(atlas, slice_number=SLICE_NUMBER):
    # Need the astype call to avoid a crash on Windows
    return atlas.annotation[slice_number, :, :].astype(np.uint32)


@pytest.fixture(scope="module")
def atlas_hemisphere(atlas, slice_number=SLICE_NUMBER):
    return atlas.hemispheres[slice_number, :, :]


@pytest.fixture(scope="module")
def sample_moving_image():
    return imread(
        Path(__file__).parent / "test_images/sample_hipp.tif"
    ).astype(np.float32)


@pytest.fixture(scope="module")
def registration_affine_only(
    atlas_reference, sample_moving_image, parameter_lists_affine_only
):
    yield run_registration(
        atlas_reference,
        sample_moving_image,
        parameter_lists_affine_only,
    )


@pytest.fixture(scope="module")
def invert_transform(
    registration_affine_only, atlas_reference, parameter_lists_affine_only
):
    transform_parameters = registration_affine_only
    invert_parameters = invert_transformation(
        atlas_reference, parameter_lists_affine_only, transform_parameters
    )

    yield invert_parameters, transform_parameters


def test_run_registration(registration_affine_only):
    transform_parameters = registration_affine_only

    expected_parameter_object = itk.ParameterObject.New()
    expected_parameter_object.AddParameterFile(
        str(Path(__file__).parent / "test_images/TransformParameters.0.txt")
    )

    compare_parameter_objects(transform_parameters, expected_parameter_object)


def test_transform_annotation_image(
    atlas_annotation, registration_affine_only
):
    transform_parameters = registration_affine_only

    transformed_annotation = transform_annotation_image(
        atlas_annotation, transform_parameters
    )

    expected_transformed_annotation = imread(
        Path(__file__).parent / "test_images/registered_atlas.tiff"
    )

    assert np.allclose(transformed_annotation, expected_transformed_annotation)


def test_invert_transformation(invert_transform):
    invert_parameters, original_parameters = invert_transform

    expected_parameter_object = itk.ParameterObject.New()
    expected_parameter_object.AddParameterFile(
        str(
            Path(__file__).parent
            / "test_images/InverseTransformParameters.0.txt"
        )
    )

    compare_parameter_objects(invert_parameters, expected_parameter_object)

    for i in range(original_parameters.GetNumberOfParameterMaps()):
        assert original_parameters.GetParameter(
            i, "FinalBSplineInterpolationOrder"
        ) == ("3",)


def test_transform_image(invert_transform, sample_moving_image):
    invert_parameters, _ = invert_transform

    transformed_image = transform_image(sample_moving_image, invert_parameters)

    expected_image = imread(
        Path(__file__).parent / "test_images/registered_sample.tiff"
    )

    assert np.allclose(transformed_image, expected_image, atol=0.1)


def test_calculate_deformation_field(
    sample_moving_image, registration_affine_only
):
    transform_parameters = registration_affine_only

    deformation_field = calculate_deformation_field(
        sample_moving_image, transform_parameters
    )[..., ::-1]

    deformation_field_0 = imread(
        Path(__file__).parent / "test_images/deformation_field_0.tiff"
    )
    deformation_field_1 = imread(
        Path(__file__).parent / "test_images/deformation_field_1.tiff"
    )
    expected_deformation_field = np.stack(
        (deformation_field_0, deformation_field_1), axis=-1
    )

    assert np.allclose(deformation_field, expected_deformation_field, atol=0.1)


def test_setup_parameter_object_empty_list():
    parameter_list = []

    param_obj = setup_parameter_object(parameter_list)

    assert param_obj.GetNumberOfParameterMaps() == 0


@pytest.mark.parametrize(
    "parameter_list, expected",
    [
        (
            [("rigid", {"Transform": ["EulerTransform"]})],
            [("EulerTransform",)],
        ),
        (
            [("affine", {"Transform": ["AffineTransform"]})],
            [("AffineTransform",)],
        ),
        (
            [("bspline", {"Transform": ["BSplineTransform"]})],
            [("BSplineTransform",)],
        ),
        (
            [
                ("rigid", {"Transform": ["EulerTransform"]}),
                ("affine", {"Transform": ["AffineTransform"]}),
                ("bspline", {"Transform": ["BSplineTransform"]}),
            ],
            [("EulerTransform",), ("AffineTransform",), ("BSplineTransform",)],
        ),
        (
            [
                ("rigid", {"Transform": ["EulerTransform"]}),
                ("rigid", {"Transform": ["EulerTransform"]}),
                ("rigid", {"Transform": ["EulerTransform"]}),
            ],
            [("EulerTransform",), ("EulerTransform",), ("EulerTransform",)],
        ),
    ],
)
def test_setup_parameter_object_one_transform(parameter_list, expected):
    param_obj = setup_parameter_object(parameter_list)

    assert param_obj.GetNumberOfParameterMaps() == len(expected)

    for index, transform_type in enumerate(expected):
        assert param_obj.GetParameterMap(index)["Transform"] == transform_type
