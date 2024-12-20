from pathlib import Path

import numpy as np
import pytest
from tifffile import imread

from brainglobe_registration.elastix.register import (
    run_registration,
    setup_parameter_object,
)


@pytest.fixture(scope="module")
def sample_atlas_slice():
    return imread(Path(__file__).parent / "test_images/Atlas_Hipp.tif")


@pytest.fixture(scope="module")
def sample_moving_image():
    return imread(Path(__file__).parent / "test_images/sample_hipp.tif")


@pytest.fixture(scope="module")
def registration_affine_only(
    sample_atlas_slice, sample_moving_image, parameter_lists_affine_only
):
    yield run_registration(
        sample_atlas_slice,
        sample_moving_image,
        parameter_lists_affine_only,
    )


def test_run_registration(registration_affine_only):
    result_image, transform_parameters = registration_affine_only

    expected_result_image = imread(
        Path(__file__).parent / "test_images/registered_atlas.tiff"
    )

    assert np.allclose(result_image, expected_result_image, atol=0.1)
    assert transform_parameters is not None


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
