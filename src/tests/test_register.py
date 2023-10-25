import itk
import pytest
from PIL import Image

from bg_elastix.elastix.register import (
    setup_parameter_object,
    run_registration,
)


@pytest.fixture
def sample_atlas():
    return Image.open("test_images/Atlas_Hipp.tif")


@pytest.fixture
def sample_moving_image():
    return Image.open("test_images/Sample_Hipp.tif")


@pytest.mark.slow
def test_run_registration(sample_atlas, sample_moving_image):
    result_image, transform_parameters = run_registration(
        sample_atlas,
        sample_moving_image,
        rigid=True,
        affine=False,
        bspline=False,
        use_default_params=True,
        affine_iterations="2048",
        log=False,
    )
    assert result_image is not None
    assert transform_parameters is not None


@pytest.mark.xfail()
def test_setup_parameter_object_rigid_custom():
    param_obj = setup_parameter_object(rigid=True, use_default=False)
    test_param_obj = itk.ParameterObject.New()
    test_param_obj.ReadParameterFile("./parameters/ara_tools/rigid.txt")

    default_params = param_obj.GetDefaultParameterMap("rigid")

    assert param_obj.GetParameterMap(0) == test_param_obj.GetParameterMap(0)
    assert param_obj.GetParameterMap(0) != default_params


def test_setup_parameter_object_affine_custom():
    param_obj = setup_parameter_object(affine=True, use_default=False)
    test_param_obj = itk.ParameterObject.New()
    test_param_obj.ReadParameterFile("./parameters/ara_tools/affine.txt")

    default_params = param_obj.GetDefaultParameterMap("affine")

    assert param_obj.GetParameterMap(0) == test_param_obj.GetParameterMap(0)
    assert param_obj.GetParameterMap(0) != default_params


def test_setup_parameter_object_bspline_custom():
    param_obj = setup_parameter_object(bspline=True, use_default=False)
    test_param_obj = itk.ParameterObject.New()
    test_param_obj.ReadParameterFile("./parameters/ara_tools/bspline.txt")

    default_params = param_obj.GetDefaultParameterMap("bspline")

    assert param_obj.GetParameterMap(0) == test_param_obj.GetParameterMap(0)
    assert param_obj.GetParameterMap(0) != default_params


@pytest.mark.parametrize(
    "rigid, affine, bspline, transform_types",
    [
        (True, False, False, ["rigid"]),
        (False, False, True, ["bspline"]),
        (True, True, False, ["rigid", "affine"]),
        (True, False, True, ["rigid", "bspline"]),
        (False, True, True, ["affine", "bspline"]),
        (True, True, True, ["rigid", "affine", "bspline"]),
    ],
)
def test_setup_parameter_object_default_params(
    rigid, affine, bspline, transform_types
):
    param_obj = setup_parameter_object(
        rigid=rigid, affine=affine, bspline=bspline, use_default=True
    )

    for index, transform_type in enumerate(transform_types):
        default_obj = param_obj.GetDefaultParameterMap(transform_type)

        assert param_obj.GetParameterMap(index) == default_obj, (
            f"Expected {default_obj['Transform']} but got "
            f"{param_obj.GetParameterMap(index)['Transform']}"
        )

    assert param_obj.GetNumberOfParameterMaps() == len(transform_types)


@pytest.mark.parametrize(
    "rigid, affine, bspline, transform_types",
    [
        pytest.param(
            True,
            False,
            False,
            ["rigid"],
            marks=pytest.mark.xfail(reason="No rigid parameter file"),
        ),
        (False, True, False, ["affine"]),
        (False, False, True, ["bspline"]),
        pytest.param(
            True,
            True,
            False,
            ["rigid", "affine"],
            marks=pytest.mark.xfail(reason="No rigid parameter file"),
        ),
        pytest.param(
            True,
            False,
            True,
            ["rigid", "bspline"],
            marks=pytest.mark.xfail(reason="No rigid parameter file"),
        ),
        (False, True, True, ["affine", "bspline"]),
        pytest.param(
            True,
            True,
            True,
            ["rigid", "affine", "bspline"],
            marks=pytest.mark.xfail(reason="No rigid parameter file"),
        ),
    ],
)
def test_setup_parameter_object_file_params(
    rigid, affine, bspline, transform_types
):
    param_obj = setup_parameter_object(
        rigid=rigid, affine=affine, bspline=bspline, use_default=False
    )
    test_param_obj = itk.ParameterObject.New()

    for index, transform_type in enumerate(transform_types):
        test_param_obj.AddParameterFile(
            f"./parameters/ara_tools/{transform_type}.txt"
        )

        assert param_obj.GetParameterMap(
            index
        ) == test_param_obj.GetParameterMap(index), (
            f"Expected {test_param_obj.GetParameterMap(index)['Transform']} but got "
            f"{param_obj.GetParameterMap(index)['Transform']}"
        )

    assert param_obj.GetNumberOfParameterMaps() == len(transform_types)
