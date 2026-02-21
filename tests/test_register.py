from pathlib import Path

import itk
import numpy as np
import pytest
from brainglobe_atlasapi import BrainGlobeAtlas
from tifffile import imread

from brainglobe_registration.elastix.register import (
    calculate_deformation_field,
    crop_atlas,
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
def atlas_hemispheres(atlas, slice_number=SLICE_NUMBER):
    return atlas.hemispheres[slice_number, :, :]


@pytest.fixture(scope="module")
def load_transform_parameters():
    transform_parameters = itk.ParameterObject.New()
    transform_parameters.AddParameterFile(
        str(Path(__file__).parent / "test_images/TransformParameters.0.txt")
    )

    return transform_parameters


@pytest.fixture(scope="module")
def load_invert_parameters():
    transform_parameters = itk.ParameterObject.New()
    transform_parameters.AddParameterFile(
        str(
            Path(__file__).parent
            / "test_images/InverseTransformParameters.0.txt"
        )
    )

    return transform_parameters


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
    atlas_annotation, load_transform_parameters
):
    transform_parameters = load_transform_parameters

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


def test_transform_image(load_invert_parameters, sample_moving_image):
    invert_parameters = load_invert_parameters

    transformed_image = transform_image(sample_moving_image, invert_parameters)

    expected_image = imread(
        Path(__file__).parent / "test_images/registered_sample.tiff"
    )

    assert np.allclose(transformed_image, expected_image, atol=0.1)


def test_calculate_deformation_field(
    sample_moving_image, load_transform_parameters
):
    transform_parameters = load_transform_parameters

    deformation_field = calculate_deformation_field(
        sample_moving_image, transform_parameters
    )

    deformation_field_0 = imread(
        Path(__file__).parent / "test_images/deformation_field_0.tiff"
    )
    deformation_field_1 = imread(
        Path(__file__).parent / "test_images/deformation_field_1.tiff"
    )
    expected_deformation_field = np.stack(
        (deformation_field_0, deformation_field_1), axis=-1
    )

    assert np.allclose(deformation_field, expected_deformation_field, atol=0.5)


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


def test_run_registration_creates_directory(
    tmp_path, atlas_reference, sample_moving_image, monkeypatch
):
    """
    Test that `run_registration` creates the output directory and writes
    transform parameter files."""
    non_existent_folder = tmp_path / "new_output_folder"
    assert not non_existent_folder.exists()

    dummy_params = [("affine", {"MaximumNumberOfIterations": ["1"]})]

    # Create a parameter object that the dummy elastix object will return.
    expected_param_obj = setup_parameter_object(dummy_params)

    class DummyElastix:
        def __init__(self, moving, fixed):
            self._param_obj = expected_param_obj

        def SetParameterObject(self, po):
            self._param_obj = po

        def UpdateLargestPossibleRegion(self):
            return None

        def GetTransformParameterObject(self):
            return self._param_obj

    monkeypatch.setattr(
        itk.ElastixRegistrationMethod,
        "New",
        lambda moving, fixed: DummyElastix(moving, fixed),
    )

    # Run registration attempting to save to the non-existent folder
    run_registration(
        atlas_reference,
        sample_moving_image,
        dummy_params,
        output_directory=non_existent_folder,
    )

    assert non_existent_folder.exists()
    assert (non_existent_folder / "TransformParameters.0.txt").exists()


@pytest.mark.parametrize(
    "brain_geometry",
    [
        "full",
        "hemisphere_l",
        "hemisphere_r",
        "quarter_al",
        "quarter_ar",
        "quarter_pl",
        "quarter_pr",
    ],
)
def test_crop_atlas(atlas, brain_geometry):
    """Test that crop_atlas returns valid atlases for all geometry types."""
    cropped = crop_atlas(atlas, brain_geometry)

    assert isinstance(cropped, BrainGlobeAtlas)
    
    # Assert shapes are preserved
    assert cropped.reference.shape == atlas.reference.shape
    assert cropped.annotation.shape == atlas.annotation.shape
    assert cropped.hemispheres.shape == atlas.hemispheres.shape

    # For full brain, should be identical
    if brain_geometry == "full":
        assert np.array_equal(cropped.reference, atlas.reference)
        assert np.array_equal(cropped.annotation, atlas.annotation)
    else:
        # For partial brains, some regions should be masked (set to 0)
        # The masked atlas should have fewer non-zero voxels
        assert np.sum(cropped.reference == 0) > np.sum(atlas.reference == 0)


def test_crop_atlas_hemisphere_l(atlas):
    """Test that left hemisphere keeps left side masked."""
    cropped = crop_atlas(atlas, "hemisphere_l")

    right_hem_locations = atlas.hemispheres == atlas.right_hemisphere_value
    assert np.all(cropped.reference[right_hem_locations] == 0)
    assert np.all(cropped.annotation[right_hem_locations] == 0)

    left_hem_locations = atlas.hemispheres == atlas.left_hemisphere_value
    assert np.any(cropped.reference[left_hem_locations] != 0)


def test_crop_atlas_hemisphere_r(atlas):
    """Test that right hemisphere keeps right side masked."""
    cropped = crop_atlas(atlas, "hemisphere_r")

    left_hem_locations = atlas.hemispheres == atlas.left_hemisphere_value
    assert np.all(cropped.reference[left_hem_locations] == 0)
    assert np.all(cropped.annotation[left_hem_locations] == 0)

    right_hem_locations = atlas.hemispheres == atlas.right_hemisphere_value
    assert np.any(cropped.reference[right_hem_locations] != 0)


def test_crop_atlas_quarter_al(atlas):
    """Test that anterior-left quarter masks correctly.
    
    BrainGlobe "asr" orientation:
    - Index 0 (AP): 0=Anterior, max=Posterior
    - Index 2 (ML): 0=Right, max=Left
    So Anterior-Left = keep small indices on axis 0, keep large indices on axis 2
    """
    cropped = crop_atlas(atlas, "quarter_al")

   
    cropped_nonzero = np.sum(cropped.reference != 0)
    full_nonzero = np.sum(atlas.reference != 0)
    assert cropped_nonzero < full_nonzero

    # Posterior (large indices on axis 0) should be completely masked
    ap_axis = 0
    ap_midpoint = atlas.reference.shape[ap_axis] // 2
    posterior_locations = np.arange(ap_midpoint, atlas.reference.shape[ap_axis])
    posterior_slice = (posterior_locations, slice(None), slice(None))
    assert np.all(cropped.reference[posterior_slice] == 0)

    # Right (small indices on axis 2) should be completely masked
    ml_axis = 2
    ml_midpoint = atlas.reference.shape[ml_axis] // 2
    right_locations = np.arange(0, ml_midpoint)
    right_slice = (slice(None), slice(None), right_locations)
    assert np.all(cropped.reference[right_slice] == 0)


def test_crop_atlas_invalid_geometry(atlas):
    """Test that invalid geometry raises ValueError."""
    with pytest.raises(ValueError):
        crop_atlas(atlas, "invalid_geometry")
