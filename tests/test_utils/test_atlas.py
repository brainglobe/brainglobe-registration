import numpy as np
import pytest
from brainglobe_atlasapi import BrainGlobeAtlas

from brainglobe_registration.utils.atlas import (
    calculate_region_size,
    convert_atlas_labels,
    generate_mask_from_atlas_annotations,
    mask_atlas,
    mask_atlas_with_annotations,
    restore_atlas_labels,
)


def test_convert_atlas_labels_no_change():
    mock_annotations = np.arange(1024).reshape((32, 32))

    result, mapping = convert_atlas_labels(mock_annotations)

    assert np.array_equal(result, mock_annotations)
    assert len(mapping) == 0


def test_convert_atlas_labels_high_labels():
    mock_annotations = np.arange(2**15, 2**15 + 1024).reshape((32, 32))

    result, mapping = convert_atlas_labels(mock_annotations)

    # Since the labels are consecutive, starting at the max label, there
    # should be no change to the array and the mapping should be empty
    assert np.array_equal(result, mock_annotations)
    assert len(mapping) == 0


def test_convert_atlas_labels():
    rng = np.random.default_rng(42)

    mock_annotations = rng.integers(
        2**32 - 1, size=(128, 128), dtype=np.uint32
    )

    result, mapping = convert_atlas_labels(mock_annotations)

    max_value = 2**15
    unique_values = np.unique(mock_annotations)
    expected_mapping_count = unique_values[unique_values >= max_value].size

    assert len(mapping) == expected_mapping_count

    restored_image = restore_atlas_labels(result, mapping)

    assert np.array_equal(restored_image, mock_annotations)


def test_calculate_areas(tmp_path):
    atlas = BrainGlobeAtlas("allen_mouse_100um")

    mid_point = atlas.annotation.shape[0] // 2
    mock_annotations = atlas.annotation[mid_point, :, :]
    hemispheres = atlas.hemispheres[mid_point, :, :]

    output_path = tmp_path / "areas.csv"

    out_df = calculate_region_size(
        atlas, mock_annotations, hemispheres, output_path
    )

    assert output_path.exists()
    assert out_df.columns.size == 4

    # Based on regression testing, the following values are expected
    assert out_df.loc[672, "structure_name"] == "Caudoputamen"
    assert out_df.loc[672, "left_area_mm2"] == 1.98
    assert out_df.loc[672, "right_area_mm2"] == 2.0
    assert out_df.loc[672, "total_area_mm2"] == 3.98

    assert out_df.loc[961, "structure_name"] == "Piriform area"
    assert out_df.loc[961, "left_area_mm2"] == 1.27
    assert out_df.loc[961, "right_area_mm2"] == 1.28
    assert out_df.loc[961, "total_area_mm2"] == 2.55


@pytest.fixture
def dummy_atlas(mocker):
    # Patch the BrainGlobeAtlas class
    mock_atlas_class = mocker.patch(
        "brainglobe_registration.utils.atlas.BrainGlobeAtlas"
    )

    # Create a mock instance
    mock_instance = mock_atlas_class.return_value

    # Assign test data
    mock_instance.reference = np.array(
        [[100, 150, 200], [50, 0, 75], [25, 25, 25]], dtype=np.uint16
    )

    mock_instance.annotation = np.array(
        [[1, 0, 2], [0, 0, 3], [4, 0, 0]], dtype=np.uint16
    )

    return mock_instance


def test_generate_mask_from_atlas_annotations(dummy_atlas):
    mask = generate_mask_from_atlas_annotations(dummy_atlas)
    expected_mask = np.array([[1, 0, 1], [0, 0, 1], [1, 0, 0]], dtype=np.uint8)

    assert np.array_equal(mask, expected_mask)


def test_mask_atlas(dummy_atlas):
    # Create a mask to apply manually
    mask = np.array([[1, 0, 1], [0, 0, 1], [1, 0, 0]], dtype=np.uint8)

    masked_image = mask_atlas(dummy_atlas, mask)
    expected = np.array(
        [[100, 0, 200], [0, 0, 75], [25, 0, 0]], dtype=np.uint16
    )

    assert np.array_equal(masked_image, expected)


def test_mask_atlas_with_annotations(dummy_atlas):
    masked_image = mask_atlas_with_annotations(dummy_atlas)
    expected = np.array(
        [[100, 0, 200], [0, 0, 75], [25, 0, 0]], dtype=np.uint16
    )

    assert np.array_equal(masked_image, expected)
