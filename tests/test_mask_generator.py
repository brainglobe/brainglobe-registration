import numpy as np
import pytest
from brainglobe_registration.preprocessing.mask_generator import generate_mask_from_atlas, mask_atlas

# Dummy class to simulate BrainGlobeAtlas
class DummyAtlas:
    def __init__(self, reference, annotation):
        self.reference = reference
        self.annotation = annotation


@pytest.fixture
def dummy_atlas():
    reference = np.array([
        [100, 150, 200],
        [50, 0, 75],
        [25, 25, 25]
    ], dtype=np.uint16)

    annotation = np.array([
        [1, 0, 2],
        [0, 0, 3],
        [4, 0, 0]
    ], dtype=np.uint16)

    return DummyAtlas(reference, annotation)


def test_generate_mask_from_atlas(dummy_atlas):
    mask = generate_mask_from_atlas(dummy_atlas)
    expected_mask = np.array([
        [1, 0, 1],
        [0, 0, 1],
        [1, 0, 0]
    ], dtype=np.uint8)

    assert np.array_equal(mask, expected_mask)


def test_mask_atlas(dummy_atlas):
    masked_image = mask_atlas(dummy_atlas)
    expected_masked_image = np.array([
        [100, 0, 200],
        [0, 0, 75],
        [25, 0, 0]
    ], dtype=np.uint16)

    assert np.array_equal(masked_image, expected_masked_image)
