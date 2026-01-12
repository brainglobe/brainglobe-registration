import numpy as np
import pytest
from pytransform3d.rotations import active_matrix_from_angle

from brainglobe_registration.utils.transforms import (
    calculate_rotated_bounding_box,
)


@pytest.mark.parametrize(
    "basis, rotation, expected_bounds",
    [
        (0, 0, (50, 100, 200)),
        (0, 90, (50, 200, 100)),
        (0, 180, (50, 100, 200)),
        (0, 45, (50, 212, 212)),
        (1, 0, (50, 100, 200)),
        (1, 90, (200, 100, 50)),
        (1, 180, (50, 100, 200)),
        (1, 45, (177, 100, 177)),
        (2, 0, (50, 100, 200)),
        (2, 90, (100, 50, 200)),
        (2, 180, (50, 100, 200)),
        (2, 45, (106, 106, 200)),
    ],
)
def test_calculate_rotated_bounding_box(basis, rotation, expected_bounds):
    image_shape = (50, 100, 200)
    rotation_matrix = np.eye(4)
    rotation_matrix[:-1, :-1] = active_matrix_from_angle(
        basis, np.deg2rad(rotation)
    )

    result_shape = calculate_rotated_bounding_box(image_shape, rotation_matrix)

    assert result_shape == expected_bounds
