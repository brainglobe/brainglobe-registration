import numpy as np

from brainglobe_registration.optical_flow import (
    compare_deformation_fields,
    compute_optical_flow_skimage,
)


def test_compute_optical_flow_2d():
    # Create two shifted squares
    img1 = np.zeros((100, 100), dtype=np.float32)
    img2 = np.zeros_like(img1)

    img1[30:70, 30:70] = 1
    img2[32:72, 35:75] = 1  # Shifted down and right

    flow = compute_optical_flow_skimage(img1, img2, method="tvl1")

    assert flow.shape == (2, 100, 100)
    assert np.any(flow != 0), "Flow field should contain non-zero displacement"


def test_compare_deformation_fields_2d():
    shape = (100, 100)
    dummy_flow = np.zeros((2, *shape), dtype=np.float32)
    dummy_elastix = np.zeros((*shape, 2), dtype=np.float32)

    # Simulate some known flow
    dummy_flow[0, 50, 50] = 2  # y
    dummy_flow[1, 50, 50] = 3  # x
    dummy_elastix[50, 50, 0] = 2
    dummy_elastix[50, 50, 1] = 3

    mse = compare_deformation_fields(dummy_elastix, dummy_flow)
    assert mse < 1e-6, "MSE should be near zero for matching fields"
