import numpy as np
import pytest
from skimage.metrics import structural_similarity

from brainglobe_registration.widgets.automated_target_selection import (
    pad_to_match_shape,
    normalise_image,
    scale_moving_image,
    prepare_images,
    safe_ncc,
    compute_similarity_metric,
    registration_objective,
)