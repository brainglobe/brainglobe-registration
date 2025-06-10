import numpy as np

from dask_image.ndinterp import affine_transform as dask_affine_transform
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_error
from pytransform3d.rotations import active_matrix_from_angle

from brainglobe_registration.utils.utils import (
    calculate_rotated_bounding_box,

)

def on_adjust_atlas_rotation(self, pitch: float, yaw: float, roll: float):
    if not (
        self._atlas
        and self._atlas_data_layer
        and self._atlas_annotations_layer
    ):
        show_error("No atlas selected. Please select an atlas before rotating")
        return

    # Create the rotation matrix
    roll_matrix = active_matrix_from_angle(0, np.deg2rad(roll))
    yaw_matrix = active_matrix_from_angle(1, np.deg2rad(yaw))
    pitch_matrix = active_matrix_from_angle(2, np.deg2rad(pitch))

    # Combine rotation matrices
    rotation_matrix = yaw_matrix @ pitch_matrix @ roll_matrix

    full_matrix = np.eye(4)
    full_matrix[:3, :3] = rotation_matrix

    # Translate the origin to the center of the image
    origin = np.asarray(self._atlas.reference.shape) / 2
    translate_matrix = np.eye(4)
    translate_matrix[:-1, -1] = -origin

    bounding_box = calculate_rotated_bounding_box(
        self._atlas.reference.shape, full_matrix
    )
    new_translation = np.asarray(bounding_box) / 2
    post_rotate_translation = np.eye(4)
    post_rotate_translation[:3, -1] = new_translation

    self._atlas_transform_matrix = np.linalg.inv(
        post_rotate_translation @ full_matrix @ translate_matrix
    )

    self._atlas_data_layer.data = dask_affine_transform(
        self._atlas.reference,
        self._atlas_transform_matrix,
        order=2,
        output_shape=bounding_box,
        output_chunks=(2, bounding_box[1], bounding_box[2]),
    ).astype(self._atlas.reference.dtype)

    self._atlas_annotations_layer.data = dask_affine_transform(
        self._atlas.annotation,
        self._atlas_transform_matrix,
        order=0,
        output_shape=bounding_box,
        output_chunks=(2, bounding_box[1], bounding_box[2]),
    ).astype(self._atlas.annotation.dtype)

    self._viewer.reset_view()
    self._viewer.grid.enabled = False
    self._viewer.grid.enabled = True

    worker = compute_atlas_rotation(self)
    worker.returned.connect(self.set_atlas_layer_data)
    worker.start()


@thread_worker
def compute_atlas_rotation(self):
    self.adjust_moving_image_widget.reset_atlas_button.setEnabled(False)
    self.adjust_moving_image_widget.adjust_atlas_rotation.setEnabled(False)

    computed_array = self._atlas_data_layer.data.compute()

    self.adjust_moving_image_widget.reset_atlas_button.setEnabled(True)
    self.adjust_moving_image_widget.adjust_atlas_rotation.setEnabled(True)

    return computed_array
