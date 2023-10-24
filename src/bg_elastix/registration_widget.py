"""
A napari widget to view atlases.

Atlases that are exposed by the Brainglobe atlas API are
shown in a table view using the Qt model/view framework
[Qt Model/View framework](https://doc.qt.io/qt-6/model-view-programming.html)

Users can download and add the atlas images/structures as layers to the viewer.
"""

from typing import List

import numpy as np

from bg_elastix.elastix.register import run_registration
from bg_elastix.widgets.select_images_view import SelectImagesView
from bg_elastix.widgets.adjust_moving_image_view import AdjustMovingImageView
from bg_elastix.widgets.registration_parameters_view import RegistrationParametersView

import napari.layers
from pytransform3d.rotations import active_matrix_from_angle
from bg_atlasapi import BrainGlobeAtlas
from bg_atlasapi.list_atlases import get_downloaded_atlases
from napari.viewer import Viewer
from qtpy.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QVBoxLayout,
    QWidget,
    QGridLayout,
    QPushButton,
)

from bg_elastix.utils.brainglobe_logo import header_widget


def adjust_napari_image_layer(
    image_layer: napari.layers.Layer, x: int, y: int, rotate: float
):
    """Adjusts the napari image layer by the given x, y, and rotation values.

    Rotation around origin code adapted from:
        https://forum.image.sc/t/napari-3d-rotation-center-change-and-scaling/66347/5
    """
    image_layer.translate = (y, x)

    rotation_matrix = active_matrix_from_angle(2, np.deg2rad(rotate))
    translate_matrix = np.eye(3)
    origin = np.asarray(image_layer.data.shape) // 2 + np.asarray([y, x])
    translate_matrix[:2, -1] = origin
    transform_matrix = (
        translate_matrix @ rotation_matrix @ np.linalg.inv(translate_matrix)
    )
    image_layer.affine = transform_matrix


class RegistrationWidget(QWidget):
    def __init__(self, napari_viewer: Viewer):
        super().__init__()

        self._viewer = napari_viewer
        self._atlas: BrainGlobeAtlas = None

        # Hacky way of having an empty first option for the dropdown
        self._available_atlases = ["------"] + get_downloaded_atlases()
        self._sample_images = ["------"] + self.get_image_layer_names()
        self._moving_image = None

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(
            header_widget(tutorial_file_name="register-2D-image.html")
        )

        self.get_atlas_widget = SelectImagesView(
            available_atlases=self._available_atlases,
            sample_image_names=self._sample_images,
            parent=self,
        )
        self.get_atlas_widget.atlas_index_change.connect(
            self._on_atlas_dropdown_index_changed
        )
        self.get_atlas_widget.moving_image_index_change.connect(
            self._on_sample_dropdown_index_changed
        )

        self.adjust_moving_image_widget = AdjustMovingImageView(parent=self)
        self.adjust_moving_image_widget.adjust_image_signal.connect(
            self._on_adjust_moving_image_button_click
        )

        self.adjust_moving_image_widget.reset_image_signal.connect(
            self._on_adjust_moving_image_reset_button_click
        )

        self.test_widget = RegistrationParametersView()

        self.registration_parameters_widget = QGroupBox()
        self.registration_parameters_widget.setLayout(QVBoxLayout())

        self.rigid_checkbox = QCheckBox("Rigid")
        self.rigid_checkbox.setChecked(True)

        self.affine_checkbox = QCheckBox("Affine")
        self.affine_checkbox.setChecked(True)

        self.bspline_checkbox = QCheckBox("B-Spline")
        self.bspline_checkbox.setChecked(True)

        self.use_default_params_checkbox = QCheckBox("Default Params")
        self.log_checkbox = QCheckBox("Log")

        self.run_button = QPushButton()
        self.run_button.setText("Run Registration")
        self.run_button.clicked.connect(self._on_run_button_click)

        self.registration_parameters_widget.layout().addWidget(
            self.rigid_checkbox
        )
        self.registration_parameters_widget.layout().addWidget(
            self.affine_checkbox
        )
        self.registration_parameters_widget.layout().addWidget(
            self.bspline_checkbox
        )
        self.registration_parameters_widget.layout().addWidget(
            self.use_default_params_checkbox
        )
        self.registration_parameters_widget.layout().addWidget(
            self.log_checkbox
        )
        self.registration_parameters_widget.layout().addWidget(self.run_button)

        self.layout().addWidget(self.get_atlas_widget)
        self.layout().addWidget(self.adjust_moving_image_widget)
        self.layout().addWidget(self.registration_parameters_widget)
        self.layout().addWidget(self.test_widget)

    def _on_atlas_dropdown_index_changed(self, index):
        # Hacky way of having an empty first dropdown
        if index == 0:
            return

        atlas_name = self._available_atlases[index]
        atlas = BrainGlobeAtlas(atlas_name)

        if self._atlas:
            curr_atlas_layer_index = self.find_layer_index(
                self._atlas.atlas_name
            )

            self._viewer.layers.pop(curr_atlas_layer_index)

        self._viewer.add_image(
            atlas.reference,
            name=atlas_name,
            colormap="gray",
            blending="translucent",
        )

        self._atlas = BrainGlobeAtlas(atlas_name=atlas_name)
        self._viewer.grid.enabled = True

    def _on_sample_dropdown_index_changed(self, index):
        if index > 0:
            viewer_index = self.find_layer_index(self._sample_images[index])
            self._moving_image = self._viewer.layers[viewer_index]
            # if (len(self.curr_images) - 1 != len(self._viewer.layers)):
            #     self.curr_images = ["-----"] + self.get_image_layer_names()
            #     self.available_sample_images.clear()
            #     self.available_sample_images.addItems(self.curr_images)

    def _on_adjust_moving_image_button_click(self, x: int, y: int, rotate: float):
        adjust_napari_image_layer(self._moving_image, x, y, rotate)

    def _on_adjust_moving_image_reset_button_click(self):
        adjust_napari_image_layer(self._moving_image, 0, 0, 0)

    def _on_run_button_click(self):
        current_atlas_slice = self._viewer.dims.current_step[0]

        result, parameters = run_registration(
            self._atlas.reference[current_atlas_slice, :, :],
            self._moving_image.data,
            self.rigid_checkbox.isChecked(),
            self.affine_checkbox.isChecked(),
            self.bspline_checkbox.isChecked(),
            self.use_default_params_checkbox.isChecked(),
        )

        self._viewer.add_image(result, name="Registered Image")
        self._viewer.add_labels(
            self._atlas.annotation[current_atlas_slice, :, :],
            name="Registered Annotations",
            visible=False,
        )

    def find_layer_index(self, layer_name: str) -> int:
        """Finds the index of a layer in the napari viewer."""
        curr_layers = self._viewer.layers

        for idx, layer in enumerate(curr_layers):
            if layer.name == layer_name:
                return idx

        return -1

    def get_image_layer_names(self) -> List[str]:
        """Returns a list of the names of the napari image layers currently in the layer."""
        return [layer.name for layer in self._viewer.layers]

