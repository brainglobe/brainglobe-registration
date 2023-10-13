"""
A napari widget to view atlases.

Atlases that are exposed by the Brainglobe atlas API are
shown in a table view using the Qt model/view framework
[Qt Model/View framework](https://doc.qt.io/qt-6/model-view-programming.html)

Users can download and add the atlas images/structures as layers to the viewer.
"""

from typing import List

from bg_elastix.elastix.register import run_registration

import napari.layers
from bg_atlasapi import BrainGlobeAtlas
from bg_atlasapi.list_atlases import get_downloaded_atlases
from napari.viewer import Viewer
from qtpy.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QComboBox,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QSpinBox,
    QDoubleSpinBox,
    QLabel,
    QFormLayout,
    QPushButton,
    QGridLayout,
)

from bg_elastix.utils.brainglobe_logo import header_widget


class RegistrationWidget(QWidget):
    def __init__(self, napari_viewer: Viewer):
        super().__init__()

        # Hacky way of having an empty first option for the dropdown
        self._available_atlases = ["-----"] + get_downloaded_atlases()
        self._viewer = napari_viewer
        self._atlas: BrainGlobeAtlas = None

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(
            header_widget(tutorial_file_name="register-2D-image.html")
        )

        self.available_atlas_dropdown_label = QLabel("Select Atlas:")

        self.available_atlas_dropdown = QComboBox()
        self.available_atlas_dropdown.addItems(self._available_atlases)
        self.available_atlas_dropdown.currentIndexChanged.connect(self._on_atlas_dropdown_index_changed)

        self.available_sample_dropdown_label = QLabel("Select sample:")
        # TODO update the layer names dropdown when new images are opened in the viewer, can then use the index directly instead of looping through the layers
        self.available_sample_dropdown = QComboBox()
        self.curr_images = self.get_image_layer_names()
        self.available_sample_dropdown.addItems(self.curr_images)
        # TODO dynamically update self._moving_image to always store the napari layer containing the sample
        # self.available_sample_images.currentIndexChanged.connect(self._on_sample_dropdown_index_changed)
        self._moving_image = self._viewer.layers[0]

        self.get_atlas_widget = QGroupBox(parent=self)
        self.get_atlas_widget.setLayout(QVBoxLayout())
        self.get_atlas_widget.layout().addWidget(self.available_atlas_dropdown_label)
        self.get_atlas_widget.layout().addWidget(self.available_atlas_dropdown)
        self.get_atlas_widget.layout().addWidget(self.available_sample_dropdown_label)
        self.get_atlas_widget.layout().addWidget(self.available_sample_dropdown)

        self.adjust_moving_image_widget = QGroupBox("Adjust the sample image: ",parent=self)
        self.adjust_moving_image_widget.setLayout(QFormLayout())

        min_offset_range = -2000
        max_offset_range = 2000

        self.adjust_moving_image_x = QSpinBox()
        self.adjust_moving_image_x.setRange(min_offset_range, max_offset_range)

        self.adjust_moving_image_y = QSpinBox()
        self.adjust_moving_image_y.setRange(min_offset_range, max_offset_range)

        self.adjust_moving_image_rotate = QDoubleSpinBox()
        self.adjust_moving_image_rotate.setRange(-360,360)
        self.adjust_moving_image_rotate.setSingleStep(0.5)

        self.adjust_moving_image_buttons = QGroupBox(parent=self.adjust_moving_image_widget)
        self.adjust_moving_image_reset_button = QPushButton()
        self.adjust_moving_image_reset_button.setText("Reset Image")
        self.adjust_moving_image_reset_button.clicked.connect(self._on_adjust_moving_image_reset_button_click)

        self.adjust_moving_image_button = QPushButton(parent=self.adjust_moving_image_buttons)
        self.adjust_moving_image_button.setText("Adjust Image")
        self.adjust_moving_image_button.clicked.connect(self._on_adjust_moving_image_button_click)

        self.adjust_moving_image_buttons.setLayout(QHBoxLayout())
        self.adjust_moving_image_buttons.layout().addWidget(self.adjust_moving_image_button)
        self.adjust_moving_image_buttons.layout().addWidget(self.adjust_moving_image_reset_button)

        self.adjust_moving_image_widget.layout().addRow("X offset:", self.adjust_moving_image_x)
        self.adjust_moving_image_widget.layout().addRow("Y offset:", self.adjust_moving_image_y)
        self.adjust_moving_image_widget.layout().addRow("Rotation (degrees):", self.adjust_moving_image_rotate)
        self.adjust_moving_image_widget.layout().addRow(self.adjust_moving_image_buttons)

        self.registration_parameters_widget = QGroupBox(parent=self)
        self.registration_parameters_widget.setLayout(QFormLayout())

        self.rigid_checkbox = QCheckBox()
        self.affine_checkbox = QCheckBox()
        self.bspline_checkbox = QCheckBox()
        self.use_default_params_checkbox = QCheckBox()
        self.log_checkbox = QCheckBox()

        self.run_button = QPushButton()
        self.run_button.setText("Run Registration")
        self.run_button.clicked.connect(self._on_run_button_click)

        self.registration_parameters_widget.layout().addRow("Rigid", self.rigid_checkbox)
        self.registration_parameters_widget.layout().addRow("Affine", self.affine_checkbox)
        self.registration_parameters_widget.layout().addRow("B-Spline", self.bspline_checkbox)
        self.registration_parameters_widget.layout().addRow("Default Params", self.use_default_params_checkbox)
        self.registration_parameters_widget.layout().addRow("Log", self.log_checkbox)
        self.registration_parameters_widget.layout().addRow(self.run_button)

        self.layout().addWidget(self.get_atlas_widget)
        self.layout().addWidget(self.adjust_moving_image_widget)
        self.layout().addWidget(self.registration_parameters_widget)

    def _on_atlas_dropdown_index_changed(self, index):
        # Hacky way of having an empty first dropdown
        if index == 0:
            return

        atlas_name = self._available_atlases[index]
        atlas = BrainGlobeAtlas(atlas_name)

        if self._atlas:
            curr_atlas_layer_index = self.find_layer_index(self._atlas.atlas_name)

            self._viewer.layers.pop(curr_atlas_layer_index)

        self._viewer.add_image(atlas.reference, name=atlas_name, colormap="gray", blending="translucent")

        self._atlas = BrainGlobeAtlas(atlas_name=atlas_name)
        self._viewer.grid.enabled = True

    # def _on_sample_dropdown_index_changed(self, index):
    #     if index > 0:
    #         self._moving_image = self._viewer.layers[index-1]
    #         if (len(self.curr_images) - 1 != len(self._viewer.layers)):
    #             self.curr_images = ["-----"] + self.get_image_layer_names()
    #             self.available_sample_images.clear()
    #             self.available_sample_images.addItems(self.curr_images)

    def _on_adjust_moving_image_button_click(self):
        self.adjust_napari_image_layer(self._moving_image, self.adjust_moving_image_x.value(),
                                       self.adjust_moving_image_y.value(), self.adjust_moving_image_rotate.value())
        # layer_name = self.available_sample_images.currentText()
        # index = self.find_layer_index(layer_name=layer_name)
        #
        # if index >= 0:
        #     self.adjust_napari_image_layer(self._viewer.layers[index], self.adjust_moving_image_x.value(),
        #                                    self.adjust_moving_image_y.value(), self.adjust_moving_image_rotate.value())

    def _on_adjust_moving_image_reset_button_click(self):
        # layer_name = self.available_sample_images.currentText()
        # index = self.find_layer_index(layer_name=layer_name)

        self.adjust_moving_image_x.setValue(0)
        self.adjust_moving_image_y.setValue(0)
        self.adjust_moving_image_rotate.setValue(0)

        self.adjust_napari_image_layer(self._moving_image, 0,0,0)

        # if index >= 0:
        #     self.adjust_napari_image_layer(self._viewer.layers[index], 0, 0, 0)

    def _on_run_button_click(self):
        current_atlas_slice = self._viewer.dims.current_step[0]

        result, parameters = run_registration(
            self._atlas.reference[current_atlas_slice,:,:],
            self._moving_image.data,
            self.rigid_checkbox.isChecked(),
            self.affine_checkbox.isChecked(),
            self.bspline_checkbox.isChecked(),
            self.use_default_params_checkbox.isChecked()
        )

        self._viewer.add_image(result, name="Registered Image")
        self._viewer.add_labels(self._atlas.annotation[current_atlas_slice,:,:], name="Registered Annotations", visible=False)

    def find_layer_index(self, layer_name: str) -> int:
        curr_layers = self._viewer.layers

        for idx, layer in enumerate(curr_layers):
            if layer.name == layer_name:
                return idx

        return -1

    def get_image_layer_names(self) -> List[str]:
        return [layer.name for layer in self._viewer.layers]

    def adjust_napari_image_layer(self, image_layer: napari.layers.Image, x: int, y: int, rotate: float):
        image_layer.translate = (y, x)
        image_layer.rotate = rotate
