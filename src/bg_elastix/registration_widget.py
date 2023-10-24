"""
A napari widget to view atlases.

Atlases that are exposed by the Brainglobe atlas API are
shown in a table view using the Qt model/view framework
[Qt Model/View framework](https://doc.qt.io/qt-6/model-view-programming.html)

Users can download and add the atlas images/structures as layers to the viewer.
"""

from typing import List

import numpy as np
from pathlib import Path

from bg_elastix.elastix.register import run_registration
from bg_elastix.widgets.select_images_view import SelectImagesView
from bg_elastix.widgets.adjust_moving_image_view import AdjustMovingImageView
from bg_elastix.widgets.run_settings_select_view import RunSettingsSelectView
from bg_elastix.widgets.parameter_list_view import RegistrationParameterListView

import napari.layers
from pytransform3d.rotations import active_matrix_from_angle
from bg_atlasapi import BrainGlobeAtlas
from bg_atlasapi.list_atlases import get_downloaded_atlases
from napari.viewer import Viewer
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QHBoxLayout,
    QBoxLayout,
    QGroupBox,
    QVBoxLayout,
    QWidget,
    QTabWidget,
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


def open_parameter_file(file_path: Path) -> dict:
    """
    Opens the parameter file and returns the transform type and the parameter dictionary.
    """
    with open(file_path, "r") as f:
        param_dict = {}
        for line in f.readlines():
            if line[0] == "(":
                split_line = line[1:-1].split()
                param_dict[split_line[0]] = split_line[1].strip("\" )")

    return param_dict


class RegistrationWidget(QWidget):
    def __init__(self, napari_viewer: Viewer):
        super().__init__()

        self._viewer = napari_viewer
        self._atlas: BrainGlobeAtlas = None

        self.transform_params = {"rigid": {}, "affine": {}, "bspline": {}}

        for transform_type in self.transform_params:
            file_path = Path() / "parameters" / "elastix_default" / f"{transform_type}.txt"

            if file_path.exists():
                self.transform_params[transform_type] = open_parameter_file(file_path)

        # Hacky way of having an empty first option for the dropdown
        self._available_atlases = ["------"] + get_downloaded_atlases()
        self._sample_images = ["------"] + self.get_image_layer_names()
        self._moving_image = None

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(
            header_widget(tutorial_file_name="register-2D-image.html")
        )

        self.main_tabs = QTabWidget(parent=self)
        self.main_tabs.setTabPosition(QTabWidget.West)
        # self.main_tabs.setStyleSheet("QTabBar::tab { height: 200px; width: 30px; font-size: 20px;}")

        self.settings_tab = QGroupBox()
        self.settings_tab.setLayout(QVBoxLayout())
        self.parameters_tab = QTabWidget()

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

        self.run_settings_widget = RunSettingsSelectView()

        self.run_settings_widget.run_signal.connect(
            self._on_run_button_click
        )

        self.run_settings_widget.rigid_checkbox_signal.connect(
            self._on_rigid_checkbox_change
        )

        self.run_settings_widget.affine_checkbox_signal.connect(
            self._on_affine_checkbox_change
        )

        self.run_settings_widget.bspline_checkbox_signal.connect(
            self._on_bspline_checkbox_change
        )

        self.run_settings_widget.default_file_signal.connect(
            self._on_default_file_change
        )

        self.settings_tab.layout().addWidget(self.get_atlas_widget)
        self.settings_tab.layout().addWidget(self.adjust_moving_image_widget)
        self.settings_tab.layout().addWidget(self.run_settings_widget)
        self.settings_tab.layout().setAlignment(Qt.AlignTop)
        self.parameter_list_tabs = {}

        for transform_type in self.transform_params:
            new_tab = RegistrationParameterListView(
                param_dict=self.transform_params[transform_type],
                transform_type=transform_type
            )

            self.parameters_tab.addTab(new_tab, transform_type)
            self.parameter_list_tabs[transform_type] = new_tab

        self.main_tabs.addTab(self.settings_tab, "Settings")
        self.main_tabs.addTab(self.parameters_tab, "Parameters")

        self.layout().addWidget(self.main_tabs)

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

    def _on_run_button_click(
            self, rigid: bool, affine: bool, bspline: bool,
            use_default_params: bool, default_params_file: str):
        current_atlas_slice = self._viewer.dims.current_step[0]

        # TODO Pass back default file to back end
        result, parameters = run_registration(
            self._atlas.reference[current_atlas_slice, :, :],
            self._moving_image.data,
            rigid,
            affine,
            bspline,
            use_default_params,
        )

        self._viewer.add_image(result, name="Registered Image")
        self._viewer.add_labels(
            self._atlas.annotation[current_atlas_slice, :, :],
            name="Registered Annotations",
            visible=False,
        )

    def _on_default_file_change(self, directory: str):
        for transform_type in self.transform_params:
            file_path = Path() / "parameters" / directory / f"{transform_type}.txt"

            if file_path.exists():
                self.transform_params[transform_type] = open_parameter_file(file_path)
                # Signal to the parameter list view to update the parameters
                self.parameter_list_tabs[transform_type].set_data(self.transform_params[transform_type])


    def _on_rigid_checkbox_change(self, state):
        self.parameters_tab.setTabEnabled(0, state)

    def _on_affine_checkbox_change(self, state):
        self.parameters_tab.setTabEnabled(1, state)

    def _on_bspline_checkbox_change(self, state):
        self.parameters_tab.setTabEnabled(2, state)

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

