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

from brainglobe_registration.elastix.register import run_registration
from brainglobe_registration.widgets.select_images_view import SelectImagesView
from brainglobe_registration.widgets.adjust_moving_image_view import (
    AdjustMovingImageView,
)
from brainglobe_registration.widgets.parameter_list_view import (
    RegistrationParameterListView,
)
from brainglobe_registration.widgets.transform_select_view import (
    TransformSelectView,
)

import napari.layers
from pytransform3d.rotations import active_matrix_from_angle
from bg_atlasapi import BrainGlobeAtlas
from bg_atlasapi.list_atlases import get_downloaded_atlases
from napari.viewer import Viewer
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QGroupBox,
    QVBoxLayout,
    QWidget,
    QTabWidget,
    QPushButton,
)

from brainglobe_registration.utils.brainglobe_logo import header_widget


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
    Opens the parameter file and returns the parameter dictionary.
    """
    with open(file_path, "r") as f:
        param_dict = {}
        for line in f.readlines():
            if line[0] == "(":
                split_line = line[1:-1].split()
                cleaned_params = []
                for i, entry in enumerate(split_line[1:]):
                    if entry != ")":
                        cleaned_params.append(entry.strip('" )'))

                param_dict[split_line[0]] = cleaned_params

    return param_dict


class RegistrationWidget(QWidget):
    def __init__(self, napari_viewer: Viewer):
        super().__init__()

        self._viewer = napari_viewer
        self._atlas: BrainGlobeAtlas = None

        self.transform_params = {"rigid": {}, "affine": {}, "bspline": {}}
        self.transform_selections = []

        for transform_type in self.transform_params:
            file_path = (
                Path()
                / "parameters"
                / "elastix_default"
                / f"{transform_type}.txt"
            )

            if file_path.exists():
                self.transform_params[transform_type] = open_parameter_file(
                    file_path
                )
                self.transform_selections.append(
                    (transform_type, self.transform_params[transform_type])
                )

        # Hacky way of having an empty first option for the dropdown
        self._available_atlases = ["------"] + get_downloaded_atlases()
        self._sample_images = self.get_image_layer_names()
        self._moving_image = self._viewer.layers[0]

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(
            header_widget(tutorial_file_name="register-2D-image.html")
        )

        self.main_tabs = QTabWidget(parent=self)
        self.main_tabs.setTabPosition(QTabWidget.West)
        # self.main_tabs.setStyleSheet(
        # "QTabBar::tab { height: 200px; width: 30px; font-size: 20px;}"
        # )

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

        self.transform_select_view = TransformSelectView()
        self.transform_select_view.transform_type_added_signal.connect(
            self._on_transform_type_added
        )
        self.transform_select_view.transform_type_removed_signal.connect(
            self._on_transform_type_removed
        )
        self.transform_select_view.file_signal.connect(
            self._on_default_file_selection_change
        )

        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self._on_run_button_click)
        self.run_button.setEnabled(False)
        self.test_button = QPushButton("Test")
        self.test_button.clicked.connect(self._on_test_button_click)

        self.settings_tab.layout().addWidget(self.get_atlas_widget)
        self.settings_tab.layout().addWidget(self.adjust_moving_image_widget)
        self.settings_tab.layout().addWidget(self.transform_select_view)
        self.settings_tab.layout().addWidget(self.run_button)
        self.settings_tab.layout().addWidget(self.test_button)
        self.settings_tab.layout().setAlignment(Qt.AlignTop)

        self.parameter_setting_tabs_lists = []

        for transform_type in self.transform_params:
            new_tab = RegistrationParameterListView(
                param_dict=self.transform_params[transform_type],
                transform_type=transform_type,
            )

            self.parameters_tab.addTab(new_tab, transform_type)
            self.parameter_setting_tabs_lists.append(new_tab)

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
        else:
            self.run_button.setEnabled(True)

        self._viewer.add_image(
            atlas.reference,
            name=atlas_name,
            colormap="gray",
            blending="translucent",
        )

        self._atlas = BrainGlobeAtlas(atlas_name=atlas_name)
        self._viewer.grid.enabled = True

    def _on_sample_dropdown_index_changed(self, index):
        viewer_index = self.find_layer_index(self._sample_images[index])
        self._moving_image = self._viewer.layers[viewer_index]
        # if (len(self.curr_images) - 1 != len(self._viewer.layers)):
        #     self.curr_images = ["-----"] + self.get_image_layer_names()
        #     self.available_sample_images.clear()
        #     self.available_sample_images.addItems(self.curr_images)

    def _on_adjust_moving_image_button_click(
        self, x: int, y: int, rotate: float
    ):
        adjust_napari_image_layer(self._moving_image, x, y, rotate)

    def _on_adjust_moving_image_reset_button_click(self):
        adjust_napari_image_layer(self._moving_image, 0, 0, 0)

    def _on_run_button_click(self):
        current_atlas_slice = self._viewer.dims.current_step[0]

        result, parameters = run_registration(
            self._atlas.reference[current_atlas_slice, :, :],
            self._moving_image.data,
            self.transform_selections,
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
        """
        Returns a list of the names of the napari image layers in the viewer.
        """
        return [layer.name for layer in self._viewer.layers]

    def _on_test_button_click(self):
        for transform in self.transform_selections:
            print(transform[0])
            print(transform[1])
        print()

    def _on_transform_type_added(
        self, transform_type: str, transform_order: int
    ) -> None:
        if transform_order > len(self.transform_selections):
            raise IndexError(
                f"Transform added out of order index: {transform_order}"
                f" is greater than length: {len(self.transform_selections)}"
            )
        elif len(self.parameter_setting_tabs_lists) == transform_order:
            self.transform_selections.append(
                (transform_type, self.transform_params[transform_type].copy())
            )
            new_tab = RegistrationParameterListView(
                param_dict=self.transform_selections[transform_order][1],
                transform_type=transform_type,
            )
            self.parameters_tab.addTab(new_tab, transform_type)
            self.parameter_setting_tabs_lists.append(new_tab)

        else:
            self.transform_selections[transform_order] = (
                transform_type,
                self.transform_params[transform_type],
            )
            self.parameters_tab.setTabText(transform_order, transform_type)
            self.parameter_setting_tabs_lists[transform_order].set_data(
                self.transform_params[transform_type].copy()
            )

    def _on_transform_type_removed(self, transform_order: int) -> None:
        if transform_order >= len(self.transform_selections):
            raise IndexError("Transform removed out of order")
        else:
            self.transform_selections.pop(transform_order)
            self.parameters_tab.removeTab(transform_order)
            self.parameter_setting_tabs_lists.pop(transform_order)

    def _on_default_file_selection_change(
        self, default_file_type: str, index: int
    ) -> None:
        if index >= len(self.transform_selections):
            raise IndexError("Transform file selection out of order")

        transform_type = self.transform_selections[index][0]
        file_path = (
            Path() / "parameters" / default_file_type / f"{transform_type}.txt"
        )

        if not file_path.exists():
            file_path = (
                Path()
                / "parameters"
                / "elastix_default"
                / f"{transform_type}.txt"
            )

        param_dict = open_parameter_file(file_path)

        self.transform_selections[index] = (transform_type, param_dict)
        self.parameter_setting_tabs_lists[index].set_data(param_dict)
