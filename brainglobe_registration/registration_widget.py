"""
A napari widget to view atlases.

Atlases that are exposed by the Brainglobe atlas API are
shown in a table view using the Qt model/view framework
[Qt Model/View framework](https://doc.qt.io/qt-6/model-view-programming.html)

Users can download and add the atlas images/structures as layers to the viewer.
"""

from pathlib import Path

import numpy as np
from bg_atlasapi import BrainGlobeAtlas
from bg_atlasapi.list_atlases import get_downloaded_atlases
from brainglobe_utils.qtpy.logo import header_widget
from napari.viewer import Viewer
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QGroupBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from skimage.segmentation import find_boundaries

from brainglobe_registration.elastix.register import run_registration
from brainglobe_registration.utils.utils import (
    adjust_napari_image_layer,
    find_layer_index,
    get_image_layer_names,
    open_parameter_file,
)
from brainglobe_registration.widgets.adjust_moving_image_view import (
    AdjustMovingImageView,
)
from brainglobe_registration.widgets.parameter_list_view import (
    RegistrationParameterListView,
)
from brainglobe_registration.widgets.select_images_view import SelectImagesView
from brainglobe_registration.widgets.transform_select_view import (
    TransformSelectView,
)


class RegistrationWidget(QWidget):
    def __init__(self, napari_viewer: Viewer):
        super().__init__()

        self._viewer = napari_viewer
        self._atlas: BrainGlobeAtlas = None
        self._moving_image = None

        self.transform_params: dict[str, dict] = {
            "rigid": {},
            "affine": {},
            "bspline": {},
        }
        self.transform_selections = []

        for transform_type in self.transform_params:
            file_path = (
                Path(__file__).parent.resolve()
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
        self._sample_images = get_image_layer_names(self._viewer)

        if len(self._sample_images) > 0:
            self._moving_image = self._viewer.layers[0]
        else:
            self._moving_image = None

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(
            header_widget(
                "brainglobe-registration",
                "Registration to a BrainGlobe atlas using Elastix",
            )
        )

        self.main_tabs = QTabWidget(parent=self)
        self.main_tabs.setTabPosition(QTabWidget.West)

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
        self.get_atlas_widget.sample_image_popup_about_to_show.connect(
            self._on_sample_popup_about_to_show
        )

        self.adjust_moving_image_widget = AdjustMovingImageView(parent=self)
        self.adjust_moving_image_widget.adjust_image_signal.connect(
            self._on_adjust_moving_image
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
        self.transform_select_view.file_option_changed_signal.connect(
            self._on_default_file_selection_change
        )

        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self._on_run_button_click)
        self.run_button.setEnabled(False)

        self.settings_tab.layout().addWidget(self.get_atlas_widget)
        self.settings_tab.layout().addWidget(self.adjust_moving_image_widget)
        self.settings_tab.layout().addWidget(self.transform_select_view)
        self.settings_tab.layout().addWidget(self.run_button)
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
            if self._atlas:
                curr_atlas_layer_index = find_layer_index(
                    self._viewer, self._atlas.atlas_name
                )

                self._viewer.layers.pop(curr_atlas_layer_index)
                self._atlas = None
                self.run_button.setEnabled(False)
                self._viewer.grid.enabled = False

            return

        atlas_name = self._available_atlases[index]
        atlas = BrainGlobeAtlas(atlas_name)

        if self._atlas:
            curr_atlas_layer_index = find_layer_index(
                self._viewer, self._atlas.atlas_name
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
        viewer_index = find_layer_index(
            self._viewer, self._sample_images[index]
        )
        self._moving_image = self._viewer.layers[viewer_index]

    def _on_adjust_moving_image(self, x: int, y: int, rotate: float):
        adjust_napari_image_layer(self._moving_image, x, y, rotate)

    def _on_adjust_moving_image_reset_button_click(self):
        adjust_napari_image_layer(self._moving_image, 0, 0, 0)

    def _on_run_button_click(self):
        current_atlas_slice = self._viewer.dims.current_step[0]

        result, parameters, registered_annotation_image = run_registration(
            self._atlas.reference[current_atlas_slice, :, :],
            self._moving_image.data,
            self._atlas.annotation[current_atlas_slice, :, :],
            self.transform_selections,
        )

        boundaries = find_boundaries(
            registered_annotation_image, mode="inner"
        ).astype(np.int8, copy=False)

        self._viewer.add_image(result, name="Registered Image", visible=False)

        atlas_layer_index = find_layer_index(
            self._viewer, self._atlas.atlas_name
        )
        self._viewer.layers[atlas_layer_index].visible = False

        self._viewer.add_labels(
            registered_annotation_image.astype(np.uint32, copy=False),
            name="Registered Annotations",
            visible=False,
        )
        self._viewer.add_image(
            boundaries,
            name="Registered Boundaries",
            visible=True,
            blending="additive",
            opacity=0.8,
        )

        self._viewer.grid.enabled = False

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
            Path(__file__).parent.resolve()
            / "parameters"
            / default_file_type
            / f"{transform_type}.txt"
        )

        if not file_path.exists():
            file_path = (
                Path(__file__).parent.resolve()
                / "parameters"
                / "elastix_default"
                / f"{transform_type}.txt"
            )

        param_dict = open_parameter_file(file_path)

        self.transform_selections[index] = (transform_type, param_dict)
        self.parameter_setting_tabs_lists[index].set_data(param_dict)

    def _on_sample_popup_about_to_show(self):
        self._sample_images = get_image_layer_names(self._viewer)
        self.get_atlas_widget.update_sample_image_names(self._sample_images)
