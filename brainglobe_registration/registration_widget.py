"""
A napari widget to view atlases.

Atlases that are exposed by the Brainglobe atlas API are
shown in a table view using the Qt model/view framework
[Qt Model/View framework](https://doc.qt.io/qt-6/model-view-programming.html)

Users can download and add the atlas images/structures as layers to the viewer.
"""

from pathlib import Path
from typing import Optional

import dask.array as da
import napari.layers
import numpy as np
import numpy.typing as npt
from brainglobe_atlasapi import BrainGlobeAtlas
from brainglobe_atlasapi.list_atlases import get_downloaded_atlases
from brainglobe_utils.qtpy.collapsible_widget import CollapsibleWidgetContainer
from brainglobe_utils.qtpy.dialog import display_info
from brainglobe_utils.qtpy.logo import header_widget
from dask_image.ndinterp import affine_transform as dask_affine_transform
from napari.qt.threading import thread_worker
from napari.utils.events import Event
from napari.utils.notifications import show_error
from napari.viewer import Viewer
from pytransform3d.rotations import active_matrix_from_angle
from qtpy.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTabWidget,
    QWidget,
)
from skimage.segmentation import find_boundaries
from skimage.transform import rescale
from tifffile import imwrite

from brainglobe_registration.utils.utils import (
    adjust_napari_image_layer,
    calculate_areas,
    calculate_rotated_bounding_box,
    check_atlas_installed,
    filter_plane,
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


class RegistrationWidget(CollapsibleWidgetContainer):
    def __init__(self, napari_viewer: Viewer):
        super().__init__()
        self.setContentsMargins(10, 10, 10, 10)

        self._viewer = napari_viewer
        self._atlas: Optional[BrainGlobeAtlas] = None
        self._atlas_data_layer: Optional[napari.layers.Image] = None
        self._atlas_annotations_layer: Optional[napari.layers.Labels] = None
        self._moving_image: Optional[napari.layers.Image] = None
        self._moving_image_data_backup: Optional[npt.NDArray] = None
        # Flag to differentiate between manual and automatic atlas deletion
        self._automatic_deletion_flag = False

        self.transform_params: dict[str, dict] = {
            "affine": {},
            "bspline": {},
        }
        self.transform_selections = []

        for transform_type in self.transform_params:
            file_path = (
                Path(__file__).parent.resolve()
                / "parameters"
                / "ara_tools"
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

        self.output_directory: Optional[Path] = None

        if len(self._sample_images) > 0:
            self._moving_image = self._viewer.layers[0]
        else:
            self._moving_image = None

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
        self.adjust_moving_image_widget.scale_image_signal.connect(
            self._on_scale_moving_image
        )
        self.adjust_moving_image_widget.atlas_rotation_signal.connect(
            self._on_adjust_atlas_rotation
        )
        self.adjust_moving_image_widget.reset_atlas_signal.connect(
            self._on_atlas_reset
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

        # Use decorator to connect to layer deletion event
        self._connect_events()

        self.filter_checkbox = QCheckBox("Filter Images")
        self.filter_checkbox.setChecked(False)

        self.output_directory_widget = QWidget()
        self.output_directory_widget.setLayout(QHBoxLayout())

        self.output_directory_text_field = QLineEdit()
        self.output_directory_text_field.editingFinished.connect(
            self._on_output_directory_text_edited
        )
        self.output_directory_widget.layout().addWidget(
            self.output_directory_text_field
        )

        self.open_file_dialog = QPushButton("Browse")
        self.open_file_dialog.clicked.connect(
            self._on_open_file_dialog_clicked
        )
        self.output_directory_widget.layout().addWidget(self.open_file_dialog)

        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self._on_run_button_click)
        self.run_button.setEnabled(False)

        self.add_widget(
            header_widget(
                "brainglobe-<br>registration",  # line break at <br>
                "Registration with Elastix",
                github_repo_name="brainglobe-registration",
            ),
            collapsible=False,
        )
        self.add_widget(self.get_atlas_widget, widget_title="Select Images")
        self.add_widget(
            self.adjust_moving_image_widget, widget_title="Prepare Images"
        )
        self.add_widget(
            self.transform_select_view, widget_title="Select Transformations"
        )

        self.parameter_setting_tabs_lists = []
        self.parameters_tab = QTabWidget(parent=self)

        for transform_type in self.transform_params:
            new_tab = RegistrationParameterListView(
                param_dict=self.transform_params[transform_type],
                transform_type=transform_type,
            )

            self.parameters_tab.addTab(new_tab, transform_type)
            self.parameter_setting_tabs_lists.append(new_tab)

        self.add_widget(
            self.parameters_tab, widget_title="Advanced Settings (optional)"
        )

        self.add_widget(self.filter_checkbox, collapsible=False)

        self.add_widget(QLabel("Output Directory"), collapsible=False)
        self.add_widget(self.output_directory_widget, collapsible=False)

        self.add_widget(self.run_button, collapsible=False)

        self.layout().itemAt(1).widget().collapse(animate=False)

        check_atlas_installed(self)

    def _connect_events(self):
        @self._viewer.layers.events.removed.connect
        def _on_layer_deleted(event: Event):
            if not self._automatic_deletion_flag:
                self._handle_layer_deletion(event)

    def _handle_layer_deletion(self, event: Event):
        deleted_layer = event.value

        # Check if the deleted layer is the moving image
        if self._moving_image == deleted_layer:
            self._moving_image = None
            self._moving_image_data_backup = None
            self._update_dropdowns()

        # Check if deleted layer is the atlas reference / atlas annotations
        if (
            self._atlas_data_layer == deleted_layer
            or self._atlas_annotations_layer == deleted_layer
        ):
            # Reset the atlas selection combobox
            self.get_atlas_widget.reset_atlas_combobox()

    def _delete_atlas_layers(self):
        # Delete atlas reference layer if it exists
        if self._atlas_data_layer in self._viewer.layers:
            self._viewer.layers.remove(self._atlas_data_layer)

        # Delete atlas annotations layer if it exists
        if self._atlas_annotations_layer in self._viewer.layers:
            self._viewer.layers.remove(self._atlas_annotations_layer)

        # Clear atlas attributes
        self._atlas = None
        self._atlas_data_layer = None
        self._atlas_annotations_layer = None
        self.run_button.setEnabled(False)
        self._viewer.grid.enabled = False

    def _update_dropdowns(self):
        # Extract the names of the remaining layers
        layer_names = get_image_layer_names(self._viewer)
        # Update the dropdowns in SelectImagesView
        self.get_atlas_widget.update_sample_image_names(layer_names)

    def _on_atlas_dropdown_index_changed(self, index):
        # Hacky way of having an empty first dropdown
        if index == 0:
            self._delete_atlas_layers()
            return

        atlas_name = self._available_atlases[index]

        if self._atlas:
            # Set flag to True when atlas is changed, not manually deleted
            # Ensures atlas index does not reset to 0
            self._automatic_deletion_flag = True
            try:
                self._delete_atlas_layers()
            finally:
                self._automatic_deletion_flag = False

        self.run_button.setEnabled(True)

        self._atlas = BrainGlobeAtlas(atlas_name)
        dask_reference = da.from_array(
            self._atlas.reference,
            chunks=(
                1,
                self._atlas.reference.shape[1],
                self._atlas.reference.shape[2],
            ),
        )
        dask_annotations = da.from_array(
            self._atlas.annotation,
            chunks=(
                1,
                self._atlas.annotation.shape[1],
                self._atlas.annotation.shape[2],
            ),
        )

        contrast_max = np.max(
            dask_reference[dask_reference.shape[0] // 2]
        ).compute()
        self._atlas_data_layer = self._viewer.add_image(
            dask_reference,
            name=atlas_name,
            colormap="gray",
            blending="translucent",
            contrast_limits=[0, contrast_max],
            multiscale=False,
        )
        self._atlas_annotations_layer = self._viewer.add_labels(
            dask_annotations,
            name="Annotations",
            visible=False,
        )

        self._viewer.grid.enabled = True

    def _on_sample_dropdown_index_changed(self, index):
        viewer_index = find_layer_index(
            self._viewer, self._sample_images[index]
        )
        if viewer_index == -1:
            self._moving_image = None
            self._moving_image_data_backup = None

            return

        self._moving_image = self._viewer.layers[viewer_index]
        self._moving_image_data_backup = self._moving_image.data.copy()

    def _on_adjust_moving_image(self, x: int, y: int, rotate: float):
        if not self._moving_image:
            show_error(
                "No moving image selected."
                "Please select a moving image before adjusting"
            )
            return
        adjust_napari_image_layer(self._moving_image, x, y, rotate)

    def _on_adjust_moving_image_reset_button_click(self):
        if not self._moving_image:
            show_error(
                "No moving image selected. "
                "Please select a moving image before adjusting"
            )
            return
        adjust_napari_image_layer(self._moving_image, 0, 0, 0)

    def _on_output_directory_text_edited(self):
        self.output_directory = Path(self.output_directory_text_field.text())

    def _on_open_file_dialog_clicked(self) -> None:
        """
        Open a file dialog to select the output directory.
        """
        output_directory_str = QFileDialog.getExistingDirectory(
            self, "Select the output directory", str(Path.home())
        )
        # A blank string is returned if the user cancels the dialog
        if not output_directory_str:
            return

        self.output_directory = Path(output_directory_str)
        self.output_directory_text_field.setText(str(self.output_directory))

    def _on_run_button_click(self):
        from brainglobe_registration.elastix.register import (
            calculate_deformation_field,
            invert_transformation,
            run_registration,
            transform_annotation_image,
            transform_image,
        )

        if self._atlas_data_layer is None:
            display_info(
                widget=self,
                title="Warning",
                message="Please select an atlas before clicking 'Run'.",
            )
            return

        if self._moving_image == self._atlas_data_layer:
            display_info(
                widget=self,
                title="Warning",
                message="Your moving image cannot be an atlas.",
            )
            return

        current_atlas_slice = self._viewer.dims.current_step[0]

        if self.filter_checkbox.isChecked():
            atlas_layer = filter_plane(
                self._atlas_data_layer.data[
                    current_atlas_slice, :, :
                ].compute()
            )
            moving_image = filter_plane(self._moving_image.data)
        else:
            atlas_layer = self._atlas_data_layer.data[
                current_atlas_slice, :, :
            ]
            moving_image = self._moving_image.data

        result, parameters = run_registration(
            atlas_layer,
            moving_image,
            self.transform_selections,
            self.output_directory,
        )

        inverse_result, inverse_parameters = invert_transformation(
            atlas_layer,
            self.transform_selections,
            parameters,
            self.output_directory,
        )

        data_in_atlas_space = transform_image(moving_image, inverse_parameters)

        # Creating fresh array is necessary to avoid a crash on Windows
        # Otherwise self._atlas_annotations_layer.data.dtype.type returns
        # np.uintc which is not supported by ITK. After creating a new array
        # annotations.dtype.type returns np.uint32 which is supported by ITK.
        annotation = np.array(
            self._atlas_annotations_layer.data[current_atlas_slice, :, :],
            dtype=np.uint32,
        )

        registered_annotation_image = transform_annotation_image(
            annotation,
            parameters,
        )

        registered_hemisphere = transform_annotation_image(
            self._atlas.hemispheres[current_atlas_slice, :, :], parameters
        )

        boundaries = find_boundaries(
            registered_annotation_image, mode="inner"
        ).astype(np.int8, copy=False)

        deformation_field = calculate_deformation_field(
            moving_image, parameters
        )

        self._viewer.add_image(result, name="Registered Image", visible=False)

        atlas_layer_index = find_layer_index(
            self._viewer, self._atlas.atlas_name
        )
        self._viewer.layers[atlas_layer_index].visible = False

        self._viewer.add_labels(
            registered_annotation_image,
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
        self._viewer.add_image(
            data_in_atlas_space,
            name="Inverse Registered Image",
            visible=False,
        )

        self._viewer.grid.enabled = False

        if self.output_directory:
            self.save_outputs(
                boundaries,
                deformation_field,
                moving_image,
                data_in_atlas_space,
                result,
                registered_annotation_image,
                registered_hemisphere,
            )

            areas_path = self.output_directory / "areas.csv"
            calculate_areas(
                self._atlas,
                registered_annotation_image,
                registered_hemisphere,
                areas_path,
            )

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

    def _on_scale_moving_image(self, x: float, y: float):
        """
        Scale the moving image to have resolution equal to the atlas.

        Parameters
        ------------
        x : float
            Moving image x pixel size (> 0.0).
        y : float
            Moving image y pixel size (> 0.0).

        Will show an error if the pixel sizes are less than or equal to 0.
        Will show an error if the moving image or atlas is not selected.
        """
        if x <= 0 or y <= 0:
            show_error("Pixel sizes must be greater than 0")
            return

        if not (self._moving_image and self._atlas):
            show_error(
                "Sample image or atlas not selected. "
                "Please select a sample image and atlas before scaling",
            )
            return

        if self._moving_image_data_backup is None:
            self._moving_image_data_backup = self._moving_image.data.copy()

        x_factor = x / self._atlas.resolution[0]
        y_factor = y / self._atlas.resolution[1]

        self._moving_image.data = rescale(
            self._moving_image_data_backup,
            (y_factor, x_factor),
            mode="constant",
            preserve_range=True,
            anti_aliasing=True,
        )

    def _on_adjust_atlas_rotation(self, pitch: float, yaw: float, roll: float):
        if not (
            self._atlas
            and self._atlas_data_layer
            and self._atlas_annotations_layer
        ):
            show_error(
                "No atlas selected. Please select an atlas before rotating"
            )
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

        # Combine the matrices. The order of operations is:
        # 1. Translate the origin to the center of the image
        # 2. Rotate the image
        # 3. Translate the origin back to the top left corner
        transform_matrix = (
            post_rotate_translation @ full_matrix @ translate_matrix
        )

        self._atlas_data_layer.data = dask_affine_transform(
            self._atlas.reference,
            np.linalg.inv(transform_matrix),
            order=2,
            output_shape=bounding_box,
            output_chunks=(2, bounding_box[1], bounding_box[2]),
        )

        self._atlas_annotations_layer.data = dask_affine_transform(
            self._atlas.annotation,
            np.linalg.inv(transform_matrix),
            order=0,
            output_shape=bounding_box,
            output_chunks=(2, bounding_box[1], bounding_box[2]),
        )

        # Resets the viewer grid to update the grid to the new atlas
        # The grid is disabled and re-enabled to force the grid to update
        self._viewer.reset_view()
        self._viewer.grid.enabled = False
        self._viewer.grid.enabled = True

        worker = self.compute_atlas_rotation(self._atlas_data_layer.data)
        worker.returned.connect(self.set_atlas_layer_data)
        worker.start()

    @thread_worker
    def compute_atlas_rotation(self, dask_array: da.Array):
        self.adjust_moving_image_widget.reset_atlas_button.setEnabled(False)
        self.adjust_moving_image_widget.adjust_atlas_rotation.setEnabled(False)

        computed_array = dask_array.compute()

        self.adjust_moving_image_widget.reset_atlas_button.setEnabled(True)
        self.adjust_moving_image_widget.adjust_atlas_rotation.setEnabled(True)

        return computed_array

    def set_atlas_layer_data(self, new_data):
        self._atlas_data_layer.data = new_data

    def _on_atlas_reset(self):
        if not self._atlas:
            show_error(
                "No atlas selected. Please select an atlas before resetting"
            )
            return

        self._atlas_data_layer.data = self._atlas.reference
        self._atlas_annotations_layer.data = self._atlas.annotation
        self._viewer.grid.enabled = False
        self._viewer.grid.enabled = True

    def save_outputs(
        self,
        boundaries: npt.NDArray,
        deformation_field: npt.NDArray,
        downsampled: npt.NDArray,
        data_in_atlas_space: npt.NDArray,
        atlas_in_data_space: npt.NDArray,
        annotation_in_data_space: npt.NDArray,
        registered_hemisphere: npt.NDArray,
    ):
        """
        Save the outputs of the registration to the output directory.

        The outputs are saved as per
        https://brainglobe.info/documentation/brainreg/user-guide/output-files.html

        Parameters
        ----------
        boundaries: npt.NDArray
            The area boundaries of the registered annotation image.
        deformation_field: npt.NDArray
            The deformation field.
        downsampled: npt.NDArray
            The downsampled moving image.
        data_in_atlas_space: npt.NDArray
            The moving image in atlas space.
        atlas_in_data_space: npt.NDArray
            The atlas in data space.
        annotation_in_data_space: npt.NDArray
            The annotation in data space.#
        registered_hemisphere: npt.NDArray
            The hemisphere annotation in data space.
        """
        assert self._moving_image
        assert self.output_directory

        imwrite(self.output_directory / "boundaries.tiff", boundaries)

        for i in range(deformation_field.shape[-1]):
            imwrite(
                self.output_directory / f"deformation_field_{i}.tiff",
                deformation_field[:, :, i],
            )

        imwrite(self.output_directory / "downsampled.tiff", downsampled)
        imwrite(
            self.output_directory
            / f"downsampled_standard_{self._moving_image.name}.tiff",
            data_in_atlas_space,
        )
        imwrite(
            self.output_directory / "registered_atlas.tiff",
            annotation_in_data_space,
        )

        imwrite(
            self.output_directory / "registered_hemispheres.tiff",
            registered_hemisphere,
        )
