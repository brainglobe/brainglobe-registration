"""
A napari widget to view atlases.

Atlases that are exposed by the Brainglobe atlas API are
shown in a table view using the Qt model/view framework
[Qt Model/View framework](https://doc.qt.io/qt-6/model-view-programming.html)

Users can download and add the atlas images/structures as layers to the viewer.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import dask.array as da
import napari.layers
import numpy as np
import numpy.typing as npt
from brainglobe_atlasapi import BrainGlobeAtlas
from brainglobe_atlasapi.config import get_brainglobe_dir
from brainglobe_atlasapi.list_atlases import get_downloaded_atlases
from brainglobe_space import AnatomicalSpace
from brainglobe_utils.qtpy.logo import header_widget
from dask_image.imread import imread as dask_imread
from dask_image.ndinterp import affine_transform as dask_affine_transform
from fancylog import fancylog
from napari.qt.threading import (
    create_worker,
    thread_worker,
)
from napari.utils.events import Event
from napari.utils.notifications import show_error
from napari.viewer import Viewer
from qt_niu.collapsible_widget import CollapsibleWidgetContainer
from qt_niu.dialog import display_info
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QWidget,
)
from skimage.segmentation import find_boundaries
from skimage.transform import rescale
from tifffile import imwrite

import brainglobe_registration
from brainglobe_registration.automated_target_selection import (
    run_bayesian_generator,
)
from brainglobe_registration.utils.atlas import calculate_region_size
from brainglobe_registration.utils.file import (
    open_parameter_file,
    serialize_registration_widget,
)
from brainglobe_registration.utils.logging import (
    StripANSIColorFilter,
    get_auto_slice_logging_args,
)
from brainglobe_registration.utils.napari import (
    check_atlas_installed,
    find_layer_index,
    get_data_from_napari_layer,
    get_image_layer_names,
)
from brainglobe_registration.utils.transforms import (
    create_rotation_matrix,
    rotate_volume,
)
from brainglobe_registration.utils.visuals import generate_checkerboard
from brainglobe_registration.widgets.adjust_moving_image_view import (
    AdjustMovingImageView,
)
from brainglobe_registration.widgets.parameter_list_view import (
    RegistrationParameterListView,
)
from brainglobe_registration.widgets.qc_widget import QCWidget
from brainglobe_registration.widgets.select_images_view import SelectImagesView
from brainglobe_registration.widgets.target_selection_widget import (
    AutoSliceDialog,
)
from brainglobe_registration.widgets.transform_select_view import (
    TransformSelectView,
)


class RegistrationWidget(QScrollArea):
    def __init__(self, napari_viewer: Viewer):
        super().__init__()
        self._widget = CollapsibleWidgetContainer()
        self._widget.setContentsMargins(10, 10, 10, 10)

        self._viewer = napari_viewer
        self._atlas: Optional[BrainGlobeAtlas] = None
        self._atlas_data_layer: Optional[napari.layers.Image] = None
        self._atlas_annotations_layer: Optional[napari.layers.Labels] = None
        self._atlas_transform_matrix: npt.NDArray = np.eye(3)
        self._atlas_matrix_inv: npt.NDArray = np.eye(3)
        self._atlas_offset = np.zeros(3)
        self._atlas_2d_slice_index: int = -1
        self._atlas_2d_slice_corners: npt.NDArray = np.zeros((4, 3))
        self._moving_image: Optional[napari.layers.Image] = None
        self._moving_image_data_backup: Optional[npt.NDArray] = None

        self.moving_anatomical_space: Optional[AnatomicalSpace] = None
        # Flag to differentiate between manual and automatic atlas deletion
        self._automatic_deletion_flag = False
        # Registered image layer reference (saved after registration)
        self._registered_image: Optional[napari.layers.Image] = None
        # Checkerboard visualization
        self._checkerboard_layer: Optional[napari.layers.Image] = None
        # Cached image data for QC (avoids repeated layer queries)
        self._cached_moving_data: Optional[npt.NDArray] = None
        self._cached_registered_data: Optional[npt.NDArray] = None
        # Timer for debouncing square size changes (real-time updates)
        self._square_size_timer = QTimer()
        self._square_size_timer.setSingleShot(True)
        self._square_size_timer.timeout.connect(self._on_square_size_changed)

        self.transform_params: dict[str, dict] = {
            "affine": {},
            "bspline": {},
        }
        self.transform_selections = []

        for transform_type in self.transform_params:
            file_path = (
                Path(__file__).parent.resolve()
                / "parameters"
                / "brainglobe_registration"
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

        self.adjust_moving_image_widget = AdjustMovingImageView(
            parent=self,
            auto_slice_callback=self._open_auto_slice_dialog,
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
        self.adjust_moving_image_widget.reset_moving_image_signal.connect(
            self._on_moving_image_reset
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

        # QC widget for quality control visualizations
        self.qc_widget = QCWidget(parent=self)
        # Plot QC button - generates all selected QC plots
        self.qc_widget.plot_qc_button.clicked.connect(self._on_plot_qc_clicked)
        # Clear QC images button
        self.qc_widget.clear_qc_button.clicked.connect(
            self._on_clear_qc_images
        )
        # Square size spinbox - real-time updates when checkerboard displayed
        self.qc_widget.square_size_spinbox.valueChanged.connect(
            self._on_square_size_value_changed
        )

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

        self._widget.add_widget(
            header_widget(
                "brainglobe-<br>registration",  # line break at <br>
                "Registration with Elastix",
                github_repo_name="brainglobe-registration",
            ),
            collapsible=False,
        )
        self._widget.add_widget(
            self.get_atlas_widget, widget_title="Select Images"
        )
        self._widget.add_widget(
            self.adjust_moving_image_widget, widget_title="Prepare Images"
        )
        self._widget.add_widget(
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

        self._widget.add_widget(
            self.parameters_tab, widget_title="Advanced Settings (optional)"
        )

        # Add QC widget after Advanced Settings
        self._widget.add_widget(self.qc_widget, widget_title="Quality Control")

        self._widget.add_widget(self.filter_checkbox, collapsible=False)

        self._widget.add_widget(QLabel("Output Directory"), collapsible=False)
        self._widget.add_widget(
            self.output_directory_widget, collapsible=False
        )
        self._widget.add_widget(self.run_button, collapsible=False)

        self._widget.layout().itemAt(1).widget().collapse(animate=False)

        check_atlas_installed(self)

        self.setWidgetResizable(True)
        self.setWidget(self._widget)
        self._update_is_3d_flag()

    def _update_is_3d_flag(self):
        if self._moving_image is None:
            return
        is_3d = self._moving_image.ndim == 3
        self.adjust_moving_image_widget.set_is_3d(is_3d)

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

        self._reset_atlas_attributes()

        self.run_button.setEnabled(False)
        self._viewer.grid.enabled = False

    def _reset_atlas_attributes(self):
        self._atlas_transform_matrix = np.eye(3)
        self._atlas_matrix_inv = np.eye(3)
        self._atlas_offset = np.zeros(3)
        self._atlas_2d_slice_index = -1
        self._atlas_2d_slice_corners = np.zeros((4, 3))

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
        ).astype(np.uint32)

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
        self._update_is_3d_flag()

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
        if not (self._atlas and self._atlas_data_layer):
            display_info(
                widget=self,
                title="Warning",
                message="Please select an atlas before clicking 'Run'.",
            )
            return

        if not self._moving_image:
            display_info(
                widget=self,
                title="Warning",
                message="Please select a moving image before clicking 'Run'.",
            )
            return

        if self._moving_image == self._atlas_data_layer:
            display_info(
                widget=self,
                title="Warning",
                message="Your moving image cannot be an atlas.",
            )
            return

        if not self.output_directory:
            display_info(
                widget=self,
                title="Warning",
                message="Please select an output directory "
                "before clicking 'Run'.",
            )
            return

        moving_image = get_data_from_napari_layer(self._moving_image).astype(
            np.uint16
        )
        self._atlas_2d_slice_index = self._viewer.dims.current_step[0]

        if self._moving_image.data.ndim == 2:
            atlas_selection = (
                slice(
                    self._atlas_2d_slice_index, self._atlas_2d_slice_index + 1
                ),
            )
            atlas_image = get_data_from_napari_layer(
                self._atlas_data_layer, atlas_selection
            ).astype(np.float32)
            annotation_image = get_data_from_napari_layer(
                self._atlas_annotations_layer, atlas_selection
            )

            moving_image = moving_image.astype(np.float32)

            for transform_selection in self.transform_selections:
                # Can't use a short for internal pixels on 2D images
                fixed_pixel_type = transform_selection[1].get(
                    "FixedInternalImagePixelType", []
                )
                moving_pixel_type = transform_selection[1].get(
                    "MovingInternalImagePixelType", []
                )
                if "float" not in fixed_pixel_type:
                    print(
                        f"Can not use {fixed_pixel_type} "
                        f"for internal pixels on 2D images, switching to float"
                    )
                    transform_selection[1]["FixedInternalImagePixelType"] = [
                        "float"
                    ]
                if "float" not in moving_pixel_type:
                    print(
                        f"Can not use {moving_pixel_type} "
                        f"for internal pixels on 2D images, switching to float"
                    )
                    transform_selection[1]["MovingInternalImagePixelType"] = [
                        "float"
                    ]

            rotated_shape = np.array(self._atlas_data_layer.data.shape)
            original_shape = np.array(self._atlas.shape)

            atlas_corners = [
                [self._atlas_2d_slice_index, 0, 0],
                [self._atlas_2d_slice_index, 0, atlas_image.shape[1]],
                [self._atlas_2d_slice_index, atlas_image.shape[0], 0],
                [
                    self._atlas_2d_slice_index,
                    atlas_image.shape[0],
                    atlas_image.shape[1],
                ],
            ]

            # Centers
            rotated_center = rotated_shape / 2.0
            original_center = original_shape / 2.0

            # Inverse transform: rotated -> original
            original_corners = np.trunc(
                (
                    self._atlas_transform_matrix
                    @ (atlas_corners - rotated_center).T
                ).T
                + original_center
            )

            self._atlas_2d_slice_corners = (
                original_corners * self._atlas.resolution
            )

        else:
            atlas_image = get_data_from_napari_layer(self._atlas_data_layer)
            annotation_image = get_data_from_napari_layer(
                self._atlas_annotations_layer
            )

        for transform_selection in self.transform_selections:
            if "FixedImageDimension" not in transform_selection[1]:
                transform_selection[1]["FixedImageDimension"] = [
                    str(self._moving_image.data.ndim)
                ]
            if "MovingImageDimension" not in transform_selection[1]:
                transform_selection[1]["MovingImageDimension"] = [
                    str(self._moving_image.data.ndim)
                ]

        # Import elastix functions locally to avoid slow widget loading
        from brainglobe_registration.elastix.register import (
            calculate_deformation_field,
            invert_transformation,
            run_registration,
            transform_annotation_image,
            transform_image,
        )

        print("Running registration")
        parameters = run_registration(
            atlas_image,
            moving_image,
            self.transform_selections,
            self.output_directory,
            filter_images=self.filter_checkbox.isChecked(),
        )

        atlas_in_data_space = da.from_array(
            transform_image(atlas_image, parameters)
        )

        # Save reference to registered image layer for QC visualizations
        self._registered_image = self._viewer.add_image(
            atlas_in_data_space, name="Registered Image", visible=False
        )

        print("Inverting transformation")
        inverse_parameters = invert_transformation(
            atlas_image,
            self.transform_selections,
            parameters,
            self.output_directory,
            filter_images=self.filter_checkbox.isChecked(),
        )

        data_in_atlas_space = da.from_array(
            transform_image(moving_image, inverse_parameters)
        )
        data_in_atlas_space_path = (
            self.output_directory
            / f"downsampled_standard_{self._moving_image.name}.tiff"
        )

        imwrite(
            data_in_atlas_space_path,
            data_in_atlas_space,
        )

        self._viewer.add_image(
            data_in_atlas_space,
            name="Inverse Registered Image",
            visible=False,
        )

        print("Transforming annotation image")
        registered_annotation_image = transform_annotation_image(
            annotation_image,
            parameters,
        )

        registered_annotation_image_path = (
            self.output_directory / "registered_atlas.tiff"
        )
        imwrite(registered_annotation_image_path, registered_annotation_image)
        hemispheres_image = self._atlas.hemispheres

        if not np.allclose(self._atlas_transform_matrix, np.eye(3)):
            hemispheres_image = dask_affine_transform(
                self._atlas.hemispheres,
                self._atlas_transform_matrix,
                offset=self._atlas_offset,
                order=0,
                output_shape=self._atlas_data_layer.data.shape,
            )

        if self._moving_image.ndim == 2:
            hemispheres_image = hemispheres_image[
                self._atlas_2d_slice_index, :, :
            ]

        if isinstance(hemispheres_image, da.Array):
            hemispheres_image = hemispheres_image.compute()

        registered_hemispheres = transform_annotation_image(
            hemispheres_image, parameters
        )

        registered_hemispheres_path = (
            self.output_directory / "registered_hemispheres.tiff"
        )
        imwrite(registered_hemispheres_path, registered_hemispheres)

        if self._moving_image.data.ndim == 2:
            region_stat_path = self.output_directory / "areas.csv"
        else:
            region_stat_path = self.output_directory / "volumes.csv"

        calculate_region_size(
            self._atlas,
            registered_annotation_image,
            registered_hemispheres,
            region_stat_path,
        )

        # Free up memory
        del registered_hemispheres
        del hemispheres_image

        boundaries = find_boundaries(
            registered_annotation_image, mode="inner"
        ).astype(np.int8, copy=False)

        imwrite(self.output_directory / "boundaries.tiff", boundaries)

        if self._moving_image.data.ndim != 2:
            # Free up memory
            del registered_annotation_image

            registered_annotation_image = dask_imread(
                registered_annotation_image_path
            )

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

        print("Calculating deformation field")
        deformation_field = calculate_deformation_field(
            moving_image, parameters
        )

        for i in range(deformation_field.shape[-1]):
            imwrite(
                self.output_directory / f"deformation_field_{i}.tiff",
                deformation_field[..., i],
            )

        self._atlas_data_layer.visible = False
        self._viewer.grid.enabled = False

        # Cache image data for QC (avoids repeated layer queries)
        # Improves performance: get_data_from_napari_layer can be slow
        # with Dask arrays or large images
        self._cached_moving_data = get_data_from_napari_layer(
            self._moving_image
        )
        self._cached_registered_data = get_data_from_napari_layer(
            self._registered_image
        )

        # Enable QC widget now that registration is complete
        self.qc_widget.set_enabled(True)

        # Set default square size based on image dimensions if not
        # user-modified (adaptive: roughly 1/16th of smallest dimension,
        # minimum 8 pixels)
        min_dimension = min(moving_image.shape[-2:])
        default_square_size = max(8, min_dimension // 16)
        # Only update if still at default value (32)
        if self.qc_widget.square_size_spinbox.value() == 32:
            self.qc_widget.square_size_spinbox.setValue(default_square_size)

        print("Saving outputs")
        imwrite(self.output_directory / "downsampled.tiff", moving_image)

        with open(
            self.output_directory / "brainglobe-registration.json", "w"
        ) as f:
            json.dump(self, f, default=serialize_registration_widget, indent=4)

    def _on_plot_qc_clicked(self):
        """
        Generate all selected QC visualizations.

        Checks which QC plots are selected and generates them.
        This prevents accidental computation from checkbox toggles.
        """
        # Check which QC plots are selected
        if self.qc_widget.checkerboard_checkbox.isChecked():
            self._show_checkerboard()

    def _show_checkerboard(self):
        """
        Generate and display checkerboard using a background thread.

        This method uses threading to keep the UI responsive during
        checkerboard generation, especially important for large 3D images.
        """
        if not self._moving_image:
            show_error(
                "No moving image available. "
                "Please select a moving image first."
            )
            return

        # Use saved reference to registered image layer
        if self._registered_image is None:
            show_error(
                "Registered Image layer not found. "
                "Please run registration first."
            )
            return

        # Use cached data if available (set after registration),
        # otherwise fetch from layers and cache (fallback)
        if self._cached_moving_data is not None:
            moving_data = self._cached_moving_data
            registered_data = self._cached_registered_data
        else:
            # Fallback if cache is empty (shouldn't happen normally)
            self._cached_moving_data = get_data_from_napari_layer(
                self._moving_image
            )
            self._cached_registered_data = get_data_from_napari_layer(
                self._registered_image
            )
            moving_data = self._cached_moving_data
            registered_data = self._cached_registered_data

        # Handle dimension mismatch: if moving image is 2D but registered is
        # 3D, extract the current slice from the registered image
        if moving_data.ndim == 2 and registered_data.ndim == 3:
            # Get current slice position from viewer
            current_slice = self._viewer.dims.current_step[0]
            # Extract the corresponding slice
            registered_data = registered_data[current_slice, :, :]
        elif moving_data.ndim == 3 and registered_data.ndim == 2:
            # If moving is 3D but registered is 2D, we can't match them
            show_error(
                f"Cannot create checkerboard: dimension mismatch. "
                f"Moving image is 3D {moving_data.shape} but "
                f"Registered image is 2D {registered_data.shape}"
            )
            return

        # Get square size from QC widget (user-configurable)
        square_size = self.qc_widget.square_size_spinbox.value()

        # Create worker to generate checkerboard in background thread
        # This keeps the UI responsive during computation
        worker = create_worker(
            self._generate_checkerboard_thread,
            moving_data,
            registered_data,
            square_size,
        )

        # When worker finishes, update the layer on main thread
        worker.returned.connect(self._update_checkerboard_layer)
        worker.errored.connect(self._on_checkerboard_error)
        worker.start()

    def _on_checkerboard_error(self, error: Exception):
        """
        Handle errors from checkerboard generation worker.

        Parameters
        ----------
        error : Exception
            The error that occurred during checkerboard generation.
        """
        show_error(f"Error generating checkerboard: {error}")
        self.qc_widget.checkerboard_checkbox.setChecked(False)

    def _generate_checkerboard_thread(
        self,
        moving_data: npt.NDArray,
        registered_data: npt.NDArray,
        square_size: int,
    ) -> npt.NDArray:
        """
        Generate checkerboard pattern in background thread.

        This function runs in a background thread and does NOT freeze the UI.
        The heavy computation (normalization, pattern generation) happens here.

        Parameters
        ----------
        moving_data : npt.NDArray
            The moving image data.
        registered_data : npt.NDArray
            The registered image data.
        square_size : int
            Size of checkerboard squares in pixels.

        Returns
        -------
        npt.NDArray
            The generated checkerboard pattern.
        """
        # Note: generate_checkerboard() handles shape mismatches automatically
        # by cropping to the overlapping region
        return generate_checkerboard(
            moving_data, registered_data, square_size=square_size
        )

    def _update_checkerboard_layer(self, checkerboard_data: npt.NDArray):
        """
        Update the checkerboard layer in Napari viewer (main thread).

        This method runs when the background computation is complete.
        It updates the layer on the main thread (required for Qt operations).

        Parameters
        ----------
        checkerboard_data : npt.NDArray
            The computed checkerboard pattern to display.
        """
        try:
            # Update existing layer if it exists (faster than remove/add)
            if self._checkerboard_layer is not None:
                if self._checkerboard_layer in self._viewer.layers:
                    # OPTIMIZATION: Update data in place instead of removing
                    # and re-adding the layer. This is much faster for Napari.
                    self._checkerboard_layer.data = checkerboard_data
                    self._checkerboard_layer.visible = True
                else:
                    # Layer was deleted manually, reset reference
                    self._checkerboard_layer = None

            # If layer doesn't exist (first run or was deleted), create it
            if self._checkerboard_layer is None:
                self._checkerboard_layer = self._viewer.add_image(
                    checkerboard_data,
                    name="Checkerboard",
                    colormap="gray",
                    blending="opaque",
                    visible=True,
                )

            # Hide original images for better visualization
            if self._moving_image is not None:
                self._moving_image.visible = False
            if self._registered_image is not None:
                self._registered_image.visible = False

        except Exception as e:
            show_error(f"Error updating checkerboard layer: {str(e)}")
            self.qc_widget.checkerboard_checkbox.setChecked(False)

    def _remove_checkerboard(self):
        """
        Remove the checkerboard visualization and restore original images.
        """
        if self._checkerboard_layer is not None:
            if self._checkerboard_layer in self._viewer.layers:
                self._viewer.layers.remove(self._checkerboard_layer)
            self._checkerboard_layer = None

        # Stop any pending square size updates
        self._square_size_timer.stop()

        # Restore visibility of original images
        if self._moving_image is not None:
            self._moving_image.visible = True

        if self._registered_image is not None:
            self._registered_image.visible = False

    def _on_square_size_value_changed(self, value: int):
        """
        Handle square size spinbox value changes with debouncing.

        This method restarts a timer. When the timer fires, if the
        checkerboard is currently displayed, it will be regenerated with the
        new square size. This provides real-time updates while avoiding
        excessive recomputation
        when the user holds the arrow button.

        Parameters
        ----------
        value : int
            The new square size value (unused, but required by signal).
        """
        # Restart the timer (debounce: wait 300ms after last change)
        self._square_size_timer.stop()
        self._square_size_timer.start(300)  # 300ms delay

    def _on_square_size_changed(self):
        """
        Regenerate checkerboard with new square size if currently displayed.

        This method is called by the debounce timer after the user stops
        changing the square size. It checks if the checkerboard is currently
        displayed and regenerates it with the new square size value.
        """
        # Only regenerate if checkerboard is currently displayed
        if (
            self._checkerboard_layer is not None
            and self._checkerboard_layer in self._viewer.layers
            and self.qc_widget.checkerboard_checkbox.isChecked()
        ):
            # Regenerate checkerboard with new square size
            self._show_checkerboard()

    def _on_clear_qc_images(self):
        """Remove all QC visualization layers."""
        # Remove checkerboard if it exists
        self._remove_checkerboard()

        # Uncheck all QC checkboxes
        self.qc_widget.checkerboard_checkbox.setChecked(False)

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

    def _on_scale_moving_image(
        self, x_res: float, y_res: float, z_res: float, orientation: str
    ):
        """
        Scale the moving image to have resolution equal to the atlas.

        Parameters
        ------------
        x_res : float
            Moving image x pixel size (> 0.0).
        y_res : float
            Moving image y pixel size (> 0.0).
        z_res : float
            Moving image z pixel size (> 0.0).
        orientation : str
            The orientation of the moving image BrainGlobe convention.
            Required for 3D scaling, can be an empty string if the image is 2D.

        Will show an error if the pixel sizes are less than or equal to 0.
        Will show an error if the moving image or atlas is not selected.
        Will show an error if the orientation is invalid.
        """
        if not (self._moving_image and self._atlas):
            show_error(
                "Sample image or atlas not selected. "
                "Please select a sample image and atlas before scaling",
            )
            return

        if (x_res <= 0 or y_res <= 0) or (
            self._moving_image.data.ndim == 3 and z_res <= 0
        ):
            show_error("Pixel sizes must be greater than 0")
            return

        valid_orientation_labels = ["p", "a", "s", "i", "l", "r"]
        orientation = orientation.lower()

        for label in orientation:
            if label not in valid_orientation_labels:
                show_error(
                    "Invalid orientation. "
                    "Please use the BrainGlobe convention (e.g. 'psl')"
                )
                return

        if self._moving_image_data_backup is None:
            self._moving_image_data_backup = self._moving_image.data.copy()

        # Debug: Print resolution information
        print(f"Atlas resolution (z,y,x): {self._atlas.resolution}")
        print(f"Input pixel sizes - x: {x_res}, y: {y_res}, z: {z_res}")

        # Atlas resolution is stored as z,y,x
        if self._moving_image.data.ndim == 3:
            self.moving_anatomical_space = AnatomicalSpace(
                origin=orientation,
                resolution=(z_res, y_res, x_res),
            )
            self._moving_image.data = (
                self.moving_anatomical_space.map_stack_to(
                    self._atlas.space, self._moving_image_data_backup
                )
            )
        else:
            x_factor = x_res / self._atlas.resolution[0]
            y_factor = y_res / self._atlas.resolution[1]
            scale: Tuple[float, ...] = (y_factor, x_factor)

            self._moving_image.data = rescale(
                self._moving_image_data_backup,
                scale,
                mode="constant",
                preserve_range=True,
                anti_aliasing=True,
            ).astype(self._moving_image_data_backup.dtype)

        print(f"Original image shape: {self._moving_image_data_backup.shape}")
        print(f"Atlas shape: {self._atlas.reference.shape}")

        # Resets the viewer grid to update the grid to the scaled moving image
        self._viewer.grid.enabled = False
        self._viewer.grid.enabled = True

        print(f"Scaled image shape: {self._moving_image.data.shape}")
        print("---")

    def _on_moving_image_reset(self) -> None:
        if not self._moving_image:
            show_error(
                "Sample image not selected. "
                "Please select a sample image before resetting"
            )
            return

        if self._moving_image_data_backup is None:
            show_error("No backup available to reset the moving image.")
            return

        self._moving_image.data = self._moving_image_data_backup.copy()
        self.moving_anatomical_space = None

        # Resets the viewer grid to update the grid to the original image
        self._viewer.grid.enabled = False
        self._viewer.grid.enabled = True

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

        transform_matrix, offset, bounding_box = create_rotation_matrix(
            roll,
            yaw,
            pitch,
            self._atlas.reference.shape,
        )

        rotated_reference = rotate_volume(
            data=self._atlas.reference,
            reference_shape=self._atlas.reference.shape,
            final_transform=transform_matrix,
            bounding_box=bounding_box,
            interpolation_order=2,
            offset=offset,
        )

        rotated_annotations = rotate_volume(
            data=self._atlas.annotation,
            reference_shape=self._atlas.reference.shape,
            final_transform=transform_matrix,
            bounding_box=bounding_box,
            interpolation_order=0,
            offset=offset,
        )

        self._atlas_transform_matrix = transform_matrix
        self._atlas_matrix_inv = np.linalg.inv(transform_matrix)
        self._atlas_offset = offset
        self._atlas_data_layer.data = rotated_reference
        self._atlas_annotations_layer.data = rotated_annotations

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

        self._reset_atlas_attributes()

        self._viewer.grid.enabled = False
        self._viewer.grid.enabled = True

    def _open_auto_slice_dialog(self):
        if not (self._atlas and self._atlas_data_layer):
            display_info(
                widget=self,
                title="Warning",
                message="Please select an atlas before "
                "clicking 'Automatic Slice Detection'.",
            )
            return

        if not self._moving_image:
            display_info(
                widget=self,
                title="Warning",
                message="Please select a moving image before "
                "clicking 'Automatic Slice Detection'.",
            )
            return

        if self._moving_image == self._atlas_data_layer:
            display_info(
                widget=self,
                title="Warning",
                message="Your moving image cannot be an atlas.",
            )
            return

        # Launch dialog
        max_z = self._atlas.reference.shape[0] - 1
        dialog = AutoSliceDialog(parent=self._widget, z_max_value=max_z)

        dialog.parameters_confirmed.connect(
            self._on_auto_slice_parameters_confirmed
        )
        dialog.exec_()

    def _on_auto_slice_parameters_confirmed(self, params: dict):
        total = 2 * (params["init_points"] + params["n_iter"])

        self.adjust_moving_image_widget.progress_bar.setVisible(True)
        self.adjust_moving_image_widget.progress_bar.setValue(0)
        self.adjust_moving_image_widget.progress_bar.setRange(0, total)

        worker = create_worker(
            self.run_auto_slice_thread,
            params,
            _progress={"total": total, "desc": "Optimising..."},
        )

        worker.yielded.connect(self.handle_auto_slice_progress)
        worker.returned.connect(self.set_optimal_rotation_params)
        worker.start()

    def run_auto_slice_thread(self, params: dict):
        atlas_image = get_data_from_napari_layer(self._atlas_data_layer)
        moving_image = get_data_from_napari_layer(self._moving_image).astype(
            np.int16
        )

        # Define a logging output directory
        logging_dir = get_brainglobe_dir() / "brainglobe_registration_logs"
        logging_dir.mkdir(parents=True, exist_ok=True)

        args_namedtuple = get_auto_slice_logging_args(params)

        fancylog.start_logging(
            output_dir=str(logging_dir),
            package=brainglobe_registration,
            filename="auto_slice_log",
            variables=args_namedtuple,
            log_header="AUTO SLICE DETECTION LOG",
            verbose=True,
            write_git=False,
        )

        for handler in logging.getLogger().handlers:
            handler.addFilter(StripANSIColorFilter())

        logging.info("Starting Bayesian slice detection...")

        result_generator = run_bayesian_generator(
            atlas_image,
            moving_image,
            params["z_range"],
            params["pitch_bounds"],
            params["yaw_bounds"],
            params["roll_bounds"],
            params["init_points"],
            params["n_iter"],
            params["metric"],
            params["weights"],
        )

        i = 0
        try:
            while True:
                next(result_generator)
                i += 1
                yield {"progress": i}
        except StopIteration as stop:
            final_result = stop.value

        root_logger = logging.getLogger()
        if root_logger.hasHandlers():
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)

        return {
            "done": True,
            "best_pitch": final_result["best_pitch"],
            "best_yaw": final_result["best_yaw"],
            "best_roll": final_result["best_roll"],
            "best_z_slice": final_result["best_z_slice"],
        }

    def handle_auto_slice_progress(self, update: dict):
        if isinstance(update, dict) and "progress" in update:
            self.adjust_moving_image_widget.progress_bar.setValue(
                update["progress"]
            )

    def set_optimal_rotation_params(self, result):
        if result.get("done"):
            pitch = result["best_pitch"]
            yaw = result["best_yaw"]
            roll = result["best_roll"]
            z_slice = result["best_z_slice"]

            # Apply rotation to atlas
            self._on_adjust_atlas_rotation(pitch, yaw, roll)
            self._viewer.dims.set_point(0, z_slice)

            # Update pitch, yaw, roll on GUI display
            self.adjust_moving_image_widget.adjust_atlas_pitch.setValue(pitch)
            self.adjust_moving_image_widget.adjust_atlas_yaw.setValue(yaw)
            self.adjust_moving_image_widget.adjust_atlas_roll.setValue(roll)

            self.adjust_moving_image_widget.progress_bar.reset()
            self.adjust_moving_image_widget.progress_bar.setVisible(False)

    def save_outputs(
        self,
        boundaries: npt.NDArray,
        deformation_field: npt.NDArray,
        downsampled: npt.NDArray,
        data_in_atlas_space: npt.NDArray,
        atlas_in_data_space: npt.NDArray,
        annotation_in_data_space: npt.NDArray,
        registered_hemispheres: npt.NDArray,
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
        registered_hemispheres: npt.NDArray
            The hemispheres annotation in data space.
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
            registered_hemispheres,
        )

    def __dict__(self):
        return {
            "atlas": self._atlas,
            "atlas_transform_matrix": self._atlas_transform_matrix.tolist(),
            "atlas_inverse_transform_matrix": self._atlas_matrix_inv.tolist(),
            "atlas_offset": self._atlas_offset.tolist(),
            "atlas_2d_slice_index": self._atlas_2d_slice_index,
            "atlas_slice_corners": self._atlas_2d_slice_corners.tolist(),
            "atlas_data_layer": self._atlas_data_layer,
            "atlas_annotations_layer": self._atlas_annotations_layer,
            "moving_image": self._moving_image,
            "adjust_moving_image_widget": self.adjust_moving_image_widget,
            "transform_selections": self.transform_selections,
            "filter": self.filter_checkbox.isChecked(),
            "output_directory": self.output_directory,
        }
