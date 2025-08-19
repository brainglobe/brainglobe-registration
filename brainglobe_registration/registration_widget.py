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
from brainglobe_registration.utils.logging import (
    StripANSIColorFilter,
    get_auto_slice_logging_args,
)
from brainglobe_registration.utils.transforms import (
    create_rotation_matrix,
    rotate_volume,
)
from brainglobe_registration.utils.utils import (
    calculate_region_size,
    check_atlas_installed,
    find_layer_index,
    get_data_from_napari_layer,
    get_image_layer_names,
    open_parameter_file,
    serialize_registration_widget,
)
from brainglobe_registration.widgets.adjust_moving_image_view import (
    AdjustMovingImageView,
)
from brainglobe_registration.widgets.parameter_list_view import (
    RegistrationParameterListView,
)
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
        self._atlas_transform_matrix: Optional[npt.NDArray] = None
        self._moving_image: Optional[napari.layers.Image] = None
        self._moving_image_data_backup: Optional[npt.NDArray] = None
        self.moving_anatomical_space: Optional[AnatomicalSpace] = None
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
        self._atlas_transform_matrix = None
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

        from brainglobe_registration.elastix.register import (
            calculate_deformation_field,
            invert_transformation,
            run_registration,
            transform_annotation_image,
            transform_image,
        )

        moving_image = get_data_from_napari_layer(self._moving_image).astype(
            np.uint16
        )
        current_atlas_slice = self._viewer.dims.current_step[0]

        if self._moving_image.data.ndim == 2:
            atlas_selection = (
                slice(current_atlas_slice, current_atlas_slice + 1),
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

        self._viewer.add_image(
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
        hemisphere_image = self._atlas.hemispheres

        if self._atlas_transform_matrix is not None:
            hemisphere_image = dask_affine_transform(
                self._atlas.hemispheres,
                self._atlas_transform_matrix,
                order=0,
                output_shape=self._atlas_data_layer.data.shape,
            )

        if self._moving_image.ndim == 2:
            hemisphere_image = hemisphere_image[current_atlas_slice, :, :]
        else:
            hemisphere_image = hemisphere_image

        if isinstance(hemisphere_image, da.Array):
            hemisphere_image = hemisphere_image.compute()

        registered_hemisphere = transform_annotation_image(
            hemisphere_image, parameters
        )

        registered_hemisphere_path = (
            self.output_directory / "registered_hemisphere.tiff"
        )
        imwrite(registered_hemisphere_path, registered_hemisphere)

        if self._moving_image.data.ndim == 2:
            region_stat_path = self.output_directory / "areas.csv"
        else:
            region_stat_path = self.output_directory / "volumes.csv"

        calculate_region_size(
            self._atlas,
            registered_annotation_image,
            registered_hemisphere,
            region_stat_path,
        )

        # Free up memory
        del registered_hemisphere
        del hemisphere_image

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

        print("Saving outputs")
        imwrite(self.output_directory / "downsampled.tiff", moving_image)

        with open(
            self.output_directory / "brainglobe-registration.json", "w"
        ) as f:
            json.dump(self, f, default=serialize_registration_widget, indent=4)

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

        transform_matrix, bounding_box = create_rotation_matrix(
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
        )

        rotated_annotations = rotate_volume(
            data=self._atlas.annotation,
            reference_shape=self._atlas.reference.shape,
            final_transform=transform_matrix,
            bounding_box=bounding_box,
            interpolation_order=0,
        )

        self._atlas_transform_matrix = transform_matrix
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
        moving_data = get_data_from_napari_layer(self._moving_image)
        is_slab = moving_data.ndim == 3

        if is_slab:
            total = 4 * (params["init_points"] + params["n_iter"])
            run_method = self.run_auto_slab_thread
            callback = self.set_optimal_rotation_params_for_slab
        else:
            total = 2 * (params["init_points"] + params["n_iter"])
            run_method = self.run_auto_slice_thread
            callback = self.set_optimal_rotation_params

        self.adjust_moving_image_widget.progress_bar.setVisible(True)
        self.adjust_moving_image_widget.progress_bar.setValue(0)
        self.adjust_moving_image_widget.progress_bar.setRange(0, total)

        worker = create_worker(
            run_method,
            params,
            _progress={"total": total, "desc": "Optimising..."},
        )

        worker.yielded.connect(self.handle_auto_slice_progress)
        worker.returned.connect(callback)
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

    def run_auto_slab_thread(self, params: dict):
        atlas_image = get_data_from_napari_layer(self._atlas_data_layer)
        slab = get_data_from_napari_layer(self._moving_image).astype(np.int16)

        # Define a logging output directory
        logging_dir = get_brainglobe_dir() / "brainglobe_registration_logs"
        logging_dir.mkdir(parents=True, exist_ok=True)

        args_namedtuple = get_auto_slice_logging_args(params)

        fancylog.start_logging(
            output_dir=str(logging_dir),
            package=brainglobe_registration,
            filename="auto_slab_log",
            variables=args_namedtuple,
            log_header="AUTO SLAB DETECTION LOG",
            verbose=True,
            write_git=False,
        )

        for handler in logging.getLogger().handlers:
            handler.addFilter(StripANSIColorFilter())

        logging.info(
            "\nBayesian slice detection for the first slice in the slab..."
        )

        first_slice = slab[0]
        last_slice = slab[-1]

        progress_i = 0
        result_first = run_bayesian_generator(
            atlas_image,
            first_slice,
            params["z_range"],
            params["pitch_bounds"],
            params["yaw_bounds"],
            params["roll_bounds"],
            params["init_points"],
            params["n_iter"],
            params["metric"],
            params["weights"],
        )

        try:
            while True:
                next(result_first)
                progress_i += 1
                yield {"progress": progress_i}
        except StopIteration as stop:
            final_first = stop.value

        logging.info(
            "\nBayesian slice detection for the last slice in the slab..."
        )

        result_last = run_bayesian_generator(
            atlas_image,
            last_slice,
            params["z_range"],
            params["pitch_bounds"],
            params["yaw_bounds"],
            params["roll_bounds"],
            params["init_points"],
            params["n_iter"],
            params["metric"],
            params["weights"],
        )

        try:
            while True:
                next(result_last)
                progress_i += 1
                yield {"progress": progress_i}
        except StopIteration as stop:
            final_last = stop.value

        logging.info(
            "\nFirst and last slices have been matched to the atlas."
            "Finding slices in between..."
        )

        # --- Z slice calculation ---
        z1 = final_first["best_z_slice"]
        z2 = final_last["best_z_slice"]
        num_slices = slab.shape[0]
        z_min = min(z1, z2)
        z_max = max(z1, z2)
        target_depth = z_max - z_min + 1

        if target_depth < num_slices:
            logging.info(
                "Case 1: Expanding outward to match number of slab slices"
            )
            current_first = z_min
            current_last = z_max
            while (current_last - current_first + 1) < num_slices:
                if current_first > 0:
                    current_first -= 1
                if (
                    current_last < atlas_image.shape[0] - 1
                    and (current_last - current_first + 1) < num_slices
                ):
                    current_last += 1
            target_z_indices = list(range(current_first, current_last + 1))
        elif target_depth == num_slices:
            logging.info("Case 2: Exact match between slab and atlas z-slices")
            target_z_indices = list(range(z_min, z_max + 1))
        else:
            logging.info("Case 3: Subsampling across wider atlas z-range")
            target_z_indices = (
                np.linspace(z_min, z_max, num_slices).astype(int).tolist()
            )

        # Interpolate pitch/yaw/roll across the number of slices
        num_slices = slab.shape[0]
        pitches = np.linspace(
            final_first["best_pitch"], final_last["best_pitch"], num_slices
        )
        yaws = np.linspace(
            final_first["best_yaw"], final_last["best_yaw"], num_slices
        )
        rolls = np.linspace(
            final_first["best_roll"], final_last["best_roll"], num_slices
        )

        per_slice_params = []
        for i in range(num_slices):
            per_slice_params.append(
                {
                    "pitch": float(pitches[i]),
                    "yaw": float(yaws[i]),
                    "roll": float(rolls[i]),
                    "z_slice": target_z_indices[i],
                }
            )

        logging.info(
            "Optimal parameters for each slice in the slab:\n"
            + "\n".join(
                f"Slice {i}: pitch={p['pitch']:.3f}, "
                f"yaw={p['yaw']:.3f}, roll={p['roll']:.3f}, "
                f"z_slice={p['z_slice']}"
                for i, p in enumerate(per_slice_params)
            )
        )

        root_logger = logging.getLogger()
        if root_logger.hasHandlers():
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)

        return {
            "done": True,
            "per_slice_params": per_slice_params,
        }

    def handle_auto_slice_progress(self, update: dict):
        if isinstance(update, dict) and "progress" in update:
            self.adjust_moving_image_widget.progress_bar.setValue(
                update["progress"]
            )

    def set_optimal_rotation_params(self, result):
        if result.get("done"):

            if "z_indices" in result:
                self.set_optimal_rotation_params_for_slab(result)
                return

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

    def set_optimal_rotation_params_for_slab(self, result):
        atlas_volume = get_data_from_napari_layer(self._atlas_data_layer)
        slab = get_data_from_napari_layer(self._moving_image).astype(np.int16)

        per_slice_params = result["per_slice_params"]
        self._per_slice_rotation_params = per_slice_params

        # 1: Create blank volume and fill it with rotated slices at target Z
        blank_volume = np.zeros_like(atlas_volume)
        for i, entry in enumerate(per_slice_params):
            slab_idx = i
            atlas_z = entry["z_slice"]
            if not (0 <= slab_idx < slab.shape[0]):
                continue
            if not (0 <= atlas_z < blank_volume.shape[0]):
                continue

            slice_data = slab[slab_idx]
            y_offset = (atlas_volume.shape[1] - slice_data.shape[0]) // 2
            x_offset = (atlas_volume.shape[2] - slice_data.shape[1]) // 2

            y_start = max(0, y_offset)
            x_start = max(0, x_offset)
            y_end = min(y_start + slice_data.shape[0], atlas_volume.shape[1])
            x_end = min(x_start + slice_data.shape[1], atlas_volume.shape[2])

            y_slice_end = y_end - y_start
            x_slice_end = x_end - x_start

            blank_volume[atlas_z, y_start:y_end, x_start:x_end] = slice_data[
                :y_slice_end, :x_slice_end
            ]

        # 2: Replace existing moving image layer
        moving_image_name = self._moving_image.name
        if moving_image_name in self._viewer.layers:
            self._viewer.layers.remove(moving_image_name)

        new_layer = self._viewer.add_image(
            blank_volume, name=moving_image_name
        )
        self._moving_image = new_layer

        # 3: Update sample image list
        new_layer_name = new_layer.name
        for i, name in enumerate(self._sample_images):
            if name == moving_image_name or name == self._moving_image.name:
                self._sample_images[i] = new_layer_name
                break
        else:
            self._sample_images.append(new_layer_name)

        self.get_atlas_widget.update_sample_image_names(self._sample_images)
        dropdown_index = self._sample_images.index(new_layer_name)
        self._on_sample_dropdown_index_changed(dropdown_index)

        # 4: Jump viewer to first Z
        if per_slice_params:
            first_z = per_slice_params[0]["z_slice"]
            self._viewer.dims.set_point(0, first_z)

        # 5: Set initial spinbox values from first slice
        self._on_adjust_atlas_rotation(
            per_slice_params[0]["pitch"],
            per_slice_params[0]["yaw"],
            per_slice_params[0]["roll"],
        )
        self.adjust_moving_image_widget.adjust_atlas_pitch.setValue(
            per_slice_params[0]["pitch"]
        )
        self.adjust_moving_image_widget.adjust_atlas_yaw.setValue(
            per_slice_params[0]["yaw"]
        )
        self.adjust_moving_image_widget.adjust_atlas_roll.setValue(
            per_slice_params[0]["roll"]
        )

        # 6: Dynamic update of spinboxes based on Z
        def _update_spinboxes_from_slice(event=None):
            # Only update if event is for axis 0 or if event is None
            if event is not None and getattr(event, "axis", 0) != 0:
                return

            current_z = int(self._viewer.dims.point[0])
            for entry in self._per_slice_rotation_params:

                if entry["z_slice"] == current_z:
                    self._on_adjust_atlas_rotation(
                        entry["pitch"],
                        entry["yaw"],
                        entry["roll"],
                    )

                    (
                        self.adjust_moving_image_widget.adjust_atlas_pitch.setValue(
                            entry["pitch"]
                        )
                    )
                    (
                        self.adjust_moving_image_widget.adjust_atlas_yaw.setValue(
                            entry["yaw"]
                        )
                    )
                    (
                        self.adjust_moving_image_widget.adjust_atlas_roll.setValue(
                            entry["roll"]
                        )
                    )
                    break

        self._viewer.dims.events.point.connect(_update_spinboxes_from_slice)

        # 7: Hide progress bar
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

    def __dict__(self):
        return {
            "atlas": self._atlas,
            "atlas_data_layer": self._atlas_data_layer,
            "atlas_annotations_layer": self._atlas_annotations_layer,
            "moving_image": self._moving_image,
            "adjust_moving_image_widget": self.adjust_moving_image_widget,
            "transform_selections": self.transform_selections,
            "filter": self.filter_checkbox.isChecked(),
            "output_directory": self.output_directory,
        }
