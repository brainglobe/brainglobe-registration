from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QGroupBox,
    QWidget,
    QVBoxLayout,
    QLabel,
    QComboBox,
)


class SelectImagesView(QGroupBox):
    """A widget to select the images to be used for registration."""

    atlas_index_change = Signal(int)
    moving_image_index_change = Signal(int)

    def __init__(
        self, available_atlases, sample_image_names, parent: QWidget = None
    ):
        super().__init__(parent)

        self.setLayout(QVBoxLayout())

        self.available_atlas_dropdown_label = QLabel("Select Atlas:")
        self.available_atlas_dropdown = QComboBox(parent=self)
        self.available_atlas_dropdown.addItems(available_atlases)

        self.available_sample_dropdown_label = QLabel("Select sample:")
        # TODO update the layer names dropdown when new images are opened
        #  in the viewer, can then use the index directly instead of looping
        #  through the layers
        self.available_sample_dropdown = QComboBox(parent=self)
        self.available_sample_dropdown.addItems(sample_image_names)

        self.available_atlas_dropdown.currentIndexChanged.connect(
            self._on_atlas_index_change
        )

        self.available_sample_dropdown.currentIndexChanged.connect(
            self._on_moving_image_index_change
        )

        self.layout().addWidget(self.available_atlas_dropdown_label)
        self.layout().addWidget(self.available_atlas_dropdown)
        self.layout().addWidget(self.available_sample_dropdown_label)
        self.layout().addWidget(self.available_sample_dropdown)

    def _on_atlas_index_change(self):
        self.atlas_index_change.emit(
            self.available_atlas_dropdown.currentIndex()
        )

    def _on_moving_image_index_change(self):
        self.moving_image_index_change.emit(
            self.available_sample_dropdown.currentIndex()
        )
