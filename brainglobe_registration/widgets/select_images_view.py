from typing import List

from brainglobe_utils.qtpy.dialog import display_info
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QComboBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)


class SampleImageComboBox(QComboBox):
    popout_about_to_show = Signal()

    def __init__(self, parent=None):
        super().__init__(parent=parent)

    def showPopup(self):
        self.popout_about_to_show.emit()
        super().showPopup()


class SelectImagesView(QWidget):
    """
    A QWidget subclass that provides a dropdown menu for selecting the image
    and atlas for registration.

    This widget provides two dropdown menus for selecting the atlas and the
    sample to be used for registration. It emits signals when the selected
    atlas or sample image changes.

    Attributes
    ----------
    atlas_index_change : Signal
        Emitted when the selected atlas changes. The signal includes the index
        of the selected atlas.
    moving_image_index_change : Signal
        Emitted when the selected sample image changes. The signal includes
        the index of the selected image.

    Methods
    -------
    _on_atlas_index_change():
        Emits the atlas_index_change signal with the index of the selected
        atlas.
    _on_moving_image_index_change():
        Emits the moving_image_index_change signal with the index of the
        selected image.
    """

    atlas_index_change = Signal(int)
    moving_image_index_change = Signal(int)
    sample_image_popup_about_to_show = Signal()

    def __init__(
        self,
        available_atlases: List[str],
        sample_image_names: List[str],
        parent: QWidget = None,
    ):
        """
        Initialize the widget.

        Parameters
        ----------
        available_atlases : List[str]
            The list of available atlases.
        sample_image_names : List[str]
            The list of available sample images.
        parent : QWidget, optional
            The parent widget, by default None
        """
        super().__init__(parent)

        self.setLayout(QVBoxLayout())
        self.available_atlas_dropdown_label = QLabel("Select Atlas:")
        self.available_atlas_dropdown = QComboBox(parent=self)
        self.available_atlas_dropdown.addItems(available_atlases)

        self.available_sample_dropdown_label = QLabel("Select sample:")
        self.available_sample_dropdown = SampleImageComboBox(parent=self)
        self.available_sample_dropdown.addItems(sample_image_names)

        self.available_atlas_dropdown.currentIndexChanged.connect(
            self._on_atlas_index_change
        )
        self.available_sample_dropdown.currentIndexChanged.connect(
            self._on_moving_image_index_change
        )
        self.available_sample_dropdown.popout_about_to_show.connect(
            self._on_sample_popup_about_to_show
        )

        self.layout().addWidget(self.available_atlas_dropdown_label)
        self.layout().addWidget(self.available_atlas_dropdown)
        self.layout().addWidget(self.available_sample_dropdown_label)
        self.layout().addWidget(self.available_sample_dropdown)

    def update_sample_image_names(self, sample_image_names: List[str]):
        self.available_sample_dropdown.clear()
        if len(sample_image_names) == 0:
            display_info(
                widget=QWidget(),
                title="Warning",
                message="Image list empty: please upload an image.",
            )
        else:
            self.available_sample_dropdown.addItems(sample_image_names)

    def _on_atlas_index_change(self):
        """
        Emit the atlas_index_change signal with the index of the selected
        atlas.

        If the selected index is invalid, emits -1.
        """
        self.atlas_index_change.emit(
            self.available_atlas_dropdown.currentIndex()
        )

    def _on_moving_image_index_change(self):
        """
        Emit the moving_image_index_change signal with the index of the
        selected image.

        If the selected index is invalid, emits -1.
        """
        self.moving_image_index_change.emit(
            self.available_sample_dropdown.currentIndex()
        )

    def reset_atlas_combobox(self):
        if self.available_atlas_dropdown is not None:
            self.available_atlas_dropdown.setCurrentIndex(0)

    def _on_sample_popup_about_to_show(self):
        self.sample_image_popup_about_to_show.emit()
