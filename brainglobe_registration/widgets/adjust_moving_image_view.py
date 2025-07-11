from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QWidget,
)


class AdjustMovingImageView(QWidget):
    """
    A QWidget subclass that provides controls for adjusting the moving image.

    This widget provides controls for adjusting the x and y offsets and
    rotation of the moving image. It emits signals when the image is adjusted,
    scaled, or reset.

    Attributes
    ----------
    scale_image_signal : Signal
        Emitted when the image is scaled. The signal includes the x, y, and z
        pixel sizes.
    atlas_rotation_signal : Signal
        Emitted when the atlas is rotated. The signal includes the pitch, yaw,
        and roll.
    reset_atlas_signal : Signal
        Emitted when the atlas is reset.

    Methods
    -------
    _on_scale_image_button_click():
        Emits the scale_image_signal with the entered pixel sizes.
    _on_adjust_atlas_rotation():
        Emits the atlas_rotation_signal with the entered pitch, yaw, and roll.
    _on_atlas_reset():
        Resets the pitch, yaw, and roll to 0 and emits the atlas_reset_signal.
    """

    scale_image_signal = Signal(float, float, float)
    atlas_rotation_signal = Signal(float, float, float)
    reset_atlas_signal = Signal()

    def __init__(self, parent=None, auto_slice_callback=None):
        """
        Initialize the widget.

        Parameters
        ----------
        parent : QWidget, optional
            The parent widget, by default None
        """
        super().__init__(parent=parent)

        self.setLayout(QFormLayout())
        self.layout().setLabelAlignment(Qt.AlignLeft)

        rotation_range = 360

        self.adjust_moving_image_pixel_size_x = QDoubleSpinBox(parent=self)
        self.adjust_moving_image_pixel_size_x.setDecimals(3)
        self.adjust_moving_image_pixel_size_x.setRange(0, 100.00)
        self.adjust_moving_image_pixel_size_x.setSpecialValueText("N/A")
        self.adjust_moving_image_pixel_size_y = QDoubleSpinBox(parent=self)
        self.adjust_moving_image_pixel_size_y.setDecimals(3)
        self.adjust_moving_image_pixel_size_y.setRange(0, 100.00)
        self.adjust_moving_image_pixel_size_y.setSpecialValueText("N/A")
        self.adjust_moving_image_pixel_size_z = QDoubleSpinBox(parent=self)
        self.adjust_moving_image_pixel_size_z.setDecimals(3)
        self.adjust_moving_image_pixel_size_z.setRange(0, 100.00)
        self.adjust_moving_image_pixel_size_z.setSpecialValueText("N/A")
        self.scale_moving_image_button = QPushButton()
        self.scale_moving_image_button.setText("Scale Image")
        self.scale_moving_image_button.clicked.connect(
            self._on_scale_image_button_click
        )

        self.adjust_atlas_pitch = QDoubleSpinBox(parent=self)
        self.adjust_atlas_pitch.setSingleStep(0.1)
        self.adjust_atlas_pitch.setRange(-rotation_range, rotation_range)

        self.adjust_atlas_yaw = QDoubleSpinBox(parent=self)
        self.adjust_atlas_yaw.setSingleStep(0.1)
        self.adjust_atlas_yaw.setRange(-rotation_range, rotation_range)

        self.adjust_atlas_roll = QDoubleSpinBox(parent=self)
        self.adjust_atlas_roll.setSingleStep(0.1)
        self.adjust_atlas_roll.setRange(-rotation_range, rotation_range)

        self.adjust_atlas_rotation = QPushButton()
        self.adjust_atlas_rotation.setText("Rotate Atlas")
        self.adjust_atlas_rotation.clicked.connect(
            self._on_adjust_atlas_rotation
        )
        self.reset_atlas_button = QPushButton()
        self.reset_atlas_button.setText("Reset Atlas")
        self.reset_atlas_button.clicked.connect(self._on_atlas_reset)

        self.layout().addRow(QLabel("Adjust the moving image scale:"))
        self.layout().addRow(
            "Sample image X pixel size (\u03bcm / pixel):",
            self.adjust_moving_image_pixel_size_x,
        )
        self.layout().addRow(
            "Sample image Y pixel size (\u03bcm / pixel):",
            self.adjust_moving_image_pixel_size_y,
        )
        self.layout().addRow(
            "Sample image Z pixel size (\u03bcm / pixel):",
            self.adjust_moving_image_pixel_size_z,
        )
        self.layout().addRow(self.scale_moving_image_button)

        automate_slice_label = QLabel(
            "Automatically choose atlas slice "
            "and rotation parameters that align "
            "most closely with your moving image:"
        )
        automate_slice_label.setWordWrap(True)
        self.layout().addRow(automate_slice_label)
        if auto_slice_callback is not None:
            self.auto_slice_button = QPushButton("Automatic Slice Detection")
            self.auto_slice_button.clicked.connect(auto_slice_callback)
            self.layout().addRow(self.auto_slice_button)

        # Progress bar wrapper to reserve height
        self.progress_bar_container = QWidget(self)
        self.progress_bar_layout = QHBoxLayout(self.progress_bar_container)
        self.progress_bar_layout.setContentsMargins(0, 0, 0, 0)
        self.progress_bar_layout.setSpacing(0)

        self.progress_bar = QProgressBar(self.progress_bar_container)
        self.progress_bar.setFixedHeight(20)
        self.progress_bar_layout.addWidget(self.progress_bar)

        self.layout().addRow(self.progress_bar_container)
        self.progress_bar.setVisible(False)

        self.layout().addRow(
            QLabel("Manually refine the atlas pitch and yaw: ")
        )
        self.layout().addRow("Pitch:", self.adjust_atlas_pitch)
        self.layout().addRow("Yaw:", self.adjust_atlas_yaw)
        self.layout().addRow("Roll:", self.adjust_atlas_roll)
        self.layout().addRow(self.adjust_atlas_rotation)
        self.layout().addRow(self.reset_atlas_button)

    def _on_scale_image_button_click(self):
        """
        Emit the scale_image_signal with the entered pixel sizes.
        """
        self.scale_image_signal.emit(
            self.adjust_moving_image_pixel_size_x.value(),
            self.adjust_moving_image_pixel_size_y.value(),
            self.adjust_moving_image_pixel_size_z.value(),
        )

    def _on_adjust_atlas_rotation(self):
        """
        Emit the atlas_rotation_signal with the entered pitch and yaw.
        """
        self.atlas_rotation_signal.emit(
            self.adjust_atlas_pitch.value(),
            self.adjust_atlas_yaw.value(),
            self.adjust_atlas_roll.value(),
        )

    def _on_atlas_reset(self):
        """
        Reset the pitch, yaw, and roll to 0 and emit the atlas_reset_signal.
        """
        self.adjust_atlas_yaw.setValue(0)
        self.adjust_atlas_pitch.setValue(0)
        self.adjust_atlas_roll.setValue(0)

        self.reset_atlas_signal.emit()

    def __dict__(self):
        return {
            "pixel_size_x": self.adjust_moving_image_pixel_size_x.value(),
            "pixel_size_y": self.adjust_moving_image_pixel_size_y.value(),
            "pixel_size_z": self.adjust_moving_image_pixel_size_z.value(),
            "atlas_pitch": self.adjust_atlas_pitch.value(),
            "atlas_yaw": self.adjust_atlas_yaw.value(),
            "atlas_roll": self.adjust_atlas_roll.value(),
        }
