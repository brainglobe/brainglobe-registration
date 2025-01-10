from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QSpinBox,
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
    adjust_image_signal : Signal
        Emitted when the image is adjusted. The signal includes the x and y
        offsets and rotation as parameters.
    reset_image_signal : Signal
        Emitted when the image is reset.

    Methods
    -------
    _on_adjust_image():
        Emits the adjust_image_signal with the current x and y offsets and
        rotation.
    _on_reset_image_button_click():
        Resets the x and y offsets and rotation to 0 and emits the
        reset_image_signal.
    _on_scale_image_button_click():
        Emits the scale_image_signal with the entered pixel sizes.
    _on_adjust_atlas_rotation():
        Emits the atlas_rotation_signal with the entered pitch, yaw, and roll.
    _on_atlas_reset():
        Resets the pitch, yaw, and roll to 0 and emits the atlas_reset_signal.
    """

    adjust_image_signal = Signal(int, int, float)
    scale_image_signal = Signal(float, float)
    atlas_rotation_signal = Signal(float, float, float)
    reset_atlas_signal = Signal()
    reset_image_signal = Signal()

    def __init__(self, parent=None):
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

        offset_range = 2000
        rotation_range = 360

        self.adjust_moving_image_pixel_size_x = QDoubleSpinBox(parent=self)
        self.adjust_moving_image_pixel_size_x.setDecimals(3)
        self.adjust_moving_image_pixel_size_x.setRange(0.001, 100.00)
        self.adjust_moving_image_pixel_size_y = QDoubleSpinBox(parent=self)
        self.adjust_moving_image_pixel_size_y.setDecimals(3)
        self.adjust_moving_image_pixel_size_y.setRange(0.001, 100.00)
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

        self.adjust_moving_image_x = QSpinBox(parent=self)
        self.adjust_moving_image_x.setRange(-offset_range, offset_range)
        self.adjust_moving_image_x.valueChanged.connect(self._on_adjust_image)

        self.adjust_moving_image_y = QSpinBox(parent=self)
        self.adjust_moving_image_y.setRange(-offset_range, offset_range)
        self.adjust_moving_image_y.valueChanged.connect(self._on_adjust_image)

        self.adjust_moving_image_rotate = QDoubleSpinBox(parent=self)
        self.adjust_moving_image_rotate.setRange(
            -rotation_range, rotation_range
        )
        self.adjust_moving_image_rotate.setSingleStep(0.5)
        self.adjust_moving_image_rotate.valueChanged.connect(
            self._on_adjust_image
        )

        self.adjust_moving_image_reset_button = QPushButton()
        self.adjust_moving_image_reset_button.setText("Reset Image")
        self.adjust_moving_image_reset_button.clicked.connect(
            self._on_reset_image_button_click
        )

        self.layout().addRow(QLabel("Adjust the moving image scale:"))
        self.layout().addRow(
            "Sample image X pixel size (\u03BCm / pixel):",
            self.adjust_moving_image_pixel_size_x,
        )
        self.layout().addRow(
            "Sample image Y pixel size (\u03BCm / pixel):",
            self.adjust_moving_image_pixel_size_y,
        )
        self.layout().addRow(self.scale_moving_image_button)

        self.layout().addRow(QLabel("Adjust the atlas pitch and yaw: "))
        self.layout().addRow("Pitch:", self.adjust_atlas_pitch)
        self.layout().addRow("Yaw:", self.adjust_atlas_yaw)
        self.layout().addRow("Roll:", self.adjust_atlas_roll)
        self.layout().addRow(self.adjust_atlas_rotation)
        self.layout().addRow(self.reset_atlas_button)

        self.layout().addRow(QLabel("Adjust the moving image position: "))
        self.layout().addRow("X offset:", self.adjust_moving_image_x)
        self.layout().addRow("Y offset:", self.adjust_moving_image_y)
        self.layout().addRow(
            "Rotation (degrees):", self.adjust_moving_image_rotate
        )
        self.layout().addRow(self.adjust_moving_image_reset_button)

    def _on_adjust_image(self):
        """
        Emit the adjust_image_signal with the current x and y offsets and
        rotation.
        """
        self.adjust_image_signal.emit(
            self.adjust_moving_image_x.value(),
            self.adjust_moving_image_y.value(),
            self.adjust_moving_image_rotate.value(),
        )

    def _on_reset_image_button_click(self):
        """
        Reset the x and y offsets and rotation to 0 and emit the
        reset_image_signal.
        """
        self.adjust_moving_image_x.setValue(0)
        self.adjust_moving_image_y.setValue(0)
        self.adjust_moving_image_rotate.setValue(0)

        self.reset_image_signal.emit()

    def _on_scale_image_button_click(self):
        """
        Emit the scale_image_signal with the entered pixel sizes.
        """
        self.scale_image_signal.emit(
            self.adjust_moving_image_pixel_size_x.value(),
            self.adjust_moving_image_pixel_size_y.value(),
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
            "atlas_pitch": self.adjust_atlas_pitch.value(),
            "atlas_yaw": self.adjust_atlas_yaw.value(),
            "atlas_roll": self.adjust_atlas_roll.value(),
            "moving_image_x_offset": self.adjust_moving_image_x.value(),
            "moving_image_y_offset": self.adjust_moving_image_y.value(),
            "moving_image_rotation": self.adjust_moving_image_rotate.value(),
        }
