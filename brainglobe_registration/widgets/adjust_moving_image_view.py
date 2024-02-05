from qtpy.QtCore import Signal
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
    """

    adjust_image_signal = Signal(int, int, float)
    scale_image_signal = Signal(float, float)
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

        offset_range = 2000
        rotation_range = 360

        self.adjust_moving_image_voxel_size_x = QDoubleSpinBox(parent=self)
        self.adjust_moving_image_voxel_size_x.setDecimals(2)
        self.adjust_moving_image_voxel_size_x.setMinimum(0.01)
        self.adjust_moving_image_voxel_size_y = QDoubleSpinBox(parent=self)
        self.adjust_moving_image_voxel_size_y.setDecimals(2)
        self.adjust_moving_image_voxel_size_y.setMinimum(0.01)
        self.scale_moving_image_button = QPushButton()
        self.scale_moving_image_button.setText("Scale Image")
        self.scale_moving_image_button.clicked.connect(
            self._on_scale_image_button_click
        )

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
            "Sample image X pixel size:", self.adjust_moving_image_voxel_size_x
        )
        self.layout().addRow(
            "Sample image Y pixel size:", self.adjust_moving_image_voxel_size_y
        )
        self.layout().addRow(self.scale_moving_image_button)

        self.layout().addRow(QLabel("Adjust the sample image position:"))
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
            self.adjust_moving_image_voxel_size_x.value(),
            self.adjust_moving_image_voxel_size_y.value(),
        )
