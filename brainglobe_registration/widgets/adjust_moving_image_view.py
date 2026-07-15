from qtpy.QtCore import Qt, QTimer, Signal
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSlider,
    QWidget,
)


class ThrottledTimer:
    """
    A throttling timer using QTimer.

    Throttling ensures a function is called at most once per interval,
    providing consistent update rate during rapid changes (e.g., slider drag).
    Unlike debouncing which waits for activity to stop, throttling fires
    immediately and then blocks further calls until the interval passes.
    """

    def __init__(self, interval_ms: int, callback, parent=None):
        """
        Initialize the throttled timer.

        Parameters
        ----------
        interval_ms : int
            Minimum interval between callback invocations in milliseconds.
        callback : callable
            Function to call when throttle fires.
        parent : QObject, optional
            Parent for the internal QTimer.
        """
        self._interval = interval_ms
        self._timer = QTimer(parent)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._on_timeout)
        self._callback = callback
        self._is_ready = True
        self._pending = False

    def trigger(self):
        """
        Request a callback invocation.

        If ready, fires immediately and starts cooldown.
        If in cooldown, marks as pending for execution when cooldown ends.
        """
        if self._is_ready:
            self._is_ready = False
            self._callback()
            self._timer.start(self._interval)
        else:
            self._pending = True

    def _on_timeout(self):
        """Handle cooldown timeout."""
        if self._pending:
            self._pending = False
            self._callback()
            self._timer.start(self._interval)
        else:
            self._is_ready = True

    def is_active(self) -> bool:
        """Return True if timer is in cooldown period."""
        return self._timer.isActive()


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
    interpolation_order_changed : Signal
        Emitted when the interpolation order changes. Emits 0 or 1.

    Methods
    -------
    _on_scale_image_button_click():
        Emits the scale_image_signal with the entered pixel sizes.
    _on_adjust_atlas_rotation():
        Emits the atlas_rotation_signal with the entered pitch, yaw, and roll.
    _on_atlas_reset():
        Resets the pitch, yaw, and roll to 0 and emits the atlas_reset_signal.
    """

    scale_image_signal = Signal(float, float, float, str)
    atlas_rotation_signal = Signal(float, float, float)
    reset_atlas_signal = Signal()
    reset_moving_image_signal = Signal()
    interpolation_order_changed = Signal(int)

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
        self.data_orientation_field = QLineEdit(parent=self)
        self.data_orientation_field.setToolTip(
            "Three characters describing the data orientation, "
            'e.g. "psl". See docs for more details.'
        )
        self.scale_moving_image_button = QPushButton()
        self.scale_moving_image_button.setText("Scale Moving Image")
        self.scale_moving_image_button.clicked.connect(
            self._on_scale_image_button_click
        )
        self.reset_moving_image_button = QPushButton()
        self.reset_moving_image_button.setText("Reset Moving Image")
        self.reset_moving_image_button.clicked.connect(
            self._on_moving_image_reset
        )

        # Create slider + spinbox pairs for rotation controls
        # Pitch
        self.adjust_atlas_pitch = QSlider(Qt.Horizontal, parent=self)
        self.adjust_atlas_pitch.setRange(-1800, 1800)  # -180 to 180 in 0.1 deg
        self.adjust_atlas_pitch.setValue(0)
        self.adjust_atlas_pitch.setTickInterval(450)  # Tick every 45 degrees
        self.adjust_atlas_pitch.valueChanged.connect(
            self._on_pitch_slider_changed
        )

        self.pitch_spinbox = QDoubleSpinBox(parent=self)
        self.pitch_spinbox.setRange(-180.0, 180.0)
        self.pitch_spinbox.setDecimals(1)
        self.pitch_spinbox.setSingleStep(0.1)
        self.pitch_spinbox.setValue(0.0)
        self.pitch_spinbox.setSuffix("°")
        self.pitch_spinbox.setFixedWidth(80)
        self.pitch_spinbox.editingFinished.connect(
            self._on_pitch_spinbox_edited
        )

        # Yaw
        self.adjust_atlas_yaw = QSlider(Qt.Horizontal, parent=self)
        self.adjust_atlas_yaw.setRange(-1800, 1800)
        self.adjust_atlas_yaw.setValue(0)
        self.adjust_atlas_yaw.setTickInterval(450)
        self.adjust_atlas_yaw.valueChanged.connect(self._on_yaw_slider_changed)

        self.yaw_spinbox = QDoubleSpinBox(parent=self)
        self.yaw_spinbox.setRange(-180.0, 180.0)
        self.yaw_spinbox.setDecimals(1)
        self.yaw_spinbox.setSingleStep(0.1)
        self.yaw_spinbox.setValue(0.0)
        self.yaw_spinbox.setSuffix("°")
        self.yaw_spinbox.setFixedWidth(80)
        self.yaw_spinbox.editingFinished.connect(self._on_yaw_spinbox_edited)

        # Roll
        self.adjust_atlas_roll = QSlider(Qt.Horizontal, parent=self)
        self.adjust_atlas_roll.setRange(-1800, 1800)
        self.adjust_atlas_roll.setValue(0)
        self.adjust_atlas_roll.setTickInterval(450)
        self.adjust_atlas_roll.valueChanged.connect(
            self._on_roll_slider_changed
        )

        self.roll_spinbox = QDoubleSpinBox(parent=self)
        self.roll_spinbox.setRange(-180.0, 180.0)
        self.roll_spinbox.setDecimals(1)
        self.roll_spinbox.setSingleStep(0.1)
        self.roll_spinbox.setValue(0.0)
        self.roll_spinbox.setSuffix("°")
        self.roll_spinbox.setFixedWidth(80)
        self.roll_spinbox.editingFinished.connect(self._on_roll_spinbox_edited)

        # Throttle timer for live slider updates (5ms)
        self._rotation_throttle_timer = ThrottledTimer(
            5, self._on_adjust_atlas_rotation, parent=self
        )

        # Debounce timer (kept but not used) - waits for activity to stop
        self._rotation_debounce_timer = QTimer(self)
        self._rotation_debounce_timer.setSingleShot(True)
        self._rotation_debounce_timer.setInterval(5)
        self._rotation_debounce_timer.timeout.connect(
            self._on_adjust_atlas_rotation
        )

        self.reset_atlas_button = QPushButton()
        self.reset_atlas_button.setText("Reset Atlas")
        self.reset_atlas_button.clicked.connect(self._on_atlas_reset)

        self.layout().addRow(QLabel("Prepare the moving image:"))
        self.x_row_label = QLabel(
            "Sample image X pixel size (\u03bcm / pixel):"
        )
        self.y_row_label = QLabel(
            "Sample image Y pixel size (\u03bcm / pixel):"
        )
        self.z_row_label = QLabel(
            "Sample image Z pixel size (\u03bcm / pixel):"
        )

        self.layout().addRow(
            self.x_row_label, self.adjust_moving_image_pixel_size_x
        )
        self.layout().addRow(
            self.y_row_label, self.adjust_moving_image_pixel_size_y
        )
        self.layout().addRow(
            self.z_row_label, self.adjust_moving_image_pixel_size_z
        )

        self.orientation_row_label = QLabel("Data orientation:")

        self.layout().addRow(
            self.orientation_row_label, self.data_orientation_field
        )

        self.layout().addRow(self.scale_moving_image_button)
        self.layout().addRow(self.reset_moving_image_button)

        self.set_is_3d(False)

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

        # Create horizontal layouts for slider + spinbox pairs
        pitch_layout = QHBoxLayout()
        pitch_layout.setContentsMargins(0, 0, 0, 0)
        pitch_layout.addWidget(self.adjust_atlas_pitch, stretch=1)
        pitch_layout.addWidget(self.pitch_spinbox)
        pitch_container = QWidget(self)
        pitch_container.setLayout(pitch_layout)
        self.layout().addRow(QLabel("Pitch:"))
        self.layout().addRow(pitch_container)

        yaw_layout = QHBoxLayout()
        yaw_layout.setContentsMargins(0, 0, 0, 0)
        yaw_layout.addWidget(self.adjust_atlas_yaw, stretch=1)
        yaw_layout.addWidget(self.yaw_spinbox)
        yaw_container = QWidget(self)
        yaw_container.setLayout(yaw_layout)
        self.layout().addRow(QLabel("Yaw:"))
        self.layout().addRow(yaw_container)

        roll_layout = QHBoxLayout()
        roll_layout.setContentsMargins(0, 0, 0, 0)
        roll_layout.addWidget(self.adjust_atlas_roll, stretch=1)
        roll_layout.addWidget(self.roll_spinbox)
        roll_container = QWidget(self)
        roll_container.setLayout(roll_layout)
        self.layout().addRow(QLabel("Roll:"))
        self.layout().addRow(roll_container)

        # Interpolation order dropdown (0=Nearest, 1=Linear)
        self.interpolation_order_dropdown = QComboBox(parent=self)
        self.interpolation_order_dropdown.addItem("0", 0)
        self.interpolation_order_dropdown.addItem("1", 1)
        self.interpolation_order_dropdown.setCurrentIndex(
            1
        )  # Default to Linear
        self.interpolation_order_dropdown.currentIndexChanged.connect(
            self._on_interpolation_order_changed
        )
        self.layout().addRow(
            "Interpolation:", self.interpolation_order_dropdown
        )

        self.layout().addRow(self.reset_atlas_button)

    def set_is_3d(self, is_3d: bool):
        """
        Show / hide Z pixel size and orientation based on data dimensionality.
        """
        self.z_row_label.setVisible(is_3d)
        self.adjust_moving_image_pixel_size_z.setVisible(is_3d)
        self.orientation_row_label.setVisible(is_3d)
        self.data_orientation_field.setVisible(is_3d)

    def _on_scale_image_button_click(self):
        """
        Emit the scale_image_signal with the entered pixel sizes.
        """
        self.scale_image_signal.emit(
            self.adjust_moving_image_pixel_size_x.value(),
            self.adjust_moving_image_pixel_size_y.value(),
            self.adjust_moving_image_pixel_size_z.value(),
            self.data_orientation_field.text(),
        )

    def _on_pitch_slider_changed(self, value: int):
        """
        Handle pitch slider value changes.
        Update spinbox and trigger rotation with throttling.
        """
        degrees = value / 10.0
        self.pitch_spinbox.blockSignals(True)
        self.pitch_spinbox.setValue(degrees)
        self.pitch_spinbox.blockSignals(False)
        self._rotation_throttle_timer.trigger()

    def _on_yaw_slider_changed(self, value: int):
        """
        Handle yaw slider value changes.
        Update spinbox and trigger rotation with throttling.
        """
        degrees = value / 10.0
        self.yaw_spinbox.blockSignals(True)
        self.yaw_spinbox.setValue(degrees)
        self.yaw_spinbox.blockSignals(False)
        self._rotation_throttle_timer.trigger()

    def _on_roll_slider_changed(self, value: int):
        """
        Handle roll slider value changes.
        Update spinbox and trigger rotation with throttling.
        """
        degrees = value / 10.0
        self.roll_spinbox.blockSignals(True)
        self.roll_spinbox.setValue(degrees)
        self.roll_spinbox.blockSignals(False)
        self._rotation_throttle_timer.trigger()

    def _on_pitch_spinbox_edited(self):
        """
        Handle pitch spinbox editing finished (Enter or focus lost).
        Update slider and trigger rotation.
        """
        degrees = self.pitch_spinbox.value()
        self.adjust_atlas_pitch.blockSignals(True)
        self.adjust_atlas_pitch.setValue(int(degrees * 10))
        self.adjust_atlas_pitch.blockSignals(False)
        self._on_adjust_atlas_rotation()

    def _on_yaw_spinbox_edited(self):
        """
        Handle yaw spinbox editing finished (Enter or focus lost).
        Update slider and trigger rotation.
        """
        degrees = self.yaw_spinbox.value()
        self.adjust_atlas_yaw.blockSignals(True)
        self.adjust_atlas_yaw.setValue(int(degrees * 10))
        self.adjust_atlas_yaw.blockSignals(False)
        self._on_adjust_atlas_rotation()

    def _on_roll_spinbox_edited(self):
        """
        Handle roll spinbox editing finished (Enter or focus lost).
        Update slider and trigger rotation.
        """
        degrees = self.roll_spinbox.value()
        self.adjust_atlas_roll.blockSignals(True)
        self.adjust_atlas_roll.setValue(int(degrees * 10))
        self.adjust_atlas_roll.blockSignals(False)
        self._on_adjust_atlas_rotation()

    def _on_adjust_atlas_rotation(self):
        """
        Emit the atlas_rotation_signal with the current slider values.
        Slider values are in tenths of degrees, so divide by 10.
        """
        self.atlas_rotation_signal.emit(
            self.adjust_atlas_pitch.value() / 10.0,
            self.adjust_atlas_yaw.value() / 10.0,
            self.adjust_atlas_roll.value() / 10.0,
        )

    def _on_atlas_reset(self):
        """
        Reset the pitch, yaw, and roll to 0 and emit the atlas_reset_signal.
        """
        # Block signals to avoid triggering rotation during reset
        self.adjust_atlas_yaw.blockSignals(True)
        self.adjust_atlas_pitch.blockSignals(True)
        self.adjust_atlas_roll.blockSignals(True)
        self.pitch_spinbox.blockSignals(True)
        self.yaw_spinbox.blockSignals(True)
        self.roll_spinbox.blockSignals(True)

        self.adjust_atlas_yaw.setValue(0)
        self.adjust_atlas_pitch.setValue(0)
        self.adjust_atlas_roll.setValue(0)
        self.pitch_spinbox.setValue(0.0)
        self.yaw_spinbox.setValue(0.0)
        self.roll_spinbox.setValue(0.0)

        self.adjust_atlas_yaw.blockSignals(False)
        self.adjust_atlas_pitch.blockSignals(False)
        self.adjust_atlas_roll.blockSignals(False)
        self.pitch_spinbox.blockSignals(False)
        self.yaw_spinbox.blockSignals(False)
        self.roll_spinbox.blockSignals(False)

        self.reset_atlas_signal.emit()

    def _on_moving_image_reset(self):
        """
        Emit the reset_moving_image_signal to restore moving image data.
        """
        self.reset_moving_image_signal.emit()

    def _on_interpolation_order_changed(self, index: int):
        """
        Emit interpolation_order_changed signal with the selected order.
        """
        order = self.interpolation_order_dropdown.currentData()
        self.interpolation_order_changed.emit(order)

    def get_interpolation_order(self) -> int:
        """Return the currently selected interpolation order (0 or 1)."""
        return self.interpolation_order_dropdown.currentData()

    def set_rotation_values(self, pitch: float, yaw: float, roll: float):
        """
        Set slider and spinbox values programmatically.
        Values are in degrees; converted to tenths internally for sliders.
        """
        # Block signals to avoid triggering live updates
        self.adjust_atlas_pitch.blockSignals(True)
        self.adjust_atlas_yaw.blockSignals(True)
        self.adjust_atlas_roll.blockSignals(True)
        self.pitch_spinbox.blockSignals(True)
        self.yaw_spinbox.blockSignals(True)
        self.roll_spinbox.blockSignals(True)

        self.adjust_atlas_pitch.setValue(int(pitch * 10))
        self.adjust_atlas_yaw.setValue(int(yaw * 10))
        self.adjust_atlas_roll.setValue(int(roll * 10))
        self.pitch_spinbox.setValue(pitch)
        self.yaw_spinbox.setValue(yaw)
        self.roll_spinbox.setValue(roll)

        self.adjust_atlas_pitch.blockSignals(False)
        self.adjust_atlas_yaw.blockSignals(False)
        self.adjust_atlas_roll.blockSignals(False)
        self.pitch_spinbox.blockSignals(False)
        self.yaw_spinbox.blockSignals(False)
        self.roll_spinbox.blockSignals(False)

    def __dict__(self):
        return {
            "pixel_size_x": self.adjust_moving_image_pixel_size_x.value(),
            "pixel_size_y": self.adjust_moving_image_pixel_size_y.value(),
            "pixel_size_z": self.adjust_moving_image_pixel_size_z.value(),
            "atlas_pitch": self.adjust_atlas_pitch.value() / 10.0,
            "atlas_yaw": self.adjust_atlas_yaw.value() / 10.0,
            "atlas_roll": self.adjust_atlas_roll.value() / 10.0,
            "interpolation_order": (
                self.interpolation_order_dropdown.currentData()
            ),
        }
