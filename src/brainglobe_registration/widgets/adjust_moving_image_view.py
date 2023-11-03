from qtpy.QtCore import Signal

from qtpy.QtWidgets import (
    QGroupBox,
    QFormLayout,
    QSpinBox,
    QDoubleSpinBox,
    QPushButton,
    QHBoxLayout,
    QWidget,
    QLabel,
)


class AdjustMovingImageView(QWidget):
    adjust_image_signal = Signal(int, int, float)
    reset_image_signal = Signal()

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self.setLayout(QFormLayout())

        min_offset_range = -2000
        max_offset_range = 2000

        self.adjust_moving_image_x = QSpinBox()
        self.adjust_moving_image_x.setRange(min_offset_range, max_offset_range)

        self.adjust_moving_image_y = QSpinBox()
        self.adjust_moving_image_y.setRange(min_offset_range, max_offset_range)

        self.adjust_moving_image_rotate = QDoubleSpinBox()
        self.adjust_moving_image_rotate.setRange(-360, 360)
        self.adjust_moving_image_rotate.setSingleStep(0.5)

        self.adjust_moving_image_buttons = QGroupBox(parent=self)
        self.adjust_moving_image_reset_button = QPushButton(
            parent=self.adjust_moving_image_buttons
        )
        self.adjust_moving_image_reset_button.setText("Reset Image")
        self.adjust_moving_image_reset_button.clicked.connect(
            self._on_reset_image_button_click
        )

        self.adjust_moving_image_button = QPushButton(
            parent=self.adjust_moving_image_buttons
        )
        self.adjust_moving_image_button.setText("Adjust Image")
        self.adjust_moving_image_button.clicked.connect(
            self._on_adjust_image_button_click
        )

        self.adjust_moving_image_buttons.setLayout(QHBoxLayout())
        self.adjust_moving_image_buttons.layout().addWidget(
            self.adjust_moving_image_button
        )
        self.adjust_moving_image_buttons.layout().addWidget(
            self.adjust_moving_image_reset_button
        )

        self.layout().addRow(QLabel("Adjust the moving image: "))
        self.layout().addRow("X offset:", self.adjust_moving_image_x)
        self.layout().addRow("Y offset:", self.adjust_moving_image_y)
        self.layout().addRow(
            "Rotation (degrees):", self.adjust_moving_image_rotate
        )
        self.layout().addRow(self.adjust_moving_image_buttons)

    def _on_adjust_image_button_click(self):
        self.adjust_image_signal.emit(
            self.adjust_moving_image_x.value(),
            self.adjust_moving_image_y.value(),
            self.adjust_moving_image_rotate.value(),
        )

    def _on_reset_image_button_click(self):
        self.adjust_moving_image_x.setValue(0)
        self.adjust_moving_image_y.setValue(0)
        self.adjust_moving_image_rotate.setValue(0)

        self.reset_image_signal.emit()
