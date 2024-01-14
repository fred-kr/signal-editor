from PySide6.QtWidgets import QHBoxLayout, QPushButton, QWidget


class ConfirmCancelButtons(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_ui()
        self.hide()

    def _init_ui(self) -> None:
        self._confirm_button = QPushButton("Confirm")
        self._cancel_button = QPushButton("Cancel")

        layout = QHBoxLayout()
        layout.addWidget(self._confirm_button)
        layout.addWidget(self._cancel_button)
        self.setLayout(layout)

    @property
    def confirm_button(self) -> QPushButton:
        return self._confirm_button

    @property
    def cancel_button(self) -> QPushButton:
        return self._cancel_button

    