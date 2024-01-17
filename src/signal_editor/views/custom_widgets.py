from PySide6.QtWidgets import QHBoxLayout, QPushButton, QWidget, QApplication
from qtconsole import inprocess
import jupyter_client

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



class JupyterConsoleWidget(inprocess.QtInProcessRichJupyterWidget):
    def __init__(self):
        super().__init__()

        self.kernel_manager: inprocess.QtInProcessKernelManager = inprocess.QtInProcessKernelManager()
        self.kernel_manager.start_kernel()
        self.kernel_client: jupyter_client.blocking.client.BlockingKernelClient = self.kernel_manager.client()
        self.kernel_client.start_channels()
        QApplication.instance().aboutToQuit.connect(self.shutdown_kernel)

    def shutdown_kernel(self):
        self.kernel_client.stop_channels()
        self.kernel_manager.shutdown_kernel()