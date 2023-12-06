from PySide6.QtWidgets import QDialog


class MessageHandler(QDialog):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)
