from typing import TYPE_CHECKING, Literal

import qdarkstyle
from PySide6.QtCore import Slot

if TYPE_CHECKING:
    from ..app import MainWindow


class ThemeSwitcher:
    def __init__(self, main_window: "MainWindow") -> None:
        self.main_window = main_window
        self.active_style = "light"

    def _set_light_style(self) -> None:
        self.main_window.setStyleSheet(
            qdarkstyle.load_stylesheet(
                qt_api="pyside6", palette=qdarkstyle.LightPalette
            )
        )
        self.main_window.plot.set_style("light")
        self.active_style = "light"

    def _set_dark_style(self) -> None:
        self.main_window.setStyleSheet(
            qdarkstyle.load_stylesheet(qt_api="pyside6", palette=qdarkstyle.DarkPalette)
        )
        self.main_window.plot.set_style("dark")
        self.active_style = "dark"

    def set_style(self, style: Literal["light", "dark"]) -> None:
        if style == "light":
            self._set_light_style()
        elif style == "dark":
            self._set_dark_style()

    @Slot()
    def switch_theme(self) -> None:
        if self.active_style == "light":
            self._set_dark_style()
        else:
            self._set_light_style()
