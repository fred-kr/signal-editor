from typing import Literal

import pyqtgraph as pg
import qdarkstyle
from PySide6.QtCore import Slot
from PySide6.QtWidgets import QApplication


class ThemeSwitcher:
    def __init__(
        self,
        plot_widgets: list[pg.PlotWidget] | None = None,
        style: Literal["light", "dark"] = "dark",
    ) -> None:
        self.active_style: Literal["light", "dark"] = style
        self.plot_widgets = plot_widgets
        self.app = QApplication.instance()

    def _set_light_style(self) -> None:
        self.app.setStyleSheet(
            qdarkstyle.load_stylesheet(
                qt_api="pyside6", palette=qdarkstyle.LightPalette
            )
        )
        self.active_style = "light"

    def _set_dark_style(self) -> None:
        self.app.setStyleSheet(
            qdarkstyle.load_stylesheet(qt_api="pyside6", palette=qdarkstyle.DarkPalette)
        )

        self.active_style = "dark"

    def set_style(self, style: Literal["light", "dark"] | None = None) -> None:
        if style is None:
            style = self.active_style
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
