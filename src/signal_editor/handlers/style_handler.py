import typing as t

import pyqtgraph as pg
import qdarkstyle
from PySide6 import QtCore, QtWidgets


class StyleHandler:
    def __init__(
        self,
        plot_widgets: list[pg.PlotWidget] | None = None,
        style: t.Literal["light", "dark"] = "dark",
    ) -> None:
        self.active_style: t.Literal["light", "dark"] = style
        self.plot_widgets = plot_widgets
        self.app = QtWidgets.QApplication.instance()

    def _set_light_style(self) -> None:
        self.app.setStyleSheet(
            qdarkstyle.load_stylesheet(qt_api="pyside6", palette=qdarkstyle.LightPalette)
        )
        self.active_style = "light"

    def _set_dark_style(self) -> None:
        self.app.setStyleSheet(
            qdarkstyle.load_stylesheet(qt_api="pyside6", palette=qdarkstyle.DarkPalette)
        )

        self.active_style = "dark"

    def set_style(self, style: t.Literal["light", "dark"] | None = None) -> None:
        if style is None:
            style = self.active_style
        if style == "light":
            self._set_light_style()
        elif style == "dark":
            self._set_dark_style()

    @QtCore.Slot(bool)
    def switch_theme(self, checked: bool) -> None:
        if checked:
            self._set_light_style()
        else:
            self._set_dark_style()
