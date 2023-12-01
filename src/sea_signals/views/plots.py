from typing import Any

import numpy as np
import pyqtgraph as pg
from numpy.typing import NDArray
from pyqtgraph.GraphicsScene import mouseEvents
from PySide6 import QtCore
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import QWidget

from ..custom_types import SignalName


class Crosshair:
    def __init__(
        self,
    ):
        self.vline = pg.InfiniteLine(angle=90, movable=False)
        self.hline = pg.InfiniteLine(angle=0, movable=False)
        self.setZValue(100)

    def setPos(self, pos: QtCore.QPointF) -> None:
        self.vline.setPos(pos.x())
        self.hline.setPos(pos.y())

    def setZValue(self, z: int) -> None:
        self.vline.setZValue(z)
        self.hline.setZValue(z)


class PlotManager(QWidget):
    """
    Class that manages showing and updating plots.
    """

    sig_peaks_edited = Signal()
    sig_bpm_updated = Signal(str)

    def __init__(
        self,
        parent: QWidget | None = None,
    ):
        super().__init__(parent=parent)

        plot_bg = str(pg.getConfigOption("background"))
        self.click_tolerance = 60

        self.hbr_plot_widget = pg.PlotWidget(
            background=plot_bg,
            useOpenGL=True,
        )
        self.ventilation_plot_widget = pg.PlotWidget(
            background=plot_bg,
            useOpenGL=True,
        )
        self.bpm_hbr_plot_widget = pg.PlotWidget(
            background=plot_bg,
            useOpenGL=True,
        )
        self.bpm_ventilation_plot_widget = pg.PlotWidget(
            background=plot_bg,
            useOpenGL=True,
        )

        self._prepare_plot_items()

    def _init_plot_items(self) -> None:
        self.hbr_signal_line: pg.PlotDataItem
        self.ventilation_signal_line: pg.PlotDataItem
        self.bpm_hbr_signal_line: pg.PlotDataItem
        self.bpm_ventilation_signal_line: pg.PlotDataItem

        self.hbr_peaks_scatter: pg.ScatterPlotItem
        self.ventilation_peaks_scatter: pg.ScatterPlotItem
        self.bpm_hbr_mean_hline: pg.InfiniteLine
        self.bpm_ventilation_mean_hline: pg.InfiniteLine

    @staticmethod
    def set_plot_titles_and_labels(
        plot_item: pg.PlotItem, title: str, left_label: str, bottom_label: str
    ) -> None:
        plot_item.setTitle(title)
        plot_item.setLabel(axis="left", text=left_label)
        plot_item.setLabel(axis="bottom", text=bottom_label)

    def add_temperature_color_bar(self, start: float, end: float) -> None:
        n_steps = int((end - start) * 10)
        img_item = pg.ImageItem(np.zeros((n_steps, 1)))
        color_bar = pg.ColorBarItem(
            values=(start, end),
            label="Temperature (°C)",
            interactive=True,
            limits=(start, end),
            rounding=0.1,
            orientation="horizontal",
            colorMap=pg.colormap.get("CET-L8"),
        )
        color_bar.setImageItem(img_item, insert_in=self.hbr_plot_widget.getPlotItem())

    def _prepare_plot_items(self) -> None:  # sourcery skip: extract-method
        plot_widgets = [
            (self.hbr_plot_widget, "hbr"),
            (self.bpm_hbr_plot_widget, "bpm_hbr"),
            (self.ventilation_plot_widget, "ventilation"),
            (self.bpm_ventilation_plot_widget, "bpm_ventilation"),
        ]
        for pw, name in plot_widgets:
            plot_item = pw.getPlotItem()
            if plot_item is not None:
                plot_item.showGrid(x=True, y=True)
                plot_item.enableAutoRange(x=False, y=True)
                plot_item.setDownsampling(auto=True)
                plot_item.setClipToView(True)
                plot_item.addLegend(offset=(0, 1), pen=pg.mkPen(color="w"), colCount=2)
                plot_item.register(name)
                plot_item.setAutoVisible(x=False, y=True)
                plot_item.setMouseEnabled(x=True, y=False)

        self.set_plot_titles_and_labels(
            self.hbr_plot_widget.getPlotItem(),
            title="<b>HBR</b>",
            left_label="Signal Amplitude",
            bottom_label="n samples",
        )
        self.set_plot_titles_and_labels(
            self.ventilation_plot_widget.getPlotItem(),
            title="<b>Ventilation</b>",
            left_label="Signal Amplitude",
            bottom_label="n samples",
        )
        self.set_plot_titles_and_labels(
            self.bpm_hbr_plot_widget.getPlotItem(),
            title="<b>Estimated BPM from HBR peaks</b>",
            left_label="beats per minute",
            bottom_label="n samples",
        )
        self.set_plot_titles_and_labels(
            self.bpm_ventilation_plot_widget.getPlotItem(),
            title="<b>Estimated BPM from Ventilation peaks</b>",
            left_label="beats per minute",
            bottom_label="n samples",
        )

        self.hbr_plot_widget.setXLink(
            self.bpm_hbr_plot_widget.getPlotItem().getViewBox()
        )
        self.ventilation_plot_widget.setXLink(
            self.bpm_ventilation_plot_widget.getPlotItem().getViewBox()
        )

        self.hbr_temp_label = pg.TextItem()
        self.hbr_crosshair_vline = pg.InfiniteLine(angle=90, movable=False)
        self.hbr_crosshair_hline = pg.InfiniteLine(angle=0, movable=False)
        self.hbr_crosshair_vline.setZValue(100)
        self.hbr_crosshair_hline.setZValue(100)

        self.ventilation_crosshair_vline = pg.InfiniteLine(angle=90, movable=False)
        self.ventilation_crosshair_hline = pg.InfiniteLine(angle=0, movable=False)
        self.ventilation_crosshair_vline.setZValue(100)
        self.ventilation_crosshair_hline.setZValue(100)

        self.hbr_plot_widget.addItem(self.hbr_crosshair_hline, ignoreBounds=True)
        self.hbr_plot_widget.addItem(self.hbr_crosshair_vline, ignoreBounds=True)
        self.hbr_plot_widget.plotItem.scene().sigMouseMoved.connect(
            lambda pos: self.mouse_moved("hbr", pos)
        )

        self.ventilation_plot_widget.addItem(
            self.ventilation_crosshair_hline, ignoreBounds=True
        )
        self.ventilation_plot_widget.addItem(
            self.ventilation_crosshair_vline, ignoreBounds=True
        )
        self.ventilation_plot_widget.plotItem.scene().sigMouseMoved.connect(
            lambda pos: self.mouse_moved("ventilation", pos)
        )

    @Slot()
    def mouse_moved(self, plot_widget: SignalName, pos: QtCore.QPointF) -> None:
        if plot_widget == "hbr":
            mouse_pos = self.hbr_plot_widget.getViewBox().mapSceneToView(pos)
            self.hbr_crosshair_vline.setPos(mouse_pos.x())
            self.hbr_crosshair_hline.setPos(mouse_pos.y())
        elif plot_widget == "ventilation":
            mouse_pos = self.ventilation_plot_widget.getViewBox().mapSceneToView(pos)
            self.ventilation_crosshair_vline.setPos(mouse_pos.x())
            self.ventilation_crosshair_hline.setPos(mouse_pos.y())

    def clear_all(self) -> None:
        self.hbr_plot_widget.plotItem.clear()
        self.ventilation_plot_widget.plotItem.clear()
        self.bpm_hbr_plot_widget.plotItem.clear()
        self.bpm_ventilation_plot_widget.plotItem.clear()

    @Slot()
    def reset_plots(self) -> None:
        if hasattr(self, "hbr_signal_line"):
            self.hbr_plot_widget.removeItem(self.hbr_signal_line)
        if hasattr(self, "ventilation_signal_line"):
            self.ventilation_plot_widget.removeItem(self.ventilation_signal_line)
        if hasattr(self, "hbr_peaks_scatter"):
            self.hbr_plot_widget.removeItem(self.hbr_peaks_scatter)
        if hasattr(self, "ventilation_peaks_scatter"):
            self.ventilation_plot_widget.removeItem(self.ventilation_peaks_scatter)
        if hasattr(self, "bpm_hbr_signal_line"):
            self.bpm_hbr_plot_widget.removeItem(self.bpm_hbr_signal_line)
        if hasattr(self, "bpm_hbr_mean_hline"):
            self.bpm_hbr_plot_widget.removeItem(self.bpm_hbr_mean_hline)
        if hasattr(self, "bpm_ventilation_signal_line"):
            self.bpm_ventilation_plot_widget.removeItem(
                self.bpm_ventilation_signal_line
            )
        if hasattr(self, "bpm_ventilation_mean_hline"):
            self.bpm_ventilation_plot_widget.removeItem(self.bpm_ventilation_mean_hline)

    def draw_signal(
        self,
        sig: NDArray[np.float32 | np.float64],
        plot_widget: pg.PlotWidget,
        signal_name: str,
    ) -> None:
        color = "crimson" if signal_name == "hbr" else "royalblue"
        signal_line = pg.PlotDataItem(
            sig,
            pen=pg.mkPen(color=color, width=1),  # type: ignore
            skipFiniteCheck=True,
            autoDownSample=True,
            name=f"{signal_name}_signal",
        )
        signal_line.curve.setSegmentedLineMode("on")
        signal_line.curve.setClickable(True, width=self.click_tolerance)

        if hasattr(self, f"{signal_name}_signal_line"):
            getattr(self, f"{signal_name}_signal_line").sigClicked.disconnect(
                self.add_clicked_point
            )
            plot_widget.removeItem(getattr(self, f"{signal_name}_signal_line"))
        if hasattr(self, f"{signal_name}_peaks_scatter"):
            plot_widget.removeItem(getattr(self, f"{signal_name}_peaks_scatter"))

        plot_widget.addItem(signal_line)
        setattr(self, f"{signal_name}_signal_line", signal_line)
        getattr(self, f"{signal_name}_signal_line").sigClicked.connect(
            self.add_clicked_point
        )

    def draw_peaks(
        self,
        pos_x: NDArray[np.int32],
        pos_y: NDArray[np.float32 | np.float64],
        plot_widget: pg.PlotWidget,
        signal_name: str,
    ) -> None:
        color = "goldenrod"
        peaks_scatter = pg.ScatterPlotItem(
            x=pos_x,
            y=pos_y,
            pxMode=True,
            size=10,
            pen=None,
            brush=pg.mkBrush(color=color),
            useCache=True,
            name=f"{signal_name}_peaks",
            hoverable=True,
            hoverPen=pg.mkPen(color="gray", width=1),  # type: ignore
            hoverSymbol="x",
            hoverBrush=pg.mkBrush(color="red"),
            hoverSize=12,
            tip=None,
        )
        peaks_scatter.setZValue(20)

        if hasattr(self, f"{signal_name}_peaks_scatter"):
            getattr(self, f"{signal_name}_peaks_scatter").sigClicked.disconnect(
                self.remove_clicked_point
            )
            plot_widget.removeItem(getattr(self, f"{signal_name}_peaks_scatter"))

        plot_widget.addItem(peaks_scatter)

        setattr(self, f"{signal_name}_peaks_scatter", peaks_scatter)
        getattr(self, f"{signal_name}_peaks_scatter").sigClicked.connect(
            self.remove_clicked_point
        )

    def draw_bpm(
        self,
        bpm_data: NDArray[np.float32 | np.float64],
        plot_widget: pg.PlotWidget,
        signal_name: str,
    ) -> None:
        mean_bpm = np.mean(bpm_data)
        color = "lightgreen"

        bpm_line = pg.PlotDataItem(
            bpm_data,
            pen=pg.mkPen(color=color, width=1),  # type: ignore
            autoDownsample=True,
            skipFiniteCheck=True,
            name=f"bpm_{signal_name}",
        )
        bpm_mean_line = pg.InfiniteLine(
            pos=mean_bpm,
            angle=0,
            pen=pg.mkPen(color="orange", width=2, style=Qt.PenStyle.DashLine),  # type: ignore
            name=f"mean_bpm_{signal_name}",
        )

        if hasattr(self, f"bpm_{signal_name}_signal_line") and hasattr(
            self, f"bpm_{signal_name}_mean_hline"
        ):
            plot_widget.removeItem(getattr(self, f"bpm_{signal_name}_signal_line"))
            plot_widget.removeItem(getattr(self, f"bpm_{signal_name}_mean_hline"))

        plot_widget.addItem(bpm_line)
        plot_widget.addItem(bpm_mean_line)

        setattr(self, f"bpm_{signal_name}_signal_line", bpm_line)
        setattr(self, f"bpm_{signal_name}_mean_hline", bpm_mean_line)

        self.sig_bpm_updated.emit(signal_name)

    @Slot(object, object, object)
    def remove_clicked_point(
        self,
        sender: pg.ScatterPlotItem,
        points: np.ndarray[pg.SpotItem, Any],
        ev: mouseEvents.MouseClickEvent,
    ) -> None:
        if points.size > 0:
            to_remove_index = points[0].index()
            if sender.name() == "hbr_peaks":
                new_points_x = np.delete(
                    self.hbr_peaks_scatter.data["x"], to_remove_index
                )
                new_points_y = np.delete(
                    self.hbr_peaks_scatter.data["y"], to_remove_index
                )
                self.hbr_peaks_scatter.setData(x=new_points_x, y=new_points_y)
            elif sender.name() == "ventilation_peaks":
                new_points_x = np.delete(
                    self.ventilation_peaks_scatter.data["x"], to_remove_index
                )
                new_points_y = np.delete(
                    self.ventilation_peaks_scatter.data["y"], to_remove_index
                )
                self.ventilation_peaks_scatter.setData(x=new_points_x, y=new_points_y)
            self.sig_peaks_edited.emit()

    @Slot(object, object)
    def add_clicked_point(
        self, sender: pg.PlotCurveItem, ev: mouseEvents.MouseClickEvent
    ) -> None:
        x_new = np.array(ev.pos().x(), dtype=np.int32).flatten()
        if sender.name() == "hbr_signal":
            assert self.hbr_signal_line.yData is not None
            y_new = np.array(
                self.hbr_signal_line.yData[x_new], dtype=np.float64
            ).flatten()
            self.hbr_peaks_scatter.addPoints(x=x_new, y=y_new)
        elif sender.name() == "ventilation_signal":
            assert self.ventilation_signal_line.yData is not None
            y_new = np.array(
                self.ventilation_signal_line.yData[x_new], dtype=np.float64
            ).flatten()
            self.ventilation_peaks_scatter.addPoints(x=x_new, y=y_new)

        self.sig_peaks_edited.emit()
