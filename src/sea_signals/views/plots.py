from typing import Any

import numpy as np
import pyqtgraph as pg
from loguru import logger
from numpy.typing import NDArray
from pyqtgraph.GraphicsScene import mouseEvents
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import QWidget


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
        self.added_points: dict[str, list[int]] = {"hbr": [], "ventilation": []}
        self.removed_points: dict[str, list[int]] = {"hbr": [], "ventilation": []}

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
        self.hbr_signal_line: pg.PlotDataItem | None = None
        self.ventilation_signal_line: pg.PlotDataItem | None = None
        self.bpm_hbr_signal_line: pg.PlotDataItem | None = None
        self.bpm_ventilation_signal_line: pg.PlotDataItem | None = None

        self.hbr_peaks_scatter: pg.ScatterPlotItem | None = None
        self.ventilation_peaks_scatter: pg.ScatterPlotItem | None = None
        self.bpm_hbr_mean_hline: pg.InfiniteLine | None = None
        self.bpm_ventilation_mean_hline: pg.InfiniteLine | None = None

    @staticmethod
    def set_plot_titles_and_labels(
        plot_item: pg.PlotItem, title: str, left_label: str, bottom_label: str
    ) -> None:
        plot_item.setTitle(title)
        plot_item.setLabel(axis="left", text=left_label)
        plot_item.setLabel(axis="bottom", text=bottom_label)

    def _prepare_plot_items(self) -> None:  # sourcery skip: extract-method
        self._init_plot_items()
        plot_widgets = [
            (self.hbr_plot_widget, "hbr"),
            (self.bpm_hbr_plot_widget, "bpm_hbr"),
            (self.ventilation_plot_widget, "ventilation"),
            (self.bpm_ventilation_plot_widget, "bpm_ventilation"),
        ]
        for pw, name in plot_widgets:
            plot_item = pw.getPlotItem()
            plot_item.showGrid(x=True, y=True)
            plot_item.enableAutoRange(y=True, enable=0.95)
            plot_item.setDownsampling(auto=True)
            plot_item.setClipToView(True)
            plot_item.addLegend(
                offset=(0.5, 1),
                pen=pg.mkPen(color="w"),  # type: ignore
                brush=pg.mkBrush(color="k"),  # type: ignore
                colCount=2,
            )
            plot_item.register(name)
            plot_item.setAutoVisible(y=True)
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

        self.hbr_plot_widget.setXLink("bpm_hbr")
        self.ventilation_plot_widget.setXLink("bpm_ventilation")

        self.hbr_plot_widget.plotItem.getViewBox().setCursor(Qt.CursorShape.CrossCursor)  # type: ignore
        self.ventilation_plot_widget.plotItem.getViewBox().setCursor(  # type: ignore
            Qt.CursorShape.CrossCursor
        )

    @Slot()
    def reset_plots(self) -> None:
        self.hbr_plot_widget.getPlotItem().clear()
        self.ventilation_plot_widget.getPlotItem().clear()
        self.bpm_hbr_plot_widget.getPlotItem().clear()
        self.bpm_ventilation_plot_widget.getPlotItem().clear()
        self._prepare_plot_items()
        # if hasattr(self, "hbr_signal_line"):
        #     self.hbr_plot_widget.removeItem(self.hbr_signal_line)
        # if hasattr(self, "ventilation_signal_line"):
        #     self.ventilation_plot_widget.removeItem(self.ventilation_signal_line)
        # if hasattr(self, "hbr_peaks_scatter"):
        #     self.hbr_plot_widget.removeItem(self.hbr_peaks_scatter)
        # if hasattr(self, "ventilation_peaks_scatter"):
        #     self.ventilation_plot_widget.removeItem(self.ventilation_peaks_scatter)
        # if hasattr(self, "bpm_hbr_signal_line"):
        #     self.bpm_hbr_plot_widget.removeItem(self.bpm_hbr_signal_line)
        # if hasattr(self, "bpm_hbr_mean_hline"):
        #     self.bpm_hbr_plot_widget.removeItem(self.bpm_hbr_mean_hline)
        # if hasattr(self, "bpm_ventilation_signal_line"):
        #     self.bpm_ventilation_plot_widget.removeItem(
        #         self.bpm_ventilation_signal_line
        #     )
        # if hasattr(self, "bpm_ventilation_mean_hline"):
        #     self.bpm_ventilation_plot_widget.removeItem(self.bpm_ventilation_mean_hline)

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

        bpm_plot_widget = getattr(self, f"bpm_{signal_name}_plot_widget")

        line_ref = getattr(self, f"{signal_name}_signal_line")
        scatter_ref = getattr(self, f"{signal_name}_peaks_scatter")
        bpm_line_ref = getattr(self, f"bpm_{signal_name}_signal_line")
        bpm_mean_line_ref = getattr(self, f"bpm_{signal_name}_mean_hline")
        if line_ref is not None:
            line_ref.sigClicked.disconnect(self.add_clicked_point)
            plot_widget.removeItem(line_ref)
        if scatter_ref is not None:
            # scatter_ref.sigClicked.disconnect(self.remove_clicked_point)
            plot_widget.removeItem(scatter_ref)
        if bpm_line_ref is not None:
            bpm_plot_widget.removeItem(bpm_line_ref)
        if bpm_mean_line_ref is not None:
            bpm_plot_widget.removeItem(bpm_mean_line_ref)

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
            brush=pg.mkBrush(color=color),  # type: ignore
            useCache=True,
            name=f"{signal_name}_peaks",
            hoverable=True,
            hoverPen=pg.mkPen(color="k", width=1),  # type: ignore
            hoverSymbol="x",
            hoverBrush=pg.mkBrush(color="red"),  # type: ignore
            hoverSize=14,
            tip=None,
        )
        peaks_scatter.setZValue(20)

        scatter_ref = getattr(self, f"{signal_name}_peaks_scatter")

        if scatter_ref is not None:
            scatter_ref.sigClicked.disconnect(self.remove_clicked_point)
            plot_widget.removeItem(scatter_ref)

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
            label=f"Mean BPM: {int(mean_bpm)}",
        )

        bpm_line_ref = getattr(self, f"bpm_{signal_name}_signal_line")
        bpm_mean_line_ref = getattr(self, f"bpm_{signal_name}_mean_hline")
        if bpm_line_ref is not None and bpm_mean_line_ref is not None:
            plot_widget.removeItem(bpm_line_ref)
            plot_widget.removeItem(bpm_mean_line_ref)

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
        ev.accept()
        if points.size <= 0:
            return
        to_remove_index = points[0].index()
        scatter_plots = {
            "hbr_peaks": getattr(self, "hbr_peaks_scatter", None),
            "ventilation_peaks": getattr(self, "ventilation_peaks_scatter", None),
        }
        if scatter_plot := scatter_plots.get(sender.name()):
            new_points_x = np.delete(scatter_plot.data["x"], to_remove_index)
            new_points_y = np.delete(scatter_plot.data["y"], to_remove_index)
            scatter_plot.setData(x=new_points_x, y=new_points_y)
            self.sig_peaks_edited.emit()
            name = "hbr" if "hbr" in sender.name() else "ventilation"
            self.removed_points[name].append(int(points[0].pos().x()))

    @Slot(object, object)
    def add_clicked_point(
        self, sender: pg.PlotCurveItem, ev: mouseEvents.MouseClickEvent
    ) -> None:
        ev.accept()
        x_new = int(ev.pos().x())
        signal_map = {
            "hbr_signal": getattr(self, "hbr_signal_line", None),
            "ventilation_signal": getattr(self, "ventilation_signal_line", None),
        }
        scatter_map = {
            "hbr_signal": getattr(self, "hbr_peaks_scatter", None),
            "ventilation_signal": getattr(self, "ventilation_peaks_scatter", None),
        }
        signal_name = sender.name()
        if signal_name in signal_map:
            y_new = signal_map[signal_name].yData[x_new]
            scatter_map[signal_name].addPoints(x=x_new, y=y_new)
            # scatter_map[signal_name].addPoints(x_new, y_new)
            self.sig_peaks_edited.emit()
            name = "hbr" if "hbr" in signal_name else "ventilation"
            logger.debug(f"{x_new=}, {y_new=}, {self.added_points=}")
            self.added_points[name].append(x_new)
