from typing import Any

import numpy as np
import pyqtgraph as pg
from loguru import logger
from numpy.typing import NDArray
from pyqtgraph.GraphicsScene import mouseEvents
from PySide6.QtCore import QPointF, Qt, Signal, Slot
from PySide6.QtWidgets import QWidget

from ..custom_types import SignalName


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

    @staticmethod
    def target_pos(x: float, y: float) -> str:
        x = max(x, 0)
        time_seconds = x / 400
        hours = time_seconds // 3600
        minutes = (time_seconds % 3600) // 60
        seconds = time_seconds % 60
        return f"time: {int(hours):02}:{int(minutes):02}:{int(seconds):02}\namplitude: {y:.4f}"

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
            plot_item.showGrid(x=False, y=True)
            plot_item.getViewBox().enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
            plot_item.setDownsampling(auto=True)
            plot_item.setClipToView(True)
            plot_item.addLegend(
                offset=(0, 1),
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
            title="<b>Estimated Rate</b>",
            left_label="beats per minute",
            bottom_label="n samples",
        )
        self.set_plot_titles_and_labels(
            self.bpm_ventilation_plot_widget.getPlotItem(),
            title="<b>Estimated Rate</b>",
            left_label="beats per minute",
            bottom_label="n samples",
        )

        self.hbr_plot_widget.setXLink("bpm_hbr")
        self.ventilation_plot_widget.setXLink("bpm_ventilation")

        self.hbr_target = pg.TargetItem(
            pos=(0, 0),
            size=18,
            label=self.target_pos,
            labelOpts={"fill": (0, 0, 0, 120)},
            movable=False,
        )
        self.ventilation_target = pg.TargetItem(
            pos=(0, 0), size=18, label=self.target_pos, labelOpts={"fill": (0, 0, 0, 120)}, movable=False
        )

        self.hbr_target.setZValue(65)
        self.ventilation_target.setZValue(65)

        self.hbr_plot_widget.addItem(self.hbr_target)
        self.ventilation_plot_widget.addItem(self.ventilation_target)

        self.hbr_plot_widget.plotItem.scene().sigMouseMoved.connect(
            lambda pos: self.update_target_pos("hbr", pos)
        )
        self.ventilation_plot_widget.plotItem.scene().sigMouseMoved.connect(
            lambda pos: self.update_target_pos("ventilation", pos)
        )

        self.hbr_plot_widget.plotItem.getViewBox().setCursor(Qt.CursorShape.BlankCursor)  # type: ignore
        self.ventilation_plot_widget.plotItem.getViewBox().setCursor(  # type: ignore
            Qt.CursorShape.BlankCursor
        )

    @Slot(QPointF)
    def update_target_pos(self, signal_name: SignalName, pos: QPointF) -> None:
        scene_pos = getattr(
            self, f"{signal_name}_plot_widget"
        ).plotItem.vb.mapSceneToView(pos)
        getattr(self, f"{signal_name}_target").setPos(scene_pos)

    @Slot()
    def reset_plots(self) -> None:
        plot_widgets = [
            "hbr_plot_widget",
            "ventilation_plot_widget",
            "bpm_hbr_plot_widget",
            "bpm_ventilation_plot_widget",
        ]
        for pw in plot_widgets:
            getattr(self, pw).getPlotItem().clear()
            getattr(self, pw).getPlotItem().legend.clear()
        self._prepare_plot_items()

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
            hoverSize=15,
            tip=None,
        )
        peaks_scatter.setZValue(60)

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
        color = "limegreen"

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
            pen=pg.mkPen(color="yellow", width=2, style=Qt.PenStyle.DashLine),  # type: ignore
            name=f"mean_bpm_{signal_name}",
        )

        bpm_line_ref = getattr(self, f"bpm_{signal_name}_signal_line")
        bpm_mean_line_ref = getattr(self, f"bpm_{signal_name}_mean_hline")

        if bpm_line_ref is not None and bpm_mean_line_ref is not None:
            plot_widget.removeItem(bpm_line_ref)
            plot_widget.removeItem(bpm_mean_line_ref)
            plot_widget.getPlotItem().legend.clear()

        plot_widget.addItem(bpm_line)
        plot_widget.addItem(bpm_mean_line)
        plot_widget.getPlotItem().legend.addItem(
            pg.PlotDataItem(
                np.array([0, 1], dtype=np.float32),
                pen=pg.mkPen(color="yellow", width=2, style=Qt.PenStyle.DotLine),
                skipFiniteCheck=True,
            ),
            f"Mean BPM: {int(mean_bpm)}",
        )

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
            self.sig_peaks_edited.emit()
            name = "hbr" if "hbr" in signal_name else "ventilation"
            self.added_points[name].append(x_new)
