from typing import TYPE_CHECKING, Any, override

import numpy as np
import pyqtgraph as pg
from numpy.typing import NDArray
from pyqtgraph.GraphicsScene import mouseEvents
from PySide6 import QtGui, QtWidgets
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, Signal, Slot

from ..type_aliases import AddedPoints, OldPeaks, PeakEdits, RemovedPoints, SignalName

if TYPE_CHECKING:
    from ..app import MainWindow


class CustomViewBox(pg.ViewBox):
    sig_selection_changed = Signal(QtGui.QPolygonF)

    def __init__(self, *args: Any, **kargs: Any) -> None:
        pg.ViewBox.__init__(self, *args, **kargs)
        self._selection_box: QtWidgets.QGraphicsRectItem | None = None
        self.mapped_peak_selection: QtGui.QPolygonF | None = None

    @property
    def selection_box(self) -> QtWidgets.QGraphicsRectItem:
        if self._selection_box is None:
            selection_box = QtWidgets.QGraphicsRectItem(0, 0, 1, 1)
            selection_box.setPen(pg.mkPen((100, 100, 255), width=1))
            selection_box.setBrush(pg.mkBrush((100, 100, 255, 100)))
            selection_box.setZValue(1e9)
            selection_box.hide()
            self._selection_box = selection_box
            self.addItem(selection_box, ignoreBounds=True)
        return self._selection_box

    @selection_box.setter
    def selection_box(self, selection_box: QtWidgets.QGraphicsRectItem | None) -> None:
        if self._selection_box is not None:
            self.removeItem(self._selection_box)
        self._selection_box = selection_box
        if selection_box is None:
            return None
        selection_box.setZValue(1e9)
        selection_box.hide()
        self.addItem(selection_box, ignoreBounds=True)
        return None

    @override
    def mouseDragEvent(
        self, ev: mouseEvents.MouseDragEvent, axis: int | float | None = None
    ) -> None:
        ev.accept()

        pos = ev.pos()
        lastPos = ev.lastPos()
        dif = pos - lastPos
        dif = dif * -1


        mouseEnabled = np.array(self.state["mouseEnabled"], dtype=np.float64)
        mask = mouseEnabled.copy()
        if axis is not None:
            mask[1 - axis] = 0.0

        if ev.button() == Qt.MouseButton.MiddleButton or (ev.button() == Qt.MouseButton.LeftButton and ev.modifiers() & Qt.KeyboardModifier.ControlModifier):
            if ev.isFinish():
                r = QRectF(ev.pos(), ev.buttonDownPos())
                data_coords: QtGui.QPolygonF = self.mapToView(r)
                self.mapped_peak_selection = data_coords
            else:
                self.updateSelectionBox(ev.pos(), ev.buttonDownPos())
                self.mapped_peak_selection = None
        elif ev.button() == Qt.MouseButton.LeftButton:
            if self.state["mouseMode"] == pg.ViewBox.RectMode and axis is None:
                if ev.isFinish():
                    self.rbScaleBox.hide()
                    ax = QRectF(pg.Point(ev.buttonDownPos(ev.button())), pg.Point(pos))
                    ax = self.childGroup.mapRectFromParent(ax)
                    self.showAxRect(ax)
                    self.axHistoryPointer += 1
                    self.axHistory = self.axHistory[: self.axHistoryPointer] + [ax]
                else:
                    self.updateScaleBox(ev.buttonDownPos(), ev.pos())
            else:
                tr = pg.invertQTransform(self.childGroup.transform())
                tr = tr.map(dif * mask) - tr.map(pg.Point(0, 0))

                x = tr.x() if mask[0] == 1 else None
                y = tr.y() if mask[1] == 1 else None

                self._resetTarget()
                if x is not None or y is not None:
                    self.translateBy(x=x, y=y)
                self.sigRangeChangedManually.emit(self.state["mouseEnabled"])
        elif ev.button() & Qt.MouseButton.RightButton:
            if self.state["aspectLocked"] is not False:
                mask[0] = 0

            dif = np.array(
                [
                    -(ev.screenPos().x() - ev.lastScreenPos().x()),
                    ev.screenPos().y() - ev.lastScreenPos().y(),
                ]
            )
            s = ((mask * 0.02) + 1) ** dif

            tr = pg.invertQTransform(self.childGroup.transform())

            x = s[0] if mouseEnabled[0] == 1 else None
            y = s[1] if mouseEnabled[1] == 1 else None

            center = pg.Point(tr.map(ev.buttonDownPos(Qt.MouseButton.RightButton)))
            self._resetTarget()
            self.scaleBy(x=x, y=y, center=center)
            self.sigRangeChangedManually.emit(self.state["mouseEnabled"])

    def updateSelectionBox(self, pos1: pg.Point, pos2: pg.Point) -> None:
        rect = QRectF(pos1, pos2)
        rect = self.childGroup.mapRectFromParent(rect)
        self.selection_box.setPos(rect.topLeft())
        tr = QtGui.QTransform.fromScale(rect.width(), rect.height())
        self.selection_box.setTransform(tr)
        self.selection_box.show()


class PlotHandler(QObject):
    """
    Class that manages showing and updating plots.
    """

    sig_bpm_updated = Signal(str)
    sig_peaks_edited = Signal(str)

    def __init__(
        self,
        parent: "MainWindow",
    ):
        super().__init__(parent=parent)
        self._parent = parent
        plot_bg = str(pg.getConfigOption("background"))
        self.click_tolerance = 80
        self.added_points: AddedPoints = AddedPoints(hbr=[], ventilation=[])
        self.removed_points: RemovedPoints = RemovedPoints(hbr=[], ventilation=[])
        self.peak_edits: PeakEdits = PeakEdits(
            added_peaks=AddedPoints(hbr=[], ventilation=[]),
            removed_peaks=RemovedPoints(hbr=[], ventilation=[]),
        )
        self._old_peaks: OldPeaks = {}

        self.hbr_plot_widget = pg.PlotWidget(
            viewBox=CustomViewBox(),
            background=plot_bg,
            useOpenGL=True,
        )
        self.ventilation_plot_widget = pg.PlotWidget(
            viewBox=CustomViewBox(),
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

        self.hbr_linear_region: pg.LinearRegionItem | None = None
        self.ventilation_linear_region: pg.LinearRegionItem | None = None

    @staticmethod
    def set_plot_titles_and_labels(
        plot_item: pg.PlotItem, title: str, left_label: str, bottom_label: str
    ) -> None:
        plot_item.setTitle(title)
        plot_item.setLabel(axis="left", text=left_label)
        plot_item.setLabel(axis="bottom", text=bottom_label)

    def _prepare_plot_items(self) -> None:
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
            plot_item.getViewBox().enableAutoRange("y")
            plot_item.setDownsampling(auto=True)
            plot_item.setClipToView(True)
            plot_item.addLegend(
                pen="white",
                brush="transparent",
                colCount=2,
            )
            plot_item.register(name)
            plot_item.getViewBox().setAutoVisible(y=True)
            plot_item.setMouseEnabled(x=True, y=False)

        # common_label = "<span style='color: lightgray; font-size: 10pt;'>n samples</span>"
        # common_amp_label = (
        #     "<span style='color: lightgray; font-size: 10pt;'>Signal Amplitude</span>"
        # )
        # common_rate_label = (
        #     "<span style='color: lightgray; font-size: 10pt;'>Signal Rate</span>"
        # )

        self.set_plot_titles_and_labels(
            self.hbr_plot_widget.getPlotItem(),
            title="<span style='color: white; font-size: 12pt; font-weight: 500'>HBR</span>",
            left_label="<span style='color: lightgray; font-size: 10pt;'>Signal Amplitude</span>",
            bottom_label="<span style='color: lightgray; font-size: 10pt;'>n samples</span>",
        )
        self.set_plot_titles_and_labels(
            self.ventilation_plot_widget.getPlotItem(),
            title="<span style='color: white; font-size: 12pt; font-weight: 500'>Ventilation</span>",
            left_label="<span style='color: lightgray; font-size: 10pt;'>Signal Amplitude</span>",
            bottom_label="<span style='color: lightgray; font-size: 10pt;'>n samples</span>",
        )
        self.set_plot_titles_and_labels(
            self.bpm_hbr_plot_widget.getPlotItem(),
            title="<span style='color: white; font-size: 12pt; font-weight: 500'>Estimated Rate</span>",
            left_label="<span style='color: lightgray; font-size: 10pt;'>Cycles per minute</span>",
            bottom_label="<span style='color: lightgray; font-size: 10pt;'>n samples</span>",
        )
        self.set_plot_titles_and_labels(
            self.bpm_ventilation_plot_widget.getPlotItem(),
            title="<span style='color: white; font-size: 12pt; font-weight: 500'>Estimated Rate</span>",
            left_label="<span style='color: lightgray; font-size: 10pt;'>Cycles per minute</span>",
            bottom_label="<span style='color: lightgray; font-size: 10pt;'>n samples</span>",
        )

        self.hbr_plot_widget.getPlotItem().getViewBox().setXLink("bpm_hbr")
        self.ventilation_plot_widget.getPlotItem().getViewBox().setXLink(
            "bpm_ventilation"
        )

        for bpm_pw in [self.bpm_hbr_plot_widget, self.bpm_ventilation_plot_widget]:
            bpm_pw.getPlotItem().getViewBox().setMouseMode(pg.ViewBox.RectMode)

    @Slot(QPointF)
    def update_target_pos(self, signal_name: SignalName, pos: QPointF) -> None:
        scene_pos: QPointF = getattr(
            self, f"{signal_name}_plot_widget"
        ).plotItem.vb.mapSceneToView(pos)
        getattr(self, f"{signal_name}_target").setPos(scene_pos)

    @Slot()
    def reset_plots(self) -> None:
        plot_widgets = [
            self.hbr_plot_widget,
            self.ventilation_plot_widget,
            self.bpm_hbr_plot_widget,
            self.bpm_ventilation_plot_widget,
        ]
        for pw in plot_widgets:
            pw.getPlotItem().clear()
            pw.getPlotItem().legend.clear()
        self.hbr_signal_line = None
        self.ventilation_signal_line = None
        self._prepare_plot_items()

    def draw_signal(
        self,
        sig: NDArray[np.float32 | np.float64],
        plot_widget: pg.PlotWidget,
        signal_name: SignalName,
    ) -> None:
        color = "crimson" if signal_name == "hbr" else "royalblue"
        signal_line = pg.PlotDataItem(
            sig,
            pen=color,
            skipFiniteCheck=True,
            autoDownSample=True,
            name=f"{signal_name}_signal",
        )
        signal_line.curve.setSegmentedLineMode("on")
        signal_line.curve.setClickable(True, width=self.click_tolerance)

        bpm_plot_widget: pg.PlotWidget = getattr(self, f"bpm_{signal_name}_plot_widget")

        line_ref: pg.PlotDataItem | None = getattr(
            self, f"{signal_name}_signal_line", None
        )
        scatter_ref: pg.ScatterPlotItem | None = getattr(
            self, f"{signal_name}_peaks_scatter", None
        )
        bpm_line_ref: pg.PlotDataItem | None = getattr(
            self, f"bpm_{signal_name}_signal_line", None
        )
        bpm_mean_line_ref: pg.InfiniteLine | None = getattr(
            self, f"bpm_{signal_name}_mean_hline", None
        )
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
        signal_name: SignalName,
    ) -> None:
        brush_color = "goldenrod"
        hover_brush_color = "red"

        peaks_scatter = pg.ScatterPlotItem(
            x=pos_x,
            y=pos_y,
            pxMode=True,
            size=10,
            pen=None,
            brush=brush_color,
            useCache=True,
            name=f"{signal_name}_peaks",
            hoverable=True,
            hoverPen="black",
            hoverSymbol="x",
            hoverBrush=hover_brush_color,
            hoverSize=15,
            tip=None,
        )
        peaks_scatter.setZValue(60)

        scatter_ref: pg.ScatterPlotItem | None = getattr(
            self, f"{signal_name}_peaks_scatter", None
        )

        if scatter_ref is not None:
            scatter_ref.sigClicked.disconnect(self.remove_clicked_point)
            plot_widget.removeItem(scatter_ref)

        plot_widget.addItem(peaks_scatter)

        setattr(self, f"{signal_name}_peaks_scatter", peaks_scatter)
        getattr(self, f"{signal_name}_peaks_scatter").sigClicked.connect(
            self.remove_clicked_point
        )

    def draw_rate(
        self,
        bpm_data: NDArray[np.float32 | np.float64],
        plot_widget: pg.PlotWidget,
        signal_name: SignalName,
        **kwargs: float,
    ) -> None:
        mean_bpm = kwargs.get("mean_peak_interval", np.mean(bpm_data))
        pen_color = "green"
        mean_pen_color = "goldenrod"

        bpm_line = pg.PlotDataItem(
            bpm_data,
            pen=pen_color,
            autoDownsample=True,
            skipFiniteCheck=True,
            name=f"bpm_{signal_name}",
        )
        bpm_mean_line = pg.InfiniteLine(
            pos=mean_bpm,
            angle=0,
            pen=dict(color=mean_pen_color, width=2, style=Qt.PenStyle.DashLine),
            name=f"mean_bpm_{signal_name}",
        )

        bpm_line_ref: pg.PlotDataItem | None = getattr(
            self, f"bpm_{signal_name}_signal_line", None
        )
        bpm_mean_line_ref: pg.InfiniteLine | None = getattr(
            self, f"bpm_{signal_name}_mean_hline", None
        )

        if bpm_line_ref is not None:
            plot_widget.removeItem(bpm_line_ref)
            plot_widget.removeItem(bpm_mean_line_ref)
        if legend := plot_widget.getPlotItem().legend:
            legend.clear()
        else:
            legend = pg.LegendItem()
            plot_widget.getPlotItem().addLegend()

        plot_widget.addItem(bpm_line)
        plot_widget.addItem(bpm_mean_line)
        legend.addItem(
            pg.PlotDataItem(
                np.array([0, 1]),
                pen=dict(color=mean_pen_color, width=2, style=Qt.PenStyle.DashLine),
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
        """
        Remove clicked point from plot.

        Parameters
        ----------
        sender : pg.ScatterPlotItem
            The object that emitted the signal
        points : np.ndarray[pg.SpotItem, Any]
            Array of points under the cursor at the moment of the event
        ev : mouseEvents.MouseClickEvent
            The mouse click event
        """
        ev.accept()
        if len(points) == 0:
            return

        to_remove_index = points[0].index()

        scatter_plots: dict[str, pg.ScatterPlotItem | None] = {
            "hbr_peaks": getattr(self, "hbr_peaks_scatter", None),
            "ventilation_peaks": getattr(self, "ventilation_peaks_scatter", None),
        }

        if scatter_plot := scatter_plots.get(sender.name()):
            new_points_x = np.delete(scatter_plot.data["x"], to_remove_index)
            new_points_y = np.delete(scatter_plot.data["y"], to_remove_index)
            scatter_plot.setData(x=new_points_x, y=new_points_y)
            name = "hbr" if "hbr" in sender.name() else "ventilation"
            self.peak_edits["removed_peaks"][name].append(int(points[0].pos().x()))
            self.sig_peaks_edited.emit(name)

    @Slot()
    def remove_selected(self) -> None:
        """
        Removes peaks inside a rectangular selection.
        """
        name = self._parent.signal_name
        vb: CustomViewBox = (
            getattr(self, f"{name}_plot_widget").getPlotItem().getViewBox()
        )
        if vb.mapped_peak_selection is None:
            return
        rect = vb.mapped_peak_selection.boundingRect().getRect()
        rect_x, rect_y, rect_width, rect_height = (
            int(rect[0]),
            rect[1],
            rect[2],
            rect[3],
        )

        x_range = (rect_x, rect_x + rect_width)
        y_range = (rect_y, rect_y + rect_height)
        scatter_ref: pg.ScatterPlotItem = getattr(self, f"{name}_peaks_scatter")

        scatter_x, scatter_y = scatter_ref.getData()
        # self.hbr_plot_widget.saveState()
        to_remove = np.argwhere(
            (scatter_x >= x_range[0])
            & (scatter_x <= x_range[1])
            & (scatter_y >= y_range[0])
            & (scatter_y <= y_range[1])
        )

        if to_remove.size == 0:
            return
        scatter_ref.setData(
            x=np.delete(scatter_x, to_remove), y=np.delete(scatter_y, to_remove)
        )

        self.peak_edits["removed_peaks"][name].extend(
            scatter_x[to_remove][:, 0].astype(int).tolist()
        )
        self.sig_peaks_edited.emit(name)
        # self.removed_points[signal_name].extend(
        #     scatter_x[to_remove][:, 0].astype(int).tolist()
        # )

    @Slot(object, object)
    def add_clicked_point(
        self, sender: pg.PlotCurveItem, ev: mouseEvents.MouseClickEvent
    ) -> None:
        """
        Add scatter point to plot.

        Parameters
        ----------
        sender : pg.PlotCurveItem
            The object that emitted the signal
        ev : mouseEvents.MouseClickEvent
            The mouse click event
        """
        ev.accept()
        click_x = int(ev.pos().x())
        signal_map: dict[SignalName, pg.PlotCurveItem | None] = {
            "hbr": getattr(self, "hbr_signal_line", None),
            "ventilation": getattr(self, "ventilation_signal_line", None),
        }
        scatter_map: dict[SignalName, pg.ScatterPlotItem | None] = {
            "hbr": getattr(self, "hbr_peaks_scatter", None),
            "ventilation": getattr(self, "ventilation_peaks_scatter", None),
        }
        name = self._parent.signal_name
        # if name not in signal_map:
        #     return
        signal_line = signal_map[name]

        if signal_line is None:
            return

        xData: NDArray[np.float64] = signal_line.xData
        yData: NDArray[np.float64] = signal_line.yData

        # Define the radius within which to search for the max y-value
        search_radius = 15

        # Find the indices within the radius around the click position
        indices = np.where(
            (xData >= click_x - search_radius) & (xData <= click_x + search_radius)
        )[0]

        # Select the subset of y-values within the radius
        y_values_in_radius = yData[indices]

        # Find the index of the max y-value
        max_y_index = indices[np.argmax(y_values_in_radius)]

        # Get the new x and y values
        x_new, y_new = xData[max_y_index], y_values_in_radius.max()
        scatter_map[name].addPoints(x=x_new, y=y_new)
        # name = "hbr" if "hbr" in name else "ventilation"
        self.peak_edits["added_peaks"][name].append(int(x_new))
        self.sig_peaks_edited.emit(name)

    @Slot()
    def show_region_selector(self) -> None:
        name = self._parent.signal_name
        plot_widget = getattr(self, f"{name}_plot_widget")
        view_box = plot_widget.getPlotItem().getViewBox()
        selection_box = view_box.selection_box
        selection_box.setVisible(not selection_box.isVisible())

    # @Slot()
    # def restore_previous_peaks(self) -> None:
    #     signal_name = self._parent.signal_name
    #     scatter_plot = getattr(self, f"{signal_name}_peaks_scatter")
    #     scatter_plot.setData(
    #         x=self._old_peaks[signal_name][0], y=self._old_peaks[signal_name][1]
    #     )
    #     self.sig_peaks_edited.emit()

    def get_state(self) -> PeakEdits:
        return self.peak_edits

    def restore_state(self, state: PeakEdits) -> None:
        self.peak_edits = state
