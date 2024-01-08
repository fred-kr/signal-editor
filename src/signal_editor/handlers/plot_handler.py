from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Iterable, Literal, Sequence, override

import numpy as np
import pyqtgraph as pg
from numpy.typing import NDArray
from pyqtgraph.GraphicsScene import mouseEvents
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QObject, QRectF, Qt, Signal, Slot
from PySide6.QtGui import QFont

from ..models.result import ManualPeakEdits
from ..type_aliases import (
    SignalName,
    SpotItemDict,
)

if TYPE_CHECKING:
    from ..app import MainWindow


def _pinv_fallback(tr: QtGui.QTransform) -> QtGui.QTransform:
    """
    Calculate the pseudo-inverse of a QTransform object.

    Parameters
    ----------
    tr : QtGui.QTransform
        The QTransform object to calculate the pseudo-inverse of.

    Returns
    -------
    QtGui.QTransform
        The pseudo-inverse of the QTransform object.
    """
    arr = np.array(
        [
            tr.m11(),
            tr.m12(),
            tr.m13(),
            tr.m21(),
            tr.m22(),
            tr.m23(),
            tr.m31(),
            tr.m32(),
            tr.m33(),
        ]
    )
    arr.shape = (3, 3)
    pinv = np.linalg.pinv(arr)
    return QtGui.QTransform(*pinv.ravel().tolist())


def invertQTransform(tr: QtGui.QTransform) -> QtGui.QTransform:
    """
    Return a QTransform that is the inverse of `tr`.
    A pseudo-inverse is returned if tr is not invertible.

    Note that this function is preferred over `QTransform.inverted()` due to
    bugs in that method. (specifically, Qt has floating-point precision issues
    when determining whether a matrix is invertible)

    Parameters
    ----------
    tr : QtGui.QTransform
        The QTransform to get the inverse of.

    Returns
    -------
    QtGui.QTransform
        The (pseudo-)inverse of the QTransform object.
    """
    try:
        det = tr.determinant()
        detr = 1.0 / det  # let singular matrices raise ZeroDivisionError
        inv = tr.adjoint()
        inv *= detr
        return inv
    except ZeroDivisionError:
        return _pinv_fallback(tr)


class ScatterPlotItemError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class CustomViewBox(pg.ViewBox):
    sig_selection_changed = Signal(QtGui.QPolygonF)

    def __init__(self, *args: Any, **kargs: Any) -> None:
        pg.ViewBox.__init__(self, *args, **kargs)
        self._selection_box: QtWidgets.QGraphicsRectItem | None = None
        self._deletion_box: QtWidgets.QGraphicsRectItem | None = None
        self.mapped_peak_selection: QtGui.QPolygonF | None = None
        self.mapped_deletion_selection: QtGui.QPolygonF | None = None

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
            return
        selection_box.setZValue(1e9)
        selection_box.hide()
        self.addItem(selection_box, ignoreBounds=True)
        return

    @property
    def deletion_box(self) -> QtWidgets.QGraphicsRectItem:
        if self._deletion_box is None:
            deletion_box = QtWidgets.QGraphicsRectItem(0, 0, 1, 1)
            deletion_box.setPen(pg.mkPen((255, 100, 100), width=1))
            deletion_box.setBrush(pg.mkBrush((255, 100, 100, 100)))
            deletion_box.setZValue(1e9)
            deletion_box.hide()
            self._deletion_box = deletion_box
            self.addItem(deletion_box, ignoreBounds=True)
        return self._deletion_box

    @deletion_box.setter
    def deletion_box(self, deletion_box: QtWidgets.QGraphicsRectItem | None) -> None:
        if self._deletion_box is not None:
            self.removeItem(self._deletion_box)
        self._deletion_box = deletion_box
        if deletion_box is None:
            return
        deletion_box.setZValue(1e9)
        deletion_box.hide()
        self.addItem(deletion_box, ignoreBounds=True)
        return

    @override
    def mouseDragEvent(
        self, ev: mouseEvents.MouseDragEvent, axis: int | float | None = None
    ) -> None:
        ev.accept()

        pos = ev.pos()
        lastPos = ev.lastPos()
        dif = (pos - lastPos) * np.array([-1, -1])

        mouseEnabled = np.array(self.state["mouseEnabled"], dtype=np.float64)
        mask = mouseEnabled.copy()
        if axis is not None:
            mask[1 - axis] = 0.0

        def is_middle_button(ev: mouseEvents.MouseDragEvent) -> bool:
            return ev.button() == Qt.MouseButton.MiddleButton

        def is_left_button_with_control(ev: mouseEvents.MouseDragEvent) -> bool:
            return (
                ev.button() == Qt.MouseButton.LeftButton
                and ev.modifiers() & Qt.KeyboardModifier.ControlModifier
            )

        def is_left_button_with_alt(ev: mouseEvents.MouseDragEvent) -> bool:
            return (
                ev.button() == Qt.MouseButton.LeftButton
                and ev.modifiers() & Qt.KeyboardModifier.AltModifier
            )

        def is_left_button(ev: mouseEvents.MouseDragEvent) -> bool:
            return ev.button() == Qt.MouseButton.LeftButton

        def is_right_button(ev: mouseEvents.MouseDragEvent) -> bool:
            return ev.button() & Qt.MouseButton.RightButton

        def create_selection(ev: mouseEvents.MouseDragEvent) -> QtGui.QPolygonF:
            r = QRectF(ev.pos(), ev.buttonDownPos())
            return self.mapToView(r)

        if is_middle_button(ev) or is_left_button_with_control(ev):
            if ev.isFinish():
                self.mapped_peak_selection = create_selection(ev)
            else:
                self.updateSelectionBox(ev.pos(), ev.buttonDownPos())
                self.mapped_peak_selection = None
        elif is_left_button_with_alt(ev):
            if ev.isFinish():
                self.mapped_deletion_selection = create_selection(ev)
            else:
                self.updateDeletionBox(ev.pos(), ev.buttonDownPos())
                self.mapped_deletion_selection = None
        elif is_left_button(ev):
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
                tr = invertQTransform(self.childGroup.transform())
                tr = tr.map(dif * mask) - tr.map(pg.Point(0, 0))

                x = tr.x() if mask[0] == 1 else None
                y = tr.y() if mask[1] == 1 else None

                self._resetTarget()
                if x is not None or y is not None:
                    self.translateBy(x=x, y=y)
                self.sigRangeChangedManually.emit(self.state["mouseEnabled"])
        elif is_right_button(ev):
            if self.state["aspectLocked"] is not False:
                mask[0] = 0

            dif = np.array(
                [
                    -(ev.screenPos().x() - ev.lastScreenPos().x()),
                    ev.screenPos().y() - ev.lastScreenPos().y(),
                ],
                dtype=np.float64,
            )
            s = ((mask * 0.02) + 1) ** dif

            tr = invertQTransform(self.childGroup.transform())

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

    def updateDeletionBox(self, pos1: pg.Point, pos2: pg.Point) -> None:
        rect = QRectF(pos1, pos2)
        rect = self.childGroup.mapRectFromParent(rect)
        self.deletion_box.setPos(rect.topLeft())
        tr = QtGui.QTransform.fromScale(rect.width(), rect.height())
        self.deletion_box.setTransform(tr)
        self.deletion_box.show()


type CustomScatterPlotItemArgs = (
    list[SpotItemDict]
    | tuple[NDArray[np.float_ | np.int_], NDArray[np.float_ | np.int_]]
)


class CustomScatterPlotItem(pg.ScatterPlotItem):
    @override
    def addPoints(
        self,
        *args: Any,
        **kargs: Any,
    ) -> None:
        if len(args) == 1:
            kargs["spots"] = args[0]
        elif len(args) == 2:
            kargs["x"] = args[0]
            kargs["y"] = args[1]
        elif len(args) > 2:
            raise ScatterPlotItemError("Only accepts up to two non-keyword arguments.")

        if "pos" in kargs:
            pos = kargs["pos"]
            if isinstance(pos, np.ndarray):
                kargs["x"] = pos[:, 0]
                kargs["y"] = pos[:, 1]
            else:
                x = []
                y = []
                for p in pos:
                    if isinstance(p, QtCore.QPointF):
                        x.append(p.x())
                        y.append(p.y())
                    else:
                        x.append(p[0])
                        y.append(p[1])
                kargs["x"] = x
                kargs["y"] = y

        if "spots" in kargs:
            numPts = len(kargs["spots"])
        elif "y" in kargs and kargs["y"] is not None and hasattr(kargs["y"], "__len__"):
            numPts = len(kargs["y"])
        elif "y" in kargs and kargs["y"] is not None:
            numPts = 1
        else:
            kargs["x"] = []
            kargs["y"] = []
            numPts = 0

        self.data["item"][...] = None

        oldData = self.data
        self.data = np.empty(len(oldData) + numPts, dtype=self.data.dtype)

        self.data[: len(oldData)] = oldData

        newData = self.data[len(oldData) :]
        newData["size"] = -1
        newData["visible"] = True

        if "spots" in kargs:
            spots = kargs["spots"]
            for i in range(len(spots)):
                spot = spots[i]
                for k in spot:
                    if k == "pos":
                        pos = spot[k]
                        if isinstance(pos, QtCore.QPointF):
                            x, y = pos.x(), pos.y()
                        else:
                            x, y = pos[0], pos[1]
                        newData[i]["x"] = x
                        newData[i]["y"] = y
                    elif k == "pen":
                        newData[i][k] = pg.mkPen(spot[k])
                    elif k == "brush":
                        newData[i][k] = pg.mkBrush(spot[k])
                    elif k in ["x", "y", "size", "symbol", "data"]:
                        newData[i][k] = spot[k]
                    else:
                        raise ScatterPlotItemError(f"Unknown spot parameter: {k}")
        elif "y" in kargs:
            newData["x"] = kargs["x"]
            newData["y"] = kargs["y"]

        if "name" in kargs:
            self.opts["name"] = kargs["name"]
        if "pxMode" in kargs:
            self.setPxMode(kargs["pxMode"])
        if "antialias" in kargs:
            self.opts["antialias"] = kargs["antialias"]
        if "hoverable" in kargs:
            self.opts["hoverable"] = bool(kargs["hoverable"])
        if "tip" in kargs:
            self.opts["tip"] = kargs["tip"]
        if "useCache" in kargs:
            self.opts["useCache"] = kargs["useCache"]

        for k in ["pen", "brush", "symbol", "size"]:
            if k in kargs:
                setMethod = getattr(self, f"set{k[0].upper()}{k[1:]}")
                setMethod(
                    kargs[k],
                    update=False,
                    dataSet=newData,
                    mask=kargs.get("mask", None),
                )
            kh = f"hover{k.title()}"
            if kh in kargs:
                vh = kargs[kh]
                if k == "pen":
                    vh = pg.mkPen(vh)
                elif k == "brush":
                    vh = pg.mkBrush(vh)
                self.opts[kh] = vh
        if "data" in kargs:
            self.setPointData(kargs["data"], dataSet=newData)

        self.prepareGeometryChange()
        self.informViewBoundsChanged()
        self.bounds = [None, None]
        self.invalidate()
        self.updateSpots(newData)
        self.sigPlotChanged.emit(self)


type PlotItemAttr = Literal["signal", "peaks", "rate", "rate_mean"]
type PlotItemVal = pg.PlotDataItem | pg.ScatterPlotItem | pg.InfiniteLine


@dataclass(slots=True, kw_only=True)
class PlotItems:
    name: SignalName | str
    signal: pg.PlotDataItem | None = None
    peaks: pg.ScatterPlotItem | None = None
    rate: pg.PlotDataItem | None = None
    rate_mean: pg.InfiniteLine | None = None
    # excluded_regions: list[pg.LinearRegionItem] | None = None

    def __getitem__(self, key: PlotItemAttr) -> PlotItemVal | None:
        return getattr(self, key)

    def __setitem__(self, key: PlotItemAttr, value: PlotItemVal) -> None:
        setattr(self, key, value)

    def as_dict(self) -> dict[str, PlotItemVal | None]:
        return {
            "signal": self["signal"],
            "peaks": self["peaks"],
            "rate": self["rate"],
            "rate_mean": self["rate_mean"],
        }


class PlotItemsContainer(dict[SignalName | str, PlotItems]):
    def __init__(self, plots: Iterable[PlotItems] = ()) -> None:
        super().__init__((plot.name, plot) for plot in plots)

    def get_plot_item(
        self, plot_name: SignalName | str, item_name: PlotItemAttr
    ) -> PlotItemVal:
        return self[plot_name][item_name]

    def set_plot_item(
        self,
        plot_name: SignalName | str,
        item_name: PlotItemAttr,
        item_value: PlotItemVal,
    ) -> None:
        self[plot_name][item_name] = item_value


def make_plot_widget(
    background_color: str, view_box: pg.ViewBox | CustomViewBox | None = None
) -> pg.PlotWidget:
    return pg.PlotWidget(
        viewBox=view_box,
        background=background_color,
        useOpenGL=True,
    )


class PlotWidgetContainer(dict[str, pg.PlotWidget]):
    def __init__(
        self, *args: Iterable[tuple[str, pg.PlotWidget]], **kwargs: pg.PlotWidget
    ) -> None:
        super().__init__(*args, **kwargs)

    def __setitem__(self, key: str, value: pg.PlotWidget) -> None:
        super().__setitem__(key, value)

    def make_plot_widget(
        self,
        name: str,
        background_color: str = "default",
        view_box: pg.ViewBox | CustomViewBox | None = None,
    ) -> None:
        widget = pg.PlotWidget(
            viewBox=view_box, background=background_color, useOpenGL=True
        )
        self[name] = widget

    def get_all_widgets(self) -> list[pg.PlotWidget]:
        return list(self.values())

    def get_view_box(self, name: str) -> pg.ViewBox | CustomViewBox:
        return self[name].getPlotItem().getViewBox()

    def get_all_view_boxes(self) -> list[pg.ViewBox | CustomViewBox]:
        return [pw.getPlotItem().getViewBox() for pw in self.values()]


class PlotHandler(QObject):
    """
    Class that manages showing and updating plots.
    """

    # sig_rate_updated = Signal(str)
    sig_peaks_edited = Signal(str)
    sig_excluded_range = Signal(int, int)

    def __init__(
        self,
        parent: "MainWindow",
    ):
        super().__init__(parent=parent)
        self._parent = parent
        self.click_tolerance = 80
        self.plot_items = PlotItemsContainer()
        self.plot_widgets = PlotWidgetContainer()
        self.peak_edits: dict[SignalName | str, ManualPeakEdits] = {name: ManualPeakEdits() for name in ("hbr", "ventilation")}
        self._last_edit_index: int = 0

        self.plot_widgets.make_plot_widget(name="hbr", view_box=CustomViewBox())
        self.plot_widgets.make_plot_widget(name="ventilation", view_box=CustomViewBox())
        self.plot_widgets.make_plot_widget(name="hbr_rate")
        self.plot_widgets.make_plot_widget(name="ventilation_rate")

        self._prepare_plot_items()

    @property
    def last_edit_index(self) -> int:
        return self._last_edit_index

    @last_edit_index.setter
    def last_edit_index(self, value: int | float) -> None:
        self._last_edit_index = int(value)

    def _init_plot_items(self) -> None:
        self.plot_items["hbr"] = PlotItems(name="hbr")
        self.plot_items["ventilation"] = PlotItems(name="ventilation")
        self.plot_items["hbr_rate"] = PlotItems(name="hbr_rate")
        self.plot_items["ventilation_rate"] = PlotItems(name="ventilation_rate")

    @staticmethod
    def set_plot_titles_and_labels(
        plot_item: pg.PlotItem, title: str, left_label: str, bottom_label: str
    ) -> None:
        plot_item.setTitle(title)
        plot_item.setLabel(axis="left", text=left_label)
        plot_item.setLabel(axis="bottom", text=bottom_label)

    def set_style(self, style: Literal["dark", "light"]) -> None:
        plot_widgets = self.plot_widgets.get_all_widgets()

        if style == "dark":
            self._set_dark_style(plot_widgets)
        elif style == "light":
            self._set_light_style(plot_widgets)

    def _set_dark_style(self, plot_widgets: list[pg.PlotWidget]) -> None:
        self._set_plot_labels(color="lightgray")
        for pw in plot_widgets:
            pw.setBackground("black")
            legend = pw.getPlotItem().legend
            if legend is not None:
                legend.setBrush("gray")
                legend.setPen("white")
                legend.setLabelTextColor("lightgray")

    def _set_light_style(self, plot_widgets: list[pg.PlotWidget]) -> None:
        self._set_plot_labels(color="black")
        for pw in plot_widgets:
            pw.setBackground("white")
            legend = pw.getPlotItem().legend
            if legend is not None:
                legend.setBrush("gray")
                legend.setPen("black")
                legend.setLabelTextColor("black")

    def _prepare_plot_items(self) -> None:
        self._init_plot_items()

        style = self._parent.active_style
        pen_col = "lightgray" if style == "dark" else "black"
        plot_font = QFont("Segoe UI", 10, 400)
        plot_font.setBold(True)
        self._set_plot_labels(color=pen_col)

        for name, pw in self.plot_widgets.items():
            plot_item = pw.getPlotItem()

            plot_item.showGrid(x=False, y=True)
            plot_item.getViewBox().enableAutoRange("y")
            plot_item.setDownsampling(auto=True)
            plot_item.setClipToView(True)
            plot_item.addLegend(
                pen=pen_col,
                labelTextColor=pen_col,
                brush="transparent",
                colCount=2,
            )
            plot_item.legend.anchor(itemPos=(0, 1), parentPos=(0, 1), offset=(5, -5))  # type: ignore
            plot_item.register(name)
            plot_item.getViewBox().setAutoVisible(y=True)
            plot_item.setMouseEnabled(x=True, y=False)
            for axis in {"left", "bottom", "top", "right"}:
                plot_item.getAxis(axis).label.setFont(plot_font)

        self.plot_widgets.get_view_box("hbr").setXLink("hbr_rate")
        self.plot_widgets.get_view_box("ventilation").setXLink("ventilation_rate")
        # self.plot_widgets.get_signal_widget("hbr").getPlotItem().getViewBox().setXLink(
        #     "rate_hbr"
        # )
        # self.plot_widgets.get_signal_widget(
        #     "ventilation"
        # ).getPlotItem().getViewBox().setXLink("rate_ventilation")
        self.plot_widgets.get_view_box("hbr_rate").setMouseMode(pg.ViewBox.RectMode)
        self.plot_widgets.get_view_box("ventilation_rate").setMouseMode(
            pg.ViewBox.RectMode
        )

    def _set_plot_labels(
        self, color: str, font_family: str = "Segoe UI", title_font_size: int = 12
    ) -> None:
        self.set_plot_titles_and_labels(
            self.plot_widgets["hbr"].getPlotItem(),
            title=f"<span style='color: {color}; font-family: {font_family}; font-size: {title_font_size}pt;'><b>Heart</b></span>",
            left_label=f"<span style='color: {color};'>Signal Amplitude</span>",
            bottom_label=f"<span style='color: {color};'>n samples</span>",
        )
        self.set_plot_titles_and_labels(
            self.plot_widgets["ventilation"].getPlotItem(),
            title=f"<span style='color: {color}; font-family: {font_family}; font-size: {title_font_size}pt;'><b>Ventilation</b></span>",
            left_label=f"<span style='color: {color};'>Signal Amplitude</span>",
            bottom_label=f"<span style='color: {color};'>n samples</span>",
        )
        self.set_plot_titles_and_labels(
            self.plot_widgets["hbr_rate"].getPlotItem(),
            title=f"<span style='color: {color}; font-family: {font_family}; font-size: {title_font_size}pt;'><b>Estimated Rate</b></span>",
            left_label=f"<span style='color: {color};'>Cycles per minute</span>",
            bottom_label=f"<span style='color: {color};'>n samples</span>",
        )
        self.set_plot_titles_and_labels(
            self.plot_widgets["ventilation_rate"].getPlotItem(),
            title=f"<span style='color: {color}; font-family: {font_family}; font-size: {title_font_size}pt;'><b>Estimated Rate</b></span>",
            left_label=f"<span style='color: {color};'>Cycles per minute</span>",
            bottom_label=f"<span style='color: {color};'>n samples</span>",
        )

    @Slot()
    def reset_plots(self) -> None:
        for name, pw in self.plot_widgets.items():
            pw.getPlotItem().clear()
            pw.getPlotItem().addLegend().clear()
            self.plot_items[name]["signal"] = pg.PlotDataItem()

        self._prepare_plot_items()

    @Slot(int)
    def reset_views(self, upper_range: int) -> None:
        for vb in self.plot_widgets.get_all_view_boxes():
            vb.setXRange(0, upper_range)

    def draw_signal(
        self,
        sig: NDArray[np.float64],
        plot_widget: pg.PlotWidget,
        signal_name: SignalName | str,
        # **kwargs: NDArray[np.float_ | np.int_],
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

        # IDEA: Add temperature as second y-axis
        # temperature = kwargs.get("temperature", None)
        # if temperature is not None:
        #     temperature_line = pg.PlotDataItem(
        #         temperature,
        #         pen="magenta",
        #         skipFiniteCheck=True,
        #         autoDownSample=True,
        #         name="temperature",
        #     )
        #     plot_widget.addItem(temperature_line)

        rate_name = f"{signal_name}_rate"
        rate_plot_widget = self.plot_widgets[rate_name]
        # scatter_ref = self.plot_items[signal_name]["peaks"]
        # rate_line_ref = self.plot_items[signal_name]["rate"]
        # rate_mean_line_ref = self.plot_items[signal_name]["rate_mean"]
        item_dict = self.plot_items[signal_name].as_dict()

        for name, item in item_dict.items():
            if item is not None and name != "name":
                if name == "signal":
                    item.sigClicked.disconnect(self.add_clicked_point)
                if "rate" in name:
                    rate_plot_widget.removeItem(item)
                else:
                    plot_widget.removeItem(item)

        # if line_ref := self.plot_items[signal_name]["signal"]:
        #     line_ref.sigClicked.disconnect(self.add_clicked_point)
        #     plot_widget.removeItem(line_ref)
        # if scatter_ref:
        #     plot_widget.removeItem(scatter_ref)
        # if rate_line_ref:
        #     rate_plot_widget.removeItem(rate_line_ref)
        # if rate_mean_line_ref:
        #     rate_plot_widget.removeItem(rate_mean_line_ref)

        plot_widget.addItem(signal_line)

        signal_line.sigClicked.connect(self.add_clicked_point)

        self.plot_items[signal_name]["signal"] = signal_line

    def draw_peaks(
        self,
        pos_x: NDArray[np.int32],
        pos_y: NDArray[np.float64],
        plot_widget: pg.PlotWidget,
        signal_name: SignalName | str,
    ) -> None:
        """
        Draw peaks on the plot.

        Draws a scatter plot of the peaks on the plot widget. If there are already peaks
        drawn, they are removed and replaced with the new peaks. The peaks are also
        connected to the `remove_clicked_point` slot.

        Parameters
        ----------
        pos_x : NDArray[np.int32]
            Array of integers representing the x positions of the peaks.
        pos_y : NDArray[np.float64]
            Array of floats corresponding to the y positions of the peaks.
        plot_widget : pg.PlotWidget
            The plot widget to draw the peaks on.
        signal_name : SignalName | str
            The name of the signal to draw the peaks for.
        """
        brush_color = "goldenrod"
        hover_brush_color = "red"

        peaks_scatter = CustomScatterPlotItem(
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

        scatter_ref = self.plot_items[signal_name]["peaks"]

        if scatter_ref is not None:
            scatter_ref.sigClicked.disconnect(self.remove_clicked_point)
            plot_widget.removeItem(scatter_ref)

        plot_widget.addItem(peaks_scatter)
        peaks_scatter.sigClicked.connect(self.remove_clicked_point)
        self.plot_items[signal_name]["peaks"] = peaks_scatter

    def draw_rate(
        self,
        rate_data: NDArray[np.float64],
        plot_widget: pg.PlotWidget,
        signal_name: SignalName | str,
        **kwargs: float,
    ) -> None:
        rate_mean = kwargs.get(
            "mean_peak_interval", np.mean(rate_data, dtype=np.float64)
        )
        pen_color = "green"
        mean_pen_color = "goldenrod"

        rate_line = pg.PlotDataItem(
            rate_data,
            pen=pen_color,
            autoDownsample=True,
            skipFiniteCheck=True,
            name=f"rate_{signal_name}",
        )
        rate_mean_line = pg.InfiniteLine(
            pos=rate_mean,
            angle=0,
            pen=dict(color=mean_pen_color, width=2, style=Qt.PenStyle.DashLine),
            name=f"rate_mean_{signal_name}",
        )

        rate_line_ref = self.plot_items[signal_name]["rate"]
        rate_mean_ref = self.plot_items[signal_name]["rate_mean"]

        if rate_line_ref is not None:
            plot_widget.removeItem(rate_line_ref)
            plot_widget.removeItem(rate_mean_ref)

        if legend := plot_widget.getPlotItem().legend:
            legend.clear()
        else:
            legend = pg.LegendItem()
            plot_widget.getPlotItem().addLegend()

        plot_widget.addItem(rate_line)
        plot_widget.addItem(rate_mean_line)
        legend.addItem(
            pg.PlotDataItem(
                np.array([0, 1]),
                pen=dict(color=mean_pen_color, width=2, style=Qt.PenStyle.DashLine),
                skipFiniteCheck=True,
            ),
            f"<span style='color: {mean_pen_color}; font-family: Segoe UI; font-size: 10pt;'>Mean Rate: {int(rate_mean)}</span>",
        )

        self.plot_items[signal_name]["rate"] = rate_line
        self.plot_items[signal_name]["rate_mean"] = rate_mean_line

    @Slot(object, object, object)
    def remove_clicked_point(
        self,
        sender: pg.ScatterPlotItem,
        points: Sequence[pg.SpotItem],
        ev: mouseEvents.MouseClickEvent,
    ) -> None:
        """
        Remove clicked point from plot.

        Parameters
        ----------
        sender : pyqtgraph.ScatterPlotItem
            The object that emitted the signal
        points : NDArray[pg.SpotItem]
            Array of points under the cursor at the moment of the event
        ev : pyqtgraph.mouseEvents.MouseClickEvent
            The mouse click event
        """
        ev.accept()
        if len(points) == 0:
            return

        spot_item = points[0]
        to_remove_index = spot_item.index()

        name = "hbr" if "hbr" in f"{sender.name()}" else "ventilation"

        if scatter_plot := self.plot_items[name]["peaks"]:
            new_points_x = np.delete(scatter_plot.data["x"], to_remove_index)  # type: ignore
            new_points_y = np.delete(scatter_plot.data["y"], to_remove_index)  # type: ignore
            scatter_plot.setData(x=new_points_x, y=new_points_y)
            self.peak_edits[name].added.append(int(points[0].pos().x()))
            self.last_edit_index = int(points[0].pos().x())
            self.sig_peaks_edited.emit(name)

    @Slot()
    def remove_selected(self) -> None:
        """
        Removes peaks inside a rectangular selection.
        """
        name = self._parent.signal_name
        vb = self.plot_widgets.get_view_box(name)
        if vb.mapped_peak_selection is None:
            return
        scatter_ref = self.plot_items[name]["peaks"]
        if scatter_ref is None:
            return
        rect: tuple[
            float, float, float, float
        ] = vb.mapped_peak_selection.boundingRect().getRect()
        rect_x, rect_y, rect_width, rect_height = (
            int(rect[0]),
            rect[1],
            rect[2],
            rect[3],
        )

        x_range = (rect_x, rect_x + rect_width)
        y_range = (rect_y, rect_y + rect_height)

        scatter_x: NDArray[np.integer[Any]] | None
        scatter_y: NDArray[np.floating[Any]] | None
        scatter_x, scatter_y = scatter_ref.getData()
        if scatter_x is None or scatter_y is None:
            return
        if scatter_x.size == 0 or scatter_y.size == 0:
            return

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

        self.peak_edits[name].removed.extend(
            scatter_x[to_remove][:, 0].astype(int).tolist()
        )
        self.last_edit_index = scatter_x[to_remove].max()
        self.sig_peaks_edited.emit(name)

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
        name = "hbr" if "hbr" in f"{sender.name()}" else "ventilation"
        signal_line = self.plot_items[name]["signal"]
        scatter_ref = self.plot_items[name]["peaks"]

        if signal_line is None or scatter_ref is None:
            return

        xData: NDArray[np.float64] = signal_line.xData
        yData: NDArray[np.float64] = signal_line.yData

        # Define the radius within which to search for the max y-value
        search_radius = 20

        # Find the indices within the radius around the click position
        indices = np.where(
            (xData >= click_x - search_radius) & (xData <= click_x + search_radius)
        )[0]

        # Select the subset of y-values within the radius
        y_values_in_radius = yData[indices]

        # Find the index of the max y-value
        max_y_index = indices[np.argmax(y_values_in_radius)]
        if max_y_index in scatter_ref.data["x"]:
            return

        # Get the new x and y values
        x_new, y_new = xData[max_y_index], y_values_in_radius.max()
        scatter_ref.addPoints(x=x_new, y=y_new)
        self.peak_edits[name].added.append(int(x_new))
        self.last_edit_index = x_new
        self.sig_peaks_edited.emit(name)

    @Slot()
    def show_region_selector(self) -> None:
        # TODO: Get name through widget (QApplication.widgetAt(cursor_pos)) at mouse cursor position (QCursor.pos())
        name = self._parent.signal_name
        view_box = self.plot_widgets.get_view_box(name)
        selection_box = view_box.selection_box
        selection_box.setVisible(not selection_box.isVisible())
        selection_box.setEnabled(not selection_box.isEnabled())

    @Slot()
    def show_exclusion_selector(self) -> None:
        # TODO: Get name through widget (QApplication.widgetAt(cursor_pos)) at mouse cursor position (QCursor.pos())
        name = self._parent.signal_name
        view_box = self.plot_widgets.get_view_box(name)
        deletion_box = view_box.deletion_box
        deletion_box.setVisible(not deletion_box.isVisible())
        deletion_box.setEnabled(not deletion_box.isEnabled())

    @Slot()
    def mark_excluded(self) -> None:
        name = self._parent.signal_name
        vb = self.plot_widgets.get_view_box(name)
        if vb.mapped_deletion_selection is None:
            return
        signal_data_ref = self.plot_items[name]["signal"]
        if signal_data_ref is None:
            return
        rect: tuple[
            float, float, float, float
        ] = vb.mapped_deletion_selection.boundingRect().getRect()
        rect_x, rect_width = int(rect[0]), int(rect[2])

        x_range = (rect_x, rect_x + rect_width)
        lr_marker = pg.LinearRegionItem(values=x_range, movable=False)
        self.sig_excluded_range.emit(x_range[0], x_range[1])

    def get_state(self) -> dict[SignalName | str, ManualPeakEdits]:
        return self.peak_edits

    def restore_state(self, state: dict[SignalName | str, ManualPeakEdits]) -> None:
        self.peak_edits = state
