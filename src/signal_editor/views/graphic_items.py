import typing as t

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from .. import type_aliases as _t

if t.TYPE_CHECKING:
    from pyqtgraph.GraphicsScene import mouseEvents


def _mk_pen(*args: t.Any, **kwargs: t.Any) -> QtGui.QPen:
    if len(args) == 1 and isinstance(args[0], QtGui.QPen):
        return args[0]
    return pg.mkPen(*args, **kwargs)


def _mk_brush(*args: t.Any, **kwargs: t.Any) -> QtGui.QBrush:
    if len(args) == 1 and isinstance(args[0], QtGui.QBrush):
        return args[0]
    return pg.mkBrush(*args, **kwargs)


class ScatterPlotItemError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class CustomViewBox(pg.ViewBox):
    """
    Custom `pyqtgraph.ViewBox` subclass that makes plot editing easier.
    """

    def __init__(self, *args: t.Any, **kargs: t.Any) -> None:
        super().__init__(*args, **kargs)
        self._selection_box: QtWidgets.QGraphicsRectItem | None = None
        self.mapped_selection_rect: QtGui.QPolygonF | None = None

    @property
    def selection_box(self) -> QtWidgets.QGraphicsRectItem:
        if self._selection_box is None:
            selection_box = QtWidgets.QGraphicsRectItem(0, 0, 1, 1)
            selection_box.setPen(pg.mkPen(color=(50, 100, 200, 255)))
            selection_box.setBrush(pg.mkBrush((50, 100, 200, 100)))
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

    @staticmethod
    def get_button_type(
        ev: "mouseEvents.MouseDragEvent",
    ) -> t.Literal["middle", "left", "left+control", "right", "unknown"]:
        if ev.button() == QtCore.Qt.MouseButton.MiddleButton:
            return "middle"
        elif ev.button() == QtCore.Qt.MouseButton.LeftButton:
            if ev.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
                return "left+control"
            else:
                return "left"
        elif ev.button() == QtCore.Qt.MouseButton.RightButton:
            return "right"
        else:
            return "unknown"

    @t.override
    def mouseDragEvent(
        self, ev: "mouseEvents.MouseDragEvent", axis: int | float | None = None
    ) -> None:
        ev.accept()

        pos = ev.pos()
        last_pos = ev.lastPos()
        dif = (pos - last_pos) * np.array([-1, -1])

        mouse_enabled = np.array(self.state["mouseEnabled"], dtype=np.float64)
        mask = mouse_enabled.copy()
        if axis is not None:
            mask[1 - axis] = 0.0

        button_type = self.get_button_type(ev)

        if button_type in {"middle", "left+control"}:
            if ev.isFinish():
                r = QtCore.QRectF(ev.pos(), ev.buttonDownPos())
                self.mapped_selection_rect = self.mapToView(r)
            else:
                self.updateSelectionBox(ev.pos(), ev.buttonDownPos())
                self.mapped_selection_rect = None
        elif button_type == "left":
            if self.state["mouseMode"] == pg.ViewBox.RectMode and axis is None:
                if ev.isFinish():
                    self.rbScaleBox.hide()
                    ax = QtCore.QRectF(pg.Point(ev.buttonDownPos(ev.button())), pg.Point(pos))
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
        elif button_type == "right":
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

            tr = pg.invertQTransform(self.childGroup.transform())

            x = s[0] if mouse_enabled[0] == 1 else None
            y = s[1] if mouse_enabled[1] == 1 else None

            center = pg.Point(tr.map(ev.buttonDownPos(QtCore.Qt.MouseButton.RightButton)))
            self._resetTarget()
            self.scaleBy(x=x, y=y, center=center)
            self.sigRangeChangedManually.emit(self.state["mouseEnabled"])

    def updateSelectionBox(self, pos1: pg.Point, pos2: pg.Point) -> None:
        rect = QtCore.QRectF(pos1, pos2)
        rect = self.childGroup.mapRectFromParent(rect)
        self.selection_box.setPos(rect.topLeft())
        tr = QtGui.QTransform.fromScale(rect.width(), rect.height())
        self.selection_box.setTransform(tr)
        self.selection_box.show()


class CustomScatterPlotItem(pg.ScatterPlotItem):
    def addPoints(self, *args: t.Any, **kargs: t.Any) -> None:
        arg_keys = ["spots", "x", "y"]
        for i, key in enumerate(arg_keys[: len(args)]):
            kargs[key] = args[i]

        pos = kargs.get("pos")
        if pos is not None:
            if isinstance(pos, np.ndarray):
                kargs["x"], kargs["y"] = pos[:, 0], pos[:, 1]
            else:
                kargs["x"] = [p.x() if isinstance(p, QtCore.QPointF) else p[0] for p in pos]
                kargs["y"] = [p.y() if isinstance(p, QtCore.QPointF) else p[1] for p in pos]

        spots: list[_t.SpotDict] | None = kargs.get("spots")
        x = kargs.get("x")
        y = kargs.get("y")

        # Calculate number of points
        num_pts = (
            len(spots)
            if spots is not None
            else len(y)
            if y is not None and hasattr(y, "__len__")
            else 1
            if y is not None
            else 0
        )

        # Initialize new data array
        self.data["item"][...] = None
        old_data = self.data
        self.data = np.empty(len(old_data) + num_pts, dtype=self.data.dtype)
        self.data[: len(old_data)] = old_data
        new_data = self.data[len(old_data) :]
        new_data["size"] = -1
        new_data["visible"] = True

        # Handle 'spots' parameter
        if spots is not None:
            for i, spot in enumerate(spots):
                for k, v in spot.items():
                    match k:
                        case "pos":
                            pos = v
                            if isinstance(pos, QtCore.QPointF):
                                x, y = pos.x(), pos.y()
                            else:
                                x, y = pos[0], pos[1]
                            new_data[i]["x"] = x
                            new_data[i]["y"] = y
                        case "pen":
                            new_data[i][k] = _mk_pen(v)
                        case "brush":
                            new_data[i][k] = _mk_brush(v)
                        case "x" | "y" | "size" | "symbol" | "data":
                            new_data[i][k] = v
                        case _:
                            raise ScatterPlotItemError(f"Unknown spot parameter: {k}")
        # Handle 'y' parameter
        elif y is not None:
            new_data["x"] = x
            new_data["y"] = y

        for k, v in kargs.items():
            match k:
                case "name":
                    self.opts["name"] = v
                case "pxMode":
                    self.setPxMode(v)
                case "antialias":
                    self.opts["antialias"] = v
                case "hoverable":
                    self.opts["hoverable"] = bool(v)
                case "tip":
                    self.opts["tip"] = v
                case "useCache":
                    self.opts["useCache"] = v
                case "pen" | "brush" | "symbol" | "size":
                    set_method = getattr(self, f"set{k.capitalize()}")
                    set_method(
                        v,
                        update=False,
                        dataSet=new_data,
                        mask=kargs.get("mask", None),
                    )
                case "hoverPen" | "hoverBrush" | "hoverSymbol" | "hoverSize":
                    vh = kargs[k]
                    if k == "hoverPen":
                        vh = _mk_pen(vh)
                    elif k == "hoverBrush":
                        vh = _mk_brush(vh)
                    self.opts[k] = vh
                case "data":
                    self.setPointData(kargs["data"], dataSet=new_data)
                case _:
                    pass

        # Update the scatter plot item
        self.prepareGeometryChange()
        self.informViewBoundsChanged()
        self.bounds = [None, None]
        self.invalidate()
        self.updateSpots(new_data)
        self.sigPlotChanged.emit(self)


class TimeAxisItem(pg.AxisItem):
    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        pg.AxisItem.__init__(self, *args, **kwargs)
        self.setLabel("Time (hh:mm:ss)")

    def tickStrings(self, values: list[float], scale: float, spacing: float) -> list[str]:
        strings: list[str] = []
        for v in values:
            vs = v * scale
            minutes, seconds = divmod(int(vs), 60)
            hours, minutes = divmod(minutes, 60)
            vstr = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            strings.append(vstr)
        return strings
