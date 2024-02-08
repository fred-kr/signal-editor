import typing as t

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

if t.TYPE_CHECKING:
    from pyqtgraph.GraphicsScene import mouseEvents


class ScatterPlotItemError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class CustomViewBox(pg.ViewBox):
    """
    Custom `pyqtgraph.ViewBox` subclass that makes plot editing easier.
    """

    def __init__(self, *args: t.Any, **kargs: t.Any) -> None:
        pg.ViewBox.__init__(self, *args, **kargs)
        self._selection_box: QtWidgets.QGraphicsRectItem | None = None
        self.mapped_selection_rect: QtGui.QPolygonF | None = None

    @property
    def selection_box(self) -> QtWidgets.QGraphicsRectItem:
        if self._selection_box is None:
            selection_box = QtWidgets.QGraphicsRectItem(0, 0, 1, 1)
            selection_box.setPen(pg.mkPen((50, 100, 200), width=1))
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
        lastPos = ev.lastPos()
        dif = (pos - lastPos) * np.array([-1, -1])

        mouseEnabled = np.array(self.state["mouseEnabled"], dtype=np.float64)
        mask = mouseEnabled.copy()
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

            x = s[0] if mouseEnabled[0] == 1 else None
            y = s[1] if mouseEnabled[1] == 1 else None

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
        # Extract the required parameters
        spots = kargs.get("spots")
        x = kargs.get("x")
        y = kargs.get("y")
        pos = kargs.get("pos")

        # Handle variable input arguments
        if len(args) == 1:
            kargs["spots"] = args[0]
        elif len(args) == 2:
            kargs["x"] = args[0]
            kargs["y"] = args[1]
        elif len(args) > 2:
            raise ScatterPlotItemError("Only accepts up to two non-keyword arguments.")

        # Handle 'pos' parameter
        if pos is not None:
            if isinstance(pos, np.ndarray):
                kargs["x"] = pos[:, 0]
                kargs["y"] = pos[:, 1]
            else:
                x = [p.x() if isinstance(p, QtCore.QPointF) else p[0] for p in pos]
                y = [p.y() if isinstance(p, QtCore.QPointF) else p[1] for p in pos]
                kargs["x"] = x
                kargs["y"] = y

        # Calculate number of points
        numPts = (
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
        oldData = self.data
        self.data = np.empty(len(oldData) + numPts, dtype=self.data.dtype)
        self.data[: len(oldData)] = oldData
        newData = self.data[len(oldData) :]
        newData["size"] = -1
        newData["visible"] = True

        # Handle 'spots' parameter
        if spots is not None:
            for i, spot in enumerate(spots):
                for k, v in spot.items():
                    if k == "pos":
                        pos = v
                        if isinstance(pos, QtCore.QPointF):
                            x, y = pos.x(), pos.y()
                        else:
                            x, y = pos[0], pos[1]
                        newData[i]["x"] = x
                        newData[i]["y"] = y
                    elif k == "pen":
                        newData[i][k] = pg.mkPen(v)
                    elif k == "brush":
                        newData[i][k] = pg.mkBrush(v)
                    elif k in ["x", "y", "size", "symbol", "data"]:
                        newData[i][k] = v
                    else:
                        raise ScatterPlotItemError(f"Unknown spot parameter: {k}")
        # Handle 'y' parameter
        elif y is not None:
            newData["x"] = x
            newData["y"] = y

        # Update the scatter plot item properties based on keyword arguments
        for k, v in kargs.items():
            if k == "name":
                self.opts["name"] = v
            elif k == "pxMode":
                self.setPxMode(v)
            elif k == "antialias":
                self.opts["antialias"] = v
            elif k == "hoverable":
                self.opts["hoverable"] = bool(v)
            elif k == "tip":
                self.opts["tip"] = v
            elif k == "useCache":
                self.opts["useCache"] = v

        # Set point-specific properties
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

        # Set point data
        if "data" in kargs:
            self.setPointData(kargs["data"], dataSet=newData)

        # Update the scatter plot item
        self.prepareGeometryChange()
        self.informViewBoundsChanged()
        self.bounds = [None, None]
        self.invalidate()
        self.updateSpots(newData)
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