import typing as t

import numpy as np
import pyqtgraph as pg
from pyqtgraph.GraphicsScene import mouseEvents
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QRectF, Qt, Signal

# class JupyterConsoleWidget(inprocess.QtInProcessRichJupyterWidget):
#     def __init__(self):
#         super().__init__()

#         self.kernel_manager: inprocess.QtInProcessKernelManager = (
#             inprocess.QtInProcessKernelManager()
#         )
#         self.kernel_manager.start_kernel()
#         self.kernel_client: jupyter_client.blocking.client.BlockingKernelClient = (
#             self.kernel_manager.client()
#         )
#         self.kernel_client.start_channels()
#         QApplication.instance().aboutToQuit.connect(self.shutdown_kernel)

#     def shutdown_kernel(self):
#         self.kernel_client.stop_channels()
#         self.kernel_manager.shutdown_kernel()


class ScatterPlotItemError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class CustomViewBox(pg.ViewBox):
    sig_selection_changed = Signal(QtGui.QPolygonF)

    def __init__(self, *args: t.Any, **kargs: t.Any) -> None:
        pg.ViewBox.__init__(self, *args, **kargs)
        self._selection_box: QtWidgets.QGraphicsRectItem | None = None
        self.mapped_peak_selection: QtGui.QPolygonF | None = None

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

    @t.override
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
                tr = pg.invertQTransform(self.childGroup.transform())
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


class CustomScatterPlotItem(pg.ScatterPlotItem):
    @t.override
    def addPoints(
        self,
        *args: t.Any,
        **kargs: t.Any,
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
