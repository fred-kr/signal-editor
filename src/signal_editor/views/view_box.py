import typing as t

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

if t.TYPE_CHECKING:
    from pyqtgraph.GraphicsScene import mouseEvents


def _get_button_type(ev: "mouseEvents.MouseDragEvent") -> t.Literal["middle", "left", "left+control", "right", "unknown"]:
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

        button_type = _get_button_type(ev)

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
                ]
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
