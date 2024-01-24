from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
)

import numpy as np
import pyqtgraph as pg
from numpy.typing import NDArray
from PySide6.QtCore import QObject, QPointF, Qt, Signal, Slot
from PySide6.QtWidgets import QVBoxLayout

from ..models.result import ManualPeakEdits
from ..views.custom_widgets import CustomScatterPlotItem, CustomViewBox

if TYPE_CHECKING:
    from pyqtgraph.GraphicsScene import mouseEvents

    from ..app import SignalEditor


SECTION_STYLES = {
    "included": {
        "brush": (0, 200, 100, 75),
        "pen": {"color": "darkgoldenrod", "width": 1},
        "hoverBrush": (0, 200, 100, 50),
        "hoverPen": {"color": "gold", "width": 3.5},
    },
    "excluded": {
        "brush": (200, 0, 100, 75),
        "pen": {"color": "darkgoldenrod", "width": 1},
        "hoverBrush": (200, 0, 100, 50),
        "hoverPen": {"color": "gold", "width": 3.5},
    },
}


class PlotHandler(QObject):
    """
    Handles plot creation and interactions.
    """

    sig_peaks_edited = Signal(str, list[int])

    def __init__(self, app: "SignalEditor", parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._app = app
        self._manual_peak_edits: dict[str, ManualPeakEdits] = {}
        self._line_clicked_tolerance: int = 80

        self._setup_plot_widgets()
        self._setup_plot_items()
        self._selector: pg.LinearRegionItem | None = None

        self._signal_item: pg.PlotDataItem | None = None
        self._scatter_item: CustomScatterPlotItem | None = None
        self._rate_item: pg.PlotDataItem | None = None
        self._rate_mean_item: pg.InfiniteLine | None = None

    # region Properties
    @property
    def region_selector(self) -> pg.LinearRegionItem | None:
        return self._selector

    @property
    def signal_item(self) -> pg.PlotDataItem | None:
        return self._signal_item

    @property
    def scatter_item(self) -> CustomScatterPlotItem | None:
        return self._scatter_item

    @property
    def rate_item(self) -> pg.PlotDataItem | None:
        return self._rate_item

    @property
    def rate_mean_item(self) -> pg.InfiniteLine | None:
        return self._rate_mean_item

    @property
    def main_plot_widget(self) -> pg.PlotWidget:
        return self._pw_main

    @property
    def rate_plot_widget(self) -> pg.PlotWidget:
        return self._pw_rate

    @property
    def view_boxes(self) -> tuple[CustomViewBox, pg.ViewBox]:
        return self._pw_main.getPlotItem().getViewBox(), self._pw_rate.getPlotItem().getViewBox()

    @property
    def is_selecting_region(self) -> bool:
        return isinstance(self._selector, pg.LinearRegionItem)
    # endregion

    def get_manual_edits(self) -> dict[str, ManualPeakEdits]:
        return self._manual_peak_edits

    def _setup_plot_widgets(self) -> None:
        widget_layout = QVBoxLayout()
        main_plot_widget = pg.PlotWidget(viewBox=CustomViewBox(name="main_plot"), useOpenGL=True)
        rate_plot_widget = pg.PlotWidget(viewBox=pg.ViewBox(name="rate_plot"), useOpenGL=True)
        widget_layout.addWidget(main_plot_widget)
        widget_layout.addWidget(rate_plot_widget)

        self._app.container_plots.setLayout(widget_layout)
        self._pw_main = main_plot_widget
        self._pw_rate = rate_plot_widget

    def _setup_plot_items(self) -> None:
        self._setup_plot_labels()
        for pw in [self._pw_main, self._pw_rate]:
            pl_item = pw.getPlotItem()
            vb = pl_item.getViewBox()

            pl_item.showGrid(x=False, y=True)
            pl_item.setDownsampling(auto=True)
            pl_item.setClipToView(True)
            pl_item.addLegend(colCount=2)
            pl_item.addLegend().anchor(itemPos=(0, 1), parentPos=(0, 1), offset=(5, -5))
            pl_item.setMouseEnabled(x=True, y=False)
            vb.enableAutoRange("y")
            vb.setAutoVisible(y=True)

        self._pw_main.getPlotItem().getViewBox().setXLink("rate_plot")
        self._pw_rate.getPlotItem().getViewBox().setMouseMode(pg.ViewBox.RectMode)
        self._pw_main.getPlotItem().scene().sigMouseMoved.connect(self._on_mouse_moved)
        self._temperature_label: pg.LabelItem = pg.LabelItem(parent=self._pw_main.getPlotItem())

    def _setup_plot_labels(self) -> None:
        main_plot_title = "<b>Input Signal</b>"
        main_left_label = "<b>Signal Amplitude</b>"

        rate_plot_title = "<b>Rate (interpolated)</b>"
        rate_left_label = "<b>bpm</b>"

        bottom_label = "<b>Index (n samples)</b>"

        main_plot_item = self._pw_main.getPlotItem()
        rate_plot_item = self._pw_rate.getPlotItem()

        main_plot_item.setLabels(title=main_plot_title, left=main_left_label, bottom=bottom_label)
        rate_plot_item.setLabels(title=rate_plot_title, left=rate_left_label, bottom=bottom_label)

    @Slot(QPointF)
    def _on_mouse_moved(self, pos: QPointF) -> None:
        if not hasattr(self._app, "data"):
            return
        try:
            temp_col = self._app.data.cas.data.get_column("temperature")
        except Exception:
            return
        pos_data_coords = self._pw_main.plotItem.vb.mapSceneToView(pos)
        index = np.clip(
            pos_data_coords.x(), 0, temp_col.len() - 1, dtype=np.int32, casting="unsafe"
        )
        temp_val = temp_col.gather(index).to_numpy(zero_copy_only=True)[0]
        temp_text = f"<span style='color: crimson; font-size: 12pt; font-weight: bold;'>Temperature: {temp_val:.1f} °C</span>"
        self._temperature_label.setText(temp_text)

    @Slot(int, float, float)
    def reset_view_range(self, len_data: int, min_data: float, max_data: float) -> None:
        for vb in self.view_boxes:
            vb.setRange(xRange=(0, len_data), yRange=(min_data, max_data), disableAutoRange=False)
            
    def update_view_limits(self, len_data: int, min_data: float, max_data: float) -> None:
        for vb in self.view_boxes:
            vb.setLimits(
                xMin=-0.25 * len_data,
                xMax=1.25 * len_data,
                maxYRange=np.abs(max_data - min_data) * 1.25,
                minYRange=0.1,
            )
        self.reset_view_range(len_data, min_data, max_data)

    @Slot()
    def reset_plots(self) -> None:
        for pw in [self._pw_main, self._pw_rate]:
            pw.getPlotItem().addLegend().clear()
            pw.getPlotItem().clear()
            pw.getPlotItem().getViewBox().clear()

        self._manual_peak_edits.clear()
        self._signal_item = None
        self._scatter_item = None
        self._rate_item = None
        self._rate_mean_item = None

        self._setup_plot_items()

    def show_section_selector(self, section_type: Literal["included", "excluded"], bounds: tuple[int, int]) -> None:
        section_style = SECTION_STYLES[section_type]
        span = abs(bounds[1] - bounds[0])
        initial_limits = (0, span / 3)
        self.remove_section_selector()

        selector = pg.LinearRegionItem(
            values=initial_limits,
            bounds=bounds,
            brush=section_style["brush"],
            pen=section_style["pen"],
            hoverBrush=section_style["hoverBrush"],
            hoverPen=section_style["hoverPen"],
        )
        selector.setZValue(1e3)
        for line in selector.lines:
            line.addMarker("<|>", position=0.5, size=12)

        self._pw_main.addItem(selector)
        self._selector = selector
        self._selector_type = section_type

    def remove_section_selector(self) -> None:
        if selector := self._selector:
            self._pw_main.removeItem(selector)
        self._selector = None
        del self._selector_type

    def draw_signal(self, sig: NDArray[np.float64], name: str, pen_color: str = "crimson") -> None:
        signal_item = pg.PlotDataItem(
            sig, pen=pen_color, skipFiniteCheck=True, autoDownSample=True, name=f"Signal ({name})"
        )
        signal_item.curve.setSegmentedLineMode("on")
        signal_item.curve.setClickable(True, self._line_clicked_tolerance)

        if self._signal_item is not None:
            self._pw_main.removeItem(self._signal_item)
        if self._scatter_item is not None:
            self._pw_main.removeItem(self._scatter_item)
        if self._rate_item is not None:
            self._pw_rate.removeItem(self._rate_item)
        if self._rate_mean_item is not None:
            self._pw_rate.removeItem(self._rate_mean_item)

        self._pw_main.addItem(signal_item)
        signal_item.sigClicked.connect(self.add_scatter)
        self._signal_item = signal_item
        self.update_view_limits(len(sig), sig.min(), sig.max())

    def draw_peaks(
        self,
        x_values: NDArray[np.uint32],
        y_values: NDArray[np.float64],
        name: str,
        **kwargs: str | tuple[int, ...],
    ) -> None:
        brush_color = kwargs.get("brush", "darkgoldenrod")
        hover_brush_color = kwargs.get("hoverBrush", "cyan")

        scatter_item = CustomScatterPlotItem(
            x=x_values,
            y=y_values,
            pxMode=True,
            size=10,
            pen=None,
            brush=brush_color,
            useCache=True,
            name=f"Peaks ({name})",
            hoverable=True,
            hoverPen="black",
            hoverSymbol="x",
            hoverBrush=hover_brush_color,
            hoverSize=15,
            tip=None,
        )
        scatter_item.setZValue(60)

        if self._scatter_item is not None:
            self._pw_main.removeItem(self._scatter_item)

        self._pw_main.addItem(scatter_item)
        scatter_item.sigClicked.connect(self.remove_scatter)
        self._scatter_item = scatter_item

    def draw_rate(
        self, rate_values: NDArray[np.float64], name: str, **kwargs: str | tuple[int, ...]
    ) -> None:
        rate_mean_val = np.mean(rate_values, dtype=np.int32)
        pen_color = kwargs.get("pen", "lightgreen")
        mean_pen_color = kwargs.get("meanPen", "goldenrod")

        rate_item = pg.PlotDataItem(
            rate_values,
            pen=pen_color,
            autoDownsample=True,
            skipFiniteCheck=True,
            name=f"Rate ({name})",
        )

        rate_mean_item = pg.InfiniteLine(
            rate_mean_val,
            angle=0,
            pen=dict(color=mean_pen_color, width=2, style=Qt.PenStyle.DashLine),
            name=f"Mean Rate ({name})",
        )

        if self._rate_item is not None:
            self._pw_rate.removeItem(self._rate_item)
        if self._rate_mean_item is not None:
            self._pw_rate.removeItem(self._rate_mean_item)

        self._pw_rate.getPlotItem().addLegend().clear()

        self._pw_rate.addItem(rate_item)
        self._pw_rate.addItem(rate_mean_item)
        self._pw_rate.getPlotItem().addLegend().addItem(
            pg.PlotDataItem(np.array([0, 1]), pen=rate_mean_item.pen, skipFiniteCheck=True),
            f"Mean Rate: {int(rate_mean_val)} bpm",
        )
        self._rate_item = rate_item
        self._rate_mean_item = rate_mean_item

    @Slot(object, object, object)
    def remove_scatter(
        self,
        sender: CustomScatterPlotItem,
        points: np.ndarray[pg.SpotItem, Any],
        ev: "mouseEvents.MouseClickEvent",
    ) -> None:
        ev.accept()
        if len(points) == 0:
            return

        spot_item = points[0]
        to_remove_index = spot_item.index()
        if scatter_plot := self._scatter_item:
            new_points_x = np.delete(scatter_plot.data["x"], to_remove_index)
            new_points_y = np.delete(scatter_plot.data["y"], to_remove_index)
            scatter_plot.setData(x=new_points_x, y=new_points_y)
            peak_edit_x = int(spot_item.pos().x())

            self.sig_peaks_edited.emit("remove", [peak_edit_x])
            
    @Slot(object, object)
    def add_scatter(self, sender: pg.PlotCurveItem, ev: "mouseEvents.MouseClickEvent") -> None:
        ev.accept()
        click_x = int(ev.pos().x())

        signal_item = self._signal_item
        scatter_item = self._scatter_item

        if signal_item is None or scatter_item is None:
            return

        x_data = signal_item.xData
        y_data = signal_item.yData
        if x_data is None or y_data is None:
            return

        search_radius = 20

        valid_indices = np.where(
            (x_data >= click_x - search_radius) & (x_data <= click_x + search_radius)
        )[0]

        valid_y_values = y_data[valid_indices]

        max_y_index = valid_indices[np.argmax(valid_y_values)]
        min_y_index = valid_indices[np.argmin(valid_y_values)]

        if np.abs(x_data[max_y_index] - click_x) > np.abs(x_data[min_y_index] - click_x):
            use_index = min_y_index
            use_val = valid_y_values.min()
        else:
            use_index = max_y_index
            use_val = valid_y_values.max()
        if use_index in scatter_item.data["x"]:
            return

        x_new, y_new = x_data[use_index], use_val
        scatter_item.addPoints(x=x_new, y=y_new)
        self.sig_peaks_edited.emit("add", [int(x_new)])

    @Slot()
    def remove_selected_scatter(self) -> None:
        vb = self._pw_main.getPlotItem().getViewBox()
        if vb.mapped_peak_selection is None:
            return
        scatter_item = self._scatter_item
        if scatter_item is None:
            return

        rect_x, rect_y, rect_width, rect_height = vb.mapped_peak_selection.boundingRect().getRect()

        scatter_x, scatter_y = scatter_item.getData()
        if scatter_x.size == 0 or scatter_y.size == 0:
            return

        to_remove = np.argwhere(
            (scatter_x >= rect_x)
            & (scatter_x <= rect_x + rect_width)
            & (scatter_y >= rect_y)
            & (scatter_y <= rect_y + rect_height)
        ).flatten()

        if len(to_remove) == 0:
            return

        scatter_item.setData(x=np.delete(scatter_x, to_remove), y=np.delete(scatter_y, to_remove))
        self.sig_peaks_edited.emit("remove", list(scatter_x[to_remove].astype(int)))
        vb.selection_box = None

    @Slot()
    def show_selection_rect(self) -> None:
        vb = self._pw_main.getPlotItem().getViewBox()
        if vb.selection_box is None:
            return
        vb.selection_box.setVisible(not vb.selection_box.isVisible())
        vb.selection_box.setEnabled(not vb.selection_box.isEnabled())

    def _mark_region(self, lower: int, upper: int, r: int, g: int, b: int) -> None:
        static_region = pg.LinearRegionItem(
            values=(lower, upper),
            brush=(r, g, b, 50),
            pen=dict(color=(r, g, b, 255), width=2),
            movable=False,
        )
        self._pw_main.getPlotItem().getViewBox().addItem(static_region)

    @Slot(int, int)
    def mark_section(self, lower: int, upper: int) -> None:
        if not hasattr(self, "_selector_type"):
            return
        section_type = self._selector_type
        if section_type == "included":
            r, g, b = 0, 200, 100
        elif section_type == "excluded":
            r, g, b = 200, 0, 100
        else:
            return
        marked_region = pg.LinearRegionItem(
            values=(lower, upper),
            brush=(r, g, b, 50),
            pen=dict(color=(r, g, b, 255), width=2),
            movable=False,
        )
        self._pw_main.addItem(marked_region)
        self.remove_section_selector()