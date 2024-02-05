import typing as t

import numpy as np
import polars as pl
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets
from numpy.typing import NDArray

from .. import constants as c
from ..models.result import ManualPeakEdits
from ..views.graphic_items import CustomScatterPlotItem, CustomViewBox

if t.TYPE_CHECKING:
    from pyqtgraph.GraphicsScene import mouseEvents

    from ..app import SignalEditor


class PlotHandler(QtCore.QObject):
    """
    Handles plot creation and interactions.
    """

    sig_peaks_edited = QtCore.Signal(str, list)
    sig_signal_drawn = QtCore.Signal(int)
    sig_peaks_drawn = QtCore.Signal()
    sig_rate_drawn = QtCore.Signal()

    scatter_search_radius: int = 20

    def __init__(self, app: "SignalEditor", parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._app = app
        self._manual_peak_edits: dict[str, ManualPeakEdits] = {}
        self._line_clicked_tolerance: int = self._app.config.click_tolerance

        self._setup_plot_widgets()
        self._setup_plot_items()
        self._selector: pg.LinearRegionItem | None = None

        self._signal_item: pg.PlotDataItem | None = None
        self._scatter_item: CustomScatterPlotItem | None = None
        self._rate_item: pg.PlotDataItem | None = None
        self._rate_mean_item: pg.InfiniteLine | pg.PlotDataItem | None = None
        self._included_regions: list[pg.LinearRegionItem] = []
        self._excluded_regions: list[pg.LinearRegionItem] = []
        self._known_names: list[str] = []
        self._line_colors: list[str] = ["crimson", "steelblue", "yellow", "lightgreen"]
        self._connect_qt_signals()

    def _connect_qt_signals(self) -> None:
        self.sig_signal_drawn.connect(self.update_view_limits)

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
    def rate_mean_item(self) -> pg.InfiniteLine | pg.PlotDataItem | None:
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
    def combined_regions(self) -> list[pg.LinearRegionItem]:
        return self._included_regions + self._excluded_regions

    # endregion

    def get_manual_edits(self) -> dict[str, ManualPeakEdits]:
        return self._manual_peak_edits

    def _setup_plot_widgets(self) -> None:
        widget_layout = QtWidgets.QVBoxLayout()
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
            pl_item.showAxis("top", show=True)
            pl_item.getAxis("top").setScale(1/200)
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

        top_label = "<b>Time</b>"
        bottom_label = "<b>Index</b>"

        main_plot_item = self._pw_main.getPlotItem()
        rate_plot_item = self._pw_rate.getPlotItem()

        main_plot_item.setLabels(title=main_plot_title, left=main_left_label, bottom=bottom_label, top=(top_label, "seconds"))
        rate_plot_item.setLabels(title=rate_plot_title, left=rate_left_label, bottom=bottom_label, top=(top_label, "seconds"))

    @QtCore.Slot(object)
    def _on_mouse_moved(self, pos: QtCore.QPointF) -> None:
        if not hasattr(self._app, "data"):
            return
        try:
            cols = self._app.data.cas.data.select("index", "section_index", "temperature")
        except Exception:
            return
        pos_data_coords = self._pw_main.plotItem.vb.mapSceneToView(pos)
        index = np.clip(pos_data_coords.x(), 0, cols.height - 1, dtype=np.int32, casting="unsafe")
        vals = cols.filter(pl.col("section_index") == index)
        temp_text = f"<span style='color: yellow; font-size: 12pt; font-weight: bold;'>Temperature: {vals.get_column("temperature")[0]:.1f} °C</span>"
        self._temperature_label.setText(temp_text)

    @QtCore.Slot(int)
    def reset_view_range(self, len_data: int) -> None:
        for vb in self.view_boxes:
            vb.setRange(xRange=(0, len_data), disableAutoRange=False)

    @QtCore.Slot(int)
    def update_view_limits(self, len_data: int) -> None:
        for vb in self.view_boxes:
            vb.setLimits(
                xMin=-0.25 * len_data,
                xMax=1.25 * len_data,
                maxYRange=1e4,
                minYRange=0.5,
            )
        self.reset_view_range(len_data)

    @QtCore.Slot()
    def reset_plots(self) -> None:
        for pw in [self._pw_main, self._pw_rate]:
            pw.getPlotItem().clear()
            pw.getPlotItem().getViewBox().clear()

        self._manual_peak_edits.clear()
        self._signal_item = None
        self._scatter_item = None
        self._rate_item = None
        self._rate_mean_item = None
        self._temperature_label.setText("")

        self._setup_plot_items()

    @QtCore.Slot(bool)
    def toggle_region_overview(self, show: bool) -> None:
        for region in self.combined_regions:
            region.setVisible(show)

    def remove_region(self, bounds: "tuple[int, int] | SectionIndices") -> None:
        def remove_region_from_list(
            region_list: list[pg.LinearRegionItem], bounds: "tuple[int, int] | SectionIndices"
        ) -> None:
            for region in region_list:
                region_bounds = region.getRegion()
                if bounds[0] == region_bounds[0] and bounds[1] == region_bounds[1]:
                    region_list.remove(region)
                    self._pw_main.removeItem(region)
                    break

        remove_region_from_list(self._included_regions, bounds)
        remove_region_from_list(self._excluded_regions, bounds)

    def clear_regions(self) -> None:
        for region in self.combined_regions:
            if region in self._pw_main.getPlotItem().items:
                self._pw_main.removeItem(region)
        self._included_regions.clear()
        self._excluded_regions.clear()

    def show_section_selector(
        self,
        section_type: t.Literal["included", "excluded"],
        bounds: "tuple[int, int] | SectionIndices",
    ) -> None:
        section_style = c.SECTION_STYLES[section_type]
        view_x = self._pw_main.plotItem.vb.viewRange()[0]
        span = view_x[1] - view_x[0]
        initial_limits = (view_x[0] + span * 0.25, view_x[0] + span * 0.75)
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
            if hasattr(self, "_selector_type"):
                del self._selector_type

    @QtCore.Slot(int, int)
    def mark_section(self, lower: int, upper: int) -> None:
        if not hasattr(self, "_selector_type"):
            return
        section_type = self._selector_type
        if section_type == "included":
            r, g, b = 0, 250, 50
            region_list = self._included_regions
        elif section_type == "excluded":
            r, g, b = 250, 0, 50
            region_list = self._excluded_regions
        else:
            return
        marked_region = pg.LinearRegionItem(
            values=(lower, upper),
            brush=(r, g, b, 25),
            pen=dict(color=(r, g, b, 255), width=1),
            movable=False,
        )
        if not self._app.action_section_overview.isChecked():
            marked_region.hide()
        marked_region.setZValue(10)
        region_list.append(marked_region)
        self._pw_main.addItem(marked_region)
        self.remove_section_selector()

    def draw_signal(self, sig: NDArray[np.float64], name: str) -> None:
        if name not in self._known_names:
            self._known_names.append(name)
        signal_item = self._signal_item
        if signal_item is None:
            self._create_signal_data_item(sig, name)
        else:
            if self._scatter_item is not None:
                self._pw_main.removeItem(self._scatter_item)
                self._scatter_item = None
            if self._rate_item is not None:
                self._pw_rate.removeItem(self._rate_item)
                self._rate_item = None
            if self._rate_mean_item is not None:
                self._pw_rate.removeItem(self._rate_mean_item)
                self._rate_mean_item = None
            signal_item.setData(sig)

        self.sig_signal_drawn.emit(sig.shape[0])

    def _create_signal_data_item(self, sig: NDArray[np.float64], name: str) -> None:
        signal_item = pg.PlotDataItem(
            sig,
            pen=self._line_colors[self._known_names.index(name)],
            skipFiniteCheck=True,
            autoDownSample=True,
            name=f"Signal ({name})",
        )
        signal_item.curve.setSegmentedLineMode("on")
        signal_item.curve.setClickable(True, self._line_clicked_tolerance)
        self._pw_main.addItem(signal_item)
        signal_item.sigClicked.connect(self.add_scatter)
        self._signal_item = signal_item

    def draw_peaks(
        self,
        x_values: NDArray[np.uint32],
        y_values: NDArray[np.float64],
        name: str,
        brush_color: str = "darkgoldenrod",
        hover_brush_color: str = "red",
    ) -> None:
        scatter_item = self._scatter_item
        if scatter_item is None:
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
            self._pw_main.addItem(scatter_item)
            scatter_item.sigClicked.connect(self.remove_scatter)
            self._scatter_item = scatter_item
        else:
            scatter_item.setData(x=x_values, y=y_values)

        self.sig_peaks_drawn.emit()

    def draw_rate(
        self,
        rate_values: NDArray[np.float64],
        name: str,
        pen_color: str = "green",
        mean_pen_color: str = "darkgoldenrod",
    ) -> None:
        rate_mean_val = np.mean(rate_values, dtype=np.int32)

        rate_item = self._rate_item
        rate_mean_item = self._rate_mean_item
        legend = self._pw_rate.getPlotItem().addLegend()

        if rate_item is None or rate_mean_item is None:
            rate_item = pg.PlotDataItem(
                rate_values,
                pen=pen_color,
                autoDownsample=True,
                skipFiniteCheck=True,
                name=f"Rate ({name})",
            )
            rate_mean_item = pg.InfiniteLine(
                rate_mean_val,
                0,
                pen={"color": mean_pen_color, "width": 2.5, "style": QtGui.Qt.PenStyle.DashLine},
                name=f"Mean Rate: {int(rate_mean_val)} bpm",
            )
            rate_mean_item.setZValue(1e3)
            rate_mean_item.opts = {
                "pen": pg.mkPen(
                    {"color": mean_pen_color, "width": 2.5, "style": QtGui.Qt.PenStyle.DashLine}
                )
            }
            legend.clear()
            self._pw_rate.addItem(rate_item)
            self._pw_rate.addItem(rate_mean_item)
            legend.addItem(rate_mean_item, name=f"Mean Rate: {int(rate_mean_val)} bpm")
            self._rate_item = rate_item
            self._rate_mean_item = rate_mean_item
        else:
            legend.clear()
            rate_item.setData(rate_values)
            rate_mean_item.setValue(rate_mean_val)
            rate_mean_item.opts = {
                "pen": pg.mkPen(
                    {"color": mean_pen_color, "width": 2.5, "style": QtGui.Qt.PenStyle.DashLine}
                )
            }
            legend.addItem(rate_item, name=f"Rate ({name})")
            legend.addItem(rate_mean_item, name=f"Mean Rate: {int(rate_mean_val)} bpm")
        self.sig_rate_drawn.emit()

    @QtCore.Slot(object, object, object)
    def remove_scatter(
        self,
        sender: CustomScatterPlotItem,
        points: np.ndarray[pg.SpotItem, t.Any],
        ev: "mouseEvents.MouseClickEvent",
    ) -> None:
        ev.accept()
        if len(points) == 0:
            return

        spot_item = points[0]
        to_remove_index = spot_item.index()
        scatter_plot = self._scatter_item
        if scatter_plot is not None:
            scatter_data = scatter_plot.data
            new_points_x = np.delete(scatter_data["x"], to_remove_index)
            new_points_y = np.delete(scatter_data["y"], to_remove_index)
            scatter_plot.setData(x=new_points_x, y=new_points_y)
            peak_edit_x = int(spot_item.pos().x())

            self.sig_peaks_edited.emit("remove", [peak_edit_x])

    @QtCore.Slot(object, object)
    def add_scatter(self, sender: pg.PlotCurveItem, ev: "mouseEvents.MouseClickEvent") -> None:
        ev.accept()
        click_x = int(ev.pos().x())
        click_y = ev.pos().y()

        signal_item = self._signal_item
        scatter_item = self._scatter_item

        if signal_item is None or scatter_item is None:
            return

        x_data = signal_item.xData
        y_data = signal_item.yData
        if x_data is None or y_data is None:
            return

        left_index = np.searchsorted(x_data, click_x - self.scatter_search_radius, side="left")
        right_index = np.searchsorted(x_data, click_x + self.scatter_search_radius, side="right")

        valid_x_values = x_data[left_index:right_index]
        valid_y_values = y_data[left_index:right_index]

        # Find the index of the nearest extreme point to the click position
        extreme_index = left_index + np.argmin(np.abs(valid_x_values - click_x))
        extreme_value = valid_y_values[np.argmin(np.abs(valid_x_values - click_x))]

        # Find the index of the nearest extreme point to the click position in the y direction
        y_extreme_index = left_index + np.argmin(np.abs(valid_y_values - click_y))
        y_extreme_value = valid_y_values[np.argmin(np.abs(valid_y_values - click_y))]

        # Use the index of the nearest extreme point in the y direction if it is closer to the click position
        if np.abs(y_extreme_value - click_y) < np.abs(extreme_value - click_y):
            extreme_index = y_extreme_index
            extreme_value = y_extreme_value

        if extreme_index in scatter_item.data["x"]:
            return

        x_new, y_new = x_data[extreme_index], extreme_value
        scatter_item.addPoints(x=x_new, y=y_new)
        self.sig_peaks_edited.emit("add", [int(x_new)])

    @QtCore.Slot()
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

        mask = (
            (scatter_x < rect_x)
            | (scatter_x > rect_x + rect_width)
            | (scatter_y < rect_y)
            | (scatter_y > rect_y + rect_height)
        )
        scatter_item.setData(x=scatter_x[mask], y=scatter_y[mask])
        self.sig_peaks_edited.emit("remove", scatter_x[~mask].astype(int).tolist())
        vb.selection_box = None

    @QtCore.Slot()
    def show_selection_rect(self) -> None:
        vb = self._pw_main.getPlotItem().getViewBox()
        if vb.selection_box is None:
            return
        vb.selection_box.setVisible(not vb.selection_box.isVisible())
        vb.selection_box.setEnabled(not vb.selection_box.isEnabled())