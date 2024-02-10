import typing as t

import numpy as np
import numpy.typing as npt
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ..models.result import ManualPeakEdits
from ..views.graphic_items import CustomScatterPlotItem, CustomViewBox, TimeAxisItem

if t.TYPE_CHECKING:
    from pyqtgraph.GraphicsScene import mouseEvents

    from ..app import SignalEditor
    from ..models.section import SectionIndices


class PlotHandler(QtCore.QObject):
    """
    Handles plot creation and interactions.
    """

    sig_peaks_edited = QtCore.Signal(str, list)
    sig_signal_drawn = QtCore.Signal(int)
    sig_peaks_drawn = QtCore.Signal()
    sig_rate_drawn = QtCore.Signal()

    scatter_search_radius: int = 20

    _known_names: list[str] = ["hbr", "ventilation"]
    _signal_colors: list[str] = ["crimson", "steelblue", "yellow", "lightgreen"]

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
        self._regions: list[pg.LinearRegionItem] = []
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
    def view_boxes(self) -> tuple[CustomViewBox | pg.ViewBox, pg.ViewBox]:
        return self._pw_main.getPlotItem().getViewBox(), self._pw_rate.getPlotItem().getViewBox()

    @property
    def plot_items(self) -> tuple[pg.PlotItem, pg.PlotItem]:
        return self._pw_main.getPlotItem(), self._pw_rate.getPlotItem()

    @property
    def regions(self) -> list[pg.LinearRegionItem]:
        return self._regions

    # endregion

    def get_manual_edits(self) -> dict[str, ManualPeakEdits]:
        return self._manual_peak_edits

    def _setup_plot_widgets(self) -> None:
        widget_layout = QtWidgets.QVBoxLayout()
        widget_layout.setContentsMargins(0, 0, 0, 0)
        widget_layout.setSpacing(2)
        main_plot_widget = pg.PlotWidget(viewBox=CustomViewBox(name="main_plot"), useOpenGL=True)
        rate_plot_widget = pg.PlotWidget(viewBox=pg.ViewBox(name="rate_plot"), useOpenGL=True)
        widget_layout.addWidget(main_plot_widget)
        widget_layout.addWidget(rate_plot_widget)

        self._app.container_plots.setLayout(widget_layout)
        self._pw_main = main_plot_widget
        self._pw_rate = rate_plot_widget

    def _setup_plot_items(self) -> None:
        for plt in self.plot_items:
            vb = plt.getViewBox()

            plt.setAxisItems(
                {
                    "top": TimeAxisItem(orientation="top"),
                }
            )
            plt.showGrid(x=False, y=True)
            plt.setDownsampling(auto=True)
            plt.setClipToView(True)
            plt.addLegend(colCount=2)
            plt.addLegend().anchor(itemPos=(0, 1), parentPos=(0, 1), offset=(5, -5))
            plt.setMouseEnabled(x=True, y=False)
            vb.enableAutoRange("y")
            vb.setAutoVisible(y=True)
            plt.scene().sigMouseMoved.connect(self._on_mouse_moved)

        self._setup_plot_labels()

        self._pw_main.getPlotItem().getViewBox().setXLink("rate_plot")
        self._temperature_label = pg.LabelItem(parent=self._pw_main.getPlotItem())
        self._bpm_label = pg.LabelItem(parent=self._pw_rate.getPlotItem())

    @QtCore.Slot(int)
    def update_time_axis_scale(self, sfreq: int) -> None:
        for plt in self.plot_items:
            plt.getAxis("top").setScale(1 / sfreq)

    def _setup_plot_labels(self) -> None:
        # main_plot_title = "<b>Input Signal</b>"
        main_left_label = "<b>Signal Amplitude</b>"

        # rate_plot_title = "<b>Rate (interpolated)</b>"
        rate_left_label = "<b>Rate (bpm)</b>"

        # top_label = "<b>Time</b>"
        bottom_label = "<b>Index</b>"

        main_plot_item = self._pw_main.getPlotItem()
        rate_plot_item = self._pw_rate.getPlotItem()

        main_plot_item.setLabels(
            left=main_left_label,
            bottom=bottom_label,
        )
        rate_plot_item.setLabels(
            left=rate_left_label,
            bottom=bottom_label,
        )

    @QtCore.Slot(object)
    def _on_mouse_moved(self, pos: QtCore.QPointF) -> None:
        if not hasattr(self._app, "data"):
            return
        if self._app.data.cas is None:
            return
        mapped_pos = self._pw_main.plotItem.vb.mapSceneToView(pos)
        cas_upper_bound = self._app.data.cas.data.height
        i = np.clip(mapped_pos.x(), 0, cas_upper_bound - 1, dtype=np.int32, casting="unsafe")

        try:
            temp_val = self._app.data.cas.data.get_column("temperature").item(i)
        except Exception:
            temp_val = np.nan

        try:
            rate_val = self._app.data.cas.rate_interp[i]
        except Exception:
            rate_val = np.nan

        temp_text = f"<span style='color: gold; font-size: 12pt; font-weight: bold;'>Temperature: {temp_val:.1f} °C</span>"
        self._temperature_label.setText(temp_text)
        rate_text = f"<span style='color: lightgreen; font-size: 12pt; font-weight: bold;'>Rate: {rate_val:.0f} bpm</span>"
        self._bpm_label.setText(rate_text)

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
                maxYRange=1e5,
                minYRange=0.5,
            )
        self.reset_view_range(len_data)

    @QtCore.Slot()
    def reset_plots(self) -> None:
        for plt in self.plot_items:
            plt.clear()
            plt.getViewBox().clear()

        self._manual_peak_edits.clear()
        self._signal_item = None
        self._scatter_item = None
        self._rate_item = None
        self._rate_mean_item = None
        self._temperature_label.setText("")

        self._setup_plot_items()

    @QtCore.Slot(bool)
    def toggle_region_overview(self, show: bool) -> None:
        for region in self.regions:
            region.setVisible(show)

    def remove_region(self, bounds: "tuple[int, int] | SectionIndices") -> None:
        for region in self.regions:
            region_bounds = region.getRegion()
            if np.allclose(bounds, region_bounds):
                self.regions.remove(region)
                self._pw_main.removeItem(region)
                break

    def clear_regions(self) -> None:
        for region in self.regions:
            if region in self._pw_main.getPlotItem().items:
                self._pw_main.removeItem(region)
        self.regions.clear()

    def show_section_selector(
        self,
        bounds: "tuple[int, int] | SectionIndices",
    ) -> None:
        view_x = self._pw_main.plotItem.vb.viewRange()[0]
        span = view_x[1] - view_x[0]
        initial_limits = (view_x[0], view_x[0] + 0.5 * span)
        self.remove_section_selector()

        selector = pg.LinearRegionItem(
            values=initial_limits,
            bounds=bounds,
            brush=(0, 200, 100, 75),
            pen={"color": "darkgoldenrod", "width": 1},
            hoverBrush=(0, 200, 100, 30),
            hoverPen={"color": "gold", "width": 3.5},
        )
        selector.setZValue(1e3)
        for line in selector.lines:
            line.addMarker("<|>", position=0.5, size=12)

        self._pw_main.addItem(selector)
        self._selector = selector

    def remove_section_selector(self) -> None:
        selector = self._selector
        if selector is not None:
            self._pw_main.removeItem(selector)
            self._selector = None

    @QtCore.Slot(int, int)
    def mark_section(self, lower: int, upper: int) -> None:
        r, g, b = 0, 200, 100
        marked_region = pg.LinearRegionItem(
            values=(lower, upper),
            brush=(r, g, b, 25),
            pen=dict(color=(r, g, b, 255), width=1),
            movable=False,
        )
        if not self._app.action_section_overview.isChecked():
            marked_region.hide()
        marked_region.setZValue(10)
        self.regions.append(marked_region)
        self._pw_main.addItem(marked_region)
        self.remove_section_selector()

    def draw_signal(self, sig: npt.NDArray[np.float64], name: str) -> None:
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

    def _create_signal_data_item(self, sig: npt.NDArray[np.float64], name: str) -> None:
        signal_item = pg.PlotDataItem(
            sig,
            pen=self._signal_colors[self._known_names.index(name)],
            skipFiniteCheck=True,
            autoDownSample=True,
            name=f"Signal ({name})",
        )
        signal_item.curve.setSegmentedLineMode("auto")
        signal_item.curve.setClickable(True, self._line_clicked_tolerance)
        self._pw_main.addItem(signal_item)
        signal_item.sigClicked.connect(self.add_scatter)
        self._signal_item = signal_item

    def draw_peaks(
        self,
        x_values: npt.NDArray[np.uint32],
        y_values: npt.NDArray[np.float64],
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
        rate_values: npt.NDArray[np.float64],
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
        points: t.Sequence[pg.SpotItem],
        ev: "mouseEvents.MouseClickEvent",
    ) -> None:
        ev.accept()
        if len(points) == 0:
            return
        spot_item = points[0]
        to_remove_val = int(spot_item.pos().x())
        to_remove_index = spot_item.index()
        scatter_plot = self._scatter_item
        if scatter_plot is None:
            return
        self._remove_scatter(to_remove_index, scatter_plot)
        self.sig_peaks_edited.emit("remove", [to_remove_val])

    def _remove_scatter(self, to_remove_index: int, scatter_plot: CustomScatterPlotItem) -> None:
        scatter_data = scatter_plot.data
        new_points_x = np.delete(scatter_data["x"], to_remove_index)
        new_points_y = np.delete(scatter_data["y"], to_remove_index)
        scatter_plot.setData(x=new_points_x, y=new_points_y)

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
        vb: CustomViewBox = self._pw_main.getPlotItem().getViewBox()
        if vb.mapped_selection_rect is None:
            return
        scatter_item = self._scatter_item
        if scatter_item is None:
            return

        r = vb.mapped_selection_rect.boundingRect()
        rx, ry, rw, rh = r.x(), r.y(), r.width(), r.height()

        scatter_x, scatter_y = scatter_item.getData()
        if scatter_x.size == 0 or scatter_y.size == 0:
            return

        mask = (scatter_x < rx) | (scatter_x > rx + rw) | (scatter_y < ry) | (scatter_y > ry + rh)
        scatter_item.setData(x=scatter_x[mask], y=scatter_y[mask])
        self.sig_peaks_edited.emit("remove", scatter_x[~mask].astype(int).tolist())
        vb.mapped_selection_rect = None
        vb.selection_box = None

    @QtCore.Slot()
    def remove_selection_rect(self) -> None:
        vb: CustomViewBox = self._pw_main.getPlotItem().getViewBox()
        vb.selection_box = None
        vb.mapped_selection_rect = None

    def get_selection_rect(self) -> QtCore.QRectF | None:
        vb: CustomViewBox = self._pw_main.plotItem.vb
        if vb.mapped_selection_rect is not None:
            return vb.mapped_selection_rect.boundingRect()
