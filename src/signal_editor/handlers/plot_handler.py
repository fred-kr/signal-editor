from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Literal,
    Sequence,
)

import numpy as np
import pyqtgraph as pg
from loguru import logger
from numpy.typing import NDArray
from PySide6.QtCore import QObject, QPointF, Qt, Signal, Slot

from ..models.result import ManualPeakEdits
from ..type_aliases import SignalName
from ..views.custom_widgets import CustomScatterPlotItem, CustomViewBox

if TYPE_CHECKING:
    from pyqtgraph.GraphicsScene import mouseEvents

    from ..app import SignalEditor


@dataclass(slots=True)
class PlotItems:
    signal: pg.PlotDataItem | None = None
    peaks: CustomScatterPlotItem | None = None
    rate: pg.PlotDataItem | None = None
    rate_mean: pg.InfiniteLine | None = None
    temperature_label: pg.LabelItem | None = None
    active_section: pg.LinearRegionItem | None = None

    def reset(self) -> None:
        self.signal = None
        self.peaks = None
        self.rate = None
        self.rate_mean = None
        self.temperature_label = None
        self.active_section = None


class PlotWidgetContainer(dict[str, pg.PlotWidget]):
    def make_plot_widget(
        self,
        name: str,
        background_color: str = "default",
        view_box: pg.ViewBox | CustomViewBox | None = None,
    ) -> None:
        widget = pg.PlotWidget(viewBox=view_box, background=background_color, useOpenGL=True)
        self[name] = widget

    def get_all_widgets(self) -> list[pg.PlotWidget]:
        return list(self.values())

    def get_view_box(self, name: str) -> pg.ViewBox | CustomViewBox:
        return self[name].getPlotItem().getViewBox()

    def get_all_view_boxes(self) -> list[pg.ViewBox | CustomViewBox]:
        return [pw.getPlotItem().getViewBox() for pw in self.values()]


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
    Class that manages showing and updating plots.
    """

    sig_peaks_edited = Signal(str)
    sig_excluded_range = Signal(int, int)
    sig_active_region_changed = Signal(int, int)

    def __init__(self, window: "SignalEditor", parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._window = window
        self._click_tolerance = 80
        self.plot_items: dict[str, PlotItems] = {
            "hbr": PlotItems(),
            "ventilation": PlotItems(),
        }
        self.plot_widgets = PlotWidgetContainer()
        self.peak_edits: dict[SignalName | str, ManualPeakEdits] = {
            "hbr": ManualPeakEdits(),
            "ventilation": ManualPeakEdits(),
        }
        self._last_edit_index: int = 0

        self._make_plot_widgets()

        self._prepare_plot_items()

    @property
    def last_edit_index(self) -> int:
        return self._last_edit_index

    @last_edit_index.setter
    def last_edit_index(self, value: int | float) -> None:
        self._last_edit_index = int(value)

    @Slot(int)
    def set_click_tolerance(self, value: int) -> None:
        self._click_tolerance = value

    def _make_plot_widgets(self) -> None:
        self.plot_widgets.make_plot_widget(name="hbr", view_box=CustomViewBox())
        self.plot_widgets.make_plot_widget(name="ventilation", view_box=CustomViewBox())
        self.plot_widgets.make_plot_widget(name="hbr_rate")
        self.plot_widgets.make_plot_widget(name="ventilation_rate")

    def _prepare_plot_items(self) -> None:
        # style = self._window.theme_switcher.active_style
        # color = "white" if style == "dark" else "black"
        self._set_plot_labels(color="white")

        for name, pw in self.plot_widgets.items():
            plot_item = pw.getPlotItem()
            view_box = plot_item.getViewBox()

            plot_item.showGrid(x=False, y=True)
            view_box.enableAutoRange("y")
            plot_item.setDownsampling(auto=True)
            plot_item.setClipToView(True)
            plot_item.addLegend(colCount=2, brush="transparent")
            plot_item.addLegend().anchor(itemPos=(0, 1), parentPos=(0, 1), offset=(5, -5))
            plot_item.register(name)
            view_box.setAutoVisible(y=True)
            plot_item.setMouseEnabled(x=True, y=False)

        linked_plots = [("hbr", "hbr_rate"), ("ventilation", "ventilation_rate")]
        for name, linked_name in linked_plots:
            self.plot_widgets.get_view_box(name).setXLink(linked_name)
            self.plot_widgets.get_view_box(linked_name).setMouseMode(pg.ViewBox.RectMode)

        for name in ["hbr", "ventilation"]:
            self.plot_widgets[name].getPlotItem().scene().sigMouseMoved.connect(self.on_mouse_moved)

    @staticmethod
    def set_plot_titles_and_labels(
        plot_item: pg.PlotItem, title: str, left_label: str, bottom_label: str
    ) -> None:
        plot_item.setTitle(title)
        plot_item.setLabel(axis="left", text=left_label)
        plot_item.setLabel(axis="bottom", text=bottom_label)

    def _generate_styled_label(
        self,
        color: str,
        content: str,
        font_family: str = "Segoe UI",
        font_size: int = 12,
    ) -> str:
        return f"<span style='color: {color}; font-family: {font_family}; font-size: {font_size}pt;'>{content}</span>"

    def _set_plot_labels(
        self,
        color: str = "gray",
        font_family: str = "Segoe UI",
        title_font_size: int = 12,
    ) -> None:
        plot_titles = {
            "hbr": "<b>Heart</b>",
            "ventilation": "<b>Ventilation</b>",
            "hbr_rate": "<b>Estimated Rate</b>",
            "ventilation_rate": "<b>Estimated Rate</b>",
        }
        for plot_key, title_content in plot_titles.items():
            title = self._generate_styled_label(color, title_content, font_family, title_font_size)
            left_label = self._generate_styled_label(
                color,
                "Signal Amplitude" if plot_key in ["hbr", "ventilation"] else "Cycles per minute",
            )
            bottom_label = self._generate_styled_label(color, "n samples")
            self.set_plot_titles_and_labels(
                self.plot_widgets[plot_key].getPlotItem(),
                title=title,
                left_label=left_label,
                bottom_label=bottom_label,
            )

    def show_section_selector(
        self,
        name: str,
        section_type: Literal["included", "excluded"],
        bounds: tuple[int, int] = (0, 1),
    ) -> None:
        styles = SECTION_STYLES[section_type]
        span = bounds[1] - bounds[0]

        active_region = pg.LinearRegionItem(
            values=[bounds[0] + span / 4, bounds[1] - span / 4],
            brush=styles["brush"],
            pen=styles["pen"],
            hoverBrush=styles["hoverBrush"],
            hoverPen=styles["hoverPen"],
            bounds=bounds,
        )
        for line in active_region.lines:
            line.addMarker("<|>", position=0.5, size=20)

        active_region.setZValue(1e3)
        self.plot_widgets[name].addItem(active_region)
        self.plot_items[name].active_section = active_region

    def hide_section_selector(self, name: str) -> None:
        active_section = self.plot_items[name].active_section
        if active_section is not None:
            active_section.hide()

    @Slot(str)
    def update_plot_view(self, name: str) -> None:
        for w_name, widget in self.plot_widgets.items():
            if w_name in name:
                plot_item = widget.getPlotItem()
                view_box = plot_item.getViewBox()
                view_box.autoRange()
                view_box.enableAutoRange("y")
                plot_item.setDownsampling(auto=True)
                plot_item.setClipToView(True)
                view_box.setAutoVisible(y=True)
                view_box.setMouseEnabled(x=True, y=False)

    @Slot()
    def reset_plots(self) -> None:
        for pw in self.plot_widgets.get_all_widgets():
            pw.getPlotItem().addLegend().clear()
            pw.getPlotItem().clear()
            pw.getPlotItem().getViewBox().clear()

        self.plot_items.clear()
        self.peak_edits.clear()
        self._last_edit_index = 0
        self.peak_edits = {"hbr": ManualPeakEdits(), "ventilation": ManualPeakEdits()}
        self.plot_items = {"hbr": PlotItems(), "ventilation": PlotItems()}
        self._prepare_plot_items()

    @Slot(QPointF)
    def on_mouse_moved(self, pos: QPointF) -> None:
        if not hasattr(self._window, "data"):
            return
        name = self._window.signal_name
        if self._window.data.sigs == {}:
            return
        sig_df = self._window.data.sigs[name].data
        if sig_df.is_empty():
            return
        temperature_label = self.plot_items[name].temperature_label
        if temperature_label is None:
            temperature_label = pg.LabelItem(parent=self.plot_widgets[name].getPlotItem())
        mapped_pos = self.plot_widgets[name].plotItem.vb.mapSceneToView(pos)
        index = int(mapped_pos.x())
        index = np.clip(index, 0, sig_df.height - 1)

        temperature = (
            sig_df.get_column("temperature").gather(index).to_numpy(zero_copy_only=True)[0]
        )

        text = f"<span style='color: orange; font-size: 12pt; font-weight: bold; font-family: Segoe UI;'>Temperature: {temperature:.1f} °C</span>"

        temperature_label.setText(text)

        # self._window.statusbar.showMessage(
        #     f"Cursor position (scene): (x = {pos.x()}, y = {pos.y()}); Cursor position (data): (x = {int(mapped_pos.x()):_}, y = {mapped_pos.y():.2f})"
        # )
        self.plot_items[name].temperature_label = temperature_label

    @Slot(int)
    def reset_views(self, upper_range: int) -> None:
        for vb in self.plot_widgets.get_all_view_boxes():
            vb.setXRange(0, upper_range)

    def draw_signal(
        self,
        sig: NDArray[np.float64],
        signal_name: SignalName | str,
    ) -> None:
        plot_widget = self.plot_widgets[signal_name]
        color = "crimson" if signal_name == "hbr" else "royalblue"
        signal_line = pg.PlotDataItem(
            sig,
            pen=color,
            skipFiniteCheck=True,
            autoDownSample=True,
            name=f"{signal_name}_signal",
        )
        signal_line.curve.setSegmentedLineMode("on")
        signal_line.curve.setClickable(True, width=self._click_tolerance)

        rate_name = f"{signal_name}_rate"
        rate_plot_widget = self.plot_widgets[rate_name]
        signal_ref = self.plot_items[signal_name].signal

        if signal_ref is not None:
            plot_widget.removeItem(signal_ref)
        if self.plot_items[signal_name].rate is not None:
            rate_plot_widget.removeItem(self.plot_items[signal_name].rate)
            rate_plot_widget.removeItem(self.plot_items[signal_name].rate_mean)

        plot_widget.addItem(signal_line)

        signal_line.sigClicked.connect(self.add_clicked_point)

        self.plot_items[signal_name].signal = signal_line

    def draw_peaks(
        self,
        pos_x: NDArray[np.int32],
        pos_y: NDArray[np.float64],
        signal_name: SignalName | str,
    ) -> None:
        plot_widget = self.plot_widgets[signal_name]
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

        peak_ref = self.plot_items[signal_name].peaks
        if peak_ref is not None:
            plot_widget.removeItem(peak_ref)

        plot_widget.addItem(peaks_scatter)
        peaks_scatter.sigClicked.connect(self.remove_clicked_point)
        self.plot_items[signal_name].peaks = peaks_scatter

    def draw_rate(
        self,
        rate_data: NDArray[np.float64],
        plot_widget: pg.PlotWidget,
        signal_name: SignalName | str,
    ) -> None:
        rate_mean = np.mean(rate_data, dtype=np.int32)
        pen_color = "green"
        mean_pen_color = "goldenrod"

        rate_line = pg.PlotDataItem(
            rate_data,
            pen=pen_color,
            autoDownsample=True,
            skipFiniteCheck=True,
            name=f"{signal_name}_rate",
        )
        rate_mean_line = pg.InfiniteLine(
            rate_mean,
            angle=0,
            pen=dict(color=mean_pen_color, width=2, style=Qt.PenStyle.DashLine),
            name=f"{signal_name}_rate_mean",
        )

        rate_ref = self.plot_items[signal_name].rate
        rate_mean_ref = self.plot_items[signal_name].rate_mean

        if rate_ref is not None:
            plot_widget.removeItem(rate_ref)
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

        self.plot_items[signal_name].rate = rate_line
        self.plot_items[signal_name].rate_mean = rate_mean_line

    @Slot(object, object, object)
    def remove_clicked_point(
        self,
        sender: CustomScatterPlotItem,
        points: Sequence[pg.SpotItem],
        ev: "mouseEvents.MouseClickEvent",
    ) -> None:
        ev.accept()
        if not points:
            return

        spot_item = points[0]
        to_remove_index = spot_item.index()
        name = "hbr" if "hbr" in sender.name() else "ventilation"

        if scatter_plot := self.plot_items[name].peaks:
            new_points_x = np.delete(scatter_plot.data["x"], to_remove_index)  # type: ignore
            new_points_y = np.delete(scatter_plot.data["y"], to_remove_index)  # type: ignore
            scatter_plot.setData(x=new_points_x, y=new_points_y)
            peak_edit_x = int(spot_item.pos().x())
            self.peak_edits[name].removed.append(peak_edit_x)
            self.last_edit_index = peak_edit_x
            self.plot_items[name].peaks = scatter_plot
            self.sig_peaks_edited.emit(name)

    @Slot()
    def remove_selected(self) -> None:
        """
        Removes peaks inside a rectangular selection.
        """
        logger.debug(f"Remove selected sender name: {self.sender().objectName()}")
        name = self._window.signal_name
        vb = self.plot_widgets.get_view_box(name)
        if vb.mapped_peak_selection is None:
            return
        scatter_ref = self.plot_items[name].peaks
        if scatter_ref is None:
            return
        rect_x, rect_y, rect_width, rect_height = vb.mapped_peak_selection.boundingRect().getRect()

        scatter_x, scatter_y = scatter_ref.getData()
        if scatter_x is None or scatter_y is None or scatter_x.size == 0 or scatter_y.size == 0:
            return

        to_remove = np.argwhere(
            (scatter_x >= rect_x)
            & (scatter_x <= rect_x + rect_width)
            & (scatter_y >= rect_y)
            & (scatter_y <= rect_y + rect_height)
        ).flatten()

        if to_remove.size == 0:
            return
        scatter_ref.setData(x=np.delete(scatter_x, to_remove), y=np.delete(scatter_y, to_remove))

        self.peak_edits[name].removed.extend(scatter_x[to_remove].astype(int))
        self.last_edit_index = scatter_x[to_remove].max()
        self.sig_peaks_edited.emit(name)
        self.plot_widgets[name].getPlotItem().getViewBox().selection_box = None

    @Slot(object, object)
    def add_clicked_point(
        self, sender: pg.PlotCurveItem, ev: "mouseEvents.MouseClickEvent"
    ) -> None:
        ev.accept()
        click_x = int(ev.pos().x())
        name = "hbr" if "hbr" in f"{sender.name()}" else "ventilation"
        signal_line = self.plot_items[name].signal
        scatter_ref = self.plot_items[name].peaks

        if signal_line is None or scatter_ref is None:
            return

        xData: NDArray[np.float64] = signal_line.xData
        yData: NDArray[np.float64] = signal_line.yData

        # Define the radius within which to search for the max y-value
        search_radius = 20

        # Find the indices within the radius around the click position
        indices = np.where((xData >= click_x - search_radius) & (xData <= click_x + search_radius))[
            0
        ]

        # Select the subset of y-values within the radius
        y_values_in_radius: NDArray[np.float64] = yData[indices]

        # Find the index of the max y-value and the index of the min y-value
        max_y_index = indices[np.argmax(y_values_in_radius)]
        min_y_index = indices[np.argmin(y_values_in_radius)]

        # Check which index is closer to the click position
        if abs(xData[max_y_index] - click_x) > abs(xData[min_y_index] - click_x):
            use_index = min_y_index
            use_val = y_values_in_radius.min()
        else:
            use_index = max_y_index
            use_val = y_values_in_radius.max()
        if use_index in scatter_ref.data["x"]:
            return

        # Get the new x and y values
        x_new, y_new = xData[use_index], use_val
        scatter_ref.addPoints(x=x_new, y=y_new)
        self.peak_edits[name].added.append(int(x_new))
        self.last_edit_index = x_new
        self.sig_peaks_edited.emit(name)

    @Slot()
    def show_region_selector(self) -> None:
        name = self._window.signal_name
        view_box: CustomViewBox = self.plot_widgets.get_view_box(name)
        selection_box = view_box.selection_box
        selection_box.setVisible(not selection_box.isVisible())
        selection_box.setEnabled(not selection_box.isEnabled())

    def _mark_included_region(self, lower: int, upper: int) -> None:
        self._mark_region(lower, upper, 0, 255, 0)

    def _mark_excluded_region(self, lower: int, upper: int) -> None:
        self._mark_region(lower, upper, 255, 0, 0)

    def _mark_region(self, lower: int, upper: int, r: int, g: int, b: int) -> None:
        name = self._window.signal_name
        static_region = pg.LinearRegionItem(
            values=(lower, upper),
            brush=(r, g, b, 50),
            pen=dict(color=(r, g, b, 255), width=2),
            movable=False,
        )
        self.plot_widgets[name].getPlotItem().getViewBox().addItem(static_region)

    @Slot()
    def emit_to_be_excluded_range(self) -> None:
        name = self._window.signal_name
        active_section = self.plot_items[name].active_section
        if not active_section.isVisible():
            return
        lower, upper = active_section.getRegion()
        lower, upper = int(lower), int(upper)
        self._mark_excluded_region(lower, upper)
        self.sig_excluded_range.emit(lower, upper)

    def restore_state(self, state: dict[SignalName | str, ManualPeakEdits]) -> None:
        self.peak_edits = state
