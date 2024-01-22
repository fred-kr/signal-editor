import pprint
import typing as t
from pathlib import Path

import numpy as np
import pdir
import polars as pl
import polars.selectors as ps
import pyqtgraph as pg
from PySide6.QtCore import (
    QDate,
    QObject,
    Signal,
    Slot,
)
from PySide6.QtWidgets import (
    QComboBox,
    QDockWidget,
    QMenu,
)

from ..handlers.plot_handler import PlotHandler
from ..models.io import parse_file_name
from ..views._ui_state_maps import (
    COMBO_BOX_ITEMS,
    FILTER_INPUT_STATES,
    INITIAL_PEAK_STATES,
    INITIAL_STATE_MAP,
    INITIAL_STATE_METHODS_MAP,
)
from ..views.custom_widgets import JupyterConsoleWidget

if t.TYPE_CHECKING:
    from ..app import SignalEditor


class UIHandler(QObject):
    sig_filter_inputs_ready = Signal()
    sig_section_confirmed = Signal()
    sig_section_canceled = Signal()

    def __init__(self, window: "SignalEditor", plot: PlotHandler) -> None:
        super(UIHandler, self).__init__()
        self._window = window
        self.plot = plot
        self.setup_widgets()
        self._connect_signals()

    def setup_widgets(self) -> None:
        self._set_combo_box_items()

        # Signal Filtering
        self._prepare_inputs()

        # File Info
        self._prepare_widgets()

        # Statusbar
        self._window.statusbar.showMessage("Idle")

        # Toolbar Plots
        self._prepare_toolbars()

        # Console
        self.create_jupyter_console_widget()

    def _prepare_widgets(self) -> None:
        self._window.container_file_info.setEnabled(False)
        self._window.btn_load_selection.setEnabled(False)
        self._window.dock_widget_sections.setVisible(False)
        self._window.container_section_confirm_cancel.setVisible(False)

        export_menu = QMenu(self._window.btn_export_focused)
        export_menu.addAction("CSV", lambda: self._window.export_focused_result("csv"))
        export_menu.addAction(
            "Text (tab-delimited)", lambda: self._window.export_focused_result("txt")
        )
        export_menu.addAction("Excel", lambda: self._window.export_focused_result("xlsx"))
        self._window.btn_export_focused.setMenu(export_menu)

    def _prepare_inputs(self) -> None:
        # Signal Filtering
        self._window.combo_box_preprocess_pipeline.setValue("custom")
        self._window.container_custom_filter_inputs.setEnabled(True)
        self._set_elgendi_cleaning_params()

        # Peak Detection
        peak_combo_box = self._window.combo_box_peak_detection_method
        stacked_peak_widget = self._window.stacked_peak_parameters
        peak_combo_box.blockSignals(True)
        peak_combo_box.clear()
        peak_combo_box.setItems(COMBO_BOX_ITEMS["combo_box_peak_detection_method"])
        peak_combo_box.setCurrentIndex(0)
        stacked_peak_widget.setCurrentIndex(0)
        peak_combo_box.blockSignals(False)
        peak_combo_box.currentIndexChanged.connect(stacked_peak_widget.setCurrentIndex)

    def _connect_signals(self) -> None:
        self._window.tabs_main.currentChanged.connect(self.on_main_tab_changed)
        self._window.combo_box_filter_method.currentTextChanged.connect(
            self.handle_filter_method_changed
        )
        self._window.combo_box_preprocess_pipeline.currentTextChanged.connect(
            self.handle_preprocess_pipeline_changed
        )

        self._window.action_open_console.triggered.connect(self.show_jupyter_console_widget)
        self._window.tabs_main.currentChanged.connect(self.on_main_tab_changed)

    @Slot()
    def _emit_section_confirmed(self) -> None:
        self.sig_section_confirmed.emit()

    @Slot()
    def _emit_section_canceled(self) -> None:
        self.sig_section_canceled.emit()

    def _prepare_toolbars(self) -> None:
        plot_toolbar = self._window.toolbar_plots
        plot_toolbar.setVisible(False)

    def _set_combo_box_items(self) -> None:
        for key, value in COMBO_BOX_ITEMS.items():
            combo_box: pg.ComboBox = getattr(self._window, key)
            combo_box.clear()
            combo_box.setItems(value)

    @staticmethod
    def _blocked_set_combo_box_items(combo_box: QComboBox, items: list[str]) -> None:
        combo_box.blockSignals(True)
        combo_box.clear()
        combo_box.addItems(items)
        combo_box.setCurrentIndex(0)
        combo_box.blockSignals(False)

    def update_data_select_ui(self, path: str) -> None:
        self._window.container_file_info.setEnabled(True)
        self._window.btn_load_selection.setEnabled(True)
        data_cols = self._window.data.df.select(
            (~ps.contains(["index", "time", "temp"])) & (ps.float())
        ).columns
        data_combo_box = self._window.combo_box_signal_column
        self._blocked_set_combo_box_items(data_combo_box, data_cols)

        try:
            parsed_date, parsed_id, parsed_oxy = parse_file_name(Path(path).name)
            self._window.date_edit_file_info.setDate(
                QDate(parsed_date.year, parsed_date.month, parsed_date.day)
            )
            self._window.line_edit_subject_id.setText(parsed_id)
            self._window.combo_box_oxygen_condition.setValue(parsed_oxy)
        except Exception:
            self._window.date_edit_file_info.setDate(QDate.currentDate())
            self._window.line_edit_subject_id.setText("unknown")
            self._window.combo_box_oxygen_condition.setValue("unknown")

    @Slot()
    def reset_widget_state(self) -> None:
        mw = self._window
        mw.tabs_main.setCurrentIndex(0)
        mapping = INITIAL_STATE_METHODS_MAP
        combined_map = INITIAL_STATE_MAP | INITIAL_PEAK_STATES
        for widget_name, state in combined_map.items():
            for attribute, value in state.items():
                getattr(mw, widget_name).__getattribute__(mapping[attribute])(value)

        self.plot.reset_plots()
        mw.statusbar.showMessage("Idle")

    @Slot(int)
    def on_main_tab_changed(self, index: int) -> None:
        is_index_one = index == 1

        self._window.toolbar_plots.setVisible(is_index_one)
        self._window.toolbar_plots.setEnabled(is_index_one)
        self._window.dock_widget_sections.setVisible(is_index_one)
        self._window.dock_widget_sections.setEnabled(is_index_one)

    def create_jupyter_console_widget(self) -> None:
        self.jupyter_console = JupyterConsoleWidget()
        self.jupyter_console.set_default_style("linux")
        self.jupyter_console_dock = QDockWidget("Jupyter Console")
        self.jupyter_console_dock.setWidget(self.jupyter_console)
        self.jupyter_console.kernel_manager.kernel.shell.push(
            dict(mw=self._window, pg=pg, np=np, pl=pl, pp=pprint.pprint, pdir=pdir)
        )
        self.jupyter_console.execute("whos()")

    @Slot()
    def show_jupyter_console_widget(self) -> None:
        if self.jupyter_console_dock.isVisible():
            self.jupyter_console_dock.close()
        else:
            self.jupyter_console_dock.show()

    @Slot(str)
    def handle_filter_method_changed(self, text: str) -> None:
        method = self._window.filter_method

        for widget_name, enabled in FILTER_INPUT_STATES[method].items():
            getattr(self._window, widget_name).setEnabled(enabled)

        self.sig_filter_inputs_ready.emit()

    def _set_elgendi_cleaning_params(self) -> None:
        self._window.combo_box_filter_method.blockSignals(True)
        self._window.combo_box_filter_method.setValue("butterworth")
        self._window.combo_box_filter_method.blockSignals(False)
        self._window.dbl_spin_box_lowcut.setValue(0.5)
        self._window.dbl_spin_box_highcut.setValue(8.0)
        self._window.spin_box_order.setValue(3)
        self._window.slider_order.setValue(3)
        self.sig_filter_inputs_ready.emit()

    @Slot()
    def handle_preprocess_pipeline_changed(self) -> None:
        pipeline_value = self._window.pipeline
        if pipeline_value == "custom":
            self._window.container_custom_filter_inputs.setEnabled(True)
            selected_filter = self._window.filter_method
            self.handle_filter_method_changed(selected_filter)

        elif pipeline_value == "ppg_elgendi":
            self._window.container_custom_filter_inputs.setEnabled(False)
            self._window.combo_box_filter_method.setValue("butterworth")
            self._set_elgendi_cleaning_params()
        else:
            # TODO: add UI and logic for other signal cleaning pipelines
            self._window.container_custom_filter_inputs.setEnabled(False)
            msg = f"Selected pipeline {pipeline_value} not yet implemented, use either 'custom' or 'ppg_elgendi'."
            self._window.sig_show_message.emit(msg, "info")
            return
