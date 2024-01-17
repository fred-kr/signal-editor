import pprint
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
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
    QDockWidget,
    QVBoxLayout,
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
from ..views.custom_widgets import ConfirmCancelButtons, JupyterConsoleWidget
from .helpers.table_view_helper import TableViewHelper

if TYPE_CHECKING:
    from ..app import MainWindow


class UIHandler(QObject):
    sig_filter_inputs_ready = Signal()
    sig_section_confirmed = Signal(bool)

    def __init__(self, window: "MainWindow", plot: PlotHandler) -> None:
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

        # Plots
        self._setup_active_plot_btn_grp()
        self.create_plot_widgets()

        # Console
        # self.create_console_widget()
        self.create_jupyter_console_widget()

    def _prepare_widgets(self) -> None:
        self._window.container_file_info.setEnabled(False)
        self._window.btn_load_selection.setEnabled(False)
        self._window.dock_widget_sections.setVisible(False)
        self.confirm_cancel_buttons = ConfirmCancelButtons()
        self._window.section_dock_contents.layout().addWidget(self.confirm_cancel_buttons)

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

    def _setup_active_plot_btn_grp(self) -> None:
        self._window.stacked_hbr_vent.setCurrentIndex(0)
        self._window.btn_group_plot_view.setId(self._window.btn_view_hbr, 0)
        self._window.btn_group_plot_view.setId(self._window.btn_view_vent, 1)

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

        self.confirm_cancel_buttons.confirm_button.clicked.connect(
            lambda: self.sig_section_confirmed.emit(True)
        )
        self.confirm_cancel_buttons.cancel_button.clicked.connect(
            lambda: self.sig_section_confirmed.emit(False)
        )

    def _prepare_toolbars(self) -> None:
        plot_toolbar = self._window.toolbar_plots
        plot_toolbar.setVisible(False)

    def _set_combo_box_items(self) -> None:
        for key, value in COMBO_BOX_ITEMS.items():
            combo_box: pg.ComboBox = getattr(self._window, key)
            combo_box.clear()
            combo_box.setItems(value)

    def update_data_selection_widgets(self, path: str) -> None:
        self._window.container_file_info.setEnabled(True)
        self._window.group_box_subset_params.setEnabled(True)
        self._window.btn_load_selection.setEnabled(True)
        available_filter_cols = self._window.data.df.select(
            ps.contains(["index", "time", "temp"])
        ).columns
        column_box = self._window.combo_box_filter_column
        column_box.blockSignals(True)
        column_box.clear()
        column_box.addItems(available_filter_cols)
        column_box.setCurrentIndex(0)
        column_box.currentTextChanged.connect(self.update_subset_param_widgets)
        if len(available_filter_cols) > 0:
            self.update_subset_param_widgets(available_filter_cols[0])
        column_box.blockSignals(False)
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

    @Slot(str)
    def update_subset_param_widgets(self, col_name: str) -> None:
        if not col_name or col_name not in self._window.data.df.columns:
            return
        lower = cast(float, self._window.data.minmax_map[col_name]["min"])
        upper = cast(float, self._window.data.minmax_map[col_name]["max"])
        fs = self._window.data.fs

        widgets = [
            self._window.dbl_spin_box_subset_min,
            self._window.dbl_spin_box_subset_max,
        ]
        for w in widgets:
            if col_name == "index":
                w.setDecimals(0)
                w.setSingleStep(1)
            elif col_name == "temperature":
                w.setDecimals(1)
                w.setSingleStep(0.1)
            elif col_name == "time_s":
                w.setDecimals(4)
                w.setSingleStep(np.round(1 / fs, 4))
            w.setMinimum(lower)
            w.setMaximum(upper)

        widgets[0].setValue(lower)
        widgets[1].setValue(upper)

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

    def create_plot_widgets(self) -> None:
        self._window.plot_widget_hbr.setLayout(QVBoxLayout())
        self._window.plot_widget_vent.setLayout(QVBoxLayout())
        hbr_pw = self.plot.plot_widgets["hbr"]
        vent_pw = self.plot.plot_widgets["ventilation"]
        hbr_rate_pw = self.plot.plot_widgets["hbr_rate"]
        vent_rate_pw = self.plot.plot_widgets["ventilation_rate"]

        self._window.plot_widget_hbr.layout().addWidget(hbr_pw)
        self._window.plot_widget_hbr.layout().addWidget(hbr_rate_pw)

        self._window.plot_widget_vent.layout().addWidget(vent_pw)
        self._window.plot_widget_vent.layout().addWidget(vent_rate_pw)

    def create_jupyter_console_widget(self) -> None:
        self.jupyter_console = JupyterConsoleWidget()
        self.jupyter_console_dock = QDockWidget("Jupyter Console")
        self.jupyter_console_dock.setWidget(self.jupyter_console)
        self.jupyter_console.kernel_manager.kernel.shell.push(
            dict(mw=self._window, pg=pg, np=np, pl=pl, pp=pprint.pprint)
        )

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
