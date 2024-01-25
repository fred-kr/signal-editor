import pprint
import typing as t

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

from .. import type_aliases as _t
from ..handlers.plot_handler import PlotHandler
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
    sig_section_confirmed = Signal(str)
    sig_section_canceled = Signal()

    def __init__(self, app: "SignalEditor", plot: PlotHandler) -> None:
        super(UIHandler, self).__init__()
        self._app = app
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
        self._app.statusbar.showMessage("Ready")

        # Toolbar Plots
        self._prepare_toolbars()

        # Console
        self.create_jupyter_console_widget()

    def _prepare_widgets(self) -> None:
        self._app.container_file_info.setEnabled(False)
        self._app.btn_load_selection.setEnabled(False)
        self._app.dock_widget_sections.setVisible(False)
        self._app.container_section_confirm_cancel.setEnabled(True)
        self._app.container_section_confirm_cancel.hide()

        export_menu = QMenu(self._app.btn_export_focused)
        export_menu.addAction("CSV", lambda: self._app.export_focused_result("csv"))
        export_menu.addAction(
            "txt (tab-delimited)", lambda: self._app.export_focused_result("txt")
        )
        export_menu.addAction("Excel", lambda: self._app.export_focused_result("xlsx"))
        self._app.btn_export_focused.setMenu(export_menu)

    def _prepare_inputs(self) -> None:
        # Signal Filtering
        self._app.combo_box_preprocess_pipeline.setValue("custom")
        self._app.container_custom_filter_inputs.setEnabled(True)
        self._set_elgendi_cleaning_params()

        # Peak Detection
        peak_combo_box = self._app.combo_box_peak_detection_method
        stacked_peak_widget = self._app.stacked_peak_parameters
        peak_combo_box.blockSignals(True)
        peak_combo_box.clear()
        peak_combo_box.setItems(COMBO_BOX_ITEMS["combo_box_peak_detection_method"])
        peak_combo_box.setCurrentIndex(0)
        stacked_peak_widget.setCurrentIndex(0)
        peak_combo_box.blockSignals(False)
        peak_combo_box.currentIndexChanged.connect(stacked_peak_widget.setCurrentIndex)

    def _connect_signals(self) -> None:
        self._app.tabs_main.currentChanged.connect(self.on_main_tab_changed)
        self._app.combo_box_filter_method.currentTextChanged.connect(
            self.handle_filter_method_changed
        )
        self._app.combo_box_preprocess_pipeline.currentTextChanged.connect(
            self.handle_preprocess_pipeline_changed
        )

        self._app.action_open_console.triggered.connect(self.show_jupyter_console_widget)
        self._app.tabs_main.currentChanged.connect(self.on_main_tab_changed)

    @Slot()
    def _emit_section_confirmed(self) -> None:
        self.sig_section_confirmed.emit()

    @Slot()
    def _emit_section_canceled(self) -> None:
        self.sig_section_canceled.emit()

    def _prepare_toolbars(self) -> None:
        plot_toolbar = self._app.toolbar_plots
        plot_toolbar.setVisible(False)

    def _set_combo_box_items(self) -> None:
        for key, value in COMBO_BOX_ITEMS.items():
            combo_box: pg.ComboBox = getattr(self._app, key)
            combo_box.clear()
            combo_box.setItems(value)

    @staticmethod
    def _blocked_set_combo_box_items(combo_box: QComboBox, items: list[str]) -> None:
        combo_box.blockSignals(True)
        combo_box.clear()
        combo_box.addItems(items)
        combo_box.setCurrentIndex(0)
        combo_box.blockSignals(False)

    @Slot()
    def update_data_select_ui(self) -> None:
        self._app.container_file_info.setEnabled(True)
        self._app.btn_load_selection.setEnabled(True)
        data_cols = self._app.data.raw_df.select(
            (~ps.contains(["index", "time", "temp"])) & (ps.float())
        ).columns
        data_combo_box = self._app.combo_box_signal_column
        self._blocked_set_combo_box_items(data_combo_box, data_cols)

        try:
            metadata = self._app.data.metadata
            meas_date = metadata["date_recorded"]

            self._app.date_edit_file_info.setDate(
                QDate(meas_date.year, meas_date.month, meas_date.day)
            )
            self._app.line_edit_subject_id.setText(metadata["animal_id"])
            self._app.combo_box_oxygen_condition.setValue(metadata["oxygen_condition"])
        except Exception:
            self._app.date_edit_file_info.setDate(QDate.currentDate())
            self._app.line_edit_subject_id.setText("unknown")
            self._app.combo_box_oxygen_condition.setValue("unknown")

    @Slot()
    def reset_widget_state(self) -> None:
        mw = self._app
        mw.tabs_main.setCurrentIndex(0)
        mapping = INITIAL_STATE_METHODS_MAP
        combined_map = INITIAL_STATE_MAP | INITIAL_PEAK_STATES
        for widget_name, state in combined_map.items():
            for attribute, value in state.items():
                getattr(mw, widget_name).__getattribute__(mapping[attribute])(value)

        self.plot.reset_plots()
        mw.statusbar.showMessage("Ready")

    @Slot(int)
    def on_main_tab_changed(self, index: int) -> None:
        is_index_one = index == 1

        self._app.toolbar_plots.setVisible(is_index_one)
        self._app.toolbar_plots.setEnabled(is_index_one)
        self._app.dock_widget_sections.setVisible(is_index_one)
        self._app.dock_widget_sections.setEnabled(is_index_one)

    def create_jupyter_console_widget(self) -> None:
        self.jupyter_console = JupyterConsoleWidget()
        self.jupyter_console.set_default_style("linux")
        self.jupyter_console_dock = QDockWidget("Jupyter Console")
        self.jupyter_console_dock.setWidget(self.jupyter_console)
        self.jupyter_console.kernel_manager.kernel.shell.push(
            dict(mw=self._app, pg=pg, np=np, pl=pl, pp=pprint.pprint, pdir=pdir)
        )
        self.jupyter_console.execute("whos")

    @Slot()
    def show_jupyter_console_widget(self) -> None:
        if self.jupyter_console_dock.isVisible():
            self.jupyter_console_dock.close()
        else:
            self.jupyter_console_dock.show()
            self.jupyter_console_dock.resize(900, 600)

    @Slot(str)
    def handle_filter_method_changed(self, text: str) -> None:
        method = self._app.filter_method

        for widget_name, enabled in FILTER_INPUT_STATES[method].items():
            getattr(self._app, widget_name).setEnabled(enabled)

        self.sig_filter_inputs_ready.emit()

    def _set_elgendi_cleaning_params(self) -> None:
        self._app.combo_box_filter_method.blockSignals(True)
        self._app.combo_box_filter_method.setValue("butterworth")
        self._app.combo_box_filter_method.blockSignals(False)
        self._app.dbl_spin_box_lowcut.setValue(0.5)
        self._app.dbl_spin_box_highcut.setValue(8.0)
        self._app.spin_box_order.setValue(3)
        self._app.slider_order.setValue(3)
        self.sig_filter_inputs_ready.emit()

    @Slot()
    def handle_preprocess_pipeline_changed(self) -> None:
        pipeline_value = self._app.pipeline
        if pipeline_value == "custom":
            self._app.container_custom_filter_inputs.setEnabled(True)
            selected_filter = self._app.filter_method
            self.handle_filter_method_changed(selected_filter)

        elif pipeline_value == "ppg_elgendi":
            self._app.container_custom_filter_inputs.setEnabled(False)
            self._app.combo_box_filter_method.setValue("butterworth")
            self._set_elgendi_cleaning_params()
        else:
            # TODO: add UI and logic for other signal cleaning pipelines
            self._app.container_custom_filter_inputs.setEnabled(False)
            msg = f"Selected pipeline {pipeline_value} not yet implemented, use either 'custom' or 'ppg_elgendi'."
            self._app.sig_show_message.emit(msg, "info")
            return

    def get_filter_parameters(self) -> _t.SignalFilterParameters:
        method = self._app.filter_method

        filter_params = _t.SignalFilterParameters(
            lowcut=None,
            highcut=None,
            method=method,
            order=2,
            window_size="default",
            powerline=50,
        )
        if method != "None":
            filter_widgets = {
                "lowcut": self._app.dbl_spin_box_lowcut,
                "highcut": self._app.dbl_spin_box_highcut,
                "order": self._app.spin_box_order,
                "window_size": self._app.spin_box_window_size,
                "powerline": self._app.dbl_spin_box_powerline,
            }
            for param, widget in filter_widgets.items():
                if widget.isEnabled():
                    filter_params[param] = widget.value()

        return filter_params

    def get_standardize_parameters(self) -> _t.StandardizeParameters:
        method = self._app.scale_method
        robust = method == "mad"
        if self._app.container_scale_window_inputs.isChecked():
            window_size = self._app.spin_box_scale_window_size.value()
        else:
            window_size = None

        return _t.StandardizeParameters(robust=robust, window_size=window_size)

    def get_peak_detection_parameters(self) -> _t.PeakDetectionParameters:
        method = self._app.peak_detection_method

        match method:
            case "elgendi_ppg":
                vals = _t.PeakDetectionElgendiPPG(
                    peakwindow=self._app.peak_elgendi_ppg_peakwindow.value(),
                    beatwindow=self._app.peak_elgendi_ppg_beatwindow.value(),
                    beatoffset=self._app.peak_elgendi_ppg_beatoffset.value(),
                    mindelay=self._app.peak_elgendi_ppg_min_delay.value(),
                )
            case "local":
                vals = _t.PeakDetectionLocalMaxima(radius=self._app.peak_local_max_radius.value())
            case "neurokit2":
                vals = _t.PeakDetectionNeurokit2(
                    smoothwindow=self._app.peak_neurokit2_smoothwindow.value(),
                    avgwindow=self._app.peak_neurokit2_avgwindow.value(),
                    gradthreshweight=self._app.peak_neurokit2_gradthreshweight.value(),
                    minlenweight=self._app.peak_neurokit2_minlenweight.value(),
                    mindelay=self._app.peak_neurokit2_mindelay.value(),
                    correct_artifacts=self._app.peak_neurokit2_correct_artifacts.isChecked(),
                )
            case "promac":
                vals = _t.PeakDetectionProMAC(
                    threshold=self._app.peak_promac_threshold.value(),
                    gaussian_sd=self._app.peak_promac_gaussian_sd.value(),
                    correct_artifacts=self._app.peak_promac_correct_artifacts.isChecked(),
                )
            case "pantompkins":
                vals = _t.PeakDetectionPantompkins(
                    correct_artifacts=self._app.peak_pantompkins_correct_artifacts.isChecked(),
                )
            case "wfdb_xqrs":
                vals = _t.PeakDetectionXQRS(
                    search_radius=self._app.peak_xqrs_search_radius.value(),
                    peak_dir=self._app.xqrs_peak_direction,
                )

        for key, val in vals.items():
            if isinstance(val, float):
                vals[key] = round(val, 3)

        return _t.PeakDetectionParameters(method=method, input_values=vals)

    def set_peak_detection_parameters(self, params: _t.PeakDetectionParameters) -> None:
        # sourcery skip: extract-method
        method = params["method"]
        self._app.combo_box_peak_detection_method.setValue(method)

        vals = params["input_values"]

        match method:
            case "elgendi_ppg":
                vals = t.cast(_t.PeakDetectionElgendiPPG, vals)
                self._app.peak_elgendi_ppg_peakwindow.setValue(vals["peakwindow"])
                self._app.peak_elgendi_ppg_beatwindow.setValue(vals["beatwindow"])
                self._app.peak_elgendi_ppg_beatoffset.setValue(vals["beatoffset"])
                self._app.peak_elgendi_ppg_min_delay.setValue(vals["mindelay"])
            case "local":
                vals = t.cast(_t.PeakDetectionLocalMaxima, vals)
                self._app.peak_local_max_radius.setValue(vals["radius"])
            case "neurokit2":
                vals = t.cast(_t.PeakDetectionNeurokit2, vals)
                self._app.peak_neurokit2_smoothwindow.setValue(vals["smoothwindow"])
                self._app.peak_neurokit2_avgwindow.setValue(vals["avgwindow"])
                self._app.peak_neurokit2_gradthreshweight.setValue(vals["gradthreshweight"])
                self._app.peak_neurokit2_minlenweight.setValue(vals["minlenweight"])
                self._app.peak_neurokit2_mindelay.setValue(vals["mindelay"])
                self._app.peak_neurokit2_correct_artifacts.setChecked(vals["correct_artifacts"])
            case "promac":
                vals = t.cast(_t.PeakDetectionProMAC, vals)
                self._app.peak_promac_threshold.setValue(vals["threshold"])
                self._app.peak_promac_gaussian_sd.setValue(vals["gaussian_sd"])
                self._app.peak_promac_correct_artifacts.setChecked(vals["correct_artifacts"])
            case "pantompkins":
                vals = t.cast(_t.PeakDetectionPantompkins, vals)
                self._app.peak_pantompkins_correct_artifacts.setChecked(vals["correct_artifacts"])
            case "wfdb_xqrs":
                vals = t.cast(_t.PeakDetectionXQRS, vals)
                self._app.peak_xqrs_search_radius.setValue(vals["search_radius"])
                self._app.peak_xqrs_peak_dir.setValue(vals["peak_dir"])
