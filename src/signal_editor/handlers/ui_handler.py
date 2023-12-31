import types
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import polars as pl
import pyqtgraph as pg
import rich as r
from pyqtgraph.console import ConsoleWidget
from PySide6.QtCore import (
    QDate,
    QObject,
    QPointF,
    Qt,
    Signal,
    Slot,
)
from PySide6.QtGui import QStandardItemModel
from PySide6.QtWidgets import (
    QDockWidget,
    QVBoxLayout,
)

from ..handlers.plot_handler import PlotHandler
from ..models.io import parse_file_name
from ..type_aliases import (
    SignalName,
)
from ..views._peak_parameter_states import INITIAL_PEAK_STATES

if TYPE_CHECKING:
    from ..app import MainWindow


COMBO_BOX_ITEMS = {
    "combo_box_peak_detection_method": {
        "Elgendi (PPG, fast)": "elgendi_ppg",
        "Local Maxima (Any, fast)": "local",
        "Neurokit (ECG, fast)": "neurokit2",
        "ProMAC (ECG, slow)": "promac",
        "Pan and Tompkins (ECG, medium)": "pantompkins",
        "XQRS (ECG, medium)": "wfdb_xqrs",
    },
    # "peak_neurokit2_algorithm_used": {
    #     "Neurokit2 (Default)": "neurokit",
    #     "Nabian (ECG)": "nabian",
    #     "Gamboa (ECG)": "gamboa",
    #     "Slope Sum Function (ECG)": "slopesumfunction",
    #     "Zong (ECG)": "zong",
    #     "Hamilton (ECG)": "hamilton",
    #     "Christov (ECG)": "christov",
    #     "Engzeemod (ECG)": "engzeemod",
    #     "Elgendi (ECG)": "elgendi",
    #     "Kalidas (ECG)": "kalidas",
    #     "Martinez (ECG)": "martinez",
    #     "Rodrigues (ECG)": "rodrigues",
    #     "VGraph (ECG)": "vgraph",
    # },
    "combo_box_filter_method": {
        "Butterworth (SOS)": "butterworth",
        "Butterworth (BA)": "butterworth_ba",
        "Savitzky-Golay": "savgol",
        "FIR": "fir",
        "Bessel": "bessel",
        "No Filter": "None",
    },
    "combo_box_oxygen_condition": {
        "Normoxic": "normoxic",
        "Hypoxic": "hypoxic",
        "Unknown": "unknown",
    },
    "combo_box_scale_method": {
        "No Standardization": "None",
        "Z-Score": "zscore",
        "Median Absolute Deviation": "mad",
    },
    "combo_box_preprocess_pipeline": {
        "Custom": "custom",
        "Elgendi (PPG)": "ppg_elgendi",
        "Neurokit (ECG)": "ecg_neurokit2",
    },
}


INITIAL_STATE_MAP = {
    "table_data_preview": {
        "model": QStandardItemModel(),
    },
    "table_data_info": {
        "model": QStandardItemModel(),
    },
    "line_edit_active_file": {"text": ""},
    "group_box_subset_params": {
        "enabled": False,
        "checked": False,
    },
    "container_file_info": {
        "enabled": True,
    },
    "date_edit_file_info": {
        "date": QDate.currentDate(),
    },
    "line_edit_subject_id": {
        "text": "",
    },
    "combo_box_oxygen_condition": {
        "value": "normoxic",
    },
    "btn_load_selection": {
        "enabled": False,
    },
    "stacked_hbr_vent": {
        "currentIndex": 0,
    },
    "btn_view_hbr": {
        "checked": True,
    },
    "btn_view_vent": {
        "checked": False,
    },
    "combo_box_preprocess_pipeline": {
        "value": "custom",
    },
    "combo_box_filter_method": {
        "enabled": True,
        "value": "None",
    },
    "combo_box_scale_method": {
        "value": "None",
    },
    "container_standardize": {
        "enabled": True,
    },
    "container_scale_window_inputs": {
        "enabled": True,
        "checked": True,
    },
    "dbl_spin_box_lowcut": {
        "value": 0.5,
    },
    "dbl_spin_box_highcut": {
        "value": 8.0,
    },
    "spin_box_order": {
        "value": 3,
    },
    "slider_order": {
        "value": 3,
    },
    "spin_box_window_size": {
        "value": 250,
    },
    "slider_window_size": {
        "value": 250,
    },
    "combo_box_peak_detection_method": {
        "value": "elgendi_ppg",
    },
    "btn_detect_peaks": {
        "enabled": False,
    },
    "btn_compute_results": {"enabled": False},
    "table_view_results_hbr": {
        "model": QStandardItemModel(),
    },
    "table_view_results_ventilation": {
        "model": QStandardItemModel(),
    },
    "tabs_result": {
        "currentIndex": 0,
    },
} | INITIAL_PEAK_STATES

INITIAL_STATE_METHODS_MAP = {
    "enabled": "setEnabled",
    "checked": "setChecked",
    "text": "setText",
    "model": "setModel",
    "value": "setValue",
    "currentText": "setCurrentText",
    "currentIndex": "setCurrentIndex",
    "date": "setDate",
    "decimals": "setDecimals",
    "minimum": "setMinimum",
    "maximum": "setMaximum",
    "singleStep": "setSingleStep",
    "stepType": "setStepType",
    "accelerated": "setAccelerated",
    "correctionMode": "setCorrectionMode",
    "isChecked": "setChecked",
    "items": "setItems",
    "specialValueText": "setSpecialValueText",
}

FILTER_INPUT_STATES = {
    "butterworth": {
        "container_lowcut": True,
        "container_highcut": True,
        "container_order_inputs": True,
        "container_window_size": False,
    },
    "butterworth_ba": {
        "container_lowcut": True,
        "container_highcut": True,
        "container_order_inputs": True,
        "container_window_size": False,
    },
    "bessel": {
        "container_lowcut": True,
        "container_highcut": True,
        "container_order_inputs": True,
        "container_window_size": False,
    },
    "fir": {
        "container_lowcut": True,
        "container_highcut": True,
        "container_order_inputs": False,
        "container_window_size": True,
    },
    "savgol": {
        "container_lowcut": False,
        "container_highcut": False,
        "container_order_inputs": True,
        "container_window_size": True,
    },
    "None": {
        "container_lowcut": False,
        "container_highcut": False,
        "container_order_inputs": False,
        "container_window_size": False,
    },
}


class UIHandler(QObject):
    sig_filter_inputs_ready = Signal()
    sig_ready_for_cleaning = Signal()

    def __init__(self, window: "MainWindow", plot: PlotHandler) -> None:
        super(UIHandler, self).__init__()
        self.window = window
        self.plot = plot
        self.setup_widgets()
        self.connect_signals()

    def setup_widgets(self) -> None:
        self._set_combo_box_items()

        # Signal Filtering
        self._prepare_inputs()

        # Peak Detection
        # self.create_peak_detection_trees()

        # File Info
        self._prepare_widgets()

        # Statusbar
        self.window.statusbar.showMessage("Idle")

        # Toolbar Plots
        self._prepare_toolbars()

        # Plots
        self._setup_active_plot_btn_grp()
        self.create_plot_widgets()

        # Console
        self.create_console_widget()

    def _prepare_widgets(self) -> None:
        self.window.container_file_info.setEnabled(False)

    def _prepare_inputs(self) -> None:
        # Signal Filtering
        self.window.combo_box_preprocess_pipeline.setValue("custom")
        self.window.container_custom_filter_inputs.setEnabled(True)
        self._set_elgendi_cleaning_params()

        # Peak Detection
        peak_combo_box = self.window.combo_box_peak_detection_method
        stacked_peak_widget = self.window.stacked_peak_parameters
        # if peak_combo_box.value() not in COMBO_BOX_ITEMS["combo_box_peak_detection_method"]:
        # peak_combo_box.clear()
        # peak_combo_box.currentIndexChanged.disconnect()
        peak_combo_box.blockSignals(True)
        peak_combo_box.clear()
        peak_combo_box.setItems(COMBO_BOX_ITEMS["combo_box_peak_detection_method"])
        peak_combo_box.setCurrentIndex(0)
        stacked_peak_widget.setCurrentIndex(0)
        peak_combo_box.blockSignals(False)
        peak_combo_box.currentIndexChanged.connect(stacked_peak_widget.setCurrentIndex)

    def _setup_active_plot_btn_grp(self) -> None:
        self.window.stacked_hbr_vent.setCurrentIndex(0)
        self.window.btn_group_plot_view.setId(self.window.btn_view_hbr, 0)
        self.window.btn_group_plot_view.setId(self.window.btn_view_vent, 1)

    def connect_signals(self) -> None:
        self.window.tabs_main.currentChanged.connect(self.on_main_tab_changed)
        self.window.combo_box_filter_method.currentTextChanged.connect(
            self.handle_filter_method_changed
        )
        self.window.combo_box_preprocess_pipeline.currentTextChanged.connect(
            self.handle_preprocess_pipeline_changed
        )

        self.window.btn_apply_filter.clicked.connect(
            lambda: self.window.btn_detect_peaks.setEnabled(True)  # type: ignore
        )
        self.window.btn_detect_peaks.clicked.connect(
            lambda: self.window.btn_compute_results.setEnabled(True)  # type: ignore
        )

        self.window.action_open_console.triggered.connect(self.show_console_widget)
        self.window.tabs_main.currentChanged.connect(self.on_main_tab_changed)

    def _prepare_toolbars(self) -> None:
        self.window.toolbar_plots.setVisible(False)

    def _set_combo_box_items(self) -> None:
        for key, value in COMBO_BOX_ITEMS.items():
            combo_box: pg.ComboBox = getattr(self.window, key)
            combo_box.clear()
            combo_box.setItems(value)

    def update_data_selection_widgets(self, path: str) -> None:
        self.window.container_file_info.setEnabled(True)
        self.window.group_box_subset_params.setEnabled(True)
        self.window.btn_load_selection.setEnabled(True)
        viable_filter_columns = ("index", "time_s", "temperature")
        column_box = self.window.combo_box_filter_column
        column_box.blockSignals(True)
        column_box.clear()
        column_box.addItems(viable_filter_columns)
        column_box.setCurrentIndex(0)
        column_box.currentTextChanged.connect(self.update_subset_param_widgets)
        self.update_subset_param_widgets(viable_filter_columns[0])
        column_box.blockSignals(False)
        try:
            parsed_date, parsed_id, parsed_oxy = parse_file_name(Path(path).name)
            self.window.date_edit_file_info.setDate(
                QDate(parsed_date.year, parsed_date.month, parsed_date.day)
            )
            self.window.line_edit_subject_id.setText(parsed_id)
            self.window.combo_box_oxygen_condition.setValue(parsed_oxy)
        except Exception:
            self.window.date_edit_file_info.setDate(QDate(1970, 1, 1))
            self.window.line_edit_subject_id.setText("unknown")
            self.window.combo_box_oxygen_condition.setValue("unknown")

    @Slot(str)
    def update_subset_param_widgets(self, col_name: str) -> None:
        lower = cast(float, self.window.data.minmax_map[col_name]["min"])
        upper = cast(float, self.window.data.minmax_map[col_name]["max"])
        fs = self.window.data.fs

        widgets = [
            self.window.dbl_spin_box_subset_min,
            self.window.dbl_spin_box_subset_max,
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
        mw = self.window
        mw.tabs_main.setCurrentIndex(0)
        mapping = INITIAL_STATE_METHODS_MAP
        for widget_name, state in INITIAL_STATE_MAP.items():
            for attribute, value in state.items():
                getattr(mw, widget_name).__getattribute__(mapping[attribute])(value)

        self.plot.reset_plots()
        self.temperature_label_hbr.setText("Temperature: -")
        self.temperature_label_ventilation.setText("Temperature: -")
        mw.statusbar.showMessage("Ready")

    @Slot(int)
    def on_main_tab_changed(self, index: int) -> None:
        is_index_one = index == 1

        self.window.toolbar_plots.setVisible(is_index_one)
        self.window.toolbar_plots.setEnabled(is_index_one)

    def create_plot_widgets(self) -> None:
        self.window.plot_widget_hbr.setLayout(QVBoxLayout())
        self.window.plot_widget_vent.setLayout(QVBoxLayout())
        hbr_pw = self.plot.plot_widgets["hbr"]
        vent_pw = self.plot.plot_widgets["ventilation"]
        hbr_rate_pw = self.plot.plot_widgets["hbr_rate"]
        vent_rate_pw = self.plot.plot_widgets["ventilation_rate"]

        self.window.plot_widget_hbr.layout().addWidget(hbr_pw)
        self.window.plot_widget_vent.layout().addWidget(vent_pw)

        self.window.plot_widget_hbr.layout().addWidget(hbr_rate_pw)
        self.window.plot_widget_vent.layout().addWidget(vent_rate_pw)

        self.temperature_label_hbr = pg.LabelItem(
            text="<span style='color: orange; font-size: 12pt; font-weight: bold; font-family: Segoe UI;'>Temperature: -</span>",
            parent=hbr_pw.getPlotItem(),
            angle=0,
        )
        self.temperature_label_ventilation = pg.LabelItem(
            text="<span style='color: orange; font-size: 12pt; font-weight: bold; font-family: Segoe UI;'>Temperature: -</span>",
            parent=vent_pw.getPlotItem(),
            angle=0,
        )
        hbr_pw.getPlotItem().scene().sigMouseMoved.connect(
            lambda pos: self.update_temperature_label("hbr", pos)  # type: ignore
        )
        vent_pw.getPlotItem().scene().sigMouseMoved.connect(
            lambda pos: self.update_temperature_label("ventilation", pos)  # type: ignore
        )

    @Slot(QPointF)
    def update_temperature_label(self, signal_name: SignalName, pos: QPointF) -> None:
        if not hasattr(self.window, "data"):
            return
        if self.window.data.df.is_empty():
            return
        # view_box = self.plot.plot_widgets.get_view_box(signal_name)
        data_pos = (
            self.plot.plot_widgets[signal_name].plotItem.vb.mapSceneToView(pos).x()
        )
        data_pos = np.clip(
            data_pos,
            0,
            self.window.data.sigs[signal_name].data.height - 1,
            dtype=np.int32,
            casting="unsafe",
        )
        temp_value = (
            self.window.data.sigs[signal_name]
            .data.get_column("temperature")
            .gather(data_pos)
            .to_numpy(zero_copy_only=True)[0]
        )
        # TODO: give temperature labels same treatment as all the other things with two versions
        if signal_name == "hbr":
            self.temperature_label_hbr.setText(
                f"<span style='color: orange; font-size: 12pt; font-weight: bold; font-family: Segoe UI;'>Temperature: {temp_value:.1f} °C, cursor position: {data_pos}</span>"
            )
        elif signal_name == "ventilation":
            self.temperature_label_ventilation.setText(
                f"<span style='color: orange; font-size: 12pt; font-weight: bold; font-family: Segoe UI;'>Temperature: {temp_value:.1f} °C, cursor position: {data_pos}</span>"
            )
        self.window.statusbar.showMessage(
            f"Temperature: {temp_value:.1f}°C, {data_pos = }"
        )

    def create_console_widget(self) -> None:
        module_names = [
            "self (MainWindow)",
            "pg (pyqtgraph)",
            "np (numpy)",
            "pl (polars)",
            "r (rich)",
        ]
        namespace: "dict[str, types.ModuleType | MainWindow]" = {
            "self": self.window,
            "pg": pg,
            "np": np,
            "pl": pl,
            "r": r,
        }
        startup_message = f"Available namespaces: {', '.join(module_names)}.\n\nUse `r.print()` for more readable formatting."
        self.console = ConsoleWidget(
            parent=self.window,
            namespace=namespace,
            historyFile="history.pickle",
            text=startup_message,
        )
        self.console_dock = QDockWidget("Debug Console")
        self.console_dock.setWidget(self.console)
        self.console_dock.setMinimumSize(800, 600)
        self.console_dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.console_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetClosable
        )

    @Slot()
    def show_console_widget(self) -> None:
        if self.console_dock.isVisible():
            self.console_dock.close()
        else:
            self.console_dock.show()
            self.console.input.setFocus()

    @Slot(str)
    def handle_filter_method_changed(self, text: str) -> None:
        method = self.window.filter_method

        for widget_name, enabled in FILTER_INPUT_STATES[method].items():
            getattr(self.window, widget_name).setEnabled(enabled)

        self.sig_filter_inputs_ready.emit()

    def _set_elgendi_cleaning_params(self) -> None:
        self.window.combo_box_filter_method.blockSignals(True)
        self.window.combo_box_filter_method.setValue("butterworth")
        self.window.combo_box_filter_method.blockSignals(False)
        self.window.dbl_spin_box_lowcut.setValue(0.5)
        self.window.dbl_spin_box_highcut.setValue(8.0)
        self.window.spin_box_order.setValue(3)
        self.window.slider_order.setValue(3)
        self.sig_filter_inputs_ready.emit()

    @Slot()
    def handle_preprocess_pipeline_changed(self) -> None:
        pipeline_value = self.window.pipeline
        if pipeline_value == "custom":
            self.window.container_custom_filter_inputs.setEnabled(True)
            selected_filter = self.window.filter_method
            self.handle_filter_method_changed(selected_filter)

        elif pipeline_value == "ppg_elgendi":
            self.window.container_custom_filter_inputs.setEnabled(False)
            self.window.combo_box_filter_method.setValue("butterworth")
            self._set_elgendi_cleaning_params()
        else:
            # TODO: add UI and logic for other signal cleaning pipelines
            self.window.container_custom_filter_inputs.setEnabled(False)
            msg = f"Selected pipeline {pipeline_value} not yet implemented, use either 'custom' or 'ppg_elgendi'."
            self.window.sig_show_message.emit(msg, "info")
            return
