import types
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import polars as pl
import pyqtgraph as pg
import rich
from pyqtgraph.console import ConsoleWidget
from pyqtgraph.parametertree import ParameterTree
from PySide6.QtCore import (
    QDate,
    QObject,
    QPointF,
    Qt,
    Signal,
    Slot,
)
from PySide6.QtGui import QActionGroup, QStandardItemModel
from PySide6.QtWidgets import (
    QDockWidget,
    QVBoxLayout,
    QWidget,
)

from .models.io import parse_file_name
from .models.peaks import UIPeakDetection
from .type_aliases import (
    FilterMethod,
    PeakDetectionMethod,
    Pipeline,
    SignalName,
)
from .views.plots import PlotHandler

if TYPE_CHECKING:
    from .app import MainWindow

COMBO_BOX_ITEMS = {
    "combo_box_peak_detection_method": {
        "Elgendi (PPG, fast)": "elgendi_ppg",
        "Local Maxima (Any, fast)": "local",
        "XQRS (ECG, medium)": "wfdb_xqrs",
        "Neurokit (ECG, fast)": "neurokit2",
        "ProMAC (ECG, slow)": "promac",
        "Pan and Tompkins (ECG, medium)": "pantompkins",
        "Nabian (ECG)": "nabian",
        "Gamboa (ECG)": "gamboa",
        "Slope Sum Function (ECG)": "slopesumfunction",
        "Zong (ECG)": "zong",
        "Hamilton (ECG)": "hamilton",
        "Christov (ECG)": "christov",
        "Engzeemod (ECG)": "engzeemod",
        "Elgendi (ECG)": "elgendi_ecg",
        "Kalidas (ECG)": "kalidas",
        "Martinez (ECG)": "martinez",
        "Rodrigues (ECG)": "rodrigues",
        "VGraph (ECG)": "vgraph",
    },
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
        "currentText": "normoxic",
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
        "currentText": "custom",
    },
    "combo_box_filter_method": {
        "enabled": True,
        "currentText": "None",
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
        "currentText": "elgendi_ppg",
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
}

INITIAL_STATE_METHODS_MAP = {
    "enabled": "setEnabled",
    "checked": "setChecked",
    "text": "setText",
    "model": "setModel",
    "value": "setValue",
    "currentText": "setCurrentText",
    "currentIndex": "setCurrentIndex",
    "date": "setDate",
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
        self.create_peak_detection_trees()

        # File Info
        self._prepare_widgets()

        # Statusbar
        self.window.statusbar.showMessage("Ready")

        # Toolbar Plots
        self._prepare_toolbars()

        # Plots
        self._prepare_plots()
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
        self.window.combo_box_peak_detection_method.setCurrentIndex(0)

    def _prepare_plots(self) -> None:
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
        self.window.combo_box_peak_detection_method.currentTextChanged.connect(
            self.on_peak_detection_method_changed
        )

        self.window.btn_apply_filter.clicked.connect(
            lambda: self.window.btn_detect_peaks.setEnabled(True)
        )
        self.window.btn_detect_peaks.clicked.connect(
            lambda: self.window.btn_compute_results.setEnabled(True)
        )

        self.window.action_open_console.triggered.connect(self.show_console_widget)
        self.window.tabs_main.currentChanged.connect(self.on_main_tab_changed)

    def _prepare_toolbars(self) -> None:
        self.window.toolbar_plots.setVisible(False)

        self.action_group_mouse_mode = QActionGroup(self.window)
        self.action_group_mouse_mode.setExclusive(True)
        self.action_group_mouse_mode.addAction(self.window.action_rect_mode)
        self.action_group_mouse_mode.addAction(self.window.action_pan_mode)

    def _set_combo_box_items(self) -> None:
        for key, value in COMBO_BOX_ITEMS.items():
            getattr(self.window, key).clear()
            getattr(self.window, key).setItems(value)

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
        self.window.combo_box_filter_column.blockSignals(False)
        try:
            parsed_date, parsed_id, parsed_oxy = parse_file_name(Path(path).name)
            self.window.date_edit_file_info.setDate(
                QDate(parsed_date.year, parsed_date.month, parsed_date.day)
            )
            self.window.line_edit_subject_id.setText(parsed_id)
            self.window.combo_box_oxygen_condition.setValue(parsed_oxy)
        except Exception:
            self.window.date_edit_file_info.setDate(QDate(1970, 1, 1))
            self.window.line_edit_subject_id.setText("")
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

    def create_peak_detection_trees(self) -> None:
        method: PeakDetectionMethod = "elgendi_ppg"

        self.peak_tree = ParameterTree()
        self.peak_params = UIPeakDetection(name="Adjustable Parameters")
        self.peak_params.set_method(method)
        self.peak_tree.setParameters(self.peak_params, showTop=True)

        self.window.container_peak_detection_sidebar.layout().addWidget(self.peak_tree)

    @Slot()
    def on_peak_detection_method_changed(self) -> None:
        method = cast(
            PeakDetectionMethod, self.window.combo_box_peak_detection_method.value()
        )
        try:
            self.peak_params.set_method(method)
        except NotImplementedError as e:
            self.window.sig_show_message.emit(str(e), "warning")

    @Slot(int)
    def on_main_tab_changed(self, index: int) -> None:
        is_index_one = index == 1

        self.window.toolbar_plots.setVisible(is_index_one)
        self.window.toolbar_plots.setEnabled(is_index_one)

    def create_plot_widgets(self) -> None:
        self.window.plot_widget_hbr.setLayout(QVBoxLayout())
        self.window.plot_widget_vent.setLayout(QVBoxLayout())

        self.window.plot_widget_hbr.layout().addWidget(self.plot.plot_widgets.get_signal_widget("hbr"))
        self.window.plot_widget_vent.layout().addWidget(self.plot.plot_widgets.get_signal_widget("ventilation"))

        self.window.plot_widget_hbr.layout().addWidget(self.plot.plot_widgets.get_rate_widget("hbr"))
        self.window.plot_widget_vent.layout().addWidget(self.plot.plot_widgets.get_rate_widget("ventilation"))

        self.temperature_label_hbr = pg.LabelItem(
            text="Temperature: -",
            parent=self.plot.plot_widgets.get_signal_widget("hbr").plotItem,
            angle=0,
        )
        self.temperature_label_ventilation = pg.LabelItem(
            text="Temperature: -",
            parent=self.plot.plot_widgets.get_signal_widget("ventilation").plotItem,
            angle=0,
        )
        self.plot.plot_widgets.get_signal_widget("hbr").plotItem.scene().sigMouseMoved.connect(
            lambda pos: self.update_temperature_label("hbr", pos)
        )
        self.plot.plot_widgets.get_signal_widget("ventilation").plotItem.scene().sigMouseMoved.connect(
            lambda pos: self.update_temperature_label("ventilation", pos)
        )

    @Slot(QPointF)
    def update_temperature_label(self, signal_name: SignalName, pos: QPointF) -> None:
        if not hasattr(self.window, "data"):
            return
        if self.window.data.df.is_empty():
            return
        data_pos = int(
            self.plot.plot_widgets.get_signal_widget(signal_name).plotItem.vb.mapSceneToView(pos).x()
        )
        try:
            temp_value = self.window.data.df.get_column("temperature").to_numpy(
                zero_copy_only=True
            )[np.clip(data_pos, 0, self.window.data.df.shape[0] - 1)]
        except Exception:
            if data_pos < 0:
                default_index = 0
            elif data_pos > self.window.data.df.shape[0]:
                default_index = -1
            else:
                default_index = data_pos
            temp_value = self.window.data.df.get_column("temperature").to_numpy(
                zero_copy_only=True
            )[default_index]
        if signal_name == "hbr":
            self.temperature_label_hbr.setText(f"Temperature: {temp_value:.1f}°C")
        elif signal_name == "ventilation":
            self.temperature_label_ventilation.setText(
                f"Temperature: {temp_value:.1f}°C, cursor position: {data_pos}"
            )
        self.window.statusbar.showMessage(
            f"Temperature: {temp_value:.1f}°C, {data_pos=}"
        )

    def create_console_widget(self) -> None:
        module_names = [
            "self (App)",
            "pg (pyqtgraph)",
            "np (numpy)",
            "pl (polars)",
            "rich (rich)",
        ]
        namespace: "dict[str, types.ModuleType | MainWindow]" = {
            "self": self.window,
            "pg": pg,
            "np": np,
            "pl": pl,
            "rich": rich,
        }
        startup_message = f"Available namespaces: {*module_names, = }"
        self.console = ConsoleWidget(
            parent=self.window,
            namespace=namespace,
            historyFile="history.pickle",
            text=startup_message,
        )
        self.console_dock = QDockWidget("Console")
        self.console_dock.setWidget(self.console)
        self.console_dock.setMinimumSize(600, 300)
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
        method = cast(FilterMethod, self.window.combo_box_filter_method.value())

        if method != "None":
            states = {
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
            }
            for widget_name, enabled in states[method].items():
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
        self.window.spin_box_window_size.setValue(250)
        self.window.slider_window_size.setValue(250)
        self.sig_filter_inputs_ready.emit()

    @Slot()
    def handle_preprocess_pipeline_changed(self) -> None:
        pipeline_value = cast(
            Pipeline, self.window.combo_box_preprocess_pipeline.value()
        )
        if pipeline_value == "custom":
            self.window.container_custom_filter_inputs.setEnabled(True)
            selected_filter = cast(str, self.window.combo_box_filter_method.value())
            self.handle_filter_method_changed(selected_filter)

        elif pipeline_value == "ppg_elgendi":
            self.window.container_custom_filter_inputs.setEnabled(False)
            self._set_elgendi_cleaning_params()
        else:
            # TODO: add UI and logic for other signal cleaning pipelines
            self.window.container_custom_filter_inputs.setEnabled(False)
            msg = f"Selected pipeline {pipeline_value} not yet implemented, use either 'custom' or 'ppg_elgendi'."
            self.window.sig_show_message.emit(msg, "info")
            return
            # logger.warning(f"Pipeline {pipeline} not implemented yet.")
            # "ecg_neurokit2"
            # "ecg_biosppy"
            # "ecg_pantompkins1985"
            # "ecg_hamilton2002"
            # "ecg_elgendi2010"
            # "ecg_engzeemod2012"
