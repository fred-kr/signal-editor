import types
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import polars as pl
import pyqtgraph as pg
import rich
from loguru import logger
from pyqtgraph.console import ConsoleWidget
from pyqtgraph.parametertree import ParameterTree
from PySide6.QtCore import (
    QObject,
    QPointF,
    Qt,
    Signal,
    Slot,
)
from PySide6.QtWidgets import (
    QDockWidget,
    QMessageBox,
    QSizePolicy,
    QVBoxLayout,
)

from .custom_types import (
    InfoProcessingParams,
    InfoWorkingData,
    NormMethod,
    OxygenCondition,
    PeakDetectionMethod,
    PeaksPPGElgendi,
    Pipeline,
    SignalFilterParameters,
    SignalName,
)
from .models.data import (
    Identifier,
)
from .models.peaks import ElgendiPPGPeaks
from .views.plots import PlotManager

if TYPE_CHECKING:
    from .app import MainWindow


class UIHandler(QObject):
    sig_filter_inputs_ready = Signal()
    sig_preprocess_pipeline_ready = Signal(str)
    sig_ready_for_cleaning = Signal()
    sig_apply_filter = Signal(dict)
    sig_peak_detection_inputs = Signal(dict)
    sig_ready_for_export = Signal()

    def __init__(self, window: "MainWindow", plot: PlotManager) -> None:
        super(UIHandler, self).__init__()
        self.window = window
        self.plot = plot
        self.setup_widgets()
        self.connect_signals()

    def setup_widgets(self) -> None:
        # Signal Filtering
        self._prepare_inputs()

        # # Peak Detection
        self.create_peak_detection_trees()

        # File Info
        self._prepare_widgets()

        # Statusbar
        self.create_statusbar()

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
        self.window.combo_box_preprocess_pipeline.setCurrentIndex(0)
        self.window.container_standard_filter_method.setEnabled(True)
        self.window.combo_box_filter_method.setCurrentIndex(0)
        self.window.combo_box_standardizing_method.setCurrentIndex(0)
        self.window.container_signal_filter_inputs.setEnabled(False)
        self.window.dbl_spin_box_lowcut.setValue(0.5)
        self.window.dbl_spin_box_highcut.setValue(8.0)
        self.window.spin_box_order.setValue(3)
        self.window.slider_order.setValue(3)
        self.window.spin_box_window_size.setValue(251)
        self.window.slider_window_size.setValue(251)
        # Peak Detection
        self.window.combo_box_peak_detection_method.setCurrentIndex(0)
        self.window.stacked_widget_peak_detection.setCurrentIndex(0)

    def _prepare_plots(self) -> None:
        self.window.stacked_hbr_vent.setCurrentIndex(0)
        self.window.btn_group_plot_view.setId(self.window.btn_view_hbr, 0)
        self.window.btn_group_plot_view.setId(self.window.btn_view_vent, 1)

    def connect_signals(self) -> None:
        self.window.tabWidget.currentChanged.connect(self.handle_tab_changed)
        self.window.combo_box_filter_method.currentTextChanged.connect(
            self.handle_filter_method_changed
        )
        self.window.combo_box_preprocess_pipeline.currentTextChanged.connect(
            self.handle_preprocess_pipeline_changed
        )

        self.window.btn_group_plot_view.idClicked.connect(
            lambda index: self.window.stacked_hbr_vent.setCurrentIndex(index)  # type: ignore
        )
        self.window.btn_apply_filter.clicked.connect(self.emit_filter_settings)
        self.window.btn_apply_filter.clicked.connect(
            lambda: self.window.btn_find_peaks.setEnabled(True)
        )
        self.window.btn_find_peaks.clicked.connect(self.emit_peak_detection_inputs)
        self.window.btn_find_peaks.clicked.connect(
            lambda: self.window.btn_compute_results.setEnabled(True)
        )

        self.window.action_open_console.triggered.connect(self.show_console_widget)
        self.window.tabWidget.currentChanged.connect(self.handle_tab_changed)

        self.window.btn_compute_results.clicked.connect(self.update_results)
        self.window.btn_export_to_csv.clicked.connect(self.export_to_csv)
        self.window.btn_export_to_excel.clicked.connect(self.export_to_excel)
        self.window.btn_export_to_text.clicked.connect(self.export_to_text)

    # Export Methods +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
    def get_export_params(self) -> tuple[str, str, str]:
        output_dir = self.window.line_edit_output_dir.text()
        signal_name = (
            "hbr"
            if self.window.tab_widget_results.currentIndex() == 0
            else "ventilation"
        )
        try:
            original_file_name = getattr(
                self.window, f"{signal_name}_results"
            ).identifier.file_name
        except Exception as e:
            error_dialog = QMessageBox(
                QMessageBox.Icon.Information,
                "Error",
                f"There are no results to export for `{signal_name}`: {e}",
            )
            error_dialog.exec()
            return "", "", ""

        return output_dir, signal_name, original_file_name

    @Slot()
    def export_to_csv(self) -> None:
        output_dir, signal_name, original_file_name = self.get_export_params()
        if output_dir == "":
            return
        self.window.get_results_table(signal_name).write_csv(
            f"{output_dir}/results_{signal_name}_{Path(original_file_name).stem}.csv",
        )

    @Slot()
    def export_to_excel(self) -> None:
        output_dir, signal_name, original_file_name = self.get_export_params()
        if output_dir == "":
            return
        self.window.get_results_table(signal_name).write_excel(
            f"{output_dir}/results_{signal_name}_{Path(original_file_name).stem}.xlsx",
        )

    @Slot()
    def export_to_text(self) -> None:
        output_dir, signal_name, original_file_name = self.get_export_params()
        if output_dir == "":
            return
        self.window.get_results_table(signal_name).write_csv(
            f"{output_dir}/results_{signal_name}_{Path(original_file_name).stem}.txt",
            separator="\t",
        )

    def _prepare_toolbars(self) -> None:
        self.window.toolbar_plots.setVisible(False)

    @Slot()
    def update_results(self) -> None:
        signal_name = self.window.get_signal_name()
        with pg.BusyCursor():
            self.window.make_results(signal_name)
            self.window.make_results_table()

        self.sig_ready_for_export.emit()

    def get_peak_detection_inputs(self) -> PeaksPPGElgendi:
        group_name = "Peak Detection (Elgendi PPG)"
        return {
            "peakwindow": self.elgendi_params.child(group_name)
            .child("peakwindow")
            .value(),
            "beatwindow": self.elgendi_params.child(group_name)
            .child("beatwindow")
            .value(),
            "beatoffset": self.elgendi_params.child(group_name)
            .child("beatoffset")
            .value(),
            "mindelay": self.elgendi_params.child(group_name).child("mindelay").value(),
        }

    @Slot()
    def emit_peak_detection_inputs(self) -> None:
        logger.debug(
            f"Emitting peak detection inputs: {self.get_peak_detection_inputs()}"
        )
        self.sig_peak_detection_inputs.emit(self.get_peak_detection_inputs())

    def create_peak_detection_trees(self) -> None:
        self.elgendi_layout = QVBoxLayout()
        self.elgendi_tree = ParameterTree()
        self.elgendi_params = ElgendiPPGPeaks().make_parameters()
        self.elgendi_tree.setParameters(self.elgendi_params, showTop=False)
        self.elgendi_layout.addWidget(self.elgendi_tree)

        self.window.page_peak_detection_elgendi.setLayout(self.elgendi_layout)

    @Slot(int)
    def handle_tab_changed(self, index: int) -> None:
        is_index_two = index == 2
        is_index_one = index == 1

        self.window.widget_sidebar.setVisible(not is_index_two)

        self.window.toolbar_plots.setVisible(is_index_one)
        self.window.toolbar_plots.setEnabled(is_index_one)

    def create_plot_widgets(self) -> None:
        self.window.plot_widget_hbr.setLayout(QVBoxLayout())
        self.window.plot_widget_vent.setLayout(QVBoxLayout())

        self.window.plot_widget_hbr.layout().addWidget(self.plot.hbr_plot_widget)
        self.window.plot_widget_vent.layout().addWidget(
            self.plot.ventilation_plot_widget
        )

        self.window.plot_widget_hbr.layout().addWidget(self.plot.bpm_hbr_plot_widget)
        self.window.plot_widget_vent.layout().addWidget(
            self.plot.bpm_ventilation_plot_widget
        )

        self.temperature_label_hbr = pg.LabelItem(
            text="Temperature: -",
            parent=self.plot.hbr_plot_widget.plotItem,
            angle=0,
        )
        self.temperature_label_ventilation = pg.LabelItem(
            text="Temperature: -",
            parent=self.plot.ventilation_plot_widget.plotItem,
            angle=0,
        )
        self.plot.hbr_plot_widget.plotItem.scene().sigMouseMoved.connect(  # type: ignore
            lambda pos: self.update_temperature_label("hbr", pos)  # type: ignore
        )
        self.plot.ventilation_plot_widget.plotItem.scene().sigMouseMoved.connect(  # type: ignore
            lambda pos: self.update_temperature_label("ventilation", pos)  # type: ignore
        )

    @Slot(QPointF)
    def update_temperature_label(self, signal_name: SignalName, pos: QPointF) -> None:
        if not hasattr(self.window, "dm"):
            return
        if not hasattr(self.window.dm, "data"):
            return
        data_pos = int(
            getattr(self.plot, f"{signal_name}_plot_widget")
            .plotItem.vb.mapSceneToView(pos)
            .x()
        )
        try:
            temp_value = self.window.dm.data.get_column("temperature").to_numpy(
                zero_copy_only=True
            )[np.clip(data_pos, 0, self.window.dm.data.shape[0] - 1)]
        except Exception:
            if data_pos < 0:
                default_index = 0
            elif data_pos > self.window.dm.data.shape[0]:
                default_index = -1
            else:
                default_index = data_pos
            temp_value = self.window.dm.data.get_column("temperature").to_numpy(
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
        startup_message = f"Available namespaces: {*module_names,=}"
        self.window.console = ConsoleWidget(
            namespace=namespace, historyFile="history.pickle", text=startup_message
        )
        dock = QDockWidget("Console", parent=self.window, flags=Qt.WindowType.SubWindow)
        dock.setWidget(self.window.console)
        dock.setFloating(True)
        dock.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        dock.setMinimumSize(200, 100)
        dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        self.console_dock = dock
        self.console_dock.setObjectName("console_dock")
        self.window.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea, self.console_dock
        )
        self.console_dock.setVisible(False)

    @Slot()
    def show_console_widget(self) -> None:
        self.console_dock.setVisible(not self.console_dock.isVisible())
        if self.console_dock.isVisible():
            self.console_dock.setFocus()

    def create_statusbar(self) -> None:
        self.window.statusbar.showMessage("Ready")

    @Slot(str)
    def handle_filter_method_changed(self, method: str) -> None:
        self.window.container_signal_filter_inputs.setEnabled(method != "None")

        if method != "None":
            states = {
                "butterworth": {
                    "dbl_spin_box_lowcut": True,
                    "dbl_spin_box_highcut": True,
                    "spin_box_order": True,
                    "spin_box_window_size": False,
                },
                "butterworth_ba": {
                    "dbl_spin_box_lowcut": True,
                    "dbl_spin_box_highcut": True,
                    "spin_box_order": True,
                    "spin_box_window_size": False,
                },
                "bessel": {
                    "dbl_spin_box_lowcut": True,
                    "dbl_spin_box_highcut": True,
                    "spin_box_order": True,
                    "spin_box_window_size": False,
                },
                "fir": {
                    "dbl_spin_box_lowcut": True,
                    "dbl_spin_box_highcut": True,
                    "spin_box_order": False,
                    "spin_box_window_size": True,
                },
                "savgol": {
                    "dbl_spin_box_lowcut": False,
                    "dbl_spin_box_highcut": False,
                    "spin_box_order": True,
                    "spin_box_window_size": True,
                },
            }
            for widget_name, enabled in states[method].items():
                getattr(self.window, widget_name).setEnabled(enabled)

        self.sig_filter_inputs_ready.emit()

    def _set_elgendi_cleaning_params(self) -> None:
        self.window.combo_box_filter_method.blockSignals(True)
        self.window.combo_box_filter_method.setCurrentText("butterworth")
        self.window.combo_box_filter_method.blockSignals(False)
        self.window.dbl_spin_box_lowcut.setValue(0.5)
        self.window.dbl_spin_box_highcut.setValue(8.0)
        self.window.spin_box_order.setValue(3)
        self.window.dbl_spin_box_lowcut.setEnabled(False)
        self.window.dbl_spin_box_highcut.setEnabled(False)
        self.window.spin_box_order.setEnabled(False)
        self.sig_filter_inputs_ready.emit()

    @Slot(str)
    def handle_preprocess_pipeline_changed(self, pipeline: str) -> None:
        logger.debug(f"Preprocess pipeline changed to {pipeline}.")
        if pipeline == "custom":
            self.window.container_standard_filter_method.setEnabled(True)
            self.window.container_signal_filter_inputs.setEnabled(True)
            selected_filter = self.window.combo_box_filter_method.currentText()
            self.handle_filter_method_changed(selected_filter)

        elif pipeline == "ppg_elgendi":
            self.window.container_standard_filter_method.setEnabled(False)
            self.window.container_signal_filter_inputs.setEnabled(False)
            self._set_elgendi_cleaning_params()
        else:
            # TODO: add UI and logic for other signal cleaning pipelines
            self.window.container_standard_filter_method.setEnabled(False)
            logger.warning(f"Pipeline {pipeline} not implemented yet.")
            # "ecg_neurokit2"
            # "ecg_biosppy"
            # "ecg_pantompkins1985"
            # "ecg_hamilton2002"
            # "ecg_elgendi2010"
            # "ecg_engzeemod2012"

        self.sig_preprocess_pipeline_ready.emit(pipeline)

    def get_filter_settings(self) -> SignalFilterParameters:
        current_pipeline = self.window.combo_box_preprocess_pipeline.currentText()

        current_method = self.window.combo_box_filter_method.currentText()
        if current_pipeline != "custom":
            current_method = "None"

        if current_method not in [
            "butterworth",
            "butterworth_ba",
            "savgol",
            "fir",
            "bessel",
            "None",
        ]:
            logger.error(f"Unknown filter method: {current_method}")
            raise ValueError(f"Unknown filter method: {current_method}")

        filter_settings = cast(
            SignalFilterParameters,
            {
                "method": current_method,
            },
        )

        if filter_settings["method"] != "None":
            if self.window.dbl_spin_box_lowcut.isEnabled():
                filter_settings["lowcut"] = self.window.dbl_spin_box_lowcut.value()

            if self.window.dbl_spin_box_highcut.isEnabled():
                filter_settings["highcut"] = self.window.dbl_spin_box_highcut.value()

            if self.window.spin_box_order.isEnabled():
                filter_settings["order"] = self.window.spin_box_order.value()

            if self.window.spin_box_window_size.isEnabled():
                filter_settings[
                    "window_size"
                ] = self.window.spin_box_window_size.value()

        logger.debug(f"Using filter settings: {filter_settings}")
        return filter_settings

    @Slot()
    def emit_filter_settings(self) -> None:
        logger.debug(f"Emitting filter settings: {self.get_filter_settings()}")
        self.sig_apply_filter.emit(self.get_filter_settings())

    def get_standardizing_method(self) -> NormMethod:
        return cast(
            NormMethod, self.window.combo_box_standardizing_method.currentText()
        )

    def get_preprocess_pipeline(self) -> Pipeline:
        return cast(Pipeline, self.window.combo_box_preprocess_pipeline.currentText())

    def get_identifier(self) -> Identifier:
        signal_name = self.window.get_signal_name()
        file_name = self.window.line_edit_active_file.text()
        subject_id = self.window.line_edit_subject_id.text()
        date_of_recording = self.window.date_edit_file_info.date().toPython()
        date_of_recording = cast(date, date_of_recording)
        oxygen_condition = self.window.combo_box_oxygen_condition.currentText()
        oxygen_condition = cast(OxygenCondition, oxygen_condition)
        return Identifier(
            signal_name=signal_name,
            file_name=file_name,
            subject_id=subject_id,
            date_of_recording=date_of_recording,
            oxygen_condition=oxygen_condition,
        )

    def get_info_working_data(self) -> InfoWorkingData:
        signal_name = self.window.get_signal_name()
        subset_column = self.window.combo_box_filter_column.currentText()
        subset_lower_bound = self.window.dbl_spin_box_subset_min.value()
        subset_upper_bound = self.window.dbl_spin_box_subset_max.value()
        n_samples = self.window.dm.data.shape[0]
        return InfoWorkingData(
            signal_name=signal_name,
            subset_column=subset_column,
            subset_lower_bound=subset_lower_bound,
            subset_upper_bound=subset_upper_bound,
            n_samples=n_samples,
        )

    def get_info_processing_params(self) -> InfoProcessingParams:
        signal_name = self.window.get_signal_name()
        sampling_rate = self.window.spin_box_fs.value()
        preprocess_pipeline = self.get_preprocess_pipeline()
        filter_parameters = self.get_filter_settings()
        standardization_method = self.get_standardizing_method()
        peak_detection_method = (
            self.window.combo_box_peak_detection_method.currentText()
        )
        peak_detection_method = cast(PeakDetectionMethod, peak_detection_method)
        peak_method_parameters = self.get_peak_detection_inputs()
        return InfoProcessingParams(
            signal_name=signal_name,
            sampling_rate=sampling_rate,
            preprocess_pipeline=preprocess_pipeline,
            filter_parameters=filter_parameters,
            standardization_method=standardization_method,
            peak_detection_method=peak_detection_method,
            peak_method_parameters=peak_method_parameters,
        )
