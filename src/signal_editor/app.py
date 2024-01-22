import cProfile
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

import h5py
import numpy as np
import pyqtgraph as pg
from loguru import logger
from PySide6.QtCore import (
    QByteArray,
    QDate,
    QFileInfo,
    QProcess,
    QSettings,
    Signal,
    Slot,
)
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QInputDialog,
    QMainWindow,
    QMenu,
    QMessageBox,
)

from . import type_aliases as _t
from .handlers.config_handler import ConfigHandler
from .handlers.data_handler import DataHandler
from .handlers.helpers import table_view_helper
from .handlers.plot_handler import PlotHandler
from .handlers.style_handler import ThemeSwitcher
from .handlers.ui_handler import UIHandler
from .models import data as _data
from .models import io as _io
from .models import result as _result
from .models.signal import SectionID
from .views.main_window import Ui_MainWindow


class SignalEditor(QMainWindow, Ui_MainWindow):
    sig_data_filtered = Signal(str)
    sig_data_loaded = Signal()
    sig_new_peak_data = Signal(str)
    sig_show_message = Signal(str, str)
    sig_data_restored = Signal()
    sig_active_section_changed = Signal(str)
    sig_update_view_range = Signal(int, float, float)

    def __init__(self) -> None:
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("Signal Editor")
        self.config = ConfigHandler("config.ini")
        self.theme_switcher = ThemeSwitcher()
        self.plot = PlotHandler(self)
        self.data = DataHandler(self)
        self._read_settings()
        self.ui = UIHandler(self, self.plot)
        self.file_info: QFileInfo = QFileInfo()
        self._connect_signals()
        self._on_init_finished()

    def _on_init_finished(self) -> None:
        self.line_edit_output_dir.setText(self.config.output_dir.as_posix())
        saved_style = self.config.style
        if saved_style in ["dark", "light"]:
            self.theme_switcher.set_style(saved_style)
        else:
            self.theme_switcher.set_style("dark")
        self._add_style_toggle()
        self._add_profiler()
        self.data.fs = int(self.config.config.get("Defaults", "SampleRate", fallback=-1))
        self._setup_section_widgets()

    # region Dev
    def _add_profiler(self) -> None:
        self.menubar.addAction("Start Profiler", self._start_profiler)
        self.menubar.addAction("Stop Profiler", self._stop_profiler)

    # endregion

    # region Properties
    @property
    def output_dir(self) -> Path:
        return self.config.output_dir

    @output_dir.setter
    def output_dir(self, value: Path | str) -> None:
        self.config.output_dir = Path(value)

    @property
    def app_dir(self) -> Path:
        return self.config.app_dir

    @property
    def data_dir(self) -> Path:
        return self.config.data_dir

    @data_dir.setter
    def data_dir(self, value: Path | str) -> None:
        self.config.data_dir = Path(value)

    @property
    def signal_name(self) -> _t.SignalName | str:
        return self.combo_box_signal_column.currentText()

    @property
    def active_section_id(self) -> SectionID:
        return SectionID(self.combo_box_section_select.currentText())

    @property
    def scale_method(self) -> _t.ScaleMethod:
        return cast(_t.ScaleMethod, self.combo_box_scale_method.value())

    @property
    def pipeline(self) -> _t.Pipeline:
        return cast(_t.Pipeline, self.combo_box_preprocess_pipeline.value())

    @property
    def filter_method(self) -> _t.FilterMethod:
        return cast(_t.FilterMethod, self.combo_box_filter_method.value())

    @property
    def peak_detection_method(self) -> _t.PeakDetectionMethod:
        return cast(_t.PeakDetectionMethod, self.combo_box_peak_detection_method.value())

    @property
    def xqrs_peak_direction(self) -> _t.WFDBPeakDirection:
        return cast(_t.WFDBPeakDirection, self.peak_xqrs_peak_dir.value())

    # endregion

    def _add_style_toggle(self) -> None:
        self.menubar.addSeparator()
        self.menubar.addAction("Switch Theme", self.theme_switcher.switch_theme)

    def _setup_section_widgets(self) -> None:
        self.combo_box_section_select.blockSignals(True)
        self.combo_box_section_select.addItem("IN_001")
        self.combo_box_section_select.blockSignals(False)
        add_section_menu = QMenu(self.btn_section_add)
        add_section_menu.addAction("Included", self._new_included_section)
        add_section_menu.addAction("Excluded", self._new_excluded_section)
        self.btn_section_add.setMenu(add_section_menu)

    def _connect_signals(self) -> None:
        """
        Connect signals to slots.
        """

        # Menu & Toolbar Actions
        self.action_exit.triggered.connect(self.close)
        self.action_load_state.triggered.connect(self.restore_state)
        self.action_save_state.triggered.connect(self.save_state)
        self.action_select_file.triggered.connect(self.select_data_file)
        self.action_next_section.triggered.connect(self._on_next_section)
        self.action_previous_section.triggered.connect(self._on_previous_section)

        # Plotting Related Actions
        self.action_remove_peak_rect.triggered.connect(self.plot.show_selection_rect)
        self.action_remove_selected_peaks.triggered.connect(self.plot.remove_selected_scatter)
        self.action_reset_view.triggered.connect(self._emit_data_range_info)
        self.sig_update_view_range.connect(self.plot.reset_view_range)
        self.plot.sig_peaks_edited.connect(self.handle_scatter_clicked)

        # Button Actions
        self.btn_apply_filter.clicked.connect(self.handle_apply_filter)
        self.btn_browse_output_dir.clicked.connect(self.select_output_location)
        self.btn_compute_results.clicked.connect(self.update_results)
        self.btn_detect_peaks.clicked.connect(self.handle_peak_detection)
        self.btn_load_selection.clicked.connect(self.handle_load_selection)
        self.btn_save_to_hdf5.clicked.connect(self.save_to_hdf5)
        self.btn_select_file.clicked.connect(self.select_data_file)

        # Data Export Actions
        self.btn_export_focused.clicked.connect(self.export_focused_result)

        # Data Related Signals
        self.sig_data_filtered.connect(self.handle_table_view_data)
        self.sig_data_loaded.connect(lambda: self.tabs_main.setCurrentIndex(1))
        self.sig_data_loaded.connect(self.handle_plot_draw)
        self.sig_data_loaded.connect(self.handle_table_view_data)
        self.sig_data_restored.connect(self.refresh_app_state)
        self.sig_new_peak_data.connect(self.handle_draw_results)
        self.sig_show_message.connect(self.show_message)
        self.sig_active_section_changed.connect(self._on_section_changed)

        # UI Handler Signals
        self.ui.sig_section_confirmed.connect(self._on_section_confirmed)
        self.ui.sig_section_canceled.connect(self._on_section_canceled)

        # Widget specific Signals
        self.spin_box_sample_rate.valueChanged.connect(self.data.update_fs)
        self.combo_box_section_select.currentTextChanged.connect(self._on_section_changed)

    @Slot()
    def _new_included_section(self) -> None:
        current_limits = self.data.sig_data.active_bounds
        self.container_section_confirm_cancel.show()
        self.plot.show_section_selector("included", current_limits)

    @Slot()
    def _new_excluded_section(self) -> None:
        current_limits = self.data.sig_data.active_bounds
        self.container_section_confirm_cancel.show()
        self.plot.show_section_selector("excluded", current_limits)

    @Slot()
    def _on_section_confirmed(self) -> None:
        active_region = self.plot.region_selector
        if active_region is None:
            return
        lower, upper = active_region.getRegion()
        lower, upper = int(lower), int(upper)
        self.data.sig_data.add_section(lower, upper, set_active=True)
        self.combo_box_section_select.addItem(self.data.active_section.section_id)
        self.combo_box_section_select.setCurrentText(self.data.active_section.section_id)
        self.container_section_confirm_cancel.hide()
        self.sig_active_section_changed.emit(self.data.active_section.section_id)

    @Slot()
    def _on_section_canceled(self) -> None:
        self.container_section_confirm_cancel.hide()
        self.plot.hide_section_selector()

    @Slot(str)
    def _on_section_changed(self, section_id: SectionID) -> None:
        self.plot.hide_section_selector()
        self.data.sig_data.set_active_section(section_id)
        self.handle_plot_draw()

    @Slot()
    def _on_next_section(self) -> None:
        self.data.sig_data.next_section(self.active_section_id)

    @Slot()
    def _on_previous_section(self) -> None:
        self.data.sig_data.previous_section(self.active_section_id)

    @Slot()
    def remove_excluded_regions(self) -> None:
        self.data.sig_data.remove_all_excluded()

    @Slot()
    def _emit_data_range_info(self) -> None:
        data = self.data.sig_data.get_active_signal_data()
        len_data, min_data, max_data = data.shape[0], data.min(), data.max()
        self.sig_update_view_range.emit(len_data, min_data, max_data)

    def read_hdf5(self, file_path: str | Path) -> None:
        with h5py.File(file_path, "r") as f:
            self.restore_state(f)

    @Slot()
    def run_hdf5view(self) -> None:
        """
        Runs `hdf5view.exe` to allow inspecting results stored as HDF5. Made by Martin
        Swarbrick. Link to github repo: https://github.com/marts/hdf5view
        """
        self.hdf5view_process = QProcess(self)
        self.hdf5view_process.finished.connect(self._process_finished)
        self.hdf5view_process.start("hdf5view")

    @Slot()
    def _process_finished(self) -> None:
        self.hdf5view_process = None

    @Slot()
    def save_to_hdf5(self) -> None:
        default_out = (
            f"{self.output_dir.as_posix()}/{self.file_info.completeBaseName()}_results.hdf5"
        )
        if file_path := QFileDialog.getSaveFileName(
            self,
            "Save to HDF5",
            default_out,
            "HDF5 Files (*.hdf5 *.h5)",
        )[0]:
            _io.write_hdf5(file_path, self.make_results(self.signal_name))

    @Slot()
    def update_results(self) -> None:
        if len(self.data.sig_data.active_peaks) == 0:
            msg = f"No peaks detected for signal '{self.signal_name}'. Please run peak detection first."
            self.sig_show_message.emit(msg, "info")
            return
        self.make_results(self.signal_name)

    @Slot(str)
    def export_focused_result(self, output_format: Literal["csv", "xlsx", "txt"]) -> None:
        if not self.data.sig_data:
            msg = "No results to export. Please compute results first."
            self.sig_show_message.emit(msg, "warning")
            return

        result_file_name = self.config.make_focused_result_name(
            self.signal_name, self.file_info.baseName()
        )
        result_location = self.output_dir / result_file_name

        result_df = (
            self.data.sig_data.get_section(self.active_section_id).get_focused_result().to_polars()
        )

        try:
            if output_format == "csv":
                result_df.write_csv(f"{result_location}.csv")
            elif output_format == "xlsx":
                result_df.write_excel(f"{result_location}.xlsx")
            elif output_format == "txt":
                result_df.write_csv(f"{result_location}.txt", separator="\t")
        except Exception as e:
            msg = f"Failed to export results: {e}"
            self.sig_show_message.emit(msg, "error")
            return

    @Slot(str, str)
    def show_message(
        self, text: str, level: Literal["info", "warning", "critical", "error"]
    ) -> None:
        icon_map = {
            "info": QMessageBox.Icon.Information,
            "warning": QMessageBox.Icon.Warning,
            "critical": QMessageBox.Icon.Critical,
            "error": QMessageBox.Icon.Critical,
        }
        msg_box = QMessageBox(icon_map[level], "Notice", text)
        msg_box.exec()

    @Slot()
    def select_output_location(self) -> None:
        """
        Prompt user to select a directory for storing the exported results.
        """
        if path := QFileDialog.getExistingDirectory(
            self,
            caption="Select Output Directory",
            dir=self.output_dir.as_posix(),
            options=QFileDialog.Option.ShowDirsOnly,
        ):
            self.line_edit_output_dir.setText(path)
            self.output_dir = path

    @Slot()
    def handle_table_view_data(self) -> None:
        """
        Update the data preview table and the data info table with the current data.
        """
        # Get top and bottom parts of the data and its description
        data = self.data.df
        n_rows = 10
        df_head = data.head(n_rows)
        df_tail = data.tail(n_rows)
        df_description = data.describe(percentiles=None)

        model = _data.CompactDFModel(df_head=df_head, df_tail=df_tail)
        raw_data_table = table_view_helper.TableViewHelper(self.table_data_preview)
        raw_data_table.make_table(model)

        info = _data.DescriptiveStatsModel(df_description)
        info_table = table_view_helper.TableViewHelper(self.table_data_info)
        info_table.make_table(info)

    @Slot()
    def select_data_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            caption="Select File",
            dir=self.config.data_dir.as_posix(),
            filter="EDF (*.edf);;CSV (*.csv);;TXT (*.txt);;Feather (*.feather);;State Files (*.pkl);;All Files (*.edf *.csv *.txt *.feather *.pkl)",
            selectedFilter="All Files (*.edf *.csv *.txt *.feather *.pkl)",
        )
        if path:
            self.ui.reset_widget_state()
            self.file_info.setFile(path)
            self.line_edit_active_file.setText(Path(path).name)
            self.data.read(path)
            self.ui.update_data_select_ui(path)

            self.config.data_dir = self.file_info.dir().path()

    @Slot()
    def handle_load_selection(self) -> None:
        self.btn_load_selection.processing("Loading data...")

        try:
            self.data.df.shrink_to_fit(in_place=True)
            signal_col = self.signal_name
            if signal_col not in self.data.df.columns:
                msg = f"Signal column '{signal_col}' not found in data."
                self.btn_load_selection.feedback(False, "Error", msg)
                self.sig_show_message.emit(msg, "error")
                return

            if "index" in self.data.df.columns:
                self.data.df.drop_in_place("index")

            self.data.df = self.data.df.with_row_index()

            self.data.new_sig_data(signal_col)
            self.btn_load_selection.feedback(True, "Success", "Data loaded successfully.")
            self.sig_data_loaded.emit()
        except Exception as e:
            msg = f"Failed to load data: {e}"
            self.btn_load_selection.feedback(False, "Error", msg)
            self.sig_show_message.emit(msg, "error")
            logger.error(msg)
            self.statusbar.showMessage("Error loading data.")

    @Slot()
    def handle_plot_draw(self) -> None:
        with pg.BusyCursor():
            data = self.data.sig_data.get_active_signal_data()
            len_data, min_data, max_data = data.shape[0], data.min(), data.max()
            self.plot.draw_signal(data, self.signal_name)
            self.plot.update_view_limits(len_data, min_data, max_data)
            self.sig_update_view_range.emit(len_data, min_data, max_data)

    @Slot()
    def handle_apply_filter(self) -> None:
        self.statusbar.showMessage("Applying filter...")
        btn = self.btn_apply_filter
        btn.processing("Working...")
        with pg.BusyCursor():
            filter_params = self.get_filter_values()
            standardize_params = self.get_standardization_values()

            pipeline = self.pipeline

            signal_name = self.signal_name

            self.data.run_preprocessing(
                name=signal_name,
                pipeline=pipeline,
                filter_params=filter_params,
                standardize_params=standardize_params,
            )
            self.handle_plot_draw()
        btn.success("Done")
        self.statusbar.showMessage("Signal filtered successfully.", 3000)
        self.sig_data_filtered.emit(signal_name)

    @Slot()
    def handle_peak_detection(self) -> None:
        btn = self.btn_detect_peaks
        btn.processing("Working...")
        self.statusbar.showMessage("Detecting peaks...")
        with pg.BusyCursor():
            peak_params = self.get_peak_detection_values()
            name = self.signal_name
            active_section = self.data.active_section
            if active_section.data.shape[0] == 0:
                msg = "Signal needs to be filtered before peak detection can be performed."
                self.sig_show_message.emit(msg, "info")
                return

            self.data.run_peak_detection(
                name=name,
                peak_parameters=peak_params,
            )
            peaks, peaks_y = active_section.get_peak_xy()

            self.plot.draw_peaks(
                x_values=peaks,
                y_values=peaks_y,
                name=name,
            )
        btn.success("Done")
        self.statusbar.showMessage("Peak detection finished.", 3000)
        section_id = active_section.section_id
        self.sig_new_peak_data.emit(section_id)

    @Slot(str)
    def handle_draw_results(self, section_id: SectionID) -> None:
        active_section = self.data.sig_data.get_section(section_id)
        active_section.calculate_rate()
        rate_interp = active_section.rate_data.rate_interpolated

        self.plot.draw_rate(
            rate_interp,
            name=self.signal_name,
        )

    # def _sync_peak_indices(self, section_id: SectionID) -> None:
    #     scatter_item = self.plot.scatter_item
    #     if scatter_item is None:
    #         return
    #     plot_peaks = scatter_item.data["x"].astype(np.int32)
    #     data_peaks = self.data.active_section.peaks
    #     plot_peaks.sort()
    #     data_peaks.sort()

    #     if not np.array_equal(plot_peaks, data_peaks):
    #         self.data.sig_data.get_section_by_id(section_id).set_peaks(plot_peaks)

    @Slot(str)
    def handle_scatter_clicked(self, section_id: SectionID) -> None:
        scatter_item = self.plot.scatter_item
        if scatter_item is None:
            return
        plot_peaks = scatter_item.data["x"].astype(np.int32)
        plot_peaks.sort()
        self.data.sig_data.get_section(section_id).set_peaks(plot_peaks)
        # self.data.sig_data.active_section.calculate_rate()
        self.sig_new_peak_data.emit(section_id)

    @Slot(QCloseEvent)
    def closeEvent(self, event: QCloseEvent) -> None:
        self._write_settings()
        if hasattr(self, "hdf5view_process") and self.hdf5view_process:
            self.hdf5view_process.kill()
            self.hdf5view_process = None
        if self.ui.jupyter_console_dock.isVisible():
            self.ui.jupyter_console_dock.close()

        super().closeEvent(event)

    def _read_settings(self) -> None:
        settings = QSettings("AWI", "Signal Editor")
        geometry = settings.value("geometry", QByteArray(), type=QByteArray)

        if geometry.size():
            self.restoreGeometry(geometry)

        self.data_dir = self.config.data_dir

        self.output_dir = self.config.output_dir

        self.theme_switcher.set_style(self.config.style)

    def _write_settings(self) -> None:
        settings = QSettings("AWI", "Signal Editor")

        geometry = self.saveGeometry()
        settings.setValue("geometry", geometry)

        data_dir = self.file_info.dir().path()
        output_dir = self.line_edit_output_dir.text()

        self.config.data_dir = Path(data_dir)
        self.config.output_dir = Path(output_dir)

        self.config.style = self.theme_switcher.active_style
        self.config.sample_rate = self.data.fs
        self.config.write_config()

    @Slot()
    def _start_profiler(self):
        self.profiler = cProfile.Profile()
        self.profiler.enable()
        self.statusbar.showMessage("Profiler started.")
        logger.debug(f"Started profiling at: {datetime.now()}")

    @Slot()
    def _stop_profiler(self):
        self.profiler.disable()
        self.statusbar.showMessage("Profiler stopped.")
        self.profiler.dump_stats(
            f"./logs/profiler_{int(datetime.timestamp(datetime.now()))}.pstats"
        )

    def get_identifier(self) -> _result.ResultIdentifier:
        metadata = self.get_file_metadata()
        result_file_name = self.config.make_complete_result_name(
            self.signal_name, self.file_info.baseName()
        )
        return _result.ResultIdentifier(
            name=self.signal_name,
            animal_id=metadata["animal_id"],
            oxygen_condition=metadata["oxygen_condition"],
            source_file_name=self.file_info.fileName(),
            date_recorded=metadata["date_recorded"],
            result_file_name=result_file_name,
            creation_date=datetime.now(),
        )

    def get_section_identifier(self, section_id: SectionID) -> _t.SectionIdentifier:
        return self.data.sig_data.get_section(section_id).get_section_info()

    def get_file_metadata(self) -> _t.FileMetadata:
        date_recorded = cast(datetime, self.date_edit_file_info.date().toPython())
        animal_id = self.line_edit_subject_id.text()
        oxygen_condition = cast(_t.OxygenCondition, self.combo_box_oxygen_condition.value())
        return _t.FileMetadata(
            date_recorded=date_recorded,
            animal_id=animal_id,
            oxygen_condition=oxygen_condition,
        )

    def get_processing_parameters(self, section_id: SectionID) -> _t.ProcessingParameters:
        return self.data.sig_data.get_section(section_id).processing_parameters

    def get_filter_values(self) -> _t.SignalFilterParameters:
        method = self.filter_method

        filter_params = _t.SignalFilterParameters(
            lowcut=None,
            highcut=None,
            method=method,
            order=2,
            window_size="default",
            powerline=50,
        )
        if filter_params["method"] != "None":
            filter_widgets = {
                "lowcut": self.dbl_spin_box_lowcut,
                "highcut": self.dbl_spin_box_highcut,
                "order": self.spin_box_order,
                "window_size": self.spin_box_window_size,
                "powerline": self.dbl_spin_box_powerline,
            }
            for param, widget in filter_widgets.items():
                if widget.isEnabled():
                    filter_params[param] = widget.value()

        return filter_params

    def get_standardization_values(self) -> _t.StandardizeParameters:
        method = self.scale_method
        robust = method.lower() == "mad"
        if self.container_scale_window_inputs.isChecked():
            window_size = self.spin_box_scale_window_size.value()
        else:
            window_size = None
        return _t.StandardizeParameters(robust=robust, window_size=window_size)

    def get_peak_detection_values(self) -> _t.PeakDetectionParameters:
        method = self.peak_detection_method
        start_index = self.data.active_section.sect_start
        stop_index = self.data.active_section.sect_stop

        if method == "elgendi_ppg":
            vals = _t.PeakDetectionElgendiPPG(
                peakwindow=self.peak_elgendi_ppg_peakwindow.value(),
                beatwindow=self.peak_elgendi_ppg_beatwindow.value(),
                beatoffset=self.peak_elgendi_ppg_beatoffset.value(),
                mindelay=self.peak_elgendi_ppg_min_delay.value(),
            )
        elif method == "local":
            vals = _t.PeakDetectionLocalMaxima(radius=self.peak_local_max_radius.value())
        elif method == "neurokit2":
            vals = _t.PeakDetectionNeurokit2(
                smoothwindow=self.peak_neurokit2_smoothwindow.value(),
                avgwindow=self.peak_neurokit2_avgwindow.value(),
                gradthreshweight=self.peak_neurokit2_gradthreshweight.value(),
                minlenweight=self.peak_neurokit2_minlenweight.value(),
                mindelay=self.peak_neurokit2_mindelay.value(),
                correct_artifacts=self.peak_neurokit2_correct_artifacts.isChecked(),
            )
        elif method == "promac":
            vals = _t.PeakDetectionProMAC(
                threshold=self.peak_promac_threshold.value(),
                gaussian_sd=self.peak_promac_gaussian_sd.value(),
                correct_artifacts=self.peak_promac_correct_artifacts.isChecked(),
            )
        elif method == "wfdb_xqrs":
            vals = _t.PeakDetectionXQRS(
                search_radius=self.peak_xqrs_search_radius.value(),
                peak_dir=self.xqrs_peak_direction,
            )
        elif method == "pantompkins":
            vals = _t.PeakDetectionPantompkins(
                correct_artifacts=self.peak_pantompkins_correct_artifacts.isChecked(),
            )
        else:
            raise ValueError(f"Unknown peak detection method: {method}")

        for key, val in vals.items():
            if isinstance(val, float):
                vals[key] = np.round(val, 2)

        return _t.PeakDetectionParameters(
            start_index=start_index,
            stop_index=stop_index,
            method=method,
            input_values=vals,
        )

    def get_processing_info(self) -> _result.ProcessingParameters:
        filter_params = self.get_filter_values()
        standardization_params = self.get_standardization_values()
        peak_detection_params = self.get_peak_detection_values()
        return _result.ProcessingParameters(
            sampling_rate=self.data.fs,
            pipeline=self.pipeline,
            filter_parameters=filter_params,
            scaling_parameters=standardization_params,
            peak_detection_parameters=peak_detection_params,
        )

    def get_descriptive_stats(self, result_name: str) -> _result.SummaryStatistics:
        return self.data.get_descriptive_stats(result_name)

    def make_results(self) -> _result.Result:
        name = self.signal_name
        btn = self.btn_compute_results
        btn.processing("Working...")
        with pg.BusyCursor():
            focused_result_df = self.data.get_focused_result_df(name)
            results = self._assemble_result_data(name)
        btn.success("Done")

        result_table = self.table_view_focused_result
        helper_result_table = table_view_helper.TableViewHelper(result_table)
        helper_result_table.make_table(_data.PolarsModel(focused_result_df))
        self.tabs_main.setCurrentIndex(2)
        self.result = results
        return results

    # FIXME: needs to be updated for new data structure
    def _assemble_result_data(self, result_name: str) -> _result.Result:
        result_name = self.signal_name
        identifier = self.get_identifier()
        section_infos = {}
        for section_id, section in self.data.sig_data.sections.get_included().items():
            section_identifier = section.get_section_info()
            section_infos[section_id] = section_identifier

        processing_info = self.get_processing_info()

        focused_result = self.data.focused_results[result_name]
        statistics = self.data.get_descriptive_stats(result_name)
        manual_edits = self.plot.get_manual_edits()
        source_data = self.data.sig_data
        return _result.Result(
            identifier=identifier,
            selection_parameters=data_info,
            processing_parameters=processing_info,
            summary_statistics=statistics,
            focused_result=focused_result,
            manual_peak_edits=manual_edits,
            source_data=source_data,
        )

    @Slot()
    def save_state(self) -> None:
        stopped_at_index, ok = QInputDialog.getInt(
            self,
            "Save State",
            "Data is clean up to (and including) index:",
            0,
            0,
            self.data.sig_data.active_section.data.height - 1,
            1,
        )
        if not ok:
            return
        if file_path := QFileDialog.getSaveFileName(
            self,
            "Save State",
            f"{self.output_dir.as_posix()}/snapshot_at_{stopped_at_index}_{self.file_info.completeBaseName()}",
            "Pickle Files (*.pkl)",
        )[0]:
            state_dict = _t.StateDict(
                active_signal=self.signal_name,
                source_file_path=self.file_info.filePath(),
                output_dir=self.output_dir.as_posix(),
                data_selection_params=self.get_section_identifier(),
                data_processing_params=self.get_processing_info(),
                file_metadata=self.get_file_metadata(),
                sampling_frequency=self.data.fs,
                peak_edits=self.plot.peak_edits,  # FIXME: update to new data structure
                data_state=self.data.get_state(),
                stopped_at_index=stopped_at_index,
            )

            with open(file_path, "wb") as f:
                pickle.dump(state_dict, f)
        else:
            msg = "Action 'Save state' cancelled by user."
            self.sig_show_message.emit(msg, "warning")

    @Slot()
    def restore_state(self, file_path: str | Path | None = None) -> None:
        if not file_path:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Restore State",
                self.output_dir.as_posix(),
                "Pickle Files (*.pkl);;_result.Result Files (*.hdf5 *.h5);;All Files (*.pkl *.hdf5 *.h5)",
            )
        if not file_path:
            return
        if Path(file_path).suffix == ".pkl":
            try:
                self.restore_from_pickle(file_path)
            except Exception as e:
                msg = f"Failed to restore state: {e}"
                self.sig_show_message.emit(msg, "warning")
        elif Path(file_path).suffix in [".hdf5", ".h5"]:
            msg = "Restore not yet implemented for HDF5 files."
            self.sig_show_message.emit(msg, "info")
            # TODO: implement way to load the result files for viewing
            # try:
            #     self.restore_from_hdf5(file_path)
            # except Exception as e:
            #     msg = f"Failed to restore state: {e}"
            #     self.sig_show_message.emit(msg, "warning")

    def restore_input_values(
        self,
        selection_params: _result.SelectionParameters,
        processing_params: _result.ProcessingParameters,
        file_metadata: _t.FileMetadata,
    ) -> None:
        self.combo_box_filter_column.setCurrentText(selection_params.filter_column or "")
        self.dbl_spin_box_subset_min.setValue(selection_params.lower_bound)
        self.dbl_spin_box_subset_max.setValue(selection_params.upper_bound)

        self.spin_box_fs.setValue(processing_params.sampling_rate)

        self.combo_box_preprocess_pipeline.setValue(processing_params.pipeline)

        self.dbl_spin_box_lowcut.setValue(processing_params.filter_parameters["lowcut"] or 0.5)
        self.dbl_spin_box_highcut.setValue(processing_params.filter_parameters["highcut"] or 8.0)
        self.spin_box_order.setValue(processing_params.filter_parameters["order"])
        if processing_params.filter_parameters["window_size"] == "default":
            filter_window = int(np.round(processing_params.sampling_rate / 3))
            if filter_window % 2 == 0:
                filter_window += 1
        else:
            filter_window = processing_params.filter_parameters["window_size"]
        self.spin_box_window_size.setValue(filter_window)
        self.dbl_spin_box_powerline.setValue(processing_params.filter_parameters["powerline"])

        method = "mad" if processing_params.scaling_parameters["robust"] else "zscore"
        self.combo_box_scale_method.setValue(method)

        if processing_params.scaling_parameters["window_size"] is None:
            self.container_scale_window_inputs.setChecked(False)
            value = self.spin_box_scale_window_size.minimum()
        else:
            self.container_scale_window_inputs.setChecked(True)
            value = processing_params.scaling_parameters["window_size"]
        self.spin_box_scale_window_size.setValue(value)

        self.combo_box_peak_detection_method.setValue(
            processing_params.peak_detection_parameters["method"]
        )

        # FIXME: update restoration of peak detection parameters to work with new UI

        # self.stacked_peak_parameters.setCurrentIndex(self.combo_box_peak_detection_method.currentIndex())
        # for name, value in processing_params.peak_detection_parameters.get(
        #     "input_values", {}
        # ).items():
        #     self.ui

        self.date_edit_file_info.setDate(
            QDate(
                file_metadata["date_recorded"].year,
                file_metadata["date_recorded"].month,
                file_metadata["date_recorded"].day,
            )
        )
        self.line_edit_subject_id.setText(file_metadata["animal_id"])
        self.combo_box_oxygen_condition.setValue(file_metadata["oxygen_condition"])

        self.btn_apply_filter.setEnabled(True)
        self.btn_detect_peaks.setEnabled(True)
        self.btn_compute_results.setEnabled(True)

    def restore_from_pickle(self, file_path: str | Path) -> None:
        with open(file_path, "rb") as f:
            state: _t.StateDict = pickle.load(f)

        self.data.restore_state(state["working_data"])
        # self.plot.restore_state(state["peak_edits"])
        # self.plot.last_edit_index = state["stopped_at_index"]

        self.spin_box_fs.setValue(state["sampling_frequency"])
        self.file_info.setFile(state["source_file_path"])

        self.output_dir = Path(state["output_dir"])
        self.line_edit_output_dir.setText(state["output_dir"])
        self.data_dir = Path(state["source_file_path"]).parent

        self.ui.update_data_select_ui(path=state["source_file_path"])
        self.line_edit_active_file.setText(self.file_info.fileName())

        self.restore_input_values(
            state["data_selection_params"],
            state["data_processing_params"],
            state["file_metadata"],
        )

        self.stopped_at_index = state["stopped_at_index"]

        self.sig_data_restored.emit()

    @Slot()
    def refresh_app_state(self) -> None:
        self.handle_table_view_data()
        for s_name in self.data.sigs:
            processed_name = self.data.sigs[s_name].processed_name
            if processed_name not in self.data.sigs[s_name].data.columns:
                self.plot.draw_signal(
                    self.data.sigs[s_name].active_section.data.get_column(s_name).to_numpy(),
                    s_name,
                )
                continue
            processed_signal = self.data.sigs[s_name].get_active_signal()
            rate = self.data.sigs[s_name].interpolated_rate
            peaks = self.data.sigs[s_name].get_all_peaks()
            peaks_y = processed_signal[peaks]
            self.plot.draw_signal(processed_signal, s_name)
            self.plot.draw_peaks(peaks, peaks_y, s_name)
            self.plot.draw_rate(rate, s_name)
            self._make_table(
                getattr(self, f"table_view_results_{s_name}"),
                _data.PolarsModel(self.data.focused_results[s_name].to_polars()),
            )


def main(dev_mode: bool = False, antialias: bool = False) -> None:
    if dev_mode:
        os.environ["QT_LOGGING_RULES"] = "qt.pyside.libpyside.warning=true"
    os.environ[
        "LOGURU_FORMAT"
    ] = "<magenta>{time:YYYY-MM-DD HH:mm:ss.SSS}</magenta> | <level>{level: <8}</level> | <yellow>{message}</yellow> | <blue>{name}</blue>.<cyan>{function}()</cyan>, l: <green>{line}</green>\n\n <red>{exception.type}: {exception.value}</red>\n\n{exception.traceback}"

    pg.setConfigOptions(
        useOpenGL=True,
        enableExperimental=True,
        segmentedLineMode="on",
        background="k",
        antialias=antialias,
    )
    logger.add(
        "./logs/debug.log",
        format=(
            "{time:YYYY-MM-DD at HH:mm:ss.SSS} | [{level}]: {message} | module: {name} in {function}, line: {line}"
        ),
        level="DEBUG",
    )
    app = QApplication(sys.argv)
    window = SignalEditor()
    window.show()
    sys.exit(app.exec())
