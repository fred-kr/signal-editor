import cProfile
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
import typing as t

import h5py
import numpy as np
import pyqtgraph as pg
import polars as pl
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
from .handlers.helpers.table_view_helper import TableViewHelper
from .handlers.plot_handler import PlotHandler
from .handlers.style_handler import ThemeSwitcher
from .handlers.ui_handler import UIHandler
from .models import data as _data
from .models import io as _io
from .models import result as _result
from .models.section import SectionID
from .views.main_window import Ui_MainWindow


class SignalEditor(QMainWindow, Ui_MainWindow):
    sig_data_loaded = Signal()
    sig_data_processed = Signal()
    sig_peaks_detected = Signal()
    sig_show_message = Signal(str, str)
    sig_data_restored = Signal()
    sig_section_added = Signal(int, int)
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
        if os.environ.get("DEV_MODE", "0") == "1":
            self._add_profiler()
        self.data.set_sfreq(int(self.config.config.get("Defaults", "SampleRate", fallback=-1)))
        self._setup_section_widgets()

    # region Dev
    def _add_profiler(self) -> None:
        self.menubar.addAction("Start Profiler", self._start_profiler)
        self.menubar.addAction("Stop Profiler", self._stop_profiler)

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

    # endregion

    # region Properties

    @property
    def sig_name(self) -> str:
        return self.combo_box_signal_column.currentText()

    @property
    def proc_sig_name(self) -> str:
        return f"{self.sig_name}_processed"

    @property
    def scale_method(self) -> _t.ScaleMethod:
        return t.cast(_t.ScaleMethod, self.combo_box_scale_method.value())

    @property
    def pipeline(self) -> _t.Pipeline:
        return t.cast(_t.Pipeline, self.combo_box_preprocess_pipeline.value())

    @property
    def filter_method(self) -> _t.FilterMethod:
        return t.cast(_t.FilterMethod, self.combo_box_filter_method.value())

    @property
    def peak_detection_method(self) -> _t.PeakDetectionMethod:
        return t.cast(_t.PeakDetectionMethod, self.combo_box_peak_detection_method.value())

    @property
    def xqrs_peak_direction(self) -> _t.WFDBPeakDirection:
        return t.cast(_t.WFDBPeakDirection, self.peak_xqrs_peak_dir.value())

    # endregion

    def _add_style_toggle(self) -> None:
        self.menubar.addSeparator()
        self.menubar.addAction("Switch Theme", self.theme_switcher.switch_theme)

    def _setup_section_widgets(self) -> None:
        self.combo_box_section_select.blockSignals(True)
        self.combo_box_section_select.addItem(f"SEC_{self.sig_name}_000")
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

        # Plotting Related Actions
        self.action_remove_peak_rect.triggered.connect(self.plot.show_selection_rect)
        self.action_remove_selected_peaks.triggered.connect(self.plot.remove_selected_scatter)
        self.action_reset_view.triggered.connect(self._emit_data_range_info)
        self.sig_update_view_range.connect(self.plot.reset_view_range)
        self.plot.sig_peaks_edited.connect(self.handle_scatter_clicked)

        # Button Actions
        self.btn_apply_filter.clicked.connect(self.process_signal)
        self.btn_browse_output_dir.clicked.connect(self.select_output_location)
        self.btn_compute_results.clicked.connect(self.update_results)
        self.btn_detect_peaks.clicked.connect(self.detect_peaks)
        self.btn_load_selection.clicked.connect(self.handle_load_data)
        self.btn_save_to_hdf5.clicked.connect(self.save_to_hdf5)
        self.btn_select_file.clicked.connect(self.select_data_file)

        # Data Export Actions
        self.btn_export_focused.clicked.connect(self.export_focused_result)

        # Data Related Signals
        self.sig_data_processed.connect(self.handle_table_view_data)
        self.sig_data_loaded.connect(lambda: self.tabs_main.setCurrentIndex(1))
        self.sig_data_loaded.connect(self.handle_draw_signal)
        self.sig_data_loaded.connect(self.handle_table_view_data)
        self.sig_data_restored.connect(self.refresh_app_state)
        
        self.sig_peaks_detected.connect(self.handle_draw_rate)
        self.sig_peaks_detected.connect(self.handle_draw_peaks)
        
        self.sig_section_added.connect(self.plot.mark_section)
        self.data.sig_cas_changed.connect(self.handle_draw_signal)
        self.data.sig_new_raw.connect(self.ui.update_data_select_ui)

        # UI Handler Signals
        self.ui.sig_section_confirmed.connect(self._on_section_confirmed)
        self.ui.sig_section_canceled.connect(self._on_section_canceled)

        # Widget specific Signals
        self.spin_box_sample_rate.valueChanged.connect(self.data.update_sfreq)
        self.combo_box_section_select.currentTextChanged.connect(self.data.set_cas)
        self.date_edit_file_info.dateChanged.connect(self.data.set_date)
        self.line_edit_subject_id.textChanged.connect(self.data.set_animal_id)
        self.combo_box_oxygen_condition.currentTextChanged.connect(self.data.set_oxy_condition)

        self.sig_show_message.connect(self.show_message)
        
    @Slot()
    def _new_included_section(self) -> None:
        current_limits = self.data.bounds
        self.container_section_confirm_cancel.show()
        self.plot.show_section_selector("included", current_limits)

    @Slot()
    def _new_excluded_section(self) -> None:
        current_limits = self.data.bounds
        self.container_section_confirm_cancel.show()
        self.plot.show_section_selector("excluded", current_limits)

    @Slot()
    def _on_section_confirmed(self) -> None:
        lin_reg = self.plot.region_selector
        if lin_reg is None:
            return
        lower, upper = lin_reg.getRegion()
        lower, upper = int(lower), int(upper)
        new_sect = self.data.new_section(lower, upper)
        self.combo_box_section_select.addItem(new_sect.section_id)
        self.combo_box_section_select.setCurrentText(new_sect.section_id)
        self.container_section_confirm_cancel.hide()
        self.sig_section_added.emit(lower, upper)

    @Slot()
    def _on_section_canceled(self) -> None:
        self.container_section_confirm_cancel.hide()
        self.plot.remove_section_selector()

    @Slot()
    def _emit_data_range_info(self) -> None:
        data = self.data.cas_proc_data.to_numpy()
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
        out_location = Path.joinpath(self.config.output_dir, self.config.make_complete_result_name(self.sig_name, self.file_info.completeBaseName()))
        if file_path := QFileDialog.getSaveFileName(
            self,
            "Save to HDF5",
            out_location.as_posix(),
            "HDF5 Files (*.hdf5 *.h5)",
        )[0]:
            _io.write_hdf5(file_path, self.make_results())

    @Slot()
    def update_results(self) -> None:
        if len(self.data.cas.peaks) == 0:
            msg = (
                f"No peaks detected for signal '{self.sig_name}'. Please run peak detection first."
            )
            self.sig_show_message.emit(msg, "info")
            return
        self.make_results()

    @Slot(str)
    def export_focused_result(self, output_format: t.Literal["csv", "xlsx", "txt"]) -> None:
        result_file_name = self.config.make_focused_result_name(
            self.sig_name, self.file_info.completeBaseName()
        )
        result_location = Path.joinpath(self.config.output_dir, result_file_name)

        results = self.data.get_focused_results()
        result_dfs: list[pl.DataFrame] = []
        for section_id, focused_result in results.items():
            fr_df = focused_result.to_polars().with_columns(pl.lit(section_id).alias("section_id"))
            result_dfs.append(fr_df)
        result_df = pl.concat(result_dfs)

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
        self, text: str, level: t.Literal["debug", "info", "warning", "critical", "error"]
    ) -> None:
        icon_map = {
            "debug": QMessageBox.Icon.NoIcon,
            "info": QMessageBox.Icon.Information,
            "warning": QMessageBox.Icon.Warning,
            "critical": QMessageBox.Icon.Critical,
            "error": QMessageBox.Icon.Critical,
        }
        if level == "debug" and os.environ.get("DEV_MODE", "0") == "0":
            return
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
            dir=self.config.output_dir.as_posix(),
            options=QFileDialog.Option.ShowDirsOnly,
        ):
            self.line_edit_output_dir.setText(path)
            self.output_dir = path

    @Slot()
    def handle_table_view_data(self) -> None:
        """
        Update the data preview table and the data info table with the current data.
        """
        data = self.data.cas.data
        n_rows = 10
        df_head = data.head(n_rows)
        df_tail = data.tail(n_rows)
        df_description = data.describe(percentiles=None)

        model = _data.CompactDFModel(df_head=df_head, df_tail=df_tail)
        raw_data_table = TableViewHelper(self.table_data_preview)
        raw_data_table.make_table(model)

        info = _data.DescriptiveStatsModel(df_description)
        info_table = TableViewHelper(self.table_data_info)
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
            self.config.data_dir = self.file_info.dir().path()

    @Slot()
    def handle_load_data(self) -> None:
        self.btn_load_selection.processing("Loading data...")
        self.statusbar.showMessage(f"Loading data from file: {self.file_info.canonicalFilePath()}")
        self.data.read_file(self.file_info.canonicalFilePath())

        signal_col = self.sig_name
        if signal_col not in self.data.raw_df.columns:
            msg = f"Selected column '{signal_col}' not found in data. Detected columns are: '{', '.join(self.data.raw_df.columns)}'"
            self.btn_load_selection.failure("Error", msg)
            self.sig_show_message.emit(msg, "error")
            self.statusbar.showMessage("Selected signal column not found in data.", 3000)
            return
        self.data.create_base_df(signal_col)

        self.sig_data_loaded.emit()
        self.btn_load_selection.success()
        self.statusbar.showMessage("Data loaded.", 3000)

    @Slot()
    def process_signal(self) -> None:
        self.statusbar.showMessage("Filtering data...")
        btn = self.btn_apply_filter
        btn.processing("Working...")
        with pg.BusyCursor():
            filter_params = self.ui.get_filter_parameters()
            standardize_params = self.ui.get_standardize_parameters()

            self.data.cas.filter_data(
                self.pipeline,
                **filter_params,
            )
            if self.scale_method != "None":
                self.data.cas.scale_data(**standardize_params)
                
            self.handle_draw_signal()
        btn.success()
        self.statusbar.showMessage("Filtering finished.", 3000)
        self.sig_data_processed.emit()

    @Slot()
    def detect_peaks(self) -> None:
        self.btn_detect_peaks.processing("Working...")
        self.statusbar.showMessage("Detecting peaks...")
        with pg.BusyCursor():
            peak_params = self.ui.get_peak_detection_parameters()
            self.data.cas.detect_peaks(**peak_params)
        self.sig_peaks_detected.emit()
        self.statusbar.showMessage("Peak detection finished.", 3000)

    @Slot()
    def handle_draw_signal(self) -> None:
        with pg.BusyCursor():
            data = self.data.cas_proc_data.to_numpy()
            self.plot.draw_signal(data, self.sig_name)

    @Slot()
    def handle_draw_peaks(self) -> None:
        with pg.BusyCursor():
            peaks_x, peaks_y = self.data.cas.get_peak_xy()
            self.plot.draw_peaks(peaks_x, peaks_y, self.sig_name)
        self.btn_detect_peaks.success()
            
    @Slot()
    def handle_draw_rate(self) -> None:
        self.plot.draw_rate(self.data.cas.rate_interp, self.sig_name)

    @Slot(str, list[int])
    def handle_scatter_clicked(self, action: t.Literal["add", "remove"], indices: list[int]) -> None:
        scatter_item = self.plot.scatter_item
        if scatter_item is None:
            return
        self.data.cas.update_peaks(action, indices)
        self.sig_peaks_detected.emit()

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
        self.config.sample_rate = self.data.sfreq
        self.config.write_config()

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
                active_signal=self.sig_name,
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
        os.environ["DEV_MODE"] = "1"
    else:
        os.environ["QT_LOGGING_RULES"] = "qt.pyside.libpyside.warning=false"
        os.environ["DEV_MODE"] = "0"

    pg.setConfigOptions(
        useOpenGL=True,
        enableExperimental=True,
        segmentedLineMode="on",
        background="k",
        antialias=antialias,
    )
    logger.add(
        "./logs/debug.log",
        level="DEBUG",
    )
    app = QApplication(sys.argv)
    window = SignalEditor()
    window.show()
    sys.exit(app.exec())
