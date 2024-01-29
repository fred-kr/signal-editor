import cProfile
import os
import pickle
import sys
import typing as t
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import polars as pl
import pyqtgraph as pg
from loguru import logger
from PySide6 import QtGui, QtWidgets, QtCore
from PySide6.QtCore import (
    QByteArray,
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
from .handlers.helpers.table_helper import TableHelper
from .handlers.plot_handler import PlotHandler
from .handlers.style_handler import ThemeSwitcher
from .handlers.ui_handler import UIHandler
from .models.io import write_hdf5
from .models.polars_df import DescriptiveStatsModel, PolarsTableModel
from .models.result import CompleteResult
from .models.section import SectionID, SectionIndices
from .views.main_window import Ui_MainWindow


def readable_section_id(section_id: SectionID) -> str:
    number = int(section_id[-3:])
    sig_name = section_id.split("_")[1]
    return f"Active Data: Signal={sig_name.upper()}, Section={number:03d}"


class SignalEditor(QMainWindow, Ui_MainWindow):
    sig_show_message = Signal(str, str)
    sig_data_loaded = Signal()
    sig_data_processed = Signal()
    sig_peaks_detected = Signal()
    sig_data_restored = Signal()
    sig_section_confirmed = Signal(int, int)
    sig_update_view_range = Signal(int)

    def __init__(self) -> None:
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("Signal Editor")
        self.unsaved_changes: bool = False
        self.config = ConfigHandler("config.ini")
        self.theme_switcher = ThemeSwitcher()
        self.plot = PlotHandler(self)
        self.data = DataHandler(self)
        self._read_settings()
        self.ui = UIHandler(self, self.plot)
        self.file_info: QFileInfo = QFileInfo()
        self._connect_signals()
        self._on_init_finished()

    def _connect_signals(self) -> None:
        """
        Connect signals to slots.
        """
        # General application actions
        self.action_exit.triggered.connect(self.close)
        self.action_load_state.triggered.connect(self.restore_state)
        self.action_save_state.triggered.connect(self.save_state)
        self.action_select_file.triggered.connect(self.select_data_file)
        self.action_light_switch.toggled.connect(self.theme_switcher.switch_theme)
        self.action_open_config_file.triggered.connect(self.open_config_file)
        self.sig_show_message.connect(self.show_message)

        # Sections
        self.data.sig_section_added.connect(self.add_section_to_widget)
        self.data.sig_section_removed.connect(self.remove_section_from_widget)
        self.data.sig_cas_changed.connect(self.handle_draw_signal)
        self.list_widget_sections.currentRowChanged.connect(self._on_active_section_changed)
        self.combo_box_section_select.currentIndexChanged.connect(self._on_active_section_changed)
        self.action_mark_section_finished.triggered.connect(self._on_section_done)
        self.action_reset_all.triggered.connect(self._on_sections_cleared)
        self.action_section_overview.toggled.connect(self.plot.toggle_region_overview)
        self.sig_section_confirmed.connect(self.plot.mark_section)
        self.btn_section_confirm.clicked.connect(self._on_section_confirmed)
        self.btn_section_cancel.clicked.connect(self._on_section_canceled)
        self.action_confirm.triggered.connect(self._on_section_confirmed)
        self.action_cancel.triggered.connect(self._on_section_canceled)
        self.btn_section_remove.clicked.connect(self._remove_section)
        self.action_remove_section.triggered.connect(self._remove_section)
        self.action_add_section.triggered.connect(self._maybe_new_included_section)

        # File I/O
        self.btn_browse_output_dir.clicked.connect(self.select_output_location)
        self.btn_save_to_hdf5.clicked.connect(self.save_to_hdf5)
        self.btn_select_file.clicked.connect(self.select_data_file)
        self.action_save_to_hdf5.triggered.connect(self.save_to_hdf5)
        self.btn_export_focused.clicked.connect(self.export_focused_result)

        # Data Handling
        self.data.sig_new_raw.connect(self.ui.update_data_select_ui)
        self.btn_load_selection.clicked.connect(self.handle_load_data)
        self.sig_data_loaded.connect(self._on_data_loaded)
        self.sig_data_restored.connect(self.refresh_app)
        self.btn_compute_results.clicked.connect(self.update_results)

        # Processing & Editing
        self.btn_apply_filter.clicked.connect(self.process_signal)
        self.btn_detect_peaks.clicked.connect(self.detect_peaks)
        self.plot.sig_peaks_edited.connect(self.handle_scatter_clicked)
        self.plot.sig_peaks_drawn.connect(self.handle_draw_rate)
        self.sig_data_processed.connect(self.update_cas_table)
        self.sig_peaks_detected.connect(self._on_peaks_detected)
        self.action_remove_peak_rect.triggered.connect(self.plot.show_selection_rect)
        self.action_remove_selected_peaks.triggered.connect(self.plot.remove_selected_scatter)

        # Plot / View actions
        self.action_reset_view.triggered.connect(self._emit_data_range_info)
        self.sig_update_view_range.connect(self.plot.reset_view_range)

        # File information / metadata
        self.spin_box_sample_rate.valueChanged.connect(self.data.update_sfreq)
        self.date_edit_file_info.dateChanged.connect(self.data.set_date)
        self.line_edit_subject_id.textChanged.connect(self.data.set_animal_id)
        self.combo_box_oxygen_condition.currentTextChanged.connect(self.data.set_oxy_condition)

    def _on_init_finished(self) -> None:
        self.line_edit_output_dir.setText(self.config.output_dir.as_posix())
        self.theme_switcher.set_style(self.config.style)
        if os.environ.get("DEV_MODE", "0") == "1":
            self._add_profiler()
        self.action_group_cc = QtGui.QActionGroup(self)
        self.action_group_cc.addAction(self.action_confirm)
        self.action_group_cc.addAction(self.action_cancel)
        self.action_group_cc.setVisible(False)
        add_section_menu = QMenu(self.btn_section_add)
        add_section_menu.addAction("Included", self._maybe_new_included_section)
        add_section_menu.addAction("Excluded", self._maybe_new_excluded_section)
        self.btn_section_add.setMenu(add_section_menu)
        self.data.set_sfreq(self.config.sample_rate)
        self._result: CompleteResult | None = None

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

    # endregion Dev

    # region Properties
    @property
    def sig_name(self) -> str:
        return self.combo_box_signal_column.currentText()

    @property
    def proc_sig_name(self) -> str:
        return f"{self.sig_name}_processed"

    @property
    def section_name(self) -> SectionID:
        return SectionID(self.combo_box_section_select.currentText())

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

    # endregion Properties

    @Slot()
    def open_config_file(self) -> None:
        file = self.config.config_file.as_posix()
        if not os.path.exists(file):
            self.sig_show_message.emit(f"Config file not found: {file}", "warning")
            return
        # Open the config.ini file in the default os editor
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(file))

    # region Sections
    @Slot(str)
    def add_section_to_widget(self, section_id: SectionID) -> None:
        self.list_widget_sections.addItem(section_id)
        self.combo_box_section_select.addItem(section_id)

    @Slot(str)
    def remove_section_from_widget(self, section_id: SectionID) -> None:
        index = self.combo_box_section_select.findText(section_id)
        self.list_widget_sections.takeItem(index)
        self.combo_box_section_select.removeItem(index)
        self.list_widget_sections.setCurrentRow(0)
        self.combo_box_section_select.setCurrentIndex(0)

    def _clear_section_widgets(self) -> None:
        self.list_widget_sections.clear()
        self.combo_box_section_select.clear()

    @Slot()
    def _on_sections_cleared(self) -> None:
        # Only ask if there is any data currently loaded (i.e. dont ask the first time)
        if len(self.data.sections) != 0:
            ok = QMessageBox.question(
                self,
                "Reset everything?",
                "This will remove all sections and set the data back to its original state (as read from the file). Continue?",
            )
            if ok == QMessageBox.StandardButton.No:
                return
        self.list_widget_sections.clear()
        self.combo_box_section_select.clear()
        self.data.clear_sections()
        self.data.create_base_df(self.sig_name)
        self.plot.clear_regions()
        self.update_cas_table()
        self._result = None

    @Slot()
    def _on_section_done(self) -> None:
        self.data.save_cas()
        self.update_focused_result_table()

    @Slot(int)
    def _on_active_section_changed(self, index: int) -> None:
        if index == -1:
            return
        sender = self.sender()
        if isinstance(sender, QtWidgets.QListWidget):
            section_id = SectionID(sender.currentItem().text())
            self.combo_box_section_select.setCurrentIndex(index)
        elif isinstance(sender, QtWidgets.QComboBox):
            section_id = SectionID(sender.currentText())
            self.list_widget_sections.setCurrentRow(index)
        else:
            return
        self.data.set_cas(section_id)
        self.update_cas_table()
        self.ui.label_currently_showing.setText(readable_section_id(section_id))
        if index != 0:
            self.action_add_section.setEnabled(False)
            self.btn_section_add.setEnabled(False)
            self.btn_section_add.setToolTip(
                "Creating a new section is currently only possible when the first section (the one ending in `000`) is selected."
            )
        else:
            self.action_add_section.setEnabled(True)
            self.btn_section_add.setEnabled(True)
            self.btn_section_add.setToolTip("Create a new section.")

    @Slot()
    def _maybe_new_included_section(self) -> None:
        current_limits = self.data.bounds
        self.container_section_confirm_cancel.show()
        self.action_group_cc.setVisible(True)
        self.plot.show_section_selector("included", current_limits)
        self._section_type = "included"

    @Slot()
    def _maybe_new_excluded_section(self) -> None:
        current_limits = self.data.bounds
        self.container_section_confirm_cancel.show()
        self.action_group_cc.setVisible(True)
        self.plot.show_section_selector("excluded", current_limits)
        self._section_type = "excluded"

    @Slot()
    def _on_section_confirmed(self) -> None:
        if not hasattr(self, "_section_type"):
            return
        lin_reg = self.plot.region_selector
        if lin_reg is None:
            return

        lower, upper = lin_reg.getRegion()
        lower, upper = int(lower), int(upper)
        match self._section_type:
            case "included":
                self.data.new_section(lower, upper)
            case "excluded":
                self.data.excluded_sections.append(SectionIndices(lower, upper))
            case _:
                raise ValueError(f"Invalid section type: {self._section_type}")
        self.action_group_cc.setVisible(False)
        self.container_section_confirm_cancel.hide()
        self.btn_section_remove.setEnabled(True)
        self.action_remove_section.setEnabled(True)
        self.sig_section_confirmed.emit(lower, upper)

    @Slot()
    def _on_section_canceled(self) -> None:
        self.action_group_cc.setVisible(False)
        self.container_section_confirm_cancel.hide()
        self.plot.remove_section_selector()

    @Slot()
    def _remove_section(self) -> None:
        excluded_sections = self.data.excluded_sections
        excluded_sect_strings = [str(sect) for sect in excluded_sections]
        combined_items = list(self.data.sections.keys())[1:] + excluded_sect_strings
        if len(combined_items) == 0:
            msg = "No sections to remove."
            self.sig_show_message.emit(msg, "info")
            return
        to_remove, ok = QInputDialog.getItem(
            self,
            "Select Section to Remove",
            "Items that look like `SEC_<signal_name>_<number>` are included, those that look like `<start_index>, <stop_index>` are excluded.",
            [str(item) for item in combined_items],
            0,
            False,
        )
        if ok:
            if to_remove in excluded_sect_strings:
                bounds = [int(float(bound)) for bound in to_remove.split(", ")]
                self.data.excluded_sections.remove(SectionIndices(*bounds))
            else:
                sect_id = SectionID(to_remove)
                bounds = self.data.get_section(sect_id).base_bounds
                self.data.remove_section(sect_id)
            self.plot.remove_region((bounds[0], bounds[1]))
            if len(self.data.combined_section_ids) == 0:
                self.btn_section_remove.setEnabled(False)
                self.action_remove_section.setEnabled(False)

    # endregion Sections

    @Slot()
    def _on_data_loaded(self) -> None:
        if self.config.switch_on_load:
            self.action_toggle_section_sidebar.setChecked(True)
            self.tabs_main.setCurrentIndex(1)
        self.handle_draw_signal()
        self.update_cas_table()

    @Slot()
    def _on_peaks_detected(self) -> None:
        self.handle_draw_peaks()
        self.handle_draw_rate()
        self.update_cas_table()

    @Slot()
    def _emit_data_range_info(self) -> None:
        self.sig_update_view_range.emit(self.data.cas.data.height)

    @Slot(int)
    def update_sfreq_blocked(self, value: int) -> None:
        self.spin_box_sample_rate.blockSignals(True)
        self.spin_box_sample_rate.setValue(value)
        self.spin_box_sample_rate.blockSignals(False)

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
        out_location = Path.joinpath(
            self.config.output_dir,
            self.config.make_complete_result_name(self.sig_name, self.file_info.completeBaseName()),
        )
        if file_path := QFileDialog.getSaveFileName(
            self,
            "Save to HDF5",
            out_location.as_posix(),
            "HDF5 Files (*.hdf5 *.h5)",
        )[0]:
            complete_results = self.data.get_complete_result()
            write_hdf5(file_path, complete_results)

            self.sig_show_message.emit(f"Saved results to {file_path}.", "info")
            self.unsaved_changes = False

    @Slot()
    def update_results(self) -> None:
        if self.data.cas.peaks.len() == 0:
            msg = (
                f"No peaks detected for signal '{self.sig_name}'. Please run peak detection first."
            )
            self.sig_show_message.emit(msg, "info")
            return
        self.statusbar.showMessage("Computing results...")
        self._result = self.data.get_complete_result()
        self.statusbar.showMessage("Results computed.")

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
        msg_box = QMessageBox(icon_map[level], "Notice", text, QMessageBox.StandardButton.Ok)
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
            self.config.output_dir = Path(path)

    @Slot()
    def update_cas_table(self) -> None:
        """
        Update the data preview table and the data info table with the current data.
        """
        cas_df = self.data.cas.data
        cas_desc_df = cas_df.describe()

        if not hasattr(self, "cas_table"):
            model = PolarsTableModel(cas_df)
            self.cas_table = TableHelper(self.table_view_cas, model)
        else:
            self.cas_table.update_cas_df(cas_df)

        if not hasattr(self, "cas_desc_table"):
            info = DescriptiveStatsModel(cas_desc_df, orig_dtypes=self.data.cas.data.dtypes)
            self.cas_desc_table = TableHelper(self.table_view_cas_description, info)
        else:
            self.cas_desc_table.update_cas_desc_df(cas_desc_df, cas_df.dtypes)

    @Slot()
    def update_focused_result_table(self) -> None:
        try:
            if self._result is None:
                res_df = pl.DataFrame()
            else:
                res_df = self.data.cas.get_focused_result().to_polars()
            if not hasattr(self, "cas_focused_table"):
                self.cas_focused_table = TableHelper(
                    self.table_view_focused_result, PolarsTableModel(res_df)
                )
            else:
                self.cas_focused_table.update_cas_df(res_df)
        except Exception as e:
            msg = f"Failed to create focused result table: {e}"
            self.sig_show_message.emit(msg, "error")

    @Slot()
    def select_data_file(self) -> None:
        if self.unsaved_changes:
            msg = "You have unsaved changes. Are you sure you want to continue?"
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                msg,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                return
        path, _ = QFileDialog.getOpenFileName(
            self,
            caption="Select File",
            dir=self.config.data_dir.as_posix(),
            filter="EDF (*.edf);;CSV (*.csv);;TXT (*.txt);;Feather (*.feather);;State Files (*.pkl);;All Files (*.edf *.csv *.txt *.feather *.pkl)",
            selectedFilter="All Files (*.edf *.csv *.txt *.feather *.pkl)",
        )
        if path:
            self.ui.reset_widget_state()
            self.plot.reset_plots()
            self.data.reset()
            self.file_info.setFile(path)
            self.line_edit_active_file.setText(Path(path).name)
            self.config.data_dir = self.file_info.dir().path()
            self.data.read_file(path)
            self._clear_section_widgets()

    @Slot()
    def handle_load_data(self) -> None:
        self.btn_load_selection.processing("Loading data...")
        self.statusbar.showMessage(f"Loading data from file: {self.file_info.canonicalFilePath()}")

        signal_col = self.sig_name
        if signal_col not in self.data.raw_df.columns:
            msg = f"Selected column '{signal_col}' not found in data. Detected columns are: '{', '.join(self.data.raw_df.columns)}'"
            self.btn_load_selection.failure("Error", msg)
            self.sig_show_message.emit(msg, "error")
            self.statusbar.showMessage("Selected signal column not found in data.")
            return
        self._on_sections_cleared()

        self.sig_data_loaded.emit()
        self.btn_load_selection.success()
        self.statusbar.showMessage("Data loaded.")

    @Slot()
    def process_signal(self) -> None:
        self.statusbar.showMessage("Filtering data...")
        with pg.BusyCursor():
            filter_params = self.ui.get_filter_parameters()
            standardize_params = self.ui.get_standardize_parameters()

            if (
                self.pipeline == "custom"
                and self.filter_method == "None"
                and self.scale_method == "None"
            ):
                self.data.cas.data = self.data.cas.data.with_columns(
                    pl.col(self.sig_name).alias(self.proc_sig_name),
                    pl.repeat(False, self.data.cas.data.height).alias("is_peak"),
                )
            else:
                self.data.cas.filter_data(
                    self.pipeline,
                    **filter_params,
                )
                if self.scale_method != "None":
                    self.data.cas.scale_data(**standardize_params)

            self.handle_draw_signal()

        self.sig_data_processed.emit()
        self.unsaved_changes = True
        self.btn_apply_filter.success(limitedTime=False)
        self.statusbar.showMessage("Filtering finished.")
        self.btn_apply_filter.reset()

    @Slot()
    def detect_peaks(self) -> None:
        self.statusbar.showMessage("Detecting peaks...")
        self.btn_detect_peaks.processing("Working...")
        peak_params = self.ui.get_peak_detection_parameters()
        if peak_params["method"] == "wfdb_xqrs":
            with pg.BusyCursor():
                self.data.cas.detect_peaks(**peak_params)
        else:
            self.data.cas.detect_peaks(**peak_params)
        self.sig_peaks_detected.emit()
        self.unsaved_changes = True
        self.btn_detect_peaks.success(limitedTime=False)
        self.statusbar.showMessage(
            f"Peak detection finished. Found {self.data.cas.peaks.len()} peaks."
        )
        self.btn_detect_peaks.reset()

    @Slot(bool)
    def handle_draw_signal(self, has_peaks: bool = False) -> None:
        self.plot.draw_signal(self.data.cas.proc_data.to_numpy(), self.sig_name)
        if has_peaks:
            self.handle_draw_peaks()

        for region in self.plot.combined_regions:
            if self.section_name.endswith("000") and self.action_section_overview.isChecked():
                if region not in self.plot.main_plot_widget.getPlotItem().items:
                    self.plot.main_plot_widget.addItem(region)
                region.show()
            else:
                region.hide()

    @Slot()
    def handle_draw_peaks(self) -> None:
        peaks_x, peaks_y = self.data.cas.get_peak_xy()
        self.plot.draw_peaks(peaks_x, peaks_y, self.sig_name)

    @Slot()
    def handle_draw_rate(self) -> None:
        self.data.cas.calculate_rate()
        self.plot.draw_rate(self.data.cas.rate_interp, self.sig_name)

    @Slot(str, list)
    def handle_scatter_clicked(
        self, action: t.Literal["add", "remove"], indices: list[int]
    ) -> None:
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
        if os.environ.get("DEV_MODE", "0") == "1" and self.ui.jupyter_console_dock.isVisible():
            self.ui.jupyter_console_dock.close()

        super().closeEvent(event)

    def _read_settings(self) -> None:
        settings = QSettings("AWI", "Signal Editor")
        geometry: QByteArray = t.cast(QByteArray, settings.value("geometry", QByteArray(), type=QByteArray))

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
        snapshot_name = self.config.make_app_state_name(datetime.now())
        if file_path := QFileDialog.getSaveFileName(
            self,
            caption="Save State",
            dir=Path.joinpath(self.config.output_dir, snapshot_name).as_posix(),
            filter="Pickle Files (*.pkl)",
        )[0]:
            state_dict = _t.StateDict(
                signal_name=self.sig_name,
                source_file_path=self.file_info.filePath(),
                output_dir=self.config.output_dir.as_posix(),
                data_state=self.data.get_state(),
            )

            with open(file_path, "wb") as f:
                pickle.dump(state_dict, f)
        else:
            msg = "Action 'Save state' cancelled by user."
            self.sig_show_message.emit(msg, "warning")

    @Slot(bool)
    def restore_state(self, checked: bool = False, file_path: str | Path | None = None) -> None:
        if not file_path:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                caption="Restore State",
                dir=self.config.output_dir.as_posix(),
                filter="Pickle Files (*.pkl);;Result Files (*.hdf5 *.h5);;All Files (*.pkl *.hdf5 *.h5)",
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

    def restore_processing_parameters(
        self,
        processing_params: _t.ProcessingParameters,
    ) -> None:
        self.spin_box_sample_rate.setValue(processing_params["sampling_rate"])
        self.combo_box_preprocess_pipeline.setValue(processing_params["pipeline"])

        filt_params = processing_params["filter_parameters"]
        if filt_params is not None:
            lowcut = filt_params.get("lowcut") or 0.5
            highcut = filt_params.get("highcut") or 8.0
            order = filt_params.get("order") or 2
            window_size = filt_params.get("window_size") or "default"
            powerline = filt_params.get("powerline") or 50.0
        else:
            lowcut, highcut, order, window_size, powerline = 0.5, 8.0, 2, "default", 50.0
        self.dbl_spin_box_lowcut.setValue(lowcut)
        self.dbl_spin_box_highcut.setValue(highcut)
        self.spin_box_order.setValue(order)
        if window_size == "default":
            filter_window = int(np.round(processing_params["sampling_rate"] / 3))
            if filter_window % 2 == 0:
                filter_window += 1
        else:
            filter_window = window_size
        self.spin_box_window_size.setValue(filter_window)
        self.dbl_spin_box_powerline.setValue(powerline)

        standardize_params = processing_params["standardize_parameters"]
        if standardize_params is not None:
            robust = standardize_params.get("robust") or False
            scale_window_size = standardize_params.get("window_size") or None
            method = "mad" if robust else "zscore"
        else:
            robust, scale_window_size = False, None
            method = "None"
        self.combo_box_scale_method.setValue(method)
        if scale_window_size is None:
            self.container_scale_window_inputs.setChecked(False)
            value = self.spin_box_scale_window_size.minimum()
        else:
            self.container_scale_window_inputs.setChecked(True)
            value = scale_window_size
        self.spin_box_scale_window_size.setValue(value)

        peak_detection_params = processing_params["peak_detection_parameters"]
        if peak_detection_params is not None:
            peak_method = peak_detection_params.get("method") or "elgendi_ppg"
            input_values = peak_detection_params.get("input_values") or _t.PeakDetectionElgendiPPG(
                peakwindow=0.111,
                beatwindow=0.667,
                beatoffset=0.02,
                mindelay=0.3,
            )
        else:
            peak_method = "elgendi_ppg"
            input_values = _t.PeakDetectionElgendiPPG(
                peakwindow=0.111,
                beatwindow=0.667,
                beatoffset=0.02,
                mindelay=0.3,
            )

        params = _t.PeakDetectionParameters(method=peak_method, input_values=input_values)
        self.ui.set_peak_detection_parameters(params)

    def restore_from_pickle(self, file_path: str | Path) -> None:
        with open(file_path, "rb") as f:
            state: _t.StateDict = pickle.load(f)

        self.data.restore_state(state["data_state"])

        self.file_info.setFile(state["source_file_path"])

        self.config.output_dir = Path(state["output_dir"])
        self.line_edit_output_dir.setText(state["output_dir"])
        self.config.data_dir = Path(state["source_file_path"]).parent

        self.ui.update_data_select_ui()
        self.line_edit_active_file.setText(self.file_info.fileName())

        self.restore_processing_parameters(self.data.cas.processing_parameters)

        self.sig_data_restored.emit()

    @Slot()
    def refresh_app(self) -> None:
        # TODO: implement
        ...


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
    # app.setAttribute(QtGui.Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    window = SignalEditor()
    window.show()
    sys.exit(app.exec())
