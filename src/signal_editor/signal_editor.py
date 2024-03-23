import cProfile
import os
import sys
import traceback
import typing as t
from datetime import datetime
from pathlib import Path

import matplotlib.colors as mcolors
import numpy as np
import numpy.typing as npt
import polars as pl
import pyqtgraph as pg
from loguru import logger
from PySide6 import QtCore, QtGui, QtWidgets

from . import type_aliases as _t
from .fileio import result_dict_to_hdf5
from .handlers import (
    ConfigHandler,
    DataHandler,
    PlotHandler,
    StyleHandler,
    TableHandler,
    UIHandler,
)
from .models import CompleteResult, SectionID
from .peaks import find_extrema, find_peaks
from .processing import rolling_rate
from .views.main_window import Ui_MainWindow

if t.TYPE_CHECKING:
    from xlsxwriter import Workbook

    from .views import CustomScatterPlotItem


def readable_section_id(section_id: SectionID, subject_id: str, oxygen_condition: str) -> str:
    return f"Active: '{subject_id} - {section_id}'; oxygen: '{oxygen_condition}'"


class SignalEditor(QtWidgets.QMainWindow, Ui_MainWindow):
    sig_show_message = QtCore.Signal(str, str)
    sig_data_loaded = QtCore.Signal()
    sig_data_processed = QtCore.Signal()
    sig_peaks_detected = QtCore.Signal()
    sig_section_confirmed = QtCore.Signal(int, int)
    sig_update_view_range = QtCore.Signal(int)

    def __init__(self) -> None:
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("Signal Editor")
        self._read_settings()
        self.unsaved_changes: bool = False
        self.config = ConfigHandler("config.ini")
        self.theme = StyleHandler()
        self.plot = PlotHandler(self)
        self.data = DataHandler(self)
        self.ui = UIHandler(self, self.plot)
        self.file_info: QtCore.QFileInfo = QtCore.QFileInfo()
        self._read_config()
        self._connect_qt_signals()
        self._result: CompleteResult | None = None
        self.tables = TableHandler()
        self._on_init_finished()

    def _connect_qt_signals(self) -> None:
        """
        Connect signals to slots.
        """
        # General application actions
        self.action_exit.triggered.connect(self.close)
        self.action_select_file.triggered.connect(self.select_data_file)
        self.action_light_switch.toggled.connect(self.theme.switch_theme)
        self.action_open_config_file.triggered.connect(self.open_config_file)
        self.sig_show_message.connect(self.show_message)

        # Sections
        self.data.sig_section_added.connect(self.add_section_to_widget)
        self.data.sig_section_removed.connect(self.remove_section_from_widget)
        self.data.sig_cas_changed.connect(self.handle_draw_signal)
        self.list_widget_sections.currentRowChanged.connect(self._on_active_section_changed)
        self.combo_box_section_select.currentIndexChanged.connect(self._on_active_section_changed)
        self.action_reset_all.triggered.connect(self._on_sections_cleared)
        self.action_section_overview.toggled.connect(self.plot.toggle_region_overview)
        self.sig_section_confirmed.connect(self.plot.mark_section)
        self.btn_section_confirm.clicked.connect(self._on_section_confirmed)
        self.btn_section_cancel.clicked.connect(self._on_section_canceled)
        self.action_confirm.triggered.connect(self._on_section_confirmed)
        self.action_cancel.triggered.connect(self._on_section_canceled)
        self.btn_section_add.clicked.connect(self._maybe_new_section)
        self.btn_section_remove.clicked.connect(self._remove_section)
        self.action_remove_section.triggered.connect(self._remove_section)
        self.action_add_section.triggered.connect(self._maybe_new_section)

        # File I/O
        self.btn_save_to_hdf5.clicked.connect(self.save_to_hdf5)
        self.btn_select_file.clicked.connect(self.select_data_file)
        self.action_save_to_hdf5.triggered.connect(self.save_to_hdf5)
        self.btn_load_focused_result.clicked.connect(self._load_focused_result)

        # Data Handling
        self.btn_show_more_rows.clicked.connect(self._update_cas_table)
        self.data.sig_new_raw.connect(self.ui.update_data_select_ui)
        self.data.sig_sfreq_changed.connect(self.update_sfreq_blocked)
        self.btn_load_selection.clicked.connect(self.handle_load_data)
        self.sig_data_loaded.connect(self._on_data_loaded)
        self.btn_compute_results.clicked.connect(self.update_results)
        self.action_get_section_result.triggered.connect(self._on_section_done)
        self.btn_calculate_rolling_rate.clicked.connect(self.handle_rolling_rate_calculation)

        # Processing & Editing
        self.btn_apply_filter.clicked.connect(self.process_signal)
        self.btn_detect_peaks.clicked.connect(self.detect_peaks)
        self.plot.sig_peaks_edited.connect(self.handle_scatter_clicked)
        self.sig_data_processed.connect(self.update_data_tables)
        self.sig_peaks_detected.connect(self._on_peaks_detected)
        self.action_remove_selected_peaks.triggered.connect(self.plot.remove_selected_scatter)
        self.action_detect_in_selection.triggered.connect(self.handle_intrasection_peak_detection)
        self.action_hide_selection_box.triggered.connect(self.plot.remove_selection_rect)

        # Plot / View actions
        self.action_reset_view.triggered.connect(self._emit_data_range_info)
        self.sig_update_view_range.connect(self.plot.reset_view_range)
        self.btn_clear_mpl_plot.clicked.connect(self.plot.clear_mpl_plot)
        self.btn_clear_rolling_rate_data.clicked.connect(self._on_clear_rolling_rate_data)
        self.btn_reset_rolling_rate_inputs.clicked.connect(self.ui.set_initial_rolling_rate_inputs)

        # File information / metadata
        self.spin_box_sample_rate.editingFinished.connect(
            lambda: self.data.update_sfreq(self.spin_box_sample_rate.value())
        )
        self.spin_box_sample_rate.editingFinished.connect(
            lambda: self.plot.update_time_axis_scale(self.spin_box_sample_rate.value())
        )
        self.date_edit_file_info.dateChanged.connect(self.data.set_date)
        self.line_edit_subject_id.textChanged.connect(self.data.set_animal_id)
        self.combo_box_oxygen_condition.currentTextChanged.connect(self.data.set_oxy_condition)

    def _on_init_finished(self) -> None:
        self.theme.set_style(self.config.style)
        if os.environ.get("PROFILE", "0") == "1":
            self._add_profiler()
        if os.environ.get("ENABLE_CONSOLE", "0") == "1":
            self._add_jupyter_console()
            self.action_open_console.setChecked(False)
        self.action_group_cc = QtGui.QActionGroup(self.toolbar_plots)
        self.action_group_cc.addAction(self.action_confirm)
        self.action_group_cc.addAction(self.action_cancel)
        self.action_group_cc.setVisible(False)
        self.combo_box_section_select.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        sfreq = self.config.sample_rate
        self.data.set_sfreq(sfreq)
        self.plot.update_time_axis_scale(sfreq)
        self._result: CompleteResult | None = None
        self.previous_filter: str | None = None
        self.current_rr_df_cols = None
        self._setup_tables()

    def _setup_tables(self) -> None:
        self.tables.clear()
        self.table_context_menu = QtWidgets.QMenu()
        self.table_context_menu.addAction("Load More", self._update_cas_table)
        self.table_view_cas.customContextMenuRequested.connect(
            lambda: self.table_context_menu.exec(QtGui.QCursor.pos())
        )
        if self.action_show_data_table.isChecked():
            for table in [
                self.table_view_cas,
                self.table_view_cas_description,
                self.table_view_focused_result,
                self.table_view_rolling_rate,
            ]:
                table.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
                self.tables.add_view(table.objectName(), table)

    # region Dev

    def _add_jupyter_console(self) -> None:
        try:
            import jupyter_client
            import numpy as np
            import polars as pl
            import pyqtgraph as pg
            from PySide6 import QtCore, QtGui, QtWidgets
            from qtconsole import inprocess
            from rich import inspect, print

        except ImportError:
            return

        class JupyterConsoleWidget(inprocess.QtInProcessRichJupyterWidget):
            def __init__(self) -> None:
                super().__init__()
                self.set_default_style("linux")
                self.kernel_manager: inprocess.QtInProcessKernelManager = (
                    inprocess.QtInProcessKernelManager()
                )
                self.kernel_manager.start_kernel()
                self.kernel_client: jupyter_client.blocking.client.BlockingKernelClient = (
                    self.kernel_manager.client()
                )
                self.kernel_client.start_channels()

                qapp_instance = QtWidgets.QApplication.instance()
                if qapp_instance is not None:
                    qapp_instance.aboutToQuit.connect(self.shutdown_kernel)

            @QtCore.Slot()
            def shutdown_kernel(self):
                self.kernel_client.stop_channels()
                self.kernel_manager.shutdown_kernel()

        self.jupyter_console = JupyterConsoleWidget()
        self.jupyter_console_dock = QtWidgets.QDockWidget("Jupyter Console")
        self.jupyter_console_dock.setWidget(self.jupyter_console)
        self.jupyter_console_dock.resize(900, 600)
        self.jupyter_console.kernel_manager.kernel.shell.push(
            dict(
                self=self,
                pg=pg,
                np=np,
                pl=pl,
                print=print,
                qtc=QtCore,
                qtw=QtWidgets,
                qtg=QtGui,
                inspect=inspect,
            )
        )
        self.jupyter_console.execute("whos")

        self.action_open_console.triggered.connect(self._toggle_console)

    @QtCore.Slot()
    def _toggle_console(self) -> None:
        if not hasattr(self, "jupyter_console_dock"):
            return

        if self.jupyter_console_dock.isVisible():
            self.jupyter_console_dock.hide()
        else:
            self.jupyter_console_dock.show()

    def _add_profiler(self) -> None:
        self.menubar.addAction("Start Profiler", self._start_profiler)
        self.menubar.addAction("Stop Profiler", self._stop_profiler)

    @QtCore.Slot()
    def _start_profiler(self):
        self.profiler = cProfile.Profile()
        self.profiler.enable()
        self.statusbar.showMessage("Profiler started.")
        logger.debug(f"Started profiling at: {datetime.now()}")
        self.sig_show_message.emit("Profiler started.", "info")
        self.ui.progress_bar.setRange(0, 0)
        self.ui.progress_bar.show()

    @QtCore.Slot()
    def _stop_profiler(self):
        self.profiler.disable()
        self.statusbar.showMessage("Profiler stopped.")
        dtm = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_path = Path(".") / "logs" / f"cprof_{dtm}.pstats"

        self.profiler.dump_stats(log_path)
        logger.debug(f"Stopped profiling at: {dtm}")
        self.sig_show_message.emit(f"Profiler stopped. Log file written to: \n\n{log_path}", "info")
        self.ui.progress_bar.setRange(0, 100)
        self.ui.progress_bar.hide()

    # endregion Dev

    # region Properties
    @property
    def sig_name(self) -> str:
        """Returns the name of the currently selected signal column."""
        return self.combo_box_signal_column.currentText()

    @property
    def scale_method(self) -> _t.ScaleMethod:
        """Returns the currently selected scale method."""
        return t.cast(_t.ScaleMethod, self.combo_box_scale_method.value())

    @property
    def pipeline(self) -> _t.Pipeline:
        """Returns the currently selected preprocessing pipeline."""
        return t.cast(_t.Pipeline, self.combo_box_preprocess_pipeline.value())

    @property
    def filter_method(self) -> _t.FilterMethod:
        """Returns the currently selected method for filtering the signal values."""
        return t.cast(_t.FilterMethod, self.combo_box_filter_method.value())

    @property
    def peak_detection_method(self) -> _t.PeakDetectionMethod:
        """Returns the currently selected peak detection method / algorithm."""
        return t.cast(_t.PeakDetectionMethod, self.combo_box_peak_detection_method.value())

    @property
    def xqrs_peak_direction(self) -> _t.WFDBPeakDirection:
        """Returns the currently selected direction value to use for the XQRS peak detection algorithm."""
        return t.cast(_t.WFDBPeakDirection, self.peak_xqrs_peak_dir.value())

    @property
    def selection_peak_type(self) -> t.Literal["Auto", "Maxima", "Minima"]:
        """What peak type to look for when detecting peaks in selection. `Auto` chooses the type based on the position of the selection rectangle."""
        return t.cast(
            t.Literal["Auto", "Maxima", "Minima"], self.combo_box_selection_peak_type.currentText()
        )

    # endregion Properties

    @QtCore.Slot(str, str)
    def show_message(
        self, text: str, level: t.Literal["debug", "info", "warning", "critical", "error"]
    ) -> None:
        icon_map = {
            "debug": QtWidgets.QMessageBox.Icon.Question,
            "info": QtWidgets.QMessageBox.Icon.Information,
            "warning": QtWidgets.QMessageBox.Icon.Warning,
            "critical": QtWidgets.QMessageBox.Icon.Critical,
            "error": QtWidgets.QMessageBox.Icon.Critical,
        }
        if level == "debug" and os.environ.get("ENABLE_CONSOLE", "0") == "0":
            return
        msg_box = QtWidgets.QMessageBox()
        msg_box.setIcon(icon_map[level])
        msg_box.setText(text)
        msg_box.setWindowTitle(level.capitalize())
        msg_box.exec()

    @QtCore.Slot()
    def open_config_file(self) -> None:
        file = self.config.config_file.as_posix()
        if not os.path.exists(file):
            self.sig_show_message.emit(f"Config file not found: {file}", "warning")
            return
        # Open the config.ini file in the default os editor
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(file))

    @QtCore.Slot()
    def _on_clear_rolling_rate_data(self) -> None:
        if hasattr(self, "_rr_df_export"):
            self._rr_df_export = None
        self.table_view_rolling_rate.setModel(None)
        self.plot.clear_mpl_plot()

    @QtCore.Slot()
    def _load_focused_result(self) -> None:
        file_paths, selected_filter = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select Focused Result Files to create combined rolling rate for",
            self.config.focused_result_dir.as_posix(),
            "CSV Files (*.csv);;Excel Files (*.xlsx);;Text Files (*.txt)",
            self.previous_filter or "Excel Files (*.xlsx)",
        )
        if not file_paths:
            return

        self.plot.clear_mpl_plot()
        self.table_view_rolling_rate.setModel(None)

        self._current_subject_id = Path(file_paths[0]).parent.name
        self._current_focused_result_file_names = [Path(file_path).name for file_path in file_paths]

        self.previous_filter = selected_filter
        file_0 = file_paths[0]
        self.config.focused_result_dir = Path(file_0).parent
        if "PM_03" in file_0 or "PM03" in file_0 or "PM_05" in file_0 or "PM05" in file_0:
            self.spin_box_focused_sample_rate.setValue(1000)
        else:
            self.spin_box_focused_sample_rate.setValue(400)

        suffix = Path(file_0).suffix
        dataframes: list[pl.DataFrame] = []
        match suffix:
            case ".csv":
                dataframes.extend(pl.read_csv(file_path) for file_path in file_paths)
            case ".xlsx":
                dataframes.extend(pl.read_excel(file_path) for file_path in file_paths)
            case ".txt":
                dataframes.extend(
                    pl.read_csv(file_path, separator="\t") for file_path in file_paths
                )
            case _:
                self.sig_show_message.emit(f"Unsupported file format: {suffix}", "warning")
                return

        # Check if all dataframes have the same column names
        df_cols = dataframes[0].columns
        for df in dataframes[1:]:
            if df.columns != df_cols:
                self.sig_show_message.emit("All dataframes must have the same columns", "warning")
                return
        self.combo_box_grp_col.clear()
        self.combo_box_temperature_col.clear()
        self.combo_box_grp_col.addItems(df_cols)
        self.combo_box_temperature_col.addItems(df_cols)
        if "temperature" in df_cols:
            self.combo_box_temperature_col.setCurrentIndex(df_cols.index("temperature"))

        self.list_widget_focused_results.clear()
        self.list_widget_focused_results.addItems(self._current_focused_result_file_names)
        self._rr_dfs = dataframes
        self.current_rr_df_cols = df_cols

    # region Sections
    @QtCore.Slot()
    def handle_intrasection_peak_detection(self) -> None:
        rect = self.plot.get_selection_rect()
        cas = self.data.cas
        if rect is None or cas is None:
            return
        left, right, top, bottom = int(rect.left()), int(rect.right()), rect.top(), rect.bottom()
        self.plot.remove_selection_rect()
        win_size = self.spin_box_selection_window_size.value()
        edge_buffer = self.spin_box_selection_edge_buffer.value()
        peak_type = self.selection_peak_type
        b_left, b_right = left + edge_buffer, right - edge_buffer
        b_left: int = np.maximum(b_left, 0)
        b_right: int = np.minimum(b_right, cas.proc_data.len())
        data = cas.proc_data[b_left:b_right].to_numpy()
        if self.check_box_use_selection_peak_find.isChecked():
            match peak_type:
                case "Maxima":
                    peaks = find_extrema(data, radius=win_size, direction="up")
                case "Minima":
                    peaks = find_extrema(data, radius=win_size, direction="down")
                case "Auto":
                    data_mean = np.mean(data)
                    if abs(top - data_mean) < abs(bottom - data_mean):
                        peaks = find_extrema(data, radius=win_size, direction="up")
                    else:
                        peaks = find_extrema(data, radius=win_size, direction="down")
        else:
            try:
                peaks = find_peaks(
                    data,
                    sampling_rate=self.data.sfreq,
                    **self.ui.get_peak_detection_parameters(),
                )
            except Exception:
                msg = f"Unable to detect peaks in selection using method: '{self.peak_detection_method}'.\nFalling back to default method (local min/max detection)."
                self.sig_show_message.emit(msg, "warning")
                match peak_type:
                    case "Maxima":
                        peaks = find_extrema(data, radius=win_size, direction="up")
                    case "Minima":
                        peaks = find_extrema(data, radius=win_size, direction="down")
                    case "Auto":
                        data_mean = np.mean(data)
                        if abs(top - data_mean) < abs(bottom - data_mean):
                            peaks = find_extrema(data, radius=win_size, direction="up")
                        else:
                            peaks = find_extrema(data, radius=win_size, direction="down")

        peaks = peaks + b_left
        cas.update_peaks("add", peaks)
        self.sig_peaks_detected.emit()

    @QtCore.Slot(str)
    def add_section_to_widget(self, section_id: SectionID) -> None:
        self.list_widget_sections.addItem(section_id)
        self.combo_box_section_select.addItem(section_id)

    @QtCore.Slot(str)
    def remove_section_from_widget(self, section_id: SectionID) -> None:
        index = self.combo_box_section_select.findText(section_id)
        self.list_widget_sections.takeItem(index)
        self.combo_box_section_select.removeItem(index)
        self.list_widget_sections.setCurrentRow(0)
        self.combo_box_section_select.setCurrentIndex(0)

    @QtCore.Slot()
    def _on_sections_cleared(self) -> None:
        # Only ask if there is any data currently loaded (i.e. dont ask the first time)
        if len(self.data.sections) != 0:
            btn_pressed = QtWidgets.QMessageBox.question(
                self,
                "Reset everything?",
                "This will remove all sections and set the data back to its original state (as read from the file). Continue?",
            )
            if btn_pressed == QtWidgets.QMessageBox.StandardButton.No:
                return
        self.list_widget_sections.clear()
        self.combo_box_section_select.clear()
        self.data.clear_sections()
        self.plot.clear_regions()
        self.data.create_base_df(self.sig_name)
        self.update_data_tables()
        self._result = None

    @QtCore.Slot()
    def _on_section_done(self) -> None:
        if self.data.cas is None:
            return
        if self.data.cas.peaks.len() == 0:
            msg = f"No peaks detected for section '{self.data.cas.section_id}'. Cannot compute results without peaks."
            self.sig_show_message.emit(msg, "info")
            return
        self.statusbar.showMessage("Updating data with values from section...")
        self.ui.progress_bar.reset()
        self.ui.progress_bar.show()
        self.ui.progress_bar.setValue(50)

        self.data.save_cas()
        self.update_result_tables()

        self.ui.progress_bar.setValue(100)
        self.statusbar.showMessage("Focused result ready for export, see 'Results' tab.")
        self.ui.progress_bar.hide()

    @QtCore.Slot()
    def handle_rolling_rate_calculation(self) -> None:
        temperature_col = f"{self.combo_box_temperature_col.currentText()}_mean"
        rate_col = "n_peaks_mean"
        try:
            if not hasattr(self, "_rr_dfs"):
                self.sig_show_message.emit(
                    "Can't find attribute '_rr_dfs' in self. Make sure to load focused result files first.",
                    "error",
                )
                return
            colors = list(mcolors.TABLEAU_COLORS)
            col_enum = pl.Enum(colors[: len(self._rr_dfs)])
            roll_dfs = [
                rolling_rate(df, **self.ui.get_rolling_rate_parameters()).with_columns(
                    plot_color=pl.repeat(pl.lit(colors[i]), pl.len(), dtype=col_enum)
                )
                for i, df in enumerate(self._rr_dfs)
            ]
            roll_df = pl.concat(roll_dfs).sort(temperature_col)
        except Exception as e:
            msg = f"Failed to calculate rolling rate:\n\n{e}\n\nTraceback: {traceback.format_exc()}"
            self.sig_show_message.emit(msg, "error")
            return
        self.tables.update_df_model(roll_df, self.table_view_rolling_rate, limit=3000)

        self.plot.draw_rolling_rate(
            x=roll_df.get_column(temperature_col).to_numpy(),
            y=roll_df.get_column(rate_col).to_numpy(),
            color=roll_df.get_column("plot_color").to_list(),
        )
        self._rr_df_export = roll_df

    @QtCore.Slot()
    def update_results(self) -> None:
        self.ui.progress_bar.reset()
        self.ui.progress_bar.show()
        self.statusbar.showMessage("Computing result for all sections...")
        self.ui.progress_bar.setValue(10)

        try:
            self._result = self.data.get_complete_result()
        except RuntimeWarning:
            msg = "At least one section has no peaks. Either remove the sections, or finish processing them."
            self.sig_show_message.emit(msg, "info")
            self.ui.progress_bar.hide()
            return

        self.ui.progress_bar.setValue(50)
        self.btn_save_to_hdf5.setEnabled(True)
        self.ui.progress_bar.setValue(100)
        msg = "Complete result ready for export, see 'Results' tab for more."
        self.statusbar.showMessage(msg)
        self.sig_show_message.emit(msg, "info")
        self.ui.progress_bar.hide()

    @QtCore.Slot(int)
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
        self.update_data_tables()
        self.ui.label_currently_showing.setText(
            readable_section_id(
                section_id,
                self.line_edit_subject_id.text(),
                str(self.combo_box_oxygen_condition.value()),
            )
        )
        if index != 0:
            self.action_add_section.setEnabled(False)
            self.btn_section_add.setEnabled(False)
            self.btn_section_add.setToolTip(
                "Creating a new section is only possible when the first section (the one ending in `000`) is selected."
            )
        else:
            self.action_add_section.setEnabled(True)
            self.btn_section_add.setEnabled(True)
            self.btn_section_add.setToolTip("Create a new section.")

    @QtCore.Slot()
    def _maybe_new_section(self) -> None:
        if self.data.bounds is None:
            return
        self.container_section_confirm_cancel.show()
        self.action_group_cc.setVisible(True)
        self.plot.show_section_selector(self.data.bounds)

    @QtCore.Slot()
    def _on_section_confirmed(self) -> None:
        lin_reg = self.plot.region_selector
        if lin_reg is None:
            return

        lower, upper = lin_reg.getRegion()
        lower, upper = int(lower), int(upper)
        self.data.new_section(lower, upper)
        self.action_group_cc.setVisible(False)
        self.container_section_confirm_cancel.hide()
        self.btn_section_remove.setEnabled(True)
        self.action_remove_section.setEnabled(True)
        self.sig_section_confirmed.emit(lower, upper)

    @QtCore.Slot()
    def _on_section_canceled(self) -> None:
        self.action_group_cc.setVisible(False)
        self.container_section_confirm_cancel.hide()
        self.plot.remove_section_selector()

    @QtCore.Slot()
    def _remove_section(self) -> None:
        sections = self.data.removable_section_ids
        if len(sections) == 0:
            msg = "No sections to remove."
            self.sig_show_message.emit(msg, "info")
            return
        to_remove, ok = QtWidgets.QInputDialog.getItem(
            self,
            "Remove Section",
            "Select the section to remove. The base section (ending in '000') cannot be removed.",
            [str(item) for item in sections],
            0,
            False,
        )
        if ok:
            sect_id = SectionID(to_remove)
            bounds = self.data.sections[sect_id].base_bounds
            self.data.remove_section(sect_id)
            self.plot.remove_region((bounds[0], bounds[1]))
            if len(self.data.removable_section_ids) == 0:
                self.btn_section_remove.setEnabled(False)
                self.action_remove_section.setEnabled(False)

    # endregion Sections

    @QtCore.Slot()
    def _on_data_loaded(self) -> None:
        if self.config.switch_on_load:
            self.action_toggle_section_sidebar.setChecked(True)
            self.tabs_main.setCurrentIndex(1)
        self.handle_draw_signal()
        self.update_data_tables()

    @QtCore.Slot()
    def _on_peaks_detected(self) -> None:
        self.handle_draw_peaks()
        self.handle_draw_rate()
        self.update_data_tables()

    @QtCore.Slot()
    def _emit_data_range_info(self) -> None:
        if self.data.cas is None:
            return
        self.sig_update_view_range.emit(self.data.cas.data.height)

    @QtCore.Slot(int)
    def update_sfreq_blocked(self, value: int) -> None:
        self.spin_box_sample_rate.blockSignals(True)
        self.spin_box_sample_rate.setValue(value)
        self.spin_box_sample_rate.blockSignals(False)

    @QtCore.Slot()
    def save_to_hdf5(self) -> None:
        out_location = Path.joinpath(
            self.config.output_dir,
            self.config.make_complete_result_name(self.sig_name, self.file_info.completeBaseName()),
        )
        if file_path := QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save to HDF5",
            out_location.as_posix(),
            "HDF5 Files (*.hdf5 *.h5)",
        )[0]:
            complete_results = self.data.get_complete_result()
            if complete_results is None:
                return
            try:
                result_dict_to_hdf5(file_path, complete_results.as_dict())
            except Exception as e:
                msg = f"Failed to write to HDF5 file:\n\n{e}\n\nTraceback: {traceback.format_exc()}"
                self.sig_show_message.emit(msg, "error")
                return
            self.sig_show_message.emit(f"Saved results to:\n\n{file_path}.", "info")
            self.unsaved_changes = False

    @QtCore.Slot(str)
    def export_focused_result(self, output_format: t.Literal["csv", "xlsx", "txt"]) -> None:
        if self.data.cas is None:
            return
        result_file_name = self.config.make_focused_result_name(
            self.sig_name, self.file_info.completeBaseName()
        )
        self.config.output_dir.mkdir(exist_ok=True)
        output_options = [
            "Only export currently active section",
            "Create a new file for each section",
            "Concatenate all sections and write to a single file",
        ]
        item, ok = QtWidgets.QInputDialog.getItem(
            self,
            "Configure Output",
            "Specify what to export:",
            output_options,
            0,
            False,
        )

        if not ok:
            return

        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Select Output Location",
            Path.joinpath(self.config.output_dir, result_file_name).as_posix(),
            f"{output_format.upper()} Files (*.{output_format})",
        )

        self.config.output_dir = Path(file_name).parent
        try:
            results = self.data.get_focused_results()
        except Exception as e:
            msg = f"Failed to create focused results:\n\n{e}\n\nTraceback: {traceback.format_exc()}"
            self.sig_show_message.emit(msg, "error")
            return
        if results is None:
            msg = "No focused results to export."
            self.sig_show_message.emit(msg, "info")
            return

        write_functions: dict[str, t.Callable[[pl.DataFrame, str | Path], "None | Workbook"]] = {
            "csv": lambda df, path: df.write_csv(path),
            "xlsx": lambda df, path: df.write_excel(path),
            "txt": lambda df, path: df.write_csv(path, separator="\t"),
        }
        match item:
            case "Only export currently active section":
                foc_df = self.data.cas.get_focused_result().to_polars()
                write_functions[output_format](foc_df, file_name)
            case "Create a new file for each section":
                for s_id, foc_res in results.items():
                    foc_df = foc_res.to_polars()
                    write_functions[output_format](foc_df, f"{file_name}_{s_id}")
            case "Concatenate all sections and write to a single file":
                result_dfs = [
                    foc_res.to_polars().with_columns(pl.lit(s_id).alias("section_id"))
                    for s_id, foc_res in results.items()
                ]
                result_df = pl.concat(result_dfs)
                write_functions[output_format](result_df, f"{file_name}_all")
            case _:
                return

        msg = f"Exported focused result to:\n\n{file_name}."
        self.sig_show_message.emit(msg, "info")

    @QtCore.Slot(str)
    def export_rolling_rate(self, output_format: t.Literal["csv", "xlsx", "txt"]) -> None:
        if self.data.cas is None and not hasattr(self, "_rr_df_export"):
            return
        result_file_name = self.config.make_rolling_rate_result_name(self._current_subject_id)

        data = self._rr_df_export

        if data is None:
            msg = "No rolling rate data to export. Please make sure to load a focused result file and then create the rolling rate version using the 'Calculate + Plot' button."
            self.sig_show_message.emit(msg, "warning")
            return

        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Select Output Location",
            Path.joinpath(self.config.output_dir, result_file_name).as_posix(),
            f"{output_format.upper()} Files (*.{output_format})",
        )

        if not file_name:
            self.sig_show_message.emit("Export cancelled. No files written.", "info")
            return

        match output_format:
            case "csv":
                data.write_csv(file_name)
            case "xlsx":
                data.write_excel(file_name)
            case "txt":
                data.write_csv(file_name, separator="\t")

        self.sig_show_message.emit(f"Exported rolling rate to:\n\n{file_name}.", "info")

    @QtCore.Slot()
    def _update_cas_table(self) -> None:
        cas = self.data.cas
        if cas is None:
            return
        rows = None
        sender = self.sender()
        if sender is self.btn_show_more_rows:
            rows, ok = QtWidgets.QInputDialog.getInt(
                self,
                "Load More Rows",
                "Enter the number of rows to load:",
                500,
                0,
                cas.data.height,
                1,
            )
            if not ok or rows == 0:
                return
        df = cas.data.lazy()
        self.tables.update_df_model(df, self.table_view_cas, limit=rows)

    @QtCore.Slot()
    def update_data_tables(self) -> None:
        cas = self.data.cas
        if cas is None:
            return
        section_df = cas.data.lazy()
        section_df_desc = cas.data.describe().lazy()

        self.tables.update_df_model(section_df, self.table_view_cas)
        self.tables.update_df_model(section_df_desc, self.table_view_cas_description)

    @QtCore.Slot()
    def update_result_tables(self) -> None:
        """
        Computes the current active sections focused result and displays it in the 'Focused' table
        in the 'Results' tab, where the user can export it to a file.
        """
        cas = self.data.cas
        if cas is None:
            return
        self.statusbar.showMessage("Creating focused result for current section...")
        try:
            cas_foc_res = cas.get_focused_result().to_polars().lazy()
        except RuntimeWarning:
            self.btn_compute_results.failure()
            msg = "At least one section has no peaks. Either remove the sections, or finish processing them."
            self.sig_show_message.emit(msg, "info")
            return
        self.tables.update_df_model(cas_foc_res, self.table_view_focused_result, limit=1000)
        self.label_focused_result_shown.setText(cas.section_id)
        self.btn_compute_results.success()
        self.statusbar.showMessage("Focused result created successfully.", 5000)

    @QtCore.Slot()
    def select_data_file(self) -> None:
        if self.unsaved_changes:
            msg = "You have unsaved changes. Are you sure you want to continue?"
            reply = QtWidgets.QMessageBox.question(
                self,
                "Unsaved Changes",
                msg,
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            if reply == QtWidgets.QMessageBox.StandardButton.No:
                return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
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
            self._setup_tables()
            self.unsaved_changes = False
            self.file_info.setFile(path)
            self.line_edit_active_file.setText(Path(path).name)
            self.config.data_dir = self.file_info.dir().path()
            self.data.read_file(path)
            self.list_widget_sections.clear()
            self.combo_box_section_select.clear()

    @QtCore.Slot()
    def handle_load_data(self) -> None:
        if self.data.raw_df is None:
            return
        self.btn_load_selection.processing("Loading data...")
        self.statusbar.showMessage(f"Loading data from file: {self.file_info.canonicalFilePath()}")

        signal_col = self.combo_box_signal_column.currentText()
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

    @QtCore.Slot()
    def process_signal(self) -> None:
        if self.data.cas is None:
            return
        self.statusbar.showMessage("Filtering data...")
        with pg.BusyCursor():
            filter_params = self.ui.get_filter_parameters()
            standardize_params = self.ui.get_standardize_parameters()
            standardize_params["method"] = self.scale_method

            if (
                self.pipeline == "custom"
                and self.filter_method == "None"
                and self.scale_method == "None"
            ):
                self.data.cas.data = (
                    self.data.cas.data.lazy()
                    .with_columns(
                        pl.col(self.sig_name).alias(f"{self.sig_name}_processed"),
                        pl.lit(False).alias("is_peak"),
                    )
                    .collect()
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

    @QtCore.Slot()
    def detect_peaks(self) -> None:
        if self.data.cas is None:
            return
        self.statusbar.showMessage("Detecting peaks...")
        self.btn_detect_peaks.processing("Working...")
        peak_params = self.ui.get_peak_detection_parameters()
        with pg.BusyCursor():
            self.data.cas.detect_peaks(**peak_params)
        self.sig_peaks_detected.emit()
        self.unsaved_changes = True
        self.btn_detect_peaks.success()
        self.statusbar.showMessage(
            f"Peak detection finished. Found {self.data.cas.peaks.len()} peaks."
        )

    @QtCore.Slot(bool)
    def handle_draw_signal(self, has_peaks: bool = False) -> None:
        if self.data.cas is None:
            return
        self.plot.update_time_axis_scale(self.data.cas.sfreq)
        self.plot.draw_signal(self.data.cas.proc_data.to_numpy(zero_copy_only=True), self.sig_name)
        if has_peaks:
            self.handle_draw_peaks()
            self.handle_draw_rate()

        for region in self.plot.regions:
            if (
                self.data.cas.section_id.endswith("000")
                and self.action_section_overview.isChecked()
            ):
                if region not in self.plot.main_plot_widget.getPlotItem().items:
                    self.plot.main_plot_widget.addItem(region)
                region.show()
            else:
                region.hide()

    @QtCore.Slot()
    def handle_draw_peaks(self) -> None:
        if self.data.cas is None:
            return
        peaks_x, peaks_y = self.data.cas.get_peak_xy()
        self.plot.draw_peaks(peaks_x, peaks_y, self.sig_name)

    @QtCore.Slot(object)
    def handle_draw_rate(self, scatter_plt_item: "CustomScatterPlotItem | None" = None) -> None:
        if self.data.cas is None:
            return
        self.data.cas.calculate_rate(
            sampling_rate=self.data.cas.sfreq, peaks=self.data.cas.peaks.to_numpy()
        )
        self.plot.draw_rate(self.data.cas.rate_interp, self.sig_name)

    @QtCore.Slot(str, list)
    def handle_scatter_clicked(
        self, action: t.Literal["add", "remove"], indices: t.Sequence[int] | npt.NDArray[np.int32]
    ) -> None:
        if self.data.cas is None:
            return
        self.data.cas.update_peaks(action, indices)
        self.sig_peaks_detected.emit()

    @QtCore.Slot(QtGui.QCloseEvent)
    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._write_settings()
        self._write_config()
        if os.environ.get("ENABLE_CONSOLE", "0") == "1":
            self.jupyter_console.shutdown_kernel()
            self.jupyter_console_dock.close()

        super().closeEvent(event)

    def _read_settings(self) -> None:
        settings = QtCore.QSettings("AWI", "Signal Editor")

        self.restoreGeometry(settings.value("geometry"))  # type: ignore
        self.restoreState(settings.value("windowState"))  # type: ignore

    def _write_settings(self) -> None:
        settings = QtCore.QSettings("AWI", "Signal Editor")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())

    def _read_config(self) -> None:
        self.data_dir = self.config.data_dir

        self.output_dir = self.config.output_dir

        self.theme.set_style(self.config.style)
        self.data.set_sfreq(self.config.sample_rate)

    def _write_config(self) -> None:
        data_dir = self.file_info.dir().path()

        self.config.data_dir = Path(data_dir)

        self.config.style = self.theme.active_style
        self.config.sample_rate = self.data.sfreq
        self.config.write_config()


def main() -> None:
    pl.Config.activate_decimals(True)
    pg.setConfigOptions(
        useOpenGL=True,
        enableExperimental=True,
        useNumba=True,
        segmentedLineMode="on",
        antialias=os.environ.get("PG_ANTIALIAS", "0") == "1",
    )
    app = QtWidgets.QApplication(sys.argv)
    window = SignalEditor()
    window.show()
    sys.exit(app.exec())
