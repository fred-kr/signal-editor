import cProfile
import os
import sys
import typing as t
from datetime import datetime
from pathlib import Path

import polars as pl
import pyqtgraph as pg
from loguru import logger
from PySide6 import QtCore, QtGui, QtWidgets

from signal_editor.io import result_dict_to_hdf5

from . import type_aliases as _t
from .handlers import (
    ConfigHandler,
    DataHandler,
    PlotHandler,
    StyleHandler,
    TableHandler,
    UIHandler,
)
from .models import (
    CompleteResult,
    SectionID,
    SectionIndices,
)
from .views.main_window import Ui_MainWindow


def readable_section_id(section_id: SectionID) -> str:
    number = int(section_id[-3:])
    sig_name = section_id.split("_")[1]
    return f"Active Data: Signal={sig_name.upper()}, Section={number:03d}"


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
        self.theme_switcher = StyleHandler()
        self.plot = PlotHandler(self)
        self.data = DataHandler(self)
        self.ui = UIHandler(self, self.plot)
        self.file_info: QtCore.QFileInfo = QtCore.QFileInfo()
        self._read_config()
        self._connect_signals()
        self._result: CompleteResult | None = None
        self.tables = TableHandler()
        self._on_init_finished()

    def _connect_signals(self) -> None:
        """
        Connect signals to slots.
        """
        # General application actions
        self.action_exit.triggered.connect(self.close)
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
        # self.sig_data_restored.connect(self.refresh_app)
        self.btn_compute_results.clicked.connect(self.update_results)
        self.action_get_section_result.triggered.connect(self._on_section_done)

        # Processing & Editing
        self.btn_apply_filter.clicked.connect(self.process_signal)
        self.btn_detect_peaks.clicked.connect(self.detect_peaks)
        self.plot.sig_peaks_edited.connect(self.handle_scatter_clicked)
        self.plot.sig_peaks_drawn.connect(self.handle_draw_rate)
        self.sig_data_processed.connect(self.update_data_tables)
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
        self.action_group_cc = QtGui.QActionGroup(self.toolbar_plots)
        self.action_group_cc.addAction(self.action_confirm)
        self.action_group_cc.addAction(self.action_cancel)
        self.action_group_cc.setVisible(False)
        add_section_menu = QtWidgets.QMenu(self.btn_section_add)
        add_section_menu.addAction("Included", self._maybe_new_included_section)
        add_section_menu.addAction("Excluded", self._maybe_new_excluded_section)
        self.btn_section_add.setMenu(add_section_menu)
        self.data.set_sfreq(self.config.sample_rate)
        self._result: CompleteResult | None = None
        self._setup_tables()

    def _setup_tables(self) -> None:
        self.table_context_menu = QtWidgets.QMenu(self.table_view_cas)
        self.table_context_menu.addAction("Load More", self._update_cas_table)
        self.table_view_cas.customContextMenuRequested.connect(
            lambda: self.table_context_menu.exec(QtGui.QCursor.pos())
        )
        for table in [
            self.table_view_cas,
            self.table_view_cas_description,
            self.table_view_focused_result,
        ]:
            table.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
            self.tables.add_table(table.objectName(), table)

    # region Dev
    def _add_profiler(self) -> None:
        self.menubar.addAction("Start Profiler", self._start_profiler)
        self.menubar.addAction("Stop Profiler", self._stop_profiler)

    @QtCore.Slot()
    def _start_profiler(self):
        self.profiler = cProfile.Profile()
        self.profiler.enable()
        self.statusbar.showMessage("Profiler started.")
        logger.debug(f"Started profiling at: {datetime.now()}")

    @QtCore.Slot()
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
        """Returns the name of the currently selected signal column."""
        return self.combo_box_signal_column.currentText()

    @property
    def cas_id(self) -> SectionID:
        """Returns the ID of the currently active section (CAS)."""
        return self.data.cas.section_id

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

    # endregion Properties

    @QtCore.Slot(str, str)
    def show_message(
        self, text: str, level: t.Literal["debug", "info", "warning", "critical", "error"]
    ) -> None:
        icon_map = {
            "debug": QtWidgets.QMessageBox.Icon.NoIcon,
            "info": QtWidgets.QMessageBox.Icon.Information,
            "warning": QtWidgets.QMessageBox.Icon.Warning,
            "critical": QtWidgets.QMessageBox.Icon.Critical,
            "error": QtWidgets.QMessageBox.Icon.Critical,
        }
        if level == "debug" and os.environ.get("DEV_MODE", "0") == "0":
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

    # region Sections
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
            ok = QtWidgets.QMessageBox.question(
                self,
                "Reset everything?",
                "This will remove all sections and set the data back to its original state (as read from the file). Continue?",
            )
            if ok == QtWidgets.QMessageBox.StandardButton.No:
                return
        self.list_widget_sections.clear()
        self.combo_box_section_select.clear()
        self.data.clear_sections()
        self.data.create_base_df(self.sig_name)
        self.plot.clear_regions()
        self.update_data_tables()
        self._result = None

    @QtCore.Slot()
    def _on_section_done(self) -> None:
        if self.data.cas.peaks.len() == 0:
            msg = f"No peaks detected for section '{self.cas_id}'. Cannot compute results without peaks."
            self.sig_show_message.emit(msg, "info")
            return
        self.statusbar.showMessage("Updating data with values from section...")
        self.ui.progress_bar.reset()
        self.ui.progress_bar.show()
        self.ui.progress_bar.setValue(10)
        self.data.save_cas()
        self.ui.progress_bar.setValue(50)
        self.update_result_tables()
        self.ui.progress_bar.setValue(100)
        self.statusbar.showMessage("Focused result ready for export, see 'Results' tab.")
        self.ui.progress_bar.hide()

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
        self.statusbar.showMessage("Complete result ready for export, see 'Results' tab.")

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

    @QtCore.Slot()
    def _maybe_new_included_section(self) -> None:
        current_limits = self.data.bounds
        self.container_section_confirm_cancel.show()
        self.action_group_cc.setVisible(True)
        self.plot.show_section_selector("included", current_limits)
        self._section_type = "included"

    @QtCore.Slot()
    def _maybe_new_excluded_section(self) -> None:
        current_limits = self.data.bounds
        self.container_section_confirm_cancel.show()
        self.action_group_cc.setVisible(True)
        self.plot.show_section_selector("excluded", current_limits)
        self._section_type = "excluded"

    @QtCore.Slot()
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

    @QtCore.Slot()
    def _on_section_canceled(self) -> None:
        self.action_group_cc.setVisible(False)
        self.container_section_confirm_cancel.hide()
        self.plot.remove_section_selector()

    @QtCore.Slot()
    def _remove_section(self) -> None:
        excluded_sections = self.data.excluded_sections
        excluded_sect_strings = [str(sect) for sect in excluded_sections]
        combined_items = list(self.data.sections.keys())[1:] + excluded_sect_strings
        if len(combined_items) == 0:
            msg = "No sections to remove."
            self.sig_show_message.emit(msg, "info")
            return
        to_remove, ok = QtWidgets.QInputDialog.getItem(
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
            try:
                result_dict_to_hdf5(file_path, complete_results.as_dict())
            except Exception as e:
                msg = f"Failed to write to HDF5 file: {e}"
                self.sig_show_message.emit(msg, "error")
                return
            self.sig_show_message.emit(f"Saved results to {file_path}.", "info")
            self.unsaved_changes = False

    @QtCore.Slot(str)
    def export_focused_result(self, output_format: t.Literal["csv", "xlsx", "txt"]) -> None:
        result_file_name = self.config.make_focused_result_name(
            self.sig_name, self.file_info.completeBaseName()
        )
        self.config.output_dir.mkdir(exist_ok=True)
        result_location = Path.joinpath(Path(self.line_edit_output_dir.text()), result_file_name)
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

        try:
            results = self.data.get_focused_results()

            write_functions: dict[str, t.Callable[[pl.DataFrame, str | Path], t.Any]] = {
                "csv": lambda df, path: df.write_csv(f"{path}.csv"),
                "xlsx": lambda df, path: df.write_excel(f"{path}.xlsx"),
                "txt": lambda df, path: df.write_csv(f"{path}.txt", separator="\t"),
            }
            if item == "Only export currently active section":
                foc_df = self.data.cas.get_focused_result().to_polars()
                write_functions[output_format](foc_df, result_location)

            elif item == "Create a new file for each section":
                for s_id, foc_res in results.items():
                    foc_df = foc_res.to_polars()
                    write_functions[output_format](foc_df, f"{result_location}_{s_id}")

            elif item == "Concatenate all sections and write to a single file":
                result_dfs = [
                    foc_res.to_polars().with_columns(pl.lit(s_id).alias("section_id"))
                    for s_id, foc_res in results.items()
                ]
                result_df = pl.concat(result_dfs)
                write_functions[output_format](result_df, f"{result_location}_all")
        except Exception as e:
            msg = f"Failed to export focused result: {e}"
            self.sig_show_message.emit(msg, "error")
            return

        msg = "Export successful."
        self.sig_show_message.emit(msg, "info")

    @QtCore.Slot()
    def select_output_location(self) -> None:
        """
        Prompt user to select a directory for storing the exported results.
        """
        if path := QtWidgets.QFileDialog.getExistingDirectory(
            self,
            caption="Select Output Directory",
            dir=self.config.output_dir.as_posix(),
            options=QtWidgets.QFileDialog.Option.ShowDirsOnly,
        ):
            self.line_edit_output_dir.setText(path)
            self.config.output_dir = Path(path)

    @QtCore.Slot()
    def _update_cas_table(self) -> None:
        self.tables.set_model_data(self.table_view_cas, self.data.cas.data.lazy())

    @QtCore.Slot()
    def update_data_tables(self) -> None:
        section_df = self.data.cas.data.lazy()
        section_df_desc = self.data.cas.data.describe().lazy()

        self.tables.set_model_data(self.table_view_cas, section_df)
        self.tables.set_model_data(self.table_view_cas_description, section_df_desc)

    @QtCore.Slot()
    def update_result_tables(self) -> None:
        """
        Computes the current active sections focused result and displays it in the 'Focused' table
        in the 'Results' tab, where the user can export it to a file.
        """
        self.statusbar.showMessage("Creating focused result for current section...")
        try:
            cas_foc_res = self.data.cas.get_focused_result().to_polars().lazy()
        except RuntimeWarning:
            self.btn_compute_results.failure()
            msg = "At least one section has no peaks. Either remove the sections, or finish processing them."
            self.sig_show_message.emit(msg, "info")
            return
        self.tables.set_model_data(self.table_view_focused_result, cas_foc_res)
        self.label_focused_result_shown.setText(self.data.cas.section_id)
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
            self.file_info.setFile(path)
            self.line_edit_active_file.setText(Path(path).name)
            self.config.data_dir = self.file_info.dir().path()
            self.data.read_file(path)
            self.list_widget_sections.clear()
            self.combo_box_section_select.clear()

    @QtCore.Slot()
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

    @QtCore.Slot()
    def process_signal(self) -> None:
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
        self.plot.draw_signal(self.data.cas.proc_data.to_numpy(), self.sig_name)
        if has_peaks:
            self.handle_draw_peaks()

        for region in self.plot.combined_regions:
            if self.cas_id.endswith("000") and self.action_section_overview.isChecked():
                if region not in self.plot.main_plot_widget.getPlotItem().items:
                    self.plot.main_plot_widget.addItem(region)
                region.show()
            else:
                region.hide()

    @QtCore.Slot()
    def handle_draw_peaks(self) -> None:
        peaks_x, peaks_y = self.data.cas.get_peak_xy()
        self.plot.draw_peaks(peaks_x, peaks_y, self.sig_name)

    @QtCore.Slot()
    def handle_draw_rate(self) -> None:
        self.data.cas.calculate_rate()
        self.plot.draw_rate(self.data.cas.rate_interp, self.sig_name)

    @QtCore.Slot(str, list)
    def handle_scatter_clicked(
        self, action: t.Literal["add", "remove"], indices: list[int]
    ) -> None:
        # scatter_item = self.plot.scatter_item
        # if scatter_item is None:
        #     return

        self.data.cas.update_peaks(action, indices)
        self.sig_peaks_detected.emit()

    @QtCore.Slot(QtGui.QCloseEvent)
    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._write_settings()
        self._write_config()
        if os.environ.get("ENABLE_CONSOLE", "0") == "1":
            self.ui.jupyter_console_dock.close()

        super().closeEvent(event)

    def _read_settings(self) -> None:
        settings = QtCore.QSettings("AWI", "Signal Editor")
        geometry: QtCore.QByteArray = t.cast(
            QtCore.QByteArray,
            settings.value("geometry", QtCore.QByteArray(), type=QtCore.QByteArray),
        )

        if geometry.size():
            self.restoreGeometry(geometry)

    def _read_config(self) -> None:
        self.data_dir = self.config.data_dir

        self.output_dir = self.config.output_dir

        self.theme_switcher.set_style(self.config.style)

    def _write_settings(self) -> None:
        settings = QtCore.QSettings("AWI", "Signal Editor")

        geometry = self.saveGeometry()
        settings.setValue("geometry", geometry)

    def _write_config(self) -> None:
        data_dir = self.file_info.dir().path()
        output_dir = self.line_edit_output_dir.text()

        self.config.data_dir = Path(data_dir)
        self.config.output_dir = Path(output_dir)

        self.config.style = self.theme_switcher.active_style
        self.config.sample_rate = self.data.sfreq
        self.config.write_config()


def main(dev_mode: bool = False, antialias: bool = False, enable_console: bool = False) -> None:
    if dev_mode:
        os.environ["QT_LOGGING_RULES"] = "qt.pyside.libpyside.warning=true"
        os.environ["DEV_MODE"] = "1"
    else:
        os.environ["QT_LOGGING_RULES"] = "qt.pyside.libpyside.warning=false"
        os.environ["DEV_MODE"] = "0"

    os.environ["ENABLE_CONSOLE"] = "1" if enable_console else "0"
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
    app = QtWidgets.QApplication(sys.argv)
    window = SignalEditor()
    window.show()
    sys.exit(app.exec())
