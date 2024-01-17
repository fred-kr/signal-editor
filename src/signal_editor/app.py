import cProfile
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal, TypedDict, cast

import h5py
import numpy as np
import pyqtgraph as pg
from loguru import logger
from PySide6.QtCore import (
    QAbstractTableModel,
    QByteArray,
    QDate,
    QFileInfo,
    QProcess,
    QSettings,
    Qt,
    Signal,
    Slot,
)
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHeaderView,
    QInputDialog,
    QMainWindow,
    QMessageBox,
    QTableView,
)

from .handlers.config_handler import ConfigHandler
from .handlers.data_handler import DataHandler, DataState
from .handlers.plot_handler import PlotHandler
from .handlers.style_handler import ThemeSwitcher
from .handlers.ui_handler import UIHandler
from .models.data import (
    CompactDFModel,
    DescriptiveStatsModel,
    PolarsModel,
)
from .models.io import write_hdf5
from .models.result import (
    ManualPeakEdits,
    ProcessingParameters,
    Result,
    ResultContainer,
    ResultIdentifier,
    SelectionParameters,
    SummaryStatistics,
)
from .models.signal import SectionID
from .type_aliases import (
    FileMetadata,
    FilterMethod,
    OxygenCondition,
    PeakDetectionElgendiPPG,
    PeakDetectionLocalMaxima,
    PeakDetectionMethod,
    PeakDetectionNeurokit2,
    PeakDetectionPantompkins,
    PeakDetectionParameters,
    PeakDetectionProMAC,
    PeakDetectionXQRS,
    Pipeline,
    ScaleMethod,
    SignalFilterParameters,
    SignalName,
    StandardizeParameters,
    StateDict,
    WFDBPeakDirection,
)
from .views.main_window import Ui_MainWindow


class MainWindow(QMainWindow, Ui_MainWindow):
    sig_data_filtered = Signal(str)
    sig_data_loaded = Signal()
    sig_draw_signal_rate = Signal(str)
    sig_plot_data_changed = Signal(str)
    sig_show_message = Signal(str, str)
    sig_data_restored = Signal()
    sig_init_finished = Signal()
    sig_active_section_changed = Signal(str)

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
        self._results: ResultContainer = ResultContainer()
        self.sig_init_finished.emit()

    @Slot()
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
    def signal_name(self) -> SignalName:
        return "hbr" if self.stacked_hbr_vent.currentIndex() == 0 else "ventilation"

    @property
    def result_name(self) -> SignalName:
        return "hbr" if self.tabs_result.currentIndex() == 0 else "ventilation"

    @property
    def scale_method(self) -> ScaleMethod:
        return cast(ScaleMethod, self.combo_box_scale_method.value())

    @property
    def pipeline(self) -> Pipeline:
        return cast(Pipeline, self.combo_box_preprocess_pipeline.value())

    @property
    def filter_method(self) -> FilterMethod:
        return cast(FilterMethod, self.combo_box_filter_method.value())

    @property
    def peak_detection_method(self) -> PeakDetectionMethod:
        return cast(PeakDetectionMethod, self.combo_box_peak_detection_method.value())

    @property
    def xqrs_peak_direction(self) -> WFDBPeakDirection:
        return cast(WFDBPeakDirection, self.peak_xqrs_peak_dir.value())

    # endregion

    def _add_style_toggle(self) -> None:
        self.menubar.addSeparator()
        self.menubar.addAction("Switch Theme", self.theme_switcher.switch_theme)

    def _setup_section_combo_box(self) -> None:
        self.combo_box_section_select.addItem("IN_001")

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
        self.action_remove_peak_rect.triggered.connect(self.plot.show_region_selector)
        self.action_remove_selected_peaks.triggered.connect(self.plot.remove_selected)
        self.action_reset_view.triggered.connect(
            lambda: self.plot.reset_views(
                self.data.sigs[self.signal_name].active_section.data.height
            )
        )
        self.plot.sig_excluded_range.connect(self.data.exclude_region)
        self.plot.sig_peaks_edited.connect(self.handle_scatter_clicked)
        self.sig_plot_data_changed.connect(self._update_plot_view)

        # Button Actions
        self.btn_apply_filter.clicked.connect(self.handle_apply_filter)
        self.btn_browse_output_dir.clicked.connect(self.select_output_location)
        self.btn_compute_results.clicked.connect(self.update_results)
        self.btn_detect_peaks.clicked.connect(self.handle_peak_detection)
        self.btn_load_selection.clicked.connect(self.handle_load_selection)
        self.btn_save_to_hdf5.clicked.connect(self.save_to_hdf5)
        self.btn_select_file.clicked.connect(self.select_data_file)
        self.btn_section_add_new.clicked.connect(
            lambda: self.ui.confirm_cancel_buttons.setVisible(True)
        )
        self.btn_section_add_new.clicked.connect(self.maybe_new_section)
        self.btn_section_exclude_new.clicked.connect(self.maybe_new_section)

        # Data Export Actions
        self.btn_export_to_csv.clicked.connect(lambda: self.export_results("csv"))
        self.btn_export_to_excel.clicked.connect(lambda: self.export_results("excel"))
        self.btn_export_to_text.clicked.connect(lambda: self.export_results("txt"))

        # Plot View Actions
        self.btn_group_plot_view.idClicked.connect(self.stacked_hbr_vent.setCurrentIndex)

        # Data Related Signals
        self.sig_data_filtered.connect(self.handle_table_view_data)
        self.sig_data_loaded.connect(lambda: self.tabs_main.setCurrentIndex(1))
        self.sig_data_loaded.connect(self.handle_plot_draw)
        self.sig_data_loaded.connect(self.handle_table_view_data)
        self.sig_data_restored.connect(self.refresh_app_state)
        self.sig_draw_signal_rate.connect(self.handle_draw_results)
        self.sig_init_finished.connect(self._on_init_finished)
        self.sig_show_message.connect(self.show_message)
        self.sig_active_section_changed.connect(self._on_section_changed)

        # UI Handler Signals
        self.ui.sig_section_confirmed.connect(self._on_section_chosen)

        # Widget specific Signals
        self.spin_box_fs.valueChanged.connect(self.data.update_fs)
        self.combo_box_section_select.currentTextChanged.connect(self._on_section_changed)

    @Slot()
    def maybe_new_section(self) -> None:
        sender_name = self.sender().objectName()
        sig_name = self.signal_name
        if sender_name == "btn_section_add_new":
            section_type = "included"
        elif sender_name == "btn_section_exclude_new":
            section_type = "excluded"
        else:
            raise ValueError(f"Unknown sender: {sender_name}")

        current_limits = self.data.sigs[sig_name].active_region_limits
        self.ui.confirm_cancel_buttons.show()
        self.plot.show_section_selector(
            name=sig_name, section_type=section_type, bounds=current_limits
        )

    @Slot(bool)
    def _on_section_chosen(self, confirmed: bool) -> None:
        sig_name = self.signal_name

        # Get the new section limits
        lower, upper = self.plot.plot_items[sig_name].active_section.getRegion()
        lower, upper = int(lower), int(upper)

        self.data.sigs[sig_name].add_section(lower, upper, set_active=confirmed, include=confirmed)
        self.combo_box_section_select.addItem(self.data.active_section.section_id)
        self.ui.confirm_cancel_buttons.hide()
        self.sig_active_section_changed.emit(self.data.active_section.section_id)

    @Slot(str)
    def _on_section_changed(self, section_id: SectionID) -> None:
        self.plot.hide_section_selector(self.signal_name)
        self.data.sigs[self.signal_name].set_active_section(section_id)
        self.handle_plot_draw()

    @Slot()
    def remove_excluded_regions(self) -> None:
        name = self.signal_name
        self.data.sigs[name].remove_all_excluded()

    @Slot(str)
    def _sync_peak_indices(self, name: SignalName | str) -> None:
        plot_item = self.plot.plot_items[name].peaks
        if plot_item is None:
            return
        plot_peaks = plot_item.data["x"].astype(np.int32)
        data_peaks = self.data.sigs[name].active_peaks
        plot_peaks.sort()
        data_peaks.sort()

        if not np.array_equal(plot_peaks, data_peaks):
            self.data.sigs[name].active_section.set_peaks(plot_peaks)

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
            write_hdf5(file_path, self._results[self.result_name])

    @Slot()
    def update_results(self) -> None:
        if len(self.data.sigs[self.signal_name].active_peaks) == 0:
            msg = f"No peaks detected for signal '{self.signal_name}'. Please run peak detection first."
            self.sig_show_message.emit(msg, "info")
            return
        self.make_results(self.signal_name)

    @Slot(str)
    def export_results(self, file_type: Literal["csv", "excel", "txt"]) -> None:
        if not self._results[self.result_name]:
            msg = f"No results for `{self.result_name}`. Results can be created using the `Compute Results` button in the `Plots` tab."
            self.sig_show_message.emit(msg, "warning")
            return

        result_location = (
            Path(self.output_dir)
            .joinpath(f"results_{self.result_name}_{self.file_info.baseName()}")
            .as_posix()
        )
        focused_result_df = self._results[self.result_name].focused_result.to_polars()
        btn = None
        try:
            if file_type == "csv":
                btn = self.btn_export_to_csv
                btn.processing("Exporting to CSV...")
                focused_result_df.write_csv(f"{result_location}.csv")
            elif file_type == "excel":
                btn = self.btn_export_to_excel
                btn.processing("Exporting to Excel...")
                focused_result_df.write_excel(f"{result_location}.xlsx")
            elif file_type == "txt":
                btn = self.btn_export_to_text
                btn.processing("Exporting to TXT...")
                focused_result_df.write_csv(f"{result_location}.txt", separator="\t")
            btn.feedback(True)
        except Exception as e:
            if btn:
                btn.feedback(False)
            msg = f"Failed to export results: {e}"
            self.sig_show_message.emit(msg, "error")

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
            self.ui.update_data_selection_widgets(path)

            self.config.data_dir = self.file_info.dir().path()
            self._file_path = path

    @Slot()
    def handle_load_selection(self) -> None:
        self.btn_load_selection.processing("Loading data...")
        try:
            if self.group_box_subset_params.isChecked():
                filter_col = self.combo_box_filter_column.currentText()
                lower = self.dbl_spin_box_subset_min.value()
                upper = self.dbl_spin_box_subset_max.value()
                if (
                    lower != self.dbl_spin_box_subset_min.minimum()
                    or upper != self.dbl_spin_box_subset_max.maximum()
                ):
                    self.data.get_subset(filter_col, lower, upper)

            self.data.df.shrink_to_fit(in_place=True)
            logger.info(
                f"Loaded {self.data.df.shape[0]} rows from {self.data.df.shape[1]} columns, size {self.data.df.estimated_size('mb'):.2f} MB"
            )

            self.btn_load_selection.feedback(True)
            self.sig_data_loaded.emit()
        except Exception as e:
            msg = f"Failed to load data: {e}"
            self.btn_load_selection.feedback(False, "Error", msg)
            logger.error(msg)
            self.statusbar.showMessage(msg)

    @Slot(str)
    def _update_plot_view(self, name: SignalName | str) -> None:
        def update(plot_widget: pg.PlotWidget) -> None:
            plot_item = plot_widget.getPlotItem()
            view_box = plot_item.getViewBox()
            view_box.autoRange()
            view_box.enableAutoRange("y")
            plot_item.setDownsampling(auto=True)
            plot_item.setClipToView(True)
            view_box.setAutoVisible(y=True)
            view_box.setMouseEnabled(x=True, y=False)

        for w_name, widget in self.plot.plot_widgets.items():
            if w_name in name:
                update(widget)

    @Slot()
    def handle_plot_draw(self) -> None:
        with pg.BusyCursor():
            hbr_line_exists = self.plot.plot_items["hbr"].signal is not None
            ventilation_line_exists = self.plot.plot_items["ventilation"].signal is not None

            if not hbr_line_exists and not ventilation_line_exists:
                self._draw_initial_signals()
            else:
                self._update_signal(self.signal_name)

    def _draw_initial_signals(self) -> None:
        for name, sig in self.data.sigs.items():
            plot_widget = self.plot.plot_widgets[name]
            self.plot.draw_signal(sig.get_active_signal(), plot_widget, name)
        self._set_x_ranges()

    def _set_x_ranges(self) -> None:
        data_length = self.data.sigs[self.signal_name].active_section.data.height
        view_boxes = self.plot.plot_widgets.get_all_view_boxes()
        for vb in view_boxes:
            vb.setLimits(
                xMin=-0.25 * data_length,
                xMax=1.25 * data_length,
                maxYRange=1e5,
                minYRange=0.5,
            )
            vb.setXRange(0, data_length)
        for name in ["hbr", "ventilation"]:
            self._update_plot_view(name)

    def _update_signal(self, name: SignalName | str) -> None:
        signal_data = self.data.sigs[name].get_active_signal()
        plot_widget = self.plot.plot_widgets[name]
        self.plot.draw_signal(signal_data, plot_widget, name)
        self._set_x_ranges()
        # self.sig_plot_data_changed.emit(name)

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

        model = CompactDFModel(df_head=df_head, df_tail=df_tail)
        self._set_table_model(self.table_data_preview, model)

        info = DescriptiveStatsModel(df_description)
        self._set_table_model(self.table_data_info, info)

    def _set_table_model(self, table_view: QTableView, model: QAbstractTableModel) -> None:
        """
        Set the model for the given table view

        Parameters
        ----------
        table_view : QTableView
            The table view for which to set the model
        model : QAbstractTableModel
            The model to use
        """
        table_view.setModel(model)
        self._customize_table_header(table_view, model.columnCount())

    @staticmethod
    def _customize_table_header(
        table_view: QTableView,
        n_columns: int,
        header_alignment: Qt.AlignmentFlag = Qt.AlignmentFlag.AlignLeft,
        resize_mode: QHeaderView.ResizeMode = QHeaderView.ResizeMode.Stretch,
    ) -> None:
        """
        Customize header alignment and resize the columns to fill the available
        horizontal space

        Parameters
        ----------
        table_view : QTableView
            The table view to customize
        n_columns : int
            The number of columns in the table
        header_alignment : Qt.AlignmentFlag, optional
            The alignment of the header, by default Qt.AlignmentFlag.AlignLeft
        resize_mode : QHeaderView.ResizeMode, optional
            The resize mode for the columns, by default QHeaderView.ResizeMode.Stretch
        """
        table_view.horizontalHeader().setDefaultAlignment(header_alignment)
        table_view.verticalHeader().setVisible(False)
        table_view.resizeColumnsToContents()
        for col in range(n_columns):
            table_view.horizontalHeader().setSectionResizeMode(col, resize_mode)

    @Slot()
    def handle_apply_filter(self) -> None:
        btn = self.btn_apply_filter
        btn.processing("Filtering...")
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
            self.plot.draw_signal(
                sig=self.data.sigs[signal_name].get_active_signal(),
                plot_widget=self.plot.plot_widgets[signal_name],
                signal_name=signal_name,
            )
        btn.success("Finished")
        self.sig_data_filtered.emit(signal_name)
        self.sig_plot_data_changed.emit(signal_name)

    @Slot()
    def handle_peak_detection(self) -> None:
        btn = self.btn_detect_peaks
        btn.processing("Detecting...")
        with pg.BusyCursor():
            peak_params = self.get_peak_detection_values()
            name = self.signal_name
            plot_widget = self.plot.plot_widgets[name]
            if self.data.sigs[name].active_section.data.shape[0] == 0:
                msg = "Signal needs to be filtered before peak detection can be performed."
                self.sig_show_message.emit(msg, "info")
                return

            self.data.run_peak_detection(
                name=name,
                peak_parameters=peak_params,
            )
            peaks = self.data.sigs[name].active_peaks
            peaks_y = self.data.sigs[name].get_active_signal()[peaks]

            self.plot.draw_peaks(
                pos_x=peaks,
                pos_y=peaks_y,
                plot_widget=plot_widget,
                signal_name=name,
            )
        btn.success("Finished")
        self.sig_draw_signal_rate.emit(name)

    @Slot(str)
    def handle_draw_results(self, name: SignalName | str) -> None:
        rate_name = f"{name}_rate"
        rate_plot_widget = self.plot.plot_widgets[rate_name]
        self.data.sigs[name].active_section.calculate_rate(sampling_rate=self.data.fs)
        rate_interp = self.data.sigs[name].active_section.rate_data.rate_interpolated

        self.plot.draw_rate(
            rate_interp,
            plot_widget=rate_plot_widget,
            signal_name=name,
        )

    @Slot(str)
    def handle_scatter_clicked(self, name: SignalName | str) -> None:
        self._sync_peak_indices(name)
        self.data.sigs[name].active_section.calculate_rate(sampling_rate=self.data.fs)
        self.sig_draw_signal_rate.emit(name)

    @Slot(QCloseEvent)
    def closeEvent(self, event: QCloseEvent) -> None:
        self._write_settings()
        if hasattr(self, "hdf5view_process") and self.hdf5view_process:
            self.hdf5view_process.kill()
            self.hdf5view_process = None
        # if self.ui.console_dock.isVisible():
        # self.ui.console_dock.close()
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

    def get_result(self, result_name: SignalName | str) -> Result:
        return self._results[result_name]

    def get_identifier(self, result_name: SignalName | str) -> ResultIdentifier:
        metadata = self.get_file_metadata()
        result_file_name = f"results_{result_name}_{self.file_info.fileName()}"
        return ResultIdentifier(
            name=result_name,
            animal_id=metadata["animal_id"],
            oxygen_condition=metadata["oxygen_condition"],
            source_file_name=self.file_info.fileName(),
            date_recorded=metadata["date_recorded"],
            result_file_name=result_file_name,
            creation_date=datetime.now(),
        )

    def get_data_selection_info(self) -> SelectionParameters:
        subset_col = self.combo_box_filter_column.currentText()
        lower_bound = self.dbl_spin_box_subset_min.value()
        upper_bound = self.dbl_spin_box_subset_max.value()
        length_overall = self.data.sigs[self.result_name].active_section.data.height
        return SelectionParameters(
            filter_column=subset_col,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            length_overall=length_overall,
        )

    def get_file_metadata(self) -> FileMetadata:
        date_recorded = cast(datetime, self.date_edit_file_info.date().toPython())
        animal_id = self.line_edit_subject_id.text()
        oxygen_condition = cast(OxygenCondition, self.combo_box_oxygen_condition.value())
        return FileMetadata(
            date_recorded=date_recorded,
            animal_id=animal_id,
            oxygen_condition=oxygen_condition,
        )

    def get_filter_values(self) -> SignalFilterParameters:
        method = self.filter_method

        filter_params = SignalFilterParameters(
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

    def get_standardization_values(self) -> StandardizeParameters:
        method = self.scale_method
        robust = method.lower() == "mad"
        if self.container_scale_window_inputs.isChecked():
            window_size = self.spin_box_scale_window_size.value()
        else:
            window_size = None
        return StandardizeParameters(robust=robust, window_size=window_size)

    def get_peak_detection_values(self) -> PeakDetectionParameters:
        method = self.peak_detection_method
        start_index = self.data.active_section.start_index
        stop_index = self.data.active_section.stop_index

        if method == "elgendi_ppg":
            vals = PeakDetectionElgendiPPG(
                peakwindow=self.peak_elgendi_ppg_peakwindow.value(),
                beatwindow=self.peak_elgendi_ppg_beatwindow.value(),
                beatoffset=self.peak_elgendi_ppg_beatoffset.value(),
                mindelay=self.peak_elgendi_ppg_min_delay.value(),
            )
        elif method == "local":
            vals = PeakDetectionLocalMaxima(radius=self.peak_local_max_radius.value())
        elif method == "neurokit2":
            vals = PeakDetectionNeurokit2(
                smoothwindow=self.peak_neurokit2_smoothwindow.value(),
                avgwindow=self.peak_neurokit2_avgwindow.value(),
                gradthreshweight=self.peak_neurokit2_gradthreshweight.value(),
                minlenweight=self.peak_neurokit2_minlenweight.value(),
                mindelay=self.peak_neurokit2_mindelay.value(),
                correct_artifacts=self.peak_neurokit2_correct_artifacts.isChecked(),
            )
        elif method == "promac":
            vals = PeakDetectionProMAC(
                threshold=self.peak_promac_threshold.value(),
                gaussian_sd=self.peak_promac_gaussian_sd.value(),
                correct_artifacts=self.peak_promac_correct_artifacts.isChecked(),
            )
        elif method == "wfdb_xqrs":
            vals = PeakDetectionXQRS(
                search_radius=self.peak_xqrs_search_radius.value(),
                peak_dir=self.xqrs_peak_direction,
            )
        elif method == "pantompkins":
            vals = PeakDetectionPantompkins(
                correct_artifacts=self.peak_pantompkins_correct_artifacts.isChecked(),
            )
        else:
            raise ValueError(f"Unknown peak detection method: {method}")

        for key, val in vals.items():
            if isinstance(val, float):
                vals[key] = np.round(val, 2)

        return PeakDetectionParameters(
            start_index=start_index,
            stop_index=stop_index,
            method=method,
            input_values=vals,
        )

    def get_processing_info(self) -> ProcessingParameters:
        filter_params = self.get_filter_values()
        standardization_params = self.get_standardization_values()
        peak_detection_params = self.get_peak_detection_values()
        return ProcessingParameters(
            sampling_rate=self.data.fs,
            pipeline=self.pipeline,
            filter_parameters=filter_params,
            scaling_parameters=standardization_params,
            peak_detection_parameters=peak_detection_params,
        )

    def get_descriptive_stats(self, result_name: str) -> SummaryStatistics:
        return self.data.get_descriptive_stats(result_name)

    def make_results(self, result_name: SignalName | str) -> None:
        btn = self.btn_compute_results
        btn.processing("Working...")
        with pg.BusyCursor():
            focused_result_df = self.data.get_focused_result_df(result_name)
            results = self._assemble_result_data(result_name)
        btn.success("Finished")

        result_table: QTableView = getattr(self, f"table_view_results_{result_name}")
        self._set_table_model(result_table, PolarsModel(focused_result_df))
        self._results[result_name] = results
        self.tabs_main.setCurrentIndex(2)

    def _assemble_result_data(self, result_name: SignalName | str) -> Result:
        identifier = self.get_identifier(result_name)
        data_info = self.get_data_selection_info()
        processing_info = self.get_processing_info()

        focused_result = self.data.focused_results[result_name]
        statistics = self.data.get_descriptive_stats(result_name)
        manual_edits = self.plot.peak_edits[result_name]
        source_data = self.data.sigs[result_name]
        return Result(
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
            self.plot.last_edit_index,
            0,
            self.data.sigs[self.signal_name].active_section.data.height - 1,
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
            state_dict = StateDict(
                active_signal=self.signal_name,
                source_file_path=self.file_info.filePath(),
                output_dir=self.output_dir.as_posix(),
                data_selection_params=self.get_data_selection_info(),
                data_processing_params=self.get_processing_info(),
                file_metadata=self.get_file_metadata(),
                sampling_frequency=self.data.fs,
                peak_edits=self.plot.peak_edits,
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
                "Pickle Files (*.pkl);;Result Files (*.hdf5 *.h5);;All Files (*.pkl *.hdf5 *.h5)",
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
        selection_params: SelectionParameters,
        processing_params: ProcessingParameters,
        file_metadata: FileMetadata,
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
            state: State = pickle.load(f)

        self.data.restore_state(state["working_data"])
        self.plot.restore_state(state["peak_edits"])
        self.plot.last_edit_index = state["stopped_at_index"]

        self.spin_box_fs.setValue(state["sampling_frequency"])
        self.file_info.setFile(state["source_file_path"])
        self._file_path = state["source_file_path"]

        self.output_dir = Path(state["output_dir"])
        self.line_edit_output_dir.setText(state["output_dir"])
        self.data_dir = Path(state["source_file_path"]).parent

        self.ui.update_data_selection_widgets(path=state["source_file_path"])
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
            signal_widget = self.plot.plot_widgets[s_name]
            processed_name = self.data.sigs[s_name].processed_name
            if processed_name not in self.data.sigs[s_name].data.columns:
                self.plot.draw_signal(
                    self.data.sigs[s_name].active_section.data.get_column(s_name).to_numpy(),
                    signal_widget,
                    s_name,
                )
                continue
            processed_signal = self.data.sigs[s_name].get_active_signal()
            rate = self.data.sigs[s_name].interpolated_rate
            peaks = self.data.sigs[s_name].get_all_peaks()
            peaks_y = processed_signal[peaks]
            rate_widget = self.plot.plot_widgets[f"{s_name}_rate"]
            self.plot.draw_signal(processed_signal, signal_widget, s_name)
            self.plot.draw_peaks(peaks, peaks_y, signal_widget, s_name)
            self.plot.draw_rate(rate, rate_widget, s_name)
            self._set_table_model(
                getattr(self, f"table_view_results_{s_name}"),
                PolarsModel(self.data.focused_results[s_name].to_polars()),
            )


class State(TypedDict):
    active_signal: SignalName
    source_file_path: str
    output_dir: str
    data_selection_params: SelectionParameters
    data_processing_params: ProcessingParameters
    file_metadata: FileMetadata
    sampling_frequency: int
    peak_edits: dict[str, ManualPeakEdits]
    working_data: DataState
    stopped_at_index: int


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
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
