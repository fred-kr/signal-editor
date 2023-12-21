import cProfile
import os
import pickle
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict, cast

import numpy as np
import polars as pl
import pyqtgraph as pg
import qdarkstyle
from loguru import logger
from PySide6.QtCore import (
    QAbstractTableModel,
    QByteArray,
    QFileInfo,
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
from typing_extensions import Literal

from .models.data import (
    CompactDFModel,
    DataHandler,
    DescriptiveStatistics,
    DescriptiveStatsModel,
    PolarsModel,
)
from .type_aliases import (
    FilterMethod,
    OxygenCondition,
    PeakDetectionParameters,
    PeakEdits,
    Pipeline,
    ScaleMethod,
    SignalFilterParameters,
    SignalName,
    StandardizeParameters,
)
from .ui_handler import UIHandler
from .views.main_window import Ui_MainWindow
from .views.plots import PlotHandler


@dataclass(slots=True, kw_only=True, frozen=True)
class ProcessingParameters:
    sampling_rate: int
    pipeline: Pipeline
    filter_parameters: SignalFilterParameters
    standardization_parameters: StandardizeParameters
    peak_detection_parameters: PeakDetectionParameters


@dataclass(slots=True, kw_only=True, frozen=True)
class SelectionParameters:
    subset_column: str
    lower_limit: int | float
    upper_limit: int | float
    length_selection: int


@dataclass(slots=True, kw_only=True, frozen=True)
class ResultIdentifier:
    name: SignalName
    animal_id: str
    environmental_condition: OxygenCondition
    data_file_name: str
    data_measured_date: datetime | None
    result_file_name: str
    result_creation_date: datetime


@dataclass(slots=True, kw_only=True, frozen=True)
class Result:
    identifier: ResultIdentifier
    info_data_selection: SelectionParameters
    info_data_processing: ProcessingParameters
    statistics: DescriptiveStatistics
    result_data: pl.DataFrame
    source_data: pl.DataFrame
    manual_edits: PeakEdits
    other: dict[str, Any] = field(default_factory=dict)


class MainWindow(QMainWindow, Ui_MainWindow):
    sig_data_filtered = Signal(str)
    sig_data_loaded = Signal()
    sig_filter_column_changed = Signal()
    sig_peaks_updated = Signal(str)
    sig_plot_data_changed = Signal(str)
    sig_results_updated = Signal(str)
    sig_show_message = Signal(str, str)
    sig_data_restored = Signal()

    def __init__(self) -> None:
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.setWindowTitle("Signal Editor")
        self._app_dir: Path = Path.cwd()
        self._data_dir: Path = Path.cwd()
        self._output_dir: Path = Path(self._app_dir / "output")
        self._output_dir.mkdir(exist_ok=True)
        self.plot = PlotHandler(self)
        self.data = DataHandler(self)
        self.ui = UIHandler(self, self.plot)
        self.file_info: QFileInfo = QFileInfo()
        # self._add_profiler()
        self.connect_signals()
        self._read_settings()
        self.line_edit_output_dir.setText(self._output_dir.as_posix())

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    @output_dir.setter
    def output_dir(self, value: Path | str) -> None:
        self._output_dir = Path(value)

    @property
    def app_dir(self) -> Path:
        return self._app_dir

    @property
    def data_dir(self) -> Path:
        return self._data_dir

    @data_dir.setter
    def data_dir(self, value: Path | str) -> None:
        self._data_dir = Path(value)

    @property
    def signal_name(self) -> SignalName:
        return "hbr" if self.stacked_hbr_vent.currentIndex() == 0 else "ventilation"

    @property
    def result_name(self) -> SignalName:
        return "hbr" if self.tabs_result.currentIndex() == 0 else "ventilation"

    def _add_profiler(self) -> None:
        self.menubar.addAction("Start Profiler", self._start_profiler)
        self.menubar.addAction("Stop Profiler", self._stop_profiler)

    def connect_signals(self) -> None:
        """
        Connect signals to slots.
        """
        self.sig_data_filtered.connect(self.handle_table_view_data)
        self.sig_data_loaded.connect(self.handle_plot_draw)
        self.sig_data_loaded.connect(self.handle_table_view_data)
        self.sig_peaks_updated.connect(self.handle_draw_results)
        self.sig_peaks_updated.connect(self.handle_table_view_data)
        self.sig_plot_data_changed.connect(self._update_plot_view)
        self.sig_show_message.connect(self.show_message)
        self.sig_data_restored.connect(self.refresh_app_state)

        self.btn_apply_filter.clicked.connect(self.handle_apply_filter)
        self.btn_detect_peaks.clicked.connect(self.handle_peak_detection)

        self.btn_browse_output_dir.clicked.connect(self.select_output_location)
        self.btn_load_selection.clicked.connect(self.handle_load_selection)
        self.btn_select_file.clicked.connect(self.select_data_file)
        self.btn_export_to_csv.clicked.connect(lambda: self.export_results("csv"))
        self.btn_export_to_excel.clicked.connect(lambda: self.export_results("excel"))
        self.btn_export_to_text.clicked.connect(lambda: self.export_results("txt"))
        self.btn_compute_results.clicked.connect(self.update_results)
        # self.btn_save_to_hdf5.clicked.connect(
        #     lambda: self.export_to_pickle(self.result_name)
        # )

        self.btn_group_plot_view.idClicked.connect(
            self.stacked_hbr_vent.setCurrentIndex
        )

        self.action_exit.triggered.connect(self.closeEvent)
        self.action_select_file.triggered.connect(self.select_data_file)
        self.action_show_roi.triggered.connect(self.plot.show_region_selector)
        self.action_remove_selected_peaks.triggered.connect(self.plot.remove_selected)
        # self.action_step_backward.triggered.connect(self.plot.restore_previous_peaks)
        self.action_run_peak_detection.triggered.connect(self.handle_peak_detection)
        self.action_run_preprocessing.triggered.connect(self.handle_apply_filter)
        self.action_save_state.triggered.connect(self.save_state)
        self.action_load_state.triggered.connect(self.restore_state)

        # self.action_rect_mode.toggled.connect(self.plot.set_rect_mode)
        # self.action_pan_mode.toggled.connect(self.plot.set_pan_mode)

        self.spin_box_fs.valueChanged.connect(self.data.update_fs)

        self.plot.sig_peaks_edited.connect(self.handle_scatter_clicked)
        self.plot.sig_peaks_edited.connect(self._sync_peaks_data_plot)

    @Slot(str)
    def _sync_peaks_data_plot(self, name: SignalName) -> None:
        plot_peak_indices = np.asarray(
            getattr(self.plot, f"{name}_peaks_scatter").data["x"], dtype=np.int32
        )
        data_peak_indices = self.data.peaks[name]
        plot_peak_indices.sort()
        data_peak_indices.sort()

        if not np.array_equal(plot_peak_indices, data_peak_indices):
            self.data.peaks.update(name, plot_peak_indices)

    @Slot()
    def update_results(self) -> None:
        self.make_results(self.signal_name)

    @Slot(str)
    def export_results(
        self, file_type: Literal["csv", "excel", "txt", "feather"]
    ) -> None:
        if not hasattr(self, f"{self.result_name}_results"):
            error_msg = f"No existing results for `{self.result_name}`. Results can be created using the `Compute Results` button in the `Plots` tab."
            self.sig_show_message.emit(error_msg, "warning")
            return

        result_file_name = (
            f"{self.output_dir}/results_{self.result_name}_{self.file_info.baseName()}"
        )
        result_table: pl.DataFrame = getattr(
            self, f"{self.result_name}_results"
        ).result_data
        if file_type == "csv":
            result_table.write_csv(f"{result_file_name}.csv")
        elif file_type == "excel":
            result_table.write_excel(f"{result_file_name}.xlsx")
        elif file_type == "txt":
            result_table.write_csv(f"{result_file_name}.txt", separator="\t")
        elif file_type == "feather":
            result_table.write_ipc(f"{result_file_name}.feather")
        else:
            warning_msg = f"Exporting to {file_type} is not supported."
            self.sig_show_message.emit(warning_msg, "warning")

    @Slot(str, str)
    def show_message(
        self, text: str, level: Literal["info", "warning", "critical"]
    ) -> None:
        icon_map = {
            "info": QMessageBox.Icon.Information,
            "warning": QMessageBox.Icon.Warning,
            "critical": QMessageBox.Icon.Critical,
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
            dir=self.data_dir.as_posix(),
            filter="EDF (*.edf);;CSV (*.csv);;TXT (*.txt);;Feather (*.feather);;All Files (*)",
            selectedFilter="All Files (*)",
        )
        if path:
            self.ui.reset_widget_state()
            self.file_info.setFile(path)
            self.line_edit_active_file.setText(Path(path).name)
            self.data.read(path)
            self.ui.update_data_selection_widgets(path)

            self.data_dir = Path(path).parent
            self._file_path = path

    @Slot()
    def handle_load_selection(self) -> None:
        try:
            if self.group_box_subset_params.isChecked():
                filter_col = self.combo_box_filter_column.currentText()
                lower = self.dbl_spin_box_subset_min.value()
                upper = self.dbl_spin_box_subset_max.value()

                self.data.get_subset(filter_col, lower, upper)

            self.data.df.shrink_to_fit(in_place=True)
            logger.info(
                f"Loaded {self.data.df.shape[0]} rows from {self.data.df.shape[1]} columns, size {self.data.df.estimated_size('mb'):.2f} MB"
            )

            self.sig_data_loaded.emit()
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            self.statusbar.showMessage(f"Failed to load data: {e}")

    @Slot(str)
    def _update_plot_view(self, signal_name: SignalName) -> None:
        widget: pg.PlotWidget = getattr(self.plot, f"{signal_name}_plot_widget")
        bpm_widget: pg.PlotWidget = getattr(self.plot, f"bpm_{signal_name}_plot_widget")
        for plot_widget in [widget, bpm_widget]:
            plot_item = plot_widget.getPlotItem()
            plot_widget.getPlotItem().getViewBox().autoRange()
            plot_item.getViewBox().enableAutoRange("y")
            plot_item.setDownsampling(auto=True)
            plot_item.setClipToView(True)
            plot_item.getViewBox().setAutoVisible(y=True)
            plot_item.getViewBox().setMouseEnabled(x=True, y=False)

            # plot_widget.getPlotItem().getViewBox().setAutoVisible(y=True)

    @Slot()
    def handle_plot_draw(self) -> None:
        with pg.BusyCursor():
            hbr_line_exists = self.plot.hbr_signal_line is not None
            ventilation_line_exists = self.plot.ventilation_signal_line is not None

            if not hbr_line_exists and not ventilation_line_exists:
                self._draw_initial_signals()
            else:
                self._update_signal(self.signal_name)

    def _draw_initial_signals(self) -> None:
        hbr_data = self.data.df.get_column("hbr").to_numpy(zero_copy_only=True)
        ventilation_data = self.data.df.get_column("ventilation").to_numpy(
            zero_copy_only=True
        )

        self.plot.draw_signal(hbr_data, self.plot.hbr_plot_widget, "hbr")
        self.plot.draw_signal(
            ventilation_data, self.plot.ventilation_plot_widget, "ventilation"
        )
        self._set_x_ranges()

    def _set_x_ranges(self) -> None:
        data_length = self.data.df.height
        plot_widgets = [
            self.plot.hbr_plot_widget,
            self.plot.bpm_hbr_plot_widget,
            self.plot.ventilation_plot_widget,
            self.plot.bpm_ventilation_plot_widget,
        ]
        for widget in plot_widgets:
            widget.getPlotItem().getViewBox().setLimits(
                xMin=-0.25 * data_length,
                xMax=1.25 * data_length,
                maxYRange=1e5,
                minYRange=0.5,
            )
            widget.getPlotItem().getViewBox().setXRange(0, data_length)
        self._update_plot_view("hbr")
        self._update_plot_view("ventilation")

    def _update_signal(self, signal_name: SignalName) -> None:
        signal_data = self.data.df.get_column(signal_name).to_numpy(zero_copy_only=True)
        plot_widget: pg.PlotWidget = getattr(self.plot, f"{signal_name}_plot_widget")
        self.plot.draw_signal(signal_data, plot_widget, signal_name)
        self.sig_plot_data_changed.emit(signal_name)

    @Slot()
    def handle_table_view_data(self) -> None:
        """
        Prepares and sets the models for data preview and info tables by fetching the
        head, tail and description of the main data. Also adjusts the alignment and
        resizes columns for these tables.
        """
        # Get top and bottom parts of the data and its description
        n_rows = 15
        df_head = self.data.df.head(n_rows).shrink_to_fit(in_place=True)
        df_tail = self.data.df.tail(n_rows).shrink_to_fit(in_place=True)
        df_description = self.data.df.describe(percentiles=None).shrink_to_fit(
            in_place=True
        )

        model = CompactDFModel(df_head=df_head, df_tail=df_tail)
        self._set_table_model(self.table_data_preview, model)

        info = DescriptiveStatsModel(df_description)
        self._set_table_model(self.table_data_info, info)

    def _set_table_model(
        self, table_view: QTableView, model: QAbstractTableModel
    ) -> None:
        """
        Set the model for the given table view

        Args:
            table_view (QTableView): The table view for which to set the model
            model (QAbstractTableModel): The model to use
        """
        table_view.setModel(model)
        self._customize_table_header(table_view, model.columnCount())

    @staticmethod
    def _customize_table_header(table_view: QTableView, n_columns: int) -> None:
        """
        Customize header alignment and resize the columns to fill the available
        horizontal space

        Args:
            table_view (QTableView): The table view to customize
            n_columns (int): The number of columns in the table
        """
        header_alignment = Qt.AlignmentFlag.AlignLeft
        resize_mode = QHeaderView.ResizeMode.Stretch
        table_view.horizontalHeader().setDefaultAlignment(header_alignment)
        table_view.verticalHeader().setVisible(False)
        table_view.resizeColumnsToContents()
        for col in range(n_columns):
            table_view.horizontalHeader().setSectionResizeMode(col, resize_mode)

    @Slot()
    def handle_apply_filter(self) -> None:
        with pg.BusyCursor():
            filter_params = self.get_filter_values()
            standardize_params = self.get_standardization_values()

            pipeline = cast(Pipeline, self.combo_box_preprocess_pipeline.value())

            signal_name = self.signal_name
            processed_signal_name = f"{signal_name}_{self.data.processed_suffix}"

            self.data.run_preprocessing(
                name=signal_name,
                pipeline=pipeline,
                filter_params=filter_params,
                standardize_params=standardize_params,
            )
            self.plot.draw_signal(
                sig=self.data.df.get_column(processed_signal_name).to_numpy(),
                plot_widget=getattr(self.plot, f"{signal_name}_plot_widget"),
                signal_name=signal_name,
            )
        self.sig_plot_data_changed.emit(signal_name)
        self.sig_data_filtered.emit(signal_name)

    @Slot()
    def handle_peak_detection(self) -> None:
        peak_params = self.get_peak_detection_values()
        signal_name = self.signal_name
        processed_signal_name = f"{signal_name}_{self.data.processed_suffix}"
        plot_widget = getattr(self.plot, f"{signal_name}_plot_widget")
        if processed_signal_name not in self.data.df.columns:
            info_msg = (
                "Signal needs to be filtered before peak detection can be performed."
            )
            self.sig_show_message.emit(info_msg, "info")
            return

        with pg.BusyCursor():
            self.data.run_peak_detection(
                name=signal_name,
                **peak_params,
            )
            peaks = self.data.peaks[signal_name]

            self.plot.draw_peaks(
                pos_x=peaks,
                pos_y=self.data.df.get_column(processed_signal_name).to_numpy()[peaks],
                plot_widget=plot_widget,
                signal_name=signal_name,
            )
        self.sig_peaks_updated.emit(signal_name)

    @Slot(str)
    def handle_draw_results(self, signal_name: SignalName) -> None:
        rate_name = f"{signal_name}_{self.data.rate_interp_suffix}"
        rate_plot_widget = getattr(self.plot, f"bpm_{signal_name}_plot_widget")
        instant_signal_rate = self.data.df.get_column(rate_name).to_numpy()

        self.plot.draw_rate(
            instant_signal_rate,
            plot_widget=rate_plot_widget,
            signal_name=signal_name,
        )

    @Slot(str)
    def handle_scatter_clicked(self, name: SignalName) -> None:
        # signal_name = self.signal_name
        # scatter_plot: pg.ScatterPlotItem = getattr(
        #     self.plot, f"{name}_peaks_scatter"
        # )
        self._sync_peaks_data_plot(name)
        # peaks = np.asarray(scatter_plot.data["x"], dtype=np.int32)
        # peaks.sort()
        # self.data.peaks.update(name, peaks)
        # setattr(self.data, peaks_name, peaks)
        self.data.compute_rate(name)
        self.sig_peaks_updated.emit(name)
        # self.handle_draw_results(name)

    @Slot()
    def closeEvent(self, event: QCloseEvent) -> None:
        self._write_settings()
        if self.ui.console_dock.isVisible():
            self.ui.console_dock.close()
        QMainWindow.closeEvent(self, event)

    # region Settings ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
    def _read_settings(self) -> None:
        settings = QSettings("AWI", "Signal Editor")
        geometry: QByteArray = settings.value("geometry", QByteArray())
        if geometry.size():
            self.restoreGeometry(geometry)
        data_dir: str = settings.value("datadir", ".")  # type: ignore
        self.data_dir = Path(data_dir)
        output_dir: str = settings.value("outputdir", ".")  # type: ignore
        self.output_dir = Path(output_dir)

    def _write_settings(self) -> None:
        settings = QSettings("AWI", "Signal Editor")
        geometry = self.saveGeometry()
        settings.setValue("geometry", geometry)
        data_dir = self.data_dir.as_posix()
        output_dir = self.output_dir.as_posix()
        settings.setValue("datadir", data_dir)
        settings.setValue("outputdir", output_dir)

    # endregion

    # region Profiler ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
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

    def get_identifier(self, result_name: SignalName) -> ResultIdentifier:
        meas_date = self.data.meas_date if hasattr(self.data, "meas_date") else None
        result_file_name = f"results_{result_name}_{self.file_info.fileName()}"
        return ResultIdentifier(
            name=result_name,
            animal_id=self.line_edit_subject_id.text(),
            environmental_condition=cast(
                OxygenCondition, self.combo_box_oxygen_condition.currentText()
            ),
            data_file_name=self.file_info.fileName(),
            data_measured_date=meas_date,
            result_file_name=result_file_name,
            result_creation_date=datetime.now(),
        )

    def get_data_info(self) -> SelectionParameters:
        subset_col = self.combo_box_filter_column.currentText()
        return SelectionParameters(
            subset_column=subset_col,
            lower_limit=cast(float, self.data.minmax_map[subset_col]["min"]),
            upper_limit=cast(float, self.data.minmax_map[subset_col]["max"]),
            length_selection=self.data.df.height,
        )

    def get_filter_values(self) -> SignalFilterParameters:
        if self.combo_box_preprocess_pipeline.value() != "custom":
            self.combo_box_filter_method.setValue("None")

        method = cast(FilterMethod, self.combo_box_filter_method.value())

        filter_params = SignalFilterParameters(method=method)
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
        method = cast(ScaleMethod, self.combo_box_scale_method.value())
        if self.container_scale_window_inputs.isChecked():
            window_size = self.spin_box_scale_window_size.value()
        else:
            window_size = "None"

        return StandardizeParameters(method=method, window_size=window_size)

    def get_peak_detection_values(self) -> PeakDetectionParameters:
        if hasattr(self.ui, "peak_params"):
            return self.ui.peak_params.get_values()
        else:
            raise ValueError(
                "Could not get peak detection values, make sure the UIHandler is properly configured."
            )

    def get_processing_info(self) -> ProcessingParameters:
        filter_params = self.get_filter_values()
        standardization_params = self.get_standardization_values()
        peak_detection_params = self.get_peak_detection_values()
        return ProcessingParameters(
            sampling_rate=self.data.fs,
            pipeline=cast(Pipeline, self.combo_box_preprocess_pipeline.value()),
            filter_parameters=filter_params,
            standardization_parameters=standardization_params,
            peak_detection_parameters=peak_detection_params,
        )

    def make_results(self, result_name: SignalName) -> None:
        with pg.ProgressDialog(
            "Computing results...", cancelText=None, parent=self, wait=0
        ) as dlg:
            dlg.setLabelText("Gathering parameters...")
            identifier = self.get_identifier(result_name)
            data_info = self.get_data_info()
            processing_info = self.get_processing_info()
            results_df = self.data.get_result_df(result_name)
            statistics = self.data.get_descriptive_stats(result_name)
            manual_edits = self.plot.peak_edits
            dlg.setValue(20)
            dlg.setLabelText("Creating result table...")
            result_table: QTableView = getattr(
                self, f"table_view_results_{result_name}"
            )
            self._set_table_model(result_table, PolarsModel(results_df))
            # additional_data = {
            #     "rate_bpm_interpolated": self.data.df.get_column(
            #         f"{result_name}_{self.data.rate_interp_suffix}"
            #     ).to_numpy(writable=True),
            #     "original_signal": self.data.df.get_column(result_name).to_numpy(writable=True),
            #     "processed_signal": self.data.df.get_column(
            #         f"{result_name}_{self.data.processed_suffix}"
            #     ).to_numpy(writable=True),
            #     "manual_edits": manual_edits,
            # }
            dlg.setValue(70)
            dlg.setLabelText("Creating results object...")
            results = Result(
                identifier=identifier,
                info_data_selection=data_info,
                info_data_processing=processing_info,
                statistics=statistics,
                result_data=results_df,
                source_data=self.data.df,
                manual_edits=manual_edits,
            )
            dlg.setValue(100)
            setattr(self, f"{result_name}_results", results)
            dlg.setLabelText("Done!")

    # @Slot(str)
    # def export_to_pickle(self, result_name: SignalName) -> None:
    #     if file_path := QFileDialog.getSaveFileName(
    #         self, "Export to Pickle", self.output_dir.as_posix(), "Pickle Files (*.pkl)"
    #     )[0]:
    #         self.save_progress_state(getattr(self, f"{result_name}_results"), file_path)

    # def save_progress_state(self, result: Result, file_path: str | Path) -> None:
    #     data_state = self.data.get_state()
    #     plot_state = self.plot.get_state()
    #     state_dict = {
    #         "identifier": result.identifier,
    #         "data_info": result.info_data_selection,
    #         "processing_info": result.info_data_processing,
    #         "data": data_state,
    #         "plot": plot_state,
    #     }
    #     with open(file_path, "wb") as f:
    #         pickle.dump(state_dict, f)

    @Slot()
    def save_state(self) -> None:
        stopped_at_index, ok = QInputDialog.getInt(
            self,
            "Save State",
            "Data is clean up to (and including) index:",
            0,
            0,
            self.data.df.height,
            1,
            flags=Qt.WindowType.WindowMinimizeButtonHint,
        )
        if not ok:
            msg = "Please enter a valid index position."
            self.sig_show_message.emit(msg, "warning")
            return
        if file_path := QFileDialog.getSaveFileName(
            self, "Save State", self.output_dir.as_posix(), "Pickle Files (*.pkl)"
        )[0]:
            state_dict = {
                "active_signal": self.signal_name,
                "source_file_path": self._file_path,
                "output_dir": self.output_dir.as_posix(),
                "data_selection_params": self.get_data_info(),
                "data_processing_params": self.get_processing_info(),
                "sampling_frequency": self.data.fs,
                "peak_edits": self.plot.peak_edits,
                "working_data": self.data.get_state(),
                "stopped_at_index": stopped_at_index,
            }

            with open(file_path, "wb") as f:
                pickle.dump(state_dict, f)
        else:
            msg = "No output file specified. Data not saved."
            self.sig_show_message.emit(msg, "warning")

    @Slot()
    def restore_state(self) -> None:
        if file_path := QFileDialog.getOpenFileName(
            self, "Restore State", self.output_dir.as_posix(), "Pickle Files (*.pkl)"
        )[0]:
            self.restore_from_pickle(file_path)
        else:
            msg = "No state file specified. Data not restored."
            self.sig_show_message.emit(msg, "warning")

    def restore_input_values(
        self,
        selection_params: SelectionParameters,
        processing_params: ProcessingParameters,
    ) -> None:
        self.combo_box_filter_column.setCurrentText(selection_params.subset_column)
        self.dbl_spin_box_subset_min.setValue(selection_params.lower_limit)
        self.dbl_spin_box_subset_max.setValue(selection_params.upper_limit)

        self.spin_box_fs.setValue(processing_params.sampling_rate)

        self.combo_box_preprocess_pipeline.setValue(processing_params.pipeline)

        self.dbl_spin_box_lowcut.setValue(
            cast(float, processing_params.filter_parameters.get("lowcut", 0.5))
        )
        self.dbl_spin_box_highcut.setValue(
            cast(float, processing_params.filter_parameters.get("highcut", 8.0))
        )
        self.spin_box_order.setValue(
            processing_params.filter_parameters.get("order", 3)
        )
        self.spin_box_window_size.setValue(
            cast(int, processing_params.filter_parameters.get("window_size", 250))
        )
        self.dbl_spin_box_powerline.setValue(
            processing_params.filter_parameters.get("powerline", 50.0)
        )

        self.combo_box_scale_method.setValue(
            processing_params.standardization_parameters.get("method", "None")
        )
        self.spin_box_scale_window_size.setValue(
            processing_params.standardization_parameters.get("window_size", 2000)
        )

        self.combo_box_peak_detection_method.setValue(
            processing_params.peak_detection_parameters.get("method", "elgendi_ppg")
        )
        self.ui.peak_params.set_method(
            processing_params.peak_detection_parameters.get("method", "elgendi_ppg")
        )
        for name, value in processing_params.peak_detection_parameters.get(
            "input_values", {}
        ).items():
            self.ui.peak_params.names[name].setValue(value)

        self.btn_apply_filter.setEnabled(True)
        self.btn_detect_peaks.setEnabled(True)
        self.btn_compute_results.setEnabled(True)

    def restore_from_pickle(self, file_path: str | Path) -> None:
        with open(file_path, "rb") as f:
            state: State = pickle.load(f)

        self.data.restore_state(state["working_data"])
        self.plot.restore_state(state["peak_edits"])

        self.spin_box_fs.setValue(state["sampling_frequency"])
        self.file_info.setFile(state["source_file_path"])
        self.line_edit_active_file.setText(self.file_info.fileName())
        self._file_path = state["source_file_path"]

        self.output_dir = Path(state["output_dir"])
        self.line_edit_output_dir.setText(state["output_dir"])
        self.data_dir = Path(state["source_file_path"]).parent

        self.ui.update_data_selection_widgets(path=state["source_file_path"])

        self.restore_input_values(
            state["data_selection_params"], state["data_processing_params"]
        )

        self.stopped_at_index = state["stopped_at_index"]


        self.sig_data_restored.emit()

    @Slot()
    def refresh_app_state(self) -> None:
        self.handle_table_view_data()
        for s_name in {"hbr", "ventilation"}:
            s_name = cast(SignalName, s_name)
            processed_name = f"{s_name}_{self.data.processed_suffix}"
            if processed_name not in self.data.df.columns:
                continue
            processed_signal = self.data.df.get_column(processed_name).to_numpy()
            rate = self.data.df.get_column(
                f"{s_name}_{self.data.rate_interp_suffix}"
            ).to_numpy()
            peaks = self.data.peaks[s_name]
            peaks_y = processed_signal[peaks]
            signal_widget: pg.PlotWidget = getattr(self.plot, f"{s_name}_plot_widget")
            rate_widget: pg.PlotWidget = getattr(self.plot, f"bpm_{s_name}_plot_widget")
            self.plot.draw_signal(processed_signal, signal_widget, s_name)
            self.plot.draw_peaks(peaks, peaks_y, signal_widget, s_name)
            self.plot.draw_rate(rate, rate_widget, s_name)
            self._set_table_model(getattr(self, f"table_view_results_{s_name}"), PolarsModel(self.data.result_dfs[s_name]))


class State(TypedDict):
    active_signal: SignalName
    source_file_path: str
    output_dir: str
    data_selection_params: SelectionParameters
    data_processing_params: ProcessingParameters
    sampling_frequency: int
    peak_edits: PeakEdits
    working_data: DataHandler.DataState
    stopped_at_index: int


# ==================================================================================== #
#                                      MAIN METHOD                                     #
# ==================================================================================== #
def main(dev_mode: bool = False) -> None:
    if dev_mode:
        os.environ["QT_LOGGING_RULES"] = "qt.pyside.libpyside.warning=true"
    os.environ[
        "LOGURU_FORMAT"
    ] = "<magenta>{time:YYYY-MM-DD HH:mm:ss.SSS}</magenta> | <level>{level: <8}</level> | <yellow>{message}</yellow> | <blue>{name}</blue>.<cyan>{function}()</cyan>, l: <green>{line}</green>\n\n <red>{exception.type}: {exception.value}</red>\n\n{exception.traceback}"

    pg.setConfigOptions(
        useOpenGL=True,
        enableExperimental=True,
        segmentedLineMode="on",
        background="transparent",
        antialias=False,
    )
    logger.add(
        "./logs/debug.log",
        format=(
            "{time:YYYY-MM-DD at HH:mm:ss.SSS} | [{level}]: {message} | module: {name} in {function}, line: {line}"
        ),
        level="DEBUG",
    )
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(
        qdarkstyle.load_stylesheet(qt_api="pyside6", palette=qdarkstyle.DarkPalette)
    )
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
