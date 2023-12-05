import cProfile
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl
import pyqtgraph as pg
import qdarkstyle
from loguru import logger
from numpy.typing import NDArray
from PySide6.QtCore import (
    QAbstractTableModel,
    QByteArray,
    QDate,
    QSettings,
    Qt,
    Signal,
    Slot,
)
from PySide6.QtGui import QCloseEvent, QStandardItemModel
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QFileDialog,
    QHeaderView,
    QMainWindow,
    QMessageBox,
    QTableView,
)

from .custom_types import (
    NormMethod,
    PeakDetectionMethod,
    PeaksPPGElgendi,
    Pipeline,
    SignalFilterParameters,
    SignalName,
)
from .models.data import (
    CompactDFModel,
    DataHandler,
    DescriptiveStatsModel,
    PolarsModel,
    Results,
)
from .models.io import parse_file_name
from .ui_handler import UIHandler
from .views.main_window import Ui_MainWindow
from .views.plots import PlotManager

INITIAL_STATE = {
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
        "enabled": False,
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
    "spin_box_fs": {
        "value": 400,
    },
    "combo_box_preprocess_pipeline": {
        "currentText": "custom",
    },
    "combo_box_filter_method": {
        "enabled": True,
        "currentText": "None",
    },
    "combo_box_standardizing_method": {
        "enabled": True,
        "currentText": "None",
    },
    "container_signal_filter_inputs": {
        "enabled": False,
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
        "value": 251,
    },
    "slider_window_size": {
        "value": 251,
    },
    "combo_box_peak_detection_method": {
        "currentText": "elgendi",
    },
    "btn_find_peaks": {
        "enabled": False,
    },
    "btn_compute_results": {"enabled": False},
    "table_view_results_hbr": {
        "model": QStandardItemModel(),
    },
    "table_view_results_ventilation": {
        "model": QStandardItemModel(),
    },
    "tab_widget_results": {
        "currentIndex": 0,
    },
}


class MainWindow(QMainWindow, Ui_MainWindow):
    sig_filter_column_changed = Signal()
    sig_lazy_ready = Signal(str)
    sig_data_loaded = Signal()
    sig_data_filtered = Signal(str)
    sig_peaks_updated = Signal(str)
    sig_plot_data_changed = Signal(str)
    sig_init_complete = Signal()
    sig_prepare_new_data = Signal()
    sig_results_updated = Signal(str)

    def __init__(self, app_wd: str) -> None:
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.setWindowTitle("Signal Editor")
        self.setDockNestingEnabled(True)
        self.plot = PlotManager()
        self.connect_signals()
        self._read_settings()
        # self._add_profiler()
        self._wd = app_wd
        self.sig_init_complete.emit()

    def _add_profiler(self) -> None:
        self.menu_info.addAction("Start Profiler", self._start_profiler)
        self.menu_info.addAction("Stop Profiler", self._stop_profiler)

    @Slot()
    def _init_complete(self) -> None:
        self.widgets = UIHandler(self, self.plot)
        self.widgets.sig_apply_filter.connect(self.handle_apply_filter)
        self.widgets.sig_peak_detection_inputs.connect(self.handle_peak_detection)
        self.line_edit_output_dir.setText(self._wd)

    def get_signal_name(self) -> SignalName:
        return "hbr" if self.stacked_hbr_vent.currentIndex() == 0 else "ventilation"

    def connect_signals(self) -> None:
        self.sig_init_complete.connect(self._init_complete)

        self.btn_select_file.clicked.connect(self.handle_select_file)
        self.btn_load_selection.clicked.connect(self.handle_load_selection)
        self.btn_browse_output_dir.clicked.connect(self.handle_browse_output_dir)

        self.action_select_file.triggered.connect(self.handle_select_file)
        self.action_exit.triggered.connect(self.closeEvent)

        self.sig_lazy_ready.connect(self.handle_filter_column_changed)
        self.sig_data_loaded.connect(self.handle_plot_draw)
        self.sig_data_loaded.connect(self.handle_table_view_data)
        self.sig_peaks_updated.connect(self.handle_draw_results)
        self.sig_peaks_updated.connect(self.handle_table_view_data)
        self.sig_plot_data_changed.connect(self._update_plot_view)
        self.sig_data_filtered.connect(self.handle_table_view_data)
        self.sig_prepare_new_data.connect(self.prepare_new_data)

        self.plot.sig_peaks_edited.connect(self.handle_scatter_clicked)

    @Slot()
    def handle_browse_output_dir(self) -> None:
        """
        Prompt user to select a directory for storing the exported results.
        """
        if path := QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            dir=Path(".").resolve().as_posix(),
            options=QFileDialog.Option.ShowDirsOnly,
        ):
            self.line_edit_output_dir.setText(path)

    @Slot()
    def handle_select_file(self) -> None:
        """
        Handles loading a new file into the editor via multiple helper methods.
        """
        if path := self.select_file_dialog():
            self.sig_prepare_new_data.emit()
            self._update_active_file_display(path)
            self._initialize_data_handler(path)
            self._update_data_selection_ui(path)

    def select_file_dialog(self) -> str:
        """
        Prompt user to select a file.

        Returns:
            str: The selected file path.
        """
        default_dir = self.get_default_directory()
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select File",
            default_dir,
            filter="EDF (*.edf);;CSV (*.csv);;TXT (*.txt);;Feather (*.feather);;All Files (*)",
            selectedFilter="EDF (*.edf)",
        )
        return path

    def get_default_directory(self) -> str:
        """
        Returns the default directory to use when selecting files.

        If a file is already loaded, the directory of that file is returned. If no file
        is loaded, the current working directory is returned.

        Returns:
            str: The default directory to use when selecting files.
        """
        if hasattr(self, "dm") and self.dm.file_path:
            return Path(self.dm.file_path).parent.as_posix()
        return Path(".").resolve().as_posix()

    def _update_active_file_display(self, path: str) -> None:
        """
        Updates the line edit widget with the name of the currently loaded file.

        Args:
            path (str): The full path to the data file.
        """
        self.line_edit_active_file.setText(Path(path).name)

    def _initialize_data_handler(self, path: str) -> None:
        """
        Initializes a DataHandler object with the given file path and sampling rate.

        Args:
            path (str): The full path to the data file.
        """
        sampling_rate = self.spin_box_fs.value()
        self.dm = DataHandler(file_path=path, sampling_rate=sampling_rate)
        self.dm.lazy_read()
        self.dm.get_min_max()

    def _update_data_selection_ui(self, path: str) -> None:
        """
        Updates UI elements related to subsetting and file metadata.

        Args:
            path (str): The full path to the data file.
        """
        self.group_box_subset_params.setEnabled(True)
        self.container_file_info.setEnabled(True)
        parsed_date, parsed_id = parse_file_name(Path(path).name)
        self.date_edit_file_info.setDate(
            QDate(parsed_date.year, parsed_date.month, parsed_date.day)
        )
        self.line_edit_subject_id.setText(parsed_id)
        self.btn_load_selection.setEnabled(True)
        self._update_filter_columns()

    def _update_filter_columns(self) -> None:
        """
        Updates the available columns in the filter combo box.

        Note:
            Currently assumes existence of `index`, `time_s`, and `temperature` columns.
        """
        # TODO: Don't hardcode viable filter columns
        viable_filter_columns = ("index", "time_s", "temperature")
        self.combo_box_filter_column.blockSignals(True)
        self.combo_box_filter_column.clear()
        self.combo_box_filter_column.addItems(viable_filter_columns)
        self.combo_box_filter_column.setCurrentIndex(0)
        self.combo_box_filter_column.currentTextChanged.connect(
            self.handle_filter_column_changed
        )
        self.combo_box_filter_column.blockSignals(False)
        self.sig_lazy_ready.emit(self.combo_box_filter_column.currentText())

    @Slot(str)
    def handle_filter_column_changed(self, text: str) -> None:
        """
        Update the min and max spin boxes based on the selected column's min and max values.

        Args:
            text (str): The name of the column.
        """
        lower = self.dm.min_max_mapping[text]["min"]
        upper = self.dm.min_max_mapping[text]["max"]

        if text == "index":
            self.dbl_spin_box_subset_min.setDecimals(0)
            self.dbl_spin_box_subset_max.setDecimals(0)
            self.dbl_spin_box_subset_min.setSingleStep(1)
            self.dbl_spin_box_subset_max.setSingleStep(1)
        elif text == "temperature":
            self.dbl_spin_box_subset_min.setDecimals(1)
            self.dbl_spin_box_subset_max.setDecimals(1)
            self.dbl_spin_box_subset_min.setSingleStep(0.1)
            self.dbl_spin_box_subset_max.setSingleStep(0.1)

        elif text == "time_s":
            self.dbl_spin_box_subset_min.setDecimals(4)
            self.dbl_spin_box_subset_max.setDecimals(4)
            self.dbl_spin_box_subset_min.setSingleStep(0.0025)
            self.dbl_spin_box_subset_max.setSingleStep(0.0025)
        self.dbl_spin_box_subset_min.setMinimum(lower)
        self.dbl_spin_box_subset_min.setMaximum(upper)
        self.dbl_spin_box_subset_min.setValue(lower)

        self.dbl_spin_box_subset_max.setMinimum(lower)
        self.dbl_spin_box_subset_max.setMaximum(upper)
        self.dbl_spin_box_subset_max.setValue(upper)

    @Slot()
    def prepare_new_data(self) -> None:
        """
        Reset all widgets to their initial state.
        """
        self.tabWidget.setCurrentIndex(0)
        method_mapping = {
            "enabled": "setEnabled",
            "checked": "setChecked",
            "text": "setText",
            "model": "setModel",
            "value": "setValue",
            "currentText": "setCurrentText",
            "currentIndex": "setCurrentIndex",
        }
        for widget_name, widget_state in INITIAL_STATE.items():
            for state_key, state_value in widget_state.items():
                if state_key in method_mapping:
                    getattr(self, widget_name).__getattribute__(
                        method_mapping[state_key]
                    )(state_value)

        self.plot.reset_plots()
        self.widgets.temperature_label_hbr.setText("Temperature: -")
        self.widgets.temperature_label_ventilation.setText("Temperature: -")
        self.statusbar.showMessage("Ready")

    # @Slot()
    # def handle_load_selection(self) -> None:
    #     try:
    #         filter_col = self.combo_box_filter_column.currentText()
    #         lower = self.dbl_spin_box_subset_min.value()
    #         upper = self.dbl_spin_box_subset_max.value()

    #         if self.group_box_subset_params.isChecked():
    #             # Because the temperature column values are not monotonically increasing/decreasing in value, just selecting all values where temperature > x leads to lots of missing values around the threshold whenever the temperature value dips below this limit.
    #             # To counteract this, the filtering looks as follows:
    #             # 1. Create a helper column thats set to 1 if the temperature is above the lower threshold, 0 otherwise
    #             # 2. Create a second helper column thats set to 1 if the temperature is above the upper threshold, 0 otherwise
    #             # 3. Filter the data frame to only keep the rows where the helper column is 1 and the second helper column is 0, i.e. between the first occurence of the lower threshold and the first occurence of the upper threshold
    #             # 4. The resulting dataframe wont have a clean lower limit cutoff, but there also won't be any missing values that mess up the plot
    #             lf_filtered = (
    #                 self.dm.lazy.with_columns(
    #                     above_lower=(
    #                         pl.when(pl.col(filter_col).ge(pl.lit(lower)))
    #                         .then(1)
    #                         .otherwise(0)
    #                     ),
    #                     above_upper=(
    #                         pl.when(pl.col(filter_col).ge(pl.lit(upper)))
    #                         .then(1)
    #                         .otherwise(0)
    #                     ),
    #                 )
    #                 .inspect()
    #                 .with_columns(
    #                     [
    #                         pl.col("above_lower").cum_max().alias("cum_above_lower"),
    #                         pl.col("above_upper").cum_max().alias("cum_above_upper"),
    #                     ]
    #                 )
    #                 .inspect()
    #                 .filter(
    #                     (pl.col("cum_above_lower") == 1)
    #                     & (pl.col("cum_above_upper") == 0)
    #                 )
    #             )
    #             self.dm.data = (
    #                 lf_filtered.select(
    #                     pl.all().exclude(
    #                         "above_lower",
    #                         "above_upper",
    #                         "cum_above_lower",
    #                         "cum_above_upper",
    #                     )
    #                 )
    #                 .collect()
    #                 .shrink_to_fit(in_place=True)
    #             )
    #         else:
    #             self.dm.data = self.dm.lazy.collect().shrink_to_fit(in_place=True)

    #         logger.info(
    #             f"Loaded {self.dm.data.shape[0]} rows from {self.dm.data.shape[1]} columns, size {self.dm.data.estimated_size('mb'):.2f} MB"
    #         )
    #         self.sig_data_loaded.emit()
    #     except Exception as e:
    #         logger.error(e)
    #         return

    @Slot()
    def handle_load_selection(self) -> None:
        try:
            lazy_frame = self.dm.lazy
            if self.group_box_subset_params.isChecked():
                filter_col = self.combo_box_filter_column.currentText()
                lower = self.dbl_spin_box_subset_min.value()
                upper = self.dbl_spin_box_subset_max.value()

                if filter_col == "temperature":
                    lazy_frame = self.dm.lazy.with_columns(
                        pl.col("temperature").round(1).alias("temperature")
                    )

                    lower_idx = (
                        lazy_frame.filter(pl.col(filter_col) >= pl.lit(lower))
                        .first()
                        .collect()
                        .get_column("index")[0]
                    )
                    upper_idx = (
                        lazy_frame.filter(pl.col(filter_col) >= pl.lit(upper))
                        .first()
                        .collect()
                        .get_column("index")[0]
                    )

                    if lower_idx is not None and upper_idx is not None:
                        self.dm.data = (
                            lazy_frame.slice(lower_idx, upper_idx - lower_idx + 1)
                            .collect()
                            .shrink_to_fit(in_place=True)
                        )
                    else:
                        logger.warning(
                            "Could not find any data in the range specified."
                        )
                elif filter_col in {"index", "time_s"}:
                    self.dm.data = (
                        lazy_frame.filter(pl.col(filter_col).is_between(lower, upper))
                        .collect()
                        .shrink_to_fit(in_place=True)
                    )
                else:
                    self.dm.data = self.dm.lazy.collect().shrink_to_fit(in_place=True)

                #     first_lower_exceed = lazy_frame.filter(
                #         pl.col(filter_col) >= pl.lit(lower)
                #     ).first()
                #     first_upper_exceed = lazy_frame.filter(
                #         pl.col(filter_col) >= pl.lit(upper)
                #     ).first()

                #     lower_idx = first_lower_exceed.collect().get_column("index")[0]
                #     upper_idx = first_upper_exceed.collect().get_column("index")[0]
                #     logger.debug(f"lower_idx: {lower_idx}, upper_idx: {upper_idx}, length: {upper_idx - lower_idx + 1}")

                #     self.dm.data = (
                #         lazy_frame.slice(lower_idx, upper_idx - lower_idx + 1)
                #         .collect()
                #         .shrink_to_fit(in_place=True)
                #     )
                # elif filter_col in {"index", "time_s"}:
                #     self.dm.data = (
                #         lazy_frame.filter(
                #             (pl.col(filter_col) >= pl.lit(lower))
                #             & (pl.col(filter_col) <= pl.lit(upper))
                #         )
                #         .collect()
                #         .shrink_to_fit(in_place=True)
                #     )

            logger.info(
                f"Loaded {self.dm.data.shape[0]} rows from {self.dm.data.shape[1]} columns, size {self.dm.data.estimated_size('mb'):.2f} MB"
            )
            self.sig_data_loaded.emit()
        except Exception as e:
            logger.error(e)
            return

    @Slot(str)
    def _update_plot_view(self, signal_name: SignalName) -> None:
        widget: pg.PlotWidget = getattr(self.plot, f"{signal_name}_plot_widget")
        widget.autoRange()
        widget.enableAutoRange(y=True, enable=0.95)
        widget.setAutoVisible(y=True)

    @Slot()
    def handle_plot_draw(self) -> None:
        with pg.BusyCursor():
            signal_name = self.get_signal_name()
            hbr_line_exists = self.plot.hbr_signal_line is not None
            ventilation_line_exists = self.plot.ventilation_signal_line is not None

            if not hbr_line_exists and not ventilation_line_exists:
                self._draw_initial_signals()
            else:
                self._update_signal(signal_name)

    def _draw_initial_signals(self) -> None:
        hbr_data = self.dm.data.get_column("hbr").to_numpy(zero_copy_only=True)
        ventilation_data = self.dm.data.get_column("ventilation").to_numpy(
            zero_copy_only=True
        )

        self.plot.draw_signal(hbr_data, self.plot.hbr_plot_widget, "hbr")
        self.plot.draw_signal(
            ventilation_data, self.plot.ventilation_plot_widget, "ventilation"
        )
        self._set_x_ranges()

    def _set_x_ranges(self) -> None:
        data_length = self.dm.data.shape[0]
        self.plot.hbr_plot_widget.setXRange(0, data_length)
        self.plot.ventilation_plot_widget.setXRange(0, data_length)
        self.plot.hbr_plot_widget.getPlotItem().getViewBox().setLimits(
            xMin=-0.5 * data_length, xMax=1.5 * data_length
        )
        self.plot.ventilation_plot_widget.getPlotItem().getViewBox().setLimits(
            xMin=-0.5 * data_length, xMax=1.5 * data_length
        )

    def _update_signal(self, signal_name: SignalName) -> None:
        signal_data = self.dm.data.get_column(signal_name).to_numpy(zero_copy_only=True)
        plot_widget = getattr(self.plot, f"{signal_name}_plot_widget")
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
        df_head = self.dm.data.head(n_rows).shrink_to_fit(in_place=True)
        df_tail = self.dm.data.tail(n_rows).shrink_to_fit(in_place=True)
        df_description = self.dm.data.describe(percentiles=None).shrink_to_fit(
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

    @Slot(dict)
    def handle_apply_filter(self, filter_params: SignalFilterParameters) -> None:
        with pg.BusyCursor():
            norm_method = self.combo_box_standardizing_method.currentText()
            norm_method = cast(NormMethod, norm_method)

            pipeline = self.combo_box_preprocess_pipeline.currentText()
            pipeline = cast(Pipeline, pipeline)

            signal_name = self.get_signal_name()

            self.dm.preprocess_signal(
                signal_name=signal_name,
                norm_method=norm_method,
                pipeline=pipeline,
                filter_params=filter_params,
            )
            self.plot.draw_signal(
                sig=self.dm.data.get_column(f"processed_{signal_name}").to_numpy(
                    zero_copy_only=True
                ),
                plot_widget=getattr(self.plot, f"{signal_name}_plot_widget"),
                signal_name=signal_name,
            )
        self.sig_plot_data_changed.emit(signal_name)
        self.sig_data_filtered.emit(signal_name)

    @Slot(dict)
    def handle_peak_detection(self, peak_params: PeaksPPGElgendi) -> None:
        signal_name = self.get_signal_name()
        if f"processed_{signal_name}" not in self.dm.data.columns:
            info_msg = (
                "Signal needs to be filtered before peak detection can be performed."
            )
            popup = QMessageBox(
                QMessageBox.Icon.Warning,
                "Warning",
                info_msg,
                QMessageBox.StandardButton.Ok,
                parent=self,
            )
            popup.show()
            return
        peak_detection_method = self.combo_box_peak_detection_method.currentText()
        peak_detection_method = cast(PeakDetectionMethod, peak_detection_method)

        with pg.BusyCursor():
            self.dm.find_peaks(
                signal_name=signal_name,
                peak_find_method=peak_detection_method,
                **peak_params,
            )
            peaks: NDArray[np.int32] = getattr(self.dm, f"{signal_name}_peaks")

            self.plot.draw_peaks(
                pos_x=peaks,
                pos_y=self.dm.data.get_column(f"processed_{signal_name}").to_numpy(
                    zero_copy_only=True
                )[peaks],
                plot_widget=getattr(self.plot, f"{signal_name}_plot_widget"),
                signal_name=signal_name,
            )
        self.sig_peaks_updated.emit(signal_name)

    @Slot(str)
    def handle_draw_results(self, signal_name: SignalName) -> None:
        instant_signal_rate: NDArray[np.float32] = getattr(
            self.dm, f"{signal_name}_rate_len_signal"
        )
        self.plot.draw_bpm(
            instant_signal_rate,
            plot_widget=getattr(self.plot, f"bpm_{signal_name}_plot_widget"),
            signal_name=signal_name,
        )

    @Slot()
    def handle_scatter_clicked(self) -> None:
        signal_name = self.get_signal_name()
        peaks = np.asarray(
            getattr(self.plot, f"{signal_name}_peaks_scatter").data["x"],
            dtype=np.int32,
        )
        peaks.sort()
        setattr(self.dm, f"{signal_name}_peaks", peaks)
        self.dm.calculate_rate(signal_name, peaks)
        self.handle_draw_results(signal_name)

    @Slot(bool)
    def closeEvent(self, event: QCloseEvent) -> None:
        self._write_settings()
        super().closeEvent(event)

    def _read_settings(self) -> None:
        settings = QSettings("AWI", "Signal Editor")
        geometry: QByteArray = settings.value("geometry", QByteArray())  # type: ignore
        if geometry.size():
            self.restoreGeometry(geometry)

    def _write_settings(self) -> None:
        settings = QSettings("AWI", "Signal Editor")
        geometry = self.saveGeometry()
        settings.setValue("geometry", geometry)

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

    def make_results(self, signal_name: SignalName) -> None:
        identifier = self.widgets.get_identifier()
        working_data_metadata = self.widgets.get_info_working_data()
        processing_metadata = self.widgets.get_info_processing_params()
        computed = self.dm.compute_results(signal_name)
        results_df = self.dm.make_results_df(
            peaks=getattr(self.dm, f"{signal_name}_peaks"),
            rate=getattr(self.dm, f"{signal_name}_rate_len_peaks"),
        )
        results = Results(
            signal_name=signal_name,
            identifier=identifier,
            working_data_metadata=working_data_metadata,
            processing_metadata=processing_metadata,
            computed=computed,
            processed_data=results_df,
        )
        setattr(self, f"{signal_name}_results", results)

    def make_results_table(self) -> None:
        signal_name = self.get_signal_name()
        main_table: pl.DataFrame = getattr(
            self, f"{signal_name}_results"
        ).processed_data
        model = PolarsModel(main_table)
        table_view: QTableView = getattr(self, f"table_view_results_{signal_name}")
        self._set_table_model(table_view, model)

    def get_results_table(self, signal_name: SignalName) -> pl.DataFrame:
        return getattr(self, f"{signal_name}_results").processed_data


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Main Method                                                                          #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def main(app_wd: str, dev_mode: bool = False) -> None:
    if dev_mode:
        os.environ["QT_LOGGING_RULES"] = "qt.pyside.libpyside.warning=true"
    os.environ[
        "LOGURU_FORMAT"
    ] = "<magenta>{time:YYYY-MM-DD HH:mm:ss.SSS}</magenta> | <level>{level: <8}</level> | <yellow>{message}</yellow> | <blue>{name}</blue>.<cyan>{function}()</cyan>, l: <green>{line}</green>\n\n <red>{exception.type}: {exception.value}</red>\n\n{exception.traceback}"

    pg.setConfigOptions(
        useOpenGL=True,
        enableExperimental=True,
        segmentedLineMode="on",
        background="black",
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
    # app.setStyle(QStyleFactory.create("default"))
    app.setStyleSheet(
        qdarkstyle.load_stylesheet(qt_api="pyside6", palette=qdarkstyle.DarkPalette)
    )

    window = MainWindow(app_wd)
    window.show()
    sys.exit(app.exec())
