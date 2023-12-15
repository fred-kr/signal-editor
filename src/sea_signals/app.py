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
    QMainWindow,
    QMessageBox,
    QTableView,
)
from typing_extensions import Literal

from .models.data import (
    CompactDFModel,
    DataHandler,
    DescriptiveStatsModel,
    PolarsModel,
    ProcessingParameters,
    Result,
    ResultIdentifier,
    SelectionParameters,
)
from .type_aliases import (
    FilterMethod,
    OxygenCondition,
    PeakDetectionManualEdited,
    PeakDetectionParameters,
    Pipeline,
    ScaleMethod,
    SignalFilterParameters,
    SignalName,
    StandardizeParameters,
)
from .ui_handler import UIHandler
from .views.main_window import Ui_MainWindow
from .views.plots import PlotHandler


class MainWindow(QMainWindow, Ui_MainWindow):
    sig_data_filtered = Signal(str)
    sig_data_loaded = Signal()
    sig_filter_column_changed = Signal()
    sig_peaks_updated = Signal(str)
    sig_plot_data_changed = Signal(str)
    sig_results_updated = Signal(str)
    sig_show_message = Signal(str, str)

    def __init__(self) -> None:
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.setWindowTitle("Signal Editor")
        self.setDockNestingEnabled(True)
        self._app_dir: Path = Path.cwd()
        self._data_dir: Path = Path.expanduser(Path("~"))
        self._output_dir: Path = Path(self._app_dir, "output")
        self._output_dir.mkdir(exist_ok=True)
        self.plot = PlotHandler()
        self.data = DataHandler(self)
        self.ui = UIHandler(self, self.plot)
        self.file_info = QFileInfo()
        self.connect_signals()
        self._read_settings()
        # self._add_profiler()
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
        self.menu_info.addAction("Start Profiler", self._start_profiler)
        self.menu_info.addAction("Stop Profiler", self._stop_profiler)

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

        self.btn_apply_filter.clicked.connect(self.handle_apply_filter)
        self.btn_detect_peaks.clicked.connect(self.handle_peak_detection)

        self.btn_browse_output_dir.clicked.connect(self.select_output_location)
        self.btn_load_selection.clicked.connect(self.handle_load_selection)
        self.btn_select_file.clicked.connect(self.select_data_file)
        self.btn_export_to_csv.clicked.connect(lambda: self.export_results("csv"))
        self.btn_export_to_excel.clicked.connect(lambda: self.export_results("excel"))
        self.btn_export_to_text.clicked.connect(lambda: self.export_results("txt"))
        self.btn_compute_results.clicked.connect(self.update_results)

        self.btn_group_plot_view.idClicked.connect(
            self.stacked_hbr_vent.setCurrentIndex
        )

        self.action_exit.triggered.connect(self.closeEvent)
        self.action_select_file.triggered.connect(self.select_data_file)

        self.spin_box_fs.valueChanged.connect(self.data.update_fs)

        self.plot.sig_peaks_edited.connect(self.handle_scatter_clicked)

    @Slot()
    def update_results(self) -> None:
        with pg.BusyCursor():
            self.make_results(self.result_name)

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
            plot_widget.getPlotItem().getViewBox().autoRange()
            plot_widget.getPlotItem().getViewBox().enableAutoRange(
                axis=pg.ViewBox.YAxis, enable=True
            )
            plot_widget.getPlotItem().getViewBox().setAutoVisible(y=True)

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
                xMin=-0.5 * data_length, xMax=1.5 * data_length
            )
            widget.getPlotItem().getViewBox().setXRange(0, data_length)
        self._update_plot_view("hbr")
        self._update_plot_view("ventilation")

    def _update_signal(self, signal_name: SignalName) -> None:
        signal_data = self.data.df.get_column(signal_name).to_numpy(zero_copy_only=True)
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
        peaks_name = f"{signal_name}_{self.data.peaks_suffix}"
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
            peaks: NDArray[np.int32] = getattr(self.data, peaks_name)

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

        self.plot.draw_bpm(
            instant_signal_rate,
            plot_widget=rate_plot_widget,
            signal_name=signal_name,
        )

    @Slot()
    def handle_scatter_clicked(self) -> None:
        signal_name = self.signal_name
        scatter_plot = getattr(self.plot, f"{signal_name}_peaks_scatter")
        peaks_name = f"{signal_name}_{self.data.peaks_suffix}"
        peaks = np.asarray(
            scatter_plot.data["x"],
            dtype=np.int32,
        )
        peaks.sort()
        setattr(self.data, peaks_name, peaks)
        self.data.compute_rate(signal_name)
        self.handle_draw_results(signal_name)

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
        result_file_name = f"results_{result_name}_{self.file_info.baseName()}"
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
            selection_size=self.data.df.height,
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
        identifier = self.get_identifier(result_name)
        data_info = self.get_data_info()
        processing_info = self.get_processing_info()
        statistics = self.data.get_descriptive_stats(result_name)
        results_df = self.data.get_result_df(result_name)
        result_table: QTableView = getattr(self, f"table_view_results_{result_name}")
        self._set_table_model(result_table, PolarsModel(results_df))
        manual_edits = PeakDetectionManualEdited(
            added_peaks=self.plot.added_points[result_name],
            removed_peaks=self.plot.removed_points[result_name],
        )
        additional_data = {
            "rate_bpm_interpolated": getattr(
                self.data,
                f"{result_name}_{self.data.rate_interp_suffix}",
                np.empty(0, dtype=np.float64),
            ),
            "processed_signal": getattr(
                self.data,
                f"{result_name}_{self.data.processed_suffix}",
                np.empty(0, dtype=np.float64),
            ),
            "manual_edits": manual_edits,
        }
        results = Result(
            identifier=identifier,
            info_data_selection=data_info,
            info_data_processing=processing_info,
            statistics=statistics,
            result_data=results_df,
            additional_data=additional_data,
        )
        setattr(self, f"{result_name}_results", results)


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
    # app.setStyle("Fusion")
    app.setStyleSheet(
        qdarkstyle.load_stylesheet(qt_api="pyside6", palette=qdarkstyle.LightPalette)
    )
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
