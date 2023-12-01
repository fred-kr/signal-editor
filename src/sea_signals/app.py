import cProfile
import os
import sys
import types
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl
import pyqtgraph as pg
import qdarkstyle
from loguru import logger
from numpy.typing import NDArray
from pyqtgraph.console import ConsoleWidget
from pyqtgraph.parametertree import ParameterTree
from PySide6 import QtCore
from PySide6.QtCore import (
    QByteArray,
    QDate,
    QObject,
    QSettings,
    QStandardPaths,
    Qt,
    Signal,
    Slot,
)
from PySide6.QtGui import QCloseEvent, QStandardItemModel
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QHeaderView,
    QMainWindow,
    QSizePolicy,
    QVBoxLayout,
)

from .custom_types import (
    InfoProcessingParams,
    InfoWorkingData,
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
    Identifier,
    InfoTableModel,
    Results,
)
from .models.io import parse_file_name
from .models.peaks import ElgendiPPGPeaks
from .views.main_window import Ui_MainWindow
from .views.plots import PlotManager


class MainWindow(QMainWindow, Ui_MainWindow):
    sig_filter_column_changed = Signal()
    sig_lazy_ready = Signal(str)
    sig_data_loaded = Signal()
    sig_data_filtered = Signal(str)
    sig_peaks_updated = Signal(str)
    sig_plot_data_changed = Signal(str)
    sig_init_complete = Signal()
    sig_prepare_new_data = Signal()

    def __init__(
        self,
    ) -> None:
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.setWindowTitle("Sea Signals")
        self.setDockNestingEnabled(True)
        self.plot = PlotManager()
        self.connect_signals()
        self._read_settings()
        self._add_profiler()
        self.sig_init_complete.emit()

    def _add_profiler(self) -> None:
        self.menu_info.addAction("Start Profiler", self._start_profiler)
        self.menu_info.addAction("Stop Profiler", self._stop_profiler)

    @Slot()
    def _init_complete(self) -> None:
        self.widgets = UIHandler(self, self.plot)
        self.widgets.sig_apply_filter.connect(self.handle_apply_filter)
        self.widgets.sig_peak_detection_inputs.connect(self.handle_peak_detection)

    def get_signal_name(self) -> SignalName:
        return "hbr" if self.stacked_hbr_vent.currentIndex() == 0 else "ventilation"

    def connect_signals(self) -> None:
        self.sig_init_complete.connect(self._init_complete)

        self.btn_load_selection.clicked.connect(self.handle_load_selection)
        self.btn_select_file.clicked.connect(self.handle_select_file)

        self.action_select_file.triggered.connect(self.handle_select_file)
        self.action_exit.triggered.connect(self.closeEvent)

        self.sig_lazy_ready.connect(self.on_filter_column_changed)
        self.sig_data_loaded.connect(self.handle_plot_draw)
        self.sig_data_loaded.connect(self.handle_table_view_data)
        self.sig_peaks_updated.connect(self.handle_draw_results)
        self.sig_plot_data_changed.connect(self._update_plot_view)
        self.sig_data_filtered.connect(self.handle_table_view_data)
        self.sig_prepare_new_data.connect(self.plot.reset_plots)

        self.plot.sig_peaks_edited.connect(self.handle_scatter_clicked)

    @Slot()
    def handle_select_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select File",
            QStandardPaths.writableLocation(
                QStandardPaths.StandardLocation.HomeLocation
            ),
            # "E:/dev-home/_readonly_datastore_/2023_HB_Vent_EDF/11092023_PA26_16°C_HB_Vent_16_25°C.edf",
            filter="EDF (*.edf);;CSV (*.csv);;TXT (*.txt);;Feather (*.feather)",
            selectedFilter="EDF (*.edf)",
        )
        if path:
            self.table_data_preview.setModel(QStandardItemModel())
            self.table_data_info.setModel(QStandardItemModel())
            self.sig_prepare_new_data.emit()

            # Show the active file's name in the line edit field
            self.line_edit_active_file.setText(Path(path).name)

            # Block any updates during the file load
            self.combo_box_filter_column.blockSignals(True)

            # Initialize a DataHandler instance and use it to read the data into a polars LazyFrame, get min/max of each column to set range limits for filtering
            sampling_rate = self.spin_box_fs.value()
            self.dm = DataHandler(file_path=path, sampling_rate=sampling_rate)
            self.dm.lazy_read()
            self.dm.get_min_max()

            # Set the UI elements to enabled and update them with the current data
            self.group_box_subset_params.setEnabled(True)
            self.group_box_subset_params.setChecked(True)

            self.container_file_info.setEnabled(True)
            parsed_date, parsed_id = parse_file_name(Path(path).name)
            self.date_edit_file_info.setDate(
                QDate(parsed_date.year, parsed_date.month, parsed_date.day)
            )
            self.line_edit_subject_id.setText(parsed_id)

            self.btn_load_selection.setEnabled(True)

            viable_filter_columns = ("index", "time_s", "temperature")
            self.combo_box_filter_column.clear()
            self.combo_box_filter_column.addItems(viable_filter_columns)
            self.combo_box_filter_column.setCurrentIndex(0)
            self.combo_box_filter_column.currentTextChanged.connect(
                self.on_filter_column_changed
            )
            self.combo_box_filter_column.blockSignals(False)

            self.sig_lazy_ready.emit(self.combo_box_filter_column.currentText())

    @Slot(str)
    def on_filter_column_changed(self, text: str) -> None:
        lower = self.dm.min_max_mapping[text]["min"]
        upper = self.dm.min_max_mapping[text]["max"]

        if text == "index":
            self.dbl_spin_box_subset_min.setDecimals(0)
            self.dbl_spin_box_subset_max.setDecimals(0)
        elif text == "time_s":
            self.dbl_spin_box_subset_min.setDecimals(4)
            self.dbl_spin_box_subset_max.setDecimals(4)
        elif text == "temperature":
            self.dbl_spin_box_subset_min.setDecimals(1)
            self.dbl_spin_box_subset_max.setDecimals(1)

        self.dbl_spin_box_subset_min.setMinimum(lower)
        self.dbl_spin_box_subset_min.setMaximum(upper)
        self.dbl_spin_box_subset_min.setValue(lower)

        self.dbl_spin_box_subset_max.setMinimum(lower)
        self.dbl_spin_box_subset_max.setMaximum(upper)
        self.dbl_spin_box_subset_max.setValue(upper)

    @Slot()
    def handle_load_selection(self) -> None:
        try:
            self.sig_prepare_new_data.emit()
            filter_col = self.combo_box_filter_column.currentText()
            lower = self.dbl_spin_box_subset_min.value()
            upper = self.dbl_spin_box_subset_max.value()

            if self.group_box_subset_params.isChecked():
                # Because the temperature column values are not monotonically increasing/decreasing in value, just selecting all values where temperature > x leads to lots of missing values around the threshold whenever the temperature value dips below this limit.
                # To counteract this, the filtering looks as follows:
                # 1. Create a helper column thats set to 1 if the temperature is above the lower threshold, 0 otherwise
                # 2. Create a second helper column thats set to 1 if the temperature is above the upper threshold, 0 otherwise
                # 3. Filter the data frame to only keep the rows where the helper column is 1 and the second helper column is 0, i.e. between the first occurence of the lower threshold and the first occurence of the upper threshold
                # 4. The resulting dataframe wont have a clean lower limit cutoff, but there also won't be any missing values that mess up the plot
                lf_filtered = (
                    self.dm.lazy.with_columns(
                        above_lower=(
                            pl.when(pl.col(filter_col).ge(pl.lit(lower)))
                            .then(1)
                            .otherwise(0)
                        ),
                        above_upper=(
                            pl.when(pl.col(filter_col).ge(pl.lit(upper)))
                            .then(1)
                            .otherwise(0)
                        ),
                    )
                    .with_columns(
                        [
                            pl.col("above_lower").cum_max().alias("cum_above_lower"),
                            pl.col("above_upper").cum_max().alias("cum_above_upper"),
                        ]
                    )
                    .filter(
                        (pl.col("cum_above_lower") == 1)
                        & (pl.col("cum_above_upper") == 0)
                    )
                )
                self.dm.data = (
                    lf_filtered.select(
                        pl.all().exclude(
                            "above_lower",
                            "above_upper",
                            "cum_above_lower",
                            "cum_above_upper",
                        )
                    )
                    .collect()
                    .shrink_to_fit(in_place=True)
                )
            else:
                self.dm.data = self.dm.lazy.collect().shrink_to_fit(in_place=True)

            logger.info(
                f"Loaded {self.dm.data.shape[0]} rows from {self.dm.data.shape[1]} columns, size {self.dm.data.estimated_size('mb'):.2f} MB"
            )
            self.sig_data_loaded.emit()
        except Exception as e:
            logger.error(e)
            return

    @Slot(str)
    def _update_plot_view(self, signal_name: str) -> None:
        getattr(self.plot, f"{signal_name}_plot_widget").autoRange()
        getattr(self.plot, f"{signal_name}_plot_widget").enableAutoRange(y=True)
        # getattr(self.plot, f"bpm_{signal_name}_plot_widget").autoRange()

    @Slot()
    def handle_plot_draw(self) -> None:
        signal_name = self.get_signal_name()
        if not hasattr(self.plot, "ventilation_signal_line") and not hasattr(
            self.plot, "hbr_signal_line"
        ):
            self.plot.draw_signal(
                sig=self.dm.data.get_column("hbr").to_numpy(zero_copy_only=True),
                plot_widget=self.plot.hbr_plot_widget,
                signal_name="hbr",
            )
            self.plot.draw_signal(
                sig=self.dm.data.get_column("ventilation").to_numpy(
                    zero_copy_only=True
                ),
                plot_widget=self.plot.ventilation_plot_widget,
                signal_name="ventilation",
            )
            self.plot.hbr_plot_widget.setXRange(0, self.dm.data.shape[0])
            self.plot.ventilation_plot_widget.setXRange(0, self.dm.data.shape[0])
        else:
            self.plot.draw_signal(
                sig=self.dm.data.get_column(signal_name).to_numpy(zero_copy_only=True),
                plot_widget=getattr(self.plot, f"{signal_name}_plot_widget"),
                signal_name=signal_name,
            )
            self.sig_plot_data_changed.emit(signal_name)

    @Slot()
    def handle_table_view_data(self) -> None:
        # model = PolarsModel(self.dm.data)
        df_head = self.dm.data.head(15).shrink_to_fit(in_place=True)
        df_tail = self.dm.data.tail(15).shrink_to_fit(in_place=True)
        df_description = self.dm.data.describe(None).shrink_to_fit(in_place=True)

        model = CompactDFModel(df_head=df_head, df_tail=df_tail)
        self.table_data_preview.setModel(model)
        self.table_data_preview.horizontalHeader().setDefaultAlignment(
            Qt.AlignmentFlag.AlignLeft
        )
        self.table_data_preview.verticalHeader().setVisible(False)
        self.table_data_preview.resizeColumnsToContents()
        for col in range(model.columnCount()):
            self.table_data_preview.horizontalHeader().setSectionResizeMode(
                col, QHeaderView.ResizeMode.Stretch
            )

        info = InfoTableModel(df_description)
        self.table_data_info.setModel(info)
        self.table_data_info.horizontalHeader().setDefaultAlignment(
            Qt.AlignmentFlag.AlignLeft
        )
        self.table_data_info.resizeColumnsToContents()
        for col in range(info.columnCount()):
            self.table_data_info.horizontalHeader().setSectionResizeMode(
                col, QHeaderView.ResizeMode.Stretch
            )

    @Slot(object)
    def handle_apply_filter(self, filter_params: SignalFilterParameters) -> None:
        logger.debug(f"Received filter params: {filter_params} in handle_apply_filter")
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

    @Slot(object)
    def handle_peak_detection(self, peak_params: PeaksPPGElgendi) -> None:
        signal_name = self.get_signal_name()
        peak_detection_method = self.combo_box_peak_detection_method.currentText()
        peak_detection_method = cast(PeakDetectionMethod, peak_detection_method)

        self.dm.find_peaks(
            signal_name=signal_name,
            peak_find_method=peak_detection_method,
            **peak_params,
        )

        self.plot.draw_peaks(
            pos_x=self.dm.__getattribute__(f"{signal_name}_peaks"),
            pos_y=self.dm.data.get_column(f"processed_{signal_name}").to_numpy(
                zero_copy_only=True
            )[self.dm.__getattribute__(f"{signal_name}_peaks")],
            plot_widget=getattr(self.plot, f"{signal_name}_plot_widget"),
            signal_name=signal_name,
        )
        self.sig_peaks_updated.emit(signal_name)

    @Slot(str)
    def handle_draw_results(self, signal_name: SignalName) -> None:
        instant_signal_rate: NDArray[np.float32] = getattr(
            self.dm, f"{signal_name}_rate"
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
            self.plot.__getattribute__(f"{signal_name}_peaks_scatter").data["x"],
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
        settings = QSettings("AWI", "Sea Signals")
        geometry: QByteArray = settings.value("geometry", QByteArray())
        if geometry.size():
            self.restoreGeometry(geometry)

    def _write_settings(self) -> None:
        settings = QSettings("AWI", "Sea Signals")
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

    def make_results(self, signal_name: SignalName) -> Results:
        identifier = self.widgets.get_identifier()
        working_data_metadata = self.widgets.get_working_metadata()
        processing_metadata = self.widgets.get_processing_metadata()
        computed = self.dm.compute_results(signal_name)
        results_df = self.dm.make_results_df(
            peaks=getattr(self.dm, f"{signal_name}_peaks"),
            rate=computed["signal_rate_from_peaks"].round(1),
        )
        return Results(
            identifier=identifier,
            working_data_metadata=working_data_metadata,
            processing_metadata=processing_metadata,
            computed=computed,
            processed_data=results_df,
        )


class UIHandler(QObject):
    sig_filter_inputs_ready = Signal()
    sig_preprocess_pipeline_ready = Signal(str)
    sig_ready_for_cleaning = Signal()
    sig_apply_filter = Signal(dict)
    sig_peak_detection_inputs = Signal(dict)

    def __init__(self, window: MainWindow, plot: PlotManager) -> None:
        super(UIHandler, self).__init__()
        self.window = window
        self.plot = plot
        self.setup_widgets()
        self.setup_toolbars()
        self.connect_signals()

    def setup_widgets(self) -> None:
        # Signal Filtering
        self.window.container_standard_filter_method.setEnabled(True)
        self.window.combo_box_preprocess_pipeline.setCurrentIndex(0)
        self.window.container_signal_filter_inputs.setEnabled(False)
        self.window.combo_box_filter_method.setCurrentIndex(0)
        self.window.combo_box_standardizing_method.setCurrentIndex(0)

        # Peak Detection
        self.window.combo_box_peak_detection_method.setCurrentIndex(0)
        self.window.stacked_widget_peak_detection.setCurrentIndex(0)
        self.create_peak_detection_trees()

        # File Info
        self.window.container_file_info.setEnabled(False)

        # Statusbar
        self.create_statusbar()

        # Plots
        self.window.stacked_hbr_vent.setCurrentIndex(0)
        self.window.btn_group_plot_view.setId(self.window.btn_view_hbr, 0)
        self.window.btn_group_plot_view.setId(self.window.btn_view_vent, 1)
        self.create_plot_widgets()

        # Console
        self.create_console_widget()

    def connect_signals(self) -> None:
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
        self.window.btn_find_peaks.clicked.connect(self.emit_peak_detection_inputs)

        self.window.action_open_console.triggered.connect(self.show_console_widget)
        self.window.tabWidget.currentChanged.connect(self.handle_tab_changed)

    def setup_toolbars(self) -> None:
        self.window.toolbar_plots.setVisible(False)

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
        if index == 1:
            self.window.toolbar_plots.setVisible(True)
            self.window.toolbar_plots.setEnabled(True)
        else:
            self.window.toolbar_plots.setVisible(False)
            self.window.toolbar_plots.setEnabled(False)

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
        self.plot.hbr_plot_widget.plotItem.scene().sigMouseMoved.connect(
            lambda pos: self.update_temperature_label("hbr", pos)
        )
        self.plot.ventilation_plot_widget.plotItem.scene().sigMouseMoved.connect(
            lambda pos: self.update_temperature_label("ventilation", pos)
        )

    @Slot(QtCore.QPointF)
    def update_temperature_label(
        self, signal_name: SignalName, pos: QtCore.QPointF
    ) -> None:
        data_pos = int(
            getattr(self.plot, f"{signal_name}_plot_widget")
            .plotItem.vb.mapSceneToView(pos)
            .x()
        )
        try:
            temp_value = self.window.dm.data.get_column("temperature").to_numpy(
                zero_copy_only=True
            )[data_pos]
        except Exception:
            temp_value = self.window.dm.data.get_column("temperature").to_numpy(
                zero_copy_only=True
            )[-1]
        if signal_name == "hbr":
            self.temperature_label_hbr.setText(f"Temperature: {temp_value:.1f}°C")
        else:
            self.temperature_label_ventilation.setText(
                f"Temperature: {temp_value:.1f}°C"
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
        ]
        namespace: dict[str, types.ModuleType | MainWindow] = {
            "self": self.window,
            "pg": pg,
            "np": np,
            "pl": pl,
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
        self.window.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea, self.console_dock
        )
        self.console_dock.setVisible(False)

    @Slot()
    def show_console_widget(self) -> None:
        self.console_dock.setVisible(not self.console_dock.isVisible())

    def create_statusbar(self) -> None:
        self.window.statusbar = self.window.statusBar()
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
        file_name = self.window.line_edit_active_file.text()
        subject_id = self.window.line_edit_subject_id.text()
        date_of_recording = self.window.date_edit_file_info.date().toPython()
        oxygen_condition = self.window.combo_box_oxygen_condition.currentText()
        return Identifier(
            file_name=file_name,
            subject_id=subject_id,
            date_of_recording=date_of_recording,
            oxygen_condition=oxygen_condition,
        )

    def get_working_metadata(self) -> InfoWorkingData:
        subset_column = self.window.combo_box_filter_column.currentText()
        subset_lower_bound = self.window.dbl_spin_box_subset_min.value()
        subset_upper_bound = self.window.dbl_spin_box_subset_max.value()
        n_samples = self.window.dm.data.shape[0]
        return InfoWorkingData(
            subset_column=subset_column,
            subset_lower_bound=subset_lower_bound,
            subset_upper_bound=subset_upper_bound,
            n_samples=n_samples,
        )

    def get_processing_metadata(self) -> InfoProcessingParams:
        sampling_rate = self.window.spin_box_fs.value()
        preprocess_pipeline = self.get_preprocess_pipeline()
        filter_parameters = self.get_filter_settings()
        standardization_method = self.get_standardizing_method()
        peak_detection_method = (
            self.window.combo_box_peak_detection_method.currentText()
        )
        peak_method_parameters = self.get_peak_detection_inputs()
        return InfoProcessingParams(
            sampling_rate=sampling_rate,
            preprocess_pipeline=preprocess_pipeline,
            filter_parameters=filter_parameters,
            standardization_method=standardization_method,
            peak_detection_method=peak_detection_method,
            peak_method_parameters=peak_method_parameters,
        )


def main(dev_mode: bool = False) -> None:
    if dev_mode:
        os.environ["QT_LOGGING_RULES"] = "qt.pyside.libpyside.warning=true"

    pg.setConfigOptions(
        useOpenGL=True,
        enableExperimental=True,
        segmentedLineMode="on",
        background="black",
        antialias=False,
    )
    # logger.add(
    # ,
    # format=(
    # "<magenta>{time:YYYY-MM-DD HH:mm:ss.SSS}</magenta> | "
    # "<level>{level: <8}</level> | "
    # "<yellow>{message}</yellow> | "
    # "source: <blue>{name}</blue>.<cyan>{function}()</cyan>, line: <green>{line}</green> | "
    # "pid: <red>{process}</red> | "
    # "tid: <red>{thread}</red>"
    # ),
    # backtrace=True,
    # diagnose=True,
    # level="DEBUG",
    # )
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
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
