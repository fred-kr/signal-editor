from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Literal, Unpack, cast

import neurokit2 as nk
import numpy as np
import polars as pl
import pyqtgraph as pg
import wfdb.processing
from loguru import logger
from numpy.typing import NDArray
from PySide6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QObject,
    QPersistentModelIndex,
    Qt,
    Signal,
    Slot,
)
from PySide6.QtWidgets import QWidget

from ..type_aliases import (
    PeakDetectionElgendiPPG,
    PeakDetectionLocalMaxima,
    PeakDetectionNeurokit2,
    PeakDetectionPantompkins,
    PeakDetectionParameters,
    PeakDetectionProMAC,
    PeakDetectionXQRS,
    Pipeline,
    ScaleMethod,
    SignalName,
    StandardizeParameters,
)
from .filters import (
    SignalFilterParameters,
    auto_correct_fir_length,
    filter_custom,
    filter_elgendi,
    scale_signal,
)
from .io import read_edf
from .peaks import (
    find_local_peaks,
    find_ppg_peaks_elgendi,
    neurokit2_find_peaks,
)

if TYPE_CHECKING:
    from ..app import MainWindow


@dataclass(slots=True, kw_only=True)
class Peaks:
    hbr: NDArray[np.int32] = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    ventilation: NDArray[np.int32] = field(
        default_factory=lambda: np.empty(0, dtype=np.int32)
    )

    def __getitem__(self, key: SignalName) -> NDArray[np.int32]:
        return getattr(self, key, np.empty(0, dtype=np.int32))

    def update(self, name: SignalName, peaks: NDArray[np.int32]) -> None:
        setattr(self, name, peaks)

    def add_peak(self, name: SignalName, peak: int) -> None:
        peaks = self[name]
        insert_index = np.searchsorted(peaks, peak)
        peaks = np.insert(peaks, insert_index, peak)
        self.update(name, peaks)

    def remove_peak(self, name: SignalName, peak: int) -> None:
        peaks = self[name]
        if peak not in peaks:
            return
        peaks = np.delete(peaks, np.where(peaks == peak))
        self.update(name, peaks)

    def diff(self, name: SignalName) -> NDArray[np.int32]:
        return np.ediff1d(self[name], to_begin=np.array([0]))


@dataclass(slots=True, kw_only=True)
class Rate:
    """
    Holds the non-interpolated rate data for each signal. Results come from the `neurokit2.signal_rate` function.
    """

    hbr: NDArray[np.float64] = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    ventilation: NDArray[np.float64] = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )

    def __getitem__(self, key: SignalName) -> NDArray[np.float64]:
        return getattr(self, key, np.empty(0, dtype=np.float64))

    def update(self, name: SignalName, rate: NDArray[np.float64]) -> None:
        setattr(self, name, rate)


@dataclass(slots=True, kw_only=True)
class ResultDFs:
    hbr: pl.DataFrame = field(default_factory=lambda: pl.DataFrame())
    ventilation: pl.DataFrame = field(default_factory=lambda: pl.DataFrame())

    def __getitem__(self, key: SignalName) -> pl.DataFrame:
        return getattr(self, key, pl.DataFrame())

    def update(self, name: SignalName, df: pl.DataFrame) -> None:
        setattr(self, name, df)


@dataclass(slots=True, kw_only=True, frozen=True)
class PeakStatistics:
    pass


@dataclass(slots=True, kw_only=True, frozen=True)
class PeakIntervalStatistics:
    mean: np.float_
    median: np.float_
    std: np.float_
    mad: np.float_
    var: np.float_


@dataclass(slots=True, kw_only=True, frozen=True)
class RateStatistics:
    mean: np.float_
    median: np.float_
    std: np.float_
    mad: np.float_
    var: np.float_


@dataclass(slots=True, kw_only=True, frozen=True)
class DescriptiveStatistics:
    peaks: PeakStatistics
    peak_intervals: PeakIntervalStatistics
    rate: RateStatistics


def standardize(
    sig: NDArray[np.float64],
    method: ScaleMethod,
    window_size: int = 500,
    rolling_window: bool = True,
) -> NDArray[np.float64]:
    if method == "None":
        return sig
    is_robust = method == "mad"

    return scale_signal(sig, robust=is_robust, window_size=window_size, rolling_window=rolling_window).to_numpy()


class DataHandler(QObject):
    """
    Handles storing and operating on data.
    """

    sig_dh_new_data = Signal()
    sig_dh_error = Signal(str)
    sig_dh_peaks_updated = Signal(str)
    sig_dh_rate_updated = Signal(str)

    def __init__(
        self,
        parent: "MainWindow",
    ) -> None:
        super().__init__(parent)
        self._parent = parent
        self.df: pl.DataFrame = pl.DataFrame()
        self._sampling_rate: int = parent.spin_box_fs.value()
        self.peaks: Peaks = Peaks()
        self.rate: Rate = Rate()
        self.result_dfs: ResultDFs = ResultDFs()
        self.processed_suffix = "processed"
        self.peaks_suffix = "peaks"
        self.rate_no_interp_suffix = "rate"
        self.rate_interp_suffix = "rate_interpolated"

    @property
    def fs(self) -> int:
        return self._sampling_rate

    @fs.setter
    def fs(self, value: int | float) -> None:
        self._sampling_rate = int(value)

    @Slot(int)
    def update_fs(self, value: int) -> None:
        self.fs = value

    def read(self, path: str | Path) -> None:
        path = Path(path)
        suffix = path.suffix
        if suffix not in {".csv", ".txt", ".edf", ".feather", ".xlsx", ".pkl"}:
            info_msg = "Currently only .csv, .txt, .xlsx, .feather, .pkl and .edf files are supported"
            self._parent.sig_show_message.emit(info_msg, "info")
            return

        if suffix == ".csv":
            self._lazy_df = pl.scan_csv(path)
        elif suffix == ".edf":
            self._lazy_df, self.meas_date, self.sampling_rate = read_edf(
                path.as_posix()
            )
        elif suffix == ".feather":
            self._lazy_df = pl.scan_ipc(path)
        elif suffix == ".txt":
            self._lazy_df = pl.scan_csv(path, separator="\t")
        elif suffix == ".xlsx":
            self.df = pl.read_excel(path)
        elif suffix == ".pkl":
            self._parent.restore_state(path)
            return
        else:
            raise NotImplementedError(f"File type `{suffix}` not supported")

        self.df = self._lazy_df.collect()
        self.calc_minmax()

    @staticmethod
    def get_slice_indices(
        lf: pl.LazyFrame, filter_col: str, lower: float, upper: float
    ) -> tuple[int, int]:
        lf = lf.with_columns(pl.col("temperature").round(1))
        b1 = (
            lf.sort(pl.col(filter_col), maintain_order=True)
            .filter(pl.col(filter_col) >= pl.lit(lower))
            .collect()
            .get_column("index")[0]
        )
        logger.debug(f"b1: {b1}")
        b2 = (
            lf.sort(pl.col(filter_col), maintain_order=True)
            .filter(pl.col(filter_col) >= pl.lit(upper))
            .collect()
            .get_column("index")[0]
        )
        logger.debug(f"b2: {b2}")
        l_index = min(b1, b2)
        u_index = max(b1, b2)

        logger.debug(f"{l_index=}, {u_index=}")
        return l_index, u_index

    def get_subset(self, subset_col: str, lower: float, upper: float) -> None:
        lf = self._lazy_df
        if (
            lower == self.minmax_map[subset_col]["min"]
            and upper == self.minmax_map[subset_col]["max"]
        ):
            return
        if subset_col in {"time_s", "temperature"}:
            lower, upper = self.get_slice_indices(lf, subset_col, lower, upper)
        self.df = lf.filter(pl.col("index").is_between(lower, upper)).collect()

    def calc_minmax(self, col_names: Iterable[str] | None = None) -> None:
        if col_names is None:
            col_names = ["index", "time_s", "temperature"]

        self.minmax_map = {
            name: {
                "min": self.df.get_column(name).min(),
                "max": self.df.get_column(name).max(),
            }
            for name in col_names
        }

    def run_preprocessing(
        self,
        name: SignalName,
        pipeline: Pipeline,
        filter_params: SignalFilterParameters,
        standardize_params: StandardizeParameters,
    ) -> None:
        if not hasattr(self, "df"):
            error_msg = "Data not loaded. Please load data first."
            self.sig_dh_error.emit(error_msg)
            return

        with pg.ProgressDialog("Filtering...", 0, 100, cancelText=None, wait=0) as dlg:
            processed_name = f"{name}_{self.processed_suffix}"

            if processed_name in self.df.columns:
                self.df = self.df.drop(processed_name)

            sig: NDArray[np.float64] = (
                self.df.get_column(name).cast(pl.Float64).to_numpy()
            )
            dlg.setValue(20)
            filter_method = filter_params.get("method", "None")
            standardize_method = standardize_params.get("method", "None")
            standardize_window_size = standardize_params.get("window_size")
            uses_rolling_window = standardize_params.get("rolling_window")

            if pipeline == "custom":
                if filter_method == "None":
                    filtered = sig
                elif filter_method == "fir":
                    filtered = auto_correct_fir_length(sig, self.fs, **filter_params)
                else:
                    filtered = filter_custom(sig, self.fs, **filter_params)
            elif pipeline == "ppg_elgendi":
                filtered = filter_elgendi(sig, self.fs)
            else:
                msg = f"Pipeline {pipeline} not implemented. Choose 'custom' or 'ppg_elgendi'."
                self._parent.sig_show_message.emit(msg, "info")
                return

            dlg.setValue(70)
            standardized = standardize(
                filtered,
                method=standardize_method,
                window_size=standardize_window_size,
                rolling_window=uses_rolling_window,
            )
            dlg.setValue(99)
            self.df.hstack([pl.Series(processed_name, standardized)], in_place=True)
            dlg.setValue(100)

    def run_peak_detection(
        self, name: SignalName, **kwargs: Unpack[PeakDetectionParameters]
    ) -> None:
        processed_name = f"{name}_{self.processed_suffix}"
        sig = self.df.get_column(processed_name).to_numpy(zero_copy_only=True)
        method = kwargs["method"]
        method_params = kwargs["input_values"]

        if method == "elgendi_ppg":
            method_params = cast(PeakDetectionElgendiPPG, method_params)
            peaks = find_ppg_peaks_elgendi(sig, self.fs, **method_params)
        elif method == "local":
            method_params = cast(PeakDetectionLocalMaxima, method_params)
            peaks = find_local_peaks(sig, **method_params)
        elif method == "neurokit2":
            method_params = cast(PeakDetectionNeurokit2, method_params)
            peaks = neurokit2_find_peaks(sig, self.fs, method, **method_params)
        elif method == "pantompkins":
            method_params = cast(PeakDetectionPantompkins, method_params)
            peaks = neurokit2_find_peaks(sig, self.fs, method, **method_params)
        elif method == "promac":
            method_params = cast(PeakDetectionProMAC, method_params)
            peaks = neurokit2_find_peaks(sig, self.fs, method, **method_params)
        elif method == "wfdb_xqrs":
            method_params = cast(PeakDetectionXQRS, method_params)
            correction_params = method_params["corrections"]
            xqrs = wfdb.processing.XQRS(sig, self.fs)
            xqrs.detect(
                sampfrom=method_params.get("sampfrom", 0),
                sampto=method_params.get("sampto", "end"),
            )
            peaks = np.asarray(
                wfdb.processing.correct_peaks(
                    sig=sig, peak_inds=xqrs.qrs_inds, **correction_params
                ),
                dtype=np.int32,
            )
        else:
            error_msg = f"Peak detection method {method} not implemented. Choose one of 'elgendi_ppg', 'local', 'neurokit2', 'pantompkins', 'promac' or 'wfdb_xqrs'."
            self._parent.sig_show_message.emit(error_msg, "warning")
            return

        self.peaks.update(name, peaks)
        self.compute_rate(name)

    def compute_rate(self, name: SignalName) -> None:
        name_rate_interp = f"{name}_{self.rate_interp_suffix}"

        peaks = self.peaks[name]
        rate_interp = np.asarray(
            nk.signal_rate(
                peaks=peaks,
                sampling_rate=self.fs,
                desired_length=self.df.height,
                interpolation_method="monotone_cubic",
                show=False,
            ),
            dtype=np.float64,
        )
        rate_no_interp = np.asarray(
            nk.signal_rate(
                peaks=peaks,
                sampling_rate=self.fs,
                desired_length=None,
                interpolation_method="monotone_cubic",
                show=False,
            ),
            dtype=np.float64,
        )

        if name_rate_interp in self.df.columns:
            self.df = self.df.drop(name_rate_interp)
        self.df.hstack(
            [pl.Series(name_rate_interp, rate_interp, dtype=pl.Float64).round(1)],
            in_place=True,
        )

        self.rate.update(name, rate_no_interp)

    def get_descriptive_stats(self, name: SignalName) -> DescriptiveStatistics:
        rate_no_interp = self.rate[name]
        diffs = self.peaks.diff(name)
        return DescriptiveStatistics(
            peaks=PeakStatistics(),
            peak_intervals=PeakIntervalStatistics(
                mean=np.mean(diffs).round(2),
                median=np.median(diffs).round(2),
                std=np.std(diffs).round(2),
                mad=np.nanmedian(np.abs(diffs - np.nanmedian(diffs))).round(2),
                var=np.var(diffs).round(2),
            ),
            rate=RateStatistics(
                mean=np.mean(rate_no_interp).round(2),
                median=np.median(rate_no_interp).round(2),
                std=np.std(rate_no_interp).round(2),
                mad=np.nanmedian(
                    np.abs(rate_no_interp - np.nanmedian(rate_no_interp))
                ).round(2),
                var=np.var(rate_no_interp).round(2),
            ),
        )

    def compute_result_df(self, name: SignalName) -> None:
        peaks = self.peaks[name]

        result_df = (
            self.df.lazy()
            .select(
                pl.col("time_s").gather(peaks).round(4),
                pl.Series(
                    "peak_index", self.df.get_column("index").gather(peaks), pl.Int32
                ),
                pl.Series("peak_intervals", self.peaks.diff(name), pl.Int32),
                pl.col("temperature").gather(peaks).round(1),
                pl.Series("rate_bpm", self.rate[name], pl.Float64).round(1),
            )
            .collect()
        )
        self.result_dfs.update(name, result_df)

    def get_result_df(self, name: SignalName, compute: bool = True) -> pl.DataFrame:
        if compute:
            self.compute_result_df(name)
        return self.result_dfs[name]

    def restore_peaks(self, peaks: Peaks) -> None:
        self.peaks = peaks

    def restore_rate(self, rate: Rate) -> None:
        self.rate = rate

    def restore_df(self, df: pl.DataFrame) -> None:
        self.df = df
        if all(
            col in df.columns
            for col in ["index", "time_s", "temperature", "hbr", "ventilation"]
        ):
            self._lazy_df = df.lazy().select(
                pl.col("index", "time_s", "temperature", "hbr", "ventilation")
            )
        else:
            self._lazy_df = df.lazy()
        self.calc_minmax()

    def restore_result_dfs(self, result_dfs: ResultDFs) -> None:
        self.result_dfs = result_dfs

    @dataclass(slots=True, kw_only=True)
    class DataState:
        df: pl.DataFrame
        result_df: ResultDFs = field(init=False)
        peaks: Peaks = field(init=False)
        rate: Rate = field(init=False)

    def get_state(self) -> DataState:
        state = self.DataState(df=self.df.clone())
        state.result_df = self.result_dfs
        state.peaks = self.peaks
        state.rate = self.rate
        return state

    def restore_state(self, state: DataState) -> None:
        self.restore_df(state.df)
        self.restore_result_dfs(state.result_df)
        self.restore_peaks(state.peaks)
        self.restore_rate(state.rate)


class CompactDFModel(QAbstractTableModel):
    """
    A model for displaying a `polars.DataFrame` in the same way as its done in the console, i.e. only column name, type, head and tail.
    ```python
    shape: (250, 2)
    ┌─────────┬─────────┐
    │ a (i32) ┆ b (i32) │
    ╞═════════╪═════════╡
    │ 123     ┆ 1142    │
    │ 325     ┆ 1020    │
    │ 288     ┆ 1248    │
    │ 289     ┆ 1697    │
    │ …       ┆ …       │
    │ 194     ┆ 1884    │
    │ 36      ┆ 1348    │
    │ 324     ┆ 1640    │
    │ 276     ┆ 1527    │
    └─────────┴─────────┘
    ```
    """

    def __init__(
        self,
        df_head: pl.DataFrame,
        df_tail: pl.DataFrame,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._columns = [
            (col, str(dtype))
            for col, dtype in zip(df_head.columns, df_head.dtypes, strict=True)
        ]
        self._data = (
            df_head.to_numpy().tolist()
            + [["..."] * len(self._columns)]
            + df_tail.to_numpy().tolist()
        )

    def rowCount(
        self, parent: QModelIndex | QPersistentModelIndex | None = None
    ) -> int:
        return len(self._data)

    def columnCount(
        self, parent: QModelIndex | QPersistentModelIndex | None = None
    ) -> int:
        return len(self._columns)

    def data(
        self,
        index: QModelIndex | QPersistentModelIndex,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if not index.isValid():
            return None

        row, column = index.row(), index.column()

        if role == Qt.ItemDataRole.DisplayRole:
            col_name = self._columns[column][0]
            if self._data[row][column] == "...":
                return "..."
            if "index" in col_name or "peak" in col_name:
                return f"{int(self._data[row][column]):_}"
            elif "time" in col_name:
                return f"{float(self._data[row][column]):_.5f}"
            elif "rate" in col_name:
                return f"{int(self._data[row][column]):_}"
            elif (
                "temp" not in col_name
                and "hb" in col_name
                or "temp" not in col_name
                and "vent" in col_name
            ):
                return f"{float(self._data[row][column]):.4f}"
            elif "temp" in col_name:
                return f"{float(self._data[row][column]):.1f}"
            return str(self._data[row][column])

        return None

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                column_name, column_type = self._columns[section]
                return f"{column_name} ({column_type})"
            if orientation == Qt.Orientation.Vertical:
                return str(section)

        return None


class PolarsModel(QAbstractTableModel):
    """
    A model for displaying polars data in a QTableView.
    """

    def __init__(self, dataframe: pl.DataFrame, parent: QWidget | None = None):
        QAbstractTableModel.__init__(self, parent)
        self._dataframe = dataframe

    def rowCount(
        self, parent: QModelIndex | QPersistentModelIndex = QModelIndex()
    ) -> int:
        return self._dataframe.shape[0] if parent == QModelIndex() else 0

    def columnCount(
        self, parent: QModelIndex | QPersistentModelIndex = QModelIndex()
    ) -> int:
        return self._dataframe.shape[1] if parent == QModelIndex() else 0

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return self._dataframe.columns[section]

            if orientation == Qt.Orientation.Vertical:
                return f"{section}"

        return None

    def data(
        self,
        index: QModelIndex | QPersistentModelIndex,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if not index.isValid():
            return None

        column = index.column()
        row = index.row()

        if role == Qt.ItemDataRole.DisplayRole:
            col_name = self._dataframe.columns[column]

            if "index" in col_name or "peak" in col_name:
                idx = self._dataframe[row, column]
                return f"{int(idx)}"
            if "time" in col_name:
                time_s = self._dataframe[row, column]
                return f"{time_s:.4f}"
            elif "temp" in col_name:
                temperature = self._dataframe[row, column]
                return f"{temperature:.1f}"
            elif "hb" in col_name:
                hbr = self._dataframe[row, column]
                return f"{hbr:.4f}"
            elif "vent" in col_name:
                ventilation = self._dataframe[row, column]
                return f"{ventilation:.4f}"
            return str(self._dataframe[row, column])

        return None


class DescriptiveStatsModel(QAbstractTableModel):
    def __init__(self, dataframe: pl.DataFrame, parent: QWidget | None = None):
        QAbstractTableModel.__init__(self, parent)
        self._dataframe = dataframe.shrink_to_fit(in_place=True)

    def rowCount(
        self, parent: QModelIndex | QPersistentModelIndex = QModelIndex()
    ) -> int:
        return self._dataframe.shape[0] if parent == QModelIndex() else 0

    def columnCount(
        self, parent: QModelIndex | QPersistentModelIndex = QModelIndex()
    ) -> int:
        return self._dataframe.shape[1] if parent == QModelIndex() else 0

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return self._dataframe.columns[section]

            if orientation == Qt.Orientation.Vertical:
                return f"{section}"

        return None

    def data(
        self,
        index: QModelIndex | QPersistentModelIndex,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if not index.isValid():
            return None

        column = index.column()
        row = index.row()

        if role == Qt.ItemDataRole.DisplayRole:
            col_name = self._dataframe.columns[column]
            info_data = self._dataframe[row, column]
            if (
                self._dataframe.get_column(col_name).dtype != pl.Float64
                or info_data > 100
            ):
                return (
                    str(info_data)
                    if isinstance(info_data, str)
                    else f"{int(info_data):_}"
                )
            return f"{info_data:.4g}"

        return None
