from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Unpack

import neurokit2 as nk
import numpy as np
import polars as pl
import polars.selectors as cs
from loguru import logger
from numpy.typing import NDArray
from PySide6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QPersistentModelIndex,
    Qt,
)
from PySide6.QtWidgets import QWidget

from ..custom_types import (
    ComputedResults,
    InfoProcessingParams,
    InfoWorkingData,
    MinMaxMapping,
    NormMethod,
    OxygenCondition,
    PeakDetectionMethod,
    PeakIntervalStats,
    PeaksPPGElgendi,
    Pipeline,
    SignalName,
    SignalRateStats,
    StatsDict,
)
from .filters import (
    SignalFilterParameters,
    auto_correct_fir_length,
    filter_custom,
    filter_elgendi,
)
from .io import read_edf
from .peaks import find_ppg_peaks_elgendi


@dataclass(slots=True, frozen=True)
class Identifier:
    signal_name: SignalName
    file_name: str
    subject_id: str
    date_of_recording: date
    oxygen_condition: OxygenCondition


@dataclass(slots=True, kw_only=True)
class Results:
    signal_name: SignalName
    identifier: Identifier
    working_data_metadata: InfoWorkingData
    processing_metadata: InfoProcessingParams
    computed: ComputedResults
    processed_data: pl.DataFrame = field(default_factory=pl.DataFrame)


@dataclass
class DataHandler:
    file_path: str | Path
    sampling_rate: int = 400
    lazy: pl.LazyFrame = field(init=False)
    data: pl.DataFrame = field(init=False)
    min_max_mapping: dict[str, MinMaxMapping] = field(init=False)
    _results: Results = field(init=False)

    def __post_init__(self) -> None:
        path = Path(self.file_path).resolve()

        self.file_path = path
        self.posix_path = path.as_posix()
        self.file_extension = path.suffix

    def lazy_read(self) -> None:
        if self.file_extension not in {".csv", ".txt", ".edf", ".feather"}:
            logger.error(f"File extension {self.file_extension} not supported.")
            raise ValueError(f"File extension {self.file_extension} not supported.")

        if self.file_extension == ".edf":
            self.lazy = read_edf(self.posix_path)
        elif self.file_extension == ".feather":
            self.lazy = pl.scan_ipc(self.posix_path)
        else:
            self.lazy = pl.scan_csv(
                self.posix_path,
                try_parse_dates=True,
                new_columns=["time_s", "temperature", "hbr", "ventilation"],
            )

    def get_min_max(self) -> None:
        min_values = (
            self.lazy.select(~cs.contains(["hbr", "ventilation"])).min().collect()
        )
        max_values = (
            self.lazy.select(~cs.contains(["hbr", "ventilation"])).max().collect()
        )

        self.min_max_mapping = {
            column: {
                "min": min_values[column][0],
                "max": max_values[column][0],
            }
            for column in self.lazy.columns
            if "hbr" not in column and "ventilation" not in column
        }

    def normalize_signal(
        self, sig: NDArray[np.float32 | np.float64], norm_method: NormMethod
    ) -> NDArray[np.float32]:
        if norm_method == "mad":
            return np.asarray(nk.standardize(sig, robust=True), dtype=np.float32)
        elif norm_method == "zscore":
            return np.asarray(nk.standardize(sig, robust=False), dtype=np.float32)
        elif norm_method == "minmax":
            return np.asarray(
                (sig - sig.min()) / (sig.max() - sig.min()), dtype=np.float32
            )
        elif norm_method == "None":
            return np.asarray(sig, dtype=np.float32)

    def preprocess_signal(
        self,
        signal_name: SignalName,
        norm_method: NormMethod,
        pipeline: Pipeline,
        filter_params: SignalFilterParameters,
    ) -> None:
        processed_name = f"processed_{signal_name}"
        to_process = self.data.get_column(signal_name).to_numpy(zero_copy_only=True)
        to_process = np.asarray(to_process, dtype=np.float64)

        if pipeline == "custom" and filter_params["method"] != "None":
            if filter_params["method"] == "fir":
                processed = auto_correct_fir_length(
                    to_process, sampling_rate=self.sampling_rate, **filter_params
                )
            else:
                processed = filter_custom(
                    sig=to_process,
                    sampling_rate=self.sampling_rate,
                    **filter_params,
                )
        elif pipeline == "ppg_elgendi":
            processed = filter_elgendi(
                sig=to_process,
                sampling_rate=self.sampling_rate,
            )
        elif pipeline == "ecg_neurokit2":
            processed = to_process
            logger.warning(
                f"Selected pipeline {pipeline} not yet implemented, using unchanged input values."
            )
        elif pipeline == "ecg_biosppy":
            processed = to_process
            logger.warning(
                f"Selected pipeline {pipeline} not yet implemented, using unchanged input values."
            )
        elif pipeline == "ecg_pantompkins1985":
            processed = to_process
            logger.warning(
                f"Selected pipeline {pipeline} not yet implemented, using unchanged input values."
            )
        elif pipeline == "ecg_hamilton2002":
            processed = to_process
            logger.warning(
                f"Selected pipeline {pipeline} not yet implemented, using unchanged input values."
            )
        elif pipeline == "ecg_elgendi2010":
            processed = to_process
            logger.warning(
                f"Selected pipeline {pipeline} not yet implemented, using unchanged input values."
            )
        elif pipeline == "ecg_engzeemod2012":
            processed = to_process
            logger.warning(
                f"Selected pipeline {pipeline} not yet implemented, using unchanged input values."
            )
        else:
            raise ValueError(f"Invalid pipeline: {pipeline}")

        if processed_name in self.data.columns:
            self.data = self.data.drop(processed_name)
        self.data.hstack(
            [
                pl.Series(
                    name=processed_name,
                    values=self.normalize_signal(processed, norm_method).round(4),
                    dtype=pl.Float32,
                )
            ],
            in_place=True,
        )

    def find_peaks(
        self,
        signal_name: SignalName,
        peak_find_method: PeakDetectionMethod = "elgendi",
        **kwargs: Unpack[PeaksPPGElgendi],
    ) -> None:
        sig_array = self.data.get_column(f"processed_{signal_name}").to_numpy(
            zero_copy_only=True
        )

        if peak_find_method == "elgendi":
            peak_indices = find_ppg_peaks_elgendi(
                sig_array, sampling_rate=self.sampling_rate, **kwargs
            )
        else:
            raise NotImplementedError(
                f"Peak finding method {peak_find_method} not implemented."
            )
        pl_peak_indices = pl.Series(name="peaks", values=peak_indices, dtype=pl.Int32)
        setattr(self, f"{signal_name}_peaks", peak_indices)
        self.data = (
            self.data.lazy()
            .with_columns(
                (
                    pl.when(pl.col("index").is_in(pl_peak_indices))
                    .then(pl.lit(1))
                    .otherwise(pl.lit(0))
                    .shrink_dtype()
                    .alias(f"{signal_name}_peaks")
                )
            )
            .collect()
        )

        self.calculate_rate(signal_name, peak_indices)

    def calculate_rate(self, signal_name: str, peaks: NDArray[np.int32]) -> None:
        rate = np.asarray(
            nk.signal_rate(
                peaks=peaks,
                sampling_rate=self.sampling_rate,
                desired_length=len(self.data),
                interpolation_method="monotone_cubic",
            ),
            dtype=np.float32,
        ).round(4)
        col_name = f"{signal_name}_rate"

        if col_name in self.data.columns:
            self.data = self.data.drop(col_name)
        self.data.hstack(
            pl.Series(name=col_name, values=rate, dtype=pl.Int32).to_frame(),
            in_place=True
        )
        # self.data = self.data.with_columns(
        #     pl.Series(name=col_name, values=rate, dtype=pl.Int32)
        # )

        setattr(self, f"{signal_name}_rate_len_signal", rate)

    @staticmethod
    def compute_stats(
        signal_name: SignalName, peaks: NDArray[np.int32], rate: NDArray[np.float32]
    ) -> StatsDict:
        return StatsDict(
            signal_name=signal_name,
            peak_stats=PeakIntervalStats(
                signal_name=signal_name,
                peak_interval_mean=np.mean(peaks),
                peak_interval_median=np.median(peaks),
                peak_interval_std=np.std(peaks),
                peak_interval_var=np.var(peaks),
            ),
            signal_rate_stats=SignalRateStats(
                signal_name=signal_name,
                signal_rate_mean=np.mean(rate),
                signal_rate_median=np.median(rate),
                signal_rate_std=np.std(rate),
                signal_rate_var=np.var(rate),
            ),
        )

    def compute_results(
        self,
        signal_name: SignalName,
    ) -> ComputedResults:
        signals = {
            "hbr": ("hbr_peaks", "hbr_rate_len_signal"),
            "ventilation": ("ventilation_peaks", "ventilation_rate_len_signal"),
        }

        peaks_attr, rate_attr = signals[signal_name]
        peaks: NDArray[np.int32] = getattr(self, peaks_attr)
        rate: NDArray[np.float32] = getattr(self, rate_attr)

        peak_intervals = np.ediff1d(peaks, to_begin=0)
        stats = self.compute_stats(signal_name, peak_intervals, rate)
        signal_rate_len_peaks = np.asarray(
            nk.signal_rate(
                peaks=peaks,
                sampling_rate=self.sampling_rate,
                desired_length=None,
                interpolation_method="monotone_cubic",
            ),
            dtype=np.float32,
        )

        setattr(
            self,
            f"{signal_name}_rate_len_peaks",
            signal_rate_len_peaks.astype(np.int32),
        )

        return ComputedResults(
            signal_name=signal_name,
            peak_intervals=peak_intervals,
            signal_rate_len_peaks=signal_rate_len_peaks,
            signal_rate_len_signal=rate,
            peak_interval_stats=stats["peak_stats"],
            signal_rate_stats=stats["signal_rate_stats"],
        )

    def make_results_df(
        self,
        peaks: NDArray[np.int32],
        rate: NDArray[np.int32],
    ) -> pl.DataFrame:
        return pl.DataFrame(
            [
                pl.Series(
                    name="time_s",
                    values=self.data.get_column("time_s").gather(peaks).round(4),
                    dtype=pl.Float64,
                ),
                pl.Series(name="peak_index", values=peaks, dtype=pl.Int32),
                pl.Series(
                    name="peak_interval",
                    values=np.ediff1d(peaks, to_begin=0),
                    dtype=pl.Int32,
                ),
                pl.Series(
                    name="temperature",
                    values=self.data.get_column("temperature").gather(peaks).round(1),
                    dtype=pl.Float32,
                ).shrink_dtype(),
                pl.Series(name="bpm", values=rate, dtype=pl.Int32),
            ]
        )


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
                return f"{float(self._data[row][column]):_.4f}"
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
