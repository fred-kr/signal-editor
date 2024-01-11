from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from loguru import logger

import numpy as np
import polars as pl
import polars.selectors as ps
from numpy.typing import NDArray
from PySide6.QtCore import QObject, Slot
from PySide6.QtWidgets import QInputDialog

from ..models.io import read_edf
from ..models.result import (
    DescriptiveStatistics,
    FocusedResult,
    SummaryStatistics,
    make_focused_result,
)
from ..models.signal import SignalData, SignalStorage
from ..type_aliases import (
    PeakDetectionParameters,
    Pipeline,
    SignalFilterParameters,
    SignalName,
    StandardizeParameters,
)

if TYPE_CHECKING:
    from ..app import MainWindow


def try_infer_sampling_rate(
    df: pl.DataFrame,
    time_col: str = "auto",
    time_unit: Literal["auto", "s", "ms"] = "auto",
) -> int:
    if time_col == "auto":
        # Try to infer the column holding the time data
        for col in df.columns:
            if "time" in col:
                time_col = col
                break

    if time_unit == "auto":
        if df.get_column(time_col).dtype.is_float():
            time_unit = "s"
        elif df.get_column(time_col).dtype.is_integer():
            time_unit = "ms"
        else:
            logger.error(f"Could not infer time unit from column '{time_col}'")
            return -1

    target = 1000 if time_unit == "ms" else 1.0
    closed = "left" if df.get_column(time_col)[0] == 0 else "both"
    lower = df.get_column(time_col)[0]
    return df.filter(
        pl.col(time_col).is_between(lower, target, closed=closed)
    ).height

    


def get_array_stats(
    array: NDArray[np.integer[Any] | np.floating[Any]], description: str
) -> DescriptiveStatistics:
    return DescriptiveStatistics(
        description,
        mean=np.mean(array),
        median=np.median(array),
        std=np.std(array),
        var=np.var(array),
    )


@dataclass(slots=True)
class DataState:
    df: pl.DataFrame
    sigs: SignalStorage
    focused_results: dict[SignalName | str, FocusedResult] = field(init=False)


class DataHandler(QObject):
    def __init__(self, window: "MainWindow") -> None:
        super().__init__()
        self._window = window
        self.df: pl.DataFrame = pl.DataFrame()
        self.sigs: SignalStorage = SignalStorage()
        self.focused_results: dict[SignalName | str, FocusedResult] = {}
        self._sampling_rate: int = -1

    @property
    def fs(self) -> int:
        return self._sampling_rate

    @fs.setter
    def fs(self, value: int | float) -> None:
        self._sampling_rate = int(value)
        self._window.spin_box_fs.blockSignals(True)
        self._window.spin_box_fs.setValue(int(value))
        self._window.spin_box_fs.blockSignals(False)

    @Slot(int)
    def update_fs(self, value: int) -> None:
        self.fs = value
        self.sigs.update_sampling_rate(value)

    def read(self, path: str | Path) -> None:
        path = Path(path)
        suffix = path.suffix
        if suffix not in {".csv", ".txt", ".edf", ".feather", ".xlsx", ".pkl"}:
            info_msg = "Currently only .csv, .txt, .xlsx, .feather, .pkl and .edf files are supported"
            self._window.sig_show_message.emit(info_msg, "info")
            return

        if suffix == ".csv":
            df = pl.read_csv(path)
        elif suffix == ".edf":
            lf, self.meas_date, self.fs = read_edf(path.as_posix())
            df = lf.collect()
        elif suffix == ".feather":
            df = pl.read_ipc(path, use_pyarrow=True)
        elif suffix == ".pkl":
            self._window.restore_state(path)
            return
        elif suffix == ".txt":
            df = pl.read_csv(path, separator="\t")
        elif suffix == ".xlsx":
            df = pl.read_excel(path)
        else:
            raise NotImplementedError(f"File type `{suffix}` not supported")

        if suffix != ".edf":
            self.fs = try_infer_sampling_rate(df)

        if self.fs == -1:
            fs, ok = QInputDialog.getInt(
                self._window,
                "Sampling rate",
                "Enter sampling rate (samples per second): ",
                200,
                1,
                10000,
                1,
            )
            if ok:
                self.fs = fs
            else:
                self._sampling_rate_unavailable()
                return
        self.df = df
        self.calc_minmax()

        for name in {"hbr", "ventilation"}:
            if name not in df.columns:
                continue
            sig = SignalData(name=name, data=df, sampling_rate=self.fs)
            self.sigs[name] = sig

    def _sampling_rate_unavailable(self) -> None:
        self.df = pl.DataFrame()
        self.fs = -1
        self.minmax_map = {}
        self.meas_date = None
        msg = "Sampling rate not set, no data loaded."
        self._window.sig_show_message.emit(msg, "warning")
        return

    def calc_minmax(self, col_names: list[str] | None = None) -> None:
        if col_names is None or len(col_names) == 0:
            col_names = ["index", "time", "temp"]
        columns = self.df.select(ps.contains(col_names)).columns
        if len(columns) == 0:
            self.minmax_map = {}
            return

        self.minmax_map = {
            name: {
                "min": self.df.get_column(name).min(),
                "max": self.df.get_column(name).max(),
            }
            for name in columns
        }

    def _get_slice_indices(
        self, filter_col: str, lower: float, upper: float
    ) -> tuple[int, int]:
        lf = self.df.lazy().with_columns(pl.col("temperature").round(1))
        b1: int = (
            lf.sort(pl.col(filter_col), maintain_order=True)
            .filter(pl.col(filter_col) >= pl.lit(lower))
            .collect()
            .get_column("index")[0]
        )
        b2: int = (
            lf.sort(pl.col(filter_col), maintain_order=True)
            .filter(pl.col(filter_col) >= pl.lit(upper))
            .collect()
            .get_column("index")[0]
        )
        out = [b1, b2]
        out.sort()
        return out[0], out[1]

    def get_subset(self, subset_col: str, lower: float, upper: float) -> None:
        lf = self.df.lazy()
        if (
            lower == self.minmax_map[subset_col]["min"]
            and upper == self.minmax_map[subset_col]["max"]
        ):
            return
        if subset_col in {"time_s", "temperature"}:
            lower, upper = self._get_slice_indices(subset_col, lower, upper)
        self.df = lf.filter(pl.col("index").is_between(lower, upper)).collect()

    def run_preprocessing(
        self,
        name: SignalName | str,
        pipeline: Pipeline,
        filter_params: SignalFilterParameters,
        standardize_params: StandardizeParameters,
    ) -> None:
        self.sigs[name].filter_values(pipeline=pipeline, col_name=None, **filter_params)

        standardize = self._window.scale_method
        if standardize.lower() == "none":
            return
        self.sigs[name].scale_values(**standardize_params)

    def run_peak_detection(
        self, name: SignalName | str, peak_parameters: PeakDetectionParameters
    ) -> None:
        self.sigs[name].detect_peaks(**peak_parameters)
        self.run_rate_calculation(name)

    def run_rate_calculation(self, name: SignalName | str) -> None:
        self.sigs[name].calculate_rate()

    def get_descriptive_stats(self, name: SignalName | str) -> SummaryStatistics:
        intervals = self.sigs[name].get_peak_diffs()
        rate = self.sigs[name].signal_rate.rate

        interval_stats = get_array_stats(intervals, "peak_intervals")
        rate_stats = get_array_stats(rate, "rate")

        return SummaryStatistics(
            peak_intervals=interval_stats,
            signal_rate=rate_stats,
        )

    def compute_result_df(self, name: SignalName | str) -> None:
        sig = self.sigs[name]
        peaks = sig.peaks
        diffs = sig.get_peak_diffs(peaks)
        rate = sig.signal_rate.rate

        data = sig.get_data()
        indices = data.get_column("index").gather(peaks)
        time_s = (indices * (1 / self.fs)).round(4)
        temperature = data.get_column("temperature").gather(peaks).round(1)
        focused_result = make_focused_result(
            time_s=time_s,
            index=indices,
            peak_intervals=diffs,
            temperature=temperature,
            rate_bpm=rate,
        )
        self.focused_results[name] = focused_result

    def get_focused_result_df(
        self, name: SignalName | str, compute: bool = True
    ) -> pl.DataFrame:
        if compute or name not in self.focused_results:
            self.compute_result_df(name)
        return self.focused_results[name].to_polars()

    def get_state(self) -> DataState:
        state = DataState(df=self.df.clone(), sigs=self.sigs.deepcopy())
        state.focused_results = self.focused_results
        return state

    def restore_state(self, state: DataState) -> None:
        self.restore_df(state.df)
        self.sigs = state.sigs
        self.focused_results = state.focused_results

    def restore_df(self, df: pl.DataFrame) -> None:
        self.df = df
        if all(
            col in df.columns
            for col in ["index", "time_s", "temperature", "hbr", "ventilation"]
        ):
            self.df = self.df.select(
                pl.col("index", "time_s", "temperature", "hbr", "ventilation")
            )
        else:
            self.df = df
        self.calc_minmax()
