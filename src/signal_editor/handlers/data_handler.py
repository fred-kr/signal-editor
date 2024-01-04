from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from numpy.typing import NDArray
from PySide6.QtCore import QObject, Signal, Slot

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
    sig_dh_error = Signal(str)
    sig_dh_peaks_updated = Signal(str)
    sig_dh_rate_updated = Signal(str)

    def __init__(self, parent: "MainWindow") -> None:
        super().__init__(parent)
        self._parent = parent
        self.df: pl.DataFrame = pl.DataFrame()
        self.sigs: SignalStorage = SignalStorage()
        self.focused_results: dict[SignalName | str, FocusedResult] = {}
        self._sampling_rate: int = parent.spin_box_fs.value()
        self._connect_signals()

    def _connect_signals(self) -> None:
        self.sig_dh_peaks_updated.connect(self.run_rate_calculation)

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

        if suffix == ".pkl":
            self._parent.restore_state(path)
            return
        elif suffix == ".csv":
            df = pl.read_csv(path)
        elif suffix == ".txt":
            df = pl.read_csv(path, separator="\t")
        elif suffix == ".edf":
            lf, self.meas_date, self.fs = read_edf(path.as_posix())
            df = lf.collect()
        elif suffix == ".feather":
            df = pl.read_ipc(path, use_pyarrow=True)
        elif suffix == ".xlsx":
            df = pl.read_excel(path)
        else:
            raise NotImplementedError(f"File type `{suffix}` not supported")

        self.df = df
        self.calc_minmax()

        for name in {"hbr", "ventilation"}:
            sig = SignalData(name=name, data=df, sampling_rate=self.fs)
            self.sigs[name] = sig

    def calc_minmax(self, col_names: list[str] | None = None) -> None:
        if col_names is None or len(col_names) == 0:
            col_names = ["index", "time_s", "temperature"]
        for name in col_names:
            if name not in self.df.columns:
                col_names.remove(name)

        self.minmax_map = {
            name: {
                "min": self.df.get_column(name).min(),
                "max": self.df.get_column(name).max(),
            }
            for name in col_names
        }

    def get_slice_indices(
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
            lower, upper = self.get_slice_indices(subset_col, lower, upper)
        self.df = lf.filter(pl.col("index").is_between(lower, upper)).collect()

    def run_preprocessing(
        self,
        name: SignalName,
        pipeline: Pipeline,
        filter_params: SignalFilterParameters,
        standardize_params: StandardizeParameters,
    ) -> None:
        self.sigs[name].filter_values(pipeline=pipeline, col_name=None, **filter_params)

        standardize = self._parent.scale_method
        if standardize.lower() == "none":
            return
        self.sigs[name].scale_values(**standardize_params)

    def run_peak_detection(
        self, name: SignalName, peak_parameters: PeakDetectionParameters
    ) -> None:
        self.sigs[name].detect_peaks(**peak_parameters)
        self.sig_dh_peaks_updated.emit(name)

    @Slot(str)
    def run_rate_calculation(self, name: SignalName | str) -> None:
        self.sigs[name].calculate_rate()
        self.sig_dh_rate_updated.emit(name)

    def get_descriptive_stats(self, name: SignalName | str) -> SummaryStatistics:
        intervals = self.sigs[name].get_peak_diffs()
        rate = self.sigs[name].signal_rate.rate

        interval_stats = get_array_stats(intervals, "peak_intervals")
        rate_stats = get_array_stats(rate, "rate")

        return SummaryStatistics(
            peak_intervals=interval_stats,
            signal_rate=rate_stats,
        )

    def compute_result_df(self, name: SignalName) -> None:
        sig = self.sigs[name]
        peaks = sig.peaks
        diffs = sig.get_peak_diffs()
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
        # result_df = (
        #     self.df.lazy()
        #     .select(
        #         pl.col("time_s").gather(peaks).round(4),
        #         pl.col("index").gather(peaks).alias("peak_index"),
        #         pl.Series("peak_intervals", diffs, pl.Int32),
        #         pl.col("temperature").gather(peaks).round(1),
        #         pl.Series("rate_bpm", rate, pl.Float64).round(1),
        #     )
        #     .collect()
        # )

        # self.result_dfs.update(name, result_df)

    def get_focused_result_df(
        self, name: SignalName, compute: bool = True
    ) -> pl.DataFrame:
        if compute:
            self.compute_result_df(name)
        # return self.result_dfs[name]
        return self.focused_results[name].to_polars()

    def get_state(self) -> DataState:
        state = DataState(df=self.df.clone(), sigs=self.sigs.deepcopy())
        state.focused_results = self.focused_results
        # state.peaks = self.sigs[name].peaks
        # state.rate = self.sigs[name].signal_rate
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
