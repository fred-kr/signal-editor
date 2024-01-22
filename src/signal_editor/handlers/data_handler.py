import typing as t
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import polars.selectors as ps
from loguru import logger
from numpy.typing import NDArray
from PySide6.QtCore import QObject, Slot
from PySide6.QtWidgets import QInputDialog

from .. import type_aliases as _t
from ..models.io import read_edf
from ..models.result import (
    DescriptiveStatistics,
    FocusedResult,
    SummaryStatistics,
    make_focused_result,
)
from ..models.signal import Section, SectionID, SignalData

if t.TYPE_CHECKING:
    from ..app import SignalEditor


def try_infer_sampling_rate(
    df: pl.DataFrame,
    time_col: str = "auto",
    time_unit: t.Literal["auto", "s", "ms"] = "auto",
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
    return df.filter(pl.col(time_col).is_between(lower, target, closed=closed)).height


def get_array_stats(
    array: NDArray[np.integer[t.Any] | np.floating[t.Any]], description: str
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
    sig_data: SignalData = field(init=False)
    focused_results: dict[str, FocusedResult] = field(init=False)


class DataHandler(QObject):

    def __init__(self, app: "SignalEditor") -> None:
        super().__init__()
        self._app = app
        self.df: pl.DataFrame = pl.DataFrame()
        self._sig: SignalData | None = None
        self.focused_results: dict[str, FocusedResult] = {}
        self._sampling_rate: int = -1
        self.minmax_map = {}

    @property
    def fs(self) -> int:
        return self._sampling_rate

    @fs.setter
    def fs(self, value: int | float) -> None:
        value = int(value)
        self._sampling_rate = value
        self._app.spin_box_sample_rate.blockSignals(True)
        self._app.spin_box_sample_rate.setValue(value)
        self._app.spin_box_sample_rate.blockSignals(False)

    @Slot(int)
    def update_fs(self, value: int) -> None:
        self.fs = value
        if self._sig is not None:
            self._sig.sampling_rate = value

    @property
    def sig_data(self) -> SignalData:
        if self._sig is None:
            raise RuntimeError("No signal data loaded")
        return self._sig

    @sig_data.setter
    def sig_data(self, value: SignalData) -> None:
        self._sig = value

    @property
    def active_section(self) -> Section:
        """Easy access to the currently active section."""
        return self.sig_data.active_section

    def read(self, path: str | Path) -> None:
        path = Path(path)
        suffix = path.suffix
        if suffix not in {".csv", ".txt", ".edf", ".feather", ".xlsx", ".pkl"}:
            info_msg = (
                "Currently only .csv, .txt, .xlsx, .feather, .pkl and .edf files are supported"
            )
            self._app.sig_show_message.emit(info_msg, "info")
            return

        if suffix == ".csv":
            df = pl.read_csv(path)
        elif suffix == ".edf":
            lf, self.meas_date, self.fs = read_edf(path.as_posix())
            df = lf.collect()
        elif suffix == ".feather":
            df = pl.read_ipc(path, use_pyarrow=True)
        elif suffix == ".pkl":
            self._app.restore_state(path)
            return
        elif suffix == ".txt":
            df = pl.read_csv(path, separator="\t")
        elif suffix == ".xlsx":
            df = pl.read_excel(path)
        else:
            raise NotImplementedError(f"File type `{suffix}` not supported")

        if suffix != ".edf":
            fs = try_infer_sampling_rate(df)

            if fs == -1:
                fs, ok = QInputDialog.getInt(
                    self._app,
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

    def _sampling_rate_unavailable(self) -> None:
        self.df = pl.DataFrame()
        self.fs = -1
        self.minmax_map = {}
        self.meas_date = None
        msg = "Sampling rate not set, no data loaded."
        self._app.sig_show_message.emit(msg, "warning")
        return

    def new_sig_data(self, name: str) -> None:
        if name not in self.df.columns:
            logger.debug(f"Tried to set signal data to non-existing column '{name}'")
            return
        self._sig = SignalData(name=name, data=self.df, sampling_rate=self.fs)

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

    def _get_slice_indices(self, filter_col: str, lower: float, upper: float) -> tuple[int, int]:
        lf = self.df.lazy().with_columns(pl.col("temperature").round(1))
        sorted_lf = lf.sort(pl.col(filter_col), maintain_order=True)
        b1: int = (
            sorted_lf.filter(pl.col(filter_col) >= pl.lit(lower)).collect().get_column("index")[0]
        )
        b2: int = (
            sorted_lf.filter(pl.col(filter_col) >= pl.lit(upper)).collect().get_column("index")[0]
        )
        return b1, b2

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
        name: str,
        pipeline: _t.Pipeline,
        filter_params: _t.SignalFilterParameters,
        standardize_params: _t.StandardizeParameters,
    ) -> None:
        self.active_section.filter_signal(pipeline=pipeline, **filter_params)

        standardize = self._app.scale_method
        if standardize.lower() == "none":
            return
        self.active_section.scale_signal(**standardize_params)

    def run_peak_detection(self, name: str, peak_parameters: _t.PeakDetectionParameters) -> None:
        self.active_section.detect_peaks(
            method=peak_parameters["method"],
            input_values=peak_parameters["input_values"],
        )
        self.run_rate_calculation(name)

    def run_rate_calculation(self, name: str) -> None:
        self.active_section.calculate_rate()

    def get_descriptive_stats(self, name: str) -> SummaryStatistics:
        intervals = self.active_section.get_peak_intervals()
        rate = self.active_section.rate_data.rate

        interval_stats = get_array_stats(intervals, "peak_intervals")
        rate_stats = get_array_stats(rate, "rate")

        return SummaryStatistics(
            peak_intervals=interval_stats,
            signal_rate=rate_stats,
        )

    def compute_result_df(self, name: str, section_id: SectionID | None = None) -> None:
        sig = self.sig_data
        # section_dict = sig.as_dict()
        # TODO: refactor result creation to work with new section structure
        if section_id is None:
            section_id = sig.active_section.section_id
        sec = sig.get_section_by_id(section_id)
        peaks = sec.get_peaks()
        diffs = sec.get_peak_intervals()
        rate = sig.default_rate
        indices = sec.data.get_column("index").gather(peaks)
        time_s = (indices * (1 / self.fs)).round(4)
        temperature = sec.data.get_column("temperature").gather(peaks).round(1)
        focused_result = make_focused_result(
            time_s=time_s,
            index=indices,
            peak_intervals=diffs,
            temperature=temperature,
            rate_bpm=rate,
        )
        self.focused_results[name] = focused_result

    def get_focused_result_df(self, name: str, compute: bool = True) -> pl.DataFrame:
        if compute or name not in self.focused_results:
            self.compute_result_df(name)
        return self.focused_results[name].to_polars()

    def get_state(self) -> DataState:
        state = DataState(df=self.df.clone())
        state.sig_data = self.sig_data
        state.focused_results = self.focused_results
        return state

    def restore_state(self, state: DataState) -> None:
        self.restore_df(state.df)
        self.sig_data = state.sig_data
        self.focused_results = state.focused_results

    def restore_df(self, df: pl.DataFrame) -> None:
        self.df = df
        if all(
            col in df.columns for col in ["index", "time_s", "temperature", "hbr", "ventilation"]
        ):
            self.df = self.df.select(pl.col("index", "time_s", "temperature", "hbr", "ventilation"))
        else:
            self.df = df
        self.calc_minmax()

    @Slot(int, int)
    def exclude_region(self, lower: int, upper: int) -> None:
        self.sig_data.add_section(lower, upper, set_active=False, include=False)
