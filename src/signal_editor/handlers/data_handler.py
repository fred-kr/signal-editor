from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, NamedTuple, Unpack, cast

import neurokit2 as nk
import numpy as np
import polars as pl
import pyqtgraph as pg
import wfdb.processing
from loguru import logger
from numpy.typing import NDArray
from PySide6.QtCore import QObject, Signal, Slot

from ..models.filters import (
    auto_correct_fir_length,
    filter_custom,
    filter_elgendi,
    scale_signal,
)
from ..models.io import read_edf
from ..models.peaks import (
    find_local_peaks,
    find_ppg_peaks_elgendi,
    neurokit2_find_peaks,
)
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
    SignalFilterParameters,
    SignalName,
    StandardizeParameters,
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
    Holds the non-interpolated rate data for each signal. Results come from the
    `neurokit2.signal_rate` function.
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

    return scale_signal(
        sig, robust=is_robust, window_size=window_size, rolling_window=rolling_window
    ).to_numpy()


class ExclusionIndices(NamedTuple):
    start: int
    stop: int


@dataclass(slots=True, kw_only=True)
class ExclusionMask:
    hbr: list[ExclusionIndices] = field(default_factory=list)
    ventilation: list[ExclusionIndices] = field(default_factory=list)

    def __getitem__(self, key: SignalName) -> list[ExclusionIndices]:
        return getattr(self, key, [])

    def add_range(self, name: SignalName, start: int, stop: int) -> None:
        self[name].append(ExclusionIndices(start=start, stop=stop))

    def remove_range(self, name: SignalName, start: int, stop: int) -> None:
        self[name].remove(ExclusionIndices(start=start, stop=stop))

    def clear_all(self) -> None:
        self.hbr = []
        self.ventilation = []

    def get_starts(self, name: SignalName) -> dict[int, int]:
        return {i: r.start for i, r in enumerate(self[name])}

    def get_stops(self, name: SignalName) -> dict[int, int]:
        return {i: r.stop for i, r in enumerate(self[name])}


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
        self.exclusion_ranges: ExclusionMask = ExclusionMask()
        self.cw_df: pl.DataFrame = pl.DataFrame()

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
        self.df = self.df.with_columns(
            pl.repeat(1, self.df.height, dtype=pl.Int8).alias("hbr_is_included"),
            pl.repeat(1, self.df.height, dtype=pl.Int8).alias(
                "ventilation_is_included"
            ),
        )

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
            self.df.replace(
                name_rate_interp,
                pl.Series(name_rate_interp, rate_interp, dtype=pl.Float64).round(1),
            )
            # self.df = self.df.drop(name_rate_interp)
        else:
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

    @Slot(str, int, int)
    def mark_excluded(self, name: SignalName, start: int, stop: int) -> None:
        if len(self.exclusion_ranges[name]) == 0:
            setattr(self, f"_{name}_all", self.df.lazy().clone())
        self.exclusion_ranges.add_range(name, start, stop)
        self.df = self.df.with_columns(
            pl.when((pl.col("index").is_between(start, stop)))
            .then(None)
            .otherwise(pl.col(f"{name}_is_included"))
            .alias(f"{name}_is_included")
        )
        self.df.drop_nulls(f"{name}_is_included")

    def reset_exclusions(self, name: SignalName) -> None:
        init_df: pl.LazyFrame = getattr(self, f"_{name}_all")
        self.df = init_df.collect()

    # def apply_exclusion_mask(self, name: SignalName) -> None:
    # sub_sections = []
    # for masked_range in self.exclusion_ranges[name]:
    # sub_df = self.df.slice(masked_range.start, masked_range.stop - masked_range.start)
    # sub_sections.append(sub_df)
    # lr_marker = pg.LinearRegionItem(values=(masked_range.start, masked_range.stop), movable=False)
