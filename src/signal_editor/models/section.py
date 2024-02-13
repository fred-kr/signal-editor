import pprint
import re
import typing as t
from collections import OrderedDict

import attrs
import neurokit2 as nk
import numpy as np
import numpy.typing as npt
import polars as pl
import polars.selectors as ps

from .. import type_aliases as _t
from ..peaks import find_peaks
from ..processing import filter_elgendi, filter_neurokit2, filter_signal, scale_signal
from .result import FocusedResult, ManualPeakEdits


class SectionIndices(t.NamedTuple):
    start: int
    stop: int
    # def __str__(self) -> str:
    #     return f"{self.start}, {self.stop}"


class SectionID(str):
    def __init__(self, value: str):
        if not re.match(r"^SEC_[a-zA-Z0-9]+_[0-9]{3}$", value):
            raise ValueError(
                f"SectionID must be of the form 'SEC_<signal_name>_000', got '{value}'"
            )
        super().__init__()

    @classmethod
    def create(cls, value: str) -> "SectionID":
        return cls(value)


def _to_section_indices(value: SectionIndices | tuple[int, int]) -> SectionIndices:
    return SectionIndices(*value)


def _to_structured_array(value: npt.NDArray[np.void] | pl.DataFrame) -> npt.NDArray[np.void]:
    if isinstance(value, pl.DataFrame):
        value = value.lazy().with_columns(ps.boolean().cast(pl.Int8)).collect()
    else:
        value = pl.from_numpy(value).lazy().with_columns(ps.boolean().cast(pl.Int8)).collect()
    return value.to_numpy(structured=True)


def _to_uint_array(value: npt.NDArray[np.uint32] | pl.Series) -> npt.NDArray[np.uint32]:
    return value.cast(pl.UInt32).to_numpy(writable=True) if isinstance(value, pl.Series) else value


def _to_float_array(value: npt.NDArray[np.float64] | pl.Series) -> npt.NDArray[np.float64]:
    return value.cast(pl.Float64).to_numpy(writable=True) if isinstance(value, pl.Series) else value


@attrs.define(slots=True, frozen=True, repr=True)
class SectionIdentifier:
    sig_name: str = attrs.field()
    section_id: SectionID = attrs.field()
    absolute_bounds: SectionIndices = attrs.field(converter=_to_section_indices)
    sampling_rate: int = attrs.field(converter=int)

    def as_dict(self) -> _t.SectionIdentifierDict:
        return _t.SectionIdentifierDict(**attrs.asdict(self))


@attrs.define(slots=True, frozen=True, repr=True)
class SectionResult:
    identifier: _t.SectionIdentifierDict = attrs.field()
    data: npt.NDArray[np.void] = attrs.field(converter=_to_structured_array)
    peaks_section: npt.NDArray[np.uint32] = attrs.field(converter=_to_uint_array)
    peaks_global: npt.NDArray[np.uint32] = attrs.field(converter=_to_uint_array)
    peak_edits: _t.ManualPeakEditsDict = attrs.field()
    rate: npt.NDArray[np.float64] = attrs.field(converter=_to_float_array)
    rate_interpolated: npt.NDArray[np.float64] = attrs.field(converter=_to_float_array)
    processing_parameters: _t.ProcessingParameters = attrs.field()

    def as_dict(self) -> _t.SectionResultDict:
        return {
            "identifier": self.identifier,
            "data": self.data,
            "peaks_section": self.peaks_section,
            "peaks_global": self.peaks_global,
            "peak_edits": self.peak_edits,
            "rate": self.rate,
            "rate_interpolated": self.rate_interpolated,
            "processing_parameters": self.processing_parameters,
        }


class Section:
    _id_counter: int = 0

    def __init__(
        self,
        data: pl.DataFrame,
        sig_name: str,
        sampling_rate: int,
        set_active: bool = False,
        _is_base: bool = False,
    ) -> None:
        self.section_id = (
            SectionID(f"SEC_{sig_name}_000") if _is_base else self._generate_id(sig_name)
        )
        if "section_index" in data.columns:
            data = data.drop("section_index")
        self.data = (
            data.with_row_index("section_index")
            .lazy()
            .select(ps.by_name("index", "section_index"), ~ps.by_name("index", "section_index"))
            .set_sorted(["index", "section_index"])
            .collect()
        )
        self.sig_name = sig_name
        self._proc_sig_name = f"{sig_name}_processed"
        self._sampling_rate = sampling_rate
        self._is_active = set_active
        abs_index_col = data.get_column("index")
        self._abs_bounds = SectionIndices(start=abs_index_col[0], stop=abs_index_col[-1])
        self._rate = np.empty(0, dtype=np.float64)
        self._rate_interp = np.empty(0, dtype=np.float64)
        self._parameters_used = _t.ProcessingParameters(
            sampling_rate=self.sfreq,
            pipeline=None,
            filter_parameters=None,
            standardize_parameters=None,
            peak_detection_parameters=None,
        )
        self._peak_edits = ManualPeakEdits()

    def __repr__(self) -> str:
        return (
            "Section("
            f"section_id={self.section_id}, "
            f"sig_name={self.sig_name}, "
            f"sampling_rate={self._sampling_rate}, "
            f"is_active={self._is_active}, "
            f"base_bounds={self._abs_bounds}, "
            f"sect_bounds={self.sect_bounds}, "
            f"raw_data={self.raw_data}, "
            f"proc_data={self.proc_data}, "
            f"peaks={self.peaks}, "
            f"rate={self.rate}, "
            f"rate_interp={self.rate_interp}, "
            ")"
        )

    @classmethod
    def get_id_counter(cls) -> int:
        return cls._id_counter

    @classmethod
    def _get_next_id(cls) -> int:
        cls._id_counter += 1
        return cls._id_counter

    @classmethod
    def reset_id_counter(cls) -> None:
        cls._id_counter = 0

    def _generate_id(self, sig_name: str) -> SectionID:
        prefix = f"SEC_{sig_name}"
        number = self._get_next_id()
        return SectionID(f"{prefix}_{number:03d}")

    @property
    def processing_parameters(self) -> _t.ProcessingParameters:
        return self._parameters_used

    @property
    def is_active(self) -> bool:
        return self._is_active

    def set_active(self, value: bool) -> None:
        self._is_active = value

    @property
    def base_bounds(self) -> SectionIndices:
        """
        The start and stop row indices of this section in the base data (see `DataHandler.base_data`).
        """
        return self._abs_bounds

    @property
    def sect_bounds(self) -> SectionIndices:
        sect_index = self.data.get_column("section_index")
        return SectionIndices(sect_index[0], sect_index[-1])

    @property
    def raw_data(self) -> pl.Series:
        return self.data.get_column(self.sig_name)

    @property
    def proc_data(self) -> pl.Series:
        return self.data.get_column(self._proc_sig_name)

    @property
    def sfreq(self) -> int:
        return self._sampling_rate

    def update_sfreq(self, new_sfreq: int) -> None:
        self._sampling_rate = new_sfreq
        if self._rate.shape[0] != 0 or self._rate_interp.shape[0] != 0:
            self.calculate_rate(new_sfreq, self.peaks.to_numpy(zero_copy_only=True))

    @property
    def peaks(self) -> pl.Series:
        return (
            self.data.lazy()
            .filter(pl.col("is_peak") == 1)
            .select("section_index")
            .collect()
            .get_column("section_index")
        )

    @property
    def peaks_global(self) -> pl.Series:
        return self.data.get_column("index").gather(self.peaks)

    @property
    def rate(self) -> npt.NDArray[np.float64]:
        return self._rate

    @property
    def rate_interp(self) -> npt.NDArray[np.float64]:
        return self._rate_interp

    def filter_data(
        self,
        pipeline: _t.Pipeline,
        **kwargs: t.Unpack[_t.SignalFilterParameters],
    ) -> None:
        method = kwargs.get("method", "None")
        raw_data = self.raw_data.to_numpy(zero_copy_only=True)
        if pipeline == "custom":
            if method == "None":
                filtered = raw_data
                filter_params = "None"
            else:
                filtered, filter_params = filter_signal(raw_data, self.sfreq, **kwargs)
        elif pipeline == "ppg_elgendi":
            filtered = filter_elgendi(raw_data, self.sfreq)
            filter_params = _t.SignalFilterParameters(
                highcut=8,
                lowcut=0.5,
                method="butterworth",
                order=3,
                window_size="default",
                powerline=50,
            )
        elif pipeline == "ecg_neurokit2":
            filtered = filter_neurokit2(raw_data, self.sfreq, powerline=kwargs.get("powerline", 50))
            filter_params = _t.SignalFilterParameters(
                highcut=None,
                lowcut=0.5,
                method="butterworth",
                order=5,
                window_size="default",
                powerline=kwargs.get("powerline", 50),
            )
        else:
            raise ValueError(f"Unknown pipeline: {pipeline}")
        self._parameters_used["pipeline"] = pipeline
        self._parameters_used["filter_parameters"] = filter_params
        self.data = self.data.with_columns(
            pl.Series(self._proc_sig_name, filtered, pl.Float64).alias(self._proc_sig_name)
        )

    def scale_data(
        self,
        robust: bool = False,
        window_size: int | None = None,
        method: t.Literal["zscore", "mad", "None"] = "None",
    ) -> None:
        scaled = scale_signal(self.proc_data, robust, window_size)

        self._parameters_used["standardize_parameters"] = _t.StandardizeParameters(
            robust=robust,
            window_size=window_size,
            method=method,
        )
        self.data = self.data.with_columns(scaled.alias(self._proc_sig_name))

    def detect_peaks(
        self,
        method: _t.PeakDetectionMethod,
        method_parameters: _t.PeakDetectionInputValues,
    ) -> None:
        sampling_rate = self.sfreq
        self._parameters_used["peak_detection_parameters"] = _t.PeakDetectionParameters(
            method=method,
            method_parameters=method_parameters,
        )
        peaks = find_peaks(
            self.proc_data.to_numpy(zero_copy_only=True), sampling_rate, method, method_parameters
        )
        self.set_peaks(peaks)

    def calculate_rate(self, sampling_rate: int, peaks: npt.NDArray[np.int32 | np.uint32]) -> None:
        self._rate = np.asarray(
            nk.signal_rate(
                peaks,
                sampling_rate,
                desired_length=None,
                interpolation_method="monotone_cubic",
            ),
            dtype=np.float64,
        )
        self._rate_interp = np.asarray(
            nk.signal_rate(
                peaks,
                sampling_rate,
                desired_length=self.proc_data.shape[0],
                interpolation_method="monotone_cubic",
            ),
            dtype=np.float64,
        )

    def set_peaks(self, peaks: npt.NDArray[np.int32 | np.uint32]) -> None:
        """
        Set the `is_peak` column, overwriting the current values.

        Parameters
        ----------
        peaks : npt.NDArray[np.int32 | np.uint32]
            The new peaks

        Raises
        ------
        ValueError
            If any peaks are not positive integers
        """
        if np.any(peaks < 0):
            raise ValueError("Peaks must be positive integers")
        pl_peaks = pl.Series("peaks", peaks, pl.UInt32)

        self.data = self.data.with_columns(
            pl.when(pl.col("section_index").is_in(pl_peaks))
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .cast(pl.Int8)
            .alias("is_peak")
        )
        self._peak_edits.clear()
        self.calculate_rate(self.sfreq, peaks)

    def update_peaks(
        self,
        action: t.Literal["add", "remove"],
        peaks: t.Sequence[int] | npt.NDArray[np.intp],
    ) -> None:
        """
        Update the `is_peak` column based on the given action and indices.
        Only modifies the values at the given indices, while leaving the rest unchanged.

        Parameters
        ----------
        action : Literal['add', 'remove']
            How to modify the peaks.
        peaks : Sequence[int]
            The indices of the peaks to modify.
        """
        pl_peaks = pl.Series("peaks", peaks, pl.Int32)
        then_value = 1 if action == "add" else 0
        updated_data = (
            self.data.lazy()
            .select(
                pl.when(pl.col("section_index").is_in(pl_peaks))
                .then(then_value)
                .otherwise(pl.col("is_peak"))
                .cast(pl.Int8)
                .alias("is_peak")
            )
            .collect()
            .get_column("is_peak")
        )

        changed_indices = pl.arg_where(updated_data != self.data.get_column("is_peak"), eager=True)

        self.data.replace("is_peak", updated_data)

        if action == "add":
            self._peak_edits.new_added(changed_indices)
        else:
            self._peak_edits.new_removed(changed_indices)

    def get_peak_edits(self) -> ManualPeakEdits:
        """
        Get the manual peak edits for this section.

        Returns
        -------
        ManualPeakEdits
            A `ManualPeakEdits` object with lists `added` and `removed` containing the sorted and deduplicated indices.
        """
        self._peak_edits.sort_and_deduplicate()
        return self._peak_edits

    def add_is_manual_column(self) -> None:
        manual_indices = pl.Series("manual", self._peak_edits.get_joined(), pl.UInt32)

        self.data = (
            self.data.lazy()
            .with_columns(
                pl.when(pl.col("section_index").is_in(manual_indices))
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
                .cast(pl.Int8)
                .alias("is_manual")
            )
            .collect()
        )

    def get_section_info(self) -> SectionIdentifier:
        return SectionIdentifier(
            sig_name=self.sig_name,
            section_id=self.section_id,
            absolute_bounds=self.base_bounds,
            sampling_rate=self.sfreq,
        )

    def get_peak_xy(self) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.float64]]:
        peaks = self.peaks.to_numpy(zero_copy_only=True)
        return peaks, self.proc_data.gather(peaks).to_numpy()

    def get_focused_result(self) -> FocusedResult:
        peaks = self.peaks
        if peaks.len() < 3:
            raise RuntimeWarning(
                f"Need at least 3 peaks to calculate focused results. Current peaks: {peaks}. No result created."
            )

        global_peaks = self.data.get_column("index").gather(peaks)
        time_global = global_peaks / self.sfreq
        time_section = peaks / self.sfreq
        intervals = peaks.diff().fill_null(0).to_numpy()
        temperature = self.data.get_column("temperature").gather(peaks).to_numpy()

        return FocusedResult(
            peaks_section_index=peaks.to_numpy(),
            peaks_global_index=global_peaks.to_numpy(),
            seconds_since_section_start=time_section.to_numpy(),
            seconds_since_global_start=time_global.to_numpy(),
            peak_intervals=intervals,
            temperature=temperature,
            rate_bpm=self.rate,
        )

    def get_section_result(self) -> SectionResult:
        self.add_is_manual_column()

        data = (
            self.data.lazy()
            .with_columns(
                (pl.col("index") / self.sfreq).alias("sec_since_global_start"),
                (pl.col("section_index") / self.sfreq).alias("sec_since_section_start"),
            )
            .select(
                pl.col("index").alias("global_index"),
                ps.by_name("section_index", "sec_since_global_start", "sec_since_section_start"),
                ~ps.by_name(
                    "index",
                    "global_index",
                    "section_index",
                    "sec_since_global_start",
                    "sec_since_section_start",
                ),
            )
            .collect()
        )

        return SectionResult(
            identifier=self.get_section_info().as_dict(),
            data=data,
            peaks_section=self.peaks,
            peaks_global=self.peaks_global,
            peak_edits=self.get_peak_edits().as_dict(),
            rate=self.rate,
            rate_interpolated=self.rate_interp,
            processing_parameters=self.processing_parameters,
        )


class SectionContainer(OrderedDict[SectionID, Section]):
    def __setitem__(self, key: SectionID, value: Section) -> None:
        super().__setitem__(key, value)
        self.move_to_end(key)

    def __getitem__(self, key: SectionID) -> Section:
        return super().__getitem__(key)

    def __repr__(self) -> str:
        return pprint.pformat(self, indent=2, width=250, compact=True)
