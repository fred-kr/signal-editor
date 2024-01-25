import re
import typing as t
from collections import OrderedDict
from pprint import pformat

import attrs
import neurokit2 as nk
import numpy as np
import numpy.typing as npt
import polars as pl

from .. import type_aliases as _t
from .peaks import find_peaks
from .processing import filter_elgendi, filter_signal, scale_signal
from .result import FocusedResult, ManualPeakEdits


def _get_summary(data: pl.Series) -> dict[str, float]:
    summary = data.describe().to_dict()
    return {k: v[0] for k, v in summary.items()}


class SectionIndices(t.NamedTuple):
    start: int
    stop: int

    def __str__(self) -> str:
        return f"{self.start}, {self.stop}"


class SectionID(str):
    def __new__(cls, value: str) -> "SectionID":
        if not re.match(r"^SEC_[a-zA-Z0-9]+_[0-9]{3}$", value):
            raise ValueError("SectionID must be of the form 'SEC_{name of signal}_001'")
        return super().__new__(cls, value)


def _to_section_indices(value: SectionIndices | tuple[int, int]) -> SectionIndices:
    return SectionIndices(*value)


def _to_structured_array(value: npt.NDArray[np.void] | pl.DataFrame) -> npt.NDArray[np.void]:
    if isinstance(value, pl.DataFrame):
        return value.to_numpy(structured=True)
    return value


def _to_index_array(value: npt.NDArray[np.uint32] | pl.Series) -> npt.NDArray[np.uint32]:
    return value.cast(pl.UInt32).to_numpy(writable=True) if isinstance(value, pl.Series) else value


def _to_float_array(value: npt.NDArray[np.float64] | pl.Series) -> npt.NDArray[np.float64]:
    return value.cast(pl.Float64).to_numpy(writable=True) if isinstance(value, pl.Series) else value


@attrs.define(slots=True, frozen=True)
class SectionResult:
    sig_name: str = attrs.field()
    section_id: SectionID = attrs.field()
    absolute_bounds: SectionIndices = attrs.field(converter=_to_section_indices)
    data: npt.NDArray[np.void] = attrs.field(converter=_to_structured_array)
    sampling_rate: int = attrs.field(converter=int)
    peaks: npt.NDArray[np.uint32] = attrs.field(converter=_to_index_array)
    peak_edits: ManualPeakEdits = attrs.field()
    rate: npt.NDArray[np.float64] = attrs.field(converter=_to_float_array)
    rate_interpolated: npt.NDArray[np.float64] = attrs.field(converter=_to_float_array)
    processing_parameters: _t.ProcessingParameters = attrs.field()
    focused_result: npt.NDArray[np.void] = attrs.field(converter=_to_structured_array)

    def as_dict(self) -> _t.SectionResultDict:
        return {
            "sig_name": self.sig_name,
            "section_id": self.section_id,
            "absolute_bounds": self.absolute_bounds,
            "data": self.data,
            "sampling_rate": self.sampling_rate,
            "peaks": self.peaks,
            "peak_edits": self.peak_edits.as_dict(),
            "rate": self.rate,
            "rate_interpolated": self.rate_interpolated,
            "processing_parameters": self.processing_parameters,
            "focused_result": self.focused_result,
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
        self.data = data.with_row_index("section_index")
        self.sig_name = sig_name
        self._proc_sig_name = f"{sig_name}_processed"
        self._sampling_rate = sampling_rate
        self._is_active = set_active
        abs_index_col = data.get_column("index")
        self._abs_bounds = SectionIndices(abs_index_col[0], abs_index_col[-1])
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

    def __str__(self) -> str:
        return pformat(self.__dict__, indent=4, width=100, compact=True)

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

    @property
    def peaks(self) -> pl.Series:
        return self.data.get_column("is_peak").arg_true()

    @property
    def rate(self) -> npt.NDArray[np.float64]:
        return self._rate

    @property
    def rate_interp(self) -> npt.NDArray[np.float64]:
        return self._rate_interp

    @property
    def rate_stats(self) -> dict[str, float]:
        return _get_summary(pl.Series("rate", self.rate, pl.Float64))

    @property
    def interval_stats(self) -> dict[str, float]:
        return _get_summary(pl.Series("peak_intervals", self.peaks.diff(null_behavior="drop")))

    def left_shrink(self, new_stop_index: int) -> "Section":
        """
        Create a new section from this section's start index (inclusive) to the given stop index (exclusive).

        Before:
            >>> 0|--------------------|len(self.data)
        After:
            >>> 0|------|new_stop_index

        Parameters
        ----------
        new_stop_index : int
            The new stop index for the section.
        """
        if new_stop_index > self.sect_bounds.stop:
            raise ValueError("New stop index must be smaller than current stop index")

        new_data = self.data.filter(pl.col("index") < new_stop_index).drop("section_index")
        return Section(new_data, self.sig_name, self.sfreq, set_active=self.is_active)

    def right_shrink(self, new_start_index: int) -> "Section":
        """
        Create a new section from the given start index (exclusive) to this section's stop index (inclusive).

        Before:
            >>> 0|--------------------|len(self.data)
        After:
            >>> new_start_index|------|len(self.data)

        Parameters
        ----------
        new_start_index : int
            The new start index for the section.
        """
        if new_start_index < self.sect_bounds.start:
            raise ValueError("New start index must be larger than current start index")

        new_data = self.data.filter(pl.col("index") > new_start_index).drop("section_index")
        return Section(new_data, self.sig_name, self.sfreq, set_active=self.is_active)

    def filter_data(
        self,
        pipeline: _t.Pipeline,
        **kwargs: t.Unpack[_t.SignalFilterParameters],
    ) -> None:
        method = kwargs.get("method", "None")
        if pipeline == "custom":
            if method == "None":
                filtered = self.raw_data.to_numpy()
            else:
                filtered = filter_signal(self.raw_data.to_numpy(), self.sfreq, **kwargs)
        elif pipeline == "ppg_elgendi":
            filtered = filter_elgendi(self.raw_data.to_numpy(), self.sfreq)
        else:
            raise ValueError(f"Unknown pipeline: {pipeline}")
        filter_parameters = _t.SignalFilterParameters(**kwargs)
        self._parameters_used["pipeline"] = pipeline
        self._parameters_used["filter_parameters"] = filter_parameters
        pl_filtered = pl.Series(self._proc_sig_name, filtered, pl.Float64)
        self.data = self.data.with_columns((pl_filtered).alias(self._proc_sig_name))

    def scale_data(self, robust: bool = False, window_size: int | None = None) -> None:
        scaled = scale_signal(self.proc_data, robust, window_size)

        self._parameters_used["standardize_parameters"] = _t.StandardizeParameters(
            robust=robust,
            window_size=window_size,
        )
        self.data = self.data.with_columns((scaled).alias(self._proc_sig_name))

    def detect_peaks(
        self,
        method: _t.PeakDetectionMethod,
        input_values: _t.PeakDetectionInputValues,
    ) -> None:
        sampling_rate = self.sfreq
        self._parameters_used["peak_detection_parameters"] = _t.PeakDetectionParameters(
            method=method,
            input_values=input_values,
        )
        peaks = find_peaks(self.proc_data.to_numpy(), sampling_rate, method, input_values)
        self.set_peaks(peaks)

    def calculate_rate(self) -> None:
        sampling_rate = self.sfreq
        peaks = self.peaks.to_numpy()
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
        if np.any(peaks < 0):
            raise ValueError("Peaks must be positive integers")
        pl_peaks = pl.Series("peaks", peaks, pl.UInt32)

        self.data = self.data.with_columns(
            pl.when(pl.col("section_index").is_in(pl_peaks)).then(True).otherwise(False).alias("is_peak")
        )
        self.calculate_rate()

    def update_peaks(self, action: t.Literal["add", "remove"], peaks: list[int]) -> None:
        pl_peaks = pl.Series("peaks", peaks, pl.Int32)
        then_value = action == "add"

        self.data = self.data.with_columns(
            pl.when(pl.col("section_index").is_in(pl_peaks))
            .then(then_value)
            .otherwise(pl.col("is_peak"))
            .alias("is_peak")
        )
        if action == "add":
            self._peak_edits.added.extend(peaks)
        else:
            self._peak_edits.removed.extend(peaks)

    def get_peak_edits(self) -> ManualPeakEdits:
        self._peak_edits.sort_and_deduplicate()
        return self._peak_edits

    def get_section_info(self) -> _t.SectionIdentifier:
        return _t.SectionIdentifier(
            sig_name=self.sig_name,
            section_id=self.section_id,
            absolute_bounds=self._abs_bounds,
            sampling_rate=self.sfreq,
        )

    def get_peak_xy(self) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.float64]]:
        peaks = self.peaks.to_numpy()
        return peaks, self.proc_data.gather(peaks).to_numpy()

    def get_focused_result(self) -> FocusedResult:
        peaks = self.peaks
        time = (peaks * (1 / self.sfreq)).round(4).to_numpy()
        intervals = peaks.diff().fill_null(0).to_numpy()
        temperature = self.data.get_column("temperature").gather(peaks).to_numpy()

        return FocusedResult(
            time_s=time,
            index=peaks.to_numpy(),
            peak_intervals=intervals,
            temperature=temperature,
            rate_bpm=self.rate,
        )

    def get_complete_result(self) -> SectionResult:
        return SectionResult(
            sig_name=self.sig_name,
            section_id=self.section_id,
            absolute_bounds=self._abs_bounds,
            data=self.data,
            sampling_rate=self.sfreq,
            peaks=self.peaks,
            peak_edits=self.get_peak_edits(),
            rate=self.rate,
            rate_interpolated=self.rate_interp,
            processing_parameters=self.processing_parameters,
            focused_result=self.get_focused_result().to_structured_array(),
        )


class SectionContainer(OrderedDict[SectionID, Section]):
    def __setitem__(self, key: SectionID, value: Section) -> None:
        super().__setitem__(key, value)
        self.move_to_end(key)

    def __str__(self) -> str:
        return pformat(self.__dict__, indent=4, width=100, compact=True)