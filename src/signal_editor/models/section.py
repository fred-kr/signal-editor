import re
import typing as t
from collections import OrderedDict
from dataclasses import dataclass

import neurokit2 as nk
import numpy as np
import numpy.typing as npt
import polars as pl

from .. import type_aliases as _t
from .peaks import find_peaks
from .result import FocusedResult
from .processing import filter_elgendi, filter_signal, scale_signal


class SectionIndices(t.NamedTuple):
    start: int
    stop: int


def _is_new_section_within_existing(
    new_limits: SectionIndices, existing_limits: SectionIndices
) -> bool:
    return existing_limits.start <= new_limits.start <= new_limits.stop <= existing_limits.stop


def _is_new_section_overlapping_start(
    new_limits: SectionIndices, existing_limits: SectionIndices
) -> bool:
    return new_limits.start < existing_limits.start < new_limits.stop


def _is_new_section_overlapping_stop(
    new_limits: SectionIndices, existing_limits: SectionIndices
) -> bool:
    return existing_limits.start < new_limits.start < existing_limits.stop


def _is_new_section_enclosing_existing(
    new_limits: SectionIndices, existing_limits: SectionIndices
) -> bool:
    return new_limits.start <= existing_limits.start and new_limits.stop >= existing_limits.stop


class SectionID(str):
    def __new__(cls, value: str) -> "SectionID":
        if not re.match(r"^(IN|EX)_[0-9]{3}$", value):
            raise ValueError("SectionID must be of the form 'IN_000' or 'EX_000'")
        return super().__new__(cls, value)


@dataclass(slots=True, frozen=True)
class SectionResult:
    name: str
    section_id: SectionID
    absolute_bounds: SectionIndices
    data: npt.NDArray[np.void]
    sampling_rate: int
    peaks: npt.NDArray[np.uint32]
    rate: npt.NDArray[np.float64]
    rate_interpolated: npt.NDArray[np.float64]
    processing_parameters: _t.ProcessingParameters

    def as_dict(self) -> _t.SectionResultDict:
        return {
            "name": self.name,
            "section_id": self.section_id,
            "absolute_bounds": self.absolute_bounds,
            "data": self.data,
            "sampling_rate": self.sampling_rate,
            "peaks": self.peaks,
            "rate": self.rate,
            "rate_interpolated": self.rate_interpolated,
            "processing_parameters": self.processing_parameters,
        }


class Section:
    _id_counter: int = 0

    def __init__(
        self,
        data: pl.DataFrame,
        name: str,
        sampling_rate: int,
        set_active: bool = False,
        include: bool = True,
    ) -> None:
        self._is_included = include
        self.section_id = self._generate_id()
        self.data = data.with_row_index("section_index")
        self.name = name
        self._sampling_rate = sampling_rate
        self._processed_name = f"{name}_processed"
        self._is_active = set_active
        index_col = data.get_column("index")
        self._abs_bounds = SectionIndices(index_col[0], index_col[-1])
        self._is_finished = False
        self._result: SectionResult | None = None
        self._rate: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
        self._rate_interp: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
        self._parameters_used: _t.ProcessingParameters = {
            "sampling_rate": self.sfreq,
            "filter_parameters": None,
            "standardize_parameters": None,
            "peak_detection_parameters": None,
        }

    def __str__(self) -> str:
        return (
            f"Section:\n"
            f"\tName: {self.name:20s}\n"
            f"\tID: {self.section_id:10s}\n"
            f"\tIndices: {self._abs_bounds:10s}\n"
            f"\tIs active: {self._is_active:10s}\n"
        )

    def __len__(self) -> int:
        return self.data.height

    @staticmethod
    def from_dict(section_dict: _t.SectionResultDict) -> "Section":
        data = pl.DataFrame(section_dict["data"])
        section = Section(
            data,
            name=section_dict["name"],
            sampling_rate=section_dict["sampling_rate"],
            set_active=False,
            include=True,
        )
        section.set_peaks(section_dict["peaks"])
        section._rate = section_dict["rate"]
        section._rate_interp = section_dict["rate_interpolated"]
        section._parameters_used = {
            "sampling_rate": section_dict["sampling_rate"],
            "filter_parameters": section_dict["processing_parameters"]["filter_parameters"],
            "standardize_parameters": section_dict["processing_parameters"][
                "standardize_parameters"
            ],
            "peak_detection_parameters": section_dict["processing_parameters"][
                "peak_detection_parameters"
            ],
        }
        return section

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

    def _generate_id(self) -> SectionID:
        prefix = "IN" if self._is_included else "EX"
        number = self._get_next_id()
        return SectionID(f"{prefix}_{number:03d}")

    def get_section_info(self) -> _t.SectionIdentifier:
        return {
            "section_id": self.section_id,
            "included": self._is_included,
            "signal_name": self.name,
            "absolute_start_index": self.abs_start,
            "absolute_stop_index": self.abs_stop,
            "finished_processing": self._is_finished,
        }

    @property
    def processing_parameters(self) -> _t.ProcessingParameters:
        return self._parameters_used

    @property
    def is_included(self) -> bool:
        return self._is_included

    @property
    def is_active(self) -> bool:
        return self._is_active

    @is_active.setter
    def is_active(self, value: bool) -> None:
        self._is_active = value

    @property
    def abs_start(self) -> int:
        """
        The index value of this section's first row in the original data.

        See Also
        --------
        abs_stop : The index value of this section's last row in the original data.
        sect_start : The first value in the `section_index` column (should always be 0).
        sect_stop : The last value in the `section_index` column (should always be the length of the section minus 1).

        """
        return self._abs_bounds.start

    @property
    def abs_stop(self) -> int:
        """
        The index value of this section's last row in the original data.

        See Also
        --------
        abs_start
        sect_start
        sect_stop

        """
        return self._abs_bounds.stop

    @property
    def sect_start(self) -> int:
        return self.data.get_column("section_index")[0]

    @property
    def sect_stop(self) -> int:
        return self.data.get_column("section_index")[-1]

    @property
    def proc_data(self) -> pl.Series:
        return self.data.get_column(self._processed_name)

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
        if new_stop_index > self.sect_stop:
            raise ValueError("New stop index must be smaller than current stop index")

        new_data = self.data.filter(pl.col("index") < new_stop_index)
        return Section(new_data, self.name, self.sfreq, set_active=self.is_active)

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
        if new_start_index < self.sect_start:
            raise ValueError("New start index must be larger than current start index")

        new_data = self.data.filter(pl.col("index") > new_start_index)
        return Section(new_data, self.name, self.sfreq, set_active=self.is_active)

    def filter_data(
        self,
        pipeline: _t.Pipeline,
        **kwargs: t.Unpack[_t.SignalFilterParameters],
    ) -> None:
        method = kwargs.get("method", "None")
        if pipeline == "custom":
            if method == "None":
                filtered = self.proc_data
            else:
                filtered = filter_signal(self.proc_data.to_numpy(), self.sfreq, **kwargs)
        elif pipeline == "ppg_elgendi":
            filtered = filter_elgendi(self.proc_data.to_numpy(), self.sfreq)
        else:
            raise ValueError(f"Unknown pipeline: {pipeline}")
        filter_parameters = _t.SignalFilterParameters(**kwargs)
        self._parameters_used["filter_parameters"] = filter_parameters
        pl_filtered = pl.Series(self._processed_name, filtered, pl.Float64)
        self.data = self.data.with_columns((pl_filtered).alias(self._processed_name))

    def scale_data(self, robust: bool = False, window_size: int | None = None) -> None:
        scaled = scale_signal(self.proc_data, robust, window_size)

        self._parameters_used["standardize_parameters"] = _t.StandardizeParameters(
            robust=robust,
            window_size=window_size,
        )
        self.data = self.data.with_columns((scaled).alias(self._processed_name))

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
            pl.when(pl.col("index").is_in(pl_peaks)).then(True).otherwise(False).alias("is_peak")
        )
        self.calculate_rate()

    def get_peak_xy(self) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.float64]]:
        peaks = self.peaks.to_numpy()
        return peaks, self.proc_data.gather(peaks).to_numpy()

    def get_focused_result(self) -> FocusedResult:
        peaks = self.peaks
        time = (peaks * (1 / self.sfreq)).round(4).to_numpy()
        intervals = peaks.diff().fill_null(0).to_numpy()

        return FocusedResult(
            time_s=time,
            index=peaks.to_numpy(),
            peak_intervals=intervals,
            temperature=self.data.get_column("temperature").to_numpy(),
            rate_bpm=self._rate,
        )

    def as_dict(self) -> _t.SectionResultDict:
        return _t.SectionResultDict(
            name=self.name,
            section_id=self.section_id,
            absolute_bounds=self._abs_bounds,
            data=self.data.to_numpy(structured=True),
            sampling_rate=self.sfreq,
            peaks=self.peaks.to_numpy(),
            rate=self._rate,
            rate_interpolated=self._rate_interp,
            processing_parameters=self._parameters_used,
        )


class SectionContainer:
    def __init__(self, name: str, base_section: Section) -> None:
        self._name = name
        self._included: OrderedDict[SectionID, Section] = OrderedDict()
        self._excluded: OrderedDict[SectionID, Section] = OrderedDict()
        self._base_section = base_section

    def __str__(self) -> str:
        included_str = "\n\t\t".join(str(section) for section in self._included.values())
        excluded_str = "\n\t\t".join(str(section) for section in self._excluded.values())

        return (
            f"SectionContainer: {self._name}\n"
            f"\tIncluded:\n"
            f"\t\t{included_str}\n"
            f"\tExcluded:\n"
            f"\t\t{excluded_str}\n"
        )

    @property
    def base_section(self) -> Section:
        return self._base_section

    def get_previous_section(self, section_id: SectionID) -> Section:
        section_ids = list(self._included.keys())
        index = section_ids.index(section_id)
        if index == 0:
            return self.base_section
        prev_id = section_ids[index - 1]
        try:
            return self.get_section(prev_id)
        except ValueError:
            raise

    def get_next_section(self, section_id: SectionID) -> Section:
        section_ids = list(self._included.keys())
        index = section_ids.index(section_id)
        if index == len(section_ids) - 1:
            return self.get_section(section_id)
        next_id = section_ids[index + 1]
        try:
            return self.get_section(next_id)
        except ValueError:
            raise

    def get_section(self, section_id: SectionID) -> Section:
        if "IN" in section_id:
            return self._included[section_id]
        elif "EX" in section_id:
            return self._excluded[section_id]
        else:
            raise ValueError(f"Section '{section_id}' does not exist")

    def get_included(self) -> list[Section]:
        return list(self._included.values())

    def get_excluded(self) -> list[Section]:
        return list(self._excluded.values())

    def get_active_section(self) -> Section:
        section = next((section for section in self._included.values() if section.is_active), None)
        if section is None:
            section_ids = list(self._included.keys())
            section = self.get_section(section_ids[0])
        return section

    def set_active_section(self, section_id: SectionID) -> None:
        for section in self._included.values():
            section.is_active = False
        self._included[section_id].is_active = True

    def add_section(self, section: Section) -> None:
        new_limits = SectionIndices(section.abs_start, section.abs_stop)
        existing_sections = list(self._included.values())
        for existing_section in existing_sections:
            existing_limits = SectionIndices(existing_section.abs_start, existing_section.abs_stop)
            if new_limits.stop < existing_limits.start or new_limits.start > existing_limits.stop:
                continue
            if _is_new_section_within_existing(new_limits, existing_limits):
                self._handle_new_within_existing(existing_section, section)
                return
            elif _is_new_section_enclosing_existing(new_limits, existing_limits):
                self._handle_new_enclosing_existing(existing_section, section)
                return
            elif _is_new_section_overlapping_start(new_limits, existing_limits):
                self._handle_new_overlapping_start(existing_section, section)
            elif _is_new_section_overlapping_stop(new_limits, existing_limits):
                self._handle_new_overlapping_stop(existing_section, section)

        if section.is_included:
            self._included[section.section_id] = section
        else:
            self._excluded[section.section_id] = section

        if section.is_active:
            self.set_active_section(section.section_id)

    def _handle_new_within_existing(self, existing_section: Section, new_section: Section) -> None:
        left_section = existing_section.left_shrink(new_section.abs_start)
        right_section = existing_section.right_shrink(new_section.abs_stop)
        self._included.pop(existing_section.section_id, None)
        self._included[left_section.section_id] = left_section
        self._included[new_section.section_id] = new_section
        self._included[right_section.section_id] = right_section

    def _handle_new_overlapping_start(
        self, existing_section: Section, new_section: Section
    ) -> None:
        existing_updated = existing_section.right_shrink(new_section.abs_stop)
        self._included.pop(existing_section.section_id, None)
        self._included[new_section.section_id] = new_section
        self._included[existing_updated.section_id] = existing_updated

    def _handle_new_overlapping_stop(self, existing_section: Section, new_section: Section) -> None:
        existing_updated = existing_section.left_shrink(new_section.abs_start)
        self._included.pop(existing_section.section_id, None)
        self._included[existing_updated.section_id] = existing_updated
        self._included[new_section.section_id] = new_section

    def _handle_new_enclosing_existing(
        self, existing_section: Section, new_section: Section
    ) -> None:
        self._included.pop(existing_section.section_id, None)
        self.add_section(new_section)
