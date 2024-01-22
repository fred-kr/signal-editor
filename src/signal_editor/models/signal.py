import re
import typing as t
from dataclasses import dataclass, field

import neurokit2 as nk
import numpy as np
import numpy.typing as npt
import polars as pl
import polars.selectors as ps
from PySide6.QtCore import QObject, Signal, Slot

from .. import type_aliases as _t
from . import filters as _filters
from . import peaks as _peaks
from . import result as _result

if t.TYPE_CHECKING:
    from .result import Result


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


@dataclass(slots=True, kw_only=True)
class RateData:
    """
    Holds both the interpolated and non-interpolated rate data for each signal. Results come from the
    `neurokit2.signal_rate` function.
    """

    rate: npt.NDArray[np.float64] = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    rate_interpolated: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )

    def update(
        self,
        rate: npt.NDArray[np.float64] | None = None,
        rate_interpolated: npt.NDArray[np.float64] | None = None,
    ) -> None:
        if rate is not None:
            self.rate = rate
        if rate_interpolated is not None:
            self.rate_interpolated = rate_interpolated

    def clear(self) -> None:
        self.rate = np.empty(0, dtype=np.float64)
        self.rate_interpolated = np.empty(0, dtype=np.float64)


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
    peaks: npt.NDArray[np.int32]
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
        self.data = data.with_row_index("relative_index")
        self.name = name
        self._sampling_rate = sampling_rate
        self._processed_name = f"{name}_processed"
        self._is_active = set_active
        index_col = data.get_column("index")
        self._abs_bounds = SectionIndices(index_col[0], index_col[-1])
        self._is_finished = False
        self._result: SectionResult | None = None
        self.rate_data = RateData()
        self._parameters_used: _t.ProcessingParameters = {
            "sampling_rate": self.sampling_rate,
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
        section.rate_data.update(
            rate=section_dict["rate"],
            rate_interpolated=section_dict["rate_interpolated"],
        )
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
        rel_start : The first value in the `relative_index` column (should always be 0).
        rel_stop : The last value in the `relative_index` column (should always be the length of the section minus 1).

        """
        return self._abs_bounds.start

    @property
    def abs_stop(self) -> int:
        """
        The index value of this section's last row in the original data.

        See Also
        --------
        abs_start
        rel_start
        rel_stop

        """
        return self._abs_bounds.stop

    @property
    def rel_start(self) -> int:
        return self.data.get_column("relative_index")[0]

    @property
    def rel_stop(self) -> int:
        return self.data.get_column("relative_index")[-1]

    @property
    def processed_signal(self) -> npt.NDArray[np.float64]:
        return self.data.get_column(self._processed_name).to_numpy()

    @property
    def sampling_rate(self) -> int:
        return self._sampling_rate

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
        if new_stop_index > self.rel_stop:
            raise ValueError("New stop index must be smaller than current stop index")

        new_data = self.data.filter(pl.col("index") < new_stop_index)
        return Section(new_data, self.name, self.sampling_rate, set_active=self.is_active)

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
        if new_start_index < self.rel_start:
            raise ValueError("New start index must be larger than current start index")

        new_data = self.data.filter(pl.col("index") > new_start_index)
        return Section(new_data, self.name, self.sampling_rate, set_active=self.is_active)

    def filter_signal(
        self,
        pipeline: _t.Pipeline,
        **kwargs: t.Unpack[_t.SignalFilterParameters],
    ) -> None:
        method = kwargs.get("method", "None")
        if pipeline == "custom":
            if method == "None":
                filtered = self.processed_signal
            elif method == "fir":
                filtered = _filters.auto_correct_fir_length(
                    self.processed_signal, self.sampling_rate, **kwargs
                )
            else:
                filtered = _filters.filter_custom(
                    self.processed_signal, self.sampling_rate, **kwargs
                )
        elif pipeline == "ppg_elgendi":
            filtered = _filters.filter_elgendi(self.processed_signal, self.sampling_rate)
        else:
            raise ValueError(f"Unknown pipeline: {pipeline}")
        filter_parameters = _t.SignalFilterParameters(**kwargs)
        self._parameters_used["filter_parameters"] = filter_parameters
        pl_filtered = pl.Series(self._processed_name, filtered, pl.Float64)
        self.data = self.data.with_columns((pl_filtered).alias(self._processed_name))

    def scale_signal(self, robust: bool = False, window_size: int | None = None) -> None:
        scaled = _filters.scale_signal(self.processed_signal, robust, window_size)

        self._parameters_used["standardize_parameters"] = {
            "robust": robust,
            "window_size": window_size,
        }
        self.data = self.data.with_columns((scaled).alias(self._processed_name))

    def detect_peaks(
        self,
        method: _t.PeakDetectionMethod,
        input_values: _t.PeakDetectionInputValues,
    ) -> None:
        sampling_rate = self.sampling_rate
        peaks = _peaks.find_peaks(self.processed_signal, sampling_rate, method, input_values)
        self._parameters_used["peak_detection_parameters"] = {
            "method": method,
            "input_values": input_values,
        }
        self.set_peaks(peaks)

    def calculate_rate(self) -> None:
        sampling_rate = self.sampling_rate
        peaks = self.get_peaks()
        rate = np.asarray(
            nk.signal_rate(
                peaks,
                sampling_rate,
                desired_length=None,
                interpolation_method="monotone_cubic",
            ),
            dtype=np.float64,
        )
        rate_interp = np.asarray(
            nk.signal_rate(
                peaks,
                sampling_rate,
                desired_length=len(self.processed_signal),
                interpolation_method="monotone_cubic",
            ),
            dtype=np.float64,
        )
        self.rate_data = RateData(rate=rate, rate_interpolated=rate_interp)

    def get_peaks(self) -> npt.NDArray[np.int32]:
        if not self._is_included:
            return np.empty(0, dtype=np.int32)
        return self.data.get_column("is_peak").arg_true().cast(pl.Int32).to_numpy()

    def set_peaks(self, peaks: npt.NDArray[np.int32]) -> None:
        pl_peaks = pl.Series("peaks", peaks, pl.Int32)

        self.data = self.data.with_columns(
            pl.when(pl.col("index").is_in(pl_peaks)).then(True).otherwise(False).alias("is_peak")
        )

    def get_peak_xy(self) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.float64]]:
        peaks = self.get_peaks()
        return peaks, self.processed_signal[peaks]

    def get_peak_intervals(self) -> npt.NDArray[np.int32]:
        if not self._is_included:
            return np.empty(0, dtype=np.int32)
        peaks = self.get_peaks()
        return np.ediff1d(peaks, to_begin=[0])

    def set_finished(self) -> None:
        self._is_finished = True

    def get_focused_result(self) -> _result.FocusedResult:
        return _result.FocusedResult(
            time_s=self.data.get_column("index").to_numpy() / self.sampling_rate,
            index=self.get_peaks(),
            peak_intervals=self.get_peak_intervals(),
            temperature=self.data.get_column("temperature").to_numpy(),
            rate_bpm=self.rate_data.rate,
        )

    def as_dict(self) -> _t.SectionResultDict:
        return _t.SectionResultDict(
            name=self.name,
            section_id=self.section_id,
            absolute_bounds=self._abs_bounds,
            data=self.data.to_numpy(structured=True),
            sampling_rate=self.sampling_rate,
            peaks=self.get_peaks(),
            rate=self.rate_data.rate,
            rate_interpolated=self.rate_data.rate_interpolated,
            processing_parameters=self._parameters_used,
        )


class SectionContainer:
    def __init__(self, name: str) -> None:
        self._name = name
        self._included: dict[SectionID, Section] = {}
        self._excluded: dict[SectionID, Section] = {}

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

    def get_previous_section(self, section_id: SectionID) -> Section:
        number = int(section_id[-3:]) - 1
        if number <= 0:
            return self.get_section_by_id(section_id)
        prev_id = SectionID(f"IN_{number:03d}")
        try:
            return self.get_section_by_id(prev_id)
        except ValueError:
            raise

    def get_next_section(self, section_id: SectionID) -> Section:
        number = int(section_id[3:]) + 1
        if number > Section.get_id_counter():
            return self.get_section_by_id(section_id)
        next_id = SectionID(f"IN_{number:03d}")
        try:
            return self.get_section_by_id(next_id)
        except ValueError:
            raise

    def get_section_by_id(self, section_id: SectionID) -> Section:
        if "IN" in section_id:
            return self._included[section_id]
        elif "EX" in section_id:
            return self._excluded[section_id]
        else:
            raise ValueError(f"Section '{section_id}' does not exist")

    def get_included(self) -> dict[SectionID, Section]:
        return self._included

    def get_excluded(self) -> dict[SectionID, Section]:
        return self._excluded

    def get_active_section(self) -> Section:
        section = next((section for section in self._included.values() if section.is_active), None)
        if section is None:
            section = self._included[list(self._included.keys())[0]]
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


class SignalData(QObject):
    sig_new_section_id = Signal(str)

    def __init__(
        self,
        name: str,
        data: pl.DataFrame,
        sampling_rate: int,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.name: str = name
        self.data: pl.DataFrame = data
        self.sampling_rate: int = sampling_rate
        self._initial_state: _t.InitialState = {
            "name": self.name,
            "sampling_rate": self.sampling_rate,
            "data": self.data,
        }
        self.result_data: "dict[str, pl.DataFrame | Result]" = {}
        self._finish_init()

    def _finish_init(self) -> None:
        self.processed_name: str = f"{self.name}_processed"
        self.is_finished: bool = False
        self._signal_rate: RateData = RateData()
        self._peak_index_offset: int = 0
        self.sections: SectionContainer = SectionContainer(self.name)
        self._setup_data()

    def _setup_data(self) -> None:
        if "index" not in self.data.columns:
            self.data = self.data.with_row_index()
        self.data = self.data.select(
            pl.col("index"),
            ps.contains("temp"),
            pl.col(self.name),
            pl.col(self.name).alias(self.processed_name),
            pl.repeat(False, self.data.height, dtype=pl.Boolean).alias("is_peak"),
        )
        self._original_data = self.data.clone()

        self.add_section(0, self.data.height - 1, set_active=True, include=True)
        section = Section(
            self.data,
            name=self.name,
            set_active=True,
            include=True,
            sampling_rate=self.sampling_rate,
        )
        self._default_section = section
        self.sections.add_section(section)

    @property
    def active_section(self) -> Section:
        return self.sections.get_active_section()

    @property
    def active_peaks(self) -> npt.NDArray[np.int32]:
        return self.active_section.get_peaks()

    @property
    def data_bounds(self) -> tuple[int, int]:
        col = self.data.get_column("index")
        return col[0], col[-1]

    @property
    def active_abs_bounds(self) -> tuple[int, int]:
        """
        The start and stop row indices of the active section in the original data.
        """
        return self.active_section.abs_start, self.active_section.abs_stop

    @property
    def default_rate(self) -> npt.NDArray[np.float64]:
        """
        Calculated instantaneous rate array with the same length as the amount of detected peaks.
        """
        return self._signal_rate.rate

    @property
    def interpolated_rate(self) -> npt.NDArray[np.float64]:
        """
        Calculated instantaneous rate array interpolated to the length of the original signal."""
        return self._signal_rate.rate_interpolated

    def get_section_by_id(self, section_id: SectionID) -> Section:
        return self.sections.get_section_by_id(section_id)

    @Slot(str)
    def next_section(self, section_id: SectionID) -> None:
        self.sections.get_next_section(section_id)

    @Slot(str)
    def previous_section(self, section_id: SectionID) -> None:
        self.sections.get_previous_section(section_id)

    def set_to_default(self) -> None:
        self.set_active_section(self._default_section.section_id)

    def save_section_changes(self) -> None:
        for excluded_section in self.sections.get_excluded().values():
            self.data = self.data.filter(
                pl.col("index")
                .is_between(excluded_section.abs_start, excluded_section.abs_stop)
                .is_not()
            )
        for section in self.sections.get_included().values():
            section_data = section.data.select(pl.col("index", self.processed_name, "is_peak"))
            self.data.update(section_data, on="index")

    def add_section(
        self, start: int, stop: int, set_active: bool = False, include: bool = True
    ) -> None:
        data = self.data.filter(pl.col("index").is_between(start, stop))
        new_section = Section(
            data,
            name=self.name,
            set_active=set_active,
            include=include,
            sampling_rate=self.sampling_rate,
        )
        self.sections.add_section(new_section)
        self.sig_new_section_id.emit(new_section.section_id)

    def get_all_peaks(self) -> npt.NDArray[np.int32]:
        self.save_section_changes()
        return self.data.filter(pl.col("is_peak")).get_column("index").cast(pl.Int32).to_numpy()

    def get_active_signal(self) -> npt.NDArray[np.float64]:
        return self.active_section.processed_signal

    def get_calculated_rate(
        self, interpolated: bool | None = None
    ) -> RateData | npt.NDArray[np.float64]:
        rate_mapping = {
            None: self._signal_rate,
            True: self._signal_rate.rate_interpolated,
            False: self._signal_rate.rate,
        }
        return rate_mapping.get(interpolated, self._signal_rate)

    def as_dict(self) -> list[dict[SectionID, _t.SectionResultDict]]:
        return [
            {
                section.section_id: section.as_dict()
                for section in self.sections.get_included().values()
            },
            {
                section.section_id: section.as_dict()
                for section in self.sections.get_excluded().values()
            },
        ]

    def set_active_section(self, section_id: SectionID) -> None:
        self.sections.set_active_section(section_id)

    def reset(self) -> None:
        self.name = self._initial_state["name"]
        self.sampling_rate = self._initial_state["sampling_rate"]
        self.data = self._initial_state["data"]
        Section.reset_id_counter()
        self._finish_init()

    def remove_all_excluded(self) -> None:
        excluded_sections = self.sections.get_excluded()
        for section in excluded_sections.values():
            self.data = self.data.filter(
                pl.col("index").is_between(section.abs_start, section.abs_stop).is_not()
            )

    def restore_excluded(self) -> None:
        excluded_sections = self.sections.get_excluded()
        for section in excluded_sections.values():
            self.data.update(section.data, on="index", how="outer")

    def set_active_peak_indices(self, indices: t.Iterable[int]) -> None:
        indices = pl.Series("peaks", indices, pl.Int32)
        self.active_section.data = self.active_section.data.with_columns(
            pl.when(pl.col("index").is_in(indices))
            .then(pl.lit(True))
            .otherwise(pl.col("is_peak"))
            .alias("is_peak")
        )

    # TODO: Actually implement this nicely
    def create_intermediate_result(self, section_ids: list[SectionID]) -> pl.DataFrame:
        sections = [self.sections.get_section_by_id(section_id) for section_id in section_ids]
        return pl.concat([section.data for section in sections])
