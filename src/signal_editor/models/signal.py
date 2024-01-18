import copy
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, NamedTuple, TypedDict, Unpack

import neurokit2 as nk
import numpy as np
import polars as pl
from numpy.typing import NDArray
from PySide6.QtCore import QObject, Signal, Slot

from ..type_aliases import (
    PeakDetectionInputValues,
    PeakDetectionMethod,
    PeakDetectionParameters,
    Pipeline,
    SignalFilterParameters,
    SignalName,
    StandardizeParameters,
)
from .filters import (
    auto_correct_fir_length,
    filter_custom,
    filter_elgendi,
    scale_signal,
)
from .peaks import find_peaks


class InitialState(TypedDict):
    name: str
    sampling_rate: int
    data: pl.DataFrame


class SignalDataDict(TypedDict):
    name: str
    sampling_rate: int
    data: NDArray[np.void]
    original_data: NDArray[np.void]
    rate_data: "RateData"
    peaks: NDArray[np.int32]


class SectionDict(TypedDict):
    section_id: "SectionID"
    name: str
    data: NDArray[np.void]
    peaks: NDArray[np.int32]
    rate: NDArray[np.float64]
    rate_interpolated: NDArray[np.float64]
    filter_parameters: SignalFilterParameters | None
    standardize_parameters: StandardizeParameters | None
    peak_detection_parameters: PeakDetectionParameters


def ensure_correct_order(start: int, stop: int) -> tuple[int, int]:
    return (stop, start) if start > stop else (start, stop)


def _is_new_section_within_existing(
    new_limits: tuple[int, int], existing_limits: tuple[int, int]
) -> bool:
    return new_limits[0] > existing_limits[0] and new_limits[1] < existing_limits[1]


def _is_new_section_overlapping_start(
    new_limits: tuple[int, int], existing_limits: tuple[int, int]
) -> bool:
    return new_limits[0] < existing_limits[0] and new_limits[1] < existing_limits[1]


def _is_new_section_overlapping_stop(
    new_limits: tuple[int, int], existing_limits: tuple[int, int]
) -> bool:
    return new_limits[0] > existing_limits[0] and new_limits[1] > existing_limits[1]


def _is_new_section_enclosing_existing(
    new_limits: tuple[int, int], existing_limits: tuple[int, int]
) -> bool:
    return new_limits[0] < existing_limits[0] and new_limits[1] > existing_limits[1]


class SectionIndices(NamedTuple):
    start: int
    stop: int


@dataclass(slots=True, kw_only=True)
class RateData:
    """
    Holds both the interpolated and non-interpolated rate data for each signal. Results come from the
    `neurokit2.signal_rate` function.
    """

    rate: NDArray[np.float64] = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    rate_interpolated: NDArray[np.float64] = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )

    def update(
        self,
        rate: NDArray[np.float64] | None = None,
        rate_interpolated: NDArray[np.float64] | None = None,
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


class Section:
    _id_counters: dict[str, int] = {}

    def __init__(
        self,
        data: pl.DataFrame,
        name: str,
        sampling_rate: int,
        set_active: bool = False,
        include: bool = True,
    ) -> None:
        self._is_included = include
        self.section_id = self._generate_id(name)
        self.data = data
        self.name = name
        self._sampling_rate = sampling_rate
        self._processed_name = f"{name}_processed"
        self._is_active = set_active
        index_col = data.get_column("index")
        self._indices = SectionIndices(index_col[0], index_col[-1])
        self._is_finished = False
        self.rate_data = RateData()
        self._parameters_used: dict[str, Any] = {
            "filter": None,
            "standardize": None,
            "peak_detection": None,
        }

    def __repr__(self) -> str:
        return (
            f"Section:\n"
            f"\tName: {self.name}\n"
            f"\tID: {self.section_id}\n"
            f"\tIndices: {self._indices}\n"
            f"\tIs active: {self._is_active}"
        )

    @classmethod
    def _get_next_id(cls, name: str) -> int:
        if name not in cls._id_counters:
            cls._id_counters[name] = 0
        cls._id_counters[name] += 1
        return cls._id_counters[name]

    @classmethod
    def reset_id_counter(cls, name: str) -> None:
        if name in cls._id_counters:
            cls._id_counters[name] = 0

    def _generate_id(self, name: str) -> SectionID:
        prefix = "IN" if self._is_included else "EX"
        number = self._get_next_id(name)
        return SectionID(f"{prefix}_{number:03d}")

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
    def start_index(self) -> int:
        return self._indices.start

    @property
    def stop_index(self) -> int:
        return self._indices.stop

    @property
    def processed_signal(self) -> NDArray[np.float64]:
        return self.data.get_column(self._processed_name).to_numpy()

    @property
    def sampling_rate(self) -> int:
        return self._sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, value: int) -> None:
        self._sampling_rate = value

    def resize(self, start_index: int, stop_index: int, data: pl.DataFrame | None = None) -> None:
        start_index, stop_index = ensure_correct_order(start_index, stop_index)
        if start_index == self.start_index and stop_index == self.stop_index:
            return
        if (start_index < self.start_index or stop_index > self.stop_index) and data is None:
            raise ValueError(
                "The `data` argument is required if the new indices are outside the current data range."
            )
        self._indices = SectionIndices(start_index, stop_index)
        if data is not None:
            self.data = data
        else:
            self.data = self.data.filter(pl.col("index").is_between(start_index, stop_index))

    def filter_signal(
        self,
        pipeline: Pipeline,
        **kwargs: Unpack[SignalFilterParameters],
    ) -> None:
        method = kwargs.get("method", "None")
        sig = self.processed_signal
        sampling_rate = self.sampling_rate
        if pipeline == "custom":
            if method == "None":
                filtered = sig
            elif method == "fir":
                filtered = auto_correct_fir_length(sig, sampling_rate, **kwargs)
            else:
                filtered = filter_custom(sig, sampling_rate, **kwargs)
        elif pipeline == "ppg_elgendi":
            filtered = filter_elgendi(sig, sampling_rate)
        else:
            raise ValueError(f"Unknown pipeline: {pipeline}")
        filter_parameters_used = {
            "pipeline": pipeline,
            "method": method,
            "method_parameters": kwargs,
        }
        self._parameters_used["filter"] = filter_parameters_used
        pl_filtered = pl.Series(self._processed_name, filtered, pl.Float64)
        self.data = self.data.with_columns((pl_filtered).alias(self._processed_name))

    def scale_signal(self, robust: bool = False, window_size: int | None = None) -> None:
        scaled = scale_signal(self.processed_signal, robust, window_size)

        self._parameters_used["standardize"] = {
            "robust": robust,
            "window_size": window_size,
        }
        self.data = self.data.with_columns(
            pl.Series(self._processed_name, scaled, pl.Float64).alias(self._processed_name)
        )

    def detect_peaks(
        self,
        method: PeakDetectionMethod,
        input_values: PeakDetectionInputValues,
    ) -> None:
        sampling_rate = self.sampling_rate
        peaks = find_peaks(self.processed_signal, sampling_rate, method, input_values)
        pl_peaks = pl.Series("peaks", peaks, pl.Int32)
        self._parameters_used["peak_detection"] = {
            "method": method,
            "input_values": input_values,
        }
        self.data = self.data.with_columns(
            (pl.when(pl.col("index").is_in(pl_peaks)).then(True).otherwise(False)).alias("is_peak")
        )

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

    def get_peaks(self) -> NDArray[np.int32]:
        if not self._is_included:
            return np.empty(0, dtype=np.int32)
        return (
            self.data.filter(pl.col("is_peak"))
            .get_column("index")
            .cast(pl.Int32)
            .to_numpy(writable=True)
        )

    def set_peaks(self, peaks: NDArray[np.int32]) -> None:
        pl_peaks = pl.Series("peaks", peaks, pl.Int32)

        self.data = self.data.with_columns(
            pl.when(pl.col("index").is_in(pl_peaks)).then(True).otherwise(False).alias("is_peak")
        )

    def get_peak_intervals(self) -> NDArray[np.int32]:
        if not self._is_included:
            return np.empty(0, dtype=np.int32)
        peaks = self.get_peaks()
        return np.ediff1d(peaks, to_begin=[0])

    def set_finished(self) -> None:
        self._is_finished = True

    def as_dict(self) -> SectionDict:
        return SectionDict(
            section_id=self.section_id,
            name=self.name,
            data=self.data.to_numpy(structured=True),
            peaks=self.get_peaks(),
            rate=self.rate_data.rate,
            rate_interpolated=self.rate_data.rate_interpolated,
            filter_parameters=self._parameters_used.get("filter", None),
            standardize_parameters=self._parameters_used.get("standardize", None),
            peak_detection_parameters=self._parameters_used.get("peak_detection", None),
        )


class SectionContainer:
    def __init__(self, name: str) -> None:
        self._name = name
        self._included: dict[SectionID, Section] = {}
        self._excluded: dict[SectionID, Section] = {}

    def __repr__(self) -> str:
        return (
            f"SectionContainer:\n"
            f"\tIncluded:\n"
            f"\t\t{self._included}\n"
            f"\tExcluded:\n"
            f"\t\t{self._excluded}\n"
        )

    def get_previous_section(self, section_id: SectionID) -> Section:
        if section_id in self._included:
            prev_id = SectionID(f"IN_{int(section_id[3:]) - 1:03d}")
        elif section_id in self._excluded:
            prev_id = SectionID(f"EX_{int(section_id[3:]) - 1:03d}")
        else:
            raise ValueError("Section not found")

        return self._included[prev_id] if prev_id in self._included else self._excluded[prev_id]

    def get_next_section(self, section_id: SectionID) -> Section:
        if section_id in self._included:
            next_id = SectionID(f"IN_{int(section_id[3:]) + 1:03d}")
        elif section_id in self._excluded:
            next_id = SectionID(f"EX_{int(section_id[3:]) + 1:03d}")
        else:
            raise ValueError("Section not found")

        return self._included[next_id] if next_id in self._included else self._excluded[next_id]

    def get_section_by_id(self, section_id: SectionID) -> Section:
        if section_id.startswith("IN"):
            return self._included[section_id]
        elif section_id.startswith("EX"):
            return self._excluded[section_id]
        else:
            raise ValueError("Section not found")

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
        if not section.is_included:
            self._excluded[section.section_id] = section
            return

        new_limits = section.start_index, section.stop_index
        right_section = None
        for existing_section in self._included.values():
            existing_limits = existing_section.start_index, existing_section.stop_index

            if _is_new_section_within_existing(new_limits, existing_limits):
                right_section = self._handle_new_within_existing(existing_section, section)
            elif _is_new_section_enclosing_existing(new_limits, existing_limits):
                self.remove_section(existing_section.section_id)
            elif _is_new_section_overlapping_start(new_limits, existing_limits):
                self._handle_new_overlapping_start(existing_section, section)
            elif _is_new_section_overlapping_stop(new_limits, existing_limits):
                self._handle_new_overlapping_stop(existing_section, section)

        self._included[section.section_id] = section
        if right_section is not None:
            self._included[right_section.section_id] = right_section

        if section.is_active:
            self.set_active_section(section.section_id)

    def _handle_new_within_existing(
        self, existing_section: Section, new_section: Section
    ) -> Section:
        right_data = existing_section.data.filter(
            pl.col("index").is_between(new_section.stop_index + 1, existing_section.stop_index)
        )
        right_section = Section(
            right_data, name=self._name, sampling_rate=existing_section.sampling_rate
        )
        existing_section.resize(existing_section.start_index, new_section.start_index - 1)
        return right_section

    def _handle_new_overlapping_start(
        self, existing_section: Section, new_section: Section
    ) -> None:
        adjusted_start = new_section.stop_index + 1
        existing_section.resize(adjusted_start, existing_section.stop_index)

    def _handle_new_overlapping_stop(self, existing_section: Section, new_section: Section) -> None:
        adjusted_stop = new_section.start_index - 1
        existing_section.resize(existing_section.start_index, adjusted_stop)

    def remove_section(self, section_id: SectionID) -> None:
        if section_id in self._included and self._included[section_id].is_active:
            try:
                new_active = self.get_previous_section(section_id)
            except ValueError:
                try:
                    new_active = self.get_next_section(section_id)
                except ValueError:
                    new_active = list(self._included.values())[0]
            self.set_active_section(new_active.section_id)
        self._included.pop(section_id, None)
        self._excluded.pop(section_id, None)


class SignalData(QObject):
    sig_new_section_id = Signal(str)

    def __init__(
        self,
        name: SignalName | str,
        data: pl.DataFrame,
        sampling_rate: int,
    ) -> None:
        super().__init__()
        self.name: str = name
        self.data: pl.DataFrame = data
        self.sampling_rate: int = sampling_rate
        self._initial_state: InitialState = {
            "name": self.name,
            "sampling_rate": self.sampling_rate,
            "data": self.data,
        }
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
            self.data = self.data.with_row_count("index")
        self.data = self.data.select(
            pl.col("index", "temperature", self.name),
            pl.col(self.name).alias(self.processed_name),
            pl.repeat(False, self.data.height, dtype=pl.Boolean).alias("is_peak"),
            pl.repeat(True, self.data.height, dtype=pl.Boolean).alias("is_included"),
            pl.repeat(False, self.data.height, dtype=pl.Boolean).alias("is_processed"),
        )
        self._original_data = self.data.clone()
        active_data = self.data.select(
            "index", "temperature", self.name, self.processed_name, "is_peak"
        )

        section = Section(
            active_data,
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
    def active_peaks(self) -> NDArray[np.int32]:
        return self.active_section.get_peaks()

    @property
    def data_bounds(self) -> tuple[int, int]:
        col = self.data.get_column("index")
        return col[0], col[-1]

    @property
    def active_region_limits(self) -> tuple[int, int]:
        return self.active_section.start_index, self.active_section.stop_index

    @property
    def default_rate(self) -> NDArray[np.float64]:
        return self._signal_rate.rate

    @property
    def interpolated_rate(self) -> NDArray[np.float64]:
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
                .is_between(excluded_section.start_index, excluded_section.stop_index)
                .is_not()
            )
        for section in self.sections.get_included().values():
            section_data = section.data.select(pl.col("index", self.processed_name, "is_peak"))
            self.data.update(section_data, on="index")

    def add_section(
        self, lower: int, upper: int, set_active: bool = False, include: bool = True
    ) -> None:
        data = self.data.filter(pl.col("index").is_between(lower, upper))
        new_section = Section(
            data,
            name=self.name,
            set_active=set_active,
            include=include,
            sampling_rate=self.sampling_rate,
        )
        self.sections.add_section(new_section)
        self.sig_new_section_id.emit(new_section.section_id)

    def get_all_peaks(self) -> NDArray[np.int32]:
        self.save_section_changes()
        return self.data.filter(pl.col("is_peak")).get_column("index").cast(pl.Int32).to_numpy()

    def get_active_signal(self) -> NDArray[np.float64]:
        return self.active_section.processed_signal

    def get_calculated_rate(
        self, interpolated: bool | None = None
    ) -> RateData | NDArray[np.float64]:
        rate_mapping = {
            None: self._signal_rate,
            True: self._signal_rate.rate_interpolated,
            False: self._signal_rate.rate,
        }
        return rate_mapping.get(interpolated, self._signal_rate)

    def as_dict(self) -> dict[SectionID, SectionDict]:
        return {
            section.section_id: section.as_dict()
            for section in self.sections.get_included().values()
        }

    def set_active_section(self, section_id: SectionID) -> None:
        self.sections.set_active_section(section_id)

    def reset(self) -> None:
        self.name = self._initial_state["name"]
        self.sampling_rate = self._initial_state["sampling_rate"]
        self.data = self._initial_state["data"]
        Section.reset_id_counter(name=self.name)
        self._finish_init()

    def remove_all_excluded(self) -> None:
        excluded_sections = self.sections.get_excluded()
        for section in excluded_sections.values():
            self.data = self.data.filter(
                pl.col("index").is_between(section.start_index, section.stop_index).is_not()
            )

    def restore_excluded(self) -> None:
        excluded_sections = self.sections.get_excluded()
        for section in excluded_sections.values():
            self.data.update(section.data, on="index", how="outer")

    def set_active_peak_indices(self, indices: Iterable[int]) -> None:
        indices = pl.Series("peaks", indices, pl.Int32)
        self.active_section.data = self.active_section.data.with_columns(
            pl.when(pl.col("index").is_in(indices))
            .then(pl.lit(True))
            .otherwise(pl.col("is_peak"))
            .alias("is_peak")
        )


class SignalStorage(dict[str, SignalData]):
    """
    A container for SignalData objects.
    """

    def deepcopy(self) -> "SignalStorage":
        return copy.deepcopy(self)

    def update_sampling_rate(self, sampling_rate: int) -> None:
        for sig in self.values():
            sig.sampling_rate = sampling_rate
