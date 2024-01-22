import datetime
import typing as t
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import polars as pl

from .. import type_aliases as _t
from .signal import SignalData


@dataclass(slots=True, frozen=True)
class DescriptiveStatistics:
    _data_description: str
    mean: np.floating[t.Any]
    median: np.floating[t.Any]
    std: np.floating[t.Any]
    var: np.floating[t.Any]

    def as_dict(self) -> dict[str, float]:
        return {
            "mean": self.mean.round(2).astype(float),
            "median": self.median.round(2).astype(float),
            "std": self.std.round(2).astype(float),
            "var": self.var.round(2).astype(float),
        }


@dataclass(slots=True, frozen=True)
class ResultIdentifier:
    name: str
    animal_id: str
    oxygen_condition: str
    source_file_name: str
    date_recorded: datetime.datetime | None
    result_file_name: str
    creation_date: datetime.datetime

    def as_dict(self) -> dict[str, str | datetime.datetime | datetime.date | None]:
        return {
            "name": self.name,
            "animal_id": self.animal_id,
            "oxygen_condition": self.oxygen_condition,
            "source_file_name": self.source_file_name,
            "date_recorded": self.date_recorded,
            "result_file_name": self.result_file_name,
            "creation_date": self.creation_date,
        }


@dataclass(slots=True, frozen=True)
class SummaryStatistics:
    peak_intervals: DescriptiveStatistics
    signal_rate: DescriptiveStatistics

    def as_dict(self) -> dict[str, dict[str, float]]:
        return {
            "peak_intervals": self.peak_intervals.as_dict(),
            "signal_rate": self.signal_rate.as_dict(),
        }


@dataclass(slots=True)
class ManualPeakEdits:
    added: list[int] = field(default_factory=list)
    removed: list[int] = field(default_factory=list)

    def add_peak(self, index: int) -> None:
        self.added.append(index)

    def remove_peak(self, index: int) -> None:
        self.removed.append(index)

    def sort_and_deduplicate(self) -> None:
        self.added = sorted(set(self.added))
        self.removed = sorted(set(self.removed))

    def as_dict(self) -> dict[str, list[int]]:
        return {
            "added": self.added,
            "removed": self.removed,
        }


@dataclass(slots=True, frozen=True)
class FocusedResult:
    time_s: npt.NDArray[np.float64]
    index: npt.NDArray[np.uint32]
    peak_intervals: npt.NDArray[np.int32 | np.uint32]
    temperature: npt.NDArray[np.float64]
    rate_bpm: npt.NDArray[np.float64]

    def to_polars(self) -> pl.DataFrame:
        data = {attr: getattr(self, attr) for attr in self.__slots__}
        return pl.DataFrame(data)

    def to_structured_array(self) -> npt.NDArray[np.void]:
        dt = np.dtype([(attr, getattr(self, attr).dtype) for attr in self.__slots__])
        values = [getattr(self, attr) for attr in self.__slots__]
        return np.array(list(zip(*values, strict=True)), dtype=dt)


@dataclass(slots=True, frozen=True)
class Result:
    identifier: ResultIdentifier
    processing_parameters: _t.ProcessingParameters
    summary_statistics: SummaryStatistics
    focused_result: FocusedResult
    manual_peak_edits: ManualPeakEdits
    source_data: SignalData

    def as_dict(self) -> _t.ResultDict:
        return {
            "identifier": self.identifier.as_dict(),
            "processing_parameters": self.processing_parameters,
            "summary_statistics": self.summary_statistics.as_dict(),
            "focused_result": self.focused_result.to_structured_array(),
            "manual_peak_edits": self.manual_peak_edits.as_dict(),
            "source_data": self.source_data.as_dict(),
        }
