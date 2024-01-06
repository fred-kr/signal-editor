import datetime
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import numpy.typing as npt
import polars as pl

from ..type_aliases import (
    PeakDetectionParameters,
    Pipeline,
    SignalFilterParameters,
    SignalName,
    StandardizeParameters,
)
from .signal import SignalData


@dataclass(slots=True, frozen=True)
class DescriptiveStatistics:
    _data_description: str
    mean: np.floating[Any]
    median: np.floating[Any]
    std: np.floating[Any]
    var: np.floating[Any]

    def __post_init__(self) -> None:
        for attr in ["mean", "median", "std", "var"]:
            np.round(getattr(self, attr), decimals=2)


@dataclass(slots=True, frozen=True)
class ResultIdentifier:
    name: SignalName | str
    animal_id: str
    oxygen_condition: str
    source_file_name: str
    date_recorded: datetime.datetime | None
    result_file_name: str
    creation_date: datetime.datetime


@dataclass(slots=True, frozen=True)
class SelectionParameters:
    filter_column: str | None
    lower_bound: int | float
    upper_bound: int | float
    length_overall: int


@dataclass(slots=True, frozen=True)
class ProcessingParameters:
    sampling_rate: int
    pipeline: Pipeline
    filter_parameters: SignalFilterParameters
    scaling_parameters: StandardizeParameters
    peak_detection_parameters: PeakDetectionParameters


@dataclass(slots=True, frozen=True)
class SummaryStatistics:
    peak_intervals: DescriptiveStatistics
    signal_rate: DescriptiveStatistics


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


@dataclass(slots=True, frozen=True)
class FocusedResult:
    time_s: npt.NDArray[np.float64]
    index: npt.NDArray[np.int32]
    peak_intervals: npt.NDArray[np.int32]
    temperature: npt.NDArray[np.float64]
    rate_bpm: npt.NDArray[np.float64]

    def to_polars(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "time_s": self.time_s,
                "index": self.index,
                "peak_intervals": self.peak_intervals,
                "temperature": self.temperature,
                "rate_bpm": self.rate_bpm,
            },
        )


def _series_to_numpy[T: (np.int32, np.float64)](
    series: pl.Series | npt.NDArray[T],
) -> npt.NDArray[T]:
    return series.to_numpy() if isinstance(series, pl.Series) else series


def make_focused_result(
    time_s: npt.NDArray[np.float64] | pl.Series,
    index: npt.NDArray[np.int32] | pl.Series,
    peak_intervals: npt.NDArray[np.int32] | pl.Series,
    temperature: npt.NDArray[np.float64] | pl.Series,
    rate_bpm: npt.NDArray[np.float64] | pl.Series,
) -> FocusedResult:
    time_s = _series_to_numpy(time_s).round(4)
    index = _series_to_numpy(index)
    peak_intervals = _series_to_numpy(peak_intervals)
    temperature = _series_to_numpy(temperature).round(1)
    rate_bpm = _series_to_numpy(rate_bpm).round(1)

    return FocusedResult(
        time_s=time_s,
        index=index,
        peak_intervals=peak_intervals,
        temperature=temperature,
        rate_bpm=rate_bpm,
    )


@dataclass(slots=True, frozen=True)
class Result:
    identifier: ResultIdentifier
    selection_parameters: SelectionParameters
    processing_parameters: ProcessingParameters
    summary_statistics: SummaryStatistics
    focused_result: FocusedResult
    manual_peak_edits: ManualPeakEdits
    source_data: SignalData
    other: dict[str, Any] = field(default_factory=dict)


class ResultContainer(dict[str, Result]):
    def __init__(self, *args: Iterable[tuple[str, Result]], **kwargs: Result) -> None:
        super().__init__(*args, **kwargs)

    def __setitem__(self, key: str, value: Result) -> None:
        super().__setitem__(key, value)
