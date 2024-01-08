import datetime
from dataclasses import dataclass, field
from typing import Any, Iterable, TypedDict

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

    def as_dict(self) -> dict[str, float]:
        return {
            "mean": self.mean.round(2).astype(float),
            "median": self.median.round(2).astype(float),
            "std": self.std.round(2).astype(float),
            "var": self.var.round(2).astype(float),
        }


@dataclass(slots=True, frozen=True)
class ResultIdentifier:
    name: SignalName | str
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
class SelectionParameters:
    filter_column: str | None
    lower_bound: int | float
    upper_bound: int | float
    length_overall: int

    def as_dict(self) -> dict[str, str | int | float | None]:
        return {
            "filter_column": self.filter_column,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "length_overall": self.length_overall,
        }


@dataclass(slots=True, frozen=True)
class ProcessingParameters:
    sampling_rate: int
    pipeline: Pipeline
    filter_parameters: SignalFilterParameters
    scaling_parameters: StandardizeParameters
    peak_detection_parameters: PeakDetectionParameters

    def as_dict(
        self,
    ) -> dict[
        str,
        int
        | Pipeline
        | SignalFilterParameters
        | StandardizeParameters
        | PeakDetectionParameters,
    ]:
        return {
            "sampling_rate": self.sampling_rate,
            "pipeline": self.pipeline,
            "filter_parameters": self.filter_parameters,
            "scaling_parameters": self.scaling_parameters,
            "peak_detection_parameters": self.peak_detection_parameters,
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

    def to_structured_array(self) -> npt.NDArray[np.void]:
        dt = np.dtype(
            [
                ("time_s", np.float64),
                ("index", np.int32),
                ("peak_intervals", np.int32),
                ("temperature", np.float64),
                ("rate_bpm", np.float64),
            ]
        )
        return np.array(
            list(
                zip(
                    self.time_s,
                    self.index,
                    self.peak_intervals,
                    self.temperature,
                    self.rate_bpm,
                    strict=True,
                )
            ),
            dtype=dt,
        )

    def as_dict(self) -> dict[str, npt.NDArray[np.float64] | npt.NDArray[np.int32]]:
        return {
            "time_s": self.time_s,
            "index": self.index,
            "peak_intervals": self.peak_intervals,
            "temperature": self.temperature,
            "rate_bpm": self.rate_bpm,
        }


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


type SignalDataHDF5 = dict[
    str, str | int | bool | npt.NDArray[np.float64] | npt.NDArray[np.int32]
]

type ResultValues = (
    ResultIdentifier
    | SelectionParameters
    | ProcessingParameters
    | SummaryStatistics
    | FocusedResult
    | ManualPeakEdits
    | SignalDataHDF5
    | dict[str, Any]
)


class ResultDict(TypedDict):
    identifier: dict[str, str | datetime.datetime | datetime.date | None]
    selection_parameters: dict[str, str | int | float | None]
    processing_parameters: dict[
        str,
        int
        | Pipeline
        | SignalFilterParameters
        | StandardizeParameters
        | PeakDetectionParameters,
    ]
    summary_statistics: dict[str, dict[str, float]]
    focused_result: npt.NDArray[np.void]
    manual_peak_edits: dict[str, list[int]]
    source_data: dict[
        str, str | int | bool | npt.NDArray[np.float64] | npt.NDArray[np.int32]
    ]


@dataclass(slots=True, frozen=True)
class Result:
    identifier: ResultIdentifier
    selection_parameters: SelectionParameters
    processing_parameters: ProcessingParameters
    summary_statistics: SummaryStatistics
    focused_result: FocusedResult
    manual_peak_edits: ManualPeakEdits
    source_data: SignalData

    def as_dict(self) -> ResultDict:
        return {
            "identifier": self.identifier.as_dict(),
            "selection_parameters": self.selection_parameters.as_dict(),
            "processing_parameters": self.processing_parameters.as_dict(),
            "summary_statistics": self.summary_statistics.as_dict(),
            "focused_result": self.focused_result.to_structured_array(),
            "manual_peak_edits": self.manual_peak_edits.as_dict(),
            "source_data": self.source_data.as_dict(),
        }


class ResultContainer(dict[str, Result]):
    def __init__(self, *args: Iterable[tuple[str, Result]], **kwargs: Result) -> None:
        super().__init__(*args, **kwargs)

    def __setitem__(self, key: str, value: Result) -> None:
        super().__setitem__(key, value)
