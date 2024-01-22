import datetime
from typing import TYPE_CHECKING, Literal, TypedDict

import numpy as np
import numpy.typing as npt
import polars as pl

if TYPE_CHECKING:
    from .handlers.data_handler import DataState
    from .models.result import (
        ManualPeakEdits,
    )
    from .models.signal import SectionID, SectionIndices

type SignalName = Literal["hbr", "ventilation"]
type ScaleMethod = Literal["mad", "zscore", "None"]
type Pipeline = Literal[
    "custom",
    "ppg_elgendi",
    "ecg_neurokit2",
    "ecg_biosppy",
    "ecg_pantompkins1985",
    "ecg_hamilton2002",
    "ecg_elgendi2010",
    "ecg_engzeemod2012",
]
type PeakDetectionMethod = Literal[
    "elgendi_ppg", "local", "neurokit2", "promac", "wfdb_xqrs", "pantompkins"
]
type WFDBPeakDirection = Literal[
    "up",
    "down",
    "both",
    "compare",
]
type FilterMethod = Literal[
    "butterworth",
    "butterworth_ba",
    "savgol",
    "fir",
    "bessel",
    "None",
]
type OxygenCondition = Literal["normoxic", "hypoxic", "unknown"]

type PeakDetectionInputValues = (
    PeakDetectionElgendiPPG
    | PeakDetectionLocalMaxima
    | PeakDetectionNeurokit2
    | PeakDetectionProMAC
    | PeakDetectionPantompkins
    | PeakDetectionXQRS
)


class FileMetadata(TypedDict):
    date_recorded: datetime.datetime
    animal_id: str
    oxygen_condition: OxygenCondition


class SignalFilterParameters(TypedDict):
    lowcut: float | None
    highcut: float | None
    method: FilterMethod
    order: int
    window_size: int | Literal["default"]
    powerline: int | float


class StandardizeParameters(TypedDict):
    robust: bool
    window_size: int | None


class PeakDetectionElgendiPPG(TypedDict):
    peakwindow: float
    beatwindow: float
    beatoffset: float
    mindelay: float


class PeakDetectionLocalMaxima(TypedDict):
    radius: int


class PeakDetectionNeurokit2(TypedDict):
    smoothwindow: float
    avgwindow: float
    gradthreshweight: float
    minlenweight: float
    mindelay: float
    correct_artifacts: bool


class PeakDetectionProMAC(TypedDict):
    threshold: float
    gaussian_sd: int
    correct_artifacts: bool


class PeakDetectionPantompkins(TypedDict):
    correct_artifacts: bool


class PeakDetectionXQRS(TypedDict):
    search_radius: int
    peak_dir: Literal["up", "down", "both", "compare"]


class PeakDetectionParameters(TypedDict):
    method: PeakDetectionMethod
    input_values: PeakDetectionInputValues


class ProcessingParameters(TypedDict):
    sampling_rate: int
    filter_parameters: SignalFilterParameters | None
    standardize_parameters: StandardizeParameters | None
    peak_detection_parameters: PeakDetectionParameters | None


class StateDict(TypedDict):
    active_signal: SignalName | str
    source_file_path: str
    output_dir: str
    data_processing_params: "ProcessingParameters"
    file_metadata: FileMetadata
    sampling_frequency: int
    peak_edits: "dict[str, ManualPeakEdits]"
    data_state: "DataState"
    stopped_at_index: int


class InitialState(TypedDict):
    name: str
    sampling_rate: int
    data: pl.DataFrame


class SectionIdentifier(TypedDict):
    section_id: "SectionID"
    included: bool
    signal_name: str
    absolute_start_index: int
    absolute_stop_index: int
    finished_processing: bool


class SectionResultDict(TypedDict):
    name: str
    section_id: "SectionID"
    absolute_bounds: "SectionIndices" | tuple[int, int]
    data: npt.NDArray[np.void]
    sampling_rate: int
    peaks: npt.NDArray[np.int32]
    rate: npt.NDArray[np.float64]
    rate_interpolated: npt.NDArray[np.float64]
    processing_parameters: ProcessingParameters


class ResultDict(TypedDict):
    identifier: dict[str, str | datetime.datetime | datetime.date | None]
    selection_parameters: dict[str, str | int | float | None]
    processing_parameters: dict[
        str,
        int | Pipeline | SignalFilterParameters | StandardizeParameters | PeakDetectionParameters,
    ]
    summary_statistics: dict[str, dict[str, float]]
    focused_result: npt.NDArray[np.void]
    manual_peak_edits: dict[str, list[int]]
    source_data: list[dict["SectionID", SectionResultDict]]
