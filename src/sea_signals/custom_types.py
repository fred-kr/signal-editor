import numpy as np
from numpy.typing import NDArray
from typing_extensions import Literal, TypedDict

type SignalName = Literal["hbr", "ventilation"]
type NormMethod = Literal["mad", "zscore", "minmax", "None"]
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
    "elgendi",
    "neurokit2",
    "local",
    "xqrs",
]
type FilterMethod = Literal[
    "butterworth",
    "butterworth_ba",
    "savgol",
    "fir",
    "bessel",
    "None",
]


class RequiredParameters(TypedDict):
    method: FilterMethod


class SignalFilterParameters(RequiredParameters, total=False):
    lowcut: float | None
    highcut: float | None
    order: int
    window_size: int | Literal["default"]
    powerline: int


class MinMaxMapping(TypedDict):
    min: float
    max: float


class PeaksPPGElgendi(TypedDict):
    peakwindow: float  # seconds
    beatwindow: float  # seconds
    beatoffset: float  # percentage as a fraction
    mindelay: float  # seconds


class PeaksLocalMax(TypedDict):
    window_size: int


class PeaksXQRS(TypedDict):
    window_size: int
    correction_direction: Literal["up", "down", "both", "compare", "None"]


type OxygenCondition = Literal["normoxic", "hypoxic"]


class PeakIntervalStats(TypedDict):
    signal_name: SignalName
    peak_interval_mean: np.float_
    peak_interval_median: np.float_
    peak_interval_std: np.float_
    peak_interval_var: np.float_


class SignalRateStats(TypedDict):
    signal_name: SignalName
    signal_rate_mean: np.float_
    signal_rate_median: np.float_
    signal_rate_std: np.float_
    signal_rate_var: np.float_


class StatsDict(TypedDict):
    peak_stats: PeakIntervalStats
    signal_rate_stats: SignalRateStats


class ComputedResults(TypedDict):
    peak_intervals: NDArray[np.int32]
    signal_rate_from_peaks: NDArray[np.float32]
    signal_rate_interpolated_signal_length: NDArray[np.float32]
    peak_interval_stats: PeakIntervalStats
    signal_rate_stats: SignalRateStats


class InfoProcessingParams(TypedDict):
    sampling_rate: int
    preprocess_pipeline: Pipeline
    filter_parameters: SignalFilterParameters
    standardization_method: NormMethod
    peak_detection_method: PeakDetectionMethod
    peak_method_parameters: PeaksPPGElgendi | PeaksLocalMax | PeaksXQRS


class InfoWorkingData(TypedDict):
    subset_column: str | None
    subset_lower_bound: int | float
    subset_upper_bound: int | float
    n_samples: int
