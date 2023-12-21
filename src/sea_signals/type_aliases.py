from typing import Any, NotRequired

import numpy as np
import polars as pl
from numpy.typing import ArrayLike, NDArray
from PySide6.QtGui import QBrush, QPainter, QPainterPath, QPen
from typing_extensions import Literal, TypedDict

# ==================================================================================== #
#                                     TYPE ALIASES                                     #
# ==================================================================================== #
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
    "elgendi_ppg",
    "local",
    "wfdb_xqrs",
    "neurokit2",
    "promac",
    "pantompkins",
    "nabian",
    "gamboa",
    "slopesumfunction",
    "zong",
    "hamilton",
    "christov",
    "engzeemod",
    "elgendi",
    "kalidas",
    "martinez",
    "rodrigues",
    "vgraph",
]
type WFDBPeakDirection = Literal[
    "up",
    "down",
    "both",
    "compare",
    "None",
]
type FilterMethod = Literal[
    "butterworth",
    "butterworth_ba",
    "savgol",
    "fir",
    "bessel",
    "None",
]
type OxygenCondition = Literal["normoxic", "hypoxic"]


# ==================================================================================== #
#                                  TYPED DICTIONARIES                                  #
# ==================================================================================== #


# File readers +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
class EDFReaderKwargs(TypedDict, total=False):
    start: int
    stop: int | None


# Data Handler +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
class MinMaxMapping(TypedDict):
    min: float
    max: float


class DataState(TypedDict):
    df: pl.DataFrame
    peaks: dict[SignalName, NDArray[np.int32]]
    rate_no_interpolation: dict[SignalName, NDArray[np.float64]]


# Signal Preprocessing +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
class RequiredParameters(TypedDict):
    method: FilterMethod


class SignalFilterParameters(RequiredParameters, total=False):
    lowcut: float | None
    highcut: float | None
    order: int
    window_size: int | Literal["default"]
    powerline: int


class StandardizeParameters(TypedDict):
    method: ScaleMethod
    window_size: NotRequired[int | Literal["None"]]


# Peak Detection +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
class OldPeaks(TypedDict, total=False):
    hbr: list[tuple[np.int32, np.float64]]
    ventilation: list[tuple[np.int32, np.float64]]


class AddedPoints(TypedDict):
    hbr: list[int]
    ventilation: list[int]


class RemovedPoints(TypedDict):
    hbr: list[int]
    ventilation: list[int]


class PeakEdits(TypedDict):
    added_peaks: AddedPoints
    removed_peaks: RemovedPoints


class PeakDetectionManualEdited(TypedDict):
    added_peaks: list[int]
    removed_peaks: list[int]


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


class CorrectXQRS(TypedDict):
    search_radius: int
    smooth_window_size: int
    peak_dir: NotRequired[WFDBPeakDirection]


class PeakDetectionXQRS(TypedDict):
    sampfrom: int
    sampto: int
    corrections: CorrectXQRS


type PeakDetectionInputValues = (
    PeakDetectionElgendiPPG
    | PeakDetectionLocalMaxima
    | PeakDetectionNeurokit2
    | PeakDetectionProMAC
    | PeakDetectionPantompkins
    | PeakDetectionXQRS
)


class PeakDetectionParameters(TypedDict):
    method: PeakDetectionMethod
    input_values: PeakDetectionInputValues


# Results ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
class PeakIntervalStatistics(TypedDict):
    mean: np.float_
    median: np.float_
    std: np.float_
    var: np.float_


class RateStatistics(TypedDict):
    mean: np.float_
    median: np.float_
    std: np.float_
    var: np.float_


class DescriptiveStatistics(TypedDict):
    name: SignalName
    interval: PeakIntervalStatistics
    rate: RateStatistics


class ComputedResults(TypedDict):
    signal_name: SignalName
    peak_intervals: NDArray[np.int32]
    signal_rate: NDArray[np.float64]
    peak_interval_stats: PeakIntervalStatistics
    signal_rate_stats: RateStatistics


class InfoProcessingParams(TypedDict):
    signal_name: SignalName
    sampling_rate: int
    preprocess_pipeline: Pipeline
    filter_parameters: SignalFilterParameters
    standardization_method: ScaleMethod
    peak_detection_method: PeakDetectionMethod
    peak_method_parameters: PeakDetectionParameters


class InfoWorkingData(TypedDict):
    signal_name: SignalName
    subset_column: str | None
    subset_lower_bound: int | float
    subset_upper_bound: int | float
    n_samples: int


class GeneralParameterOptions(TypedDict, total=False):
    name: str
    readonly: bool
    removable: bool
    visible: bool
    disabled: bool
    title: str
    default: Any
    expanded: bool


type PGSymbols = (
    Literal[
        "o",  # circle
        "s",  # square
        "t",  # triangle pointing down
        "d",  # diamond
        "+",  # plus
        "t1",  # triangle pointing up
        "t2",  # triangle pointing right
        "t3",  # triangle pointing left
        "p",  # pentagon
        "h",  # hexagon
        "star",  # star
        "x",  # cross
        "arrow_up",  # arrow pointing up
        "arrow_right",  # arrow pointing right
        "arrow_down",  # arrow pointing down
        "arrow_left",  # arrow pointing left
        "crosshair",  # crosshair
    ]
    | QPainterPath
)


class SpotItemDict(TypedDict):
    pos: tuple[int | float, int | float]
    size: int
    pen: QPen
    brush: QBrush
    symbol: PGSymbols


class SpotItemKargs(TypedDict, total=False):
    spots: list[SpotItemDict]
    x: ArrayLike
    y: ArrayLike
    pos: ArrayLike | list[tuple[int | float, int | float]]
    pxMode: bool
    symbol: PGSymbols | list[PGSymbols]
    pen: QPen | list[QPen]
    brush: QBrush | list[QBrush]
    size: int | list[int]
    data: list[object]
    hoverable: bool
    tip: str | None
    hoverSymbol: PGSymbols | None
    hoverSize: int | Literal[-1]
    hoverPen: QPen | None
    hoverBrush: QBrush | None
    useCache: bool
    antialias: bool
    compositionMode: QPainter.CompositionMode
    name: str | None
