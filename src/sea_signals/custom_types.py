from typing import Any, Iterable, NotRequired, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from PySide6.QtGui import QBrush, QPainter, QPainterPath, QPen
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
    "elgendi_ppg",
    "local",
    "wfdb_xqrs",
    "wfdb_local",
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


class PeakAlgorithmInputsDict(TypedDict):
    method: PeakDetectionMethod
    input_values: dict[str, float] | dict[str, WFDBPeakDirection | int] | dict[
        str, int
    ] | dict[str, float | int | bool] | dict[str, float | bool] | dict[str, bool]

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
    signal_name: SignalName
    peak_stats: PeakIntervalStats
    signal_rate_stats: SignalRateStats


class ComputedResults(TypedDict):
    signal_name: SignalName
    peak_intervals: NDArray[np.int32]
    signal_rate_len_peaks: NDArray[np.float32]
    signal_rate_len_signal: NDArray[np.float32]
    peak_interval_stats: PeakIntervalStats
    signal_rate_stats: SignalRateStats


class InfoProcessingParams(TypedDict):
    signal_name: SignalName
    sampling_rate: int
    preprocess_pipeline: Pipeline
    filter_parameters: SignalFilterParameters
    standardization_method: NormMethod
    peak_detection_method: PeakDetectionMethod
    peak_method_parameters: PeakAlgorithmInputsDict


class InfoWorkingData(TypedDict):
    signal_name: SignalName
    subset_column: str | None
    subset_lower_bound: int | float
    subset_upper_bound: int | float
    n_samples: int


class ParamChild(TypedDict, total=False):
    name: str
    type: str
    value: str | int | float | bool
    title: str | None


class ParamsType(TypedDict, total=False):
    name: str
    type: str
    readonly: bool
    children: list[ParamChild]


class ParameterOpts(TypedDict, total=False):
    type: str
    value: str | int | float | bool | None
    default: str | int | float | bool | None
    children: list[ParamsType]
    readonly: bool
    enabled: bool
    visible: bool
    renamable: bool
    removable: bool
    expanded: bool
    syncExpanded: bool
    title: str | None


class GeneralParameterOptions(TypedDict, total=False):
    name: str
    readonly: bool
    removable: bool
    visible: bool
    disabled: bool
    title: str
    default: Any
    expanded: bool


class SliderParameterOptions(GeneralParameterOptions):
    limits: Iterable[int | float]
    step: int | float
    span: NotRequired[Sequence[int | float]]
    format: str
    precision: int


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
