import datetime
from typing import TYPE_CHECKING, Any, NotRequired

import numpy.typing as npt
from PySide6 import QtGui
from typing_extensions import Literal, TypedDict

if TYPE_CHECKING:
    from .handlers.data_handler import DataState
    from .models.result import (
        ManualPeakEdits,
        ProcessingParameters,
        SelectionParameters,
    )

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


class FileMetadata(TypedDict):
    date_recorded: datetime.datetime
    animal_id: str
    oxygen_condition: OxygenCondition


class EDFReaderKwargs(TypedDict, total=False):
    start: int
    stop: int | None


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


# class CorrectXQRS(TypedDict):
#     search_radius: int
#     # smooth_window_size: NotRequired[int]
#     peak_dir: Literal["up", "down", "both", "compare"]


class PeakDetectionXQRS(TypedDict):
    # sampfrom: int
    # sampto: int
    # corrections: CorrectXQRS
    search_radius: int
    peak_dir: Literal["up", "down", "both", "compare"]


type PeakDetectionInputValues = (
    PeakDetectionElgendiPPG
    | PeakDetectionLocalMaxima
    | PeakDetectionNeurokit2
    | PeakDetectionProMAC
    | PeakDetectionPantompkins
    | PeakDetectionXQRS
)


class PeakDetectionParameters(TypedDict):
    start_index: NotRequired[int]
    stop_index: NotRequired[int]
    method: PeakDetectionMethod
    input_values: PeakDetectionInputValues


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
    | QtGui.QPainterPath
)


class SpotItemDict(TypedDict):
    pos: tuple[int | float, int | float]
    size: int
    pen: QtGui.QPen
    brush: QtGui.QBrush
    symbol: PGSymbols


class SpotItemKargs(TypedDict, total=False):
    spots: list[SpotItemDict]
    x: npt.ArrayLike
    y: npt.ArrayLike
    pos: npt.ArrayLike | list[tuple[int | float, int | float]]
    pxMode: bool
    symbol: PGSymbols | list[PGSymbols]
    pen: QtGui.QPen | list[QtGui.QPen]
    brush: QtGui.QBrush | list[QtGui.QBrush]
    size: int | list[int]
    data: list[object]
    hoverable: bool
    tip: str | None
    hoverSymbol: PGSymbols | None
    hoverSize: int | Literal[-1]
    hoverPen: QtGui.QPen | None
    hoverBrush: QtGui.QBrush | None
    useCache: bool
    antialias: bool
    compositionMode: QtGui.QPainter.CompositionMode
    name: str | None


class SignalSection(TypedDict):
    index_start: int
    index_stop: int


class StateDict(TypedDict):
    active_signal: SignalName | str
    source_file_path: str
    output_dir: str
    data_selection_params: "SelectionParameters"
    data_processing_params: "ProcessingParameters"
    file_metadata: FileMetadata
    sampling_frequency: int
    peak_edits: "dict[SignalName | str, ManualPeakEdits]"
    data_state: "DataState"
    stopped_at_index: int
