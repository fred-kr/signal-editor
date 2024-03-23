import datetime
import typing as t

import numpy as np
import numpy.typing as npt
import polars as pl

if t.TYPE_CHECKING:
    from PySide6 import QtCore, QtGui

    from .handlers import DataState
    from .models import SectionContainer, SectionID, SectionIndices


type ScaleMethod = t.Literal["mad", "zscore", "None"]
type Pipeline = t.Literal[
    "custom",
    "ppg_elgendi",
    "ecg_neurokit2",
    "ecg_biosppy",
    "ecg_pantompkins1985",
    "ecg_hamilton2002",
    "ecg_elgendi2010",
    "ecg_engzeemod2012",
]
type PeakDetectionMethod = t.Literal[
    "elgendi_ppg", "local", "neurokit2", "promac", "wfdb_xqrs", "pantompkins", "local_min"
]
type WFDBPeakDirection = t.Literal[
    "up",
    "down",
    "both",
    "compare",
]
type FilterMethod = t.Literal[
    "butterworth",
    "butterworth_ba",
    "savgol",
    "fir",
    "bessel",
    "powerline",
    "None",
]
type OxygenCondition = t.Literal["normoxic", "hypoxic", "unknown"]

type PeakDetectionInputValues = (
    PeakDetectionElgendiPPG
    | PeakDetectionLocalMaxima
    | PeakDetectionNeurokit2
    | PeakDetectionProMAC
    | PeakDetectionPantompkins
    | PeakDetectionXQRS
)

type SmoothingKernels = t.Literal[
    "barthann",
    "bartlett",
    "blackman",
    "blackmanharris",
    "bohman",
    "boxcar",
    "chebwin",
    "cosine",
    "dpss",
    "exponential",
    "flattop",
    "gaussian",
    "general_cosine",
    "general_gaussian",
    "general_hamming",
    "hamming",
    "hann",
    "kaiser",
    "kaiser_bessel_derived",
    "lanczos",
    "nuttall",
    "parzen",
    "taylor",
    "triangle",
    "tukey",
    "boxzen",
    "median",
]

type NKECGAlgorithms = t.Literal[
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


class FileMetadata(t.TypedDict):
    date_recorded: datetime.date | None
    animal_id: str
    oxygen_condition: OxygenCondition


class SignalFilterParameters(t.TypedDict):
    lowcut: float | None
    highcut: float | None
    method: FilterMethod
    order: int
    window_size: int | t.Literal["default"]
    powerline: int | float


class StandardizeParameters(t.TypedDict):
    robust: bool
    window_size: int | None
    method: t.NotRequired[t.Literal["mad", "zscore", "None"]]


class PeakDetectionElgendiPPG(t.TypedDict):
    peakwindow: float
    beatwindow: float
    beatoffset: float
    mindelay: float


class PeakDetectionLocalMaxima(t.TypedDict):
    radius: int


class PeakDetectionNeurokit2(t.TypedDict):
    smoothwindow: float
    avgwindow: float
    gradthreshweight: float
    minlenweight: float
    mindelay: float
    correct_artifacts: bool


class PeakDetectionProMAC(t.TypedDict):
    threshold: float
    gaussian_sd: int
    correct_artifacts: bool


class PeakDetectionPantompkins(t.TypedDict):
    correct_artifacts: bool


class PeakDetectionXQRS(t.TypedDict):
    search_radius: int
    peak_dir: t.Literal["up", "down", "both", "compare"]


class PeakDetectionParameters(t.TypedDict):
    method: PeakDetectionMethod
    method_parameters: PeakDetectionInputValues


class ProcessingParameters(t.TypedDict):
    sampling_rate: int
    pipeline: Pipeline | None
    filter_parameters: SignalFilterParameters | None
    standardize_parameters: StandardizeParameters | None
    peak_detection_parameters: PeakDetectionParameters | None


class StateDict(t.TypedDict):
    signal_name: str
    source_file_path: str
    output_dir: str
    data_state: "DataState"


class DataStateDict(t.TypedDict):
    raw_df: pl.DataFrame | None
    base_df: pl.DataFrame | None
    sections: "SectionContainer"
    sampling_rate: int
    metadata: FileMetadata | None


class SectionIdentifierDict(t.TypedDict):
    sig_name: str
    section_id: "SectionID"
    absolute_bounds: "SectionIndices"
    sampling_rate: int


class ManualPeakEditsDict(t.TypedDict):
    added: list[int]
    removed: list[int]


class SummaryDict(t.TypedDict):
    min: np.float_ | float
    max: np.float_ | float
    mean: np.float_ | float
    std: np.float_ | float
    median: np.float_ | float
    skew: np.float_ | float
    kurtosis: np.float_ | float


class SectionResultDict(t.TypedDict):
    identifier: SectionIdentifierDict
    data: npt.NDArray[np.void]
    peaks_section: npt.NDArray[np.uint32]
    peaks_global: npt.NDArray[np.uint32]
    peak_edits: ManualPeakEditsDict
    rate: npt.NDArray[np.float64]
    rate_interpolated: npt.NDArray[np.float64]
    processing_parameters: ProcessingParameters


class ResultIdentifierDict(t.TypedDict):
    signal_name: str
    source_file_name: str
    date_recorded: datetime.date | None
    animal_id: str
    oxygen_condition: OxygenCondition


class CompleteResultDict(t.TypedDict):
    identifier: ResultIdentifierDict
    global_dataframe: npt.NDArray[np.void]
    complete_section_results: dict["SectionID", SectionResultDict]
    focused_section_results: dict["SectionID", npt.NDArray[np.void]]


class SpotDict(t.TypedDict, total=False):
    pos: "tuple[float, float] | QtCore.QPointF"
    size: float
    pen: "QtGui.QPen | str | None"
    brush: "QtGui.QBrush | str | None"
    symbol: str


class SpotItemSetDataKwargs(t.TypedDict, total=False):
    spots: list[SpotDict]
    x: npt.ArrayLike
    y: npt.ArrayLike
    pos: npt.ArrayLike | list[tuple[float, float]]
    pxMode: bool
    symbol: str
    pen: "QtGui.QPen | str | None"
    brush: "QtGui.QBrush | str | None"
    size: float
    data: list[object]
    hoverable: bool
    tip: str | None
    hoverSymbol: str
    hoverSize: float
    hoverPen: "QtGui.QPen | str | None"
    hoverBrush: "QtGui.QBrush | str | None"
    useCache: bool
    antialias: bool
    compositionMode: "QtGui.QPainter.CompositionMode | None"
    name: str | None


class RollingRateParameters(t.TypedDict):
    grp_col: str
    temperature_col: str
    sec_every: int
    sec_period: int
    sec_offset: int
    sampling_rate: int
    start_by: t.NotRequired[t.Literal["window", "datapoint"]]
    label: t.NotRequired[t.Literal["left", "right", "datapoint"]]
    include_boundaries: t.NotRequired[bool]
    edge_behavior: t.NotRequired[t.Literal["approximate", "remove"]]
