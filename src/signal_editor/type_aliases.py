import datetime
import typing as t

import numpy as np
import numpy.typing as npt

if t.TYPE_CHECKING:
    from .handlers.data_handler import DataState
    from .models.section import SectionID, SectionIndices


type SignalName = t.Literal["hbr", "ventilation"]
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
    "elgendi_ppg", "local", "neurokit2", "promac", "wfdb_xqrs", "pantompkins"
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
    date_recorded: datetime.date
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
    input_values: PeakDetectionInputValues


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


class SectionIdentifier(t.TypedDict):
    sig_name: str
    section_id: "SectionID"
    absolute_bounds: "SectionIndices"
    sampling_rate: int


class ManualPeakEditsDict(t.TypedDict):
    added: list[int]
    removed: list[int]


class SectionResultDict(t.TypedDict):
    sig_name: str
    section_id: "SectionID"
    absolute_bounds: "SectionIndices"
    sampling_rate: int
    data: npt.NDArray[np.void]
    peaks: npt.NDArray[np.uint32]
    peak_interval_stats: dict[str, float]
    peak_edits: ManualPeakEditsDict
    rate: npt.NDArray[np.float64]
    rate_stats: dict[str, float]
    rate_interpolated: npt.NDArray[np.float64]
    processing_parameters: ProcessingParameters
    focused_result: npt.NDArray[np.void]


class ResultIdentifierDict(t.TypedDict):
    signal_name: str
    source_file_name: str
    date_recorded: datetime.date
    animal_id: str
    oxygen_condition: OxygenCondition


class CompleteResultDict(t.TypedDict):
    identifier: ResultIdentifierDict
    processed_dataframe: npt.NDArray[np.void]
    complete_section_results: dict["SectionID", SectionResultDict]
    focused_section_results: dict["SectionID", npt.NDArray[np.void]]
