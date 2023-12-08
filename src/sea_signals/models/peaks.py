import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Any, Callable, Literal, Unpack

import neurokit2 as nk
import numpy as np
import pyqtgraph as pg
import wfdb.processing as wfproc
from loguru import logger
from neurokit2.ecg.ecg_findpeaks import _ecg_findpeaks_findmethod
from neurokit2.signal.signal_fixpeaks import _signal_fixpeaks_kubios
from numpy.typing import NDArray
from pyqtgraph.parametertree import parameterTypes as pTypes
from scipy import ndimage, signal, stats

from ..custom_types import (
    GeneralParameterOptions,
    PeakAlgorithmInputsDict,
    PeakDetectionMethod,
    WFDBPeakDirection,
)


@dataclass(slots=True, frozen=True, kw_only=True)
class PeaksElgendiPPG:
    peakwindow: float  # seconds
    beatwindow: float  # seconds
    beatoffset: float  # percentage as a fraction
    mindelay: float  # seconds

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


# @dataclass(slots=True, frozen=True, kw_only=True)
# class PeaksWFDBLocal:
#     peak_type: WFDBPeakDirection
#     radius: int

#     def as_dict(self) -> dict[str, WFDBPeakDirection | int]:
#         return asdict(self)


@dataclass(slots=True, frozen=True, kw_only=True)
class PeaksLocalMax:
    radius: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(slots=True, frozen=True, kw_only=True)
class PeaksNeurokit2:
    smoothwindow: float
    avgwindow: float
    gradthreshweight: float
    minlenweight: float
    mindelay: float
    correct_artifacts: bool

    def as_dict(self) -> dict[str, float | bool]:
        return asdict(self)


@dataclass(slots=True, frozen=True, kw_only=True)
class PeaksPromac:
    threshold: float
    gaussian_sd: int
    correct_artifacts: bool

    def as_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


@dataclass(slots=True, frozen=True, kw_only=True)
class PeaksPantompkins:
    correct_artifacts: bool

    def as_dict(self) -> dict[str, bool]:
        return asdict(self)


@dataclass(slots=True, frozen=True, kw_only=True)
class PeaksWFDBXQRS:
    sampfrom: int
    sampto: int
    radius: int
    peak_dir: WFDBPeakDirection

    def as_dict(self) -> dict[str, int | WFDBPeakDirection]:
        return asdict(self)


type PeaksInputValues = (
    PeaksElgendiPPG
    | PeaksLocalMax
    | PeaksNeurokit2
    | PeaksPromac
    | PeaksPantompkins
    | PeaksWFDBXQRS
)


@dataclass(slots=True, frozen=True, kw_only=True)
class PeakAlgorithmInputs:
    method: PeakDetectionMethod
    input_values: dict[str, float] | dict[str, WFDBPeakDirection | int] | dict[
        str, int
    ] | dict[str, float | int | bool] | dict[str, float | bool] | dict[str, bool]

    def as_dict(self) -> PeakAlgorithmInputsDict:
        return PeakAlgorithmInputsDict(
            method=self.method,
            input_values=self.input_values,
        )


class PeakDetectionParameter(pTypes.GroupParameter):
    def __init__(
        self,
        **kwargs: Unpack[GeneralParameterOptions],
    ) -> None:
        pTypes.GroupParameter.__init__(self, **kwargs)

        self._active_method: PeakDetectionMethod | None = None
        self._relevant_children = []
        self._method_map: dict[PeakDetectionMethod, Callable[[], None]] = {}
        self._values_map: dict[PeakDetectionMethod, type[PeaksInputValues]] = {}
        self.make_method_map()
        self.make_values_map()

    def make_method_map(self) -> None:
        self._method_map = {
            "elgendi_ppg": self._set_elgendi_ppg_parameters,
            "local": self._set_localmax_parameters,
            "neurokit2": self._set_neurokit2_parameters,
            "promac": self._set_promac_parameters,
            "wfdb_xqrs": self._set_wfdb_xqrs_parameters,
            "pantompkins": self._set_pantompkins_parameters,
            # "nabian": self._set_nabian_parameters,
            # "gamboa": self._set_gamboa_parameters,
            # "slopesumfunction": self._set_slopesumfunction_parameters,
            # "zong": self._set_zong_parameters,
            # "hamilton": self._set_hamilton_parameters,
            # "christov": self._set_christov_parameters,
            # "engzeemod": self._set_engzeemod_parameters,
            # "elgendi": self._set_elgendi_ecg_parameters,
            # "kalidas": self._set_kalidas_parameters,
            # "martinez": self._set_martinez_parameters,
            # "rodrigues": self._set_rodrigues_parameters,
            # "vgraph": self._set_vgraph_parameters,
        }

    def make_values_map(self) -> None:
        self._values_map = {
            "elgendi_ppg": PeaksElgendiPPG,
            "local": PeaksLocalMax,
            "neurokit2": PeaksNeurokit2,
            "promac": PeaksPromac,
            "wfdb_xqrs": PeaksWFDBXQRS,
            "pantompkins": PeaksPantompkins,
        }

    def set_method(self, method: PeakDetectionMethod) -> None:
        self._active_method = method
        self.clearChildren()
        self._method_map[method]()

    def _set_elgendi_ppg_parameters(self) -> None:
        self.addChildren(
            [
                pTypes.TextParameter(
                    name="elgendiinfo",
                    title="Info",
                    readonly=True,
                    value=(
                        "Implementation of peak detection algorithm described here: "
                        "https://doi.org/10.1371/journal.pone.0076585, taken from "
                        "`neurokit2`."
                    ),
                ),
                pTypes.SliderParameter(
                    name="peakwindow",
                    title="Peak window",
                    type="float",
                    value=0.111,
                    default=0.111,
                    step=0.001,
                    limits=(0.0, 1.0),
                    precision=3,
                ),
                pTypes.SliderParameter(
                    name="beatwindow",
                    title="Beat window",
                    type="float",
                    value=0.667,
                    default=0.667,
                    step=0.001,
                    limits=(0.050, 2.500),
                    precision=3,
                ),
                pTypes.SliderParameter(
                    name="beatoffset",
                    title="Beat offset",
                    type="float",
                    value=0.02,
                    default=0.02,
                    step=0.01,
                    limits=(0.01, 1.0),
                    precision=2,
                ),
                pTypes.SliderParameter(
                    name="mindelay",
                    title="Minimum delay",
                    type="float",
                    value=0.3,
                    default=0.3,
                    step=0.01,
                    limits=(0.1, 5.0),
                    precision=2,
                ),
            ]
        )
        self._relevant_children = [
            "peakwindow",
            "beatwindow",
            "beatoffset",
            "mindelay",
        ]

    # XXX: Takes ages and gives bad results
    # def _set_wfdb_find_peaks_parameters(self) -> None:
    #     self.addChildren(
    #         [
    #             pTypes.TextParameter(
    #                 name="wfdbfindpeaksinfo",
    #                 title="Info",
    #                 readonly=True,
    #                 value=(
    #                     "Using `soft` peak detection includes plateaus in the signal "
    #                     "(i.e. a maximum where multiple points share the same "
    #                     "maximum value) by assigning the middle point as the peak. With "
    #                     "`hard` peak detection, a value needs to be higher than both his "
    #                     "left and right neighbours to be considered a peak."
    #                     "`all` looks for any type of peak without factoring in its shape."
    #                 ),
    #             ),
    #             pTypes.ListParameter(
    #                 name="peak_type",
    #                 title="Peak Type",
    #                 limits=["all", "soft", "hard"],
    #             ),
    #             pTypes.SliderParameter(
    #                 name="radius",
    #                 title="Search radius",
    #                 type="int",
    #                 value=110,
    #                 default=110,
    #                 step=1,
    #                 limits=(5, 3333),
    #                 expanded=True,
    #             ),
    #         ]
    #     )
    #     self._relevant_children = ["peak_type", "radius"]

    def _set_localmax_parameters(self) -> None:
        self.addChildren(
            [
                pTypes.TextParameter(
                    name="localmaxinfo",
                    title="Info",
                    readonly=True,
                    value=(
                        "Finds local maxima in the signal. The search radius is the "
                        "amount of samples before and after a given point that a value "
                        "needs to be higher than to be considered a peak."
                    ),
                ),
                pTypes.SliderParameter(
                    name="radius",
                    title="Search radius",
                    type="int",
                    value=110,
                    default=110,
                    step=1,
                    limits=(5, 3333),
                ),
            ]
        )
        self._relevant_children = ["radius"]

    def _set_neurokit2_parameters(self) -> None:
        self.addChildren(
            [
                pTypes.TextParameter(
                    name="neurokit2info",
                    title="Info",
                    readonly=True,
                    value=(
                        "QRS complexes are detected based on the steepness of the "
                        "absolute gradient of the ECG signal. Subsequently, R-peaks "
                        "are detected as local maxima in the QRS complexes. See "
                        "https://github.com/neuropsychology/NeuroKit/issues/476 for more "
                        "details."
                    ),
                ),
                pTypes.SliderParameter(
                    name="smoothwindow",
                    title="Smoothing window",
                    type="float",
                    value=0.1,
                    default=0.1,
                    step=0.01,
                    limits=(0.01, 1.0),
                    precision=2,
                ),
                pTypes.SliderParameter(
                    name="avgwindow",
                    title="Average window",
                    type="float",
                    value=0.75,
                    default=0.75,
                    step=0.01,
                    limits=(0.01, 1.0),
                    precision=2,
                ),
                pTypes.SliderParameter(
                    name="gradthreshweight",
                    title="Gradient threshold weight",
                    type="float",
                    value=1.5,
                    default=1.5,
                    step=0.1,
                    limits=(0.1, 10.0),
                    precision=1,
                ),
                pTypes.SliderParameter(
                    name="minlenweight",
                    title="Minimum length weight",
                    type="float",
                    value=0.4,
                    default=0.4,
                    step=0.1,
                    limits=(0.1, 10.0),
                    precision=1,
                ),
                pTypes.SliderParameter(
                    name="mindelay",
                    title="Minimum delay",
                    type="float",
                    value=0.3,
                    default=0.3,
                    step=0.01,
                    limits=(0.01, 1.0),
                    precision=2,
                ),
                pTypes.SimpleParameter(
                    name="correct_artifacts",
                    title="Run with artifact correction (slow)",
                    type="bool",
                    value=False,
                    default=False,
                ),
            ]
        )
        self._relevant_children = [
            "smoothwindow",
            "avgwindow",
            "gradthreshweight",
            "minlenweight",
            "mindelay",
            "correct_artifacts",
        ]

    def _set_promac_parameters(self) -> None:
        self.addChildren(
            [
                pTypes.TextParameter(
                    name="promacinfo",
                    title="Info",
                    readonly=True,
                    value=(
                        "SLOW ALGORITHM."
                        "ProMAC combines the result of several R-peak detectors in a "
                        "probabilistic way. For a given peak detector, the binary signal "
                        "representing the peak locations is convolved with a Gaussian "
                        "distribution, resulting in a probabilistic representation of each "
                        "peak location. This procedure is repeated for all selected methods "
                        "and the resulting signals are accumulated. Finally, a threshold is "
                        "used to accept or reject the peak locations. See this discussion for "
                        "more information on the origins of the method:"
                        "https://github.com/neuropsychology/NeuroKit/issues/222"
                    ),
                ),
                pTypes.SimpleParameter(
                    name="threshold",
                    title=r"Threshold",
                    type="float",
                    value=0.33,
                    default=0.33,
                    limits=(0.0, 1.0),
                    precision=2,
                    step=0.01,
                ),
                pTypes.SimpleParameter(
                    name="gaussian_sd",
                    title="Size of QRS complex (in ms)",
                    type="int",
                    value=100,
                    default=100,
                    limits=(0, None),
                    step=1,
                ),
                pTypes.SimpleParameter(
                    name="correct_artifacts",
                    title="Run with artifact correction (slow)",
                    type="bool",
                    value=False,
                    default=False,
                ),
            ]
        )
        self._relevant_children = ["threshold", "gaussian_sd", "correct_artifacts"]

    def _set_pantompkins_parameters(self) -> None:
        self.addChildren(
            [
                pTypes.TextParameter(
                    name="pantompkinsinfo",
                    title="Info",
                    readonly=True,
                    value="Algorithm by Pan & Tompkins (1985).",
                ),
                pTypes.SimpleParameter(
                    name="correct_artifacts",
                    title="Run with artifact correction (slow)",
                    type="bool",
                    value=False,
                    default=False,
                ),
            ]
        )
        self._relevant_children = ["correct_artifacts"]

    def _set_wfdb_xqrs_parameters(self) -> None:
        self.addChildren(
            [
                pTypes.TextParameter(
                    name="wfdbxqrsinfo",
                    title="Info",
                    readonly=True,
                    value=(
                        "CAN BE SLOW IF THE SIGNAL IS LARGE. "
                        "Uses wavelet "
                        "Adjust a set of detected peaks to coincide with local signal maxima."
                        " - up: peaks are shifted to local maxima"
                        " - down: peaks are shifted to local minima"
                        " - both: peaks are shifted to local maxima of the rectified (absolute) signal"
                        " - compare: tries both up and down and selects the direction with the largest mean distance from the signal"
                        " - None: no adjustment"
                    ),
                ),
                pTypes.SimpleParameter(
                    name="sampfrom",
                    title="Starting point",
                    type="int",
                    value=0,
                    default=0,
                    limits=(0, None),
                ),
                pTypes.SimpleParameter(
                    name="sampto",
                    title="Stopping point (set to 0 for no limit)",
                    type="int",
                    value=0,
                    default=0,
                    limits=(0, None),
                ),
                pTypes.SliderParameter(
                    name="radius",
                    title="Max adjustment radius",
                    type="int",
                    value=90,
                    default=90,
                    step=1,
                    limits=(5, 999),
                ),
                pTypes.ListParameter(
                    name="peak_dir",
                    title="Adjustment direction",
                    limits=["compare", "up", "down", "both", "None"],
                ),
                pTypes.TextParameter(
                    name="detailed_xqrs_info",
                    title="Detailed info",
                    readonly=True,
                    value=(
                        "1. Load the signal and configuration parameters.\n"
                        "2. Bandpass filter the signal between 5 and 20 Hz, to get the filtered signal.\n"
                        "3. Apply moving wave integration (MWI) with a Ricker (Mexican hat) wavelet onto the filtered signal, and save the square of the integrated signal.\n"
                        "4. Conduct learning if specified, to initialize running parameters of noise and QRS amplitudes, the QRS detection threshold, and recent R-R intervals. If learning is unspecified or fails, use default parameters. See the docstring for the _learn_init_params method of this class for details.\n"
                        "5. Run the main detection. Iterate through the local maxima of the MWI signal. For each local maxima:\n"
                        "6. Check if it is a QRS complex. To be classified as a QRS, it must come after the refractory period, cross the QRS detection threshold, and not be classified as a T-wave if it comes close enough to the previous QRS. If successfully classified, update running detection threshold and heart rate parameters.\n"
                        "7. If not a QRS, classify it as a noise peak and update running parameters.\n"
                        "8. Before continuing to the next local maxima, if no QRS was detected within 1.66 times the recent R-R interval, perform backsearch QRS detection. This checks previous peaks using a lower QRS detection threshold.\n"
                    ),
                    expanded=False,
                ),
            ]
        )
        self._relevant_children = ["sampfrom", "sampto", "radius", "peak_dir"]

    def get_values(self) -> PeakAlgorithmInputsDict:
        """
        Get the current values of the parameters as a dictionary with the keys:
        - method (the name of the method)
        - input_values (dictionary of `parameter_name: parameter_value` for the method's parameters)

        Raises:
            ValueError: If no method is set

        Returns:
            PeakAlgorithmInputsDict: Dictionary with the selected method name and its parameters
        """
        if not hasattr(self, "_active_method") or self._active_method is None:
            raise ValueError("No method is set. Use `set_method` to set one.")
        if self._relevant_children == []:
            return {
                "method": self._active_method,
                "input_values": {},
            }
        values: dict[str, Any] = {
            name: self.names[name].value() for name in self._relevant_children
        }
        params = self._values_map[self._active_method](**values).as_dict()
        return PeakAlgorithmInputs(
            method=self._active_method, input_values=params
        ).as_dict()


def find_ppg_peaks_elgendi(
    sig: NDArray[np.float32 | np.float64],
    sampling_rate: int = 400,
    peakwindow: float = 0.111,
    beatwindow: float = 0.667,
    beatoffset: float = 0.02,
    mindelay: float = 0.3,
) -> NDArray[np.int32]:
    """
    Find the peaks in a PPG (Photoplethysmography) signal using the algorithm proposed
    by: `Elgendi M, Norton I, Brearley M, Abbott D, Schuurmans D (2013) Systolic Peak Detection in
    Acceleration Photoplethysmograms Measured from Emergency Responders in Tropical Conditions. PLoS ONE 8(10): e76585.
    doi:10.1371/journal.pone.0076585.`
    Based on the implementation of Neurokit's `_ppg_findpeaks_elgendi` function.

    Parameters:
        sig (NDArray[np.float32 | np.float64]): Input PPG signal.
        sampling_rate (int, optional): Sampling rate of the input signal. Defaults to 400.
        peakwindow (float, optional): Size of the peak window in seconds. Defaults to 0.111.
        beatwindow (float, optional): Size of the beat window in seconds. Defaults to 0.667.
        beatoffset (float, optional): Offset for beat threshold. Defaults to 0.02.
        mindelay (float, optional): Minimum delay between peaks in seconds. Defaults to 0.3.

    Returns:
        NDArray[np.int32]: Array of peak indices.

    """
    sig_abs = sig.copy()
    sig_abs[sig_abs < 0] = 0
    sqrd = sig_abs**2

    peakwindow_samples = int(np.rint(peakwindow * sampling_rate))
    ma_peak = nk.signal_smooth(sqrd, kernel="boxcar", size=peakwindow_samples)

    beatwindow_samples = int(np.rint(beatwindow * sampling_rate))
    ma_beat = nk.signal_smooth(sqrd, kernel="boxcar", size=beatwindow_samples)

    thr1 = ma_beat + beatoffset * np.mean(sqrd)

    waves = ma_peak > thr1
    wave_changes = np.diff(waves.astype(int))
    beg_waves = np.flatnonzero(wave_changes == 1)
    end_waves = np.flatnonzero(wave_changes == -1)

    if end_waves[0] < beg_waves[0]:
        end_waves = end_waves[1:]

    min_len = peakwindow_samples
    min_delay_samples = int(np.rint(mindelay * sampling_rate))
    peaks = []

    for beg, end in zip(beg_waves, end_waves, strict=False):
        if end - beg < min_len:
            continue

        data = sig[beg:end]
        locmax, props = signal.find_peaks(data, prominence=(None, None))

        if locmax.size > 0:
            peak = beg + locmax[np.argmax(props["prominences"])]

            if not peaks or peak - peaks[-1] > min_delay_samples:
                peaks.append(peak)

    return np.array(peaks, dtype=np.int32)


def find_local_peaks(
    sig: NDArray[np.float32 | np.float64],
    radius: int,
) -> NDArray[np.int32]:
    if len(sig) == 0:
        return np.array([], dtype=np.int32)

    if np.min(sig) == np.max(sig):
        return np.array([], dtype=np.int32)

    peak_inds = []

    max_vals = ndimage.maximum_filter(sig, size=2 * radius + 1, mode="constant")
    i = 0
    while i < len(sig):
        if sig[i] == max_vals[i]:
            peak_inds.append(i)
            i += radius
        else:
            i += 1

    return np.array(peak_inds, dtype=np.int32)


def _shift_peaks(
    sig: NDArray[np.float32 | np.float64],
    peaks: NDArray[np.int32],
    radius: int,
    dir_is_up: bool,
) -> NDArray[np.int32]:
    sig_len = sig.size
    shifted_peaks = np.zeros_like(peaks)

    for i, peak_index in enumerate(peaks):
        start = max(9, peak_index - radius)
        end = min(peak_index + radius, sig_len)
        local_indices = np.arange(start, end, dtype=np.int32)
        local_sig = sig[local_indices]

        if dir_is_up:
            shifted_peaks[i] = local_indices[np.argmax(local_sig)] - peak_index
        else:
            shifted_peaks[i] = local_indices[np.argmin(local_sig)] - peak_index

    return peaks + shifted_peaks


def adjust_peak_positions(
    sig: NDArray[np.float32 | np.float64],
    peaks: NDArray[np.int32],
    radius: int,
    direction: WFDBPeakDirection,
) -> NDArray[np.int32]:
    if direction == "None":
        return peaks
    elif direction == "up":
        return _shift_peaks(sig, peaks, radius, dir_is_up=True)
    elif direction == "down":
        return _shift_peaks(sig, peaks, radius, dir_is_up=False)
    elif direction == "both":
        return _shift_peaks(np.abs(sig), peaks, radius, dir_is_up=True)
    elif direction == "compare":
        shifted_up = _shift_peaks(sig, peaks, radius, dir_is_up=True)
        shifted_down = _shift_peaks(sig, peaks, radius, dir_is_up=False)

        up_dist = np.mean(np.abs(sig[shifted_up]), dtype=np.float32)
        down_dist = np.mean(np.abs(sig[shifted_down]), dtype=np.float32)

        return shifted_up if np.greater_equal(up_dist, down_dist) else shifted_down
    else:
        logger.warning(f"Unknown direction: `{direction}`. Returning original peaks.")
        return peaks


def neurokit2_peak_algorithms(
    sig: NDArray[np.float32 | np.float64],
    sampling_rate: int,
    algorithm: Literal[
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
    ],
    **options: Any,
) -> NDArray[np.int32]:
    artifact_correction = options.pop("correct_artifacts", False)
    if algorithm == "promac":
        return _ecg_findpeaks_promac_sequential(
            sig, sampling_rate, artifact_correction, **options
        )
    peak_finder = _ecg_findpeaks_findmethod(algorithm)
    peaks = peak_finder(sig, sampling_rate=sampling_rate, **options)
    peaks = (
        _signal_fixpeaks_kubios(peaks, sampling_rate)[1]
        if artifact_correction
        else peaks
    )
    return np.asarray(peaks, dtype=np.int32)


def _ecg_findpeaks_promac_addconvolve(
    sig: NDArray[np.float32 | np.float64],
    sampling_rate: int,
    x: NDArray[np.float32 | np.float64],
    fun: Callable[..., Any],
    gaussian_sd: int = 100,
    **kwargs: Any,
) -> NDArray[np.floating[Any]]:
    peaks = fun(sig, sampling_rate=sampling_rate, **kwargs)
    mask = np.zeros(len(sig))
    mask[peaks] = 1

    sd = sampling_rate * gaussian_sd / 1000
    shape = stats.norm.pdf(
        np.linspace(-sd * 4, sd * 4, num=int(sd * 8)), loc=0, scale=sd
    )

    x += np.convolve(mask, shape, mode="same")
    return x


def _ecg_findpeaks_promac_parallel(
    sig: NDArray[np.float32 | np.float64],
    sampling_rate: int,
    correct_artifacts: bool,
    threshold: float = 0.33,
    gaussian_sd: int = 100,
    **kwargs: Any,
) -> NDArray[np.int32]:
    with pg.ProgressDialog("Running ProMAC...", wait=0, cancelText=None) as progress:
        progress.setValue(0)
        progress.setMaximum(9)
        x = np.zeros_like(sig)
        promac_methods = [
            "neurokit",
            "gamboa",
            "ssf",
            "elgendi",
            "manikandan",
            "kalidas",
            "rodrigues",
        ]

        with ProcessPoolExecutor(max_workers=min(4, os.cpu_count() // 2)) as executor:
            futures = {
                executor.submit(
                    _ecg_findpeaks_promac_addconvolve,
                    sig,
                    sampling_rate,
                    x,
                    _ecg_findpeaks_findmethod(method),
                    gaussian_sd=gaussian_sd,
                    **kwargs,
                ): method
                for method in promac_methods
            }

            for future in as_completed(futures):
                # logger.debug(f"Completed: {futures[future]}")
                x += future.result()
                progress += 1
                progress.setLabelText(
                    f"Computing probability density function\nfor method: {futures[future]}"
                )

        x = x / np.max(x)

        x[x < threshold] = 0

        peaks = nk.signal_findpeaks(x, relative_height_min=threshold)["Peaks"]

        progress += 1

        if correct_artifacts:
            progress.setLabelText("Fixing artifacts...")
            peaks = _signal_fixpeaks_kubios(peaks, sampling_rate)[1]

        progress.setLabelText("Done!")
        progress.setValue(9)

        return np.asarray(peaks, dtype=np.int32)


def _ecg_findpeaks_promac_sequential(
    sig: NDArray[np.float32 | np.float64],
    sampling_rate: int,
    correct_artifacts: bool,
    threshold: float = 0.33,
    gaussian_sd: int = 100,
    **kwargs: Any,
) -> NDArray[np.int32]:
    with pg.ProgressDialog("Running ProMAC...", wait=0, cancelText=None) as progress:
        progress.setMaximum(9)
        x = np.zeros_like(sig)
        promac_methods = [
            "neurokit",
            "gamboa",
            "ssf",
            "elgendi",
            "manikandan",
            "kalidas",
            "rodrigues",
        ]

        for method in promac_methods:
            progress.setLabelText(
                f"Computing probability density function\nfor method: {method}"
            )
            func = _ecg_findpeaks_findmethod(method)
            x = _ecg_findpeaks_promac_addconvolve(
                sig, sampling_rate, x, func, gaussian_sd=gaussian_sd, **kwargs
            )
            progress += 1

        x = x / np.max(x)

        x[x < threshold] = 0

        progress.setLabelText("Finding peaks...")
        peaks = nk.signal_findpeaks(x, height_min=threshold)["Peaks"]

        progress += 1

        if correct_artifacts:
            progress.setLabelText("Fixing artifacts...")
            peaks = _signal_fixpeaks_kubios(peaks, sampling_rate)[1]

        progress.setLabelText("Done!")
        progress += 1

        return np.asarray(peaks, dtype=np.int32)


def _handle_close_peaks(
    sig: NDArray[np.float32 | np.float64],
    peak_idx: NDArray[np.int32],
    find_peak_fn: Callable[[NDArray[np.float32 | np.float64]], int],
) -> NDArray[np.int32]:
    qrs_diffs = np.diff(peak_idx)
    close_inds = np.where(qrs_diffs <= 40)[0]

    if not close_inds.size:
        return peak_idx

    to_remove = []
    for ind in close_inds:
        if find_peak_fn == np.argmax:
            comparison_fn = np.less_equal
        elif find_peak_fn == np.argmin:
            comparison_fn = np.greater_equal
        else:
            raise ValueError(f"find_peak_fn {find_peak_fn} not supported.")

        to_remove.append(
            ind
            if comparison_fn(sig[peak_idx[ind]], sig[peak_idx[ind + 1]])
            else ind + 1
        )

    peak_idx = np.delete(peak_idx, to_remove)
    return peak_idx


def remove_outliers(
    sig: NDArray[np.float32 | np.float64],
    peak_idx: NDArray[np.int32],
    n_std: float,
    find_peak_fn: Callable[[NDArray[np.float32 | np.float64]], int],
) -> NDArray[np.int32]:
    outliers = []
    window_size = 5
    for i, peak in enumerate(peak_idx):
        if i < 3:
            start_ind = 0
            end_ind = max(window_size, i + 3)
        else:
            start_ind = max(0, i - 2)
            end_ind = min(peak_idx.size, i + 3)

        surrounding_peaks = peak_idx[start_ind:end_ind]
        surrounding_vals = sig[surrounding_peaks]

        local_mean = np.mean(surrounding_vals)
        local_std = np.std(surrounding_vals)

        if find_peak_fn == np.argmax:
            threshold = local_mean - n_std * local_std
            comparison_fn = np.less_equal
        elif find_peak_fn == np.argmin:
            threshold = local_mean + n_std * local_std
            comparison_fn = np.greater_equal
        else:
            raise ValueError("find_peak_fn must be np.argmax or np.argmin")

        if comparison_fn(sig[peak], threshold):
            outliers.append(i)

    peak_idx = np.delete(peak_idx, outliers)
    return _handle_close_peaks(sig, peak_idx, find_peak_fn)


def sanitize_qrs_inds(
    sig: NDArray[np.float32 | np.float64],
    qrs_inds: NDArray[np.int32],
    n_std: float = 4.0,
) -> NDArray[np.int32]:
    peak_idx = qrs_inds
    if np.mean(sig) < np.mean(sig[peak_idx]):
        find_peak_fn = np.argmax
    else:
        find_peak_fn = np.argmin

    peak_idx = remove_outliers(sig, peak_idx, n_std, find_peak_fn)

    sorted_indices = np.argsort(peak_idx)

    sorted_indices = sorted_indices[peak_idx[sorted_indices] >= 0]

    return peak_idx[sorted_indices]


def find_peaks_xqrs(
    sig: NDArray[np.float64],
    processed_sig: NDArray[np.float64],
    sampling_rate: int,
    radius: int,
    peak_dir: WFDBPeakDirection,
    sampfrom: int = 0,
    sampto: int = 0,
) -> NDArray[np.int32]:
    if sampto == 0:
        sampto = sig.size
    xqrs_out = wfproc.XQRS(sig=sig, fs=sampling_rate)
    xqrs_out.detect(sampfrom=sampfrom, sampto=sampto)
    qrs_inds = np.array(xqrs_out.qrs_inds, dtype=np.int32)
    peak_inds = adjust_peak_positions(
        processed_sig, peaks=qrs_inds, radius=radius, direction=peak_dir
    )
    return sanitize_qrs_inds(processed_sig, peak_inds)


def find_local_peaks_wfdb(
    sig: NDArray[np.float32 | np.float64],
    radius: int | None = None,
    peak_type: Literal["all", "soft", "hard"] = "all",
) -> NDArray[np.int32]:
    if peak_type in {"soft", "hard"}:
        hard_peaks, soft_peaks = wfproc.find_peaks(sig)
        return (
            np.asarray(soft_peaks, dtype=np.int32)
            if peak_type == "soft"
            else np.asarray(hard_peaks, dtype=np.int32)
        )
    if radius is None:
        radius = 30

    return np.asarray(wfproc.find_local_peaks(sig, radius), dtype=np.int32)
