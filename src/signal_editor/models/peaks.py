import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from enum import Enum
from functools import partial
from typing import Any, Callable, Literal, TypeVar

import neurokit2 as nk
import numpy as np
import numpy.typing as npt
import pyqtgraph as pg
import wfdb.processing as wfproc
from neurokit2.ecg.ecg_findpeaks import _ecg_findpeaks_findmethod
from scipy import ndimage, signal, stats

from ..type_aliases import (
    PeakDetectionElgendiPPG,
    PeakDetectionInputValues,
    PeakDetectionLocalMaxima,
    PeakDetectionMethod,
    PeakDetectionNeurokit2,
    PeakDetectionPantompkins,
    PeakDetectionProMAC,
    PeakDetectionXQRS,
    WFDBPeakDirection,
)

MIN_DIST = 15

PeakDetectionReturnType = TypeVar(
    "PeakDetectionReturnType",
    PeakDetectionElgendiPPG,
    PeakDetectionLocalMaxima,
    PeakDetectionNeurokit2,
    PeakDetectionPantompkins,
    PeakDetectionProMAC,
    PeakDetectionXQRS,
)


class EnumPeakDetectionMethod(Enum):
    elgendi_ppg = 0
    local = 1
    neurokit2 = 2
    promac = 3
    wfdb_xqrs = 4
    pantompkins = 5


class EnumPeakDetectionStackedPage(Enum):
    page_peak_elgendi_ppg = 0
    page_peak_local = 1
    page_peak_neurokit2 = 2
    page_peak_promac = 3
    page_peak_wfdb_xqrs = 4
    page_peak_pantompkins = 5


# region Peak Detection Parameters
# class UIPeakDetection(pTypes.GroupParameter):
#     def __init__(
#         self,
#         method: PeakDetectionMethod = "elgendi_ppg",
#         **kwargs: Unpack[GeneralParameterOptions],
#     ) -> None:
#         pTypes.GroupParameter.__init__(self, **kwargs)

#         self.active_method = method
#         self._relevant_children = []
#         self._PARAMETER_MAP = {
#             "elgendi_ppg": ELGENDI_PPG,
#             "local": LOCAL_MAX,
#             "neurokit2": NEUROKIT2,
#             "promac": PROMAC,
#             "wfdb_xqrs": XQRS,
#             "pantompkins": PANTOMPKINS,
#         }

#     def set_method(self, method: PeakDetectionMethod) -> None:
#         self.active_method = method
#         for child in self.children():
#             self.removeChild(child)

#         if self.hasChildren():
#             self.childs.clear()
#         # self.clearChildren()
#         if method not in self._PARAMETER_MAP:
#             raise NotImplementedError(f"Method `{method}` not implemented.")
#         self.addChildren(self._PARAMETER_MAP[method])
#         self._relevant_children = [name for name in self.names if "info" not in name]

#     @overload
#     def get_input_values(self, method: PeakDetectionMethod) -> PeakDetectionReturnType:
#         ...

#     def get_input_values(self, method: PeakDetectionMethod) -> PeakDetectionInputValues:
#         if self.active_method != method:
#             raise ValueError("Active method does not match the selected method")
#         vals = {name: self.names[name].value() for name in self._relevant_children}
#         if method == "wfdb_xqrs":
#             vals["corrections"] = {
#                 name: vals.pop(name)
#                 for name in {"search_radius", "smooth_window_size", "peak_dir"}
#             }
#             if vals["sampto"] == 0:
#                 vals["sampto"] = "end"
#         return vals

#     def get_values(self) -> PeakDetectionParameters:
#         if not hasattr(self, "active_method"):
#             raise ValueError("No method is set. Use `set_method` to set one.")
#         values = self.get_input_values(self.active_method)
#         return PeakDetectionParameters(method=self.active_method, input_values=values)
# endregion


type SmoothingKernels = Literal[
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


def _signal_smoothing_median[T: (np.float32, np.float64)](
    sig: npt.NDArray[T], size: int = 5
) -> npt.NDArray[T]:
    if size % 2 == 0:
        size += 1

    return signal.medfilt(sig, kernel_size=size)  # type: ignore


def _signal_smoothing(
    sig: npt.NDArray[np.floating[Any]], kernel: SmoothingKernels, size: int = 5
) -> npt.NDArray[np.float64]:
    window: npt.NDArray[np.float64] = signal.get_window(kernel, size)  # type: ignore
    w: npt.NDArray[np.float64] = window / window.sum()  # type: ignore

    x: npt.NDArray[np.float64] = np.concatenate(
        (sig[0] * np.ones(size), sig, sig[-1] * np.ones(size))
    )

    smoothed = np.convolve(w, x, mode="same")  # type: ignore
    return smoothed[size:-size]


def signal_smooth(
    sig: npt.NDArray[np.float64],
    method: Literal["convolution", "loess"] = "convolution",
    kernel: SmoothingKernels = "boxzen",
    size: int = 10,
    alpha: float = 0.1,
) -> npt.NDArray[np.float64]:
    length = sig.size

    if size > length or size < 1:
        raise ValueError(f"Size must be between 1 and {length}")

    if method == "loess":
        smoothed, _ = nk.fit_loess(sig, alpha=alpha)

    elif method == "convolution":
        if kernel == "boxcar":
            smoothed = np.asarray(
                ndimage.uniform_filter1d(sig, size=size, mode="nearest"),
                dtype=np.float64,
            )
        elif kernel == "boxzen":
            x: npt.NDArray[np.float64] = ndimage.uniform_filter1d(
                sig, size=size, mode="nearest"
            )  # type: ignore

            smoothed = _signal_smoothing(x, kernel="parzen", size=size)
        elif kernel == "median":
            smoothed = _signal_smoothing_median(sig, size=size)
        else:
            smoothed = _signal_smoothing(sig, kernel=kernel, size=size)

    return smoothed


def find_ppg_peaks_elgendi(
    sig: npt.NDArray[np.float64],
    sampling_rate: int,
    peakwindow: float = 0.111,
    beatwindow: float = 0.667,
    beatoffset: float = 0.02,
    mindelay: float = 0.3,
    **kwargs: Any,
) -> npt.NDArray[np.int32]:
    """
    Finds peaks in a PPG (photoplethysmography) signal using the method described by Elgendi et al. (see Notes)

    Parameters
    ----------
    sig : NDArray[np.float64]
        The PPG signal as a 1-dimensional NumPy array.
    sampling_rate : int
        The sampling rate of the PPG signal in samples per second.
    peakwindow : float, optional
        The width of the window used for smoothing the squared PPG signal to find peaks (in seconds).
    beatwindow : float, optional
        The width of the window used for smoothing the squared PPG signal to find beats (in seconds).
    beatoffset : float, optional
        The offset added to the smoothed beat signal to determine the threshold for detecting waves.
    mindelay : float, optional
        The minimum delay between consecutive peaks (in seconds).
    **kwargs : Any
        Additional keyword arguments.

    Returns
    -------
    npt.NDArray[np.int32]
        An array of peak indices as a 1-dimensional NumPy array.

    Notes
    -----
    This function implements the peak detection algorithm proposed by Elgendi et al. for
    PPG signals. The algorithm involves squaring the signal, applying a moving average
    with different window sizes for peak detection, and finding the local maxima in the
    resulting signal.

    For more information, see [Elgendi et al.](https://doi.org/10.1371/journal.pone.0076585).
    """
    sig_abs = sig.copy()
    sig_abs[sig_abs < 0] = 0
    sqrd = sig_abs**2

    peakwindow_samples = int(np.rint(peakwindow * sampling_rate))
    ma_peak = signal_smooth(sqrd, kernel="boxcar", size=peakwindow_samples)

    beatwindow_samples = int(np.rint(beatwindow * sampling_rate))
    ma_beat = signal_smooth(sqrd, kernel="boxcar", size=beatwindow_samples)

    thr1 = ma_beat + beatoffset * np.mean(sqrd, dtype=float)

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
        locmax, props = signal.find_peaks(data, prominence=(None, None))  # type: ignore

        if locmax.size > 0:
            peak = beg + locmax[np.argmax(props["prominences"])]  # type: ignore

            if not peaks or peak - peaks[-1] > min_delay_samples:
                peaks.append(peak)

    return np.array(peaks, dtype=np.int32)


def find_local_peaks(
    sig: npt.NDArray[np.float64],
    radius: int,
) -> npt.NDArray[np.intp]:
    if len(sig) == 0 or np.min(sig) == np.max(sig):
        return np.empty(0, dtype=np.int32)

    max_vals = ndimage.maximum_filter(sig, size=2 * radius + 1, mode="constant")  # type: ignore
    return np.nonzero(sig == max_vals)[0]  # type: ignore


def _shift_peaks(
    sig: npt.NDArray[np.float32 | np.float64],
    peaks: npt.NDArray[np.int32],
    radius: int,
    dir_is_up: bool,
) -> npt.NDArray[np.int32]:
    sig_len = sig.size
    start_indices = np.maximum(peaks - radius, 0)
    end_indices = np.minimum(peaks + radius, sig_len)

    shifted_peaks = np.zeros_like(peaks)

    for i, (start, end) in enumerate(zip(start_indices, end_indices, strict=False)):
        local_sig = sig[start:end]
        if dir_is_up:
            shifted_peaks[i] = np.subtract(np.argmax(local_sig), radius)
        else:
            shifted_peaks[i] = np.subtract(np.argmin(local_sig), radius)

    peaks += shifted_peaks
    return peaks
    # for i, peak_index in enumerate(peaks):
    #     start = max(0, peak_index - radius)
    #     end = min(peak_index + radius, sig_len)
    #     local_indices = np.arange(start, end)
    #     local_sig = sig[local_indices]

    #     if dir_is_up:
    #         shifted_peaks[i] = local_indices[np.argmax(local_sig)] - peak_index
    #     else:
    #         shifted_peaks[i] = local_indices[np.argmin(local_sig)] - peak_index

    # return peaks + shifted_peaks


def adjust_peak_positions(
    sig: npt.NDArray[np.float32 | np.float64],
    peaks: npt.NDArray[np.int32],
    radius: int,
    direction: WFDBPeakDirection,
) -> npt.NDArray[np.int32]:
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

        up_dist = np.mean(np.abs(sig[shifted_up]))
        down_dist = np.mean(np.abs(sig[shifted_down]))

        return shifted_up if np.greater_equal(up_dist, down_dist) else shifted_down
    else:
        raise ValueError(f"Unknown direction: {direction}")


def _handle_close_peaks(
    sig: npt.NDArray[np.float32 | np.float64],
    peak_idx: npt.NDArray[np.int32],
    find_peak_fn: Callable[[npt.NDArray[np.float32 | np.float64]], int],
) -> npt.NDArray[np.int32]:
    qrs_diffs = np.diff(peak_idx)
    close_inds = np.where(qrs_diffs <= MIN_DIST)[0]

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
    sig: npt.NDArray[np.float32 | np.float64],
    peak_idx: npt.NDArray[np.int32],
    n_std: float,
    find_peak_fn: Callable[[npt.NDArray[np.float32 | np.float64]], int],
) -> npt.NDArray[np.int32]:
    outliers = []
    window_size = 5
    for i, peak in enumerate(peak_idx):
        if i < 3:
            start_ind = 0
            end_ind = max(window_size, i + 3)
        else:
            start_ind = max(0, i - 2)
            end_ind = min(len(peak_idx), i + 3)

        surrounding_peaks = peak_idx[start_ind:end_ind]
        surrounding_vals = sig[surrounding_peaks]

        local_mean = np.mean(surrounding_vals)
        local_std = np.std(surrounding_vals)

        if find_peak_fn == np.argmax:
            threshold = local_mean - np.multiply(n_std, local_std)
            comparison_fn = np.less_equal
        elif find_peak_fn == np.argmin:
            threshold = local_mean + np.multiply(n_std, local_std)
            comparison_fn = np.greater_equal
        else:
            raise ValueError("find_peak_fn must be np.argmax or np.argmin")

        if comparison_fn(sig[peak], threshold):
            outliers.append(i)

    peak_idx = np.delete(peak_idx, outliers)
    return _handle_close_peaks(sig, peak_idx, find_peak_fn)


def sanitize_qrs_inds(
    sig: npt.NDArray[np.float32 | np.float64],
    qrs_inds: npt.NDArray[np.int32],
    n_std: float = 5.0,
) -> npt.NDArray[np.int32]:
    peak_idx = qrs_inds
    if np.less(np.mean(sig), np.mean(sig[peak_idx])):
        find_peak_fn = np.argmax
    else:
        find_peak_fn = np.argmin

    peak_idx = remove_outliers(sig, peak_idx, n_std, find_peak_fn)

    sorted_indices = np.argsort(peak_idx)

    sorted_indices = sorted_indices[peak_idx[sorted_indices] >= 0]

    return peak_idx[sorted_indices]


def combine_peak_indices(
    localmax_inds: npt.NDArray[np.int32],
    xqrs_inds: npt.NDArray[np.int32],
    threshold: int,
) -> npt.NDArray[np.int32]:
    combined_inds: list[int] = []
    used_xqrs_inds: set[int] = set()

    for localmax_ind in localmax_inds:
        for xqrs_ind in xqrs_inds:
            if (
                np.abs(localmax_ind - xqrs_ind) <= threshold
                and xqrs_ind not in used_xqrs_inds
            ):
                combined_inds.append((localmax_ind + xqrs_ind) // 2)
                used_xqrs_inds.add(xqrs_ind)
                break

    combined_inds.extend(
        xqrs_ind for xqrs_ind in xqrs_inds if xqrs_ind not in used_xqrs_inds
    )
    return np.array(combined_inds, dtype=np.int32)


def find_peaks_xqrs(
    sig: npt.NDArray[np.float64],
    processed_sig: npt.NDArray[np.float64],
    sampling_rate: int,
    radius: int,
    peak_dir: WFDBPeakDirection,
    sampfrom: int = 0,
    sampto: int = 0,
    **kwargs: Any,
) -> npt.NDArray[np.int32]:
    if sampto == 0:
        sampto = len(sig)
    xqrs_out = wfproc.XQRS(sig=sig, fs=sampling_rate)
    xqrs_out.detect(sampfrom=sampfrom, sampto=sampto)  # type: ignore
    qrs_inds = np.array(xqrs_out.qrs_inds, dtype=np.int32)
    peak_inds = adjust_peak_positions(
        processed_sig, peaks=qrs_inds, radius=radius, direction=peak_dir
    )
    return sanitize_qrs_inds(processed_sig, peak_inds)


type NKECGAlgorithms = Literal[
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


def neurokit2_find_peaks(
    sig: npt.NDArray[np.float32 | np.float64],
    sampling_rate: int,
    method: NKECGAlgorithms,
    **options: Any,
) -> npt.NDArray[np.int32]:
    artifact_correction = options.pop("correct_artifacts", False)
    if method == "promac":
        return _ecg_findpeaks_promac_sequential(
            sig, sampling_rate, artifact_correction, **options
        )
    peak_finder = _ecg_findpeaks_findmethod(method)  # type: ignore
    peaks = peak_finder(sig, sampling_rate=sampling_rate, **options)  # type: ignore
    peaks = (  # type: ignore
        nk.signal_fixpeaks(peaks, sampling_rate)[1] if artifact_correction else peaks
    )
    return peaks  # type: ignore


def _ecg_findpeaks_promac_addconvolve(
    sig: npt.NDArray[np.float32 | np.float64],
    sampling_rate: int,
    x: npt.NDArray[np.float32 | np.float64],
    fun: Callable[..., Any],
    gaussian_sd: int = 100,
    **kwargs: Any,
) -> npt.NDArray[np.floating[Any]]:
    peaks = fun(sig, sampling_rate=sampling_rate, **kwargs)
    mask = np.zeros(len(sig))
    mask[peaks] = 1

    sd = sampling_rate * gaussian_sd / 1000
    shape = stats.norm.pdf(
        np.linspace(-sd * 4, sd * 4, num=int(sd * 8)), loc=0, scale=sd
    )

    x += np.convolve(mask, shape, mode="same")
    return x


def ecg_findpeaks_promac_parallel(
    sig: npt.NDArray[np.float32 | np.float64],
    sampling_rate: int,
    correct_artifacts: bool,
    threshold: float = 0.33,
    gaussian_sd: int = 100,
    **kwargs: Any,
) -> npt.NDArray[np.int32]:
    with pg.ProgressDialog("Running ProMAC...", wait=0, cancelText=None) as progress:  # type: ignore
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
        cpu_count = os.cpu_count()
        workers = min(4, cpu_count // 2) if cpu_count is not None else 1
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _ecg_findpeaks_promac_addconvolve,
                    sig,
                    sampling_rate,
                    x,
                    _ecg_findpeaks_findmethod(method),  # type: ignore
                    gaussian_sd=gaussian_sd,
                    **kwargs,
                ): method
                for method in promac_methods
            }

            for future in as_completed(futures):
                x += future.result()
                progress += 1
                progress.setLabelText(
                    f"Computing probability density function\nfor method: {futures[future]}"
                )

        x = x / np.max(x)

        x[x < threshold] = 0

        peaks = nk.signal_findpeaks(x, relative_height_min=threshold)["Peaks"]  # type: ignore

        progress += 1

        if correct_artifacts:
            progress.setLabelText("Fixing artifacts...")
            peaks = nk.signal_fixpeaks(peaks, sampling_rate)[1]  # type: ignore

        progress.setLabelText("Done!")
        progress.setValue(9)

        return peaks  # type: ignore


def _ecg_findpeaks_promac_sequential(
    sig: npt.NDArray[np.float32 | np.float64],
    sampling_rate: int,
    correct_artifacts: bool,
    threshold: float = 0.33,
    gaussian_sd: int = 100,
    **kwargs: Any,
) -> npt.NDArray[np.int32]:
    with pg.ProgressDialog("Running ProMAC...", wait=0, cancelText=None) as progress:  # type: ignore
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
            func = _ecg_findpeaks_findmethod(method)  # type: ignore
            x = _ecg_findpeaks_promac_addconvolve(
                sig,
                sampling_rate,
                x,
                func,  # type: ignore
                gaussian_sd=gaussian_sd,
                **kwargs,
            )
            progress += 1

        x = x / np.max(x)

        x[x < threshold] = 0

        progress.setLabelText("Finding peaks...")
        peak_dict = nk.signal_findpeaks(x, relative_height_min=threshold)["Peaks"]  # type: ignore

        peaks = peak_dict["Peaks"]  # type: ignore
        progress += 1

        if correct_artifacts:
            progress.setLabelText("Fixing artifacts...")
            _, peaks = nk.signal_fixpeaks(peaks, sampling_rate)  # type: ignore

        progress.setLabelText("Done!")
        progress += 1

        return peaks  # type: ignore


def find_peak_function(
    method: PeakDetectionMethod,
) -> Callable[..., npt.NDArray[np.int32]]:
    if method == "elgendi_ppg":
        func = find_ppg_peaks_elgendi
    elif method == "local":
        func = find_local_peaks
    elif method == "neurokit2":
        func = partial(neurokit2_find_peaks, method="neurokit2")
    elif method == "pantompkins":
        func = partial(neurokit2_find_peaks, method="pantompkins")
    elif method == "promac":
        func = partial(neurokit2_find_peaks, method="promac")
    elif method == "wfdb_xqrs":
        func = find_peaks_xqrs
    else:
        raise ValueError(f"Method `{method}` unknown or not implemented.")

    return func


def find_peaks(
    sig: npt.NDArray[np.float64],
    sampling_rate: int,
    method: PeakDetectionMethod,
    input_values: PeakDetectionInputValues,
) -> npt.NDArray[np.int32]:
    peak_detection_func = find_peak_function(method)
    if method == "local":
        return peak_detection_func(sig, **input_values)
    return peak_detection_func(sig, sampling_rate, **input_values)
