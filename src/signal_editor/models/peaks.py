from functools import partial
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Literal, Unpack, cast

import neurokit2 as nk
import numpy as np
import pyqtgraph as pg
import wfdb.processing as wfproc
from neurokit2.ecg.ecg_findpeaks import _ecg_findpeaks_findmethod
from neurokit2.signal.signal_fixpeaks import _signal_fixpeaks_kubios
from numpy.typing import NDArray
from pyqtgraph.parametertree import parameterTypes as pTypes
from scipy import ndimage, signal, stats

from ..type_aliases import (
    GeneralParameterOptions,
    PeakDetectionInputValues,
    PeakDetectionMethod,
    PeakDetectionParameters,
    WFDBPeakDirection,
)
from ..views.widgets._parameter_tree_schemas import (
    ELGENDI_PPG,
    LOCAL_MAX,
    NEUROKIT2,
    PANTOMPKINS,
    PROMAC,
    XQRS,
)

MIN_DIST = 15


class UIPeakDetection(pTypes.GroupParameter):
    def __init__(
        self,
        method: PeakDetectionMethod = "elgendi_ppg",
        **kwargs: Unpack[GeneralParameterOptions],
    ) -> None:
        pTypes.GroupParameter.__init__(self, **kwargs)

        self.active_method = method
        self._relevant_children = []
        self._PARAMETER_MAP = {
            "elgendi_ppg": ELGENDI_PPG,
            "local": LOCAL_MAX,
            "neurokit2": NEUROKIT2,
            "promac": PROMAC,
            "wfdb_xqrs": XQRS,
            "pantompkins": PANTOMPKINS,
        }

    def set_method(self, method: PeakDetectionMethod) -> None:
        self.active_method = method
        self.clearChildren()
        if method not in self._PARAMETER_MAP:
            raise NotImplementedError(f"Method `{method}` not implemented.")
        self.addChildren(self._PARAMETER_MAP[method])
        self._relevant_children = [name for name in self.names if "info" not in name]

    def get_values(self) -> PeakDetectionParameters:
        """
        Get the current values of the parameters as a dictionary with the keys:
        - method (the name of the method)
        - input_values (dictionary of `parameter_name: parameter_value` for the method's parameters)

        Raises:
            ValueError: If no method is set

        Returns:
            PeakDetectionKwargs: Dictionary with the selected method name and its parameters
        """
        if not hasattr(self, "active_method"):
            raise ValueError("No method is set. Use `set_method` to set one.")
        values = {name: self.names[name].value() for name in self._relevant_children}
        if self.active_method == "wfdb_xqrs":
            values["corrections"] = {
                name: values.pop(name)
                for name in {"search_radius", "smooth_window_size", "peak_dir"}
            }
            if values["sampto"] == 0:
                values["sampto"] = "end"
        return PeakDetectionParameters(
            method=cast(PeakDetectionMethod, self.active_method),
            input_values=cast(PeakDetectionInputValues, values),
        )


def find_ppg_peaks_elgendi(
    sig: NDArray[np.float32 | np.float64],
    sampling_rate: int,
    peakwindow: float = 0.111,
    beatwindow: float = 0.667,
    beatoffset: float = 0.02,
    mindelay: float = 0.3,
    **kwargs: Any,
) -> NDArray[np.int32]:
    """
    Find the peaks in a PPG (Photoplethysmography) signal using the algorithm proposed
    by: `Elgendi M, Norton I, Brearley M, Abbott D, Schuurmans D (2013) Systolic Peak Detection in
    Acceleration Photoplethysmograms Measured from Emergency Responders in Tropical Conditions. PLoS ONE 8(10): e76585.
    doi:10.1371/journal.pone.0076585.`
    Based on the implementation of Neurokit's `_ppg_findpeaks_elgendi` function.

    Parameters:
        sig (NDArray[np.float32 | np.float64]): Input PPG signal.
        sampling_rate (int, optional): Sampling rate of the input signal.
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
    radius: np.int32,
    dir_is_up: bool,
) -> NDArray[np.int32]:
    sig_len = len(sig)
    shifted_peaks = np.zeros_like(peaks)

    for i, peak_index in enumerate(peaks):
        start = max(0, peak_index - radius)
        end = min(peak_index + radius, sig_len)
        local_indices = np.arange(start, end)
        local_sig = sig[local_indices]

        if dir_is_up:
            shifted_peaks[i] = local_indices[np.argmax(local_sig)] - peak_index
        else:
            shifted_peaks[i] = local_indices[np.argmin(local_sig)] - peak_index

    return peaks + shifted_peaks


def adjust_peak_positions(
    sig: NDArray[np.float32 | np.float64],
    peaks: NDArray[np.int32],
    radius: np.int32,
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

        up_dist = np.mean(np.abs(sig[shifted_up]))
        down_dist = np.mean(np.abs(sig[shifted_down]))

        return shifted_up if up_dist >= down_dist else shifted_down
    else:
        raise ValueError(f"Unknown direction: {direction}")


def _handle_close_peaks(
    sig: NDArray[np.float32 | np.float64],
    peak_idx: NDArray[np.int32],
    find_peak_fn: Callable[[NDArray[np.float32 | np.float64]], int],
) -> NDArray[np.int32]:
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
            end_ind = min(len(peak_idx), i + 3)

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
    peak_idx = np.array(peak_idx, dtype=np.int32)
    return _handle_close_peaks(sig, peak_idx, find_peak_fn)


def sanitize_qrs_inds(
    sig: NDArray[np.float32 | np.float64],
    qrs_inds: NDArray[np.int32],
    n_std: float = 5.0,
) -> NDArray[np.int32]:
    peak_idx = qrs_inds
    if np.mean(sig) < np.mean(sig[peak_idx]):
        find_peak_fn = np.argmax
    else:
        find_peak_fn = np.argmin

    peak_idx = remove_outliers(sig, peak_idx, n_std, find_peak_fn)

    sorted_indices = np.argsort(peak_idx)

    sorted_indices = sorted_indices[peak_idx[sorted_indices] >= 0]

    return np.asarray(peak_idx[sorted_indices], dtype=np.int32)


def combine_peak_indices(
    localmax_inds: NDArray[np.int32], xqrs_inds: NDArray[np.int32], threshold: int
) -> NDArray[np.int32]:
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
    return np.asarray(combined_inds, dtype=np.int32)


def find_peaks_xqrs(
    sig: NDArray[np.float64],
    processed_sig: NDArray[np.float64],
    sampling_rate: int,
    radius: np.int32,
    peak_dir: WFDBPeakDirection,
    sampfrom: int = 0,
    sampto: int = 0,
    **kwargs: Any,
) -> NDArray[np.int32]:
    if sampto == 0:
        sampto = len(sig)
    xqrs_out = wfproc.XQRS(sig=sig, fs=sampling_rate)
    xqrs_out.detect(sampfrom=sampfrom, sampto=sampto)
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
    sig: NDArray[np.float32 | np.float64],
    sampling_rate: int,
    method: NKECGAlgorithms,
    **options: Any,
) -> NDArray[np.int32]:
    artifact_correction = options.pop("correct_artifacts", False)
    if method == "promac":
        return _ecg_findpeaks_promac_sequential(
            sig, sampling_rate, artifact_correction, **options
        )
    peak_finder = _ecg_findpeaks_findmethod(method)
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


def ecg_findpeaks_promac_parallel(
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
        peaks = nk.signal_findpeaks(x, relative_height_min=threshold)["Peaks"]

        progress += 1

        if correct_artifacts:
            progress.setLabelText("Fixing artifacts...")
            peaks = _signal_fixpeaks_kubios(peaks, sampling_rate)[1]

        progress.setLabelText("Done!")
        progress += 1

        return np.asarray(peaks, dtype=np.int32)


def find_peak_function(method: PeakDetectionMethod) -> Callable[..., NDArray[np.int32]]:
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
    sig: NDArray[np.float64],
    sampling_rate: int,
    method: PeakDetectionMethod,
    input_values: PeakDetectionInputValues,
) -> NDArray[np.int32]:
    peak_detection_func = find_peak_function(method)
    if method == "local":
        return peak_detection_func(sig, **input_values)
    return peak_detection_func(sig, sampling_rate, **input_values)