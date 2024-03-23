import typing as t

import neurokit2 as nk
import numpy as np
import numpy.typing as npt
import wfdb.processing as wfproc
from scipy import ndimage, signal

from . import type_aliases as _t

MIN_DIST = 20


def _signal_smoothing_median(
    sig: npt.NDArray[np.float64], size: int = 5
) -> npt.NDArray[np.float64]:
    if size % 2 == 0:
        size += 1

    return signal.medfilt(sig, kernel_size=size)  # type: ignore


def _signal_smoothing(
    sig: npt.NDArray[np.floating[t.Any]], kernel: _t.SmoothingKernels, size: int = 5
) -> npt.NDArray[np.float64]:
    window: npt.NDArray[np.float64] = signal.get_window(kernel, size)  # type: ignore
    w: npt.NDArray[np.float64] = window / window.sum()  # type: ignore

    x: npt.NDArray[np.float64] = np.concatenate(
        (sig[0] * np.ones(size), sig, sig[-1] * np.ones(size))
    )

    smoothed = np.convolve(w, x, mode="same")  # type: ignore
    return smoothed[size:-size]


def _signal_smooth(
    sig: npt.NDArray[np.float64],
    method: t.Literal["convolution", "loess"] = "convolution",
    kernel: _t.SmoothingKernels = "boxzen",
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
            x = ndimage.uniform_filter1d(sig, size=size, mode="nearest")  # type: ignore

            smoothed = _signal_smoothing(x, kernel="parzen", size=size)  # type: ignore
        elif kernel == "median":
            smoothed = _signal_smoothing_median(sig, size=size)
        else:
            smoothed = _signal_smoothing(sig, kernel=kernel, size=size)

    return smoothed


def _find_ppg_peaks_elgendi(
    sig: npt.NDArray[np.float64],
    sampling_rate: int,
    peakwindow: float = 0.111,
    beatwindow: float = 0.667,
    beatoffset: float = 0.02,
    mindelay: float = 0.3,
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
    sig_clipped_squared = np.clip(sig, 0, None) ** 2

    peakwindow_samples = np.rint(peakwindow * sampling_rate).astype(np.int32)
    ma_peak = _signal_smooth(sig_clipped_squared, kernel="boxcar", size=peakwindow_samples)

    beatwindow_samples = np.rint(beatwindow * sampling_rate).astype(np.int32)
    ma_beat = _signal_smooth(sig_clipped_squared, kernel="boxcar", size=beatwindow_samples)

    thr1 = ma_beat + beatoffset * np.mean(sig_clipped_squared)

    waves = ma_peak > thr1
    wave_changes = np.diff(waves.astype(np.int32))
    beg_waves = np.flatnonzero(wave_changes == 1)
    end_waves = np.flatnonzero(wave_changes == -1)

    if end_waves[0] < beg_waves[0]:
        end_waves = end_waves[1:]
    if end_waves[-1] < beg_waves[-1]:
        beg_waves = beg_waves[:-1]

    diff_waves = end_waves - beg_waves
    valid_waves = diff_waves >= peakwindow_samples
    beg_waves = beg_waves[valid_waves]
    end_waves = end_waves[valid_waves]

    min_delay_samples = np.rint(mindelay * sampling_rate).astype(np.int32)
    peaks: list[int] = []

    for beg, end in zip(beg_waves, end_waves, strict=False):
        data = sig[beg:end]
        locmax, props = signal.find_peaks(data, prominence=(None, None))

        if locmax.size > 0:
            peak = beg + locmax[props["prominences"].argmax()]

            if not peaks or peak - peaks[-1] > min_delay_samples:
                peaks.append(peak)

    return np.array(peaks, dtype=np.int32)


def _find_local_maxima(
    sig: npt.NDArray[np.float64],
    radius: int,
) -> npt.NDArray[np.intp]:
    if sig.size == 0 or np.min(sig) == np.max(sig):
        return np.array([], dtype=np.int32)

    max_vals = ndimage.maximum_filter1d(sig, size=2 * radius + 1, mode="constant")
    return np.flatnonzero(sig == max_vals)


def _find_local_minima(
    sig: npt.NDArray[np.float64],
    radius: int,
) -> npt.NDArray[np.intp]:
    if sig.size == 0 or np.min(sig) == np.max(sig):
        return np.array([], dtype=np.int32)

    min_vals = ndimage.minimum_filter1d(sig, size=2 * radius + 1, mode="constant")
    return np.flatnonzero(sig == min_vals)


def find_extrema(
    sig: npt.NDArray[np.float64], radius: int, direction: t.Literal["up", "down"]
) -> npt.NDArray[np.intp]:
    if direction == "up":
        peaks = _find_local_maxima(sig, radius)
    elif direction == "down":
        peaks = _find_local_minima(sig, radius)
    else:
        raise ValueError(f"Unknown direction: {direction}")

    # Merge peaks that are too close together
    peak_diffs = np.diff(peaks)
    close_peaks = np.where(peak_diffs <= MIN_DIST)[0]
    while len(close_peaks) > 0:
        # Replace the two close peaks with their midpoint
        for i in close_peaks:
            peaks[i] = (peaks[i] + peaks[i + 1]) // 2
        peaks = np.delete(peaks, close_peaks + 1)
        peak_diffs = np.diff(peaks)
        close_peaks = np.where(peak_diffs <= MIN_DIST)[0]

    return peaks


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


def _adjust_peak_positions(
    sig: npt.NDArray[np.float32 | np.float64],
    peaks: npt.NDArray[np.int32],
    radius: int,
    direction: t.Literal["up", "down", "both", "compare"],
) -> npt.NDArray[np.int32]:
    if direction == "up":
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


def _get_comparison_fn(find_peak_fn: t.Callable[..., np.int_]) -> t.Callable[..., np.bool_]:
    if find_peak_fn == np.argmax:
        return np.less_equal
    elif find_peak_fn == np.argmin:
        return np.greater_equal
    else:
        raise ValueError(f"find_peak_fn {find_peak_fn} not supported.")


def _remove_outliers(
    sig: npt.NDArray[np.float64],
    peak_idx: npt.NDArray[np.int32],
    n_std: float,
    find_peak_fn: t.Callable[..., np.intp],
) -> npt.NDArray[np.int32]:
    comparison_ops = {np.argmax: (np.less_equal, -1), np.argmin: (np.greater_equal, 1)}

    if find_peak_fn not in comparison_ops:
        raise ValueError("find_peak_fn must be np.argmax or np.argmin")

    comparison_fn, direction = comparison_ops[find_peak_fn]
    outliers_mask = np.zeros_like(peak_idx, dtype=bool)

    for i, peak in enumerate(peak_idx):
        start_ind = max(0, i - 2)
        end_ind = min(len(peak_idx), i + 3)

        surrounding_peaks = peak_idx[start_ind:end_ind]
        surrounding_vals = sig[surrounding_peaks]
        local_mean = np.mean(surrounding_vals)
        local_std = np.std(surrounding_vals)
        threshold = local_mean + direction * n_std * local_std

        if comparison_fn(sig[peak], threshold):
            outliers_mask[i] = True

    peak_idx = peak_idx[~outliers_mask]
    return peak_idx


def _handle_close_peaks(
    sig: npt.NDArray[np.float64],
    peak_idx: npt.NDArray[np.int32],
    n_std: float,
    find_peak_fn: t.Callable[..., np.intp],
) -> npt.NDArray[np.int32]:
    qrs_diffs = np.diff(peak_idx)
    close_inds = np.where(qrs_diffs <= MIN_DIST)[0]

    if not close_inds.size:
        return peak_idx

    comparison_fn = _get_comparison_fn(find_peak_fn)
    to_remove = [
        ind if comparison_fn(sig[peak_idx[ind]], sig[peak_idx[ind + 1]]) else ind + 1
        for ind in close_inds
    ]

    peak_idx = np.delete(peak_idx, to_remove)
    return _remove_outliers(sig, peak_idx, n_std, find_peak_fn)


def _sanitize_qrs_inds(
    sig: npt.NDArray[np.float64],
    qrs_inds: npt.NDArray[np.int32],
    n_std: float = 4.0,
) -> npt.NDArray[np.int32]:
    find_peak_fn = np.argmax if np.mean(sig) < np.mean(sig[qrs_inds]) else np.argmin
    peak_idx = _handle_close_peaks(sig, qrs_inds, n_std, find_peak_fn)
    sorted_indices = np.argsort(peak_idx)
    return peak_idx[
        sorted_indices[(peak_idx[sorted_indices] > 0) & (peak_idx[sorted_indices] < sig.size)]
    ]


def _find_peaks_xqrs(
    sig: npt.NDArray[np.float64],
    sampling_rate: int,
    radius: int,
    peak_dir: t.Literal["up", "down", "both", "compare"] = "up",
) -> npt.NDArray[np.int32]:
    xqrs_out = wfproc.XQRS(sig=sig, fs=sampling_rate)
    xqrs_out.detect()
    qrs_inds = np.array(xqrs_out.qrs_inds, dtype=np.int32)
    peak_inds = _adjust_peak_positions(sig, peaks=qrs_inds, radius=radius, direction=peak_dir)
    return _sanitize_qrs_inds(sig, peak_inds)


def find_peaks(
    sig: npt.NDArray[np.float64],
    sampling_rate: int,
    method: _t.PeakDetectionMethod,
    method_parameters: _t.PeakDetectionInputValues,
) -> npt.NDArray[np.int32]:
    """
    Finds peaks in a signal using the specified method.

    Parameters
    ----------
    sig : npt.NDArray[np.float64]
        Signal as a 1-dimensional NumPy array.
    sampling_rate : int
        Sampling rate of the signal in samples per second.
    method : PeakDetectionMethod
        The peak detection method to use.
    input_values : PeakDetectionInputValues
        Dictionary of method-specific input values.

    Returns
    -------
    npt.NDArray[np.int32]
        An array of peak indices as a 1-dimensional NumPy array.

    """
    match method:
        case "local":
            return find_extrema(
                sig, radius=method_parameters.get("radius", sampling_rate // 2), direction="up"
            )
        case "local_min":
            return find_extrema(
                sig, radius=method_parameters.get("radius", sampling_rate // 2), direction="down"
            )
        case "elgendi_ppg":
            return _find_ppg_peaks_elgendi(
                sig,
                sampling_rate,
                peakwindow=method_parameters.get("peakwindow", 0.111),
                beatwindow=method_parameters.get("beatwindow", 0.667),
                beatoffset=method_parameters.get("beatoffset", 0.02),
                mindelay=method_parameters.get("mindelay", 0.3),
            )
        case "wfdb_xqrs":
            return _find_peaks_xqrs(
                sig,
                sampling_rate,
                radius=method_parameters.get("search_radius", sampling_rate // 2),
                peak_dir=method_parameters.get("peak_dir", "up"),
            )
        case "neurokit2" | "promac" | "pantompkins":
            return nk.ecg_peaks(
                ecg_cleaned=sig,
                sampling_rate=sampling_rate,
                method=method,
                correct_artifacts=method_parameters.get("correct_artifacts", False),
            )[1]["ECG_R_Peaks"]  # type: ignore


# region Unused functions
# def _combine_peak_indices(
#     localmax_inds: npt.NDArray[np.int32],
#     xqrs_inds: npt.NDArray[np.int32],
#     threshold: int,
# ) -> npt.NDArray[np.int32]:
#     combined_inds = []
#     used_xqrs_inds: set[int] = set()

#     for localmax_ind in localmax_inds:
#         # Find the index in xqrs_inds that is within the threshold and not used
#         close_inds = np.abs(xqrs_inds - localmax_ind) <= threshold
#         if unused_close_inds := [ind for ind in xqrs_inds[close_inds] if ind not in used_xqrs_inds]:
#             xqrs_ind = unused_close_inds[0]
#             combined_inds.append((localmax_ind + xqrs_ind) // 2)
#             used_xqrs_inds.add(xqrs_ind)

#     # Add the remaining unused xqrs_inds to combined_inds
#     combined_inds.extend(xqrs_inds[~np.isin(xqrs_inds, list(used_xqrs_inds))])
#     return np.array(combined_inds, dtype=np.int32)
# endregion
