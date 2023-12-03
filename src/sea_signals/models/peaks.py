from typing import Literal

import neurokit2 as nk
import numpy as np
from attrs import define, field, validators
from loguru import logger
from numpy.typing import NDArray
from pyqtgraph.parametertree import Parameter
from scipy import ndimage, signal
import wfdb.processing as wfp


@define
class ElgendiPPGPeaks:
    peakwindow: float = field(
        default=0.111,
        validator=[validators.ge(0.025), validators.le(0.500)],
        metadata={
            "unit": "seconds",
            "description": "window size of the systolic-peak duration in seconds",
        },
    )
    beatwindow: float = field(
        default=0.667,
        validator=[validators.ge(0.05), validators.le(2.5)],
        metadata={
            "unit": "seconds",
            "description": "window size of approximately one beat duration in seconds",
        },
    )
    beatoffset: float = field(
        default=0.02,
        validator=[validators.gt(0.0), validators.le(1.0)],
        metadata={
            "unit": "percentage",
            "description": "percentage value that gets multiplied with the mean of the squared signal to get the offset value",
        },
    )
    mindelay: float = field(
        default=0.3,
        metadata={
            "unit": "seconds",
            "description": "minimum delay between peaks in seconds",
        },
    )

    def make_parameters(self) -> Parameter:
        params = [
            {
                "name": "Peak Detection (Elgendi PPG)",
                "type": "group",
                "children": [
                    {
                        "name": "peakwindow",
                        "type": "slider",
                        "value": self.peakwindow,
                        "default": 0.111,
                        "step": 0.001,
                        "limits": [0.025, 0.500],
                        "precision": 3,
                    },
                    {
                        "name": "beatwindow",
                        "type": "slider",
                        "value": self.beatwindow,
                        "default": 0.667,
                        "step": 0.001,
                        "limits": [0.050, 2.500],
                        "precision": 3,
                    },
                    {
                        "name": "beatoffset",
                        "type": "slider",
                        "value": self.beatoffset,
                        "default": 0.02,
                        "step": 0.01,
                        "limits": [0.01, 1.0],
                        "precision": 2,
                    },
                    {
                        "name": "mindelay",
                        "type": "slider",
                        "value": self.mindelay,
                        "default": 0.3,
                        "step": 0.01,
                        "limits": [0.1, 15.0],
                        "precision": 2,
                    },
                ],
            },
        ]
        return Parameter.create(
            name="params_elgendi",
            type="group",
            children=params,
        )


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
    if sig.size == 0:
        logger.warning("Input signal size is 0. Returning an empty array.")
        return np.array([], dtype=np.int32)

    if np.min(sig) == np.max(sig):
        logger.warning("Input signal is constant. Returning an empty array.")
        return np.array([], dtype=np.int32)

    max_vals = np.asarray(
        ndimage.maximum_filter1d(sig, size=2 * radius + 1, mode="constant"),
        dtype=np.float32,
    )
    peak_indices = [i for i in range(sig.size) if sig[i] == max_vals[i]]

    # Skip the next `radius` elements after finding a peak
    peaks = []
    skip_until = -1
    for i in peak_indices:
        if i <= skip_until:
            continue
        peaks.append(i)
        skip_until = i + radius

    return np.array(peaks, dtype=np.int32)


def shift_peaks(
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
    direction: Literal["up", "down", "both", "compare", "None"],
) -> NDArray[np.int32]:
    if direction == "None":
        return peaks
    elif direction == "up":
        return shift_peaks(sig, peaks, radius, dir_is_up=True)
    elif direction == "down":
        return shift_peaks(sig, peaks, radius, dir_is_up=False)
    elif direction == "both":
        return shift_peaks(np.abs(sig), peaks, radius, dir_is_up=True)
    elif direction == "compare":
        shifted_up = shift_peaks(sig, peaks, radius, dir_is_up=True)
        shifted_down = shift_peaks(sig, peaks, radius, dir_is_up=False)

        up_dist = np.mean(np.abs(sig[shifted_up]), dtype=np.float32)
        down_dist = np.mean(np.abs(sig[shifted_down]), dtype=np.float32)

        return shifted_up if np.greater_equal(up_dist, down_dist) else shifted_down
    else:
        logger.warning(f"Unknown direction: `{direction}`. Returning original peaks.")
        return peaks
