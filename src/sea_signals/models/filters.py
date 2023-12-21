from typing import Literal, Unpack

import neurokit2 as nk
import numpy as np
import polars as pl
from loguru import logger
from numpy.typing import NDArray

from ..type_aliases import (
    SignalFilterParameters,
)


def mad_value[T: (pl.Expr, pl.Series)](sig: T) -> T | float | None:
    return (sig - sig.median()).abs().median()


def scale_mad[T: (pl.Expr, pl.Series)](sig: T, constant: float = 1.4826) -> T:
    mad_val = mad_value(sig)
    if mad_val is None:
        raise ValueError("MAD value is None")
    else:
        return (sig - sig.median()) / (mad_val * constant)


def scale_z[T: (pl.Expr, pl.Series)](sig: T) -> T:
    return (sig - sig.mean()) / sig.std()


def rolling_mad(
    sig: pl.Series, window_size: int, constant: float = 1.4826
) -> pl.Series:
    if window_size <= 0:
        raise ValueError("Window size must be greater than 0")
    deviation = sig - sig.rolling_median(window_size, min_periods=0)
    mad = deviation.abs().rolling_median(window_size, min_periods=0) * constant

    scaled_signal = deviation / mad
    return scaled_signal.fill_nan(0)
    # return (
    #     (sig - sig.rolling_median(window_size, min_periods=0))
    #     / (
    #         (sig - sig.rolling_median(window_size, min_periods=0))
    #         .abs()
    #         .rolling_median(window_size, min_periods=0)
    #         * constant
    #     )
    # ).fill_nan(sig[0])


def rolling_z(sig: pl.Series, window_size: int) -> pl.Series:
    return (
        (sig - sig.rolling_mean(window_size, min_periods=0))
        / sig.rolling_std(window_size, min_periods=0)
    ).fill_nan(sig[0])


def scale_signal(
    sig: pl.Series | NDArray[np.float32 | np.float64],
    robust: bool = False,
    window_size: int | Literal["None"] = "None",
) -> pl.Series:
    """
    Scales a signal series using either Z-score or median absolute
    deviation (MAD) scaling. The function can apply scaling on a
    rolling window basis if a window size is provided.

    Parameters
    ----------
    sig : polars.Series
        The input signal to scale.
    robust : bool, optional
        If True, use MAD for scaling, otherwise use Z-score.
        Defaults to False.
    window_size : int | None, optional
        The size of the rolling window over which to compute the
        scaling. If None, scale the entire series. Defaults to None.

    Returns
    -------
    polars.Series
        The scaled signal series.

    Notes
    -----
    Implementation based on the
    [neurokit2.standardize](https://neuropsychology.github.io/NeuroKit/functions/stats.html#standardize)
    function.
    """
    if isinstance(sig, np.ndarray):
        sig = pl.Series("", sig)
    sig = sig.cast(pl.Float64)

    if window_size == "None":
        return scale_mad(sig) if robust else scale_z(sig)
    else:
        return rolling_mad(sig, window_size) if robust else rolling_z(sig, window_size)


def filter_elgendi(sig: NDArray[np.float64], sampling_rate: int) -> NDArray[np.float64]:
    return np.asarray(
        nk.signal_filter(
            sig,
            sampling_rate=sampling_rate,
            lowcut=0.5,
            highcut=8,
            method="butterworth",
            order=3,
        ),
        dtype=np.float64,
    )


def filter_custom(
    sig: NDArray[np.float64],
    sampling_rate: int,
    **kwargs: Unpack[SignalFilterParameters],
) -> NDArray[np.float64]:
    return np.asarray(
        nk.signal_filter(
            sig,
            sampling_rate=sampling_rate,
            **kwargs,  # type: ignore
        ),
        dtype=np.float64,
    )


def auto_correct_fir_length(
    sig: NDArray[np.float64],
    sampling_rate: int,
    **kwargs: Unpack[SignalFilterParameters],
) -> NDArray[np.float64]:
    """
    Tries to apply an FIR filter to the input signal. If it fails with an error about the filter length being too short, it reads the error message and sets the window size to the number given in the error message. This happens twice in case of bandpass filters.

    Args:
        sig (NDArray[np.float64]): The signal to be filtered.
        sampling_rate (int): The sampling rate of the signal.
        **kwargs (Unpack[SignalFilterParameters]): The parameters of the filter, passed to `nk.signal_filter`.

    Returns:
        NDArray[np.float64]: The filtered signal.
    """
    while True:
        try:
            logger.info(f"Attempting FIR filtering with length {kwargs['window_size']}")
            return filter_custom(
                sig,
                sampling_rate=sampling_rate,
                **kwargs,
            )
        except ValueError as e:
            message = str(e)
            if "which requires" in message:
                required_samples = int(
                    message.split("requires")[1].split("samples")[0].strip()
                )
                logger.info(
                    f"Initial window size ({kwargs['window_size']}) too small, setting to {required_samples}"
                )
                kwargs["window_size"] = required_samples
            else:
                raise
