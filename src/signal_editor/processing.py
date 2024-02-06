from typing import Unpack

import neurokit2 as nk
import numpy as np
import polars as pl
from loguru import logger
from numpy.typing import NDArray

from . import type_aliases as _t


def _mad_value(sig: pl.Series) -> float:
    mad = abs(sig - sig.median()).median()
    if isinstance(mad, (float, int)):
        return float(mad)
    else:
        raise ValueError("MAD value is not a float")


def _scale_mad(sig: pl.Series, constant: float = 1.4826) -> pl.Series:
    mad_val = _mad_value(sig)
    return (sig - sig.median()) / (mad_val * constant)


def _scale_z[T: (pl.Expr, pl.Series)](sig: T) -> T:
    return (sig - sig.mean()) / sig.std()


def _rolling_mad(sig: pl.Series, window_size: int, constant: float = 1.4826) -> pl.Series:
    if window_size <= 0:
        raise ValueError("Window size must be greater than 0")
    deviation = sig - sig.rolling_median(window_size, min_periods=0)
    mad = deviation.abs().rolling_median(window_size, min_periods=0) * constant

    scaled_signal = deviation / mad
    return scaled_signal.fill_nan(0)


def _rolling_z(sig: pl.Series, window_size: int) -> pl.Series:
    return (
        (sig - sig.rolling_mean(window_size, min_periods=0))
        / sig.rolling_std(window_size, min_periods=0)
    ).fill_nan(sig[0])


def scale_signal(
    sig: pl.Series | NDArray[np.float64],
    robust: bool = False,
    window_size: int | None = None,
) -> pl.Series:
    """
    Scales a signal series using either Z-score or median absolute deviation (MAD) scaling. The
    function can apply scaling on a rolling window basis if a window size is provided.

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

    if window_size:
        return _rolling_mad(sig, window_size) if robust else _rolling_z(sig, window_size)
    else:
        return _scale_mad(sig) if robust else _scale_z(sig)


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


# TODO: Look up how window size for FIR filters is calculated and replace the loop with that
def filter_signal(
    sig: NDArray[np.float64],
    sampling_rate: int,
    **kwargs: Unpack[_t.SignalFilterParameters],
) -> tuple[NDArray[np.float64], _t.SignalFilterParameters]:
    method = kwargs["method"]
    if method == "fir":
        max_attempts = 5  # Define a maximum number of attempts for FIR filtering

        for _ in range(max_attempts):
            try:
                logger.info(f"Attempting FIR filtering with length {kwargs['window_size']}")
                out = nk.signal_filter(sig, sampling_rate=sampling_rate, **kwargs)  # type: ignore
                break  # Exit the loop if filtering is successful
            except ValueError as e:
                message = str(e)
                if "which requires" not in message:
                    raise
                required_samples = int(message.split("requires")[1].split("samples")[0].strip())
                kwargs["window_size"] = required_samples
        else:
            raise RuntimeError(f"FIR filtering failed after {max_attempts} attempts")
    else:
        out = nk.signal_filter(sig, sampling_rate=sampling_rate, **kwargs)  # type: ignore

    return np.asarray(out, dtype=np.float64), kwargs
