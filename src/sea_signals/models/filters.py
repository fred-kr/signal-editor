from typing import Unpack

import neurokit2 as nk
import numpy as np
import polars as pl
from loguru import logger
from numpy.typing import NDArray

from ..custom_types import (
    SignalFilterParameters,
)


def mad_value(sig: pl.Expr) -> pl.Expr:
    return ((sig - sig.median()).abs()).median()


def mad(sig: pl.Expr, constant: float = 1.4826) -> pl.Expr:
    """
    Scale the given signal using the Median Absolute Deviation (MAD) method.

    Args:
        sig (pl.Expr): The signal to be scaled.
        constant (float, optional): The scaling constant. Defaults to 1.4826 (similar to R).

    Returns:
        pl.Expr: The scaled signal.
    """
    return (sig - sig.median()) / (mad_value(sig) * constant)


def z_score(sig: pl.Expr) -> pl.Expr:
    return (sig - sig.mean()) / sig.std()


def min_max(sig: pl.Expr) -> pl.Expr:
    return (sig - sig.min()) / (sig.max() - sig.min())


def filter_elgendi(sig: NDArray[np.float64], sampling_rate: int) -> NDArray[np.float32]:
    return np.asarray(
        nk.signal_filter(
            sig,
            sampling_rate=sampling_rate,
            lowcut=0.5,
            highcut=8,
            method="butterworth",
            order=3,
        ),
        dtype=np.float32,
    )


def filter_custom(
    sig: NDArray[np.float64],  # using a float32 array as input causes errors
    sampling_rate: int = 400,
    **kwargs: Unpack[SignalFilterParameters],
) -> NDArray[np.float32]:
    return np.asarray(
        nk.signal_filter(
            sig,
            sampling_rate=sampling_rate,
            **kwargs,  # type: ignore
        ),
        dtype=np.float32,
    )


def auto_correct_fir_length(
    sig: NDArray[np.float64],
    sampling_rate: int,
    **kwargs: Unpack[SignalFilterParameters],
) -> NDArray[np.float32]:
    """
    Tries to apply an FIR filter to the input signal. If it fails with an error about the filter length being too short, it reads the error message and sets the window size to the number given in the error message. This happens twice in case of bandpass filters.

    Args:
        sig (NDArray[np.float64]): The signal to be filtered.
        sampling_rate (int): The sampling rate of the signal.
        **kwargs (Unpack[SignalFilterParameters]): The parameters of the filter, passed to `nk.signal_filter`.

    Returns:
        NDArray[np.float32]: The filtered signal.
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
