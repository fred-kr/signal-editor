from dataclasses import dataclass, field
from typing import NamedTuple, Unpack

import neurokit2 as nk
import numpy as np
import polars as pl
from numpy.typing import NDArray

from ..type_aliases import (
    PeakDetectionInputValues,
    PeakDetectionMethod,
    PeakDetectionParameters,
    Pipeline,
    SignalFilterParameters,
    SignalName,
)
from .filters import (
    auto_correct_fir_length,
    filter_custom,
    filter_elgendi,
    scale_signal,
)
from .peaks import find_peaks


class SectionIndices(NamedTuple):
    start: int
    stop: int


@dataclass(slots=True, kw_only=True)
class RateData:
    """
    Holds both the interpolated and non-interpolated rate data for each signal. Results come from the
    `neurokit2.signal_rate` function.
    """

    rate: NDArray[np.float64] = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    rate_interpolated: NDArray[np.float64] = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )

    def update(
        self,
        rate: NDArray[np.float64] | None = None,
        rate_interpolated: NDArray[np.float64] | None = None,
    ) -> None:
        if rate is not None:
            self.rate = rate
        if rate_interpolated is not None:
            self.rate_interpolated = rate_interpolated

    def clear(self) -> None:
        self.rate = np.empty(0, dtype=np.float64)
        self.rate_interpolated = np.empty(0, dtype=np.float64)


@dataclass(slots=True)
class SignalData:
    """
    Signal class representing a signal with associated data and operations.

    Parameters
    ----------
    name : SignalName | str
        The name of the signal.
    data : pl.DataFrame
        The data associated with the signal.
    sampling_rate : int
        The sampling rate of the signal.
    excluded_sections : list[ExclusionIndices], optional
        List of excluded sections. Defaults to an empty list.
    active_section : pl.DataFrame, optional
        The currently active section of the signal. Defaults to the entire data.
    _original_data : pl.DataFrame, optional
        The original data of the signal. Defaults to a clone of the input data.
    _processed_name : str, optional
        The name of the processed signal. Defaults to "{name}_processed".
    is_finished : bool, optional
        Flag indicating if the signal processing is finished. Defaults to False.
    """

    name: SignalName | str
    data: pl.DataFrame
    sampling_rate: int
    excluded_sections: list[SectionIndices] = field(default_factory=list)
    active_section: pl.DataFrame = field(init=False, repr=False)
    _signal_rate: RateData = field(default_factory=RateData)
    _original_data: pl.DataFrame = field(init=False, repr=False)
    processed_name: str = field(init=False, repr=False)
    _processed_data: NDArray[np.float64] = field(init=False, repr=False)
    peaks: NDArray[np.int32] = field(
        default_factory=lambda: np.empty(0, dtype=np.int32)
    )
    is_finished: bool = False

    def __post_init__(self) -> None:
        self.processed_name = f"{self.name}_processed"
        if "index" not in self.data.columns:
            self.data = self.data.with_row_count("index")
        self.data = self.data.select(
            pl.col("index", "temperature", self.name),
            pl.repeat(None, self.data.height, dtype=pl.Float64).alias(
                self.processed_name
            ),
            pl.repeat(False, self.data.height, dtype=pl.Boolean).alias("is_peak"),
            pl.repeat(True, self.data.height, dtype=pl.Boolean).alias("is_included"),
            pl.repeat(False, self.data.height, dtype=pl.Boolean).alias("is_processed"),
        )
        self._original_data = self.data.clone()
        self.active_section = self.data.select(
            "index", "temperature", self.name, self.processed_name, "is_peak"
        )
        self._processed_data = np.empty(self.data.height, dtype=np.float64)

    @property
    def processed_data(self) -> NDArray[np.float64]:
        return self._processed_data

    @processed_data.setter
    def processed_data(self, value: NDArray[np.float64]) -> None:
        self._processed_data = value

    @property
    def signal_rate(self) -> RateData:
        return self._signal_rate

    def set_active(self, start: int, stop: int) -> None:
        """
        Set the active section of the data based on the given start and stop indices.

        Parameters
        ----------
        start : int
            The start index of the active section.
        stop : int
            The stop index of the active section.
        """
        self.active_section = self.data.select(
            "index", "temperature", self.name, self.processed_name, "is_peak"
        ).filter(pl.col("index").is_between(start, stop))

    def get_active(self) -> pl.DataFrame:
        """
        Get the currently active section of the data.

        Returns
        -------
        pl.DataFrame
            Subset of the total data for this `Signal` instance.
        """
        return self.active_section

    def reset(self) -> None:
        self.data = self._original_data.clone()
        self.active_section = self.data.select(
            "index", "temperature", self.name, self.processed_name, "is_peak"
        )
        self.excluded_sections = []

    def mark_excluded(self, start: int, stop: int) -> None:
        """
        Mark the excluded section of the data based on the given start and stop indices.

        Parameters
        ----------
        start : int
            Start index of the excluded section.
        stop : int
            Stop index of the excluded section.
        """
        self.excluded_sections.append(SectionIndices(start=start, stop=stop))
        self.excluded_sections.sort(key=lambda x: x.start)
        self.data = self.data.with_columns(
            pl.when((pl.col("index").is_between(start, stop)))
            .then(False)
            .otherwise(pl.col("is_included"))
            .alias("is_included")
        )

    def get_excluded_starts(self) -> list[int]:
        """
        Returns the start indices of the current excluded sections as a list of integers.
        """
        return [exclusion.start for exclusion in self.excluded_sections]

    def get_excluded_stops(self) -> list[int]:
        """
        Returns the stop indices of the current excluded sections as a list of integers.
        """
        return [exclusion.stop for exclusion in self.excluded_sections]

    def apply_exclusion_mask(self) -> None:
        """
        Filters out the excluded sections from the data.
        """
        self.data = self.data.filter(pl.col("is_included"))

    def mark_processed(
        self, start_index: int | None = None, stop_index: int | None = None
    ) -> None:
        """
        Marks the section between `start_index` and `stop_index` (both inclusive) as
        processed by setting the corresponding rows in the `is_processed` column to
        `True`.

        You can omit the `start_index` and `stop_index` arguments, in which case
        it marks the entire currently active region as processed.
        """
        if start_index is None:
            start_index = self.active_section["index"][0]
        if stop_index is None:
            stop_index = self.active_section["index"][-1]

        self.data = self.data.with_columns(
            pl.when(pl.col("index").is_between(start_index, stop_index))
            .then(True)
            .otherwise(pl.col("is_processed"))
            .alias("is_processed")
        )

        if self.data["is_processed"].all():
            self.is_finished = True

    def filter_values(
        self,
        pipeline: Pipeline | None = None,
        col_name: SignalName | str | None = None,
        **kwargs: Unpack[SignalFilterParameters],
    ) -> None:
        """
        Filters the signal values of a specified column using a given pipeline
        or custom method with additional parameters.

        Parameters
        ----------
        pipeline : Pipeline | None, optional
            The pipeline or method to use for filtering. If set to 'custom',
            the method specified in kwargs is used. If set to 'ppg_elgendi',
            the Elgendi method is used. If None, no filtering is applied.
        col_name : SignalName | str | None, optional
            The name of the column to filter. If None, the name of the signal
            itself is used.
        **kwargs : Unpack[SignalFilterParameters]
            Additional keyword arguments specifying filter parameters. The
            expected parameters depend on the selected pipeline or custom
            method.

        Returns
        -------
        None
        """
        method = kwargs.get("method", "None")
        fs = self.sampling_rate
        if col_name is None or col_name not in self.active_section.columns:
            col_name = self.name
        sig: NDArray[np.float64] = self.active_section.get_column(col_name).to_numpy()
        if pipeline == "custom":
            if method == "None":
                filtered = sig
            elif method == "fir":
                filtered = auto_correct_fir_length(sig, fs, **kwargs)
            else:
                filtered = filter_custom(sig, fs, **kwargs)
        elif pipeline == "ppg_elgendi":
            filtered = filter_elgendi(sig, fs)
        else:
            raise NotImplementedError(f"Pipeline `{pipeline}` not implemented.")

        self.active_section = self.active_section.with_columns(
            pl.Series(self.processed_name, filtered, pl.Float64).alias(
                self.processed_name
            )
        )

        self.processed_data = filtered

    def scale_values(
        self, robust: bool = False, window_size: int | None = None
    ) -> None:
        """
        Standardize a signal using either Z-score or median absolute deviation
        (MAD). Can be applied using a rolling window if a window size is provided.

        Parameters
        ----------
        robust : bool, optional
            If True, use MAD for scaling, otherwise use Z-score. Defaults to False.
        window_size : int | None, optional
            The size of the rolling window over which to compute the standardization.
            If None, standardize the entire signal. Defaults to None.
        """
        if self.active_section.get_column(self.processed_name).is_null().all():
            use_col = self.name
        else:
            use_col = self.processed_name
        scaled = scale_signal(self.active_section.get_column(use_col), robust, window_size)
        
        self.active_section = self.active_section.with_columns(
            (scaled).alias(self.processed_name)
        )

        self.processed_data = scaled.to_numpy()

    def save_active(self) -> None:
        """
        Writes the processed values from the active section to the original data.
        """
        # Join the active section with the original data
        joined_data = self.data.join(
            self.active_section.select("index", self.processed_name, "is_peak"),
            on="index",
            how="left",
            suffix="_active",
        )

        # Coalesce the processed columns and update the original processed column
        coalesced_data = joined_data.with_columns(
            pl.coalesce(
                pl.col(f"{self.processed_name}_active"),
                pl.col(self.processed_name),
            ).alias(self.processed_name),
            pl.coalesce(
                pl.col("is_peak_active"),
                pl.col("is_peak"),
            ).alias("is_peak"),
        )

        # Drop the temporary active column
        self.data = coalesced_data.drop(f"{self.processed_name}_active", "is_peak_active")

    def get_remaining(self) -> pl.DataFrame:
        """
        Returns the data where `is_processed` is False.
        """
        return self.data.filter(pl.col("is_processed").not_())

    def detect_peaks(self, method: PeakDetectionMethod, input_values: PeakDetectionInputValues) -> None:
        sig = self.active_section.get_column(self.processed_name).to_numpy()

        self.peaks = find_peaks(
            sig, self.sampling_rate, method=method, input_values=input_values
        )

        self.active_section = self.active_section.with_columns(
            pl.when(pl.col("index").is_in(self.peaks))
            .then(True)
            .otherwise(pl.col("is_peak"))
            .alias("is_peak")
        )

    def get_peak_indices(self) -> NDArray[np.int32]:
        return (
            self.active_section.filter(pl.col("is_peak")).get_column("index").to_numpy()
        )

    def get_peak_diffs(self) -> NDArray[np.int32]:
        if self.peaks.size < 2:
            return np.array([], dtype=np.int32)
        return np.ediff1d(self.peaks, to_begin=[0])

    def calculate_rate(self) -> None:
        peaks = self.peaks
        if peaks.size < 3:
            raise ValueError("Too few peaks to calculate rate.")
        fs = self.sampling_rate

        rate = np.asarray(
            nk.signal_rate(
                peaks,
                fs,
                desired_length=None,
                interpolation_method="monotone_cubic",
                show=False,
            ),
            dtype=np.float64,
        )
        rate_interp = np.asarray(
            nk.signal_rate(
                peaks,
                fs,
                desired_length=self.data.height,
                interpolation_method="monotone_cubic",
                show=False,
            ),
            dtype=np.float64,
        )
        self.signal_rate.update(rate=rate, rate_interpolated=rate_interp)
