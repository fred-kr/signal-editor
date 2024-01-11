import copy
from dataclasses import dataclass, field
from typing import Iterable, NamedTuple, Unpack

import neurokit2 as nk
import numpy as np
import polars as pl
from numpy.typing import NDArray

from ..type_aliases import (
    PeakDetectionInputValues,
    PeakDetectionMethod,
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
    name: SignalName | str
    data: pl.DataFrame
    sampling_rate: int
    excluded_sections: list[SectionIndices] = field(default_factory=list)
    active_section: pl.DataFrame = field(init=False, repr=False)
    _signal_rate: RateData = field(default_factory=RateData)
    _original_data: pl.DataFrame = field(init=False, repr=False)
    processed_name: str = field(init=False, repr=False)
    _processed_data: NDArray[np.float64] = field(init=False, repr=False)
    peaks: NDArray[np.int32] = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    is_finished: bool = False
    _peak_index_offset: int = 0

    def __post_init__(self) -> None:
        self.processed_name = f"{self.name}_processed"
        if "index" not in self.data.columns:
            self.data = self.data.with_row_count("index")
        self.data = self.data.select(
            pl.col("index", "temperature", self.name),
            pl.col(self.name).alias(self.processed_name),
            pl.repeat(False, self.data.height, dtype=pl.Boolean).alias("is_peak"),
            pl.repeat(True, self.data.height, dtype=pl.Boolean).alias("is_included"),
            pl.repeat(False, self.data.height, dtype=pl.Boolean).alias("is_processed"),
        )
        self._original_data = self.data.clone()
        self.active_section = self.data.select(
            "index", "temperature", self.name, self.processed_name, "is_peak"
        )
        self._processed_data = self.active_section.get_column(
            self.processed_name
        ).to_numpy()

    @property
    def processed_data(self) -> NDArray[np.float64]:
        """Quick access to the filtered + scaled data column as a numpy array."""
        return self._processed_data

    @processed_data.setter
    def processed_data(self, value: NDArray[np.float64]) -> None:
        self._processed_data = value

    @property
    def signal_rate(self) -> RateData:
        return self._signal_rate

    @signal_rate.setter
    def signal_rate(self, value: RateData) -> None:
        self._signal_rate = value

    def as_dict(
        self,
    ) -> dict[str, str | int | bool | NDArray[np.float64] | NDArray[np.int32] | None]:
        """
        Returns a dictionary containing all information necessary to create a `Result`
        object instance.
        """
        return {
            "name": self.name,
            "sampling_rate": self.sampling_rate,
            "is_finished": self.is_finished,
            "data": self.data.to_numpy(structured=True),
            "excluded_sections": np.array(self.excluded_sections, dtype=np.int32),
            "original_data": self._original_data.to_numpy(structured=True),
            "rate": self.signal_rate.rate,
            "rate_interpolated": self.signal_rate.rate_interpolated,
            "peaks": self.peaks,
        }

    def _active_is_saved(self) -> bool:
        active_section_indices = self.active_section.get_column("index")
        filtered_data = self.data.select(
            "index", "temperature", self.name, self.processed_name, "is_peak"
        ).filter(pl.col("index").is_in(active_section_indices))
        return filtered_data.equals(self.active_section)

    def get_data(self) -> pl.DataFrame:
        """
        Saves any changes from the currently active section to `self.data` and returns it.

        Returns
        -------
        pl.DataFrame
            The updated `self.data` DataFrame.
        """
        if not self._active_is_saved():
            self.save_active()
        return self.data

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

    def reset(self) -> None:
        self.data = self._original_data.clone()
        self.active_section = self.data.select(
            "index", "temperature", self.name, self.processed_name, "is_peak"
        )
        self.excluded_sections = []
        self.is_finished = False
        self.peaks = np.zeros(0, dtype=np.int32)
        self._signal_rate.clear()
        self._processed_data = self.data.get_column(self.processed_name).to_numpy()

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
        kwargs : Unpack[SignalFilterParameters]
            Additional keyword arguments specifying filter parameters. The
            expected parameters depend on the selected pipeline or custom
            method.

        Returns
        -------
        None
        """
        method = kwargs.get("method", "None")
        fs = self.sampling_rate
        if col_name is None or col_name not in self.data.columns:
            col_name = self.name
        sig: NDArray[np.float64] = self.data.get_column(col_name).to_numpy()
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

        self.data = self.data.with_columns(
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
        if self.data.get_column(self.processed_name).is_null().all():
            use_col = self.name
        else:
            use_col = self.processed_name
        scaled = scale_signal(
            self.data.get_column(use_col), robust, window_size
        )

        self.data = self.data.with_columns(
            pl.Series(self.processed_name, scaled, pl.Float64).alias(
                self.processed_name
            )
        )

        self.processed_data = scaled.to_numpy(writable=True)

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
        self.data = coalesced_data.drop(
            f"{self.processed_name}_active", "is_peak_active"
        )
        self.mark_processed()

    def get_remaining(self) -> pl.DataFrame:
        """
        Returns the data where `is_processed` is False.
        """
        return self.data.filter(pl.col("is_processed").not_())

    def detect_peaks(
        self,
        method: PeakDetectionMethod,
        input_values: PeakDetectionInputValues,
        start_index: int = 0,
        stop_index: int = 0,
    ) -> None:
        if stop_index == 0:
            stop_index = len(self.processed_data)

        self._peak_index_offset = start_index
        sig = (
            self.active_section.filter(
                pl.col("index").is_between(start_index, stop_index)
            )
            .get_column(self.processed_name)
            .to_numpy()
        )

        peaks = find_peaks(
            sig=sig,
            sampling_rate=self.sampling_rate,
            method=method,
            input_values=input_values,
        )

        pl_peaks = pl.Series("peaks", peaks + start_index, pl.Int32)
        self.active_section = self.active_section.with_columns(
            (
                pl.when(pl.col("index").is_in(pl_peaks))
                .then(pl.lit(True))
                .otherwise(pl.col("is_peak"))
            ).alias("is_peak")
        )

        self.peaks = pl_peaks.to_numpy(writable=True)
        self.calculate_rate()

    def get_peak_indices(self) -> NDArray[np.int32]:
        self.peaks = (
            self.active_section.filter(pl.col("is_peak"))
            .get_column("index")
            .cast(pl.Int32)
            .to_numpy(writable=True)
        )
        return self.peaks

    def get_peak_diffs(
        self, peaks: NDArray[np.int32] | None = None
    ) -> NDArray[np.int32]:
        if peaks is None:
            peaks = self.peaks
        if len(peaks) < 2:
            return np.empty(0, dtype=np.int32)
        return np.ediff1d(peaks, to_begin=[0])

    def calculate_rate(self) -> None:
        """
        Calculates the rate of the signal using the detected peaks.

        Raises
        ------
        ValueError
            If there are less than 3 peaks.
        """
        if len(self.peaks) < 3:
            raise ValueError("Too few peaks to calculate rate.")
        fs = self.sampling_rate

        rate = np.asarray(
            nk.signal_rate(
                self.peaks,
                fs,
                desired_length=None,
                interpolation_method="monotone_cubic",
                show=False,
            ),
            dtype=np.float64,
        )
        rate_interp = np.asarray(
            nk.signal_rate(
                self.peaks,
                fs,
                desired_length=len(self.processed_data),
                interpolation_method="monotone_cubic",
                show=False,
            ),
            dtype=np.float64,
        )
        self.signal_rate.rate = rate
        self.signal_rate.rate_interpolated = rate_interp

    @property
    def data_bounds(self) -> tuple[int, int]:
        col = self.data.get_column("index")
        return col[0], col[-1]

    @property
    def active_region_limits(self) -> tuple[int, int]:
        col = self.active_section.get_column("index")
        return col[0], col[-1]


class SignalStorage(dict[SignalName | str, SignalData]):
    """
    A container for SignalData objects.
    """

    def __init__(
        self, *args: Iterable[tuple[str, SignalData]], **kwargs: SignalData
    ) -> None:
        super().__init__(*args, **kwargs)

    def __setitem__(self, key: SignalName | str, value: SignalData) -> None:
        super().__setitem__(key, value)

    def deepcopy(self) -> "SignalStorage":
        return copy.deepcopy(self)

    def update_sampling_rate(self, sampling_rate: int) -> None:
        for sig in self.values():
            sig.sampling_rate = sampling_rate
