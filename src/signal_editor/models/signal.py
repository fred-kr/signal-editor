from dataclasses import dataclass, field
from typing import Unpack

import polars as pl

from ..handlers.data_handler import ExclusionIndices
from ..type_aliases import Pipeline, SignalFilterParameters, SignalName
from .filters import (
    auto_correct_fir_length,
    filter_custom,
    filter_elgendi,
    scale_signal,
)


@dataclass(slots=True)
class Signal:
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
    excluded_sections: list[ExclusionIndices] = field(default_factory=list)
    active_section: pl.DataFrame = field(init=False, repr=False)
    _original_data: pl.DataFrame = field(init=False, repr=False)
    _processed_name: str = field(init=False, repr=False)
    is_finished: bool = False

    def __post_init__(self) -> None:
        self._processed_name = f"{self.name}_processed"
        if "index" not in self.data.columns:
            self.data = self.data.with_row_count("index")
        self.data = self.data.select(
            pl.col("index", "temperature", self.name),
            pl.zeros(self.data.height, dtype=pl.Float64).alias(self._processed_name),
            pl.repeat(True, self.data.height, dtype=pl.Boolean).alias("is_included"),
            pl.repeat(False, self.data.height, dtype=pl.Boolean).alias("is_processed"),
        )
        self._original_data = self.data.clone()
        self.active_section = self.data

    def set_active(self, start: int, stop: int) -> None:
        """
        Set the active section of the data based on the given start and stop indices.

        Parameters
        ----------
        start : int
            The start index of the active section.
        stop : int
            The stop index of the active section.

        Returns
        -------
        None
        """
        self.active_section = self.data.filter(pl.col("index").is_between(start, stop))

    def get_active(self) -> pl.DataFrame:
        return self.active_section

    def scale_values(
        self, robust: bool = False, window_size: int | None = None
    ) -> None:
        """
        Scale values in the active section using NumPy.

        Parameters
        ----------
        robust : bool, optional
            If True, use robust scaling. If False, use standard scaling.
            Default is False.
        window_size : int or None, optional
            Window size for scaling. Use None for no window.
            Default is None.

        Returns
        -------
        None
        """
        self.active_section = self.active_section.with_columns(
            (
                scale_signal(
                    self.active_section.get_column(self.name), robust, window_size
                )
            ).alias(self._processed_name)
        )

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
        sig = self.active_section.get_column(col_name).to_numpy()
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
            pl.Series(self._processed_name, filtered, pl.Float64).alias(
                self._processed_name
            )
        )

    def reset(self) -> None:
        self.data = self._original_data.clone()
        self.active_section = self.data
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
        self.excluded_sections.append(ExclusionIndices(start=start, stop=stop))
        self.data = self.data.with_columns(
            pl.when((pl.col("index").is_between(start, stop)))
            .then(False)
            .otherwise(pl.col("is_included"))
            .alias("is_included")
        )

    def apply_exclusion_mask(self) -> None:
        """
        Filters out the excluded sections from the data.
        """
        self.data = self.data.filter(~pl.col("is_included"))

    def mark_processed(self, start: int | None, stop: int | None) -> None:
        """
        Mark the processed section of the data based on the given start and stop indices.

        Parameters
        ----------
        start : int | None
            Start index of the processed section.
        stop : int | None
            Stop index of the processed section.
        """
        if start is None and stop is None:
            start = self.active_section["index"][0]
            stop = self.active_section["index"][-1]
        self.data = self.data.with_columns(
            pl.when((pl.col("index").is_between(start, stop)))
            .then(True)
            .otherwise(pl.col("is_processed"))
            .alias("is_processed")
        )

        if self.active_section.get_column("is_processed").all():
            self.is_finished = True
