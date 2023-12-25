import numpy as np
import polars as pl
from numpy.typing import ArrayLike, NDArray

from ..type_aliases import SignalName


class Signal(pl.Series):
    def __init__(
        self,
        name: SignalName,
        values: ArrayLike,
        dtype: pl.PolarsDataType = pl.Float64,
        *,
        strict: bool = True,
        nan_to_null: bool = False,
        dtype_if_empty: pl.PolarsDataType = pl.Null,
    ) -> None:
        super().__init__(
            name,
            values,
            dtype=dtype,
            strict=strict,
            nan_to_null=nan_to_null,
            dtype_if_empty=dtype_if_empty,
        )
        self._processed = None

    @property
    def processed(self) -> NDArray[np.float64]:
        return self._processed

    def standardize(self, robust: bool = False, window_size: int | None = None) -> None:
        pass
