import typing as t
from datetime import date, datetime
from pathlib import Path

import h5py
import mne.io
import numpy as np
import polars as pl

from .. import type_aliases as _t

if t.TYPE_CHECKING:
    from ..models.result import CompleteResult


def read_edf(
    file_path: str,
    start: int = 0,
    stop: int | None = None,
) -> tuple[pl.LazyFrame, datetime, float]:
    """
    Reads data from an EDF file into a polars DataFrame.

    Parameters
    ----------
    file_path : str
        Path to the EDF file.
    start : int, optional
        Index of the first sample to read, by default 0
    stop : int | None, optional
        Index of the last sample to read. Set to `None` (default) to read all samples

    Returns
    -------
    pl.LazyFrame
        The polars DataFrame containing the data from the EDF file.
    datetime
        The date of measurement stored in the EDF file.
    float
        The sampling rate stored in the EDF file.

    """
    raw_edf = mne.io.read_raw_edf(file_path)
    channel_names = t.cast(list[str], raw_edf.info.ch_names)
    date_measured = t.cast(datetime, raw_edf.info["meas_date"])
    sampling_rate = t.cast(float, raw_edf.info["sfreq"])

    rename_map = {
        "temp": "temperature",
        "hb": "hbr",
        "vent": "ventilation",
    }
    column_names = [
        next(
            (rename_map[key] for key in rename_map if key in name.lower()),
            f"channel_{i}",
        )
        for i, name in enumerate(channel_names)
    ]
    data, times = raw_edf.get_data(start=start, stop=stop, return_times=True)  # type: ignore
    lf = (
        pl.from_numpy(data.transpose(), schema={name: pl.Float64 for name in column_names})  # type: ignore
        .lazy()
        .with_row_count("index", offset=start)
        .with_columns(
            pl.Series("time_s", times, dtype=pl.Decimal),
            pl.col("temperature").round(1),
        )
        .filter((pl.col("temperature") != 0) & (pl.col("hbr") != 0) & (pl.col("ventilation") != 0))
        .select("index", "time_s", *column_names)
    )
    return lf, date_measured, sampling_rate


def create_hdf5_groups(
    group: h5py.Group,
    data: _t.CompleteResultDict,
) -> None:
    """
    Creates HDF5 groups from a dictionary.

    Parameters
    ----------
    group : h5py.Group
        The HDF5 group to create the groups in.
    data : ResultDict
        A dictionary containing the groups to create.
    """
    for key, value in data.items():
        if value is None:
            value = ()
            group.attrs.create(key, value)
        elif isinstance(value, (datetime, date)):
            value = value.isoformat()
            group.attrs.create(key, value)
        elif isinstance(value, (str, int, bool, float)):
            group.attrs.create(key, value)
        elif isinstance(value, (np.ndarray, list)):
            group.create_dataset(key, data=value)
        else:
            subgroup = group.create_group(key)
            create_hdf5_groups(subgroup, value)  # type: ignore


def write_hdf5(file_path: str | Path, result: "CompleteResult") -> None:
    file_path = Path(file_path).resolve().as_posix()

    with h5py.File(file_path, "a") as f:
        main_grp = f.create_group(f"{result.identifier.source_file_name}_results")
        res_dict = result.as_dict()

        create_hdf5_groups(main_grp, res_dict)
