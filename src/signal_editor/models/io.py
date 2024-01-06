import re
import sys
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, cast

import h5py
import mne.io
import polars as pl
from loguru import logger

if TYPE_CHECKING:
    from ..models.result import Result


def parse_file_name(
    file_name: str,
    date_pattern: str = r"\d{8}",
    id_pattern: str = r"(?:P[AM]|F)\d{1,2}",
) -> tuple[date, str, str]:
    """
    Parses the file name for the date, id, and oxygen condition.

    Parameters
    ----------
    file_name : str
        String representing the file name.
    date_pattern : str, optional
        The regular expression pattern for the date, by default r"\\d{8}"
    id_pattern : str, optional
        The regular expression pattern for the id, by default r"(?:P[AM]|F)\\d{1,2}"

    Returns
    -------
    tuple[date, str, str]
        The date, id, and oxygen condition parsed from the file name, or 'unknown' if
        the respective pattern was not found.
    """
    date_match = re.search(date_pattern, file_name)
    id_match = re.search(id_pattern, file_name)
    if "hyp" in file_name:
        oxy_condition = "hypoxic"
    elif "norm" in file_name:
        oxy_condition = "normoxic"
    else:
        oxy_condition = "unknown"

    if not date_match:
        date_ddmmyyyy = datetime.now()
    else:
        date_ddmmyyyy = date(
            year=int(date_match[0][4:8], base=10),
            month=int(date_match[0][2:4], base=10),
            day=int(date_match[0][:2], base=10),
        )
    id_str = str(id_match[0]) if id_match else "unknown"
    return date_ddmmyyyy, id_str, oxy_condition


def format_column_names(columns: Iterable[str]) -> list[str]:
    """
    Formats a list of column names by sanitizing unwanted characters and spaces, and
    ensures all names are lowercase. If any column name is empty, assigns a default name
    with an incremented index.

    Parameters
    ----------
    columns : Iterable[str]
        An iterable of column names.

    Returns
    -------
    list[str]
        A list containing the formatted column names.

    Raises
    ------
    ValueError
        If input iterable is empty.
    """
    if not columns:
        raise ValueError("Column names cannot be empty.")
    logger.debug(f"Original column names: {columns}")
    default_column_name = "column"
    unique_columns: list[str] = []
    for i, col in enumerate(columns, start=1):
        formatted_col = (
            re.sub(r"\W+|_+", "_", col.lower().replace(" ", "_"))
            if col
            else f"{default_column_name}_{i}"
        )
        formatted_col = re.sub(r"_+", "_", formatted_col)
        formatted_col = formatted_col.rstrip("_")
        unique_columns.append(formatted_col)
    logger.debug(f"Formatted column names: {unique_columns}")
    return unique_columns


def read_excel(
    file_path: str | Path,
    sheet_id: int = 0,
    **kwargs: Any,
) -> pl.DataFrame:
    """
    Reads an Excel file and returns the contents as a pandas DataFrame.

    Parameters:
        file_path (str | Path): The path to the Excel file.
        sheet_id (int, optional): The index of the sheet to read. Defaults to 0.
        **kwargs: Additional keyword arguments to be passed to the pandas `read_excel` function.

    Returns:
        pl.DataFrame: The contents of the Excel file as a pandas DataFrame.
    """
    return pl.read_excel(
        file_path,
        sheet_id=sheet_id,
        read_csv_options={"try_parse_dates": True},
        **kwargs,
    )


# TODO: Make generic
def read_edf(
    file_path: str,
    start: int = 0,
    stop: int | None = None,
) -> tuple[pl.LazyFrame, datetime, float]:
    """
    Reads data from an EDF file into a polars DataFrame.

    Args:
        file_path (str): The path to the EDF file.
        start (int, optional): The start index of the data to read. Defaults to 0.
        stop (int, optional): The end index of the data to read. Defaults to None.

    Returns:
        pl.DataFrame: A polars DataFrame containing the data from the EDF file.
    """
    raw_edf = mne.io.read_raw_edf(file_path)
    channel_names = cast(list[str], raw_edf.info.ch_names)
    date_measured = cast(datetime, raw_edf.info["meas_date"])
    sampling_rate = cast(float, raw_edf.info["sfreq"])

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
    data, times = raw_edf.get_data(start=start, stop=stop, return_times=True)
    lf = (
        pl.from_numpy(
            data.transpose(), schema={name: pl.Float64 for name in column_names}
        )
        .lazy()
        .with_row_count("index", offset=start)
        .with_columns(
            pl.Series("time_s", times, dtype=pl.Float64),
            pl.col("temperature").round(1),
        )
        .filter(
            (pl.col("temperature") != 0)
            & (pl.col("hbr") != 0)
            & (pl.col("ventilation") != 0)
        )
        .select("index", "time_s", *column_names)
    )
    return lf, date_measured, sampling_rate


def create_hdf_attrs(group: h5py.Group, data: dict[str, Any]) -> None:
    """
    Creates HDF5 attributes from a dictionary.

    Parameters
    ----------
    group : h5py.Group
        The HDF5 group to create the attributes in.
    data : dict[str, Any]
        A dictionary containing the attributes to create.
    """
    for key, value in data.items():
        if value is None:
            value = ()
            group.attrs.create(key, value)
        elif isinstance(value, datetime):
            value = value.isoformat()
            group.attrs.create(key, value)
        elif isinstance(value, dict):
            subgroup = group.create_group(key)
            create_hdf_attrs(subgroup, value)
        else:
            group.attrs.create(key, value)


def write_hdf5(file_path: str | Path, result: "Result") -> None:
    file_path = Path(file_path)
    try:
        with h5py.File(file_path, "a") as f:
            group_result = f.create_group("result")
            result_dict = asdict(result)
            group_names = [
                "identifier",
                "selection_parameters",
                "processing_parameters",
                "summary_statistics",
                "manual_peak_edits"
            ]
            for name, data in result_dict.items():
                if name in group_names or sys.getsizeof(data) < 64_000:
                    grp = group_result.create_group(name)
                    data = asdict(data) if name in group_names else data
                    if isinstance(data, dict):
                        create_hdf_attrs(grp, data)
                    elif name == "manual_peak_edits":
                        for key, value in data.items():
                            grp.create_dataset(key, data=value)
                else:
                    group_result.create_dataset(name, data=asdict(data) if name in ["focused_result", "source_data"] else data)
    except Exception as e:
        logger.error(f"Error writing to HDF5 file: {e}")
