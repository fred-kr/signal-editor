import re
from datetime import date
from pathlib import Path
from typing import Iterable, Any

import mne.io
import numpy as np
import polars as pl
from loguru import logger


def parse_file_name(
    file_name: str, date_pattern: str = r"\d{8}", id_pattern: str = r"P[AM]\d{1,2}"
) -> tuple[date, str]:
    date_match = re.search(date_pattern, file_name)
    id_match = re.search(id_pattern, file_name)

    date_ddmmyyyy = date(
        year=int(date_match[0][4:8], base=10),
        month=int(date_match[0][2:4], base=10),
        day=int(date_match[0][:2], base=10),
    )
    id_str = str(id_match[0])

    return date_ddmmyyyy, id_str


def format_column_names(columns: Iterable[str]) -> list[str]:
    """
    Formats a list of column names.

    Args:
        columns (Iterable[str]): The list of column names.

    Returns:
        list[str]: The formatted column names.

    Raises:
        ValueError: If the column names list is empty.
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


# TODO: Refactor (single responsibility principle), allow reading just one of the signal columns
def read_edf(
    file_path: str,
) -> pl.LazyFrame:
    """
    Reads an EDF file and returns a `LazyFrame` object containing the data.

    Args:
        file_path (str): The path to the EDF file.

    Returns:
        pl.LazyFrame: A `LazyFrame` object containing the EDF data.

    Raises:
        FileNotFoundError: If the specified file does not exist.

    Examples:
        >>> read_edf("path/to/file.edf")
        <polars.LazyFrame>

    Note:
        This function assumes that the EDF file is in the European Data Format (EDF) and
        uses the `mne.io.read_raw_edf` function from the MNE library to read the file.

        The function reads the raw EDF data, extracts the data and time information,
        renames the columns based on a predefined mapping, creates an index column,
        defines the schema for the `LazyFrame`, and filters out rows where the temperature,
        hbr, and ventilation columns are all zero.

        The resulting `LazyFrame` object is returned.
    """
    raw_edf = mne.io.read_raw_edf(file_path)
    data, times = raw_edf.get_data(return_times=True)
    channel_names: list[str] = raw_edf.info.ch_names

    rename_map = {
        "temp": "temperature",
        "hb": "hbr",
        "vent": "ventilation",
    }
    column_names = [
        next(
            (rename_map[key] for key in rename_map if key in name.lower()),
            f"channel_{i+1}",
        )
        for i, name in enumerate(channel_names)
    ]

    index_column = np.arange(0, len(times), 1, dtype=np.uint32)

    schema = {
        "index": pl.UInt32,
        "time_s": pl.Float64,
        **{name: pl.Float32 for name in column_names},
    }

    return (
        pl.LazyFrame(
            data=np.column_stack((index_column, times, data.T)),
            schema=schema,
        )
        .with_columns(
            pl.col("time_s").cast(pl.Float64).round(4),
            pl.col("temperature").cast(pl.Float32).round(1),
            pl.col("hbr").cast(pl.Float32).round(4),
            pl.col("ventilation").cast(pl.Float32).round(4),
        )
        .filter(  # This gets rid of the ending sections where the sensor was already disconnected from power
            (pl.col("temperature") != 0)
            & (pl.col("hbr") != 0)
            & (pl.col("ventilation") != 0)
        )
    )
