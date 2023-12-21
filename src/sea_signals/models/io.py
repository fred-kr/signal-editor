import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, cast

import mne.io
import polars as pl
from loguru import logger


def parse_file_name(
    file_name: str, date_pattern: str = r"\d{8}", id_pattern: str = r"P[AM]\d{1,2}"
) -> tuple[date, str]:
    date_match = re.search(date_pattern, file_name)
    id_match = re.search(id_pattern, file_name)

    if not date_match or not id_match:
        return date.today(), ""
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


# def save_to_hdf5(result: Result, file_path: str | Path) -> None:
#     with h5py.File(Path(file_path).resolve().as_posix(), "w") as hdf:

#         for attr, value in asdict(result).items():
#             if isinstance(value, (str, int, float, datetime)):
#                 hdf.attrs[attr] = value
#             elif isinstance(value, list):
#                 hdf.create_dataset(attr, data=np.array(value), compression="gzip")
#             elif isinstance(value, dict):
#                 grp = hdf.create_group(attr)
#                 for k, v in value.items():
#                     grp.create_dataset(k, data=np.array(v), compression="gzip")
#             elif isinstance(value, np.ndarray):
#                 hdf.create_dataset(attr, data=value, compression="gzip")
#             # elif isinstance(value, pl.DataFrame):
#                 # np_df = value.to_numpy(structured=True)
#                 # hdf.create_dataset(attr, data=np_df, compression="gzip")
#             else:
#                 raise ValueError(f"Unsupported type: {type(value)}")

#         result_df = result.result_data
#         df_grp = hdf.create_group("result_data")
#         for column in result_df.columns:
#             df_grp.create_dataset(column, data=result_df[column].to_numpy(), compression="gzip")
