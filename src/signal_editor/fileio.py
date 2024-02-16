import typing as t
from datetime import datetime
from pathlib import Path

import mne.io
import polars as pl
import tables as tb

from . import type_aliases as _t


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
        .with_row_index(offset=start)
        .with_columns(
            pl.Series("time_s", times, dtype=pl.Decimal),
            pl.col("temperature").round(1),
        )
        .filter((pl.col("temperature") != 0) & (pl.col("hbr") != 0) & (pl.col("ventilation") != 0))
        .select("index", "time_s", *column_names)
    )
    return lf, date_measured, sampling_rate


def unpack_dict_to_attrs(
    data: (
        _t.ResultIdentifierDict
        | _t.SignalFilterParameters
        | _t.StandardizeParameters
        | _t.PeakDetectionParameters
        | _t.SummaryDict
        | dict[str, str | object]
        | None
    ),
    file: tb.File,
    node: tb.Node | str,
) -> None:
    """
    Unpacks a dictionary of attributes and sets them as node attributes in a PyTables file.

    Parameters
    ----------
    data : _t.ResultIdentifierDict | _t.SignalFilterParameters | _t.StandardizeParameters | _t.PeakDetectionParameters | _t.SummaryDict | None
        A dictionary containing the attributes to be set as node attributes. Can be one of the following types:
        - _t.ResultIdentifierDict: A dictionary containing result identifier attributes.
        - _t.SignalFilterParameters: A dictionary containing signal filter parameters.
        - _t.StandardizeParameters: A dictionary containing standardize parameters.
        - _t.PeakDetectionParameters: A dictionary containing peak detection parameters.
        - _t.SummaryDict: A dictionary containing summary attributes.
        - None: If data is None, the function returns without performing any action.

    file : tb.File
        The PyTables file object.

    node : tb.Node | str
        The node in the PyTables file where the attributes will be set. Can be either a PyTables Node object or a string representing the path to the node.
    """
    if data is None:
        return
    if isinstance(data, str):
        file.set_node_attr(node, "", data)
    for key, value in data.items():
        if value is None:
            value = "unknown"
        file.set_node_attr(node, key, value)


def result_dict_to_hdf5(file_path: str | Path, data: _t.CompleteResultDict) -> None:
    file_path = Path(file_path).resolve().as_posix()

    with tb.open_file(file_path, "w", title=f"Results_{Path(file_path).stem}") as h5f:
        # Set the root groups attributes to key-value pairs of the `identifier` dictionary
        unpack_dict_to_attrs(data["identifier"], h5f, h5f.root)

        # Global DataFrame
        h5f.create_table(
            h5f.root,
            name="global_dataframe",
            description=data["global_dataframe"],
            title="Global DataFrame",
            expectedrows=data["global_dataframe"].shape[0],
        )

        # Focused Section Results
        h5f.create_group(h5f.root, "focused_section_results", title="Focused Section Results")
        for section_id, focused_result in data["focused_section_results"].items():
            h5f.create_table(
                "/focused_section_results",
                name=f"focused_result_{section_id}",
                description=focused_result,
                title=f"Focused Result ({section_id})",
                expectedrows=focused_result.shape[0],
            )

        # Complete Section Results
        h5f.create_group(h5f.root, "complete_section_results", title="Complete Section Results")
        for section_id, section_result in data["complete_section_results"].items():
            h5f.create_group(
                "/complete_section_results",
                name=f"complete_result_{section_id}",
                title=f"Complete Result ({section_id})",
            )

            # region Section DataFrame
            h5f.create_table(
                f"/complete_section_results/complete_result_{section_id}",
                name="section_dataframe",
                description=section_result["data"],
                title=f"DataFrame ({section_id})",
                expectedrows=section_result["data"].shape[0],
            )
            # endregion

            # region Peaks
            h5f.create_group(
                f"/complete_section_results/complete_result_{section_id}",
                name="peaks",
                title="Peaks",
            )
            h5f.create_array(
                f"/complete_section_results/complete_result_{section_id}/peaks",
                name="peak_indices_section",
                obj=section_result["peaks_section"],
                title="Peak indices (section)",
            )
            h5f.create_array(
                f"/complete_section_results/complete_result_{section_id}/peaks",
                name="peak_indices_global",
                obj=section_result["peaks_global"],
                title="Peak indices (global)",
            )
            h5f.create_array(
                f"/complete_section_results/complete_result_{section_id}/peaks",
                name="manually_added_peak_indices",
                obj=section_result["peak_edits"]["added"],
                title="Manually added (section)",
            )
            h5f.create_array(
                f"/complete_section_results/complete_result_{section_id}/peaks",
                name="manually_removed_peak_indices",
                obj=section_result["peak_edits"]["removed"],
                title="Manually removed (section)",
            )
            # endregion

            # region Rate
            h5f.create_group(
                f"/complete_section_results/complete_result_{section_id}",
                name="rate",
                title="Calculated rate",
            )
            h5f.create_array(
                f"/complete_section_results/complete_result_{section_id}/rate",
                name="not_interpolated",
                obj=section_result["rate"],
                title="Rate (no interpolation)",
            )
            h5f.create_array(
                f"/complete_section_results/complete_result_{section_id}/rate",
                name="interpolated",
                obj=section_result["rate_interpolated"],
                title="Rate (interpolated to length of section)",
            )
            # endregion

            # region Processing Parameters
            h5f.create_group(
                f"/complete_section_results/complete_result_{section_id}",
                name="processing_parameters",
                title=f"Processing parameters ({section_id})",
            )
            h5f.set_node_attr(
                f"/complete_section_results/complete_result_{section_id}/processing_parameters",
                attrname="sampling_rate",
                attrvalue=section_result["processing_parameters"]["sampling_rate"],
            )
            h5f.set_node_attr(
                f"/complete_section_results/complete_result_{section_id}/processing_parameters",
                attrname="pipeline",
                attrvalue=section_result["processing_parameters"]["pipeline"],
            )

            # Filter parameters
            h5f.create_group(
                f"/complete_section_results/complete_result_{section_id}/processing_parameters",
                name="filter_parameters",
                title="Filter parameters",
            )
            unpack_dict_to_attrs(
                section_result["processing_parameters"]["filter_parameters"],
                h5f,
                f"/complete_section_results/complete_result_{section_id}/processing_parameters/filter_parameters",
            )

            # Standardize parameters
            h5f.create_group(
                f"/complete_section_results/complete_result_{section_id}/processing_parameters",
                name="standardize_parameters",
                title="Standardize parameters",
            )
            unpack_dict_to_attrs(
                section_result["processing_parameters"]["standardize_parameters"],
                h5f,
                f"/complete_section_results/complete_result_{section_id}/processing_parameters/standardize_parameters",
            )

            # Peak detection parameters
            h5f.create_group(
                f"/complete_section_results/complete_result_{section_id}/processing_parameters",
                name="peak_detection_parameters",
                title="Peak detection parameters",
            )
            _peak_params = section_result["processing_parameters"]["peak_detection_parameters"]
            if _peak_params is not None:
                _method = _peak_params["method"]
                _method_params = _peak_params["method_parameters"]
                flattened_peak_detection_parameters = {"method": _method, **_method_params}
                unpack_dict_to_attrs(
                    flattened_peak_detection_parameters,
                    h5f,
                    f"/complete_section_results/complete_result_{section_id}/processing_parameters/peak_detection_parameters",
                )
            # endregion
