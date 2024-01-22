import typing as t
from dataclasses import dataclass, field
from pathlib import Path
from datetime import date, datetime
import re
from PySide6 import QtWidgets

import numpy as np
import polars as pl
from loguru import logger
from numpy.typing import NDArray
from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtWidgets import QInputDialog, QWidget

from .. import type_aliases as _t
from ..models.io import read_edf
from ..models.result import (
    DescriptiveStatistics,
    FocusedResult,
    SummaryStatistics,
)
from ..models.signal import SignalData
from ..models.section import Section, SectionID, SectionContainer

if t.TYPE_CHECKING:
    from ..app import SignalEditor

def parse_file_name(
    file_name: str,
    date_pattern: str = r"\d{8}",
    id_pattern: str = r"(?:P[AM]|F)\d{1,2}",
) -> tuple[date, str, _t.OxygenCondition]:
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


def infer_sampling_rate(
    df: pl.DataFrame,
    time_col: str = "auto",
    time_unit: t.Literal["auto", "s", "ms"] = "auto",
) -> int:
    if time_col == "auto":
        # Try to infer the column holding the time data
        for col in df.columns:
            if "time" in col:
                time_col = col
                break

    if time_unit == "auto":
        if df.get_column(time_col).dtype.is_float():
            time_unit = "s"
        elif df.get_column(time_col).dtype.is_integer():
            time_unit = "ms"
        else:
            raise ValueError("Could not infer time unit")

    target = 1000 if time_unit == "ms" else 1.0
    closed = "left" if df.get_column(time_col)[0] == 0 else "both"
    lower = df.get_column(time_col)[0]
    return df.filter(pl.col(time_col).is_between(lower, target, closed=closed)).height


def get_array_stats(
    array: NDArray[np.intp | np.uintp | np.float_], description: str
) -> DescriptiveStatistics:
    return DescriptiveStatistics(
        description,
        mean=np.mean(array),
        median=np.median(array),
        std=np.std(array),
        var=np.var(array),
    )


@dataclass(slots=True)
class DataState:
    df: pl.DataFrame
    sig_data: SignalData = field(init=False)
    focused_results: dict[str, FocusedResult] = field(init=False)


class DataHandler(QObject):
    def __init__(self, app: "SignalEditor") -> None:
        super().__init__()
        self._app = app
        self.df: pl.DataFrame = pl.DataFrame()
        self._sig: SignalData | None = None
        self.focused_results: dict[str, FocusedResult] = {}
        self._sampling_rate: int = -1
        self.minmax_map = {}

    @property
    def fs(self) -> int:
        return self._sampling_rate

    @fs.setter
    def fs(self, value: int | float) -> None:
        value = int(value)
        self._sampling_rate = value
        self._app.spin_box_sample_rate.blockSignals(True)
        self._app.spin_box_sample_rate.setValue(value)
        self._app.spin_box_sample_rate.blockSignals(False)

    @Slot(int)
    def update_fs(self, value: int) -> None:
        self.fs = value
        if self._sig is not None:
            self._sig.sampling_rate = value

    @property
    def sig_data(self) -> SignalData:
        if self._sig is None:
            raise RuntimeError("No signal data loaded")
        return self._sig

    @sig_data.setter
    def sig_data(self, value: SignalData) -> None:
        self._sig = value

    @property
    def active_section(self) -> Section:
        """Easy access to the currently active section."""
        return self.sig_data.sections.get_active_section()

    def read(self, path: str | Path) -> None:
        path = Path(path)
        suffix = path.suffix
        if suffix not in {".csv", ".txt", ".edf", ".feather", ".xlsx", ".pkl"}:
            info_msg = (
                "Currently only .csv, .txt, .xlsx, .feather, .pkl and .edf files are supported"
            )
            self._app.sig_show_message.emit(info_msg, "info")
            return

        if suffix == ".pkl":
            self._app.restore_state(path)
            return
        elif suffix == ".csv":
            df = pl.read_csv(path)
        elif suffix == ".edf":
            lf, self.meas_date, self.fs = read_edf(path.as_posix())
            df = lf.collect()
        elif suffix == ".feather":
            df = pl.read_ipc(path, use_pyarrow=True)
        elif suffix == ".txt":
            df = pl.read_csv(path, separator="\t")
        elif suffix == ".xlsx":
            df = pl.read_excel(path)
        else:
            raise NotImplementedError(f"File type `{suffix}` not supported")

        if suffix != ".edf":
            fs = infer_sampling_rate(df)

            if fs == -1:
                fs, ok = QInputDialog.getInt(
                    self._app,
                    "Sampling rate",
                    "Enter sampling rate (samples per second): ",
                    200,
                    1,
                    10000,
                    1,
                )
                if ok:
                    self.fs = fs
                else:
                    self._sampling_rate_unavailable()
                    return

        self.df = df

    def _sampling_rate_unavailable(self) -> None:
        self.df = pl.DataFrame()
        self.fs = -1
        self.minmax_map = {}
        self.meas_date = None
        msg = "Sampling rate not set, no data loaded."
        self._app.sig_show_message.emit(msg, "warning")
        return

    def new_sig_data(self, name: str) -> None:
        if name not in self.df.columns:
            logger.debug(f"Column '{name}' doesn't exist in the current data frame")
            self._app.sig_show_message.emit(f"Column '{name}' doesn't exist", "warning")
            return
        self._sig = SignalData(name=name, data=self.df, sampling_rate=self.fs)

    def run_preprocessing(
        self,
        pipeline: _t.Pipeline,
        filter_params: _t.SignalFilterParameters,
        standardize_params: _t.StandardizeParameters,
    ) -> None:
        self.active_section.filter_data(pipeline=pipeline, **filter_params)

        standardize = self._app.scale_method
        if standardize == "None":
            return
        self.active_section.scale_data(**standardize_params)

    def run_peak_detection(self, peak_parameters: _t.PeakDetectionParameters) -> None:
        self.active_section.detect_peaks(**peak_parameters)
        self.run_rate_calculation()

    def run_rate_calculation(self) -> None:
        self.active_section.calculate_rate()

    def get_descriptive_stats(self) -> SummaryStatistics:
        intervals = np.ediff1d(self.active_section.peaks, to_begin=[0])
        rate = self.active_section.rate

        interval_stats = get_array_stats(intervals, "peak_intervals")
        rate_stats = get_array_stats(rate, "rate")

        return SummaryStatistics(
            peak_intervals=interval_stats,
            signal_rate=rate_stats,
        )

    def compute_result_df(self, name: str, section_id: SectionID | None = None) -> None:
        sig = self.sig_data
        # section_dict = sig.as_dict()
        # TODO: refactor result creation to work with new section structure
        if section_id is None:
            section_id = sig.active_section.section_id
        sec = sig.get_section(section_id)
        peaks = sec.peaks
        diffs = sec.get_peak_intervals()
        rate = sig.default_rate
        indices = sec.data.get_column("index").gather(peaks)
        time_s = (indices * (1 / self.fs)).round(4)
        temperature = sec.data.get_column("temperature").gather(peaks).round(1)
        focused_result = make_focused_result(
            time_s=time_s,
            index=indices,
            peak_intervals=diffs,
            temperature=temperature,
            rate_bpm=rate,
        )
        self.focused_results[name] = focused_result

    def get_focused_result_df(self, name: str, compute: bool = True) -> pl.DataFrame:
        if compute or name not in self.focused_results:
            self.compute_result_df(name)
        return self.focused_results[name].to_polars()

    def get_state(self) -> DataState:
        state = DataState(df=self.df.clone())
        state.sig_data = self.sig_data
        state.focused_results = self.focused_results
        return state

    def restore_state(self, state: DataState) -> None:
        self.restore_df(state.df)
        self.sig_data = state.sig_data
        self.focused_results = state.focused_results

    def restore_df(self, df: pl.DataFrame) -> None:
        self.df = df
        if all(
            col in df.columns for col in ["index", "time_s", "temperature", "hbr", "ventilation"]
        ):
            self.df = self.df.select(pl.col("index", "time_s", "temperature", "hbr", "ventilation"))
        else:
            self.df = df

    @Slot(int, int)
    def exclude_region(self, lower: int, upper: int) -> None:
        self.sig_data.add_section(lower, upper, set_active=False, include=False)


def _blocked_update(widget: QWidget, value: int) -> None:
    if isinstance(widget, QtWidgets.QSpinBox):
        widget.blockSignals(True)
        widget.setValue(value)
        widget.blockSignals(False)

class DataHandler2(QObject):
    sig_new_raw = Signal()
    sig_sfreq_changed = Signal(int)
    
    def __init__(self, app: "SignalEditor", parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._app = app

        self._raw_df: pl.DataFrame | None = None
        self._base_df: pl.DataFrame | None = None
        self._sections: SectionContainer | None = None
        self._sampling_rate: int = -1
        self._metadata: _t.FileMetadata | None = None
        self._connect_qt_signals()

    def _connect_qt_signals(self) -> None:
        self._app.spin_box_sample_rate.valueChanged.connect(self.update_sfreq)
        

    @property
    def raw_df(self) -> pl.DataFrame:
        """The raw data as loaded from the file."""
        if self._raw_df is None:
            raise RuntimeError("No signal data loaded")
        return self._raw_df

    @property
    def base_df(self) -> pl.DataFrame:
        """
        A modified version of the raw data from which new sections are created.

        Schema:
        
        - index: pl.UInt32
        - temperature: pl.Float64
        - {name_of_data_column}: pl.Float64
        - {name_of_data_column}_processed: pl.Float64
        - is_peak: pl.Boolean
        """
        if self._base_df is None:
            raise RuntimeError("No signal data loaded")
        return self._base_df
    
    @property
    def sfreq(self) -> int:
        return self._sampling_rate

    def set_sfreq(self, value: int) -> None:
        self._sampling_rate = value
        _blocked_update(self._app.spin_box_sample_rate, value)

    def update_sfreq(self, value: int) -> None:
        for section in self.sections.get_included():
            section.update_sfreq(value)

    @property
    def sections(self) -> SectionContainer:
        if self._sections is None:
            raise RuntimeError("No signal data loaded")
        return self._sections

    @property
    def cas(self) -> Section:
        """Shorthand for the currently active section (cas = current active section)."""
        return self.sections.get_active_section()

    @property
    def cas_proc_data(self) -> pl.Series:
        """Shorthand for the currently active section's processed signal."""
        return self.sections.get_active_section().proc_data

    @property
    def metadata(self) -> _t.FileMetadata:
        if self._metadata is None:
            raise RuntimeError("No signal data loaded")
        return self._metadata
    
    def read_file(self, path: Path | str) -> None:
        path = Path(path).resolve()
        suffix = path.suffix
        if suffix not in {".csv", ".txt", ".edf", ".feather", ".xlsx", ".pkl"}:
            info_msg = (
                "Supported file types are .csv, .txt, .xlsx, .feather, .pkl and .edf"
            )
            self._app.sig_show_message.emit(info_msg, "info")
            return

        sfreq = -1
        meas_date = None
        animal_id = None
        oxy_condition = None
        match suffix:
            case ".pkl":
                self._app.restore_state(path)
                return
            case ".csv":
                df = pl.read_csv(path)
            case ".edf":
                lf, meas_date, sfreq = read_edf(path.as_posix())
                df = lf.collect()
            case ".feather":
                df = pl.read_ipc(path, use_pyarrow=True)
            case ".txt":
                df = pl.read_csv(path, separator="\t")
            case ".xlsx":
                df = pl.read_excel(path)
            case _:
                raise NotImplementedError(f"File type `{suffix}` not supported")

        if sfreq == -1:
            try:
                sfreq = infer_sampling_rate(df)
            except ValueError:
                sfreq, ok = QInputDialog.getInt(
                    self._app,
                    "Sampling rate",
                    "Enter sampling rate (samples per second): ",
                    200,
                    1,
                    10000,
                    1,
                )
                if not ok:
                    self._reset()
                    return

        if meas_date is None:
            meas_date, animal_id, oxy_condition = parse_file_name(path.name)
        else:
            _, animal_id, oxy_condition = parse_file_name(path.name)
            
            
        self._metadata = _t.FileMetadata(
            date_recorded=meas_date,
            animal_id=animal_id,
            oxygen_condition=oxy_condition,
        )
            
        self._raw_df = df
        self.sig_new_raw.emit()

    def _reset(self) -> None:
        self._raw_df = None
        self._sections = None
        self.set_sfreq(-1)
        self.meas_date = None
        msg = "Sampling rate not set, no data loaded."
        self._app.sig_show_message.emit(msg, "warning")
        return

        
