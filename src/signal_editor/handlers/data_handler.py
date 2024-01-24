import datetime
import re
import typing as t
from datetime import date
from pathlib import Path

import attrs
import numpy as np
import numpy.typing as npt
import polars as pl
import polars.selectors as ps
from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Signal, Slot

from .. import type_aliases as _t
from ..models.io import read_edf
from ..models.result import DescriptiveStatistics, FocusedResult
from ..models.section import Section, SectionContainer, SectionID, SectionIndices, SectionResult

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
        date_ddmmyyyy = date.today()
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
    array: npt.NDArray[np.intp | np.uintp | np.float_], description: str
) -> DescriptiveStatistics:
    return DescriptiveStatistics(
        description,
        mean=np.mean(array),
        median=np.median(array),
        std=np.std(array),
        var=np.var(array),
    )


@attrs.define
class DataState:
    raw_df: pl.DataFrame | None = attrs.field(default=None)
    base_df: pl.DataFrame | None = attrs.field(default=None)
    sections: SectionContainer = attrs.field(default=SectionContainer)
    sampling_rate: int = attrs.field(default=-1)
    metadata: _t.FileMetadata | None = attrs.field(default=None)

    def as_dict(self) -> dict[str, t.Any]:
        return {
            "raw_df": self.raw_df,
            "base_df": self.base_df,
            "sections": self.sections,
            "sampling_rate": self.sampling_rate,
            "metadata": self.metadata,
        }


class DataHandler(QtCore.QObject):
    sig_new_raw = Signal()
    sig_sfreq_changed = Signal(int)
    sig_cas_changed = Signal()

    def __init__(self, app: "SignalEditor", parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._app = app

        self._raw_df: pl.DataFrame | None = None
        self._base_df: pl.DataFrame | None = None
        self._sections: SectionContainer = SectionContainer()
        self._sampling_rate: int = -1
        self._metadata: _t.FileMetadata | None = None
        self._sig_name: str | None = None
        self._connect_qt_signals()

    def _connect_qt_signals(self) -> None:
        self.sig_sfreq_changed.connect(self._app.spin_box_sample_rate.setValue)

    def _clear(self) -> None:
        self._raw_df = None
        self._base_df = None
        self._sections = SectionContainer()
        self.set_sfreq(-1)
        self._metadata = None
        msg = "Sampling rate not set, no data loaded."
        self._app.sig_show_message.emit(msg, "warning")
        return

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
    def bounds(self) -> SectionIndices:
        return SectionIndices(0, self.base_df.height - 1)

    @property
    def sfreq(self) -> int:
        return self._sampling_rate

    def set_sfreq(self, value: int) -> None:
        self._sampling_rate = value
        self.sig_sfreq_changed.emit(value)

    def update_sfreq(self, value: int) -> None:
        for section in self.sections.values():
            section.update_sfreq(value)

    @property
    def sections(self) -> SectionContainer:
        if not self._sections:
            raise RuntimeError("No signal data loaded")
        return self._sections

    @property
    def base_section(self) -> Section:
        try:
            return Section(
                self.base_df, sig_name=self.sig_name, sampling_rate=self.sfreq, _is_base=True
            )
        except RuntimeError:
            self._app.sig_show_message.emit("No data to create base section from", "warning")
            raise

    @property
    def cas(self) -> Section:
        """Shorthand for the currently active section (cas = current active section)."""
        section = next((section for section in self.sections.values() if section.is_active), None)
        if section is None:
            section = self.base_section
        return section

    @property
    def cas_proc_data(self) -> pl.Series:
        """Shorthand for the currently active section's processed signal column as a `polars.Series`"""
        return self.cas.proc_data

    @property
    def metadata(self) -> _t.FileMetadata:
        if self._metadata is None:
            raise RuntimeError("No signal data loaded")
        return self._metadata

    @property
    def sig_name(self) -> str:
        """The name of the column holding the raw signal data."""
        if self._sig_name is None:
            raise RuntimeError("No signal data loaded")
        return self._sig_name

    @property
    def proc_sig_name(self) -> str:
        """The name of the column holding the processed signal data."""
        if self._sig_name is None:
            raise RuntimeError("No signal data loaded")
        return f"{self._sig_name}_processed"

    @Slot(QtCore.QDate)
    def set_date(self, date: QtCore.QDate) -> None:
        py_date = t.cast(datetime.date, date.toPython())
        if self._metadata is None:
            self._metadata = _t.FileMetadata(
                date_recorded=py_date,
                animal_id="unknown",
                oxygen_condition="unknown",
            )
        else:
            self._metadata["date_recorded"] = py_date

    @Slot(str)
    def set_animal_id(self, animal_id: str) -> None:
        if self._metadata is None:
            self._metadata = _t.FileMetadata(
                date_recorded=date.today(), animal_id=animal_id, oxygen_condition="unknown"
            )
        else:
            self._metadata["animal_id"] = animal_id

    @Slot(str)
    def set_oxy_condition(self, oxy_condition: _t.OxygenCondition) -> None:
        if self._metadata is None:
            self._metadata = _t.FileMetadata(
                date_recorded=date.today(), animal_id="unknown", oxygen_condition=oxy_condition
            )
        else:
            self._metadata["oxygen_condition"] = oxy_condition

    def read_file(self, path: Path | str) -> None:
        path = Path(path).resolve()
        if not path.exists():
            info_msg = f"No file named '{path.name}' found in '{path.parent}'"
            self._app.sig_show_message.emit(info_msg, "error")
            return
        suffix = path.suffix
        if suffix not in {".csv", ".txt", ".edf", ".feather", ".xlsx", ".pkl"}:
            info_msg = "Supported file types are .csv, .txt, .xlsx, .feather, .pkl and .edf"
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
                sfreq, ok = QtWidgets.QInputDialog.getInt(
                    self._app,
                    "Sampling rate",
                    "Enter sampling rate (samples per second): ",
                    200,
                    1,
                    10000,
                    1,
                )
                if not ok:
                    self._clear()
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

    def create_base_df(self, sig_name: str) -> None:
        if self._raw_df is None:
            return

        self._sig_name = sig_name
        self._base_df = (
            self._raw_df.lazy()
            .select(
                ps.contains("temp"),
                pl.col(sig_name),
                pl.col(sig_name).alias(self.proc_sig_name),
                pl.repeat(False, self._raw_df.height, dtype=pl.Boolean).alias("is_peak"),
            )
            .with_row_index()
            .collect()
            .rechunk()
        )
        self._app.sig_show_message.emit(
            f"Created base df with columns: {self._base_df.columns}", "debug"
        )

    @Slot()
    def reset(self) -> None:
        self._base_df = self.create_base_df(self.sig_name)
        self.sections.clear()
        Section.reset_id_counter()

    def update_base(self, section_df: pl.DataFrame) -> None:
        self._base_df = (
            self.base_df.lazy().update(section_df.lazy(), on="index", how="left").collect()
        )

    @Slot(int, int)
    def remove_slice(self, start: int, stop: int) -> None:
        self._base_df = (
            self.base_df.lazy().filter(~(pl.col("index").is_between(start, stop))).collect()
        )

    def get_section_results(self) -> dict[SectionID, SectionResult]:
        return {
            section.section_id: section.get_complete_result() for section in self.sections.values()
        }

    def get_focused_results(self) -> dict[SectionID, FocusedResult]:
        return {
            section.section_id: section.get_focused_result() for section in self.sections.values()
        }

    def new_section(self, start: int, stop: int) -> Section:
        data = self.base_df.filter(pl.col("index").is_between(start, stop))
        section = Section(data, sig_name=self.sig_name, sampling_rate=self.sfreq)
        self._sections[section.section_id] = section
        return section

    def remove_section(self, section_id: SectionID) -> None:
        del self._sections[section_id]

    def get_section(self, section_id: SectionID) -> Section:
        return self.sections[section_id]

    def save_sections_to_base(self) -> None:
        """
        Updates the base data frame with the data from every current section. Later sections overwrite earlier ones if they overlap.
        """
        section_dfs = [section.data for section in self.sections.values()]
        for section_df in section_dfs:
            self.update_base(section_df)

    def get_state(self) -> DataState:
        return DataState(
            raw_df=self.raw_df,
            base_df=self.base_df,
            sections=self.sections,
            sampling_rate=self.sfreq,
            metadata=self.metadata,
        )

    def restore_state(self, state: DataState) -> None:
        self._raw_df = state.raw_df
        self._base_df = state.base_df
        self._sections = state.sections
        self.set_sfreq(state.sampling_rate)
        self._metadata = state.metadata

    @Slot(str)
    def set_cas(self, section_id: SectionID) -> None:
        for section in self.sections.values():
            section.set_active(section.section_id == section_id)
        self.sig_cas_changed.emit()
