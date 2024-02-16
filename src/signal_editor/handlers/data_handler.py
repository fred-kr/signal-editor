import datetime
import re
import typing as t
from pathlib import Path

import attrs
import dateutil.parser as dt_parser
import polars as pl
import polars.selectors as ps
from PySide6 import QtCore, QtWidgets

from .. import type_aliases as _t
from ..fileio import read_edf
from ..models import (
    CompleteResult,
    FocusedResult,
    ResultIdentifier,
    Section,
    SectionContainer,
    SectionID,
    SectionIndices,
    SectionResult,
)
from ..processing import rolling_rate

if t.TYPE_CHECKING:
    from ..signal_editor import SignalEditor


def parse_file_name(
    file_name: str,
    date_pattern: str = r"\d{8}",
    id_pattern: str = r"(?:P[AM]|F)\d{1,2}",
) -> tuple[datetime.date, str, _t.OxygenCondition]:
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
    tuple[datetime.date, str, _t.OxygenCondition]
        The date, id, and oxygen condition parsed from the file name, or 'unknown' if
        the respective pattern was not found.
    """
    date_match = re.search(date_pattern, file_name)
    id_match = re.search(id_pattern, file_name)
    oxy_condition: t.Literal["hypoxic", "normoxic", "unknown"] = "unknown"

    if "hyp" in file_name:
        oxy_condition = "hypoxic"
    elif "norm" in file_name:
        oxy_condition = "normoxic"

    if not date_match:
        try:
            parsed_date = dt_parser.parse(file_name, fuzzy=True, dayfirst=True).date()
        except (ValueError, dt_parser.ParserError):
            parsed_date = datetime.date(year=2000, month=1, day=1)
    else:
        parsed_date = datetime.date(
            year=int(date_match[0][4:8], base=10),
            month=int(date_match[0][2:4], base=10),
            day=int(date_match[0][:2], base=10),
        )
    id_str = str(id_match[0]) if id_match else "unknown"
    return parsed_date, id_str, oxy_condition


def infer_sampling_rate(
    df: pl.DataFrame,
    time_col: str = "auto",
    time_unit: t.Literal["auto", "s", "ms"] = "auto",
) -> int:
    if time_col == "auto":
        # Try to infer the column holding the time data
        for col in df.columns:
            if "time" in col or df.get_column(col).dtype.is_temporal():
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


@attrs.define
class DataState:
    raw_df: pl.DataFrame | None = attrs.field(default=None)
    base_df: pl.DataFrame | None = attrs.field(default=None)
    sections: SectionContainer = attrs.field(default=SectionContainer)
    sampling_rate: int = attrs.field(default=-1)
    metadata: _t.FileMetadata | None = attrs.field(default=None)

    def as_dict(self) -> _t.DataStateDict:
        return {
            "raw_df": self.raw_df,
            "base_df": self.base_df,
            "sections": self.sections,
            "sampling_rate": self.sampling_rate,
            "metadata": self.metadata,
        }


class DataHandler(QtCore.QObject):
    sig_new_raw = QtCore.Signal()
    sig_sfreq_changed = QtCore.Signal(int)
    sig_cas_changed = QtCore.Signal(bool)
    sig_section_added = QtCore.Signal(str)
    sig_section_removed = QtCore.Signal(str)

    def __init__(self, app: "SignalEditor", parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._app = app

        self._raw_df: pl.DataFrame | None = None
        self._base_df: pl.DataFrame | None = None
        self._base_section: Section | None = None
        self._sections: SectionContainer = SectionContainer()
        self._sampling_rate: int = -1
        self._metadata: _t.FileMetadata | None = None
        self._sig_name: str | None = None
        self._error_no_data_available_shown = False
        self._section_results: dict[SectionID, SectionResult] = {}
        self._focused_results: dict[SectionID, FocusedResult] = {}

    def _clear(self) -> None:
        self._raw_df = None
        self._base_df = None
        self._sections = SectionContainer()
        self.set_sfreq(-1)
        self._metadata = None
        self._sig_name = None
        Section.reset_id_counter()
        msg = "Sampling rate not set, no data loaded."
        self._app.sig_show_message.emit(msg, "warning")
        return

    @property
    def raw_df(self) -> pl.DataFrame | None:
        """The raw data as loaded from the file."""
        return self._raw_df

    @property
    def base_df(self) -> pl.DataFrame | None:
        """
        A modified version of the raw data from which new sections are created.

        Schema:

        - index: pl.UInt32
        - temperature: pl.Float64
        - {name_of_data_column}: pl.Float64
        - {name_of_data_column}_processed: pl.Float64
        - is_peak: pl.Int8
        """
        return self._base_df

    @property
    def bounds(self) -> SectionIndices | None:
        """The row indices of the first and last row of the base data frame."""
        if self.base_df is None:
            return None
        index = self.base_df.get_column("index")
        return SectionIndices(index[0], index[-1])

    @property
    def sfreq(self) -> int:
        return self._sampling_rate

    def set_sfreq(self, value: int) -> None:
        # Change sampling rate from inside the code, then update the corresponding UI widget (block signals to avoid infinite recursion)
        self._sampling_rate = value
        self.sig_sfreq_changed.emit(value)

    @QtCore.Slot(int)
    def update_sfreq(self, value: int) -> None:
        for section in self.sections.values():
            section.update_sfreq(value)

    @property
    def sections(self) -> SectionContainer:
        return self._sections

    @property
    def base_section(self) -> Section | None:
        if self.base_df is None or self.sig_name is None or self.sfreq == -1:
            return None
        if self._base_section is None:
            self._base_section = Section(
                self.base_df,
                sig_name=self.sig_name,
                sampling_rate=self.sfreq,
                set_active=True,
                _is_base=True,
            )
        return self._base_section

    @property
    def cas(self) -> Section | None:
        """Shorthand for the currently active section (cas = current active section)."""
        return next(
            (section for section in self.sections.values() if section.is_active),
            self.base_section,
        )

    @QtCore.Slot(str)
    def set_cas(self, section_id: SectionID) -> None:
        if section_id not in self.sections or self.cas is None:
            return
        for section in self.sections.values():
            section.set_active(section.section_id == section_id)
        has_peaks = not self.cas.peaks.is_empty()
        self.sig_cas_changed.emit(has_peaks)

    @property
    def removable_section_ids(self) -> list[SectionID]:
        return list(self.sections.keys())[1:]

    @property
    def metadata(self) -> _t.FileMetadata | None:
        return self._metadata

    @property
    def sig_name(self) -> str | None:
        """The name of the column holding the raw signal data."""
        return self._sig_name

    @QtCore.Slot(QtCore.QDate)
    def set_date(self, date: QtCore.QDate) -> None:
        py_date = t.cast(datetime.date, date.toPython())
        if py_date.year == 2000 and py_date.month == 1 and py_date.day == 1:
            py_date = None
        if self._metadata is None:
            self._metadata = _t.FileMetadata(
                date_recorded=py_date,
                animal_id="unknown",
                oxygen_condition="unknown",
            )
        else:
            self._metadata["date_recorded"] = py_date

    @QtCore.Slot(str)
    def set_animal_id(self, animal_id: str) -> None:
        if self._metadata is None:
            self._metadata = _t.FileMetadata(
                date_recorded=None, animal_id=animal_id, oxygen_condition="unknown"
            )
        else:
            self._metadata["animal_id"] = animal_id

    @QtCore.Slot(str)
    def set_oxy_condition(self, oxy_condition: _t.OxygenCondition) -> None:
        if self._metadata is None:
            self._metadata = _t.FileMetadata(
                date_recorded=None,
                animal_id="unknown",
                oxygen_condition=oxy_condition,
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
        if suffix not in {".csv", ".txt", ".edf", ".feather", ".xlsx"}:
            info_msg = "Supported file types are .csv, .txt, .xlsx, .feather and .edf"
            self._app.sig_show_message.emit(info_msg, "info")
            return

        sfreq = -1
        meas_date = None
        animal_id = None
        oxy_condition = None
        match suffix:
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
                raise NotImplementedError(f"File type '{suffix}' not supported")

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
        self.set_sfreq(int(sfreq))
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
                pl.col(sig_name).alias(f"{sig_name}_processed"),
                pl.lit(0).cast(pl.Int8).alias("is_peak"),
            )
            .with_row_index()
            .set_sorted("index")
            .collect()
        )

        base_section = self.base_section
        if base_section is None:
            return
        self._sections[base_section.section_id] = base_section
        self.sig_section_added.emit(base_section.section_id)

    @QtCore.Slot()
    def reset(self) -> None:
        self.clear_sections()
        self._raw_df = None
        self._base_df = None
        self._sections = SectionContainer()
        sfreq = self._app.config.sample_rate
        self.set_sfreq(sfreq)
        self._metadata = None
        self._sig_name = None
        Section.reset_id_counter()
        self._error_no_data_available_shown = False
        self._base_section = None
        self._metadata = None
        self._section_results.clear()
        self._focused_results.clear()

    @QtCore.Slot()
    def clear_sections(self) -> None:
        self._sections.clear()
        Section.reset_id_counter()
        self._base_df = None

    def update_base(self, section_df: pl.DataFrame) -> None:
        if self.base_df is None:
            return
        self._base_df = (
            self.base_df.lazy().update(section_df.lazy(), on="index", how="left").collect()
        )

    def new_section(self, start: int, stop: int) -> None:
        if self.base_df is None or self.sig_name is None or self.sfreq == -1:
            return
        data = self.base_df.filter(pl.col("index").is_between(start, stop))
        section = Section(data, sig_name=self.sig_name, sampling_rate=self.sfreq)
        self._sections[section.section_id] = section
        self.sig_section_added.emit(section.section_id)

    def remove_section(self, section_id: SectionID) -> None:
        if section_id not in self.sections:
            return
        del self._sections[section_id]
        self.sig_section_removed.emit(section_id)

    def save_all_sections(self) -> None:
        """
        Saves all sections.

        Updates the base data with each section's data.
        Stores the section result and focused result of each section.
        """
        for section in list(self.sections.values())[1:]:
            self.update_base(section.data)
            self._section_results[section.section_id] = section.get_section_result()
            self._focused_results[section.section_id] = section.get_focused_result()

    def save_cas(self) -> None:
        """
        Saves the current CAS data.

        Updates the base data with the CAS data.
        Stores the section result and focused result of the CAS.
        """
        if self.cas is None:
            return
        self.update_base(self.cas.data)
        self._section_results[self.cas.section_id] = self.cas.get_section_result()
        self._focused_results[self.cas.section_id] = self.cas.get_focused_result()

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

    def get_result_identifier(self) -> ResultIdentifier | None:
        if self.sig_name is None or self.metadata is None:
            return None
        return ResultIdentifier(
            signal_name=self.sig_name,
            source_file_name=self._app.file_info.fileName(),
            date_recorded=self.metadata["date_recorded"],
            animal_id=self.metadata["animal_id"],
            oxygen_condition=self.metadata["oxygen_condition"],
        )

    def get_section_results(self) -> dict[SectionID, "SectionResult"] | None:
        if len(self.sections) in {0, 1} or self.base_section is None:
            msg = "Can't create section results without sections / from only the base section."
            self._app.sig_show_message.emit(msg, "warning")
            return None
        return {
            section_id: section.get_section_result()
            for section_id, section in self.sections.items()
            if section_id != self.base_section.section_id
        }

    def get_focused_results(self) -> dict[SectionID, "FocusedResult"] | None:
        if len(self.sections) in {0, 1} or self.base_section is None:
            msg = "Can't create focused results without sections / from only the base section."
            self._app.sig_show_message.emit(msg, "warning")
            return None
        return {
            section_id: section.get_focused_result()
            for section_id, section in self.sections.items()
            if section_id != self.base_section.section_id
        }

    def get_complete_result(self) -> CompleteResult | None:
        self.save_all_sections()
        identifier = self.get_result_identifier()
        processed_dataframe = self.base_df
        complete_section_results = self.get_section_results()
        focused_section_results = self.get_focused_results()

        if (
            identifier is None
            or processed_dataframe is None
            or complete_section_results is None
            or focused_section_results is None
        ):
            msg = f"Can't create complete result without identifier / processed dataframe / complete section results / focused section results.\n\n{identifier = }\n{processed_dataframe = }\n{complete_section_results = }\n{focused_section_results = }"
            self._app.sig_show_message.emit(msg, "warning")
            return None
        return CompleteResult(
            identifier=identifier,
            processed_dataframe=processed_dataframe,
            complete_section_results=complete_section_results,
            focused_section_results=focused_section_results,
        )

    def get_bpm_per_temperature(
        self, grp_col: str, temperature_col: str, every: int, period: int, offset: int
    ) -> pl.DataFrame:
        section_results = self.get_focused_results()
        if section_results is None:
            raise RuntimeError("Can't get bpm per temperature without section results.")
        dfs = [result.to_polars() for result in section_results.values()]
        grouped_dfs = [
            rolling_rate(df, grp_col, temperature_col, every, period, offset) for df in dfs
        ]
        return pl.concat(grouped_dfs)
