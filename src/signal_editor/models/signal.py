import typing as t

import numpy as np
import numpy.typing as npt
import polars as pl
import polars.selectors as ps
from PySide6.QtCore import QObject, Signal, Slot

from .. import type_aliases as _t

if t.TYPE_CHECKING:
    from .result import Result


class SignalData(QObject):
    sig_new_section_id = Signal(str)

    def __init__(
        self,
        name: str,
        data: pl.DataFrame,
        sampling_rate: int,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.name: str = name
        self.data: pl.DataFrame = data
        self.sampling_rate: int = sampling_rate
        self._initial_state: _t.InitialState = {
            "name": self.name,
            "sampling_rate": self.sampling_rate,
            "data": self.data,
        }
        self.result_data: "dict[str, pl.DataFrame | Result]" = {}
        self._finish_init()

    def _finish_init(self) -> None:
        self.processed_name: str = f"{self.name}_processed"
        self.is_finished: bool = False
        self._peak_index_offset: int = 0
        self.sections: SectionContainer = SectionContainer(self.name)
        self._setup_data()

    def _setup_data(self) -> None:
        if "index" not in self.data.columns:
            self.data = self.data.with_row_index()
        self.data = self.data.select(
            pl.col("index"),
            ps.contains("temp"),
            pl.col(self.name),
            pl.col(self.name).alias(self.processed_name),
            pl.repeat(False, self.data.height, dtype=pl.Boolean).alias("is_peak"),
        )
        self._original_data = self.data.clone()

        self.add_section(0, self.data.height - 1, set_active=True, include=True)
        section = Section(
            self.data,
            name=self.name,
            set_active=True,
            include=True,
            sampling_rate=self.sampling_rate,
        )
        self._default_section = section
        self.sections.add_section(section)

    @property
    def active_section(self) -> Section:
        return self.sections.get_active_section()

    @property
    def active_peaks(self) -> npt.NDArray[np.uint32]:
        return self.active_section.peaks.to_numpy()

    @property
    def data_bounds(self) -> tuple[int, int]:
        col = self.data.get_column("index")
        return col[0], col[-1]

    @property
    def active_bounds(self) -> tuple[int, int]:
        """
        The start and stop row indices of the active section in the original data.
        """
        return self.active_section.abs_start, self.active_section.abs_stop

    def get_section(self, section_id: SectionID) -> Section:
        return self.sections.get_section(section_id)

    @Slot(str)
    def next_section(self, section_id: SectionID) -> None:
        new = self.sections.get_next_section(section_id)
        self.set_active_section(new.section_id)

    @Slot(str)
    def previous_section(self, section_id: SectionID) -> None:
        new = self.sections.get_previous_section(section_id)
        self.set_active_section(new.section_id)

    def set_to_default(self) -> None:
        self.set_active_section(self._default_section.section_id)

    def save_section_changes(self) -> None:
        for excluded_section in self.sections.get_excluded().values():
            self.data = self.data.filter(
                ~(pl.col("index").is_between(excluded_section.abs_start, excluded_section.abs_stop))
            )
        for section in self.sections.get_included().values():
            section_data = section.data.select(pl.col("index", self.processed_name, "is_peak"))
            self.data.update(section_data, on="index")

    def add_section(
        self, start: int, stop: int, set_active: bool = False, include: bool = True
    ) -> None:
        data = self.data.filter(pl.col("index").is_between(start, stop))
        new_section = Section(
            data,
            name=self.name,
            set_active=set_active,
            include=include,
            sampling_rate=self.sampling_rate,
        )
        self.sections.add_section(new_section)
        self.sig_new_section_id.emit(new_section.section_id)

    def get_all_peaks(self) -> dict[SectionID, npt.NDArray[np.uint32]]:
        return {
            section.section_id: section.peaks for section in self.sections.get_included().values()
        }

    def get_active_signal_data(self) -> npt.NDArray[np.float64]:
        return self.active_section.sig_vals

    def as_dict(self) -> list[dict[SectionID, _t.SectionResultDict]]:
        return [
            {
                section.section_id: section.as_dict()
                for section in self.sections.get_included().values()
            },
            {
                section.section_id: section.as_dict()
                for section in self.sections.get_excluded().values()
            },
        ]

    def set_active_section(self, section_id: SectionID) -> None:
        self.sections.set_active_section(section_id)

    def reset(self) -> None:
        self.name = self._initial_state["name"]
        self.sampling_rate = self._initial_state["sampling_rate"]
        self.data = self._initial_state["data"]
        Section.reset_id_counter()
        self._finish_init()

    def remove_all_excluded(self) -> None:
        excluded_sections = self.sections.get_excluded()
        for section in excluded_sections.values():
            self.data = self.data.filter(
                ~(pl.col("index").is_between(section.abs_start, section.abs_stop))
            )

    def restore_excluded(self) -> None:
        excluded_sections = self.sections.get_excluded()
        for section in excluded_sections.values():
            self.data.update(section.data, on="index", how="outer")

    # TODO: Actually implement this nicely
    def create_intermediate_result(self, section_ids: list[SectionID]) -> pl.DataFrame:
        sections = [self.sections.get_section(section_id) for section_id in section_ids]
        return pl.concat([section.data for section in sections])
