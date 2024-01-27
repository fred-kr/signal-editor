import datetime
import typing as t
import pprint as pp

import attrs
import numpy as np
import numpy.typing as npt
import polars as pl

from .. import type_aliases as _t

if t.TYPE_CHECKING:
    from .section import SectionID, SectionResult

p = pp.PrettyPrinter(indent=4, width=120, compact=True, underscore_numbers=True)

@attrs.define(slots=True, frozen=True)
class ResultIdentifier:
    signal_name: str = attrs.field()
    source_file_name: str = attrs.field()
    date_recorded: datetime.date = attrs.field()
    animal_id: str = attrs.field()
    oxygen_condition: _t.OxygenCondition = attrs.field()

    def __str__(self) -> str:
        return f"ResultIdentifier({self.signal_name}, {self.source_file_name})"
    
    def __repr__(self) -> str:
        return f"""
ResultIdentifier(
    signal_name={self.signal_name},
    source_file_name={self.source_file_name},
    date_recorded={self.date_recorded},
    animal_id={self.animal_id},
    oxygen_condition={self.oxygen_condition},
)
"""
        
    def as_dict(self) -> _t.ResultIdentifierDict:
        return {
            "signal_name": self.signal_name,
            "animal_id": self.animal_id,
            "oxygen_condition": self.oxygen_condition,
            "source_file_name": self.source_file_name,
            "date_recorded": self.date_recorded,
        }


@attrs.define(slots=True)
class ManualPeakEdits:
    added: list[int] = attrs.field(factory=list)
    removed: list[int] = attrs.field(factory=list)

    def new_added(self, index: int) -> None:
        self.added.append(index)

    def new_removed(self, index: int) -> None:
        self.removed.append(index)

    def sort_and_deduplicate(self) -> None:
        self.added = sorted(set(self.added))
        self.removed = sorted(set(self.removed))

    def as_dict(self) -> _t.ManualPeakEditsDict:
        return {
            "added": self.added,
            "removed": self.removed,
        }


def _to_uint_array(array: npt.NDArray[np.int_]) -> npt.NDArray[np.uint32]:
    return array.astype(np.uint32)


@attrs.define(slots=True, frozen=True)
class FocusedResult:
    time_s: npt.NDArray[np.float64] = attrs.field()
    index: npt.NDArray[np.uint32] = attrs.field()
    peak_intervals: npt.NDArray[np.uint32] = attrs.field(converter=_to_uint_array)
    temperature: npt.NDArray[np.float64] = attrs.field()
    rate_bpm: npt.NDArray[np.float64] = attrs.field()

    def __repr__(self) -> str:
        return f"""
FocusedResult(
    time_s={self.time_s},
    index={self.index},
    peak_intervals={self.peak_intervals},
    temperature={self.temperature},
    rate_bpm={self.rate_bpm},
)
""" 
    
    def to_polars(self) -> pl.DataFrame:
        data = {attr: getattr(self, attr) for attr in self.__slots__ if attr != "__weakref__"}
        return pl.DataFrame(data)

    def to_structured_array(self) -> npt.NDArray[np.void]:
        dt = np.dtype(
            [(attr, getattr(self, attr).dtype) for attr in self.__slots__ if attr != "__weakref__"]
        )
        values = [getattr(self, attr) for attr in self.__slots__ if attr != "__weakref__"]
        return np.array(list(zip(*values, strict=True)), dtype=dt)


@attrs.define(slots=True, frozen=True)
class CompleteResult:
    identifier: ResultIdentifier = attrs.field()
    base_df_with_changes: pl.DataFrame = attrs.field()
    complete_section_results: dict["SectionID", "SectionResult"] = attrs.field()
    focused_section_results: dict["SectionID", FocusedResult] = attrs.field()
    peak_interval_stats: dict["SectionID", dict[str, float]] = attrs.field()
    rate_stats: dict["SectionID", dict[str, float]] = attrs.field()

    def __str__(self) -> str:
        return f"CompleteResult({self.identifier})"

    def __repr__(self) -> str:
        return f"""
CompleteResult(
    identifier={self.identifier},
    base_df_with_changes={self.base_df_with_changes},
    complete_section_results={self.complete_section_results},
    focused_section_results={self.focused_section_results},
    peak_interval_stats={self.peak_interval_stats},
    rate_stats={self.rate_stats},
)
"""

    def as_dict(self) -> _t.CompleteResultDict:
        return {
            "identifier": self.identifier.as_dict(),
            "base_df_with_changes": self.base_df_with_changes.to_numpy(structured=True),
            "complete_section_results": {
                s_id: s_res.as_dict() for s_id, s_res in self.complete_section_results.items()
            },
            "focused_section_results": {
                s_id: s_res.to_structured_array()
                for s_id, s_res in self.focused_section_results.items()
            },
            "peak_interval_stats": self.peak_interval_stats,
            "rate_stats": self.rate_stats,
        }
