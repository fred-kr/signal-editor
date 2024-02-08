import datetime
import typing as t

import attrs
import numpy as np
import numpy.typing as npt
import polars as pl

from .. import type_aliases as _t

if t.TYPE_CHECKING:
    from .section import SectionID, SectionResult


def _format_long_sequence(seq: t.Sequence[int | float]) -> str:
    if len(seq) > 10:
        return f"[{', '.join(map(str, seq[:5]))}, ..., {', '.join(map(str, seq[-5:]))}]"
    else:
        return str(seq)


@attrs.define(slots=True, frozen=True)
class ResultIdentifier:
    signal_name: str = attrs.field()
    source_file_name: str = attrs.field()
    date_recorded: datetime.date | None = attrs.field()
    animal_id: str = attrs.field()
    oxygen_condition: _t.OxygenCondition = attrs.field()

    def as_dict(self) -> _t.ResultIdentifierDict:
        return {
            "signal_name": self.signal_name,
            "source_file_name": self.source_file_name,
            "date_recorded": self.date_recorded,
            "animal_id": self.animal_id,
            "oxygen_condition": self.oxygen_condition,
        }


@attrs.define(slots=True)
class ManualPeakEdits:
    added: list[int] = attrs.field(factory=list)
    removed: list[int] = attrs.field(factory=list)

    def __repr__(self) -> str:
        return f"ManualPeakEdits(added={_format_long_sequence(self.added)}, removed={_format_long_sequence(self.removed)})\nTotal added: {len(self.added)}\nTotal removed: {len(self.removed)}"

    def clear(self) -> None:
        self.added.clear()
        self.removed.clear()

    def new_added(self, value: int | t.Sequence[int] | pl.Series) -> None:
        if isinstance(value, int):
            if value in self.removed:
                self.removed.remove(value)
            else:
                self.added.append(value)
        else:
            for v in value:
                if v in self.removed:
                    self.removed.remove(v)
                else:
                    self.added.append(v)

    def new_removed(self, value: int | t.Sequence[int] | pl.Series) -> None:
        if isinstance(value, int):
            if value in self.added:
                self.added.remove(value)
            else:
                self.removed.append(value)
        else:
            for v in value:
                if v in self.added:
                    self.added.remove(v)
                else:
                    self.removed.append(v)

    def sort_and_deduplicate(self) -> None:
        self.added = sorted(set(self.added))
        self.removed = sorted(set(self.removed))

    def get_joined(self) -> list[int]:
        return sorted(set(self.added + self.removed))

    def as_dict(self) -> _t.ManualPeakEditsDict:
        self.sort_and_deduplicate()
        return {
            "added": self.added,
            "removed": self.removed,
        }


def _to_uint_array(array: npt.NDArray[np.int_]) -> npt.NDArray[np.uint32]:
    return array.astype(np.uint32)


@attrs.define(slots=True, frozen=True)
class FocusedResult:
    peaks_section_index: npt.NDArray[np.uint32] = attrs.field()
    peaks_global_index: npt.NDArray[np.uint32] = attrs.field()
    seconds_since_global_start: npt.NDArray[np.float64] = attrs.field()
    seconds_since_section_start: npt.NDArray[np.float64] = attrs.field()
    peak_intervals: npt.NDArray[np.uint32] = attrs.field(converter=_to_uint_array)
    temperature: npt.NDArray[np.float64] = attrs.field()
    rate_bpm: npt.NDArray[np.float64] = attrs.field()

    def to_polars(self) -> pl.DataFrame:
        data = attrs.asdict(self)
        return pl.DataFrame(data)

    def to_structured_array(self) -> npt.NDArray[np.void]:
        dt = np.dtype(
            [
                (field.name, getattr(self, field.name).dtype)
                for field in attrs.fields(self.__class__)
            ]
        )
        values = tuple(getattr(self, field.name) for field in attrs.fields(self.__class__))
        return np.array(list(zip(*values, strict=True)), dtype=dt)


@attrs.define(slots=True, frozen=True)
class CompleteResult:
    identifier: ResultIdentifier = attrs.field()
    processed_dataframe: pl.DataFrame = attrs.field()
    complete_section_results: dict["SectionID", "SectionResult"] = attrs.field()
    focused_section_results: dict["SectionID", FocusedResult] = attrs.field()

    def __str__(self) -> str:
        return f"CompleteResult({self.identifier})"

    def as_dict(self) -> _t.CompleteResultDict:
        return {
            "identifier": self.identifier.as_dict(),
            "global_dataframe": self.processed_dataframe.to_numpy(structured=True),
            "complete_section_results": {
                s_id: s_res.as_dict() for s_id, s_res in self.complete_section_results.items()
            },
            "focused_section_results": {
                s_id: s_res.to_structured_array()
                for s_id, s_res in self.focused_section_results.items()
            },
        }
