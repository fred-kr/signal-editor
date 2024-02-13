import datetime
import typing as t

import attrs
import numpy as np
import numpy.typing as npt
import polars as pl
import polars.selectors as ps

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
            "global_dataframe": self.processed_dataframe.lazy()
            .with_columns(ps.boolean().cast(pl.Int8))
            .collect()
            .to_numpy(structured=True),
            "complete_section_results": {
                s_id: s_res.as_dict() for s_id, s_res in self.complete_section_results.items()
            },
            "focused_section_results": {
                s_id: s_res.to_structured_array()
                for s_id, s_res in self.focused_section_results.items()
            },
        }

"""
{'complete_section_results': {
    'SEC_hbr_001': {
        'data': np.array(
                [
                    (0,       0, 0.000000e+00, 0.000000e+00, 15.5, 2.359375, 0.18033665, 0, 0),
                    (1,       1, 5.000000e-03, 5.000000e-03, 15.5, 2.397188, 0.70710678, 0, 0),
                    (2,       2, 1.000000e-02, 1.000000e-02, 15.5, 2.446875, 1.00067924, 0, 0),
                    ...,
                    (1632695, 1632695, 8.163475e+03, 8.163475e+03, 18.9, 1.4425  , 0.25499085, 0, 0),
                    (1632696, 1632696, 8.163480e+03, 8.163480e+03, 18.9, 1.516563, 0.25989655, 0, 0),
                    (1632697, 1632697, 8.163485e+03, 8.163485e+03, 18.9, 1.636875, 0.26847276, 0, 0)
                ],
                dtype=[
                    ('global_index', '<u4'),
                    ('section_index', '<u4'),
                    ('sec_since_global_start', '<f8'),
                    ('sec_since_section_start', '<f8'),
                    ('temperature', '<f8'),
                    ('hbr', '<f8'),
                    ('hbr_processed', '<f8'),
                    ('is_peak', 'i1'),
                    ('is_manual', 'i1')
                ]
            ),
            'identifier': {
                'absolute_bounds': SectionIndices(start=0, stop=1632697),
                'sampling_rate': 200,
                'section_id': 'SEC_hbr_001',
                'sig_name': 'hbr'
            },
            'peak_edits': {
                'added': [],
                'removed': []
            },
            'peaks_global': np.array(
                [81,     195,     312, ..., 1632411, 1632500, 1632664],
                dtype=uint32
            ),
            'peaks_section': np.array(
                [81,     195,     312, ..., 1632411, 1632500, 1632664],
                dtype=uint32
            ),
            'processing_parameters': {
                'filter_parameters': {
                    'highcut': 8,
                    'lowcut': 0.5,
                    'method': 'butterworth',
                    'order': 3,
                    'powerline': 50,
                    'window_size': 'default'
                },
                'peak_detection_parameters': {
                    'method': 'elgendi_ppg',
                    'method_parameters': { 
                        'beatoffset': 0.02,
                        'beatwindow': 0.667,
                        'mindelay': 0.3,
                        'peakwindow': 0.111
                    }
                },
                'pipeline': 'ppg_elgendi',
                'sampling_rate': 200,
                'standardize_parameters': {
                    'method': 'zscore',
                    'robust': False,
                    'window_size': 2000
                }
            },
            'rate': np.array([125.54338738, 105.26315789, 102.56410256, ..., 141.17647059, 134.83146067,  73.17073171]),
            'rate_interpolated': np.array([125.54338738, 125.54338738, 125.54338738, ...,  73.17073171, 73.17073171,  73.17073171])
        }
    },
    'focused_section_results': {
        'SEC_hbr_001': np.array(
            [
                (     81,      81, 4.050000e-01, 4.050000e-01,   0, 15.5, 125.54338738),
                (    195,     195, 9.750000e-01, 9.750000e-01, 114, 15.5, 105.26315789),
                (    312,     312, 1.560000e+00, 1.560000e+00, 117, 15.5, 102.56410256),
                ...,
                (1632411, 1632411, 8.162055e+03, 8.162055e+03,  85, 18.9, 141.17647059),
                (1632500, 1632500, 8.162500e+03, 8.162500e+03,  89, 18.9, 134.83146067),
                (1632664, 1632664, 8.163320e+03, 8.163320e+03, 164, 18.9,  73.17073171)
            ],
            dtype=[
                ('peaks_section_index', '<u4'),
                ('peaks_global_index', '<u4'),
                ('seconds_since_global_start', '<f8'),
                ('seconds_since_section_start', '<f8'),
                ('peak_intervals', '<u4'),
                ('temperature', '<f8'),
                ('rate_bpm', '<f8')
            ]
        )
    },
    'global_dataframe': np.array(
        [
            (      0, 15.5, 2.359375, 2.359375, 0),
            (      1, 15.5, 2.397188, 2.397188, 0),
            (      2, 15.5, 2.446875, 2.446875, 0),
            ...,
            (5063297, 26.6, 2.504062, 2.504062, 0),
            (5063298, 26.6, 2.497187, 2.497187, 0),
            (5063299, 26.6, 2.499375, 2.499375, 0)
        ],
        dtype=[
            ('index', '<u4'),
            ('temperature', '<f8'),
            ('hbr', '<f8'),
            ('hbr_processed', '<f8'),
            ('is_peak', 'i1')
        ]
    ),
    'identifier': {
        'animal_id': 'F07',
        'date_recorded': None,
        'oxygen_condition': 'Hypoxic',
        'signal_name': 'hbr',
        'source_file_name': 'F07--15_5C26_6C--hypoxic.feather'
    }
}
"""