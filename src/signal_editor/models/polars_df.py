import typing as t

import polars as pl
import polars.selectors as ps
from PySide6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QPersistentModelIndex,
    Qt,
)
from PySide6.QtWidgets import QWidget


class PolarsTableModel(QAbstractTableModel):
    def __init__(
        self,
        dataframe: pl.DataFrame,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._df = (
            dataframe.lazy()
            .with_columns(
                ps.contains("time").cast(pl.Decimal(12, 3)),
                ps.contains("temp").cast(pl.Decimal(3, 1)),
            )
            .collect()
            .rechunk()
        )

    def set_df(self, dataframe: pl.DataFrame) -> None:
        self.beginResetModel()
        self._df = (
            dataframe.lazy()
            .with_columns(
                ps.contains("time").cast(pl.Decimal(12, 3)),
                ps.contains("temp").cast(pl.Decimal(3, 1)),
            )
            .collect()
            .rechunk()
        )
        self.endResetModel()

    def rowCount(self, parent: QModelIndex | QPersistentModelIndex | None = None) -> int:
        return self._df.height

    def columnCount(self, parent: QModelIndex | QPersistentModelIndex | None = None) -> int:
        return self._df.width

    def data(
        self,
        index: QModelIndex | QPersistentModelIndex,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> t.Any:
        if not index.isValid():
            return None

        if role == Qt.ItemDataRole.DisplayRole:
            row, column = index.row(), index.column()
            value = self._df.get_column(self._df.columns[column])[row]
            dtype = self._df.dtypes[column]

            match dtype:
                case (
                    pl.Int8
                    | pl.Int16
                    | pl.Int32
                    | pl.Int64
                    | pl.UInt8
                    | pl.UInt16
                    | pl.UInt32
                    | pl.UInt64
                ):
                    return f"{value:_}"
                case pl.Float32 | pl.Float64:
                    return f"{value:_.4f}"
                case pl.Decimal:
                    return f"{value}"
                case _:
                    return str(value)

        return None

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> t.Any:
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                name, dtype = self._df.columns[section], self._df.dtypes[section]
                return f"{name}\n---\n{dtype}"
            if orientation == Qt.Orientation.Vertical:
                return str(section)

        return None


class DescriptiveStatsModel(QAbstractTableModel):
    def __init__(
        self, desc_df: pl.DataFrame, orig_dtypes: list[pl.DataType], parent: QWidget | None = None
    ):
        QAbstractTableModel.__init__(self, parent)
        self._dtypes = orig_dtypes
        self._desc_df = desc_df

    @property
    def desc_df(self):
        return self._desc_df

    def set_df(self, desc_df: pl.DataFrame):
        self.beginResetModel()
        self._desc_df = desc_df
        self.endResetModel()

    @property
    def dtypes(self):
        return [pl.String, *self._dtypes]

    def set_dtypes(self, dtypes: list[pl.DataType]):
        self._dtypes = dtypes

    def rowCount(self, parent: QModelIndex | QPersistentModelIndex | None = None) -> int:
        return self._desc_df.height

    def columnCount(self, parent: QModelIndex | QPersistentModelIndex | None = None) -> int:
        return self._desc_df.width

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> t.Any:
        if role == Qt.ItemDataRole.DisplayRole:
            match orientation:
                case Qt.Orientation.Horizontal:
                    name, dtype = self.desc_df.columns[section], self.dtypes[section]
                    return f"{name}\n---\n{dtype}"
                case Qt.Orientation.Vertical:
                    return str(section)

        return None

    def data(
        self,
        index: QModelIndex | QPersistentModelIndex,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> t.Any:
        if not index.isValid():
            return None

        row, column = index.row(), index.column()

        if role == Qt.ItemDataRole.DisplayRole:
            value = self._desc_df.get_column(self.desc_df.columns[column])[row]
            dtype = self.dtypes[column]

            match dtype:
                case (
                    pl.Int8
                    | pl.Int16
                    | pl.Int32
                    | pl.Int64
                    | pl.UInt8
                    | pl.UInt16
                    | pl.UInt32
                    | pl.UInt64
                ):
                    return f"{value:_}"
                case pl.Float32 | pl.Float64:
                    return f"{value:_.4f}"
                case pl.Decimal:
                    return f"{value}"
                case _:
                    return str(value)

        return None
