from typing import Any

import polars as pl
from PySide6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QPersistentModelIndex,
    Qt,
)
from PySide6.QtWidgets import QWidget


class CompactDFModel(QAbstractTableModel):
    """
    A
    """

    def __init__(
        self,
        df_head: pl.DataFrame,
        df_tail: pl.DataFrame,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._columns = [
            (col, str(dtype))
            for col, dtype in zip(df_head.columns, df_head.dtypes, strict=True)
        ]
        self._data = (
            df_head.to_numpy().tolist()
            + [["..."] * len(self._columns)]
            + df_tail.to_numpy().tolist()
        )

    def rowCount(
        self, parent: QModelIndex | QPersistentModelIndex | None = None
    ) -> int:
        return len(self._data)

    def columnCount(
        self, parent: QModelIndex | QPersistentModelIndex | None = None
    ) -> int:
        return len(self._columns)

    def data(
        self,
        index: QModelIndex | QPersistentModelIndex,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if not index.isValid():
            return None

        row, column = index.row(), index.column()

        if role == Qt.ItemDataRole.DisplayRole:
            col_name = self._columns[column][0]
            if self._data[row][column] == "...":
                return "..."
            if "index" in col_name or "peak" in col_name:
                return f"{int(self._data[row][column]):_}"
            elif "is_included" in col_name:
                return f"{self._data[row][column]}"
            elif "time" in col_name:
                return f"{float(self._data[row][column]):_.5f}"
            elif "rate" in col_name:
                return f"{int(self._data[row][column]):_}"
            elif (
                "temp" not in col_name
                and "hb" in col_name
                or "temp" not in col_name
                and "vent" in col_name
            ):
                return f"{float(self._data[row][column]):.4f}"
            elif "temp" in col_name:
                return f"{float(self._data[row][column]):.1f}"
            return str(self._data[row][column])

        return None

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                column_name, column_type = self._columns[section]
                return f"{column_name} ({column_type})"
            if orientation == Qt.Orientation.Vertical:
                return str(section)

        return None


class PolarsModel(QAbstractTableModel):
    """
    A TableModel for use with polars dataframes.
    """

    def __init__(self, dataframe: pl.DataFrame, parent: QWidget | None = None):
        super().__init__(parent)
        self._dataframe = dataframe

    def rowCount(
        self, parent: QModelIndex | QPersistentModelIndex | None = None
    ) -> int:
        return self._dataframe.shape[0]

    def columnCount(
        self, parent: QModelIndex | QPersistentModelIndex | None = None
    ) -> int:
        return self._dataframe.shape[1]

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return self._dataframe.columns[section]

            if orientation == Qt.Orientation.Vertical:
                return f"{section}"

        return None

    def data(
        self,
        index: QModelIndex | QPersistentModelIndex,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if not index.isValid():
            return None

        column = index.column()
        row = index.row()

        if role == Qt.ItemDataRole.DisplayRole:
            col_name = self._dataframe.columns[column]

            if "index" in col_name or "peak" in col_name:
                idx = self._dataframe[row, column]
                return f"{int(idx)}"
            if "time" in col_name:
                time_s = self._dataframe[row, column]
                return f"{time_s:.4f}"
            elif "temp" in col_name:
                temperature = self._dataframe[row, column]
                return f"{temperature:.1f}"
            elif "hb" in col_name:
                hbr = self._dataframe[row, column]
                return f"{hbr:.4f}"
            elif "vent" in col_name:
                ventilation = self._dataframe[row, column]
                return f"{ventilation:.4f}"
            return str(self._dataframe[row, column])

        return None


class DescriptiveStatsModel(QAbstractTableModel):
    def __init__(self, dataframe: pl.DataFrame, parent: QWidget | None = None):
        QAbstractTableModel.__init__(self, parent)
        self._dataframe = dataframe.shrink_to_fit(in_place=True)

    def rowCount(
        self, parent: QModelIndex | QPersistentModelIndex | None = None
    ) -> int:
        return self._dataframe.shape[0]

    def columnCount(
        self, parent: QModelIndex | QPersistentModelIndex | None = None
    ) -> int:
        return self._dataframe.shape[1]

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return self._dataframe.columns[section]

            if orientation == Qt.Orientation.Vertical:
                return f"{section}"

        return None

    def data(
        self,
        index: QModelIndex | QPersistentModelIndex,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if not index.isValid():
            return None

        column = index.column()
        row = index.row()

        if role == Qt.ItemDataRole.DisplayRole:
            col_name = self._dataframe.columns[column]
            info_data = self._dataframe[row, column]
            if (
                self._dataframe.get_column(col_name).dtype != pl.Float64
                or info_data > 100
            ):
                return (
                    str(info_data)
                    if isinstance(info_data, str)
                    else f"{int(info_data):_}"
                )
            return f"{info_data:.4g}"

        return None
