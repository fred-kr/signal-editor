import polars as pl
import polars.selectors as ps
from PySide6 import QtCore

type ModelIndex = QtCore.QModelIndex | QtCore.QPersistentModelIndex


class PolarsDFModel(QtCore.QAbstractTableModel):
    def __init__(
        self,
        dataframe: pl.DataFrame,
        is_description: bool = False,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._is_description = is_description
        self._df = dataframe
        self.set_data(self._df, is_description=is_description)

    def set_data(self, dataframe: pl.DataFrame, is_description: bool = False) -> None:
        self.beginResetModel()
        if hasattr(self, "_df"):
            del self._df
        if is_description:
            self._df = dataframe
        else:
            self._df = (
                dataframe.lazy()
                .with_columns(
                    ps.contains(["time", "second"]).cast(pl.Decimal(12, 4)),
                    ps.contains("temp").cast(pl.Decimal(3, 1)),
                )
                .collect()
            )
        self.endResetModel()

    def rowCount(self, parent: ModelIndex | None = None) -> int:
        return self._df.height

    def columnCount(self, parent: ModelIndex | None = None) -> int:
        return self._df.width

    def data(
        self,
        index: ModelIndex,
        role: int | QtCore.Qt.ItemDataRole = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> str | None:
        if not index.isValid():
            return None
        if self._df.is_empty():
            return None

        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            row, column = index.row(), index.column()
            value = self._df.item(row, column)
            dtype = self._df.dtypes[column]

            if value is None:
                return "N/A"
            if dtype in pl.INTEGER_DTYPES:
                return f"{value:_}"
            elif dtype in pl.FLOAT_DTYPES:
                return f"{value:_.4f}"
            elif dtype == pl.Decimal:
                return f"{value}"
            else:
                return str(value)
        return None

    def headerData(
        self,
        section: int,
        orientation: QtCore.Qt.Orientation,
        role: int | QtCore.Qt.ItemDataRole = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> str | None:
        if role != QtCore.Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == QtCore.Qt.Orientation.Horizontal:
            name = self._df.columns[section]
            if self._is_description:
                return name
            dtype = self._df.dtypes[section]
            return f"{name}\n---\n{dtype}"
        return str(section)

    def flags(self, index: QtCore.QModelIndex | QtCore.QPersistentModelIndex) -> QtCore.Qt.ItemFlag:
        return QtCore.Qt.ItemFlag.ItemIsEnabled
