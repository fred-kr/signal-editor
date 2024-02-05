import polars as pl
import polars.selectors as ps
from PySide6 import QtCore


class PolarsDFModel(QtCore.QAbstractTableModel):
    def __init__(
        self,
        dataframe: pl.DataFrame,
        is_description: bool = False,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._is_description = is_description
        self.set_data(dataframe, is_description=is_description)

    def set_data(self, dataframe: pl.DataFrame, is_description: bool = False) -> None:
        self.beginResetModel()
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
                .rechunk()
            )
        self.endResetModel()

    def rowCount(
        self, parent: QtCore.QModelIndex | QtCore.QPersistentModelIndex | None = None
    ) -> int:
        return self._df.height

    def columnCount(
        self, parent: QtCore.QModelIndex | QtCore.QPersistentModelIndex | None = None
    ) -> int:
        return self._df.width

    def data(
        self,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> str | None:
        if not index.isValid():
            return None
        if index.row() >= self.rowCount() or index.column() >= self.columnCount():
            return None
        if self._df.is_empty():
            return None

        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            row, column = index.row(), index.column()
            value = self._df.get_column(self._df.columns[column])[row]
            dtype = self._df.dtypes[column]

            if value is None:
                return "N/A"
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
        orientation: QtCore.Qt.Orientation,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> str | None:
        if role != QtCore.Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == QtCore.Qt.Orientation.Horizontal:
            name = self._df.columns[section]
            dtype = self._df.dtypes[section]
            return f"{name}\n---\n{dtype}"
        return str(section)

    def flags(self, index: QtCore.QModelIndex) -> QtCore.Qt.ItemFlag:
        return QtCore.Qt.ItemFlag.ItemIsEnabled


# class PolarsDataFrameDescriptionModel(QtCore.QAbstractTableModel):
#     def __init__(
#         self,
#         desc_df: pl.DataFrame,
#         orig_dtypes: list[pl.DataType],
#         parent: QtCore.QObject | None = None,
#     ):
#         super().__init__(parent)
#         self._dtypes = orig_dtypes
#         self._desc_df = desc_df
#
#     @property
#     def desc_df(self):
#         return self._desc_df
#
#     def set_df(self, desc_df: pl.DataFrame):
#         self.beginResetModel()
#         self._desc_df = desc_df
#         self.endResetModel()
#
#     @property
#     def dtypes(self):
#         return [pl.String, *self._dtypes]
#
#     def set_dtypes(self, dtypes: list[pl.DataType]):
#         self._dtypes = dtypes
#
#     def rowCount(
#         self, parent: QtCore.QModelIndex | QtCore.QPersistentModelIndex | None = None
#     ) -> int:
#         return self._desc_df.height
#
#     def columnCount(
#         self, parent: QtCore.QModelIndex | QtCore.QPersistentModelIndex | None = None
#     ) -> int:
#         return self._desc_df.width
#
#     def headerData(
#         self,
#         section: int,
#         orientation: QtCore.Qt.Orientation,
#         role: int = QtCore.Qt.ItemDataRole.DisplayRole,
#     ) -> t.Any:
#         if role == QtCore.Qt.ItemDataRole.DisplayRole:
#             match orientation:
#                 case QtCore.Qt.Orientation.Horizontal:
#                     name, dtype = self.desc_df.columns[section], self.dtypes[section]
#                     return f"{name}\n---\n{dtype}"
#                 case QtCore.Qt.Orientation.Vertical:
#                     return str(section)
#
#         return None
#
#     def data(
#         self,
#         index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
#         role: int = QtCore.Qt.ItemDataRole.DisplayRole,
#     ) -> t.Any:
#         if not index.isValid():
#             return None
#
#         row, column = index.row(), index.column()
#
#         if role == QtCore.Qt.ItemDataRole.DisplayRole:
#             value = self._desc_df.get_column(self.desc_df.columns[column])[row]
#             dtype = self.dtypes[column]
#
#             match dtype:
#                 case (
#                     pl.Int8
#                     | pl.Int16
#                     | pl.Int32
#                     | pl.Int64
#                     | pl.UInt8
#                     | pl.UInt16
#                     | pl.UInt32
#                     | pl.UInt64
#                 ):
#                     return f"{value:_}"
#                 case pl.Float32 | pl.Float64:
#                     return f"{value:_.4f}"
#                 case pl.Decimal:
#                     return f"{value}"
#                 case _:
#                     return str(value)
#
#         return None