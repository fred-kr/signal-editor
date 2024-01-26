import typing as t

from PySide6 import QtCore


class SectionListModel(QtCore.QAbstractListModel):
    def __init__(self, sections: list[str], parent: QtCore.QObject | None = None):
        super().__init__(parent)
        self._sections = sections

    def rowCount(
        self, parent: QtCore.QModelIndex | QtCore.QPersistentModelIndex | None = None
    ) -> int:
        return len(self._sections)

    def data(
        self,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> t.Any:
        if not index.isValid():
            return None

        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            return self._sections[index.row()]

        return None

    def flags(self, index: QtCore.QModelIndex | QtCore.QPersistentModelIndex) -> QtCore.Qt.ItemFlag:
        if not index.isValid():
            return QtCore.Qt.ItemFlag.NoItemFlags

        return QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable

    def insertRows(
        self,
        row: int,
        count: int,
        parent: QtCore.QModelIndex | QtCore.QPersistentModelIndex | None = None,
    ) -> bool:
        self.beginInsertRows(parent or QtCore.QModelIndex(), row, row + count - 1)
        self._sections[row : row + count] = ["" for _ in range(count)]
        self.endInsertRows()
        return True

    def removeRows(
        self,
        row: int,
        count: int,
        parent: QtCore.QModelIndex | QtCore.QPersistentModelIndex | None = None,
    ) -> bool:
        self.beginRemoveRows(parent or QtCore.QModelIndex(), row, row + count - 1)
        del self._sections[row : row + count]
        self.endRemoveRows()
        return True

    def setData(
        self,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
        value: t.Any,
        role: int = QtCore.Qt.ItemDataRole.EditRole,
    ) -> bool:
        if not index.isValid():
            return False

        if role == QtCore.Qt.ItemDataRole.EditRole:
            self._sections[index.row()] = value
            self.dataChanged.emit(index, index)
            return True

        return False

    def headerData(
        self,
        section: int,
        orientation: QtCore.Qt.Orientation,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> t.Any:
        if (
            role == QtCore.Qt.ItemDataRole.DisplayRole
            and orientation == QtCore.Qt.Orientation.Horizontal
        ):
            return f"Section {section}"

        return None
