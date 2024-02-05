import polars as pl
from PySide6 import QtCore, QtWidgets

from ..models.polars_df import PolarsDFModel


class TableHandler:
    def __init__(self) -> None:
        self._tables: dict[str, QtWidgets.QTableView] = {}

    def add_table(self, name: str, table: QtWidgets.QTableView) -> None:
        self._tables[name] = table

    def get_table(self, name: str) -> QtWidgets.QTableView:
        return self._tables[name]

    def set_model_data(self, name: str | QtWidgets.QTableView, df: pl.DataFrame) -> None:
        is_description = df.columns[0] == "statistic"
        table = name if isinstance(name, QtWidgets.QTableView) else self._tables[name]
        model = table.model()
        if not isinstance(model, PolarsDFModel):
            model = PolarsDFModel(dataframe=df, is_description=is_description)
            table.setModel(model)
        else:
            table.model().set_data(df, is_description=is_description)
        self._set_header_style(table=table, n_columns=df.width)

    @staticmethod
    def _set_header_style(
        table: QtWidgets.QTableView,
        n_columns: int,
        header_alignment: QtCore.Qt.AlignmentFlag = QtCore.Qt.AlignmentFlag.AlignLeft,
        resize_mode: QtWidgets.QHeaderView.ResizeMode = QtWidgets.QHeaderView.ResizeMode.Stretch,
    ) -> None:
        table.horizontalHeader().setSelectionMode(QtWidgets.QHeaderView.SelectionMode.NoSelection)
        table.setSortingEnabled(False)
        table.horizontalHeader().setDefaultAlignment(header_alignment)
        table.verticalHeader().setVisible(False)
        table.resizeColumnsToContents()
        for col in range(n_columns):
            table.horizontalHeader().setSectionResizeMode(col, resize_mode)