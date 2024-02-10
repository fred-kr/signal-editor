import polars as pl
from PySide6 import QtCore, QtWidgets

from ..models import PolarsDFModel


class TableHandler:
    def __init__(self, **kwargs: QtWidgets.QTableView) -> None:
        self._tables: dict[str, QtWidgets.QTableView] = {**kwargs}
        self.loaded_rows: int = 0

    def add_table(self, name: str, table: QtWidgets.QTableView) -> None:
        self._tables[name] = table

    def get_table(self, name: str) -> QtWidgets.QTableView:
        return self._tables[name]

    def set_model_data(self, table: QtWidgets.QTableView, lf: pl.LazyFrame) -> None:
        self.loaded_rows += 500
        is_description = lf.columns[0] == "statistic"
        df = lf.collect() if is_description else lf.limit(self.loaded_rows).collect()
        model = table.model()
        if not isinstance(model, PolarsDFModel):
            model = PolarsDFModel(dataframe=df, is_description=is_description)
            table.setModel(model)
        else:
            model.set_data(df, is_description=is_description)
        self._set_header_style(table=table, n_columns=df.width)

    @staticmethod
    def _set_header_style(
        table: QtWidgets.QTableView,
        n_columns: int,
        header_alignment: QtCore.Qt.AlignmentFlag = QtCore.Qt.AlignmentFlag.AlignLeft,
        resize_mode: QtWidgets.QHeaderView.ResizeMode = QtWidgets.QHeaderView.ResizeMode.Stretch,
    ) -> None:
        table.horizontalHeader().setSelectionMode(QtWidgets.QHeaderView.SelectionMode.NoSelection)
        table.horizontalHeader().setHighlightSections(False)
        table.horizontalHeader().setSectionsClickable(False)
        table.horizontalHeader().setSectionsMovable(False)
        table.horizontalHeader().setDefaultAlignment(header_alignment)
        table.setSortingEnabled(False)
        table.verticalHeader().setVisible(False)
        table.resizeColumnsToContents()
        for col in range(n_columns):
            table.horizontalHeader().setSectionResizeMode(col, resize_mode)
