import polars as pl
from PySide6 import QtCore, QtWidgets

from ..models import PolarsDFModel


class TableHandler:
    def __init__(self, *args: QtWidgets.QTableView, **kwargs: QtWidgets.QTableView) -> None:
        self._tables: dict[str, QtWidgets.QTableView] = {}
        if args:
            self._tables |= {table.objectName(): table for table in args}

        if kwargs:
            self._tables |= kwargs

        self._row_limit: int = 500

    def add_view(self, name: str, table_view: QtWidgets.QTableView) -> None:
        self._tables[name] = table_view

    def get_view_by_name(self, name: str) -> QtWidgets.QTableView:
        return self._tables[name]

    def set_row_limit(self, limit: int) -> None:
        self._row_limit = limit

    def create_df_model(self, lf: pl.LazyFrame, name: str) -> None:
        limit = self._row_limit
        table = self._tables.get(name, None)
        if table is None:
            raise ValueError(f"Table with name {name} does not exist")
        df = lf.head(limit).collect()
        is_description = df.columns[0] == "statistic"
        model = PolarsDFModel(dataframe=df, is_description=is_description)
        table.setModel(model)
        self._set_header_style(table=table, n_columns=df.width)

    def update_df_model(
        self,
        data: pl.DataFrame | pl.LazyFrame,
        name: str | QtWidgets.QTableView,
        limit: int | None = None,
    ) -> None:
        if isinstance(name, QtWidgets.QTableView):
            table = name
            name = table.objectName()
        else:
            table = self._tables.get(name, None)
        if table is None:
            raise ValueError(f"Table with name {name} does not exist")
        if limit is None:
            limit = self._row_limit
        is_description = data.columns[0] == "statistic"
        if isinstance(data, pl.DataFrame):
            df = data if is_description else data.head(limit)
        else:
            df = data.collect() if is_description else data.head(limit).collect()
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
        table_header = table.horizontalHeader()
        table_header.setSelectionMode(QtWidgets.QHeaderView.SelectionMode.NoSelection)
        table_header.setHighlightSections(False)
        table_header.setSectionsClickable(False)
        table_header.setSectionsMovable(False)
        table_header.setDefaultAlignment(header_alignment)
        table.setSortingEnabled(False)
        table.verticalHeader().setVisible(False)
        table.resizeColumnsToContents()
        for col in range(n_columns):
            table.horizontalHeader().setSectionResizeMode(col, resize_mode)

    def clear(self) -> None:
        for table in self._tables.values():
            table.setModel(None)
        self._tables.clear()
