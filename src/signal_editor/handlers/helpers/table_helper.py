import typing as t

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHeaderView, QTableView
import polars as pl

from ...models.polars_df import DescriptiveStatsModel, PolarsTableModel


class TableHelper:
    def __init__(self, view: QTableView, model: PolarsTableModel | DescriptiveStatsModel) -> None:
        self._view = view
        self._view.setModel(model)
        self._customize_table_header(self._view.model().columnCount())

    @property
    def view(self) -> QTableView:
        return self._view

    @property
    def cas_model(self) -> PolarsTableModel | None:
        if isinstance(self._view.model(), PolarsTableModel):
            return t.cast(PolarsTableModel, self._view.model())
        return None
    
    @property
    def cas_desc_model(self) -> DescriptiveStatsModel | None:
        if isinstance(self._view.model(), DescriptiveStatsModel):
            return t.cast(DescriptiveStatsModel, self._view.model())
        return None
        
    
    def update_cas_df(self, df: pl.DataFrame) -> None:
        if self.cas_model is not None:
            self.cas_model.set_df(df)

    def update_cas_desc_df(self, df: pl.DataFrame, dtypes: list[pl.DataType]) -> None:
        if self.cas_desc_model is not None:
            self.cas_desc_model.set_dtypes(dtypes)
            self.cas_desc_model.set_df(df)
        
    def _customize_table_header(
        self,
        n_columns: int,
        header_alignment: Qt.AlignmentFlag = Qt.AlignmentFlag.AlignLeft,
        resize_mode: QHeaderView.ResizeMode = QHeaderView.ResizeMode.Stretch,
    ) -> None:
        self._view.horizontalHeader().setDefaultAlignment(header_alignment)
        self._view.verticalHeader().setVisible(False)
        self._view.resizeColumnsToContents()
        for col in range(n_columns):
            self._view.horizontalHeader().setSectionResizeMode(col, resize_mode)
