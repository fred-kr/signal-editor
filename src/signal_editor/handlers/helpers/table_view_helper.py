from PySide6.QtCore import Qt, QAbstractTableModel
from PySide6.QtWidgets import QHeaderView, QTableView


class TableViewHelper:
    def __init__(self, table_view: QTableView | None = None) -> None:
        if table_view is not None:
            self.table_view = table_view

    def set_table_view(self, table_view: QTableView) -> None:
        self.table_view = table_view

    def make_table(self, model: QAbstractTableModel) -> None:
        self.table_view.setModel(model)
        self.customize_table_header(model.columnCount())

    def customize_table_header(
        self,
        n_columns: int,
        header_alignment: Qt.AlignmentFlag = Qt.AlignmentFlag.AlignLeft,
        resize_mode: QHeaderView.ResizeMode = QHeaderView.ResizeMode.Stretch,
    ) -> None:
        self.table_view.horizontalHeader().setDefaultAlignment(header_alignment)
        self.table_view.verticalHeader().setVisible(False)
        self.table_view.resizeColumnsToContents()
        for col in range(n_columns):
            self.table_view.horizontalHeader().setSectionResizeMode(col, resize_mode)