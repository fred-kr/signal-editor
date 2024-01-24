# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'main_window.ui'
##
## Created by: Qt User Interface Compiler version 6.6.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QAbstractItemView, QAbstractScrollArea, QAbstractSpinBox, QApplication,
    QCheckBox, QComboBox, QDateEdit, QDockWidget,
    QDoubleSpinBox, QFormLayout, QFrame, QGridLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QMainWindow, QMenu, QMenuBar,
    QPushButton, QSizePolicy, QSlider, QSpacerItem,
    QSpinBox, QStackedWidget, QStatusBar, QTabWidget,
    QTableView, QTextBrowser, QToolBar, QVBoxLayout,
    QWidget)

from pyqtgraph import (ComboBox, FeedbackButton)
from . import icons_rc

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(2348, 1219)
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        icon = QIcon()
        icon.addFile(u":/custom-icons/app-icon-v2.svg", QSize(), QIcon.Normal, QIcon.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setToolButtonStyle(Qt.ToolButtonFollowStyle)
        MainWindow.setDockOptions(QMainWindow.AllowNestedDocks|QMainWindow.AnimatedDocks|QMainWindow.VerticalTabs)
        self.action_open_console = QAction(MainWindow)
        self.action_open_console.setObjectName(u"action_open_console")
        self.action_open_console.setChecked(False)
        icon1 = QIcon()
        icon1.addFile(u":/material-symbols/terminal_FILL0_wght400_GRAD0_opsz24.png", QSize(), QIcon.Normal, QIcon.Off)
        self.action_open_console.setIcon(icon1)
        self.action_open_console.setVisible(True)
        self.action_open_console.setMenuRole(QAction.NoRole)
        self.action_toggle_sidebar = QAction(MainWindow)
        self.action_toggle_sidebar.setObjectName(u"action_toggle_sidebar")
        self.action_toggle_sidebar.setCheckable(True)
        self.action_toggle_sidebar.setChecked(True)
        icon2 = QIcon()
        icon2.addFile(u":/material-symbols/thumbnail_bar_FILL0_wght400_GRAD0_opsz24.png", QSize(), QIcon.Normal, QIcon.Off)
        self.action_toggle_sidebar.setIcon(icon2)
        self.action_toggle_sidebar.setMenuRole(QAction.NoRole)
        self.action_select_file = QAction(MainWindow)
        self.action_select_file.setObjectName(u"action_select_file")
        icon3 = QIcon()
        icon3.addFile(u":/material-symbols/folder_open_FILL0_wght400_GRAD0_opsz24.png", QSize(), QIcon.Normal, QIcon.Off)
        self.action_select_file.setIcon(icon3)
        self.action_exit = QAction(MainWindow)
        self.action_exit.setObjectName(u"action_exit")
        icon4 = QIcon()
        icon4.addFile(u":/material-symbols/close_FILL0_wght400_GRAD0_opsz24.png", QSize(), QIcon.Normal, QIcon.Off)
        self.action_exit.setIcon(icon4)
        self.action_reset_all = QAction(MainWindow)
        self.action_reset_all.setObjectName(u"action_reset_all")
        icon5 = QIcon()
        icon5.addFile(u":/material-symbols/frame_reload_FILL0_wght400_GRAD0_opsz24.png", QSize(), QIcon.Normal, QIcon.Off)
        self.action_reset_all.setIcon(icon5)
        self.action_reset_all.setMenuRole(QAction.NoRole)
        self.action_pan_mode = QAction(MainWindow)
        self.action_pan_mode.setObjectName(u"action_pan_mode")
        self.action_pan_mode.setCheckable(True)
        icon6 = QIcon()
        icon6.addFile(u":/material-symbols/drag_pan_FILL0_wght400_GRAD0_opsz24.png", QSize(), QIcon.Normal, QIcon.Off)
        self.action_pan_mode.setIcon(icon6)
        self.action_pan_mode.setMenuRole(QAction.NoRole)
        self.action_pan_mode.setIconVisibleInMenu(True)
        self.action_rect_mode = QAction(MainWindow)
        self.action_rect_mode.setObjectName(u"action_rect_mode")
        self.action_rect_mode.setCheckable(True)
        icon7 = QIcon()
        icon7.addFile(u":/material-symbols/crop_free_FILL0_wght400_GRAD0_opsz24.png", QSize(), QIcon.Normal, QIcon.Off)
        self.action_rect_mode.setIcon(icon7)
        self.action_rect_mode.setMenuRole(QAction.NoRole)
        self.action_reset_view = QAction(MainWindow)
        self.action_reset_view.setObjectName(u"action_reset_view")
        icon8 = QIcon()
        icon8.addFile(u":/material-symbols/fit_screen_FILL0_wght400_GRAD0_opsz24.png", QSize(), QIcon.Normal, QIcon.Off)
        self.action_reset_view.setIcon(icon8)
        self.action_reset_view.setMenuRole(QAction.NoRole)
        self.action_previous_section = QAction(MainWindow)
        self.action_previous_section.setObjectName(u"action_previous_section")
        icon9 = QIcon()
        icon9.addFile(u":/material-symbols/arrow_back_FILL0_wght400_GRAD0_opsz24.png", QSize(), QIcon.Normal, QIcon.Off)
        self.action_previous_section.setIcon(icon9)
        self.action_previous_section.setMenuRole(QAction.NoRole)
        self.action_next_section = QAction(MainWindow)
        self.action_next_section.setObjectName(u"action_next_section")
        icon10 = QIcon()
        icon10.addFile(u":/material-symbols/arrow_forward_FILL0_wght400_GRAD0_opsz24.png", QSize(), QIcon.Normal, QIcon.Off)
        self.action_next_section.setIcon(icon10)
        self.action_next_section.setMenuRole(QAction.NoRole)
        self.action_toggle_whats_this_mode = QAction(MainWindow)
        self.action_toggle_whats_this_mode.setObjectName(u"action_toggle_whats_this_mode")
        self.action_toggle_whats_this_mode.setCheckable(True)
        icon11 = QIcon()
        icon11.addFile(u":/material-symbols/help_FILL0_wght400_GRAD0_opsz24.png", QSize(), QIcon.Normal, QIcon.Off)
        self.action_toggle_whats_this_mode.setIcon(icon11)
        self.action_toggle_whats_this_mode.setMenuRole(QAction.NoRole)
        self.action_remove_peak_rect = QAction(MainWindow)
        self.action_remove_peak_rect.setObjectName(u"action_remove_peak_rect")
        icon12 = QIcon()
        icon12.addFile(u":/material-symbols/select_FILL0_wght400_GRAD0_opsz24.png", QSize(), QIcon.Normal, QIcon.Off)
        self.action_remove_peak_rect.setIcon(icon12)
        self.action_remove_peak_rect.setMenuRole(QAction.NoRole)
        self.action_run_preprocessing = QAction(MainWindow)
        self.action_run_preprocessing.setObjectName(u"action_run_preprocessing")
        icon13 = QIcon()
        icon13.addFile(u":/material-symbols/data_thresholding_FILL0_wght400_GRAD0_opsz24.png", QSize(), QIcon.Normal, QIcon.Off)
        self.action_run_preprocessing.setIcon(icon13)
        self.action_run_preprocessing.setAutoRepeat(False)
        self.action_run_peak_detection = QAction(MainWindow)
        self.action_run_peak_detection.setObjectName(u"action_run_peak_detection")
        icon14 = QIcon()
        icon14.addFile(u":/material-symbols/query_stats_FILL0_wght400_GRAD0_opsz24.png", QSize(), QIcon.Normal, QIcon.Off)
        self.action_run_peak_detection.setIcon(icon14)
        self.action_run_peak_detection.setAutoRepeat(False)
        self.action_get_results = QAction(MainWindow)
        self.action_get_results.setObjectName(u"action_get_results")
        icon15 = QIcon()
        icon15.addFile(u":/material-symbols/start_FILL0_wght400_GRAD0_opsz24.png", QSize(), QIcon.Normal, QIcon.Off)
        self.action_get_results.setIcon(icon15)
        self.action_get_results.setMenuRole(QAction.NoRole)
        self.action_remove_selected_peaks = QAction(MainWindow)
        self.action_remove_selected_peaks.setObjectName(u"action_remove_selected_peaks")
        self.action_remove_selected_peaks.setCheckable(False)
        icon16 = QIcon()
        icon16.addFile(u":/material-symbols/remove_selection_FILL0_wght400_GRAD0_opsz24.png", QSize(), QIcon.Normal, QIcon.Off)
        self.action_remove_selected_peaks.setIcon(icon16)
        self.action_remove_selected_peaks.setMenuRole(QAction.NoRole)
        self.action_save_state = QAction(MainWindow)
        self.action_save_state.setObjectName(u"action_save_state")
        self.action_save_state.setEnabled(True)
        icon17 = QIcon()
        icon17.addFile(u":/material-symbols/file_save_FILL0_wght400_GRAD0_opsz24.png", QSize(), QIcon.Normal, QIcon.Off)
        self.action_save_state.setIcon(icon17)
        self.action_save_state.setMenuRole(QAction.NoRole)
        self.action_load_state = QAction(MainWindow)
        self.action_load_state.setObjectName(u"action_load_state")
        icon18 = QIcon()
        icon18.addFile(u":/material-symbols/upload_file_FILL0_wght400_GRAD0_opsz24.png", QSize(), QIcon.Normal, QIcon.Off)
        self.action_load_state.setIcon(icon18)
        self.action_load_state.setMenuRole(QAction.NoRole)
        self.action_save_to_hdf5 = QAction(MainWindow)
        self.action_save_to_hdf5.setObjectName(u"action_save_to_hdf5")
        icon19 = QIcon()
        icon19.addFile(u":/material-symbols/save_as_FILL0_wght400_GRAD0_opsz24.png", QSize(), QIcon.Normal, QIcon.Off)
        self.action_save_to_hdf5.setIcon(icon19)
        self.action_save_to_hdf5.setMenuRole(QAction.NoRole)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self._2 = QHBoxLayout(self.centralwidget)
        self._2.setSpacing(4)
        self._2.setContentsMargins(7, 7, 7, 7)
        self._2.setObjectName(u"_2")
        self._2.setContentsMargins(0, 0, 0, 0)
        self.tabs_main = QTabWidget(self.centralwidget)
        self.tabs_main.setObjectName(u"tabs_main")
        sizePolicy1 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.tabs_main.sizePolicy().hasHeightForWidth())
        self.tabs_main.setSizePolicy(sizePolicy1)
        self.tabs_main.setTabShape(QTabWidget.Rounded)
        self.tabs_main.setDocumentMode(True)
        self.tab_data = QWidget()
        self.tab_data.setObjectName(u"tab_data")
        self.gridLayout_4 = QGridLayout(self.tab_data)
        self.gridLayout_4.setSpacing(4)
        self.gridLayout_4.setContentsMargins(7, 7, 7, 7)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.table_data_info = QTableView(self.tab_data)
        self.table_data_info.setObjectName(u"table_data_info")
        sizePolicy2 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.table_data_info.sizePolicy().hasHeightForWidth())
        self.table_data_info.setSizePolicy(sizePolicy2)
        self.table_data_info.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.table_data_info.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table_data_info.setProperty("showDropIndicator", False)
        self.table_data_info.horizontalHeader().setMinimumSectionSize(75)
        self.table_data_info.horizontalHeader().setHighlightSections(True)
        self.table_data_info.verticalHeader().setVisible(False)

        self.gridLayout_4.addWidget(self.table_data_info, 0, 1, 1, 1)

        self.table_data_preview = QTableView(self.tab_data)
        self.table_data_preview.setObjectName(u"table_data_preview")
        sizePolicy2.setHeightForWidth(self.table_data_preview.sizePolicy().hasHeightForWidth())
        self.table_data_preview.setSizePolicy(sizePolicy2)
        self.table_data_preview.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.table_data_preview.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table_data_preview.setProperty("showDropIndicator", False)
        self.table_data_preview.setAlternatingRowColors(True)
        self.table_data_preview.setSelectionMode(QAbstractItemView.NoSelection)
        self.table_data_preview.setWordWrap(False)
        self.table_data_preview.setCornerButtonEnabled(False)
        self.table_data_preview.horizontalHeader().setMinimumSectionSize(75)
        self.table_data_preview.horizontalHeader().setStretchLastSection(False)
        self.table_data_preview.verticalHeader().setVisible(False)

        self.gridLayout_4.addWidget(self.table_data_preview, 0, 0, 3, 1)

        self.container_text_info = QFrame(self.tab_data)
        self.container_text_info.setObjectName(u"container_text_info")
        sizePolicy.setHeightForWidth(self.container_text_info.sizePolicy().hasHeightForWidth())
        self.container_text_info.setSizePolicy(sizePolicy)
        self.gridLayout_13 = QGridLayout(self.container_text_info)
        self.gridLayout_13.setSpacing(4)
        self.gridLayout_13.setContentsMargins(7, 7, 7, 7)
        self.gridLayout_13.setObjectName(u"gridLayout_13")
        self.gridLayout_13.setContentsMargins(-1, -1, -1, 0)
        self.label_19 = QLabel(self.container_text_info)
        self.label_19.setObjectName(u"label_19")
        sizePolicy3 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.label_19.sizePolicy().hasHeightForWidth())
        self.label_19.setSizePolicy(sizePolicy3)

        self.gridLayout_13.addWidget(self.label_19, 0, 0, 1, 1)

        self.text_info_loading_data = QTextBrowser(self.container_text_info)
        self.text_info_loading_data.setObjectName(u"text_info_loading_data")
        sizePolicy1.setHeightForWidth(self.text_info_loading_data.sizePolicy().hasHeightForWidth())
        self.text_info_loading_data.setSizePolicy(sizePolicy1)
        self.text_info_loading_data.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContentsOnFirstShow)

        self.gridLayout_13.addWidget(self.text_info_loading_data, 1, 0, 1, 1)


        self.gridLayout_4.addWidget(self.container_text_info, 1, 1, 1, 1)

        icon20 = QIcon()
        icon20.addFile(u":/material-symbols/dataset_FILL0_wght400_GRAD0_opsz24.png", QSize(), QIcon.Normal, QIcon.Off)
        self.tabs_main.addTab(self.tab_data, icon20, "")
        self.tab_plots = QWidget()
        self.tab_plots.setObjectName(u"tab_plots")
        self.verticalLayout_4 = QVBoxLayout(self.tab_plots)
        self.verticalLayout_4.setSpacing(4)
        self.verticalLayout_4.setContentsMargins(7, 7, 7, 7)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.container_plots = QWidget(self.tab_plots)
        self.container_plots.setObjectName(u"container_plots")

        self.verticalLayout_4.addWidget(self.container_plots)

        icon21 = QIcon()
        icon21.addFile(u":/material-symbols/earthquake_FILL0_wght400_GRAD0_opsz24.png", QSize(), QIcon.Normal, QIcon.Off)
        self.tabs_main.addTab(self.tab_plots, icon21, "")
        self.tab_results = QWidget()
        self.tab_results.setObjectName(u"tab_results")
        sizePolicy2.setHeightForWidth(self.tab_results.sizePolicy().hasHeightForWidth())
        self.tab_results.setSizePolicy(sizePolicy2)
        self.gridLayout_12 = QGridLayout(self.tab_results)
        self.gridLayout_12.setSpacing(4)
        self.gridLayout_12.setContentsMargins(7, 7, 7, 7)
        self.gridLayout_12.setObjectName(u"gridLayout_12")
        self.container_results = QWidget(self.tab_results)
        self.container_results.setObjectName(u"container_results")
        self.gridLayout_8 = QGridLayout(self.container_results)
        self.gridLayout_8.setSpacing(4)
        self.gridLayout_8.setContentsMargins(7, 7, 7, 7)
        self.gridLayout_8.setObjectName(u"gridLayout_8")
        self.gridLayout_8.setContentsMargins(0, 0, 0, 0)
        self.container_results_table = QWidget(self.container_results)
        self.container_results_table.setObjectName(u"container_results_table")
        self.gridLayout_11 = QGridLayout(self.container_results_table)
        self.gridLayout_11.setSpacing(4)
        self.gridLayout_11.setContentsMargins(7, 7, 7, 7)
        self.gridLayout_11.setObjectName(u"gridLayout_11")
        self.gridLayout_11.setContentsMargins(0, 0, 0, 0)
        self.tab_container_result_views = QTabWidget(self.container_results_table)
        self.tab_container_result_views.setObjectName(u"tab_container_result_views")
        self.tab_container_result_views.setLayoutDirection(Qt.LeftToRight)
        self.tab_focused_result = QWidget()
        self.tab_focused_result.setObjectName(u"tab_focused_result")
        self.verticalLayout_6 = QVBoxLayout(self.tab_focused_result)
        self.verticalLayout_6.setSpacing(4)
        self.verticalLayout_6.setContentsMargins(7, 7, 7, 7)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.table_view_focused_result = QTableView(self.tab_focused_result)
        self.table_view_focused_result.setObjectName(u"table_view_focused_result")
        self.table_view_focused_result.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table_view_focused_result.setTabKeyNavigation(False)
        self.table_view_focused_result.setProperty("showDropIndicator", False)
        self.table_view_focused_result.setDragDropOverwriteMode(False)
        self.table_view_focused_result.setSortingEnabled(True)

        self.verticalLayout_6.addWidget(self.table_view_focused_result)

        self.tab_container_result_views.addTab(self.tab_focused_result, "")
        self.tab_complete_result = QWidget()
        self.tab_complete_result.setObjectName(u"tab_complete_result")
        self.gridLayout_21 = QGridLayout(self.tab_complete_result)
        self.gridLayout_21.setSpacing(4)
        self.gridLayout_21.setContentsMargins(7, 7, 7, 7)
        self.gridLayout_21.setObjectName(u"gridLayout_21")
        self.gridLayout_21.setContentsMargins(0, 0, 0, 0)
        self.container_complete_result = QWidget(self.tab_complete_result)
        self.container_complete_result.setObjectName(u"container_complete_result")

        self.gridLayout_21.addWidget(self.container_complete_result, 0, 0, 1, 1)

        self.tab_container_result_views.addTab(self.tab_complete_result, "")

        self.gridLayout_11.addWidget(self.tab_container_result_views, 1, 0, 4, 3)


        self.gridLayout_8.addWidget(self.container_results_table, 0, 0, 2, 2)

        self.gridLayout_8.setColumnStretch(0, 2)

        self.gridLayout_12.addWidget(self.container_results, 4, 0, 2, 2)

        icon22 = QIcon()
        icon22.addFile(u":/material-symbols/table_chart_view_FILL0_wght400_GRAD0_opsz24.png", QSize(), QIcon.Normal, QIcon.Off)
        self.tabs_main.addTab(self.tab_results, icon22, "")
        self.tab_analysis = QWidget()
        self.tab_analysis.setObjectName(u"tab_analysis")
        self.gridLayout_16 = QGridLayout(self.tab_analysis)
        self.gridLayout_16.setSpacing(4)
        self.gridLayout_16.setContentsMargins(7, 7, 7, 7)
        self.gridLayout_16.setObjectName(u"gridLayout_16")
        self.container_analysis_tab = QWidget(self.tab_analysis)
        self.container_analysis_tab.setObjectName(u"container_analysis_tab")

        self.gridLayout_16.addWidget(self.container_analysis_tab, 0, 0, 1, 1)

        self.tabs_main.addTab(self.tab_analysis, icon14, "")

        self._2.addWidget(self.tabs_main)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 2348, 22))
        self.menubar.setNativeMenuBar(True)
        self.menu_editing_tools = QMenu(self.menubar)
        self.menu_editing_tools.setObjectName(u"menu_editing_tools")
        self.menu_editing_tools.setToolTipsVisible(True)
        self.menu_file = QMenu(self.menubar)
        self.menu_file.setObjectName(u"menu_file")
        self.menu_info = QMenu(self.menubar)
        self.menu_info.setObjectName(u"menu_info")
        self.menu_debug = QMenu(self.menubar)
        self.menu_debug.setObjectName(u"menu_debug")
        MainWindow.setMenuBar(self.menubar)
        self.toolbar = QToolBar(MainWindow)
        self.toolbar.setObjectName(u"toolbar")
        self.toolbar.setMovable(False)
        self.toolbar.setAllowedAreas(Qt.AllToolBarAreas)
        self.toolbar.setIconSize(QSize(16, 16))
        self.toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.toolbar.setFloatable(False)
        MainWindow.addToolBar(Qt.TopToolBarArea, self.toolbar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.dock_widget_sidebar = QDockWidget(MainWindow)
        self.dock_widget_sidebar.setObjectName(u"dock_widget_sidebar")
        self.dock_widget_sidebar.setFeatures(QDockWidget.DockWidgetFloatable|QDockWidget.DockWidgetMovable)
        self.dock_widget_sidebar.setAllowedAreas(Qt.LeftDockWidgetArea|Qt.RightDockWidgetArea)
        self.sidebar_dock_contents = QWidget()
        self.sidebar_dock_contents.setObjectName(u"sidebar_dock_contents")
        self.gridLayout_17 = QGridLayout(self.sidebar_dock_contents)
        self.gridLayout_17.setSpacing(4)
        self.gridLayout_17.setContentsMargins(7, 7, 7, 7)
        self.gridLayout_17.setObjectName(u"gridLayout_17")
        self.gridLayout_17.setContentsMargins(0, -1, 0, -1)
        self.sidebar = QStackedWidget(self.sidebar_dock_contents)
        self.sidebar.setObjectName(u"sidebar")
        self.sidebar.setFrameShape(QFrame.StyledPanel)
        self.sidebar_page_data = QWidget()
        self.sidebar_page_data.setObjectName(u"sidebar_page_data")
        self.gridLayout = QGridLayout(self.sidebar_page_data)
        self.gridLayout.setSpacing(4)
        self.gridLayout.setContentsMargins(7, 7, 7, 7)
        self.gridLayout.setObjectName(u"gridLayout")
        self.label_4 = QLabel(self.sidebar_page_data)
        self.label_4.setObjectName(u"label_4")

        self.gridLayout.addWidget(self.label_4, 0, 0, 1, 1)

        self.label_3 = QLabel(self.sidebar_page_data)
        self.label_3.setObjectName(u"label_3")
        sizePolicy4 = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.label_3.sizePolicy().hasHeightForWidth())
        self.label_3.setSizePolicy(sizePolicy4)
        self.label_3.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)

        self.gridLayout.addWidget(self.label_3, 2, 0, 1, 1)

        self.btn_info_data_selection = QPushButton(self.sidebar_page_data)
        self.btn_info_data_selection.setObjectName(u"btn_info_data_selection")
        self.btn_info_data_selection.setIcon(icon11)
        self.btn_info_data_selection.setFlat(True)

        self.gridLayout.addWidget(self.btn_info_data_selection, 0, 1, 1, 1, Qt.AlignRight)

        self.container_data_selection = QFrame(self.sidebar_page_data)
        self.container_data_selection.setObjectName(u"container_data_selection")
        self.container_data_selection.setFrameShape(QFrame.StyledPanel)
        self.gridLayout_15 = QGridLayout(self.container_data_selection)
        self.gridLayout_15.setSpacing(4)
        self.gridLayout_15.setContentsMargins(7, 7, 7, 7)
        self.gridLayout_15.setObjectName(u"gridLayout_15")
        self.container_data_required_info = QWidget(self.container_data_selection)
        self.container_data_required_info.setObjectName(u"container_data_required_info")
        self.formLayout = QFormLayout(self.container_data_required_info)
        self.formLayout.setSpacing(4)
        self.formLayout.setContentsMargins(7, 7, 7, 7)
        self.formLayout.setObjectName(u"formLayout")
        self.formLayout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.label_23 = QLabel(self.container_data_required_info)
        self.label_23.setObjectName(u"label_23")
        sizePolicy4.setHeightForWidth(self.label_23.sizePolicy().hasHeightForWidth())
        self.label_23.setSizePolicy(sizePolicy4)

        self.formLayout.setWidget(0, QFormLayout.LabelRole, self.label_23)

        self.combo_box_signal_column = QComboBox(self.container_data_required_info)
        self.combo_box_signal_column.setObjectName(u"combo_box_signal_column")
        self.combo_box_signal_column.setInsertPolicy(QComboBox.InsertAlphabetically)

        self.formLayout.setWidget(0, QFormLayout.FieldRole, self.combo_box_signal_column)

        self.sampleRateLabel = QLabel(self.container_data_required_info)
        self.sampleRateLabel.setObjectName(u"sampleRateLabel")

        self.formLayout.setWidget(1, QFormLayout.LabelRole, self.sampleRateLabel)

        self.spin_box_sample_rate = QSpinBox(self.container_data_required_info)
        self.spin_box_sample_rate.setObjectName(u"spin_box_sample_rate")
        self.spin_box_sample_rate.setMaximum(9999)

        self.formLayout.setWidget(1, QFormLayout.FieldRole, self.spin_box_sample_rate)


        self.gridLayout_15.addWidget(self.container_data_required_info, 2, 0, 1, 3)

        self.btn_select_file = QPushButton(self.container_data_selection)
        self.btn_select_file.setObjectName(u"btn_select_file")
        sizePolicy5 = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy5.setHorizontalStretch(0)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(self.btn_select_file.sizePolicy().hasHeightForWidth())
        self.btn_select_file.setSizePolicy(sizePolicy5)
        font = QFont()
        font.setPointSize(9)
        font.setBold(False)
        self.btn_select_file.setFont(font)

        self.gridLayout_15.addWidget(self.btn_select_file, 0, 0, 1, 1)

        self.btn_load_selection = FeedbackButton(self.container_data_selection)
        self.btn_load_selection.setObjectName(u"btn_load_selection")
        self.btn_load_selection.setEnabled(False)
        sizePolicy3.setHeightForWidth(self.btn_load_selection.sizePolicy().hasHeightForWidth())
        self.btn_load_selection.setSizePolicy(sizePolicy3)
        self.btn_load_selection.setMinimumSize(QSize(0, 50))
        font1 = QFont()
        font1.setPointSize(14)
        font1.setBold(True)
        self.btn_load_selection.setFont(font1)

        self.gridLayout_15.addWidget(self.btn_load_selection, 3, 0, 1, 3)

        self.line = QFrame(self.container_data_selection)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)

        self.gridLayout_15.addWidget(self.line, 1, 0, 1, 3)

        self.line_edit_active_file = QLineEdit(self.container_data_selection)
        self.line_edit_active_file.setObjectName(u"line_edit_active_file")
        self.line_edit_active_file.setFont(font)
        self.line_edit_active_file.setReadOnly(True)

        self.gridLayout_15.addWidget(self.line_edit_active_file, 0, 1, 1, 2)


        self.gridLayout.addWidget(self.container_data_selection, 1, 0, 1, 2)

        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout.addItem(self.verticalSpacer_2, 4, 0, 1, 2)

        self.container_file_info = QFrame(self.sidebar_page_data)
        self.container_file_info.setObjectName(u"container_file_info")
        self.container_file_info.setEnabled(True)
        sizePolicy6 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        sizePolicy6.setHorizontalStretch(0)
        sizePolicy6.setVerticalStretch(0)
        sizePolicy6.setHeightForWidth(self.container_file_info.sizePolicy().hasHeightForWidth())
        self.container_file_info.setSizePolicy(sizePolicy6)
        self.container_file_info.setFrameShape(QFrame.StyledPanel)
        self.formLayout_4 = QFormLayout(self.container_file_info)
        self.formLayout_4.setSpacing(4)
        self.formLayout_4.setContentsMargins(7, 7, 7, 7)
        self.formLayout_4.setObjectName(u"formLayout_4")
        self.formLayout_4.setRowWrapPolicy(QFormLayout.WrapAllRows)
        self.label_13 = QLabel(self.container_file_info)
        self.label_13.setObjectName(u"label_13")

        self.formLayout_4.setWidget(0, QFormLayout.LabelRole, self.label_13)

        self.date_edit_file_info = QDateEdit(self.container_file_info)
        self.date_edit_file_info.setObjectName(u"date_edit_file_info")
        self.date_edit_file_info.setWrapping(False)
        self.date_edit_file_info.setTime(QTime(1, 0, 0))
        self.date_edit_file_info.setMinimumDateTime(QDateTime(QDate(1970, 1, 1), QTime(1, 0, 0)))
        self.date_edit_file_info.setMinimumDate(QDate(1970, 1, 1))
        self.date_edit_file_info.setCalendarPopup(True)
        self.date_edit_file_info.setTimeSpec(Qt.LocalTime)

        self.formLayout_4.setWidget(0, QFormLayout.FieldRole, self.date_edit_file_info)

        self.label_14 = QLabel(self.container_file_info)
        self.label_14.setObjectName(u"label_14")

        self.formLayout_4.setWidget(1, QFormLayout.LabelRole, self.label_14)

        self.line_edit_subject_id = QLineEdit(self.container_file_info)
        self.line_edit_subject_id.setObjectName(u"line_edit_subject_id")
        self.line_edit_subject_id.setClearButtonEnabled(True)

        self.formLayout_4.setWidget(1, QFormLayout.FieldRole, self.line_edit_subject_id)

        self.label_15 = QLabel(self.container_file_info)
        self.label_15.setObjectName(u"label_15")

        self.formLayout_4.setWidget(2, QFormLayout.LabelRole, self.label_15)

        self.combo_box_oxygen_condition = ComboBox(self.container_file_info)
        self.combo_box_oxygen_condition.setObjectName(u"combo_box_oxygen_condition")

        self.formLayout_4.setWidget(2, QFormLayout.FieldRole, self.combo_box_oxygen_condition)


        self.gridLayout.addWidget(self.container_file_info, 3, 0, 1, 2)

        self.btn_info_metadata = QPushButton(self.sidebar_page_data)
        self.btn_info_metadata.setObjectName(u"btn_info_metadata")
        self.btn_info_metadata.setIcon(icon11)
        self.btn_info_metadata.setFlat(True)

        self.gridLayout.addWidget(self.btn_info_metadata, 2, 1, 1, 1, Qt.AlignRight)

        self.sidebar.addWidget(self.sidebar_page_data)
        self.sidebar_page_plots = QWidget()
        self.sidebar_page_plots.setObjectName(u"sidebar_page_plots")
        self.gridLayout_6 = QGridLayout(self.sidebar_page_plots)
        self.gridLayout_6.setSpacing(4)
        self.gridLayout_6.setContentsMargins(7, 7, 7, 7)
        self.gridLayout_6.setObjectName(u"gridLayout_6")
        self.btn_apply_filter = FeedbackButton(self.sidebar_page_plots)
        self.btn_apply_filter.setObjectName(u"btn_apply_filter")
        sizePolicy2.setHeightForWidth(self.btn_apply_filter.sizePolicy().hasHeightForWidth())
        self.btn_apply_filter.setSizePolicy(sizePolicy2)
        font2 = QFont()
        font2.setPointSize(11)
        font2.setBold(True)
        self.btn_apply_filter.setFont(font2)
        self.btn_apply_filter.setLocale(QLocale(QLocale.English, QLocale.Germany))

        self.gridLayout_6.addWidget(self.btn_apply_filter, 3, 0, 1, 3)

        self.container_signal_filtering_sidebar = QWidget(self.sidebar_page_plots)
        self.container_signal_filtering_sidebar.setObjectName(u"container_signal_filtering_sidebar")
        self.gridLayout_2 = QGridLayout(self.container_signal_filtering_sidebar)
        self.gridLayout_2.setSpacing(4)
        self.gridLayout_2.setContentsMargins(7, 7, 7, 7)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.gridLayout_2.setContentsMargins(1, 1, 1, 1)
        self.container_standardize = QFrame(self.container_signal_filtering_sidebar)
        self.container_standardize.setObjectName(u"container_standardize")
        font3 = QFont()
        font3.setBold(False)
        self.container_standardize.setFont(font3)
        self.container_standardize.setFrameShape(QFrame.StyledPanel)
        self.gridLayout_9 = QGridLayout(self.container_standardize)
        self.gridLayout_9.setSpacing(4)
        self.gridLayout_9.setContentsMargins(7, 7, 7, 7)
        self.gridLayout_9.setObjectName(u"gridLayout_9")
        self.combo_box_scale_method = ComboBox(self.container_standardize)
        self.combo_box_scale_method.setObjectName(u"combo_box_scale_method")

        self.gridLayout_9.addWidget(self.combo_box_scale_method, 0, 1, 1, 1)

        self.container_scale_window_inputs = QGroupBox(self.container_standardize)
        self.container_scale_window_inputs.setObjectName(u"container_scale_window_inputs")
        sizePolicy6.setHeightForWidth(self.container_scale_window_inputs.sizePolicy().hasHeightForWidth())
        self.container_scale_window_inputs.setSizePolicy(sizePolicy6)
        self.container_scale_window_inputs.setCheckable(True)
        self.container_scale_window_inputs.setChecked(True)
        self.horizontalLayout_2 = QHBoxLayout(self.container_scale_window_inputs)
        self.horizontalLayout_2.setSpacing(4)
        self.horizontalLayout_2.setContentsMargins(7, 7, 7, 7)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.label_2 = QLabel(self.container_scale_window_inputs)
        self.label_2.setObjectName(u"label_2")
        sizePolicy7 = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        sizePolicy7.setHorizontalStretch(0)
        sizePolicy7.setVerticalStretch(0)
        sizePolicy7.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy7)

        self.horizontalLayout_2.addWidget(self.label_2)

        self.slider_scale_window_size = QSlider(self.container_scale_window_inputs)
        self.slider_scale_window_size.setObjectName(u"slider_scale_window_size")
        sizePolicy6.setHeightForWidth(self.slider_scale_window_size.sizePolicy().hasHeightForWidth())
        self.slider_scale_window_size.setSizePolicy(sizePolicy6)
        self.slider_scale_window_size.setMinimum(5)
        self.slider_scale_window_size.setMaximum(10000)
        self.slider_scale_window_size.setValue(2000)
        self.slider_scale_window_size.setOrientation(Qt.Horizontal)

        self.horizontalLayout_2.addWidget(self.slider_scale_window_size)

        self.spin_box_scale_window_size = QSpinBox(self.container_scale_window_inputs)
        self.spin_box_scale_window_size.setObjectName(u"spin_box_scale_window_size")
        sizePolicy8 = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        sizePolicy8.setHorizontalStretch(0)
        sizePolicy8.setVerticalStretch(0)
        sizePolicy8.setHeightForWidth(self.spin_box_scale_window_size.sizePolicy().hasHeightForWidth())
        self.spin_box_scale_window_size.setSizePolicy(sizePolicy8)
        self.spin_box_scale_window_size.setCorrectionMode(QAbstractSpinBox.CorrectToNearestValue)
        self.spin_box_scale_window_size.setMinimum(5)
        self.spin_box_scale_window_size.setMaximum(10000)
        self.spin_box_scale_window_size.setValue(2000)

        self.horizontalLayout_2.addWidget(self.spin_box_scale_window_size)


        self.gridLayout_9.addWidget(self.container_scale_window_inputs, 1, 0, 1, 2)

        self.label = QLabel(self.container_standardize)
        self.label.setObjectName(u"label")
        sizePolicy6.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy6)

        self.gridLayout_9.addWidget(self.label, 0, 0, 1, 1)


        self.gridLayout_2.addWidget(self.container_standardize, 5, 0, 1, 2)

        self.label_12 = QLabel(self.container_signal_filtering_sidebar)
        self.label_12.setObjectName(u"label_12")
        sizePolicy4.setHeightForWidth(self.label_12.sizePolicy().hasHeightForWidth())
        self.label_12.setSizePolicy(sizePolicy4)
        self.label_12.setFont(font)

        self.gridLayout_2.addWidget(self.label_12, 2, 0, 1, 1)

        self.label_16 = QLabel(self.container_signal_filtering_sidebar)
        self.label_16.setObjectName(u"label_16")

        self.gridLayout_2.addWidget(self.label_16, 4, 0, 1, 1)

        self.combo_box_preprocess_pipeline = ComboBox(self.container_signal_filtering_sidebar)
        self.combo_box_preprocess_pipeline.setObjectName(u"combo_box_preprocess_pipeline")
        self.combo_box_preprocess_pipeline.setFont(font)

        self.gridLayout_2.addWidget(self.combo_box_preprocess_pipeline, 2, 1, 1, 1)

        self.container_custom_filter_inputs = QFrame(self.container_signal_filtering_sidebar)
        self.container_custom_filter_inputs.setObjectName(u"container_custom_filter_inputs")
        sizePolicy6.setHeightForWidth(self.container_custom_filter_inputs.sizePolicy().hasHeightForWidth())
        self.container_custom_filter_inputs.setSizePolicy(sizePolicy6)
        self.container_custom_filter_inputs.setFrameShape(QFrame.StyledPanel)
        self.gridLayout_10 = QGridLayout(self.container_custom_filter_inputs)
        self.gridLayout_10.setSpacing(4)
        self.gridLayout_10.setContentsMargins(7, 7, 7, 7)
        self.gridLayout_10.setObjectName(u"gridLayout_10")
        self.container_order_inputs = QWidget(self.container_custom_filter_inputs)
        self.container_order_inputs.setObjectName(u"container_order_inputs")
        sizePolicy.setHeightForWidth(self.container_order_inputs.sizePolicy().hasHeightForWidth())
        self.container_order_inputs.setSizePolicy(sizePolicy)
        self.container_order_inputs.setFont(font)
        self.horizontalLayout_5 = QHBoxLayout(self.container_order_inputs)
        self.horizontalLayout_5.setSpacing(4)
        self.horizontalLayout_5.setContentsMargins(7, 7, 7, 7)
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.horizontalLayout_5.setContentsMargins(1, -1, 1, -1)
        self.label_10 = QLabel(self.container_order_inputs)
        self.label_10.setObjectName(u"label_10")
        sizePolicy7.setHeightForWidth(self.label_10.sizePolicy().hasHeightForWidth())
        self.label_10.setSizePolicy(sizePolicy7)
        self.label_10.setFont(font)

        self.horizontalLayout_5.addWidget(self.label_10)

        self.slider_order = QSlider(self.container_order_inputs)
        self.slider_order.setObjectName(u"slider_order")
        sizePolicy6.setHeightForWidth(self.slider_order.sizePolicy().hasHeightForWidth())
        self.slider_order.setSizePolicy(sizePolicy6)
        self.slider_order.setFont(font)
        self.slider_order.setMinimum(2)
        self.slider_order.setMaximum(10)
        self.slider_order.setOrientation(Qt.Horizontal)
        self.slider_order.setTickPosition(QSlider.TicksBelow)
        self.slider_order.setTickInterval(1)

        self.horizontalLayout_5.addWidget(self.slider_order)

        self.spin_box_order = QSpinBox(self.container_order_inputs)
        self.spin_box_order.setObjectName(u"spin_box_order")
        sizePolicy4.setHeightForWidth(self.spin_box_order.sizePolicy().hasHeightForWidth())
        self.spin_box_order.setSizePolicy(sizePolicy4)
        self.spin_box_order.setMinimumSize(QSize(45, 0))
        self.spin_box_order.setFont(font)
        self.spin_box_order.setFrame(True)
        self.spin_box_order.setCorrectionMode(QAbstractSpinBox.CorrectToNearestValue)
        self.spin_box_order.setMinimum(2)
        self.spin_box_order.setMaximum(10)

        self.horizontalLayout_5.addWidget(self.spin_box_order)


        self.gridLayout_10.addWidget(self.container_order_inputs, 3, 0, 1, 3)

        self.label_7 = QLabel(self.container_custom_filter_inputs)
        self.label_7.setObjectName(u"label_7")
        sizePolicy7.setHeightForWidth(self.label_7.sizePolicy().hasHeightForWidth())
        self.label_7.setSizePolicy(sizePolicy7)
        self.label_7.setFont(font)

        self.gridLayout_10.addWidget(self.label_7, 0, 0, 1, 1)

        self.combo_box_filter_method = ComboBox(self.container_custom_filter_inputs)
        self.combo_box_filter_method.setObjectName(u"combo_box_filter_method")
        self.combo_box_filter_method.setFont(font)

        self.gridLayout_10.addWidget(self.combo_box_filter_method, 0, 1, 1, 2)

        self.container_highcut = QWidget(self.container_custom_filter_inputs)
        self.container_highcut.setObjectName(u"container_highcut")
        self.gridLayout_3 = QGridLayout(self.container_highcut)
        self.gridLayout_3.setSpacing(4)
        self.gridLayout_3.setContentsMargins(7, 7, 7, 7)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.gridLayout_3.setContentsMargins(1, 1, 1, 1)
        self.dbl_spin_box_highcut = QDoubleSpinBox(self.container_highcut)
        self.dbl_spin_box_highcut.setObjectName(u"dbl_spin_box_highcut")
        sizePolicy9 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        sizePolicy9.setHorizontalStretch(0)
        sizePolicy9.setVerticalStretch(0)
        sizePolicy9.setHeightForWidth(self.dbl_spin_box_highcut.sizePolicy().hasHeightForWidth())
        self.dbl_spin_box_highcut.setSizePolicy(sizePolicy9)
        self.dbl_spin_box_highcut.setFont(font)
        self.dbl_spin_box_highcut.setFrame(True)
        self.dbl_spin_box_highcut.setCorrectionMode(QAbstractSpinBox.CorrectToNearestValue)
        self.dbl_spin_box_highcut.setDecimals(1)
        self.dbl_spin_box_highcut.setMaximum(1000.000000000000000)
        self.dbl_spin_box_highcut.setSingleStep(0.100000000000000)
        self.dbl_spin_box_highcut.setValue(10.000000000000000)

        self.gridLayout_3.addWidget(self.dbl_spin_box_highcut, 0, 1, 1, 1)

        self.label_9 = QLabel(self.container_highcut)
        self.label_9.setObjectName(u"label_9")
        sizePolicy4.setHeightForWidth(self.label_9.sizePolicy().hasHeightForWidth())
        self.label_9.setSizePolicy(sizePolicy4)
        self.label_9.setFont(font)
        self.label_9.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)

        self.gridLayout_3.addWidget(self.label_9, 0, 0, 1, 1)


        self.gridLayout_10.addWidget(self.container_highcut, 1, 1, 1, 1)

        self.container_lowcut = QWidget(self.container_custom_filter_inputs)
        self.container_lowcut.setObjectName(u"container_lowcut")
        self.gridLayout_5 = QGridLayout(self.container_lowcut)
        self.gridLayout_5.setSpacing(4)
        self.gridLayout_5.setContentsMargins(7, 7, 7, 7)
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.gridLayout_5.setContentsMargins(1, 1, 1, 1)
        self.label_8 = QLabel(self.container_lowcut)
        self.label_8.setObjectName(u"label_8")
        sizePolicy4.setHeightForWidth(self.label_8.sizePolicy().hasHeightForWidth())
        self.label_8.setSizePolicy(sizePolicy4)
        self.label_8.setFont(font)

        self.gridLayout_5.addWidget(self.label_8, 0, 0, 1, 1)

        self.dbl_spin_box_lowcut = QDoubleSpinBox(self.container_lowcut)
        self.dbl_spin_box_lowcut.setObjectName(u"dbl_spin_box_lowcut")
        sizePolicy9.setHeightForWidth(self.dbl_spin_box_lowcut.sizePolicy().hasHeightForWidth())
        self.dbl_spin_box_lowcut.setSizePolicy(sizePolicy9)
        self.dbl_spin_box_lowcut.setFont(font)
        self.dbl_spin_box_lowcut.setWrapping(False)
        self.dbl_spin_box_lowcut.setFrame(True)
        self.dbl_spin_box_lowcut.setCorrectionMode(QAbstractSpinBox.CorrectToNearestValue)
        self.dbl_spin_box_lowcut.setDecimals(1)
        self.dbl_spin_box_lowcut.setMaximum(1000.000000000000000)
        self.dbl_spin_box_lowcut.setSingleStep(0.100000000000000)
        self.dbl_spin_box_lowcut.setStepType(QAbstractSpinBox.AdaptiveDecimalStepType)
        self.dbl_spin_box_lowcut.setValue(0.500000000000000)

        self.gridLayout_5.addWidget(self.dbl_spin_box_lowcut, 0, 1, 1, 1)


        self.gridLayout_10.addWidget(self.container_lowcut, 1, 0, 1, 1)

        self.container_window_size = QWidget(self.container_custom_filter_inputs)
        self.container_window_size.setObjectName(u"container_window_size")
        self.container_window_size.setFont(font)
        self.horizontalLayout_3 = QHBoxLayout(self.container_window_size)
        self.horizontalLayout_3.setSpacing(4)
        self.horizontalLayout_3.setContentsMargins(7, 7, 7, 7)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_3.setContentsMargins(1, -1, 1, -1)
        self.label_11 = QLabel(self.container_window_size)
        self.label_11.setObjectName(u"label_11")
        sizePolicy7.setHeightForWidth(self.label_11.sizePolicy().hasHeightForWidth())
        self.label_11.setSizePolicy(sizePolicy7)
        self.label_11.setFont(font)
        self.label_11.setScaledContents(True)

        self.horizontalLayout_3.addWidget(self.label_11)

        self.slider_window_size = QSlider(self.container_window_size)
        self.slider_window_size.setObjectName(u"slider_window_size")
        sizePolicy6.setHeightForWidth(self.slider_window_size.sizePolicy().hasHeightForWidth())
        self.slider_window_size.setSizePolicy(sizePolicy6)
        self.slider_window_size.setFont(font)
        self.slider_window_size.setMinimum(5)
        self.slider_window_size.setMaximum(3333)
        self.slider_window_size.setSingleStep(1)
        self.slider_window_size.setValue(250)
        self.slider_window_size.setOrientation(Qt.Horizontal)

        self.horizontalLayout_3.addWidget(self.slider_window_size)

        self.spin_box_window_size = QSpinBox(self.container_window_size)
        self.spin_box_window_size.setObjectName(u"spin_box_window_size")
        sizePolicy4.setHeightForWidth(self.spin_box_window_size.sizePolicy().hasHeightForWidth())
        self.spin_box_window_size.setSizePolicy(sizePolicy4)
        self.spin_box_window_size.setMinimumSize(QSize(45, 0))
        self.spin_box_window_size.setFont(font)
        self.spin_box_window_size.setFrame(True)
        self.spin_box_window_size.setAccelerated(True)
        self.spin_box_window_size.setCorrectionMode(QAbstractSpinBox.CorrectToNearestValue)
        self.spin_box_window_size.setMinimum(5)
        self.spin_box_window_size.setMaximum(3333)
        self.spin_box_window_size.setSingleStep(1)
        self.spin_box_window_size.setValue(250)

        self.horizontalLayout_3.addWidget(self.spin_box_window_size)


        self.gridLayout_10.addWidget(self.container_window_size, 5, 0, 1, 3)

        self.container_powerline = QWidget(self.container_custom_filter_inputs)
        self.container_powerline.setObjectName(u"container_powerline")
        self.gridLayout_14 = QGridLayout(self.container_powerline)
        self.gridLayout_14.setSpacing(4)
        self.gridLayout_14.setContentsMargins(7, 7, 7, 7)
        self.gridLayout_14.setObjectName(u"gridLayout_14")
        self.gridLayout_14.setContentsMargins(1, 1, 1, 1)
        self.label_20 = QLabel(self.container_powerline)
        self.label_20.setObjectName(u"label_20")
        sizePolicy4.setHeightForWidth(self.label_20.sizePolicy().hasHeightForWidth())
        self.label_20.setSizePolicy(sizePolicy4)

        self.gridLayout_14.addWidget(self.label_20, 0, 0, 1, 1)

        self.dbl_spin_box_powerline = QDoubleSpinBox(self.container_powerline)
        self.dbl_spin_box_powerline.setObjectName(u"dbl_spin_box_powerline")
        self.dbl_spin_box_powerline.setDecimals(1)
        self.dbl_spin_box_powerline.setMaximum(500.000000000000000)
        self.dbl_spin_box_powerline.setValue(50.000000000000000)

        self.gridLayout_14.addWidget(self.dbl_spin_box_powerline, 0, 1, 1, 1)


        self.gridLayout_10.addWidget(self.container_powerline, 1, 2, 1, 1)


        self.gridLayout_2.addWidget(self.container_custom_filter_inputs, 3, 0, 1, 2)

        self.btn_info_standardize = QPushButton(self.container_signal_filtering_sidebar)
        self.btn_info_standardize.setObjectName(u"btn_info_standardize")
        self.btn_info_standardize.setIcon(icon11)
        self.btn_info_standardize.setFlat(True)

        self.gridLayout_2.addWidget(self.btn_info_standardize, 4, 1, 1, 1, Qt.AlignRight)

        self.label_5 = QLabel(self.container_signal_filtering_sidebar)
        self.label_5.setObjectName(u"label_5")
        sizePolicy4.setHeightForWidth(self.label_5.sizePolicy().hasHeightForWidth())
        self.label_5.setSizePolicy(sizePolicy4)

        self.gridLayout_2.addWidget(self.label_5, 1, 0, 1, 1)

        self.btn_info_preprocess = QPushButton(self.container_signal_filtering_sidebar)
        self.btn_info_preprocess.setObjectName(u"btn_info_preprocess")
        self.btn_info_preprocess.setIcon(icon11)
        self.btn_info_preprocess.setFlat(True)

        self.gridLayout_2.addWidget(self.btn_info_preprocess, 1, 1, 1, 1, Qt.AlignRight)


        self.gridLayout_6.addWidget(self.container_signal_filtering_sidebar, 1, 0, 1, 3)

        self.container_peak_detection_sidebar = QWidget(self.sidebar_page_plots)
        self.container_peak_detection_sidebar.setObjectName(u"container_peak_detection_sidebar")
        self.layout_container_peak_detection_sidebar = QGridLayout(self.container_peak_detection_sidebar)
        self.layout_container_peak_detection_sidebar.setSpacing(4)
        self.layout_container_peak_detection_sidebar.setContentsMargins(7, 7, 7, 7)
        self.layout_container_peak_detection_sidebar.setObjectName(u"layout_container_peak_detection_sidebar")
        self.layout_container_peak_detection_sidebar.setContentsMargins(0, 0, 0, 0)
        self.stacked_peak_parameters = QStackedWidget(self.container_peak_detection_sidebar)
        self.stacked_peak_parameters.setObjectName(u"stacked_peak_parameters")
        self.stacked_peak_parameters.setFrameShape(QFrame.StyledPanel)
        self.page_peak_elgendi_ppg = QWidget()
        self.page_peak_elgendi_ppg.setObjectName(u"page_peak_elgendi_ppg")
        self.formLayout_3 = QFormLayout(self.page_peak_elgendi_ppg)
        self.formLayout_3.setSpacing(4)
        self.formLayout_3.setContentsMargins(7, 7, 7, 7)
        self.formLayout_3.setObjectName(u"formLayout_3")
        self.label_peak_window = QLabel(self.page_peak_elgendi_ppg)
        self.label_peak_window.setObjectName(u"label_peak_window")
        font4 = QFont()
        font4.setBold(True)
        font4.setItalic(False)
        self.label_peak_window.setFont(font4)

        self.formLayout_3.setWidget(1, QFormLayout.LabelRole, self.label_peak_window)

        self.peak_elgendi_ppg_peakwindow = QDoubleSpinBox(self.page_peak_elgendi_ppg)
        self.peak_elgendi_ppg_peakwindow.setObjectName(u"peak_elgendi_ppg_peakwindow")
        self.peak_elgendi_ppg_peakwindow.setAccelerated(True)
        self.peak_elgendi_ppg_peakwindow.setDecimals(3)
        self.peak_elgendi_ppg_peakwindow.setMinimum(0.050000000000000)
        self.peak_elgendi_ppg_peakwindow.setMaximum(5.000000000000000)
        self.peak_elgendi_ppg_peakwindow.setSingleStep(0.001000000000000)
        self.peak_elgendi_ppg_peakwindow.setStepType(QAbstractSpinBox.AdaptiveDecimalStepType)
        self.peak_elgendi_ppg_peakwindow.setValue(0.111000000000000)

        self.formLayout_3.setWidget(1, QFormLayout.FieldRole, self.peak_elgendi_ppg_peakwindow)

        self.beatWindowLabel = QLabel(self.page_peak_elgendi_ppg)
        self.beatWindowLabel.setObjectName(u"beatWindowLabel")
        self.beatWindowLabel.setFont(font4)

        self.formLayout_3.setWidget(2, QFormLayout.LabelRole, self.beatWindowLabel)

        self.peak_elgendi_ppg_beatwindow = QDoubleSpinBox(self.page_peak_elgendi_ppg)
        self.peak_elgendi_ppg_beatwindow.setObjectName(u"peak_elgendi_ppg_beatwindow")
        self.peak_elgendi_ppg_beatwindow.setDecimals(3)
        self.peak_elgendi_ppg_beatwindow.setMinimum(0.100000000000000)
        self.peak_elgendi_ppg_beatwindow.setMaximum(5.000000000000000)
        self.peak_elgendi_ppg_beatwindow.setSingleStep(0.001000000000000)
        self.peak_elgendi_ppg_beatwindow.setStepType(QAbstractSpinBox.AdaptiveDecimalStepType)
        self.peak_elgendi_ppg_beatwindow.setValue(0.667000000000000)

        self.formLayout_3.setWidget(2, QFormLayout.FieldRole, self.peak_elgendi_ppg_beatwindow)

        self.beatOffsetLabel = QLabel(self.page_peak_elgendi_ppg)
        self.beatOffsetLabel.setObjectName(u"beatOffsetLabel")
        self.beatOffsetLabel.setFont(font4)

        self.formLayout_3.setWidget(3, QFormLayout.LabelRole, self.beatOffsetLabel)

        self.peak_elgendi_ppg_beatoffset = QDoubleSpinBox(self.page_peak_elgendi_ppg)
        self.peak_elgendi_ppg_beatoffset.setObjectName(u"peak_elgendi_ppg_beatoffset")
        self.peak_elgendi_ppg_beatoffset.setDecimals(2)
        self.peak_elgendi_ppg_beatoffset.setMaximum(1.000000000000000)
        self.peak_elgendi_ppg_beatoffset.setSingleStep(0.010000000000000)
        self.peak_elgendi_ppg_beatoffset.setStepType(QAbstractSpinBox.AdaptiveDecimalStepType)
        self.peak_elgendi_ppg_beatoffset.setValue(0.020000000000000)

        self.formLayout_3.setWidget(3, QFormLayout.FieldRole, self.peak_elgendi_ppg_beatoffset)

        self.minimumDelayLabel = QLabel(self.page_peak_elgendi_ppg)
        self.minimumDelayLabel.setObjectName(u"minimumDelayLabel")
        self.minimumDelayLabel.setFont(font4)

        self.formLayout_3.setWidget(4, QFormLayout.LabelRole, self.minimumDelayLabel)

        self.peak_elgendi_ppg_min_delay = QDoubleSpinBox(self.page_peak_elgendi_ppg)
        self.peak_elgendi_ppg_min_delay.setObjectName(u"peak_elgendi_ppg_min_delay")
        self.peak_elgendi_ppg_min_delay.setDecimals(2)
        self.peak_elgendi_ppg_min_delay.setMaximum(10.000000000000000)
        self.peak_elgendi_ppg_min_delay.setSingleStep(0.010000000000000)
        self.peak_elgendi_ppg_min_delay.setStepType(QAbstractSpinBox.AdaptiveDecimalStepType)
        self.peak_elgendi_ppg_min_delay.setValue(0.300000000000000)

        self.formLayout_3.setWidget(4, QFormLayout.FieldRole, self.peak_elgendi_ppg_min_delay)

        self.peak_elgendi_ppg_info = QTextBrowser(self.page_peak_elgendi_ppg)
        self.peak_elgendi_ppg_info.setObjectName(u"peak_elgendi_ppg_info")
        sizePolicy10 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        sizePolicy10.setHorizontalStretch(0)
        sizePolicy10.setVerticalStretch(0)
        sizePolicy10.setHeightForWidth(self.peak_elgendi_ppg_info.sizePolicy().hasHeightForWidth())
        self.peak_elgendi_ppg_info.setSizePolicy(sizePolicy10)
        self.peak_elgendi_ppg_info.setMaximumSize(QSize(16777215, 100))
        self.peak_elgendi_ppg_info.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)

        self.formLayout_3.setWidget(0, QFormLayout.SpanningRole, self.peak_elgendi_ppg_info)

        self.stacked_peak_parameters.addWidget(self.page_peak_elgendi_ppg)
        self.page_peak_local_max = QWidget()
        self.page_peak_local_max.setObjectName(u"page_peak_local_max")
        self.formLayout_5 = QFormLayout(self.page_peak_local_max)
        self.formLayout_5.setSpacing(4)
        self.formLayout_5.setContentsMargins(7, 7, 7, 7)
        self.formLayout_5.setObjectName(u"formLayout_5")
        self.label_22 = QLabel(self.page_peak_local_max)
        self.label_22.setObjectName(u"label_22")
        font5 = QFont()
        font5.setBold(True)
        self.label_22.setFont(font5)

        self.formLayout_5.setWidget(1, QFormLayout.LabelRole, self.label_22)

        self.peak_local_max_radius = QSpinBox(self.page_peak_local_max)
        self.peak_local_max_radius.setObjectName(u"peak_local_max_radius")
        self.peak_local_max_radius.setAccelerated(True)
        self.peak_local_max_radius.setMinimum(5)
        self.peak_local_max_radius.setMaximum(9999)
        self.peak_local_max_radius.setStepType(QAbstractSpinBox.AdaptiveDecimalStepType)
        self.peak_local_max_radius.setValue(111)

        self.formLayout_5.setWidget(1, QFormLayout.FieldRole, self.peak_local_max_radius)

        self.peak_local_max_info = QTextBrowser(self.page_peak_local_max)
        self.peak_local_max_info.setObjectName(u"peak_local_max_info")
        sizePolicy2.setHeightForWidth(self.peak_local_max_info.sizePolicy().hasHeightForWidth())
        self.peak_local_max_info.setSizePolicy(sizePolicy2)
        self.peak_local_max_info.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.peak_local_max_info.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)

        self.formLayout_5.setWidget(0, QFormLayout.SpanningRole, self.peak_local_max_info)

        self.stacked_peak_parameters.addWidget(self.page_peak_local_max)
        self.page_peak_neurokit2 = QWidget()
        self.page_peak_neurokit2.setObjectName(u"page_peak_neurokit2")
        self.formLayout_6 = QFormLayout(self.page_peak_neurokit2)
        self.formLayout_6.setSpacing(4)
        self.formLayout_6.setContentsMargins(7, 7, 7, 7)
        self.formLayout_6.setObjectName(u"formLayout_6")
        self.formLayout_6.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.algorithmLabel = QLabel(self.page_peak_neurokit2)
        self.algorithmLabel.setObjectName(u"algorithmLabel")
        self.algorithmLabel.setFont(font5)

        self.formLayout_6.setWidget(1, QFormLayout.LabelRole, self.algorithmLabel)

        self.peak_neurokit2_algorithm_used = ComboBox(self.page_peak_neurokit2)
        self.peak_neurokit2_algorithm_used.setObjectName(u"peak_neurokit2_algorithm_used")

        self.formLayout_6.setWidget(1, QFormLayout.FieldRole, self.peak_neurokit2_algorithm_used)

        self.smoothingWindowLabel = QLabel(self.page_peak_neurokit2)
        self.smoothingWindowLabel.setObjectName(u"smoothingWindowLabel")
        self.smoothingWindowLabel.setFont(font5)

        self.formLayout_6.setWidget(2, QFormLayout.LabelRole, self.smoothingWindowLabel)

        self.peak_neurokit2_smoothwindow = QDoubleSpinBox(self.page_peak_neurokit2)
        self.peak_neurokit2_smoothwindow.setObjectName(u"peak_neurokit2_smoothwindow")
        self.peak_neurokit2_smoothwindow.setMinimum(0.010000000000000)
        self.peak_neurokit2_smoothwindow.setMaximum(10.000000000000000)
        self.peak_neurokit2_smoothwindow.setSingleStep(0.010000000000000)

        self.formLayout_6.setWidget(2, QFormLayout.FieldRole, self.peak_neurokit2_smoothwindow)

        self.label_27 = QLabel(self.page_peak_neurokit2)
        self.label_27.setObjectName(u"label_27")
        self.label_27.setFont(font5)

        self.formLayout_6.setWidget(3, QFormLayout.LabelRole, self.label_27)

        self.peak_neurokit2_avgwindow = QDoubleSpinBox(self.page_peak_neurokit2)
        self.peak_neurokit2_avgwindow.setObjectName(u"peak_neurokit2_avgwindow")
        self.peak_neurokit2_avgwindow.setDecimals(2)
        self.peak_neurokit2_avgwindow.setMinimum(0.010000000000000)
        self.peak_neurokit2_avgwindow.setMaximum(10.000000000000000)
        self.peak_neurokit2_avgwindow.setSingleStep(0.010000000000000)
        self.peak_neurokit2_avgwindow.setValue(0.750000000000000)

        self.formLayout_6.setWidget(3, QFormLayout.FieldRole, self.peak_neurokit2_avgwindow)

        self.label_28 = QLabel(self.page_peak_neurokit2)
        self.label_28.setObjectName(u"label_28")
        self.label_28.setFont(font5)

        self.formLayout_6.setWidget(4, QFormLayout.LabelRole, self.label_28)

        self.peak_neurokit2_gradthreshweight = QDoubleSpinBox(self.page_peak_neurokit2)
        self.peak_neurokit2_gradthreshweight.setObjectName(u"peak_neurokit2_gradthreshweight")
        self.peak_neurokit2_gradthreshweight.setDecimals(1)
        self.peak_neurokit2_gradthreshweight.setMinimum(0.100000000000000)
        self.peak_neurokit2_gradthreshweight.setMaximum(10.000000000000000)
        self.peak_neurokit2_gradthreshweight.setSingleStep(0.100000000000000)
        self.peak_neurokit2_gradthreshweight.setValue(1.500000000000000)

        self.formLayout_6.setWidget(4, QFormLayout.FieldRole, self.peak_neurokit2_gradthreshweight)

        self.label_29 = QLabel(self.page_peak_neurokit2)
        self.label_29.setObjectName(u"label_29")
        self.label_29.setFont(font5)

        self.formLayout_6.setWidget(5, QFormLayout.LabelRole, self.label_29)

        self.peak_neurokit2_minlenweight = QDoubleSpinBox(self.page_peak_neurokit2)
        self.peak_neurokit2_minlenweight.setObjectName(u"peak_neurokit2_minlenweight")
        self.peak_neurokit2_minlenweight.setDecimals(1)
        self.peak_neurokit2_minlenweight.setMinimum(0.100000000000000)
        self.peak_neurokit2_minlenweight.setMaximum(10.000000000000000)
        self.peak_neurokit2_minlenweight.setSingleStep(0.100000000000000)
        self.peak_neurokit2_minlenweight.setValue(0.400000000000000)

        self.formLayout_6.setWidget(5, QFormLayout.FieldRole, self.peak_neurokit2_minlenweight)

        self.label_30 = QLabel(self.page_peak_neurokit2)
        self.label_30.setObjectName(u"label_30")
        self.label_30.setFont(font5)

        self.formLayout_6.setWidget(6, QFormLayout.LabelRole, self.label_30)

        self.peak_neurokit2_mindelay = QDoubleSpinBox(self.page_peak_neurokit2)
        self.peak_neurokit2_mindelay.setObjectName(u"peak_neurokit2_mindelay")
        self.peak_neurokit2_mindelay.setMinimum(0.010000000000000)
        self.peak_neurokit2_mindelay.setMaximum(10.000000000000000)
        self.peak_neurokit2_mindelay.setSingleStep(0.010000000000000)
        self.peak_neurokit2_mindelay.setValue(0.300000000000000)

        self.formLayout_6.setWidget(6, QFormLayout.FieldRole, self.peak_neurokit2_mindelay)

        self.label_31 = QLabel(self.page_peak_neurokit2)
        self.label_31.setObjectName(u"label_31")
        self.label_31.setFont(font5)

        self.formLayout_6.setWidget(7, QFormLayout.LabelRole, self.label_31)

        self.peak_neurokit2_correct_artifacts = QCheckBox(self.page_peak_neurokit2)
        self.peak_neurokit2_correct_artifacts.setObjectName(u"peak_neurokit2_correct_artifacts")

        self.formLayout_6.setWidget(7, QFormLayout.FieldRole, self.peak_neurokit2_correct_artifacts)

        self.peak_neurokit2_info = QTextBrowser(self.page_peak_neurokit2)
        self.peak_neurokit2_info.setObjectName(u"peak_neurokit2_info")
        sizePolicy2.setHeightForWidth(self.peak_neurokit2_info.sizePolicy().hasHeightForWidth())
        self.peak_neurokit2_info.setSizePolicy(sizePolicy2)
        self.peak_neurokit2_info.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)

        self.formLayout_6.setWidget(0, QFormLayout.SpanningRole, self.peak_neurokit2_info)

        self.stacked_peak_parameters.addWidget(self.page_peak_neurokit2)
        self.page_peak_promac = QWidget()
        self.page_peak_promac.setObjectName(u"page_peak_promac")
        self.formLayout_7 = QFormLayout(self.page_peak_promac)
        self.formLayout_7.setSpacing(4)
        self.formLayout_7.setContentsMargins(7, 7, 7, 7)
        self.formLayout_7.setObjectName(u"formLayout_7")
        self.thresholdLabel = QLabel(self.page_peak_promac)
        self.thresholdLabel.setObjectName(u"thresholdLabel")
        self.thresholdLabel.setFont(font5)

        self.formLayout_7.setWidget(1, QFormLayout.LabelRole, self.thresholdLabel)

        self.peak_promac_threshold = QDoubleSpinBox(self.page_peak_promac)
        self.peak_promac_threshold.setObjectName(u"peak_promac_threshold")
        self.peak_promac_threshold.setMaximum(1.000000000000000)
        self.peak_promac_threshold.setSingleStep(0.010000000000000)
        self.peak_promac_threshold.setValue(0.330000000000000)

        self.formLayout_7.setWidget(1, QFormLayout.FieldRole, self.peak_promac_threshold)

        self.qRSComplexSizeLabel = QLabel(self.page_peak_promac)
        self.qRSComplexSizeLabel.setObjectName(u"qRSComplexSizeLabel")
        self.qRSComplexSizeLabel.setFont(font5)

        self.formLayout_7.setWidget(2, QFormLayout.LabelRole, self.qRSComplexSizeLabel)

        self.correctArtifactsLabel_2 = QLabel(self.page_peak_promac)
        self.correctArtifactsLabel_2.setObjectName(u"correctArtifactsLabel_2")
        self.correctArtifactsLabel_2.setFont(font5)

        self.formLayout_7.setWidget(3, QFormLayout.LabelRole, self.correctArtifactsLabel_2)

        self.peak_promac_correct_artifacts = QCheckBox(self.page_peak_promac)
        self.peak_promac_correct_artifacts.setObjectName(u"peak_promac_correct_artifacts")

        self.formLayout_7.setWidget(3, QFormLayout.FieldRole, self.peak_promac_correct_artifacts)

        self.peak_promac_gaussian_sd = QSpinBox(self.page_peak_promac)
        self.peak_promac_gaussian_sd.setObjectName(u"peak_promac_gaussian_sd")
        self.peak_promac_gaussian_sd.setMaximum(100000)
        self.peak_promac_gaussian_sd.setValue(100)

        self.formLayout_7.setWidget(2, QFormLayout.FieldRole, self.peak_promac_gaussian_sd)

        self.peak_promac_info = QTextBrowser(self.page_peak_promac)
        self.peak_promac_info.setObjectName(u"peak_promac_info")
        sizePolicy2.setHeightForWidth(self.peak_promac_info.sizePolicy().hasHeightForWidth())
        self.peak_promac_info.setSizePolicy(sizePolicy2)
        self.peak_promac_info.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)

        self.formLayout_7.setWidget(0, QFormLayout.SpanningRole, self.peak_promac_info)

        self.stacked_peak_parameters.addWidget(self.page_peak_promac)
        self.page_peak_pantompkins = QWidget()
        self.page_peak_pantompkins.setObjectName(u"page_peak_pantompkins")
        self.formLayout_8 = QFormLayout(self.page_peak_pantompkins)
        self.formLayout_8.setSpacing(4)
        self.formLayout_8.setContentsMargins(7, 7, 7, 7)
        self.formLayout_8.setObjectName(u"formLayout_8")
        self.correctArtifactsLabel = QLabel(self.page_peak_pantompkins)
        self.correctArtifactsLabel.setObjectName(u"correctArtifactsLabel")
        self.correctArtifactsLabel.setFont(font5)

        self.formLayout_8.setWidget(1, QFormLayout.LabelRole, self.correctArtifactsLabel)

        self.peak_pantompkins_correct_artifacts = QCheckBox(self.page_peak_pantompkins)
        self.peak_pantompkins_correct_artifacts.setObjectName(u"peak_pantompkins_correct_artifacts")

        self.formLayout_8.setWidget(1, QFormLayout.FieldRole, self.peak_pantompkins_correct_artifacts)

        self.peak_pantompkins_info = QTextBrowser(self.page_peak_pantompkins)
        self.peak_pantompkins_info.setObjectName(u"peak_pantompkins_info")
        sizePolicy2.setHeightForWidth(self.peak_pantompkins_info.sizePolicy().hasHeightForWidth())
        self.peak_pantompkins_info.setSizePolicy(sizePolicy2)
        self.peak_pantompkins_info.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)

        self.formLayout_8.setWidget(0, QFormLayout.SpanningRole, self.peak_pantompkins_info)

        self.stacked_peak_parameters.addWidget(self.page_peak_pantompkins)
        self.page_peak_xqrs = QWidget()
        self.page_peak_xqrs.setObjectName(u"page_peak_xqrs")
        self.formLayout_9 = QFormLayout(self.page_peak_xqrs)
        self.formLayout_9.setSpacing(4)
        self.formLayout_9.setContentsMargins(7, 7, 7, 7)
        self.formLayout_9.setObjectName(u"formLayout_9")
        self.peak_xqrs_info = QTextBrowser(self.page_peak_xqrs)
        self.peak_xqrs_info.setObjectName(u"peak_xqrs_info")
        sizePolicy2.setHeightForWidth(self.peak_xqrs_info.sizePolicy().hasHeightForWidth())
        self.peak_xqrs_info.setSizePolicy(sizePolicy2)
        self.peak_xqrs_info.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)

        self.formLayout_9.setWidget(0, QFormLayout.SpanningRole, self.peak_xqrs_info)

        self.searchRadiusLabel = QLabel(self.page_peak_xqrs)
        self.searchRadiusLabel.setObjectName(u"searchRadiusLabel")
        self.searchRadiusLabel.setFont(font5)

        self.formLayout_9.setWidget(1, QFormLayout.LabelRole, self.searchRadiusLabel)

        self.peak_xqrs_search_radius = QSpinBox(self.page_peak_xqrs)
        self.peak_xqrs_search_radius.setObjectName(u"peak_xqrs_search_radius")
        self.peak_xqrs_search_radius.setAccelerated(True)
        self.peak_xqrs_search_radius.setCorrectionMode(QAbstractSpinBox.CorrectToNearestValue)
        self.peak_xqrs_search_radius.setMinimum(5)
        self.peak_xqrs_search_radius.setMaximum(99999)
        self.peak_xqrs_search_radius.setValue(90)

        self.formLayout_9.setWidget(1, QFormLayout.FieldRole, self.peak_xqrs_search_radius)

        self.adjustPeaksLabel = QLabel(self.page_peak_xqrs)
        self.adjustPeaksLabel.setObjectName(u"adjustPeaksLabel")
        self.adjustPeaksLabel.setFont(font5)

        self.formLayout_9.setWidget(2, QFormLayout.LabelRole, self.adjustPeaksLabel)

        self.peak_xqrs_peak_dir = ComboBox(self.page_peak_xqrs)
        self.peak_xqrs_peak_dir.setObjectName(u"peak_xqrs_peak_dir")

        self.formLayout_9.setWidget(2, QFormLayout.FieldRole, self.peak_xqrs_peak_dir)

        self.stacked_peak_parameters.addWidget(self.page_peak_xqrs)

        self.layout_container_peak_detection_sidebar.addWidget(self.stacked_peak_parameters, 2, 0, 1, 2)

        self.container_peak_detection_method = QWidget(self.container_peak_detection_sidebar)
        self.container_peak_detection_method.setObjectName(u"container_peak_detection_method")
        self.container_peak_detection_method.setFont(font)
        self.layout_container_peak_detection_method = QFormLayout(self.container_peak_detection_method)
        self.layout_container_peak_detection_method.setSpacing(4)
        self.layout_container_peak_detection_method.setContentsMargins(7, 7, 7, 7)
        self.layout_container_peak_detection_method.setObjectName(u"layout_container_peak_detection_method")
        self.label_method = QLabel(self.container_peak_detection_method)
        self.label_method.setObjectName(u"label_method")
        font6 = QFont()
        font6.setPointSize(9)
        font6.setBold(True)
        self.label_method.setFont(font6)

        self.layout_container_peak_detection_method.setWidget(0, QFormLayout.LabelRole, self.label_method)

        self.combo_box_peak_detection_method = ComboBox(self.container_peak_detection_method)
        self.combo_box_peak_detection_method.setObjectName(u"combo_box_peak_detection_method")
        sizePolicy6.setHeightForWidth(self.combo_box_peak_detection_method.sizePolicy().hasHeightForWidth())
        self.combo_box_peak_detection_method.setSizePolicy(sizePolicy6)
        self.combo_box_peak_detection_method.setFont(font)
        self.combo_box_peak_detection_method.setMaxVisibleItems(10)
        self.combo_box_peak_detection_method.setMaxCount(50)
        self.combo_box_peak_detection_method.setInsertPolicy(QComboBox.NoInsert)

        self.layout_container_peak_detection_method.setWidget(0, QFormLayout.FieldRole, self.combo_box_peak_detection_method)


        self.layout_container_peak_detection_sidebar.addWidget(self.container_peak_detection_method, 1, 0, 1, 2)

        self.label_18 = QLabel(self.container_peak_detection_sidebar)
        self.label_18.setObjectName(u"label_18")
        sizePolicy4.setHeightForWidth(self.label_18.sizePolicy().hasHeightForWidth())
        self.label_18.setSizePolicy(sizePolicy4)

        self.layout_container_peak_detection_sidebar.addWidget(self.label_18, 0, 0, 1, 1)

        self.btn_info_peak_detect = QPushButton(self.container_peak_detection_sidebar)
        self.btn_info_peak_detect.setObjectName(u"btn_info_peak_detect")
        self.btn_info_peak_detect.setIcon(icon11)
        self.btn_info_peak_detect.setFlat(True)

        self.layout_container_peak_detection_sidebar.addWidget(self.btn_info_peak_detect, 0, 1, 1, 1, Qt.AlignRight)


        self.gridLayout_6.addWidget(self.container_peak_detection_sidebar, 5, 0, 1, 3)

        self.btn_compute_results = FeedbackButton(self.sidebar_page_plots)
        self.btn_compute_results.setObjectName(u"btn_compute_results")
        self.btn_compute_results.setEnabled(True)
        sizePolicy2.setHeightForWidth(self.btn_compute_results.sizePolicy().hasHeightForWidth())
        self.btn_compute_results.setSizePolicy(sizePolicy2)
        self.btn_compute_results.setFont(font2)

        self.gridLayout_6.addWidget(self.btn_compute_results, 6, 1, 1, 1)

        self.btn_detect_peaks = FeedbackButton(self.sidebar_page_plots)
        self.btn_detect_peaks.setObjectName(u"btn_detect_peaks")
        self.btn_detect_peaks.setEnabled(True)
        sizePolicy2.setHeightForWidth(self.btn_detect_peaks.sizePolicy().hasHeightForWidth())
        self.btn_detect_peaks.setSizePolicy(sizePolicy2)
        self.btn_detect_peaks.setFont(font2)

        self.gridLayout_6.addWidget(self.btn_detect_peaks, 6, 0, 1, 1)

        self.sidebar.addWidget(self.sidebar_page_plots)
        self.sidebar_page_result = QWidget()
        self.sidebar_page_result.setObjectName(u"sidebar_page_result")
        self.gridLayout_22 = QGridLayout(self.sidebar_page_result)
        self.gridLayout_22.setSpacing(4)
        self.gridLayout_22.setContentsMargins(7, 7, 7, 7)
        self.gridLayout_22.setObjectName(u"gridLayout_22")
        self.label_21 = QLabel(self.sidebar_page_result)
        self.label_21.setObjectName(u"label_21")
        sizePolicy6.setHeightForWidth(self.label_21.sizePolicy().hasHeightForWidth())
        self.label_21.setSizePolicy(sizePolicy6)

        self.gridLayout_22.addWidget(self.label_21, 0, 0, 1, 1)

        self.btn_export_focused = QPushButton(self.sidebar_page_result)
        self.btn_export_focused.setObjectName(u"btn_export_focused")

        self.gridLayout_22.addWidget(self.btn_export_focused, 2, 0, 1, 1)

        self.btn_save_to_hdf5 = FeedbackButton(self.sidebar_page_result)
        self.btn_save_to_hdf5.setObjectName(u"btn_save_to_hdf5")

        self.gridLayout_22.addWidget(self.btn_save_to_hdf5, 2, 1, 1, 1)

        self.container_results_output_dir = QWidget(self.sidebar_page_result)
        self.container_results_output_dir.setObjectName(u"container_results_output_dir")
        sizePolicy3.setHeightForWidth(self.container_results_output_dir.sizePolicy().hasHeightForWidth())
        self.container_results_output_dir.setSizePolicy(sizePolicy3)
        self.gridLayout_19 = QGridLayout(self.container_results_output_dir)
        self.gridLayout_19.setSpacing(4)
        self.gridLayout_19.setContentsMargins(7, 7, 7, 7)
        self.gridLayout_19.setObjectName(u"gridLayout_19")
        self.gridLayout_19.setContentsMargins(0, 0, 0, 0)
        self.btn_browse_output_dir = QPushButton(self.container_results_output_dir)
        self.btn_browse_output_dir.setObjectName(u"btn_browse_output_dir")
        sizePolicy5.setHeightForWidth(self.btn_browse_output_dir.sizePolicy().hasHeightForWidth())
        self.btn_browse_output_dir.setSizePolicy(sizePolicy5)
        icon23 = QIcon()
        icon23.addFile(u":/material-symbols/folder_FILL0_wght400_GRAD0_opsz24.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_browse_output_dir.setIcon(icon23)

        self.gridLayout_19.addWidget(self.btn_browse_output_dir, 1, 0, 1, 1)

        self.line_edit_output_dir = QLineEdit(self.container_results_output_dir)
        self.line_edit_output_dir.setObjectName(u"line_edit_output_dir")
        self.line_edit_output_dir.setClearButtonEnabled(True)

        self.gridLayout_19.addWidget(self.line_edit_output_dir, 1, 1, 1, 2)


        self.gridLayout_22.addWidget(self.container_results_output_dir, 1, 0, 1, 2)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_22.addItem(self.verticalSpacer, 3, 0, 1, 2)

        self.sidebar.addWidget(self.sidebar_page_result)

        self.gridLayout_17.addWidget(self.sidebar, 0, 0, 1, 1, Qt.AlignTop)

        self.dock_widget_sidebar.setWidget(self.sidebar_dock_contents)
        MainWindow.addDockWidget(Qt.LeftDockWidgetArea, self.dock_widget_sidebar)
        self.toolbar_plots = QToolBar(MainWindow)
        self.toolbar_plots.setObjectName(u"toolbar_plots")
        self.toolbar_plots.setEnabled(False)
        sizePolicy6.setHeightForWidth(self.toolbar_plots.sizePolicy().hasHeightForWidth())
        self.toolbar_plots.setSizePolicy(sizePolicy6)
        self.toolbar_plots.setLayoutDirection(Qt.RightToLeft)
        self.toolbar_plots.setMovable(False)
        self.toolbar_plots.setAllowedAreas(Qt.RightToolBarArea|Qt.TopToolBarArea)
        self.toolbar_plots.setIconSize(QSize(16, 16))
        self.toolbar_plots.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        MainWindow.addToolBar(Qt.TopToolBarArea, self.toolbar_plots)
        self.dock_widget_sections = QDockWidget(MainWindow)
        self.dock_widget_sections.setObjectName(u"dock_widget_sections")
        self.dock_widget_sections.setEnabled(False)
        self.dock_widget_sections.setFeatures(QDockWidget.DockWidgetFloatable|QDockWidget.DockWidgetMovable)
        self.dock_widget_sections.setAllowedAreas(Qt.LeftDockWidgetArea|Qt.RightDockWidgetArea)
        self.section_dock_contents = QWidget()
        self.section_dock_contents.setObjectName(u"section_dock_contents")
        self.gridLayout_7 = QGridLayout(self.section_dock_contents)
        self.gridLayout_7.setSpacing(4)
        self.gridLayout_7.setContentsMargins(7, 7, 7, 7)
        self.gridLayout_7.setObjectName(u"gridLayout_7")
        self.gridLayout_7.setContentsMargins(0, -1, 0, -1)
        self.section_widgets_container = QWidget(self.section_dock_contents)
        self.section_widgets_container.setObjectName(u"section_widgets_container")
        sizePolicy.setHeightForWidth(self.section_widgets_container.sizePolicy().hasHeightForWidth())
        self.section_widgets_container.setSizePolicy(sizePolicy)
        self.gridLayout_18 = QGridLayout(self.section_widgets_container)
        self.gridLayout_18.setSpacing(4)
        self.gridLayout_18.setContentsMargins(7, 7, 7, 7)
        self.gridLayout_18.setObjectName(u"gridLayout_18")
        self.btn_info_section = QPushButton(self.section_widgets_container)
        self.btn_info_section.setObjectName(u"btn_info_section")
        sizePolicy6.setHeightForWidth(self.btn_info_section.sizePolicy().hasHeightForWidth())
        self.btn_info_section.setSizePolicy(sizePolicy6)
        self.btn_info_section.setIcon(icon11)
        self.btn_info_section.setFlat(True)

        self.gridLayout_18.addWidget(self.btn_info_section, 0, 1, 1, 1, Qt.AlignRight)

        self.label_6 = QLabel(self.section_widgets_container)
        self.label_6.setObjectName(u"label_6")
        sizePolicy3.setHeightForWidth(self.label_6.sizePolicy().hasHeightForWidth())
        self.label_6.setSizePolicy(sizePolicy3)

        self.gridLayout_18.addWidget(self.label_6, 0, 0, 1, 1)

        self.btn_section_remove = QPushButton(self.section_widgets_container)
        self.btn_section_remove.setObjectName(u"btn_section_remove")
        sizePolicy6.setHeightForWidth(self.btn_section_remove.sizePolicy().hasHeightForWidth())
        self.btn_section_remove.setSizePolicy(sizePolicy6)

        self.gridLayout_18.addWidget(self.btn_section_remove, 2, 1, 1, 1)

        self.combo_box_section_select = QComboBox(self.section_widgets_container)
        self.combo_box_section_select.setObjectName(u"combo_box_section_select")
        self.combo_box_section_select.setInputMethodHints(Qt.ImhNone)

        self.gridLayout_18.addWidget(self.combo_box_section_select, 1, 0, 1, 2)

        self.btn_section_add = QPushButton(self.section_widgets_container)
        self.btn_section_add.setObjectName(u"btn_section_add")
        sizePolicy6.setHeightForWidth(self.btn_section_add.sizePolicy().hasHeightForWidth())
        self.btn_section_add.setSizePolicy(sizePolicy6)

        self.gridLayout_18.addWidget(self.btn_section_add, 2, 0, 1, 1)

        self.container_section_confirm_cancel = QWidget(self.section_widgets_container)
        self.container_section_confirm_cancel.setObjectName(u"container_section_confirm_cancel")
        self.container_section_confirm_cancel.setEnabled(False)
        self.gridLayout_20 = QGridLayout(self.container_section_confirm_cancel)
        self.gridLayout_20.setSpacing(4)
        self.gridLayout_20.setContentsMargins(7, 7, 7, 7)
        self.gridLayout_20.setObjectName(u"gridLayout_20")
        self.btn_section_confirm = QPushButton(self.container_section_confirm_cancel)
        self.btn_section_confirm.setObjectName(u"btn_section_confirm")
        self.btn_section_confirm.setStyleSheet(u"background-color: rgba(85, 170, 127, 100);")
        icon24 = QIcon()
        icon24.addFile(u":/material-symbols/check_FILL0_wght400_GRAD0_opsz24.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_section_confirm.setIcon(icon24)
        self.btn_section_confirm.setIconSize(QSize(24, 24))
        self.btn_section_confirm.setFlat(False)

        self.gridLayout_20.addWidget(self.btn_section_confirm, 1, 0, 1, 1)

        self.btn_section_cancel = QPushButton(self.container_section_confirm_cancel)
        self.btn_section_cancel.setObjectName(u"btn_section_cancel")
        self.btn_section_cancel.setStyleSheet(u"background-color: rgba(255, 0, 0, 100);")
        icon25 = QIcon()
        icon25.addFile(u":/material-symbols/cancel_FILL0_wght400_GRAD0_opsz24.png", QSize(), QIcon.Normal, QIcon.Off)
        self.btn_section_cancel.setIcon(icon25)
        self.btn_section_cancel.setIconSize(QSize(24, 24))
        self.btn_section_cancel.setFlat(False)

        self.gridLayout_20.addWidget(self.btn_section_cancel, 1, 1, 1, 1)

        self.label_24 = QLabel(self.container_section_confirm_cancel)
        self.label_24.setObjectName(u"label_24")
        self.label_24.setAlignment(Qt.AlignCenter)

        self.gridLayout_20.addWidget(self.label_24, 0, 0, 1, 2)


        self.gridLayout_18.addWidget(self.container_section_confirm_cancel, 3, 0, 1, 2)


        self.gridLayout_7.addWidget(self.section_widgets_container, 0, 0, 1, 1, Qt.AlignTop)

        self.dock_widget_sections.setWidget(self.section_dock_contents)
        MainWindow.addDockWidget(Qt.RightDockWidgetArea, self.dock_widget_sections)
#if QT_CONFIG(shortcut)
        self.label_2.setBuddy(self.slider_scale_window_size)
        self.label.setBuddy(self.combo_box_scale_method)
        self.label_12.setBuddy(self.combo_box_preprocess_pipeline)
#endif // QT_CONFIG(shortcut)

        self.menubar.addAction(self.menu_file.menuAction())
        self.menubar.addAction(self.menu_info.menuAction())
        self.menubar.addAction(self.menu_editing_tools.menuAction())
        self.menubar.addAction(self.menu_debug.menuAction())
        self.menu_editing_tools.addAction(self.action_remove_peak_rect)
        self.menu_editing_tools.addAction(self.action_remove_selected_peaks)
        self.menu_editing_tools.addAction(self.action_reset_view)
        self.menu_editing_tools.addSeparator()
        self.menu_editing_tools.addAction(self.action_previous_section)
        self.menu_editing_tools.addAction(self.action_next_section)
        self.menu_editing_tools.addSeparator()
        self.menu_editing_tools.addAction(self.action_reset_all)
        self.menu_file.addAction(self.action_select_file)
        self.menu_file.addSeparator()
        self.menu_file.addAction(self.action_save_state)
        self.menu_file.addAction(self.action_load_state)
        self.menu_file.addSeparator()
        self.menu_file.addAction(self.action_exit)
        self.menu_info.addAction(self.action_toggle_whats_this_mode)
        self.menu_debug.addAction(self.action_open_console)
        self.toolbar.addAction(self.action_select_file)
        self.toolbar.addAction(self.action_save_to_hdf5)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.action_save_state)
        self.toolbar.addAction(self.action_load_state)
        self.toolbar.addAction(self.action_open_console)
        self.toolbar_plots.addAction(self.action_reset_view)
        self.toolbar_plots.addSeparator()
        self.toolbar_plots.addAction(self.action_next_section)
        self.toolbar_plots.addAction(self.action_previous_section)

        self.retranslateUi(MainWindow)
        self.tabs_main.currentChanged.connect(self.sidebar.setCurrentIndex)
        self.slider_scale_window_size.valueChanged.connect(self.spin_box_scale_window_size.setValue)
        self.spin_box_scale_window_size.valueChanged.connect(self.slider_scale_window_size.setValue)
        self.slider_order.valueChanged.connect(self.spin_box_order.setValue)
        self.spin_box_order.valueChanged.connect(self.slider_order.setValue)
        self.slider_window_size.valueChanged.connect(self.spin_box_window_size.setValue)
        self.spin_box_window_size.valueChanged.connect(self.slider_window_size.setValue)

        self.tabs_main.setCurrentIndex(0)
        self.tab_container_result_views.setCurrentIndex(0)
        self.sidebar.setCurrentIndex(0)
        self.stacked_peak_parameters.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Signal Editor", None))
        self.action_open_console.setText(QCoreApplication.translate("MainWindow", u"Console", None))
#if QT_CONFIG(tooltip)
        self.action_open_console.setToolTip(QCoreApplication.translate("MainWindow", u"Open a python console with access to the application state", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(shortcut)
        self.action_open_console.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+Shift+P", None))
#endif // QT_CONFIG(shortcut)
        self.action_toggle_sidebar.setText(QCoreApplication.translate("MainWindow", u"Show Sidebar", None))
#if QT_CONFIG(tooltip)
        self.action_toggle_sidebar.setToolTip(QCoreApplication.translate("MainWindow", u"Show / Hide sidebar", None))
#endif // QT_CONFIG(tooltip)
        self.action_select_file.setText(QCoreApplication.translate("MainWindow", u"Select File", None))
#if QT_CONFIG(tooltip)
        self.action_select_file.setToolTip(QCoreApplication.translate("MainWindow", u"Select a file to work on", None))
#endif // QT_CONFIG(tooltip)
        self.action_exit.setText(QCoreApplication.translate("MainWindow", u"Exit", None))
#if QT_CONFIG(tooltip)
        self.action_exit.setToolTip(QCoreApplication.translate("MainWindow", u"Exit program", None))
#endif // QT_CONFIG(tooltip)
        self.action_reset_all.setText(QCoreApplication.translate("MainWindow", u"Reset All", None))
#if QT_CONFIG(tooltip)
        self.action_reset_all.setToolTip(QCoreApplication.translate("MainWindow", u"Resets state of data and app back to how it was just after being loaded. Only use in case of weird app behaviour or sth, as all unsaved progress will be lost.", None))
#endif // QT_CONFIG(tooltip)
        self.action_pan_mode.setText(QCoreApplication.translate("MainWindow", u"Pan Mode", None))
        self.action_rect_mode.setText(QCoreApplication.translate("MainWindow", u"Rect Mode", None))
        self.action_reset_view.setText(QCoreApplication.translate("MainWindow", u"Reset View", None))
#if QT_CONFIG(tooltip)
        self.action_reset_view.setToolTip(QCoreApplication.translate("MainWindow", u"Adjusts the plot view to include all existing values", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(shortcut)
        self.action_reset_view.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+R", None))
#endif // QT_CONFIG(shortcut)
        self.action_previous_section.setText(QCoreApplication.translate("MainWindow", u"Previous Section", None))
#if QT_CONFIG(tooltip)
        self.action_previous_section.setToolTip(QCoreApplication.translate("MainWindow", u"Show the previous section", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(shortcut)
        self.action_previous_section.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+Left", None))
#endif // QT_CONFIG(shortcut)
        self.action_next_section.setText(QCoreApplication.translate("MainWindow", u"Next Section", None))
#if QT_CONFIG(tooltip)
        self.action_next_section.setToolTip(QCoreApplication.translate("MainWindow", u"Show the next section", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(shortcut)
        self.action_next_section.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+Right", None))
#endif // QT_CONFIG(shortcut)
        self.action_toggle_whats_this_mode.setText(QCoreApplication.translate("MainWindow", u"Whats this?", None))
#if QT_CONFIG(tooltip)
        self.action_toggle_whats_this_mode.setToolTip(QCoreApplication.translate("MainWindow", u"When enabled, shows descriptions of UI elements you click on", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(shortcut)
        self.action_toggle_whats_this_mode.setShortcut(QCoreApplication.translate("MainWindow", u"F1", None))
#endif // QT_CONFIG(shortcut)
        self.action_remove_peak_rect.setText(QCoreApplication.translate("MainWindow", u"Hide blue rectangle", None))
#if QT_CONFIG(tooltip)
        self.action_remove_peak_rect.setToolTip(QCoreApplication.translate("MainWindow", u"Hides/shows the selection rectangle for removing multiple peaks at once. Shortcut: Ctrl+F", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(shortcut)
        self.action_remove_peak_rect.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+F", None))
#endif // QT_CONFIG(shortcut)
        self.action_run_preprocessing.setText(QCoreApplication.translate("MainWindow", u"Run pre-processing", None))
#if QT_CONFIG(tooltip)
        self.action_run_preprocessing.setToolTip(QCoreApplication.translate("MainWindow", u"Filter and standardize the active signal with the currently selected values", None))
#endif // QT_CONFIG(tooltip)
        self.action_run_peak_detection.setText(QCoreApplication.translate("MainWindow", u"Run peak detection", None))
#if QT_CONFIG(tooltip)
        self.action_run_peak_detection.setToolTip(QCoreApplication.translate("MainWindow", u"Detect peaks in the current signal using the selected method and input values", None))
#endif // QT_CONFIG(tooltip)
        self.action_get_results.setText(QCoreApplication.translate("MainWindow", u"Get Results", None))
#if QT_CONFIG(tooltip)
        self.action_get_results.setToolTip(QCoreApplication.translate("MainWindow", u"Creates exportable results from the current data", None))
#endif // QT_CONFIG(tooltip)
        self.action_remove_selected_peaks.setText(QCoreApplication.translate("MainWindow", u"Remove Selected Peaks", None))
#if QT_CONFIG(tooltip)
        self.action_remove_selected_peaks.setToolTip(QCoreApplication.translate("MainWindow", u"Remove the peaks inside the selection rectangle. Shortcut: Ctrl+D", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(shortcut)
        self.action_remove_selected_peaks.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+D", None))
#endif // QT_CONFIG(shortcut)
        self.action_save_state.setText(QCoreApplication.translate("MainWindow", u"Save State", None))
#if QT_CONFIG(tooltip)
        self.action_save_state.setToolTip(QCoreApplication.translate("MainWindow", u"Saves the current state of editor and data to disk so it can be resumed later on", None))
#endif // QT_CONFIG(tooltip)
        self.action_load_state.setText(QCoreApplication.translate("MainWindow", u"Load State", None))
#if QT_CONFIG(tooltip)
        self.action_load_state.setToolTip(QCoreApplication.translate("MainWindow", u"Load a state snapshot file", None))
#endif // QT_CONFIG(tooltip)
        self.action_save_to_hdf5.setText(QCoreApplication.translate("MainWindow", u"Save to HDF5", None))
#if QT_CONFIG(tooltip)
        self.action_save_to_hdf5.setToolTip(QCoreApplication.translate("MainWindow", u"Store results as a HDF5 file", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(shortcut)
        self.action_save_to_hdf5.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+S", None))
#endif // QT_CONFIG(shortcut)
        self.label_19.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p><span style=\" font-size:14pt; font-weight:700;\">Info</span></p></body></html>", None))
        self.text_info_loading_data.setHtml(QCoreApplication.translate("MainWindow", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"hr { height: 1px; border-width: 0; }\n"
"li.unchecked::marker { content: \"\\2610\"; }\n"
"li.checked::marker { content: \"\\2612\"; }\n"
"</style></head><body style=\" font-family:'Segoe UI'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:12px; margin-bottom:4px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>", None))
        self.tabs_main.setTabText(self.tabs_main.indexOf(self.tab_data), QCoreApplication.translate("MainWindow", u"Data", None))
        self.tabs_main.setTabText(self.tabs_main.indexOf(self.tab_plots), QCoreApplication.translate("MainWindow", u"Plots", None))
        self.tab_container_result_views.setTabText(self.tab_container_result_views.indexOf(self.tab_focused_result), QCoreApplication.translate("MainWindow", u"Focused", None))
        self.tab_container_result_views.setTabText(self.tab_container_result_views.indexOf(self.tab_complete_result), QCoreApplication.translate("MainWindow", u"Complete", None))
        self.tabs_main.setTabText(self.tabs_main.indexOf(self.tab_results), QCoreApplication.translate("MainWindow", u"Results", None))
        self.tabs_main.setTabText(self.tabs_main.indexOf(self.tab_analysis), QCoreApplication.translate("MainWindow", u"Analysis", None))
        self.menu_editing_tools.setTitle(QCoreApplication.translate("MainWindow", u"Editing Tools", None))
        self.menu_file.setTitle(QCoreApplication.translate("MainWindow", u"File", None))
        self.menu_info.setTitle(QCoreApplication.translate("MainWindow", u"Info", None))
        self.menu_debug.setTitle(QCoreApplication.translate("MainWindow", u"Debug", None))
        self.toolbar.setWindowTitle(QCoreApplication.translate("MainWindow", u"toolBar", None))
        self.dock_widget_sidebar.setWindowTitle(QCoreApplication.translate("MainWindow", u"Main Sidebar", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p><span style=\" font-size:10pt; font-weight:700;\">Data Selection</span></p></body></html>", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p><span style=\" font-size:10pt; font-weight:700;\">Metadata</span></p></body></html>", None))
        self.label_23.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p><span style=\" font-weight:700;\">Signal Column</span></p></body></html>", None))
        self.sampleRateLabel.setText(QCoreApplication.translate("MainWindow", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"hr { height: 1px; border-width: 0; }\n"
"li.unchecked::marker { content: \"\\2610\"; }\n"
"li.checked::marker { content: \"\\2612\"; }\n"
"</style></head><body style=\" font-family:'Segoe UI'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:700;\">Sample Rate </span>(samples/second)</p></body></html>", None))
        self.spin_box_sample_rate.setSuffix(QCoreApplication.translate("MainWindow", u" Hz", None))
        self.btn_select_file.setText(QCoreApplication.translate("MainWindow", u"Select File", None))
        self.btn_load_selection.setText(QCoreApplication.translate("MainWindow", u"Load Selected", None))
        self.label_13.setText(QCoreApplication.translate("MainWindow", u"Date of recording (yyyy-mm-dd):", None))
        self.date_edit_file_info.setDisplayFormat(QCoreApplication.translate("MainWindow", u"yyyy-MM-dd", None))
        self.label_14.setText(QCoreApplication.translate("MainWindow", u"Subject ID:", None))
        self.label_15.setText(QCoreApplication.translate("MainWindow", u"Oxygen condition:", None))
#if QT_CONFIG(whatsthis)
        self.btn_apply_filter.setWhatsThis(QCoreApplication.translate("MainWindow", u"Applies the currently selected pipeline or custom filter to the active signal. Repeated filtering will overwrite the previous values, so its currently not possible to apply a filter to an already filtered signal.", None))
#endif // QT_CONFIG(whatsthis)
        self.btn_apply_filter.setText(QCoreApplication.translate("MainWindow", u"Run pre-processing", None))
#if QT_CONFIG(tooltip)
        self.container_standardize.setToolTip(QCoreApplication.translate("MainWindow", u"Standardization is applied post-filtering", None))
#endif // QT_CONFIG(tooltip)
        self.container_scale_window_inputs.setTitle(QCoreApplication.translate("MainWindow", u"Apply using rolling window", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"Window Size:", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"Method:", None))
        self.label_12.setText(QCoreApplication.translate("MainWindow", u"Pipeline", None))
#if QT_CONFIG(tooltip)
        self.label_16.setToolTip(QCoreApplication.translate("MainWindow", u"Standardization is applied post-filtering", None))
#endif // QT_CONFIG(tooltip)
        self.label_16.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p><span style=\" font-size:10pt; font-weight:700;\">Standardize</span></p></body></html>", None))
#if QT_CONFIG(tooltip)
        self.combo_box_preprocess_pipeline.setToolTip(QCoreApplication.translate("MainWindow", u"<html><head/><body><p>Select the pre-processing pipeline for the signal or create a custom filter</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.label_10.setText(QCoreApplication.translate("MainWindow", u"Order:", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"Filter Type:", None))
        self.dbl_spin_box_highcut.setSuffix(QCoreApplication.translate("MainWindow", u" Hz", None))
        self.label_9.setText(QCoreApplication.translate("MainWindow", u"Highcut:", None))
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"Lowcut:", None))
        self.dbl_spin_box_lowcut.setSuffix(QCoreApplication.translate("MainWindow", u" Hz", None))
        self.label_11.setText(QCoreApplication.translate("MainWindow", u"Window Size:", None))
        self.label_20.setText(QCoreApplication.translate("MainWindow", u"Powerline", None))
#if QT_CONFIG(tooltip)
        self.dbl_spin_box_powerline.setToolTip(QCoreApplication.translate("MainWindow", u"The powerline frequency (usually 50 Hz or 60 Hz)", None))
#endif // QT_CONFIG(tooltip)
        self.dbl_spin_box_powerline.setSuffix(QCoreApplication.translate("MainWindow", u" Hz", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p><span style=\" font-size:10pt; font-weight:700;\">Pre-process</span></p></body></html>", None))
        self.label_peak_window.setText(QCoreApplication.translate("MainWindow", u"Peak Window", None))
        self.peak_elgendi_ppg_peakwindow.setSuffix(QCoreApplication.translate("MainWindow", u" s", None))
        self.beatWindowLabel.setText(QCoreApplication.translate("MainWindow", u"Beat Window", None))
        self.peak_elgendi_ppg_beatwindow.setSuffix(QCoreApplication.translate("MainWindow", u" s", None))
        self.beatOffsetLabel.setText(QCoreApplication.translate("MainWindow", u"Beat Offset", None))
        self.peak_elgendi_ppg_beatoffset.setSuffix("")
        self.minimumDelayLabel.setText(QCoreApplication.translate("MainWindow", u"Minimum Delay", None))
        self.peak_elgendi_ppg_min_delay.setSuffix(QCoreApplication.translate("MainWindow", u" s", None))
        self.peak_elgendi_ppg_info.setHtml(QCoreApplication.translate("MainWindow", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"hr { height: 1px; border-width: 0; }\n"
"li.unchecked::marker { content: \"\\2610\"; }\n"
"li.checked::marker { content: \"\\2612\"; }\n"
"</style></head><body style=\" font-family:'Segoe UI'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Algorithm for detecting peaks in a PPG signal, described here: <a href=\"https://doi.org/10.1371/journal.pone.0076585\"><span style=\" text-decoration: underline; color:#038387;\">Paper</span></a></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Implementation based on `<span style=\" font-style:italic;\">neurokit2._ppg_fi"
                        "ndpeaks_elgendi`</span> function</p></body></html>", None))
        self.label_22.setText(QCoreApplication.translate("MainWindow", u"Search Radius", None))
        self.peak_local_max_info.setHtml(QCoreApplication.translate("MainWindow", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"hr { height: 1px; border-width: 0; }\n"
"li.unchecked::marker { content: \"\\2610\"; }\n"
"li.checked::marker { content: \"\\2612\"; }\n"
"</style></head><body style=\" font-family:'Segoe UI'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Detects peaks by finding the highest point in a window of size 'Search Radius'</p></body></html>", None))
        self.algorithmLabel.setText(QCoreApplication.translate("MainWindow", u"Algorithm", None))
        self.smoothingWindowLabel.setText(QCoreApplication.translate("MainWindow", u"Smoothing Window", None))
        self.peak_neurokit2_smoothwindow.setSuffix(QCoreApplication.translate("MainWindow", u" s", None))
        self.label_27.setText(QCoreApplication.translate("MainWindow", u"Average Window", None))
        self.peak_neurokit2_avgwindow.setSuffix(QCoreApplication.translate("MainWindow", u" s", None))
        self.label_28.setText(QCoreApplication.translate("MainWindow", u"Grad. Thresh. Weight", None))
        self.label_29.setText(QCoreApplication.translate("MainWindow", u"Min. Length Weight", None))
        self.label_30.setText(QCoreApplication.translate("MainWindow", u"Minimum Delay", None))
        self.peak_neurokit2_mindelay.setSuffix(QCoreApplication.translate("MainWindow", u" s", None))
        self.label_31.setText(QCoreApplication.translate("MainWindow", u"Correct Artifacts", None))
        self.peak_neurokit2_info.setHtml(QCoreApplication.translate("MainWindow", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"hr { height: 1px; border-width: 0; }\n"
"li.unchecked::marker { content: \"\\2610\"; }\n"
"li.checked::marker { content: \"\\2612\"; }\n"
"</style></head><body style=\" font-family:'Segoe UI'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Finds R-peaks in an ECG signal using the specified method/algorithm.</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">More Info: <a href=\"https://github.com/neuropsychology/NeuroKit/issues/476\"><span style=\" text-decoration: underline; color:#038387;\">Github Discussion</span></a></p>\n"
"<p style=\" margin-top:0px; mar"
                        "gin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Algorithms: <a href=\"https://neuropsychology.github.io/NeuroKit/functions/ecg.html#ecg-peaks\"><span style=\" text-decoration: underline; color:#038387;\">Function documentation</span></a></p></body></html>", None))
        self.thresholdLabel.setText(QCoreApplication.translate("MainWindow", u"Threshold", None))
        self.qRSComplexSizeLabel.setText(QCoreApplication.translate("MainWindow", u"QRS Complex Size", None))
        self.correctArtifactsLabel_2.setText(QCoreApplication.translate("MainWindow", u"Correct Artifacts", None))
        self.peak_promac_gaussian_sd.setSuffix(QCoreApplication.translate("MainWindow", u" ms", None))
        self.peak_promac_info.setHtml(QCoreApplication.translate("MainWindow", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"hr { height: 1px; border-width: 0; }\n"
"li.unchecked::marker { content: \"\\2610\"; }\n"
"li.checked::marker { content: \"\\2612\"; }\n"
"</style></head><body style=\" font-family:'Segoe UI'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Finds peaks by running multiple peak detection algorithms and combining the results. Takes a while to run</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Source: <a href=\"https://github.com/neuropsychology/NeuroKit/issues/222\"><span style=\" text-decoration: underline; color:#038387;\">https://github.com/neuropsycholo"
                        "gy/NeuroKit/issues/222</span></a></p></body></html>", None))
        self.correctArtifactsLabel.setText(QCoreApplication.translate("MainWindow", u"Correct Artifacts", None))
        self.peak_pantompkins_info.setHtml(QCoreApplication.translate("MainWindow", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"hr { height: 1px; border-width: 0; }\n"
"li.unchecked::marker { content: \"\\2610\"; }\n"
"li.checked::marker { content: \"\\2612\"; }\n"
"</style></head><body style=\" font-family:'Segoe UI'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Uses the algorithm for ECG R-Peak detection by Pan &amp; Tompkins (1985).</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Uses implementation from 'neurokit2'.</p></body></html>", None))
        self.peak_xqrs_info.setHtml(QCoreApplication.translate("MainWindow", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"hr { height: 1px; border-width: 0; }\n"
"li.unchecked::marker { content: \"\\2610\"; }\n"
"li.checked::marker { content: \"\\2612\"; }\n"
"</style></head><body style=\" font-family:'Segoe UI'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Uses XQRS detection from the 'wfdb' library, with a slightly modified peak correction step afterwards.</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Can take a while to finish when using on sections longer than 1e6 samples.</p></body></html>", None))
        self.searchRadiusLabel.setText(QCoreApplication.translate("MainWindow", u"Search Radius", None))
        self.adjustPeaksLabel.setText(QCoreApplication.translate("MainWindow", u"Adjust Peaks", None))
        self.label_method.setText(QCoreApplication.translate("MainWindow", u"Method:", None))
        self.label_18.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p><span style=\" font-size:10pt; font-weight:700;\">Peak Detection</span></p></body></html>", None))
#if QT_CONFIG(whatsthis)
        self.btn_compute_results.setWhatsThis(QCoreApplication.translate("MainWindow", u"Takes the current state of the app, signal, detected peaks, etc. and produces a results table shown in the \"Results\" tab. From there, the results can be exported to various file formats.", None))
#endif // QT_CONFIG(whatsthis)
        self.btn_compute_results.setText(QCoreApplication.translate("MainWindow", u"Get Results", None))
#if QT_CONFIG(whatsthis)
        self.btn_detect_peaks.setWhatsThis(QCoreApplication.translate("MainWindow", u"Runs the currently selected peak detection method using the shown parameters (editable in the \"Peak Detection\" section above).", None))
#endif // QT_CONFIG(whatsthis)
        self.btn_detect_peaks.setText(QCoreApplication.translate("MainWindow", u"Run peak detection", None))
        self.label_21.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p><span style=\" font-size:10pt; font-weight:700;\">Output Directory</span></p></body></html>", None))
        self.btn_export_focused.setText(QCoreApplication.translate("MainWindow", u"Export Focused Result", None))
        self.btn_save_to_hdf5.setText(QCoreApplication.translate("MainWindow", u"Export Complete Result (HDF5)", None))
        self.btn_browse_output_dir.setText(QCoreApplication.translate("MainWindow", u"Browse", None))
        self.line_edit_output_dir.setPlaceholderText(QCoreApplication.translate("MainWindow", u"Specify an output directory for the exported files. By default (when left empty), use the same folder as the applications .exe file", None))
        self.toolbar_plots.setWindowTitle(QCoreApplication.translate("MainWindow", u"toolbar_plots", None))
        self.dock_widget_sections.setWindowTitle(QCoreApplication.translate("MainWindow", u"Section Controls", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"Currently showing:", None))
        self.btn_section_remove.setText(QCoreApplication.translate("MainWindow", u"Remove Section", None))
        self.btn_section_add.setText(QCoreApplication.translate("MainWindow", u"Add Section", None))
        self.btn_section_confirm.setText(QCoreApplication.translate("MainWindow", u"Confirm", None))
        self.btn_section_cancel.setText(QCoreApplication.translate("MainWindow", u"Cancel", None))
        self.label_24.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p><span style=\" font-size:10pt; font-weight:700;\">Selection</span></p></body></html>", None))
    # retranslateUi

