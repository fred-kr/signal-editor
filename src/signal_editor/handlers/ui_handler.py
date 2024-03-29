import datetime
import typing as t

import polars.selectors as ps
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from .. import type_aliases as _t
from ..handlers import PlotHandler
from ..views.ui_state_maps import (
    COMBO_BOX_ITEMS,
    FILTER_INPUT_STATES,
    INITIAL_PEAK_STATES,
    INITIAL_STATE_MAP,
    INITIAL_STATE_METHODS_MAP,
)

if t.TYPE_CHECKING:
    from ..app import SignalEditor


class UIHandler(QtCore.QObject):
    def __init__(self, app: "SignalEditor", plot: PlotHandler) -> None:
        super(UIHandler, self).__init__()
        self._app = app
        self.plot = plot
        self.setup_ui()
        self._connect_qt_signals()

    def _connect_qt_signals(self) -> None:
        self._app.tabs_main.currentChanged.connect(self.on_main_tab_changed)
        self._app.combo_box_filter_method.currentTextChanged.connect(
            self.handle_filter_method_changed
        )
        self._app.combo_box_preprocess_pipeline.currentTextChanged.connect(
            self.handle_preprocess_pipeline_changed
        )

        self._app.tabs_main.currentChanged.connect(self.on_main_tab_changed)
        self._app.btn_reset_peak_detection_values.clicked.connect(
            self.set_initial_peak_detection_parameters
        )

    def setup_ui(self) -> None:
        self._set_combo_box_items()

        # Signal Filtering
        self._setup_inputs()

        # File Info
        self._setup_widgets()

        # Statusbar
        self._setup_statusbar()

        # Toolbar Plots
        self._setup_toolbars()

    def _setup_statusbar(self) -> None:
        sb = self._app.statusbar
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed
        )
        sb.addWidget(self.progress_bar)
        self.progress_bar.hide()
        self.label_currently_showing = QtWidgets.QLabel("Currently showing: ")
        sb.addPermanentWidget(self.label_currently_showing)
        sb.showMessage("Ready")

    def _setup_widgets(self) -> None:
        self._app.container_file_info.setEnabled(True)
        self._app.btn_load_selection.setEnabled(False)
        self._app.dock_widget_sections.setVisible(False)
        self._app.container_section_confirm_cancel.setEnabled(True)
        self._app.btn_section_remove.setEnabled(False)
        self._app.action_remove_section.setEnabled(False)
        self._app.container_section_confirm_cancel.hide()

        export_menu = QtWidgets.QMenu(self._app.btn_export_focused)
        export_menu.addAction("CSV", lambda: self._app.export_focused_result("csv"))
        export_menu.addAction(
            "Text (tab-delimited)", lambda: self._app.export_focused_result("txt")
        )
        export_menu.addAction("Excel", lambda: self._app.export_focused_result("xlsx"))
        self._app.btn_export_focused.setMenu(export_menu)

    def _setup_inputs(self) -> None:
        # Signal Filtering
        self._app.combo_box_preprocess_pipeline.setValue("custom")
        self._app.container_custom_filter_inputs.setEnabled(True)
        self._set_elgendi_cleaning_params()

        # Peak Detection
        peak_combo_box = self._app.combo_box_peak_detection_method
        stacked_peak_widget = self._app.stacked_peak_parameters
        peak_combo_box.blockSignals(True)
        peak_combo_box.setItems(COMBO_BOX_ITEMS["combo_box_peak_detection_method"])
        peak_combo_box.setCurrentIndex(0)
        stacked_peak_widget.setCurrentIndex(0)
        peak_combo_box.blockSignals(False)
        peak_combo_box.currentIndexChanged.connect(stacked_peak_widget.setCurrentIndex)

    def _setup_toolbars(self) -> None:
        edit_tb = self._app.toolbar_plots
        edit_tb.insertWidget(self._app.action_confirm, self._app.combo_box_section_select)
        edit_tb.setVisible(False)
        self._app.action_toggle_section_sidebar.setChecked(False)

    def _set_combo_box_items(self) -> None:
        for key, value in COMBO_BOX_ITEMS.items():
            combo_box: pg.ComboBox = getattr(self._app, key)
            combo_box.setItems(value)

    @QtCore.Slot()
    def set_initial_peak_detection_parameters(self) -> None:
        for widget_name, properties in INITIAL_PEAK_STATES.items():
            for property_name, value in properties.items():
                getattr(self._app, widget_name).__getattribute__(
                    INITIAL_STATE_METHODS_MAP[property_name]
                )(value)

    @staticmethod
    def _blocked_set_combo_box_items(combo_box: QtWidgets.QComboBox, items: list[str]) -> None:
        combo_box.blockSignals(True)
        combo_box.clear()
        combo_box.addItems(items)
        combo_box.setCurrentIndex(0)
        combo_box.blockSignals(False)

    @QtCore.Slot()
    def update_data_select_ui(self) -> None:
        if self._app.data.raw_df is None:
            return
        self._app.btn_load_selection.setEnabled(True)
        data_cols = self._app.data.raw_df.select(
            (~ps.contains(["index", "time", "temp"])) & (ps.float())
        ).columns
        self._blocked_set_combo_box_items(self._app.combo_box_signal_column, data_cols)

        metadata = self._app.data.metadata
        if metadata is None:
            self._app.date_edit_file_info.setDate(QtCore.QDate(2000, 1, 1))
            self._app.line_edit_subject_id.setText("unknown")
            self._app.combo_box_oxygen_condition.setValue("unknown")
        else:
            meas_date = metadata["date_recorded"] or datetime.date(2000, 1, 1)
            subject_id = metadata["animal_id"] or "unknown"
            oxygen_condition = metadata["oxygen_condition"] or "unknown"
            self._app.date_edit_file_info.setDate(
                QtCore.QDate(meas_date.year, meas_date.month, meas_date.day)
            )
            self._app.line_edit_subject_id.setText(subject_id)
            self._app.combo_box_oxygen_condition.setValue(oxygen_condition)

    @QtCore.Slot()
    def reset_widget_state(self) -> None:
        mw = self._app
        mw.tabs_main.setCurrentIndex(0)
        mapping = INITIAL_STATE_METHODS_MAP
        combined_map = INITIAL_STATE_MAP | INITIAL_PEAK_STATES
        for widget_name, state in combined_map.items():
            for attribute, value in state.items():
                getattr(mw, widget_name).__getattribute__(mapping[attribute])(value)

        self.plot.reset_plots()
        mw.statusbar.showMessage("Ready")

    @QtCore.Slot(int)
    def on_main_tab_changed(self, index: int) -> None:
        show_section_dock = index == 1 or self._app.action_toggle_section_sidebar.isChecked()
        self._app.toolbar_plots.setVisible(index == 1)
        self._app.dock_widget_sections.setVisible(show_section_dock)

    @QtCore.Slot(str)
    def handle_filter_method_changed(self, text: str) -> None:
        method = self._app.filter_method

        for widget_name, enabled in FILTER_INPUT_STATES[method].items():
            getattr(self._app, widget_name).setEnabled(enabled)

    def _set_elgendi_cleaning_params(self) -> None:
        self._app.combo_box_filter_method.blockSignals(True)
        self._app.combo_box_filter_method.setValue("butterworth")
        self._app.combo_box_filter_method.blockSignals(False)
        self._app.dbl_spin_box_lowcut.setValue(0.5)
        self._app.dbl_spin_box_highcut.setValue(8.0)
        self._app.spin_box_order.setValue(3)
        self._app.slider_order.setValue(3)

    def _set_neurokit2_cleaning_params(self) -> None:
        self._app.combo_box_filter_method.blockSignals(True)
        self._app.combo_box_filter_method.setValue("butterworth")
        self._app.combo_box_filter_method.blockSignals(False)
        self._app.dbl_spin_box_lowcut.setValue(0.5)
        self._app.dbl_spin_box_highcut.setValue(
            self._app.dbl_spin_box_highcut.minimum()
        )  # This displays the SpecialValueText 'None'
        self._app.spin_box_order.setValue(5)
        self._app.slider_order.setValue(5)

    @QtCore.Slot()
    def handle_preprocess_pipeline_changed(self) -> None:
        pipeline_value = self._app.pipeline
        if pipeline_value == "custom":
            self._app.container_custom_filter_inputs.setEnabled(True)
            selected_filter = self._app.filter_method
            self.handle_filter_method_changed(selected_filter)

        elif pipeline_value == "ppg_elgendi":
            self._app.container_custom_filter_inputs.setEnabled(False)
            self._app.combo_box_filter_method.setValue("butterworth")
            self._set_elgendi_cleaning_params()
        elif pipeline_value == "ecg_neurokit2":
            self._app.container_custom_filter_inputs.setEnabled(False)
            self._app.container_powerline.setEnabled(True)
            self._set_neurokit2_cleaning_params()
        else:
            self._app.container_custom_filter_inputs.setEnabled(False)
            msg = f"Selected pipeline {pipeline_value} not yet implemented."
            self._app.sig_show_message.emit(msg, "info")
            return

    def get_filter_parameters(self) -> _t.SignalFilterParameters:
        method = self._app.filter_method

        filter_params = _t.SignalFilterParameters(
            lowcut=None,
            highcut=None,
            method=method,
            order=2,
            window_size="default",
            powerline=50,
        )
        if method != "None":
            filter_widgets = {
                "lowcut": self._app.dbl_spin_box_lowcut,
                "highcut": self._app.dbl_spin_box_highcut,
                "order": self._app.spin_box_order,
                "window_size": self._app.spin_box_window_size,
                "powerline": self._app.dbl_spin_box_powerline,
            }
            for param, widget in filter_widgets.items():
                if widget.isEnabled():
                    if param in ("lowcut", "highcut") and widget.value() == 0:
                        filter_params[param] = None
                    filter_params[param] = widget.value()

        return filter_params

    def get_standardize_parameters(self) -> _t.StandardizeParameters:
        method = self._app.scale_method
        robust = method == "mad"
        if self._app.container_scale_window_inputs.isChecked():
            window_size = self._app.spin_box_scale_window_size.value()
        else:
            window_size = None

        return {"robust": robust, "window_size": window_size, "method": method}

    def get_peak_detection_parameters(self) -> _t.PeakDetectionParameters:
        method = self._app.peak_detection_method

        match method:
            case "elgendi_ppg":
                vals = _t.PeakDetectionElgendiPPG(
                    peakwindow=self._app.peak_elgendi_ppg_peakwindow.value(),
                    beatwindow=self._app.peak_elgendi_ppg_beatwindow.value(),
                    beatoffset=self._app.peak_elgendi_ppg_beatoffset.value(),
                    mindelay=self._app.peak_elgendi_ppg_min_delay.value(),
                )
            case "local":
                vals = _t.PeakDetectionLocalMaxima(radius=self._app.peak_local_max_radius.value())
            case "neurokit2":
                vals = _t.PeakDetectionNeurokit2(
                    smoothwindow=self._app.peak_neurokit2_smoothwindow.value(),
                    avgwindow=self._app.peak_neurokit2_avgwindow.value(),
                    gradthreshweight=self._app.peak_neurokit2_gradthreshweight.value(),
                    minlenweight=self._app.peak_neurokit2_minlenweight.value(),
                    mindelay=self._app.peak_neurokit2_mindelay.value(),
                    correct_artifacts=self._app.peak_neurokit2_correct_artifacts.isChecked(),
                )
            case "promac":
                vals = _t.PeakDetectionProMAC(
                    threshold=self._app.peak_promac_threshold.value(),
                    gaussian_sd=self._app.peak_promac_gaussian_sd.value(),
                    correct_artifacts=self._app.peak_promac_correct_artifacts.isChecked(),
                )
            case "pantompkins":
                vals = _t.PeakDetectionPantompkins(
                    correct_artifacts=self._app.peak_pantompkins_correct_artifacts.isChecked(),
                )
            case "wfdb_xqrs":
                vals = _t.PeakDetectionXQRS(
                    search_radius=self._app.peak_xqrs_search_radius.value(),
                    peak_dir=self._app.xqrs_peak_direction,
                )
            case _:
                raise NotImplementedError(f"Peak detection method {method} not yet implemented.")

        for key, val in vals.items():
            if isinstance(val, float):
                vals[key] = round(val, 3)

        return {"method": method, "method_parameters": vals}

    # def set_peak_detection_parameters(self, params: _t.PeakDetectionParameters) -> None:
    #     # sourcery skip: extract-method
    #     method = params["method"]
    #     self._app.combo_box_peak_detection_method.setValue(method)

    #     vals = params["method_parameters"]

    #     match method:
    #         case "elgendi_ppg":
    #             vals = t.cast(_t.PeakDetectionElgendiPPG, vals)
    #             self._app.peak_elgendi_ppg_peakwindow.setValue(vals["peakwindow"])
    #             self._app.peak_elgendi_ppg_beatwindow.setValue(vals["beatwindow"])
    #             self._app.peak_elgendi_ppg_beatoffset.setValue(vals["beatoffset"])
    #             self._app.peak_elgendi_ppg_min_delay.setValue(vals["mindelay"])
    #         case "local":
    #             vals = t.cast(_t.PeakDetectionLocalMaxima, vals)
    #             self._app.peak_local_max_radius.setValue(vals["radius"])
    #         case "neurokit2":
    #             vals = t.cast(_t.PeakDetectionNeurokit2, vals)
    #             self._app.peak_neurokit2_smoothwindow.setValue(vals["smoothwindow"])
    #             self._app.peak_neurokit2_avgwindow.setValue(vals["avgwindow"])
    #             self._app.peak_neurokit2_gradthreshweight.setValue(vals["gradthreshweight"])
    #             self._app.peak_neurokit2_minlenweight.setValue(vals["minlenweight"])
    #             self._app.peak_neurokit2_mindelay.setValue(vals["mindelay"])
    #             self._app.peak_neurokit2_correct_artifacts.setChecked(vals["correct_artifacts"])
    #         case "promac":
    #             vals = t.cast(_t.PeakDetectionProMAC, vals)
    #             self._app.peak_promac_threshold.setValue(vals["threshold"])
    #             self._app.peak_promac_gaussian_sd.setValue(vals["gaussian_sd"])
    #             self._app.peak_promac_correct_artifacts.setChecked(vals["correct_artifacts"])
    #         case "pantompkins":
    #             vals = t.cast(_t.PeakDetectionPantompkins, vals)
    #             self._app.peak_pantompkins_correct_artifacts.setChecked(vals["correct_artifacts"])
    #         case "wfdb_xqrs":
    #             vals = t.cast(_t.PeakDetectionXQRS, vals)
    #             self._app.peak_xqrs_search_radius.setValue(vals["search_radius"])
    #             self._app.peak_xqrs_peak_dir.setValue(vals["peak_dir"])
    #         case _:
    #             raise NotImplementedError(f"Peak detection method {method} not yet implemented.")
