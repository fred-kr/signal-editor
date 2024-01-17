from PySide6.QtCore import QDate
from PySide6.QtGui import QStandardItemModel
from PySide6.QtWidgets import QAbstractSpinBox

# ------------------------------------------------------------------------------------------------ #
#                                      Peak detection widgets                                      #
# ------------------------------------------------------------------------------------------------ #
INITIAL_PEAK_STATES = {
    "peak_elgendi_ppg_peakwindow": {
        "accelerated": True,
        "decimals": 3,
        "minimum": 0.050000000000000,
        "maximum": 5.000000000000000,
        "singleStep": 0.001000000000000,
        "value": 0.111000000000000,
    },
    "peak_elgendi_ppg_beatwindow": {
        "decimals": 3,
        "minimum": 0.100000000000000,
        "maximum": 5.000000000000000,
        "singleStep": 0.001000000000000,
        "value": 0.667000000000000,
    },
    "peak_elgendi_ppg_beatoffset": {
        "decimals": 2,
        "maximum": 1.000000000000000,
        "singleStep": 0.010000000000000,
        "value": 0.020000000000000,
    },
    "peak_elgendi_ppg_min_delay": {
        "decimals": 2,
        "maximum": 10.000000000000000,
        "singleStep": 0.010000000000000,
        "value": 0.300000000000000,
    },
    "peak_local_max_radius": {
        "accelerated": True,
        "minimum": 5,
        "maximum": 9999,
        "value": 111,
    },
    "peak_neurokit2_smoothwindow": {
        "minimum": 0.010000000000000,
        "maximum": 10.000000000000000,
        "singleStep": 0.010000000000000,
    },
    "peak_neurokit2_avgwindow": {
        "decimals": 2,
        "minimum": 0.010000000000000,
        "maximum": 10.000000000000000,
        "singleStep": 0.010000000000000,
        "value": 0.750000000000000,
    },
    "peak_neurokit2_gradthreshweight": {
        "decimals": 1,
        "minimum": 0.100000000000000,
        "maximum": 10.000000000000000,
        "singleStep": 0.100000000000000,
        "value": 1.500000000000000,
    },
    "peak_neurokit2_minlenweight": {
        "decimals": 1,
        "minimum": 0.100000000000000,
        "maximum": 10.000000000000000,
        "singleStep": 0.100000000000000,
        "value": 0.400000000000000,
    },
    "peak_neurokit2_mindelay": {
        "minimum": 0.010000000000000,
        "maximum": 10.000000000000000,
        "singleStep": 0.010000000000000,
        "value": 0.300000000000000,
    },
    "peak_neurokit2_correct_artifacts": {
        "isChecked": False,
    },
    "peak_promac_threshold": {
        "maximum": 1.000000000000000,
        "singleStep": 0.010000000000000,
        "value": 0.330000000000000,
    },
    "peak_promac_gaussian_sd": {
        "maximum": 100_000,
        "value": 100,
    },
    "peak_promac_correct_artifacts": {
        "isChecked": False,
    },
    "peak_pantompkins_correct_artifacts": {
        "isChecked": False,
    },
    "peak_xqrs_search_radius": {
        "accelerated": True,
        "correctionMode": QAbstractSpinBox.CorrectionMode.CorrectToNearestValue,
        "minimum": 5,
        "maximum": 99_999,
        "value": 110,
    },
    "peak_xqrs_peak_dir": {
        "items": {
            "Up": "up",
            "Down": "down",
            "Both": "both",
            "Compare": "compare",
        },
    },
    "peak_neurokit2_algorithm_used": {
        "enabled": False,
    },
}


METHODS_TO_WIDGETS = {
    "elgendi_ppg": [
        "peak_elgendi_ppg_peakwindow",
        "peak_elgendi_ppg_beatwindow",
        "peak_elgendi_ppg_beatoffset",
        "peak_elgendi_ppg_min_delay",
    ],
    "local": ["peak_local_max_radius"],
    "neurokit2": [
        "peak_neurokit2_smoothwindow",
        "peak_neurokit2_avgwindow",
        "peak_neurokit2_gradthreshweight",
        "peak_neurokit2_minlenweight",
        "peak_neurokit2_mindelay",
        "peak_neurokit2_correct_artifacts",
    ],
    "promac": [
        "peak_promac_threshold",
        "peak_promac_gaussian_sd",
        "peak_promac_correct_artifacts",
    ],
    "pantompkins": ["peak_pantompkins_correct_artifacts"],
    "wfdb_xqrs": [
        "peak_xqrs_search_radius",
        "peak_xqrs_peak_dir",
    ],
}

# ------------------------------------------------------------------------------------------------ #
#                                       PyQtGraph ComboBoxes                                       #
# ------------------------------------------------------------------------------------------------ #
COMBO_BOX_ITEMS = {
    "combo_box_peak_detection_method": {
        "Elgendi (PPG, fast)": "elgendi_ppg",
        "Local Maxima (Any, fast)": "local",
        "Neurokit (ECG, fast)": "neurokit2",
        "ProMAC (ECG, slow)": "promac",
        "Pan and Tompkins (ECG, medium)": "pantompkins",
        "XQRS (ECG, medium)": "wfdb_xqrs",
    },
    "combo_box_filter_method": {
        "Butterworth (SOS)": "butterworth",
        "Butterworth (BA)": "butterworth_ba",
        "Savitzky-Golay": "savgol",
        "FIR": "fir",
        "Bessel": "bessel",
        "No Filter": "None",
    },
    "combo_box_oxygen_condition": {
        "Normoxic": "normoxic",
        "Hypoxic": "hypoxic",
        "Unknown": "unknown",
    },
    "combo_box_scale_method": {
        "No Standardization": "None",
        "Z-Score": "zscore",
        "Median Absolute Deviation": "mad",
    },
    "combo_box_preprocess_pipeline": {
        "Custom": "custom",
        "Elgendi (PPG)": "ppg_elgendi",
        "Neurokit (ECG)": "ecg_neurokit2",
    },
}

# ------------------------------------------------------------------------------------------------ #
#                                              General                                             #
# ------------------------------------------------------------------------------------------------ #
INITIAL_STATE_MAP = {
    "table_data_preview": {
        "model": QStandardItemModel(),
    },
    "table_data_info": {
        "model": QStandardItemModel(),
    },
    "line_edit_active_file": {"text": ""},
    "group_box_subset_params": {
        "enabled": False,
        "checked": False,
    },
    "container_file_info": {
        "enabled": True,
    },
    "date_edit_file_info": {
        "date": QDate.currentDate(),
    },
    "line_edit_subject_id": {
        "text": "",
    },
    "combo_box_oxygen_condition": {
        "value": "normoxic",
    },
    "btn_load_selection": {
        "enabled": False,
    },
    "stacked_hbr_vent": {
        "currentIndex": 0,
    },
    "btn_view_hbr": {
        "checked": True,
    },
    "btn_view_vent": {
        "checked": False,
    },
    "combo_box_preprocess_pipeline": {
        "value": "custom",
    },
    "combo_box_filter_method": {
        "enabled": True,
        "value": "None",
    },
    "combo_box_scale_method": {
        "value": "None",
    },
    "container_standardize": {
        "enabled": True,
    },
    "container_scale_window_inputs": {
        "enabled": True,
        "checked": True,
    },
    "dbl_spin_box_lowcut": {
        "value": 0.5,
    },
    "dbl_spin_box_highcut": {
        "value": 8.0,
    },
    "spin_box_order": {
        "value": 3,
    },
    "slider_order": {
        "value": 3,
    },
    "spin_box_window_size": {
        "value": 250,
    },
    "slider_window_size": {
        "value": 250,
    },
    "combo_box_peak_detection_method": {
        "value": "elgendi_ppg",
    },
    "btn_detect_peaks": {
        "enabled": True,
    },
    "btn_compute_results": {"enabled": True},
    "table_view_results_hbr": {
        "model": QStandardItemModel(),
    },
    "table_view_results_ventilation": {
        "model": QStandardItemModel(),
    },
    "tabs_result": {
        "currentIndex": 0,
    },
    "dock_widget_sections": {
        "visible": False,
    },
}


INITIAL_STATE_METHODS_MAP = {
    "enabled": "setEnabled",
    "visible": "setVisible",
    "checked": "setChecked",
    "text": "setText",
    "model": "setModel",
    "value": "setValue",
    "currentText": "setCurrentText",
    "currentIndex": "setCurrentIndex",
    "date": "setDate",
    "decimals": "setDecimals",
    "minimum": "setMinimum",
    "maximum": "setMaximum",
    "singleStep": "setSingleStep",
    "stepType": "setStepType",
    "accelerated": "setAccelerated",
    "correctionMode": "setCorrectionMode",
    "isChecked": "setChecked",
    "items": "setItems",
    "specialValueText": "setSpecialValueText",
}

# ------------------------------------------------------------------------------------------------ #
#                                    Filter Method Input Widgets                                   #
# ------------------------------------------------------------------------------------------------ #
FILTER_INPUT_STATES = {
    "butterworth": {
        "container_lowcut": True,
        "container_highcut": True,
        "container_order_inputs": True,
        "container_window_size": False,
        "container_powerline": False,
    },
    "butterworth_ba": {
        "container_lowcut": True,
        "container_highcut": True,
        "container_order_inputs": True,
        "container_window_size": False,
        "container_powerline": False,
    },
    "bessel": {
        "container_lowcut": True,
        "container_highcut": True,
        "container_order_inputs": True,
        "container_window_size": False,
        "container_powerline": False,
    },
    "fir": {
        "container_lowcut": True,
        "container_highcut": True,
        "container_order_inputs": False,
        "container_window_size": True,
        "container_powerline": False,
    },
    "savgol": {
        "container_lowcut": False,
        "container_highcut": False,
        "container_order_inputs": True,
        "container_window_size": True,
        "container_powerline": False,
    },
    "None": {
        "container_lowcut": False,
        "container_highcut": False,
        "container_order_inputs": False,
        "container_window_size": False,
        "container_powerline": False,
    },
}
