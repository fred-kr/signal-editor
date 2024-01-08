from PySide6.QtWidgets import QAbstractSpinBox

type PeakParameterState = dict[
    str,
    dict[
        str,
        str
        | bool
        | int
        | float
        | QAbstractSpinBox.StepType
        | QAbstractSpinBox.CorrectionMode,
    ],
]
INITIAL_PEAK_STATES = {
    "peak_elgendi_ppg_peakwindow": {
        "accelerated": True,
        "decimals": 3,
        "minimum": 0.050000000000000,
        "maximum": 5.000000000000000,
        "singleStep": 0.001000000000000,
        "stepType": QAbstractSpinBox.StepType.AdaptiveDecimalStepType,
        "value": 0.111000000000000,
    },
    "peak_elgendi_ppg_beatwindow": {
        "decimals": 3,
        "minimum": 0.100000000000000,
        "maximum": 5.000000000000000,
        "singleStep": 0.001000000000000,
        "stepType": QAbstractSpinBox.StepType.AdaptiveDecimalStepType,
        "value": 0.667000000000000,
    },
    "peak_elgendi_ppg_beatoffset": {
        "decimals": 2,
        "maximum": 1.000000000000000,
        "singleStep": 0.010000000000000,
        "stepType": QAbstractSpinBox.StepType.AdaptiveDecimalStepType,
        "value": 0.020000000000000,
    },
    "peak_elgendi_ppg_min_delay": {
        "decimals": 2,
        "maximum": 10.000000000000000,
        "singleStep": 0.010000000000000,
        "stepType": QAbstractSpinBox.StepType.AdaptiveDecimalStepType,
        "value": 0.300000000000000,
    },
    "peak_local_max_radius": {
        "accelerated": True,
        "minimum": 5,
        "maximum": 9999,
        "stepType": QAbstractSpinBox.StepType.AdaptiveDecimalStepType,
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
        "maximum": 100000,
        "value": 100,
    },
    "peak_promac_correct_artifacts": {
        "isChecked": False,
    },
    "peak_pantompkins_correct_artifacts": {
        "isChecked": False,
    },
    "peak_xqrs_sampfrom": {
        "accelerated": True,
        "correctionMode": QAbstractSpinBox.CorrectionMode.CorrectToNearestValue,
        "maximum": 50_000_000,
    },
    "peak_xqrs_sampto": {
        "accelerated": True,
        "correctionMode": QAbstractSpinBox.CorrectionMode.CorrectToNearestValue,
        "maximum": 50_000_000,
    },
    "peak_xqrs_search_radius": {
        "accelerated": True,
        "correctionMode": QAbstractSpinBox.CorrectionMode.CorrectToNearestValue,
        "minimum": 5,
        "maximum": 99_999,
        "value": 90,
        "stepType": QAbstractSpinBox.StepType.AdaptiveDecimalStepType,
    },
    "peak_xqrs_peak_dir": {
        "items": {
            "Up": "up",
            "Down": "down",
            "Both": "both",
            "Compare": "compare",
        },
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
        "peak_xqrs_sampfrom",
        "peak_xqrs_sampto",
        "peak_xqrs_search_radius",
        "peak_xqrs_peak_dir",
    ],
}
