import functools
from collections import OrderedDict
from typing import Any, TypedDict

from PySide6.QtWidgets import QWidget
from loguru import logger
from pyqtgraph.parametertree import Parameter, ParameterTree, RunOptions, Interactor

from ...models.peaks import find_ppg_peaks_elgendi


class ParamChild(TypedDict, total=False):
    name: str
    type: str
    value: str | int | float | bool
    title: str | None


class ParamsType(TypedDict, total=False):
    name: str
    type: str
    readonly: bool
    children: list[ParamChild]


ppg_clean_params_elgendi: list[ParamsType] = [
    {
        "name": "Filter parameters",
        "type": "group",
        "readonly": True,
        "children": [
            {
                "name": "method",
                "type": "str",
                "value": "butterworth",
                "title": "Filter method",
            },
            {"name": "lowcut", "type": "float", "value": 0.5},
            {"name": "highcut", "type": "float", "value": 8.0},
            {"name": "order", "type": "int", "value": 3},
        ],
    },
]


# Peak Detection parameters
peak_method_elgendi = {
        "name": "elgendi",
        "type": "group",
        "children": [
            {
                "name": "peakwindow",
                "type": "float",
                "value": 0.111,
                "default": 0.111,
            },
            {
                "name": "beatwindow",
                "type": "float",
                "value": 0.667,
                "default": 0.667,
            },
            {
                "name": "beatoffset",
                "type": "float",
                "value": 0.02,
                "default": 0.02,
            },
            {
                "name": "mindelay",
                "type": "float",
                "value": 0.3,
                "default": 0.3,
            },
        ],
    },

peak_method_neurokit2 = {
    "name": "neurokit2",
    "type": "group",
    "children": [
        {
            "name": "radius",
            "type": "int",
            "value": 100,
            "default": 100,
            "limits": [5, 9999],
            "step": 2,
        },
    ]
}

peak_method_local = {
    "name": "local",
    "type": "group",
    "children": [
        {
            "name": "radius",
            "type": "int",
            "value": 100,
            "default": 100,
            "limits": [5, 9999],
            "step": 2,
        },
        {
            "name": "min_or_max",
            "type": "list",
            "values": ["min", "max"],
            "default": "max",
        },
    ]
}

peak_method_xqrs = {
    "name": "xqrs",
    "type": "group",
    "children": [
        {
            "name": "radius",
            "type": "int",
            "value": 100,
            "default": 100,
            "limits": [5, 9999],
            "step": 2,
        },
        {
            "name": "correction",
            "type": "list",
            "values": ["up", "down", "both", "compare", "None"],
        },
    ]
}


def flatten_ordered_dict(
    d: OrderedDict[str, Any], parent_key: str = "", sep: str = "_"
) -> dict[str, Any]:
    """
    Flattens an ordered dictionary into a regular dictionary.

    Parameters:
        d (OrderedDict[str, Any]): The ordered dictionary to be flattened.
        parent_key (str): The prefix to be added to each key in the flattened dictionary. Defaults to an empty string.
        sep (str): The separator to be used between the parent key and child key in the flattened dictionary. Defaults to "_".

    Returns:
        dict[str, Any]: The flattened dictionary.

    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, tuple) and isinstance(v[1], OrderedDict):
            if v[1]:  # if the OrderedDict is not empty, recurse
                items.extend(flatten_ordered_dict(v[1], new_key, sep=sep).items())
            else:  # if the OrderedDict is empty, just add the value
                items.append((new_key, v[0]))
        else:
            items.append((new_key, v))
    return dict(items)


class AlgoParamTree(ParameterTree):
    def __init__(
        self,
        params_schema: list[ParamsType],
        parent: QWidget | None = None,
        **kwargs: Any,
    ) -> None:
        ParameterTree.__init__(self, parent=parent, **kwargs)
        self._initial_params = params_schema
        self.params = Parameter.create(
            name="params", type="group", children=params_schema
        )

    def set_params(self, params_schema: list[ParamsType]) -> None:
        params = Parameter.create(name="params", type="group", children=params_schema)
        self.setParameters(params, showTop=True)

    def reset_params(self) -> None:
        self.set_params(self._initial_params)

    def get_current_values(self) -> dict[str, Any]:
        values = flatten_ordered_dict(self.params.getValues())
        formatted_logger_output = [
            f"{k}: {v}\n"
            for k, v in values.items()
            if k not in ["name", "type", "children"]
        ]
        logger.debug(f"Current filter values: {formatted_logger_output}")
        return values


def create_interactor_tree(host: Parameter):
    interactor = Interactor(parent=host, runOptions=RunOptions.ON_ACTION)

    @interactor.decorate()
    def _interact_elgendi_peaks(
        peakwindow: float = 0.111,
        beatwindow: float = 0.667,
        beatoffset: float = 0.02,
        mindelay: float = 0.3,
    ):
        return functools.partial(
            find_ppg_peaks_elgendi,
            peakwindow=peakwindow,
            beatwindow=beatwindow,
            beatoffset=beatoffset,
            mindelay=mindelay,
        )
