from collections import OrderedDict
from typing import Any, TypedDict, Unpack

import pyqtgraph.parametertree.parameterTypes as pTypes
from loguru import logger
from pyqtgraph.parametertree import Parameter, ParameterTree
from PySide6.QtWidgets import QWidget

from ...custom_types import GeneralParameterOptions, PeakDetectionMethod


class PeakDetectionParameter(pTypes.GroupParameter):
    def __init__(
        self, method: PeakDetectionMethod, **opts: Unpack[GeneralParameterOptions]
    ) -> None:
        pTypes.GroupParameter.__init__(self, **opts)

        self._method = method

    def set_elgendi_parameters(self) -> None:
        self.clearChildren()
        self.addChildren(
            [
                pTypes.SliderParameter(
                    name="peakwindow",
                    title="Peak window",
                    type="float",
                    value=0.111,
                    default=0.111,
                    step=0.001,
                    limits=(0.0, 1.0),
                    precision=3,
                ),
                pTypes.SliderParameter(
                    name="beatwindow",
                    title="Beat window",
                    type="float",
                    value=0.667,
                    default=0.667,
                    step=0.001,
                    limits=(0.050, 2.500),
                    precision=3,
                ),
                pTypes.SliderParameter(
                    name="beatoffset",
                    title="Beat offset",
                    type="float",
                    value=0.02,
                    default=0.02,
                    step=0.01,
                    limits=(0.01, 1.0),
                    precision=2,
                ),
                pTypes.SliderParameter(
                    name="mindelay",
                    title="Minimum delay",
                    type="float",
                    value=0.3,
                    default=0.3,
                    step=0.01,
                    limits=(0.1, 5.0),
                    precision=2,
                ),
            ]
        )

    def set_wfdb_find_peaks_parameters(self) -> None:
        self.clearChildren()
        self.addChildren(
            [
                pTypes.ListParameter(
                    name="peaktype",
                    title="Type of peak",
                    limits=["soft", "hard"],
                ),
                pTypes.TextParameter(
                    name="wfdbfindpeaksinfo",
                    title="Info",
                    readonly=True,
                    value=(
                        "Using soft peak detection includes plateaus in the signal "
                        "(i.e. a maximum where multiple points share the same "
                        "maximum value) by assigning the middle point as the peak. With "
                        "hard peak detection, a value needs to be higher than both his "
                        "left and right neighbours to be considered a peak.",
                    )
                ),
            ]
        )

    def set_localmax_parameters(self) -> None:
        self.clearChildren()
        self.addChildren(
            [
                pTypes.SliderParameter(
                    name="windowsize",
                    title="Window size",
                    type="int",
                    value=110,
                    default=110,
                    step=1,
                    limits=(5, 9999),
                )
            ]
        )

    def set_neurokit_parameters(self) -> None:
        self.clearChildren()
        self.addChildren(
            [
                pTypes.SliderParameter(
                    name="smoothwindow",
                    title="Smoothing window",
                    type="float",
                    value=0.1,
                    default=0.1,
                    step=0.01,
                    limits=(0.01, 1.0),
                    precision=2,
                )
            ]
        )

    def set_xqrs_parameters(self) -> None:
        self.clearChildren()
        self.addChildren(
            [
                pTypes.SliderParameter(
                    name="windowsize",
                    title="Window size",
                    type="int",
                    value=110,
                    default=110,
                    step=2,
                    limits=(5, 9999),
                )
            ]
        )


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


# Peak Detection parameters
peak_method_elgendi = (
    {
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
)

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
    ],
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
    ],
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
    ],
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
        showHeader: bool = True,
    ) -> None:
        ParameterTree.__init__(self, parent=parent, showHeader=showHeader)
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
