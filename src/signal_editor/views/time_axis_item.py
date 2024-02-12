import typing as t

import pyqtgraph as pg


class TimeAxisItem(pg.AxisItem):
    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        pg.AxisItem.__init__(self, *args, **kwargs)
        self.setLabel("Time (hh:mm:ss)")

    def tickStrings(self, values: list[float], scale: float, spacing: float) -> list[str]:
        strings: list[str] = []
        for v in values:
            vs = v * scale
            minutes, seconds = divmod(int(vs), 60)
            hours, minutes = divmod(minutes, 60)
            vstr = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            strings.append(vstr)
        return strings
