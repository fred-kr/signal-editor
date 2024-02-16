import datetime
import os
import typing as t
from configparser import ConfigParser
from pathlib import Path


def create_config_file(filename: str | Path) -> ConfigParser:
    config = ConfigParser()

    config["General"] = {
        "style": "dark",
        "auto_switch_tabs": "True",
        "add_peak_click_tolerance_px": "80",
    }

    config["Processing"] = {
        "sample_rate": "-1",
    }

    config["FileLocations"] = {"data_files": ".", "output_files": ".\\output"}

    config["FilePatterns"] = {
        "focused_result": "FocusedResult_{SIGNAL_NAME}_{SOURCE_FILE_NAME}",
        "complete_result": "CompleteResult_{SIGNAL_NAME}_{SOURCE_FILE_NAME}",
        "rolling_rate_result": "RollingRate_{SIGNAL_NAME}_{SOURCE_FILE_NAME}",
        "app_state_snapshot": "SignalEditorStateSnapshot_{TIMESTAMP}",
    }

    with open(filename, "w") as configfile:
        config.write(configfile)

    return config


class ConfigHandler:
    def __init__(self, config_file: str | os.PathLike[str]) -> None:
        self.config_file = Path(config_file)
        if not self.config_file.exists():
            self.config = create_config_file(self.config_file)
        else:
            self.config = ConfigParser()
            self.config.read(config_file)

        self._general = dict(self.config.items("General"))
        self._processing = dict(self.config.items("Processing"))
        self._file_locations = dict(self.config.items("FileLocations"))
        self._file_patterns = dict(self.config.items("FilePatterns"))

    def write_config(self) -> None:
        with open(self.config_file, "w") as configfile:
            self.config.write(configfile)

    def _set_path(self, section: str, value: Path | str) -> None:
        path = Path(value)
        if path.is_dir():
            self.config.set("FileLocations", section, path.as_posix())
        else:
            e = NotADirectoryError()
            e.add_note(f"{path} is not a directory.")
            raise e

    @property
    def switch_on_load(self) -> bool:
        return bool(self._general.get("auto_switch_tabs", "True"))

    @property
    def click_tolerance(self) -> int:
        return int(self._general.get("add_peak_click_tolerance_px", "80"))

    @property
    def style(self) -> t.Literal["light", "dark"]:
        return t.cast(t.Literal["light", "dark"], self._general.get("style", "dark"))

    @style.setter
    def style(self, value: str | t.Literal["light", "dark"]) -> None:
        self.config.set("General", "style", value)
        self.write_config()

    @property
    def sample_rate(self) -> int:
        return self.config.getint("Processing", "sample_rate", fallback=-1)

    @sample_rate.setter
    def sample_rate(self, value: int | float) -> None:
        self.config.set("Processing", "sample_rate", str(int(value)))
        self.write_config()

    @property
    def data_dir(self) -> Path:
        return Path(self._file_locations.get("data_files", ".")).resolve()

    @data_dir.setter
    def data_dir(self, value: Path | str) -> None:
        self._set_path("data_files", value)
        self.write_config()

    @property
    def output_dir(self) -> Path:
        return Path(self._file_locations.get("output_files", ".\\output")).resolve()

    @output_dir.setter
    def output_dir(self, value: Path | str) -> None:
        self._set_path("output_files", value)
        self.write_config()

    def make_focused_result_name(self, signal_name: str, source_file_name: str) -> str:
        return self._file_patterns.get(
            "focused_result", "FocusedResult_{SIGNAL_NAME}_{SOURCE_FILE_NAME}"
        ).format(SIGNAL_NAME=signal_name, SOURCE_FILE_NAME=source_file_name)

    def make_complete_result_name(self, signal_name: str, source_file_name: str) -> str:
        return self._file_patterns.get(
            "complete_result", "CompleteResult_{SIGNAL_NAME}_{SOURCE_FILE_NAME}"
        ).format(SIGNAL_NAME=signal_name, SOURCE_FILE_NAME=source_file_name)

    def make_rolling_rate_result_name(
        self, signal_name: str, source_file_name: str
    ) -> str:
        return self._file_patterns.get(
            "rolling_rate_result", "RollingRate_{SIGNAL_NAME}_{SOURCE_FILE_NAME}"
        ).format(SIGNAL_NAME=signal_name, SOURCE_FILE_NAME=source_file_name)

    def make_app_state_name(self, timestamp: datetime.datetime | str) -> str:
        if isinstance(timestamp, datetime.datetime):
            timestamp = timestamp.strftime("%Y%m%d_%H%M%S")
        return self._file_patterns.get(
            "app_state_snapshot", "SignalEditorStateSnapshot_{TIMESTAMP}"
        ).format(TIMESTAMP=timestamp)
