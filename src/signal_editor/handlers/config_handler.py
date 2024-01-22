import datetime
import os
import typing as t
from configparser import ConfigParser
from pathlib import Path


class ConfigHandler:
    def __init__(self, config_file: str | os.PathLike[str]) -> None:
        self.config_file = config_file
        self.config = ConfigParser()
        if not Path(config_file).exists():
            self.create_default_file(config_file)
        self.config.read(config_file)
        current_dir = Path.cwd()
        self._app_dir = self._get_path("PATHS", "AppDir", fallback=current_dir)
        self._data_dir = self._get_path("PATHS", "DataDir", fallback=self._app_dir)
        self._output_dir = self._get_path("PATHS", "OutputDir", fallback=self._app_dir / "output")
        self._style = self.config.get("DEFAULT", "Style", fallback="dark")
        self._sample_rate = int(self.config.get("DEFAULT", "SampleRate", fallback=-1))
        self._focused_result_file_name_pattern = self.config.get(
            "NAMEPATTERNS",
            "FocusedResult",
            fallback="FocusedResult_{SIGNAL_NAME}_{SOURCE_FILE_NAME}",
        )
        self._complete_result_file_name_pattern = self.config.get(
            "NAMEPATTERNS",
            "CompleteResult",
            fallback="CompleteResult_{SIGNAL_NAME}_{SOURCE_FILE_NAME}",
        )
        self._app_state_file_name_pattern = self.config.get(
            "NAMEPATTERNS", "AppState", fallback="SignalEditorStateSnapshot_{TIMESTAMP}"
        )

    @classmethod
    def create_default_file(cls, config_file_location: str | os.PathLike[str]) -> None:
        config = ConfigParser()
        current_dir = Path.cwd()
        config["PATHS"] = {
            "AppDir": str(current_dir),
            "DataDir": str(current_dir),
            "OutputDir": str(current_dir / "output"),
        }
        config["DEFAULT"] = {"Style": "dark", "SampleRate": str(-1)}
        config["NAMEPATTERNS"] = {
            "FocusedResult": "FocusedResult_{SIGNAL_NAME}_{SOURCE_FILE_NAME}",
            "CompleteResult": "CompleteResult_{SIGNAL_NAME}_{SOURCE_FILE_NAME}",
            "AppState": "SignalEditorStateSnapshot_{TIMESTAMP}",
        }
        with open(config_file_location, "w") as configfile:
            config.write(configfile)

    def write_config(self) -> None:
        with open(self.config_file, "w") as configfile:
            self.config.write(configfile)

    def _get_path(self, section: str, option: str, fallback: str | Path) -> Path:
        return Path(self.config.get(section, option, fallback=fallback))

    def _set_path(self, section: str, value: Path | str) -> None:
        path = Path(value)
        if path.is_dir():
            self.config.set("PATHS", section, str(path))

    @property
    def style(self) -> t.Literal["light", "dark"]:
        return t.cast(t.Literal["light", "dark"], self._style)

    @style.setter
    def style(self, value: str | t.Literal["light", "dark"]) -> None:
        self.config.set("DEFAULT", "Style", value)
        self.write_config()

    @property
    def app_dir(self) -> Path:
        return self._app_dir

    @property
    def data_dir(self) -> Path:
        return self._data_dir

    @data_dir.setter
    def data_dir(self, value: Path | str) -> None:
        self._set_path("DataDir", value)
        self.write_config()

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    @output_dir.setter
    def output_dir(self, value: Path | str) -> None:
        self._set_path("OutputDir", value)
        self.write_config()

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value: int | float) -> None:
        self.config.set("DEFAULT", "SampleRate", str(int(value)))
        self.write_config()

    def make_focused_result_name(self, signal_name: str, source_file_name: str) -> str:
        return self._focused_result_file_name_pattern.format(
            SIGNAL_NAME=signal_name, SOURCE_FILE_NAME=source_file_name
        )

    def make_complete_result_name(self, signal_name: str, source_file_name: str) -> str:
        return self._complete_result_file_name_pattern.format(
            SIGNAL_NAME=signal_name, SOURCE_FILE_NAME=source_file_name
        )

    def make_app_state_name(self, timestamp: datetime.datetime | str) -> str:
        if isinstance(timestamp, datetime.datetime):
            timestamp = timestamp.strftime("%Y%m%d_%H%M%S")
        return self._app_state_file_name_pattern.format(TIMESTAMP=timestamp)
