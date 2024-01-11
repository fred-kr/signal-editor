from configparser import ConfigParser
from pathlib import Path
from typing import Literal, cast


class ConfigHandler:
    @classmethod
    def create_default_file(cls, config_file_location: Path | str) -> None:
        config = ConfigParser()
        config["Paths"] = {
            "AppDir": str(Path.cwd()),
            "DataDir": str(Path.cwd()),
            "OutputDir": str(Path.cwd() / "output"),
        }
        config["Defaults"] = {"Style": "dark", "SampleRate": str(-1)}

        with open(config_file_location, "w") as configfile:
            config.write(configfile)

    def __init__(self, config_file: str) -> None:
        self.config = ConfigParser()
        if not Path(config_file).exists():
            ConfigHandler.create_default_file(config_file)
        self.config.read(config_file)
        self._app_dir = self._get_path("Paths", "AppDir", fallback=str(Path.cwd()))
        self._data_dir = self._get_path("Paths", "DataDir", fallback=str(self._app_dir))
        self._output_dir = self._get_path(
            "Paths", "OutputDir", fallback=str(self._app_dir / "output")
        )
        self.validate_paths()
        if "Defaults" not in self.config.sections():
            self.config["Defaults"] = {"Style": "dark", "SampleRate": str(200)}
        self._style = self.config.get("Defaults", "Style", fallback="dark")
        self._sample_rate = int(self.config.get("Defaults", "SampleRate", fallback=-1))

    def _get_path(self, section: str, option: str, fallback: str) -> Path:
        return Path(self.config.get(section, option, fallback=fallback))

    def validate_paths(self) -> None:
        for path in [self._app_dir, self._data_dir, self._output_dir]:
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)

    @property
    def app_dir(self) -> Path:
        return self._app_dir

    @property
    def data_dir(self) -> Path:
        return self._data_dir

    @data_dir.setter
    def data_dir(self, value: Path | str) -> None:
        self._set_path("DataDir", value)

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    @output_dir.setter
    def output_dir(self, value: Path | str) -> None:
        self._set_path("OutputDir", value)

    def _set_path(self, section: str, value: Path | str) -> None:
        path = Path(value)
        if path.is_dir():
            self.config.set("Paths", section, str(path))
            self.write_config()

    @property
    def style(self) -> Literal["light", "dark"]:
        return cast(Literal["light", "dark"], self._style)

    @style.setter
    def style(self, value: str | Literal["light", "dark"]) -> None:
        self.config.set("Defaults", "Style", value)
        self.write_config()

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value: int | float) -> None:
        self.config.set("Defaults", "SampleRate", str(int(value)))
        self.write_config()

    def write_config(self) -> None:
        with open("config.ini", "w") as configfile:
            self.config.write(configfile)
