"""Self-contained runtime configuration for the copied YOPO primitives."""

from __future__ import annotations

from pathlib import Path

from ruamel.yaml import YAML


class Config:
    def __init__(self):
        with (Path(__file__).resolve().parent / "traj_opt.yaml").open("r") as stream:
            self._data = YAML().load(stream)
        self._data["train"] = True
        self._data["goal_length"] = 2.0 * self._data["radio_range"]
        self._data["sgm_time"] = (
            2.0 * self._data["radio_range"] / self._data["vel_max_train"]
        )
        self._data["traj_num"] = (
            self._data["horizon_num"]
            * self._data["vertical_num"]
            * self._data["radio_num"]
        )

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value


cfg = Config()
