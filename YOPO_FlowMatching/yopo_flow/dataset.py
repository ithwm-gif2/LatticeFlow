"""Map-level YOPO dataset splits with reproducible evaluation states.

The original YOPO dataset performs an image-level 90/10 split inside every
map and samples a new vehicle state and goal on each access.  That behavior is
useful for training, but it leaks map identity into validation and makes
paper-level comparisons difficult to reproduce.  This module keeps YOPO's
image preprocessing and state distribution while allowing complete maps to be
held out and making validation/test observations deterministic per image.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset

from .bootstrap import add_original_yopo_to_path

add_original_yopo_to_path()

from config.config import cfg as yopo_cfg  # noqa: E402


def configure_yopo_data_root(root: str | Path) -> Path:
    """Point the reused YOPO loss code at the selected dataset root."""

    path = Path(root).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"YOPO dataset root does not exist: {path}")
    yopo_cfg["dataset_path"] = str(path)
    return path


class MapSplitYOPODataset(Dataset):
    """YOPO depth dataset restricted to a list of complete map identifiers."""

    def __init__(
        self,
        root: str | Path,
        map_ids: Iterable[int],
        mode: str,
        seed: int = 0,
        camera_pitch_deg: float = 0.0,
    ):
        super().__init__()
        if mode not in {"train", "valid", "test"}:
            raise ValueError(f"Unsupported dataset mode: {mode}")

        self.root = configure_yopo_data_root(root)
        self.map_ids = tuple(int(value) for value in map_ids)
        if not self.map_ids:
            raise ValueError(f"No maps configured for {mode} split")
        self.mode = mode
        self.seed = int(seed)
        self.camera_pitch_deg = float(camera_pitch_deg)
        self.rotation_body_camera = R.from_euler(
            "Y", self.camera_pitch_deg, degrees=True
        )
        self.deterministic_observation = mode != "train"

        self.height = int(yopo_cfg["image_height"])
        self.width = int(yopo_cfg["image_width"])
        self.vel_max = float(yopo_cfg["vel_max_train"])
        self.acc_max = float(yopo_cfg["acc_max_train"])
        self.vx_lognorm_mean = float(np.log(1 - yopo_cfg["vx_mean_unit"]))
        self.vx_lognorm_sigma = float(np.log(yopo_cfg["vx_std_unit"]))
        self.v_mean = np.asarray(
            [
                yopo_cfg["vx_mean_unit"],
                yopo_cfg["vy_mean_unit"],
                yopo_cfg["vz_mean_unit"],
            ],
            dtype=np.float64,
        )
        self.v_std = np.asarray(
            [
                yopo_cfg["vx_std_unit"],
                yopo_cfg["vy_std_unit"],
                yopo_cfg["vz_std_unit"],
            ],
            dtype=np.float64,
        )
        self.a_mean = np.asarray(
            [
                yopo_cfg["ax_mean_unit"],
                yopo_cfg["ay_mean_unit"],
                yopo_cfg["az_mean_unit"],
            ],
            dtype=np.float64,
        )
        self.a_std = np.asarray(
            [
                yopo_cfg["ax_std_unit"],
                yopo_cfg["ay_std_unit"],
                yopo_cfg["az_std_unit"],
            ],
            dtype=np.float64,
        )
        self.goal_length = float(yopo_cfg["goal_length"])
        self.goal_pitch_std = float(yopo_cfg["goal_pitch_std"])
        self.goal_yaw_std = float(yopo_cfg["goal_yaw_std"])

        image_paths: list[str] = []
        positions: list[np.ndarray] = []
        quaternions: list[np.ndarray] = []
        sample_map_ids: list[int] = []
        for map_id in self.map_ids:
            image_dir = self.root / str(map_id)
            pose_path = self.root / f"pose-{map_id}.csv"
            pointcloud_path = self.root / f"pointcloud-{map_id}.ply"
            if not image_dir.is_dir() or not pose_path.is_file() or not pointcloud_path.is_file():
                raise FileNotFoundError(
                    f"Map {map_id} is incomplete under {self.root}; expected image directory, "
                    "pose CSV and point cloud"
                )

            map_images = sorted(
                image_dir.glob("*.png"),
                key=lambda path: int(path.stem.split("_")[-1]),
            )
            states = np.loadtxt(pose_path, delimiter=",", skiprows=1).astype(np.float32)
            if len(map_images) != states.shape[0]:
                raise ValueError(
                    f"Map {map_id} has {len(map_images)} images but {states.shape[0]} poses"
                )
            image_paths.extend(str(path) for path in map_images)
            positions.append(states[:, :3])
            quaternions.append(states[:, 3:7])
            sample_map_ids.extend([map_id] * len(map_images))

        self.img_list = image_paths
        self.positions = np.concatenate(positions, axis=0).astype(np.float32)
        self.quaternions = np.concatenate(quaternions, axis=0).astype(np.float32)
        self.sample_map_ids = np.asarray(sample_map_ids, dtype=np.int64)
        print(
            f"{mode.capitalize()} split: maps={list(self.map_ids)}, "
            f"images={len(self.img_list)}, deterministic_obs={self.deterministic_observation}"
        )

    def __len__(self) -> int:
        return len(self.img_list)

    def _rng(self, item: int):
        if self.deterministic_observation:
            # Large odd multiplier avoids nearby seeds for adjacent images.
            return np.random.default_rng(self.seed + 1_000_003 * int(item))
        return np.random

    def _get_random_state(self, rng) -> tuple[np.ndarray, np.ndarray]:
        while True:
            vel = self.vel_max * (self.v_mean + self.v_std * rng.standard_normal(3))
            right_skewed_vx = -1.0
            while right_skewed_vx < 0.0:
                sample = self.vel_max * rng.lognormal(
                    mean=self.vx_lognorm_mean,
                    sigma=self.vx_lognorm_sigma,
                )
                right_skewed_vx = -sample + 1.2 * self.vel_max
            vel[0] = right_skewed_vx
            if np.linalg.norm(vel) < 1.2 * self.vel_max:
                break

        while True:
            acc = self.acc_max * (self.a_mean + self.a_std * rng.standard_normal(3))
            if np.linalg.norm(acc) < 1.2 * self.acc_max:
                break
        return vel, acc

    def _get_random_goal(self, rng) -> np.ndarray:
        pitch = np.radians(rng.normal(0.0, self.goal_pitch_std))
        yaw = np.radians(rng.normal(0.0, self.goal_yaw_std))
        direction = np.asarray(
            [
                np.cos(yaw) * np.cos(pitch),
                np.sin(yaw) * np.cos(pitch),
                np.sin(pitch),
            ]
        )
        near = float(rng.random())
        if near < 0.1:
            direction = near * 10.0 * direction
        return self.goal_length * direction

    def __getitem__(self, item: int):
        image = cv2.imread(self.img_list[item], -1)
        if image is None:
            raise OSError(f"Failed to read depth image: {self.img_list[item]}")
        image = image.astype(np.float32)
        image = cv2.resize(
            image,
            (self.width, self.height),
            interpolation=cv2.INTER_NEAREST,
        ) / 65535.0
        image = image[None, ...]

        quaternion_wxyz = self.quaternions[item]
        rotation_world_camera = R.from_quat(
            [
                quaternion_wxyz[1],
                quaternion_wxyz[2],
                quaternion_wxyz[3],
                quaternion_wxyz[0],
            ]
        )
        # Simulator pose CSV stores the rendered camera pose.  For a pitched
        # MID-360 virtual camera, recover the vehicle body pose before sampling
        # body-frame state or evaluating trajectories in the world map:
        # R_WC = R_WB R_BC  =>  R_WB = R_WC R_BC^{-1}.
        rotation_world_body = (
            rotation_world_camera * self.rotation_body_camera.inv()
        )
        yaw, pitch, roll = rotation_world_body.as_euler("ZYX", degrees=False)
        del yaw
        rotation_body_level = R.from_euler("ZYX", [0.0, pitch, roll]).inv()

        rng = self._rng(item)
        velocity_level, acceleration_level = self._get_random_state(rng)
        velocity_body = rotation_body_level.apply(velocity_level)
        acceleration_body = rotation_body_level.apply(acceleration_level)
        goal_body = rotation_body_level.apply(self._get_random_goal(rng))
        observation = np.hstack(
            (velocity_body, acceleration_body, goal_body)
        ).astype(np.float32)

        return (
            image.astype(np.float32),
            self.positions[item],
            rotation_world_body.as_matrix().astype(np.float32),
            observation,
            self.sample_map_ids[item],
        )


def dataset_from_config(config: dict, split: str) -> MapSplitYOPODataset:
    data_config = config["data"]
    split_key = {"train": "train_maps", "valid": "valid_maps", "test": "test_maps"}[split]
    return MapSplitYOPODataset(
        root=data_config["root"],
        map_ids=data_config[split_key],
        mode=split,
        seed=int(config["runtime"]["seed"]),
        camera_pitch_deg=float(data_config.get("camera_pitch_deg", 0.0)),
    )
