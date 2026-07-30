#!/usr/bin/env python3
"""Run the teacher-free physical-anchor LatticeFlow on a LiDAR-only Jetson.

The node keeps the trained policy's single-frame input contract.  It subscribes
to a world-frame registered cloud and fused LiDAR odometry, projects the cloud
into a 96x160 forward depth image, optionally adds a body-relative virtual
ceiling, and publishes the unchanged ``quadrotor_msgs/PositionCommand``
interface used by cfyopo.

Edit the configuration block below.  No M-Detector input is required.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from collections import deque
from pathlib import Path
from threading import Event, Lock, Thread

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent

import rospy  # noqa: E402
import std_msgs.msg  # noqa: E402
import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402
from geometry_msgs.msg import PoseStamped  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402
from sensor_msgs import point_cloud2  # noqa: E402
from sensor_msgs.msg import Image, PointCloud2, PointField  # noqa: E402

from config.config import cfg as yopo_cfg  # noqa: E402
from quadrotor_msgs.msg import PositionCommand  # noqa: E402
from policy.poly_solver import Poly5Solver, Polys5Solver, calculate_yaw  # noqa: E402

from yopo_flow.checkpoint import load_policy_checkpoint  # noqa: E402
from yopo_flow.selection import ContinuityAwareSelector  # noqa: E402


# =============================================================================
# USER CONFIGURATION
# =============================================================================

CHECKPOINT = PROJECT_ROOT / "checkpoints/best.pt"

# The default deployment uses the split FP16 TensorRT export generated directly
# from checkpoints/best.pt by export_lattice_flow_trt_split.py.
INFERENCE_BACKEND = "tensorrt"  # "torch" or "tensorrt"
TRT_ENGINE_METADATA = (
    PROJECT_ROOT / "engines/lattice_flow_nfe6_fp16_metadata.json"
)

NFE = 6
ODOM_TOPIC = "/ekf_quat/ekf_odom"
LIDAR_TOPIC = "/cloud_registered"
CONTROL_TOPIC = "/setpoints_cmd"
GOAL_TOPIC = "/move_base_simple/goal"
WORLD_FRAME = "world"

# The cfyopo point cloud is world-frame.  ``auto`` also accepts a body-frame
# cloud if a different topic is selected later.
LIDAR_FRAME_MODE = "auto"  # "auto", "world", or "body"
LIDAR_ACCUM_FRAMES = 2
# Keep no-return pixels at MAX_DEPTH, matching the simulator.  Only fill an
# invalid pixel when it is locally surrounded by enough valid LiDAR returns.
LIDAR_FILL_ITERATIONS = 1
LIDAR_LOCAL_HOLE_KERNEL = 3
LIDAR_LOCAL_HOLE_MIN_NEIGHBORS = 5
LIDAR_FPS = 10.0

IMAGE_HEIGHT = 96
IMAGE_WIDTH = 160
HORIZONTAL_FOV_DEG = 90.0
VERTICAL_FOV_DEG = 60.0
# ``training_camera`` uses PITCH_ANGLE_DEG and VERTICAL_FOV_DEG.  The MID-360
# mode centers the virtual camera on the LiDAR's actual elevation coverage.
LIDAR_PROJECTION_MODE = "mid360_fov_aligned"  # or "training_camera"
LIDAR_MOUNT_PITCH_DEG = 0.0  # flat installation; positive pitches the LiDAR up
MID360_ELEVATION_MIN_DEG = -7.0
MID360_ELEVATION_MAX_DEG = 52.0
# Policy/trajectory camera-to-body pitch.  Keep independent from LiDAR image
# projection pitch so changing the LiDAR mounting does not rotate trajectories.
PITCH_ANGLE_DEG = 0.0
MIN_DEPTH = 0.05
MAX_DEPTH = 20.0

# The ceiling is added after filling real-LiDAR small holes.  ``world`` uses an
# absolute world-z plane; ``body`` uses a plane above the current vehicle.
VIRTUAL_CEILING_ENABLED = True
VIRTUAL_CEILING_MODE = "world"  # "world" or "body"
VIRTUAL_CEILING_WORLD_Z = 2.0
VIRTUAL_CEILING_BODY_HEIGHT = 2.5
# Full-resolution ceiling rays make this LatticeFlow checkpoint overreact;
# stride 2 preserves the verified geometric plane without dominating the image.
VIRTUAL_CEILING_STRIDE = 2
VIRTUAL_CEILING_MIN_FORWARD = 0.4
VIRTUAL_CEILING_MAX_FORWARD = 0.0  # <=0 means MAX_DEPTH

ARRIVAL_RADIUS = 1.0
# RViz 2D goals have z=0.  A value <=0 means hold the vehicle altitude at the
# instant the goal is received; a positive value is an absolute world height.
GOAL_HEIGHT = 1.5
# The first valid odometry message creates a real startup goal automatically,
# so inference and control do not wait for RViz.  The forward offset follows
# the vehicle yaw at startup; height is an absolute world-z target.
STARTUP_GOAL_ENABLED = True
STARTUP_GOAL_FORWARD_DISTANCE = 4.0
STARTUP_GOAL_WORLD_Z = 1.5
PLAN_FROM_REFERENCE = False
PUBLISH_CONTROL = True
VISUALIZE_TRAJECTORIES = True

CONTINUITY_SELECTOR_ENABLED = False
SELECTOR_ENDPOINT_WEIGHT = 0.02
SELECTOR_HYSTERESIS_MARGIN = 0.05

PUBLISH_DEPTH_DEBUG = True
PUBLISH_DEPTH_CLOUD = True
DEPTH_CLOUD_STRIDE = 2

DIAGNOSTIC_LOG_ENABLED = True
DIAGNOSTIC_LOG_DIR = PROJECT_ROOT / "logs"
DIAGNOSTIC_ROS_THROTTLE_SEC = 0.5

# Runtime timing diagnostics.  Every TIMING_PRINT_EVERY inference frames, print
# both the current-frame latency and the running average.  CUDA is synchronized
# around the network call so the reported inference time includes GPU work.
TIMING_ENABLED = True
TIMING_PRINT_EVERY = 10


class FrontDepthProjector:
    """Project FLU body-frame points into the model's forward depth image."""

    def __init__(
        self,
        height: int,
        width: int,
        horizontal_fov_deg: float,
        vertical_fov_deg: float,
        pitch_deg: float,
        min_depth: float,
        max_depth: float,
        virtual_ceiling: bool,
        virtual_ceiling_mode: str,
        virtual_ceiling_world_z: float,
        virtual_ceiling_body_height: float,
        virtual_ceiling_stride: int,
        virtual_ceiling_min_forward: float,
        virtual_ceiling_max_forward: float,
    ):
        self.height = int(height)
        self.width = int(width)
        self.pitch = np.deg2rad(float(pitch_deg))
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.fx = (self.width / 2.0) / np.tan(np.deg2rad(horizontal_fov_deg) / 2.0)
        self.fy = (self.height / 2.0) / np.tan(np.deg2rad(vertical_fov_deg) / 2.0)
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0
        self.virtual_ceiling_enabled = bool(virtual_ceiling)
        self.virtual_ceiling_mode = str(virtual_ceiling_mode).lower()
        self.virtual_ceiling_world_z = float(virtual_ceiling_world_z)
        self.virtual_ceiling_body_height = float(virtual_ceiling_body_height)
        if self.virtual_ceiling_mode not in {"world", "body"}:
            raise ValueError("virtual_ceiling_mode must be 'world' or 'body'")
        self.virtual_ceiling_stride = max(1, int(virtual_ceiling_stride))
        self.virtual_ceiling_min_forward = max(
            self.min_depth, float(virtual_ceiling_min_forward)
        )
        self.virtual_ceiling_max_forward = (
            self.max_depth
            if float(virtual_ceiling_max_forward) <= 0.0
            else min(self.max_depth, float(virtual_ceiling_max_forward))
        )
        pixels_u = np.arange(0, self.width, dtype=np.float64)
        pixels_v = np.arange(0, self.height, dtype=np.float64)
        uu, vv = np.meshgrid(pixels_u, pixels_v)
        forward_ray = np.ones_like(uu, dtype=np.float64)
        left_ray = (self.cx - uu) / self.fx
        up_ray = (self.cy - vv) / self.fy
        sin_pitch = np.sin(self.pitch)
        cos_pitch = np.cos(self.pitch)
        self._ray_body = np.stack(
            (
                cos_pitch * forward_ray - sin_pitch * up_ray,
                left_ray,
                sin_pitch * forward_ray + cos_pitch * up_ray,
            ),
            axis=-1,
        )
        self._ceiling_sample_mask = np.zeros((self.height, self.width), dtype=bool)
        self._ceiling_sample_mask[
            :: self.virtual_ceiling_stride, :: self.virtual_ceiling_stride
        ] = True

    def apply_virtual_ceiling(
        self,
        depth: np.ndarray,
        rotation_wb: np.ndarray,
        position_w: np.ndarray,
    ) -> np.ndarray:
        if not self.virtual_ceiling_enabled:
            return depth
        if self.virtual_ceiling_mode == "world":
            plane_delta = self.virtual_ceiling_world_z - float(position_w[2])
            if plane_delta <= 0.0:
                return depth
            ray_vertical = (self._ray_body @ rotation_wb.T)[..., 2]
        else:
            plane_delta = self.virtual_ceiling_body_height
            if plane_delta <= 0.0:
                return depth
            ray_vertical = self._ray_body[..., 2]
        valid = ray_vertical > 1.0e-3
        scale = plane_delta / np.where(valid, ray_vertical, np.inf)
        x_body = scale * self._ray_body[..., 0]
        valid &= (
            self._ceiling_sample_mask
            & (scale > self.min_depth)
            & (scale < self.max_depth)
            & (x_body > self.virtual_ceiling_min_forward)
            & (x_body < self.virtual_ceiling_max_forward)
        )
        depth[valid] = np.minimum(depth[valid], scale[valid].astype(depth.dtype))
        return depth

    def project_points(self, points_body: np.ndarray) -> np.ndarray:
        depth = np.full(
            (self.height, self.width), self.max_depth, dtype=np.float32
        )
        points = np.asarray(points_body, dtype=np.float64)
        if points.size:
            sin_pitch = np.sin(self.pitch)
            cos_pitch = np.cos(self.pitch)
            forward = cos_pitch * points[:, 0] + sin_pitch * points[:, 2]
            left = points[:, 1]
            up = -sin_pitch * points[:, 0] + cos_pitch * points[:, 2]
            valid = (
                np.isfinite(points).all(axis=1)
                & (forward > self.min_depth)
                & (forward < self.max_depth)
            )
            if np.any(valid):
                forward = forward[valid]
                left = left[valid]
                up = up[valid]
                pixels_u = np.rint(self.cx - self.fx * left / forward).astype(np.int32)
                pixels_v = np.rint(self.cy - self.fy * up / forward).astype(np.int32)
                in_image = (
                    (pixels_u >= 0)
                    & (pixels_u < self.width)
                    & (pixels_v >= 0)
                    & (pixels_v < self.height)
                )
                np.minimum.at(
                    depth,
                    (pixels_v[in_image], pixels_u[in_image]),
                    forward[in_image].astype(np.float32),
                )
        return depth


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class PhysicalFlowCodec:
    """Small CUDA-PyTorch portion left outside the split TensorRT engines."""

    def __init__(self, state_transform, device: torch.device):
        primitive = state_transform.lattice_primitive
        self.vertical_num = int(primitive.vertical_num)
        self.horizon_num = int(primitive.horizon_num)
        self.position_scale = float(2.0 * primitive.radio_range)
        self.velocity_scale = float(math.sqrt(3.0) * yopo_cfg["vel_max_train"])
        self.acceleration_scale = float(
            math.sqrt(3.0) * yopo_cfg["acc_max_train"]
        )
        self.primitive_velocity_scale = float(primitive.vel_max)
        self.primitive_acceleration_scale = float(primitive.acc_max)
        self.radio_range = float(primitive.radio_range)
        self.yaw_diff = float(primitive.yaw_diff)
        self.pitch_diff = float(primitive.pitch_diff)

        zero_raw = torch.zeros(
            1,
            9,
            self.vertical_num,
            self.horizon_num,
            dtype=torch.float32,
            device=device,
        )
        anchor_body = state_transform.pred_to_endstate(zero_raw)
        self.anchor_source = self._normalize_physical(anchor_body)
        anchor_position = anchor_body[:, 0:3]
        anchor_radius = anchor_position.square().sum(dim=1).sqrt().clamp_min(1e-6)
        self.anchor_yaw = torch.atan2(
            anchor_position[:, 1], anchor_position[:, 0]
        )
        self.anchor_pitch = torch.asin(
            (anchor_position[:, 2] / anchor_radius).clamp(-1.0, 1.0)
        )
        self.rotation_body_from_primitive = primitive.getRotation().flip(0).reshape(
            self.vertical_num, self.horizon_num, 3, 3
        )

    def _normalize_physical(self, endstate_body: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (
                endstate_body[:, 0:3] / self.position_scale,
                endstate_body[:, 3:6] / self.velocity_scale,
                endstate_body[:, 6:9] / self.acceleration_scale,
            ),
            dim=1,
        )

    def canonical_source(self, batch: int, dtype: torch.dtype) -> torch.Tensor:
        return self.anchor_source.to(dtype=dtype).expand(batch, -1, -1, -1)

    @staticmethod
    def project_state(state: torch.Tensor) -> torch.Tensor:
        position = state[:, 0:3]
        norm = position.square().sum(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
        position = position / torch.maximum(norm, torch.ones_like(norm))
        velocity_acceleration = state[:, 3:9].clamp(-1.0, 1.0)
        return torch.cat((position, velocity_acceleration), dim=1)

    def flow_state_to_raw(self, state: torch.Tensor) -> torch.Tensor:
        position = state[:, 0:3] * self.position_scale
        radius = position.square().sum(dim=1).sqrt().clamp_min(1e-6)
        yaw = torch.atan2(position[:, 1], position[:, 0])
        pitch = torch.asin((position[:, 2] / radius).clamp(-1.0, 1.0))
        yaw_delta = torch.atan2(
            torch.sin(yaw - self.anchor_yaw),
            torch.cos(yaw - self.anchor_yaw),
        )
        raw_yaw = yaw_delta / self.yaw_diff
        raw_pitch = (pitch - self.anchor_pitch) / self.pitch_diff
        raw_radius = radius / self.radio_range - 1.0

        rotation_t = self.rotation_body_from_primitive.to(
            device=state.device, dtype=state.dtype
        ).transpose(-1, -2).unsqueeze(0)

        def body_to_primitive(vector: torch.Tensor) -> torch.Tensor:
            vector_grid = vector.permute(0, 2, 3, 1).unsqueeze(-1)
            primitive_grid = torch.matmul(rotation_t, vector_grid).squeeze(-1)
            return primitive_grid.permute(0, 3, 1, 2)

        raw_velocity = body_to_primitive(state[:, 3:6]) * (
            self.velocity_scale / self.primitive_velocity_scale
        )
        raw_acceleration = body_to_primitive(state[:, 6:9]) * (
            self.acceleration_scale / self.primitive_acceleration_scale
        )
        return torch.cat(
            (
                raw_yaw[:, None],
                raw_pitch[:, None],
                raw_radius[:, None],
                raw_velocity,
                raw_acceleration,
            ),
            dim=1,
        ).clamp(-1.0, 1.0)


class TensorRTPolicy:
    """Run the split backbone, flow-step and score TensorRT engines."""

    def __init__(
        self,
        metadata_path: Path,
        checkpoint_path: Path,
        device: torch.device,
        state_transform,
        nfe: int,
    ):
        if device.type != "cuda":
            raise RuntimeError("TensorRT backend requires CUDA")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"TensorRT metadata not found: {metadata_path}")
        self.metadata_path = metadata_path
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("mode") != "split":
            raise ValueError("LatticeFlow TensorRT metadata must use split mode")
        if int(self.metadata.get("nfe", -1)) != int(nfe):
            raise ValueError(
                f"TensorRT NFE={self.metadata.get('nfe')} does not match NFE={nfe}"
            )
        expected_hash = self.metadata.get("checkpoint_sha256")
        actual_hash = file_sha256(checkpoint_path)
        if expected_hash != actual_hash:
            raise ValueError(
                "TensorRT engine/checkpoint mismatch: "
                f"metadata={expected_hash}, checkpoint={actual_hash}"
            )

        from torch2trt import TRTModule

        def resolve_engine(name: str) -> Path:
            path = Path(self.metadata["engines"][name]).expanduser()
            if not path.is_absolute():
                path = metadata_path.parent / path
            if not path.is_file():
                raise FileNotFoundError(f"TensorRT {name} engine not found: {path}")
            return path

        def load_engine(name: str):
            module = TRTModule()
            module.load_state_dict(
                torch.load(str(resolve_engine(name)), map_location=device)
            )
            return module.to(device).eval()

        self.backbone = load_engine("backbone")
        self.flow = load_engine("flow")
        self.score = load_engine("score")
        self.state_transform = state_transform
        self.codec = PhysicalFlowCodec(state_transform, device)
        self.nfe = int(nfe)
        self.dt = 1.0 / float(self.nfe)
        self.lattice_embedding = torch.tensor(
            self.metadata["lattice_embedding"],
            dtype=torch.float32,
            device=device,
        ).contiguous()

        time_dim = int(self.metadata["input_shapes"]["time_features"][1])
        half = time_dim // 2
        frequencies = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half, dtype=torch.float32, device=device)
            / max(half - 1, 1)
        )
        rows = []
        for step in range(self.nfe):
            value = torch.full(
                (1, 1, 3, 5),
                step / float(self.nfe),
                dtype=torch.float32,
                device=device,
            )
            angles = value * frequencies.view(1, half, 1, 1)
            rows.append(torch.cat((angles.sin(), angles.cos()), dim=1))
        self.time_features = torch.stack(rows, dim=0).contiguous()

    @staticmethod
    def _first_output(output):
        return output[0] if isinstance(output, (tuple, list)) else output

    @torch.inference_mode()
    def __call__(self, depth: torch.Tensor, obs_body: torch.Tensor):
        normalized_obs = self.state_transform.normalize_obs(obs_body.clone())
        prepared_obs = self.state_transform.prepare_input(normalized_obs)
        depth_features = self._first_output(self.backbone(depth))
        state = self.codec.canonical_source(depth.shape[0], depth.dtype).contiguous()
        for step in range(self.nfe):
            velocity = self._first_output(
                self.flow(
                    depth_features,
                    prepared_obs,
                    state,
                    self.time_features[step],
                    self.lattice_embedding,
                )
            )
            state = self.codec.project_state(state + self.dt * velocity)
        raw = self.codec.flow_state_to_raw(state)
        score_logits = self._first_output(
            self.score(
                depth_features,
                prepared_obs,
                raw,
                self.lattice_embedding,
            )
        )
        return raw, F.softplus(score_logits.squeeze(1))


class LatticeFlowLidarNode:
    def __init__(self):
        if not CHECKPOINT.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")
        if NFE < 1:
            raise ValueError("NFE must be positive")
        if LIDAR_LOCAL_HOLE_KERNEL < 1 or LIDAR_LOCAL_HOLE_KERNEL % 2 == 0:
            raise ValueError("LIDAR_LOCAL_HOLE_KERNEL must be a positive odd number")
        max_local_neighbors = LIDAR_LOCAL_HOLE_KERNEL**2 - 1
        if not 1 <= LIDAR_LOCAL_HOLE_MIN_NEIGHBORS <= max_local_neighbors:
            raise ValueError(
                "LIDAR_LOCAL_HOLE_MIN_NEIGHBORS must be in "
                f"[1, {max_local_neighbors}]"
            )
        projection_mode = str(LIDAR_PROJECTION_MODE).lower()
        if projection_mode not in {"training_camera", "mid360_fov_aligned"}:
            raise ValueError(
                "LIDAR_PROJECTION_MODE must be 'training_camera' or "
                "'mid360_fov_aligned'"
            )
        if MID360_ELEVATION_MAX_DEG <= MID360_ELEVATION_MIN_DEG:
            raise ValueError(
                "MID360_ELEVATION_MAX_DEG must exceed MID360_ELEVATION_MIN_DEG"
            )

        rospy.init_node("lattice_flow_lidar", anonymous=False)
        yopo_cfg["train"] = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.height = IMAGE_HEIGHT
        self.width = IMAGE_WIDTH
        self.min_depth = MIN_DEPTH
        self.max_depth = MAX_DEPTH
        self.nfe = int(NFE)
        self.goal = np.zeros(3, dtype=np.float64)
        self.goal_received = False
        self.arrive = False
        self.arrival_radius = float(ARRIVAL_RADIUS)
        self.plan_from_reference = bool(PLAN_FROM_REFERENCE)
        self.publish_control = bool(PUBLISH_CONTROL)
        self.visualize = bool(VISUALIZE_TRAJECTORIES)
        self.Rotation_bc = R.from_euler(
            "ZYX", [0.0, PITCH_ANGLE_DEG, 0.0], degrees=True
        ).as_matrix()
        self.world_frame = WORLD_FRAME.lstrip("/")
        self.lidar_frame_mode = LIDAR_FRAME_MODE
        self.projection_mode = projection_mode
        if self.projection_mode == "mid360_fov_aligned":
            self.projection_pitch_deg = float(LIDAR_MOUNT_PITCH_DEG) + 0.5 * (
                float(MID360_ELEVATION_MIN_DEG)
                + float(MID360_ELEVATION_MAX_DEG)
            )
            self.projection_vertical_fov_deg = float(
                MID360_ELEVATION_MAX_DEG - MID360_ELEVATION_MIN_DEG
            )
        else:
            self.projection_pitch_deg = float(PITCH_ANGLE_DEG)
            self.projection_vertical_fov_deg = float(VERTICAL_FOV_DEG)

        self.projector = FrontDepthProjector(
            self.height,
            self.width,
            HORIZONTAL_FOV_DEG,
            self.projection_vertical_fov_deg,
            self.projection_pitch_deg,
            self.min_depth,
            self.max_depth,
            VIRTUAL_CEILING_ENABLED,
            VIRTUAL_CEILING_MODE,
            VIRTUAL_CEILING_WORLD_Z,
            VIRTUAL_CEILING_BODY_HEIGHT,
            VIRTUAL_CEILING_STRIDE,
            VIRTUAL_CEILING_MIN_FORWARD,
            VIRTUAL_CEILING_MAX_FORWARD,
        )

        self.policy = None
        self.model_config = None
        if INFERENCE_BACKEND == "torch":
            self.policy, self.model_config, _ = load_policy_checkpoint(
                str(CHECKPOINT), self.device
            )
            self.policy.eval()
            variant = self.model_config.get("project", {}).get("policy_variant")
            if variant != "physical_anchor":
                raise ValueError(
                    f"Expected physical_anchor checkpoint, got policy_variant={variant!r}"
                )
            self.state_transform = self.policy.state_transform
        elif INFERENCE_BACKEND == "tensorrt":
            self.state_transform = self._make_state_transform()
            self.policy = TensorRTPolicy(
                TRT_ENGINE_METADATA,
                CHECKPOINT,
                self.device,
                self.state_transform,
                self.nfe,
            )
        else:
            raise ValueError(f"Unsupported INFERENCE_BACKEND: {INFERENCE_BACKEND}")

        primitive = self._get_primitive()
        velocity_ratio = float(yopo_cfg["velocity"]) / float(yopo_cfg["vel_max_train"])
        self.traj_time = float(yopo_cfg["sgm_time"]) / max(velocity_ratio, 1e-6)
        self.lattice_traj_num = int(primitive.traj_num)

        self.selector = ContinuityAwareSelector(
            endpoint_weight=SELECTOR_ENDPOINT_WEIGHT,
            hysteresis_margin=SELECTOR_HYSTERESIS_MARGIN,
        )
        self.odom = Odometry()
        self.odom_init = False
        self.desire_init = False
        self.desire_pos = None
        self.desire_vel = None
        self.desire_acc = None
        self.last_yaw = 0.0
        self.ctrl_time = None
        self.cur_traj_time = self.traj_time
        self.optimal_poly_x = None
        self.optimal_poly_y = None
        self.optimal_poly_z = None
        self.last_control_msg = None
        self.lock = Lock()

        self.last_raw_best_id = -1
        self.last_selected_id = -1
        self.last_endpoint_z_body = float("nan")
        self.last_endpoint_dz_world = float("nan")
        self.last_all_endpoint_dz_world_min = float("nan")
        self.last_all_endpoint_dz_world_max = float("nan")
        self.last_goal_rel_z_world = float("nan")
        self.last_goal_rel_z_body = float("nan")
        self.last_depth_raw_valid_ratio = float("nan")
        self.last_depth_input_far_ratio = float("nan")
        self.last_depth_input_mean = float("nan")
        self.diagnostic_lock = Lock()
        self.diagnostic_file = None
        self.diagnostic_writer = None
        self.diagnostic_path = None
        self._open_diagnostic_log()

        self.timing_count = 0
        self.timing_preprocess = 0.0
        self.timing_inference = 0.0
        self.timing_postprocess = 0.0
        self.timing_visualize = 0.0
        self.timing_total = 0.0

        self.lidar_buffer = deque(maxlen=max(1, int(LIDAR_ACCUM_FRAMES)))
        self.lidar_buffer_lock = Lock()
        self.lidar_latest_stamp = None
        self.lidar_new_data = Event()
        self.last_lidar_frame_warning = None

        self.depth_raw_pub = None
        self.depth_input_pub = None
        self.depth_cloud_pub = None
        if PUBLISH_DEPTH_DEBUG:
            self.depth_raw_pub = rospy.Publisher(
                "/lattice_flow/depth_front_raw", Image, queue_size=1
            )
            self.depth_input_pub = rospy.Publisher(
                "/lattice_flow/depth_front", Image, queue_size=1
            )
        if PUBLISH_DEPTH_CLOUD:
            self.depth_cloud_pub = rospy.Publisher(
                "/lattice_flow/depth_cloud_world", PointCloud2, queue_size=1
            )
        self.best_traj_pub = rospy.Publisher(
            "/lattice_flow/best_traj_visual", PointCloud2, queue_size=1
        )
        self.all_trajs_pub = rospy.Publisher(
            "/lattice_flow/trajs_visual", PointCloud2, queue_size=1
        )
        self.ctrl_pub = (
            rospy.Publisher(CONTROL_TOPIC, PositionCommand, queue_size=1)
            if self.publish_control
            else None
        )

        self.odom_sub = rospy.Subscriber(
            ODOM_TOPIC, Odometry, self.callback_odometry, queue_size=1, tcp_nodelay=True
        )
        self.lidar_sub = rospy.Subscriber(
            LIDAR_TOPIC,
            PointCloud2,
            self.callback_lidar,
            queue_size=1,
            tcp_nodelay=True,
        )
        self.goal_sub = rospy.Subscriber(
            GOAL_TOPIC, PoseStamped, self.callback_set_goal, queue_size=1
        )
        self.lidar_thread = Thread(target=self.lidar_inference_loop, daemon=True)
        self.lidar_thread.start()
        self.timer_ctrl = (
            rospy.Timer(rospy.Duration(0.02), self.control_pub)
            if self.publish_control
            else None
        )
        rospy.on_shutdown(self.shutdown_report)
        self.warm_up()

        rospy.loginfo(
            "[LatticeFlow] ready backend=%s checkpoint=%s odom=%s lidar=%s ctrl=%s",
            INFERENCE_BACKEND,
            CHECKPOINT,
            ODOM_TOPIC,
            LIDAR_TOPIC,
            CONTROL_TOPIC,
        )
        rospy.loginfo(
            "[LatticeFlow] depth=%dx%d projection=%s pitch=%.1fdeg FOV=%.1fx%.1f NFE=%d selector=%s",
            self.width,
            self.height,
            self.projection_mode,
            self.projection_pitch_deg,
            HORIZONTAL_FOV_DEG,
            self.projection_vertical_fov_deg,
            self.nfe,
            CONTINUITY_SELECTOR_ENABLED,
        )
        rospy.loginfo(
            "[LatticeFlow] ceiling enabled=%s mode=%s world_z=%.2fm body_height=%.2fm stride=%d",
            VIRTUAL_CEILING_ENABLED,
            VIRTUAL_CEILING_MODE,
            VIRTUAL_CEILING_WORLD_Z,
            VIRTUAL_CEILING_BODY_HEIGHT,
            VIRTUAL_CEILING_STRIDE,
        )
        rospy.loginfo(
            "[LatticeFlow] local hole fill kernel=%d min_neighbors=%d iterations=%d; no-return remains %.1fm",
            LIDAR_LOCAL_HOLE_KERNEL,
            LIDAR_LOCAL_HOLE_MIN_NEIGHBORS,
            LIDAR_FILL_ITERATIONS,
            self.max_depth,
        )

    @staticmethod
    def _get_primitive():
        from policy.primitive import LatticePrimitive

        return LatticePrimitive.get_instance()

    @staticmethod
    def _make_state_transform():
        from policy.state_transform import StateTransform

        return StateTransform()

    def _open_diagnostic_log(self) -> None:
        if not DIAGNOSTIC_LOG_ENABLED:
            return
        DIAGNOSTIC_LOG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.diagnostic_path = DIAGNOSTIC_LOG_DIR / f"runtime_diag_{stamp}.csv"
        self.diagnostic_file = self.diagnostic_path.open(
            "w", newline="", encoding="utf-8", buffering=1
        )
        self.diagnostic_writer = csv.writer(self.diagnostic_file)
        self.diagnostic_writer.writerow(
            [
                "ros_time",
                "event",
                "raw_best_id",
                "selected_id",
                "endpoint_z_body",
                "endpoint_dz_world",
                "all_endpoint_dz_world_min",
                "all_endpoint_dz_world_max",
                "goal_rel_z_world",
                "goal_rel_z_body",
                "odom_z",
                "expected_z",
                "depth_raw_valid_ratio",
                "depth_input_far_ratio",
                "depth_input_mean",
            ]
        )
        rospy.loginfo("[LatticeFlow] diagnostic CSV: %s", self.diagnostic_path)

    def _record_diagnostic(self, event: str, expected_z: float = float("nan")) -> None:
        odom_z = (
            float(self.odom.pose.pose.position.z) if self.odom_init else float("nan")
        )
        with self.diagnostic_lock:
            if (
                self.diagnostic_writer is None
                or self.diagnostic_file is None
                or self.diagnostic_file.closed
            ):
                return
            self.diagnostic_writer.writerow(
                [
                    rospy.Time.now().to_sec(),
                    event,
                    self.last_raw_best_id,
                    self.last_selected_id,
                    self.last_endpoint_z_body,
                    self.last_endpoint_dz_world,
                    self.last_all_endpoint_dz_world_min,
                    self.last_all_endpoint_dz_world_max,
                    self.last_goal_rel_z_world,
                    self.last_goal_rel_z_body,
                    odom_z,
                    expected_z,
                    self.last_depth_raw_valid_ratio,
                    self.last_depth_input_far_ratio,
                    self.last_depth_input_mean,
                ]
            )

    @staticmethod
    def pointcloud_xyz(message: PointCloud2) -> np.ndarray:
        fields = {field.name: field for field in message.fields}
        for name in ("x", "y", "z"):
            if name not in fields:
                raise ValueError(f"PointCloud2 is missing field {name!r}")
            field = fields[name]
            if field.datatype != PointField.FLOAT32 or field.count != 1:
                raise ValueError(f"PointCloud2 field {name!r} must be FLOAT32[1]")
            if field.offset + 4 > message.point_step:
                raise ValueError(f"PointCloud2 field {name!r} exceeds point_step")
        if message.height <= 0 or message.width <= 0 or message.point_step <= 0:
            return np.empty((0, 3), dtype=np.float32)
        dtype = np.dtype(">f4" if message.is_bigendian else "<f4")
        values = []
        for name in ("x", "y", "z"):
            values.append(
                np.ndarray(
                    shape=(message.height, message.width),
                    dtype=dtype,
                    buffer=message.data,
                    offset=fields[name].offset,
                    strides=(message.row_step, message.point_step),
                ).reshape(-1)
            )
        return np.stack(values, axis=1).astype(np.float32, copy=False)

    def callback_set_goal(self, message: PoseStamped) -> None:
        if GOAL_HEIGHT > 0.0:
            goal_z = float(GOAL_HEIGHT)
        elif self.odom_init:
            goal_z = float(self.odom.pose.pose.position.z)
        else:
            goal_z = float(message.pose.position.z)
        self.goal = np.asarray(
            [message.pose.position.x, message.pose.position.y, goal_z],
            dtype=np.float64,
        )
        self.goal_received = True
        self.arrive = False
        self.selector.reset()
        rospy.loginfo(
            "[LatticeFlow] new goal: %s (height_mode=%s)",
            self.goal.tolist(),
            "absolute" if GOAL_HEIGHT > 0.0 else "hold_current",
        )

    def callback_odometry(self, message: Odometry) -> None:
        self.odom = message
        if not self.desire_init:
            self.desire_pos = np.asarray(
                [message.pose.pose.position.x, message.pose.pose.position.y, message.pose.pose.position.z],
                dtype=np.float64,
            )
            self.desire_vel = np.asarray(
                [message.twist.twist.linear.x, message.twist.twist.linear.y, message.twist.twist.linear.z],
                dtype=np.float64,
            )
            self.desire_acc = np.zeros(3, dtype=np.float64)
            quat = message.pose.pose.orientation
            self.last_yaw = R.from_quat([quat.x, quat.y, quat.z, quat.w]).as_euler("ZYX")[0]
        self.odom_init = True
        position = np.asarray(
            [message.pose.pose.position.x, message.pose.pose.position.y, message.pose.pose.position.z],
            dtype=np.float64,
        )
        if STARTUP_GOAL_ENABLED and not self.goal_received:
            quat = message.pose.pose.orientation
            startup_yaw = R.from_quat(
                [quat.x, quat.y, quat.z, quat.w]
            ).as_euler("ZYX")[0]
            self.goal = np.asarray(
                [
                    position[0]
                    + STARTUP_GOAL_FORWARD_DISTANCE * np.cos(startup_yaw),
                    position[1]
                    + STARTUP_GOAL_FORWARD_DISTANCE * np.sin(startup_yaw),
                    STARTUP_GOAL_WORLD_Z,
                ],
                dtype=np.float64,
            )
            self.goal_received = True
            self.arrive = False
            self.selector.reset()
            rospy.loginfo(
                "[LatticeFlow] automatic startup goal: %s (forward=%.1fm world_z=%.2fm); control enabled",
                self.goal.tolist(),
                STARTUP_GOAL_FORWARD_DISTANCE,
                STARTUP_GOAL_WORLD_Z,
            )
        if self.goal_received and np.linalg.norm(position - self.goal) < self.arrival_radius:
            self.arrive = True

    def odom_pose(self):
        quat = self.odom.pose.pose.orientation
        rotation_wb = R.from_quat([quat.x, quat.y, quat.z, quat.w]).as_matrix()
        position_w = np.asarray(
            [self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z],
            dtype=np.float64,
        )
        return rotation_wb, position_w

    def callback_lidar(self, message: PointCloud2) -> None:
        if not self.odom_init:
            return
        try:
            points = self.pointcloud_xyz(message)
        except (TypeError, ValueError) as error:
            rospy.logwarn_throttle(2.0, "[LatticeFlow] invalid LiDAR cloud: %s", error)
            return
        points = points[np.isfinite(points).all(axis=1)]
        if points.size == 0:
            return
        source_frame = (message.header.frame_id or "").lstrip("/")
        odom_frame = (self.odom.header.frame_id or "").lstrip("/")
        world_frames = {self.world_frame, odom_frame}
        world_frames.discard("")
        use_world = self.lidar_frame_mode == "world" or (
            self.lidar_frame_mode == "auto" and source_frame in world_frames
        )
        if use_world:
            points_world = points.astype(np.float64, copy=False)
        else:
            rotation_wb, position_w = self.odom_pose()
            points_world = points.astype(np.float64) @ rotation_wb.T + position_w
            if self.lidar_frame_mode == "auto" and source_frame not in ("", "body", "base_link"):
                if source_frame != self.last_lidar_frame_warning:
                    rospy.logwarn(
                        "[LatticeFlow] treating frame %s as body frame; set LIDAR_FRAME_MODE explicitly if needed",
                        source_frame,
                    )
                    self.last_lidar_frame_warning = source_frame
        with self.lidar_buffer_lock:
            self.lidar_buffer.append(points_world)
            self.lidar_latest_stamp = message.header.stamp
        self.lidar_new_data.set()

    def densify_sparse_depth(self, depth: np.ndarray) -> np.ndarray:
        result = np.asarray(depth, dtype=np.float32).copy()
        if LIDAR_FILL_ITERATIONS <= 0:
            return result
        kernel_size = int(LIDAR_LOCAL_HOLE_KERNEL)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (kernel_size, kernel_size)
        )
        for _ in range(int(LIDAR_FILL_ITERATIONS)):
            valid = (
                np.isfinite(result)
                & (result > self.min_depth)
                & (result < self.max_depth * 0.999)
            )
            invalid = ~valid
            if not np.any(invalid) or not np.any(valid):
                break
            neighbor_count = cv2.boxFilter(
                valid.astype(np.float32),
                ddepth=-1,
                ksize=(kernel_size, kernel_size),
                normalize=False,
                borderType=cv2.BORDER_CONSTANT,
            )
            local_minimum = cv2.erode(result, kernel, iterations=1)
            fill_mask = (
                invalid
                & (neighbor_count >= float(LIDAR_LOCAL_HOLE_MIN_NEIGHBORS))
                & np.isfinite(local_minimum)
                & (local_minimum > self.min_depth)
                & (local_minimum < self.max_depth * 0.999)
            )
            if not np.any(fill_mask):
                break
            result[fill_mask] = local_minimum[fill_mask]
        return result

    def normalize_depth(self, depth: np.ndarray) -> np.ndarray:
        normalized = np.clip(depth / self.max_depth, 0.0, 1.0).astype(np.float32)
        # Match the simulator callback: MAX_DEPTH is a valid far observation.
        # Only non-finite or physically invalid near-zero pixels are inpainted.
        invalid = (
            ~np.isfinite(normalized)
            | (normalized < self.min_depth / self.max_depth)
        )
        if np.all(invalid):
            return np.ones_like(normalized, dtype=np.float32)
        if np.any(invalid):
            source = np.uint8(
                np.nan_to_num(normalized, nan=1.0, posinf=1.0, neginf=0.0) * 255
            )
            normalized = cv2.inpaint(
                source, np.uint8(invalid), 1, cv2.INPAINT_NS
            ).astype(np.float32) / 255.0
        return normalized.astype(np.float32)

    @staticmethod
    def make_float_image(image: np.ndarray, stamp, frame_id: str = "body") -> Image:
        array = np.ascontiguousarray(image, dtype=np.float32)
        message = Image()
        message.header.stamp = stamp if stamp is not None else rospy.Time.now()
        message.header.frame_id = frame_id
        message.height = array.shape[0]
        message.width = array.shape[1]
        message.encoding = "32FC1"
        message.is_bigendian = False
        message.step = array.shape[1] * array.dtype.itemsize
        message.data = array.tobytes()
        return message

    def publish_depth_cloud(self, raw_depth: np.ndarray, rotation_wb, position_w, stamp) -> None:
        if self.depth_cloud_pub is None or self.depth_cloud_pub.get_num_connections() <= 0:
            return
        stride = max(1, int(DEPTH_CLOUD_STRIDE))
        depth = raw_depth[::stride, ::stride]
        vv, uu = np.mgrid[0:self.height:stride, 0:self.width:stride]
        valid = np.isfinite(depth) & (depth > self.min_depth) & (depth < self.max_depth * 0.999)
        if not np.any(valid):
            return
        forward = depth[valid].astype(np.float64)
        u = uu[valid].astype(np.float64)
        v = vv[valid].astype(np.float64)
        left = (self.projector.cx - u) * forward / self.projector.fx
        up = (self.projector.cy - v) * forward / self.projector.fy
        sin_pitch = np.sin(self.projector.pitch)
        cos_pitch = np.cos(self.projector.pitch)
        x_body = cos_pitch * forward - sin_pitch * up
        z_body = sin_pitch * forward + cos_pitch * up
        points_body = np.stack((x_body, left, z_body), axis=1)
        points_world = points_body @ rotation_wb.T + position_w
        header = std_msgs.msg.Header()
        header.stamp = stamp if stamp is not None else rospy.Time.now()
        header.frame_id = self.world_frame
        self.depth_cloud_pub.publish(point_cloud2.create_cloud_xyz32(header, points_world.astype(np.float32)))

    def process_odom(self) -> torch.Tensor:
        quat = self.odom.pose.pose.orientation
        rotation_wb = R.from_quat([quat.x, quat.y, quat.z, quat.w]).as_matrix()
        rotation_wc = rotation_wb @ self.Rotation_bc
        rotation_cw = rotation_wc.T
        velocity_w = self.desire_vel if self.plan_from_reference else np.asarray(
            [self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z],
            dtype=np.float64,
        )
        acceleration_w = self.desire_acc
        velocity_c = rotation_cw @ velocity_w
        acceleration_c = rotation_cw @ acceleration_w
        if self.goal_received:
            goal_delta_w = self.goal - self.desire_pos
        else:
            goal_delta_w = rotation_wb @ np.asarray(
                [STARTUP_GOAL_FORWARD_DISTANCE, 0.0, 0.0], dtype=np.float64
            )
        goal_c = rotation_cw @ goal_delta_w
        return torch.from_numpy(np.concatenate((velocity_c, acceleration_c, goal_c), axis=0).astype(np.float32)[None, :])

    @torch.inference_mode()
    def infer(self, depth_input: np.ndarray):
        depth = torch.from_numpy(depth_input[None, None]).to(self.device)
        obs_body = self.process_odom().to(self.device)
        if INFERENCE_BACKEND == "torch":
            prepared = self.policy.prepare_observation(obs_body)
            raw, score = self.policy(depth, prepared, num_steps=self.nfe)
        else:
            raw, score = self.policy(depth, obs_body)
        return raw[0].detach().cpu().numpy(), score[0].detach().cpu().numpy()

    def choose_endpoint(self, raw: np.ndarray, score: np.ndarray):
        raw_cells = raw.reshape(9, self.lattice_traj_num).T
        scores = score.reshape(self.lattice_traj_num).astype(np.float64)
        lattice_ids = torch.arange(self.lattice_traj_num - 1, -1, -1)
        endpoints = self.state_transform.pred_to_endstate_cpu(raw_cells, lattice_ids)
        if CONTINUITY_SELECTOR_ENABLED:
            selection = self.selector.select(scores, endpoints)
            return endpoints, selection.index, selection.adjusted_scores
        return endpoints, int(np.argmin(scores)), scores

    def run_inference(self, normalized_depth: np.ndarray, stamp) -> tuple:
        rotation_wb, position_w = self.odom_pose()
        rotation_wc = rotation_wb @ self.Rotation_bc

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        inference_start = time.perf_counter()
        raw, score = self.infer(normalized_depth)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        inference_end = time.perf_counter()

        postprocess_start = inference_end
        endpoints, action_id, display_scores = self.choose_endpoint(raw, score)
        raw_best_id = int(np.argmin(score.reshape(-1)))
        endpoints_c = endpoints.reshape(-1, 3, 3).transpose(0, 2, 1)
        endpoints_w = np.matmul(rotation_wc, endpoints_c)
        start_pos = self.desire_pos if self.plan_from_reference else position_w
        start_vel = self.desire_vel if self.plan_from_reference else np.asarray(
            [self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z],
            dtype=np.float64,
        )
        goal_reference = self.desire_pos if self.desire_pos is not None else position_w
        if self.goal_received:
            goal_relative_world = self.goal - goal_reference
        else:
            goal_relative_world = rotation_wb @ np.asarray(
                [STARTUP_GOAL_FORWARD_DISTANCE, 0.0, 0.0], dtype=np.float64
            )
        goal_relative_body = rotation_wc.T @ goal_relative_world
        self.last_raw_best_id = raw_best_id
        self.last_selected_id = int(action_id)
        self.last_endpoint_z_body = float(endpoints[action_id, 2])
        self.last_endpoint_dz_world = float(endpoints_w[action_id, 2, 0])
        self.last_all_endpoint_dz_world_min = float(np.min(endpoints_w[:, 2, 0]))
        self.last_all_endpoint_dz_world_max = float(np.max(endpoints_w[:, 2, 0]))
        self.last_goal_rel_z_world = float(goal_relative_world[2])
        self.last_goal_rel_z_body = float(goal_relative_body[2])
        self._record_diagnostic("inference")
        # rospy.loginfo_throttle(
        #     DIAGNOSTIC_ROS_THROTTLE_SEC,
        #     "[LatticeFlowDiag] raw_best_id=%d selected_id=%d endpoint_z_body=%.3f endpoint_dz_world=%.3f all_endpoint_dz_world=[%.3f,%.3f] goal_rel_z_world=%.3f goal_rel_z_body=%.3f odom_z=%.3f",
        #     self.last_raw_best_id,
        #     self.last_selected_id,
        #     self.last_endpoint_z_body,
        #     self.last_endpoint_dz_world,
        #     self.last_all_endpoint_dz_world_min,
        #     self.last_all_endpoint_dz_world_max,
        #     self.last_goal_rel_z_world,
        #     self.last_goal_rel_z_body,
        #     float(position_w[2]),
        # )
        with self.lock:
            self.optimal_poly_x = Poly5Solver(start_pos[0], start_vel[0], self.desire_acc[0], endpoints_w[action_id, 0, 0] + start_pos[0], endpoints_w[action_id, 0, 1], endpoints_w[action_id, 0, 2], self.traj_time)
            self.optimal_poly_y = Poly5Solver(start_pos[1], start_vel[1], self.desire_acc[1], endpoints_w[action_id, 1, 0] + start_pos[1], endpoints_w[action_id, 1, 1], endpoints_w[action_id, 1, 2], self.traj_time)
            self.optimal_poly_z = Poly5Solver(start_pos[2], start_vel[2], self.desire_acc[2], endpoints_w[action_id, 2, 0] + start_pos[2], endpoints_w[action_id, 2, 1], endpoints_w[action_id, 2, 2], self.traj_time)
            self.ctrl_time = 0.0
            self.cur_traj_time = self.traj_time
        postprocess_end = time.perf_counter()

        visualize_start = postprocess_end
        self.visualize_trajectories(endpoints_w, display_scores, start_pos, start_vel, stamp)
        visualize_end = time.perf_counter()
        return (
            inference_end - inference_start,
            postprocess_end - postprocess_start,
            visualize_end - visualize_start,
        )

    def print_time(
        self,
        preprocess_time: float,
        inference_time: float,
        postprocess_time: float,
        visualize_time: float,
        total_time: float,
    ) -> None:
        """Print current and running-average latency for the inference pipeline."""
        if not TIMING_ENABLED:
            return

        self.timing_preprocess += preprocess_time
        self.timing_inference += inference_time
        self.timing_postprocess += postprocess_time
        self.timing_visualize += visualize_time
        self.timing_total += total_time
        self.timing_count += 1

        print_every = max(1, int(TIMING_PRINT_EVERY))
        budget_ms = 1000.0 / max(float(LIDAR_FPS), 1e-6)
        total_ms = total_time * 1000.0
        over_budget = total_ms > budget_ms
        if self.timing_count % print_every != 0 and not over_budget:
            return

        count = float(self.timing_count)
        values = (
            preprocess_time * 1000.0,
            inference_time * 1000.0,
            postprocess_time * 1000.0,
            visualize_time * 1000.0,
            total_ms,
            self.timing_preprocess * 1000.0 / count,
            self.timing_inference * 1000.0 / count,
            self.timing_postprocess * 1000.0 / count,
            self.timing_visualize * 1000.0 / count,
            self.timing_total * 1000.0 / count,
        )
        message = (
            "[LatticeFlowTime] current(ms): preprocess=%.2f inference=%.2f "
            "postprocess=%.2f trajectory_visualize=%.2f total=%.2f | "
            "average[%d](ms): preprocess=%.2f inference=%.2f postprocess=%.2f "
            "trajectory_visualize=%.2f total=%.2f"
        )
        args = values[:5] + (self.timing_count,) + values[5:]
        if over_budget:
            rospy.logwarn_throttle(
                1.0,
                message + " (budget=%.2f ms)",
                *(args + (budget_ms,)),
            )
        else:
            rospy.loginfo(message, *args)

    def lidar_inference_loop(self) -> None:
        while not rospy.is_shutdown():
            if not self.lidar_new_data.wait(timeout=1.0):
                continue
            self.lidar_new_data.clear()
            if not self.odom_init:
                continue
            frame_start = time.perf_counter()
            with self.lidar_buffer_lock:
                if not self.lidar_buffer:
                    continue
                points_world = np.concatenate(list(self.lidar_buffer), axis=0)
                stamp = self.lidar_latest_stamp
            rotation_wb, position_w = self.odom_pose()
            points_body = (points_world - position_w) @ rotation_wb
            raw_depth = self.projector.project_points(points_body)
            real_valid = (
                np.isfinite(raw_depth)
                & (raw_depth > self.min_depth)
                & (raw_depth < self.max_depth * 0.999)
            )
            # Fill only holes supported by real LiDAR returns.  Add the
            # synthetic ceiling afterwards so it cannot seed hole expansion.
            dense_depth = self.densify_sparse_depth(raw_depth)
            model_depth = self.projector.apply_virtual_ceiling(
                dense_depth, rotation_wb, position_w
            )
            normalized_depth = self.normalize_depth(model_depth)
            if self.depth_raw_pub is not None:
                self.depth_raw_pub.publish(self.make_float_image(np.clip(raw_depth / self.max_depth, 0.0, 1.0), stamp))
            if self.depth_input_pub is not None:
                self.depth_input_pub.publish(self.make_float_image(normalized_depth, stamp))
            self.publish_depth_cloud(model_depth, rotation_wb, position_w, stamp)
            model_nonfar = normalized_depth < 0.999
            self.last_depth_raw_valid_ratio = float(np.mean(real_valid))
            self.last_depth_input_far_ratio = float(np.mean(~model_nonfar))
            self.last_depth_input_mean = float(np.mean(normalized_depth))
            preprocess_end = time.perf_counter()
            rospy.loginfo_throttle(
                2.0,
                "[LatticeFlow] real lidar valid=%d/%d (%.1f%%), model_nonfar=%.1f%% far=%.1f%% mean=%.3f min=%.2fm",
                int(np.count_nonzero(real_valid)),
                int(raw_depth.size),
                100.0 * self.last_depth_raw_valid_ratio,
                100.0 * float(np.mean(model_nonfar)),
                100.0 * self.last_depth_input_far_ratio,
                self.last_depth_input_mean,
                float(np.min(raw_depth[real_valid])) if np.any(real_valid) else self.max_depth,
            )
            if not self.goal_received:
                rospy.loginfo_throttle(
                    2.0,
                    "[LatticeFlow] no goal: inference active with %.1fm forward virtual goal; control output disabled",
                    STARTUP_GOAL_FORWARD_DISTANCE,
                )
            if not np.any(real_valid):
                rospy.logwarn_throttle(
                    2.0, "[LatticeFlow] no real LiDAR return in the forward image; skipping inference"
                )
                continue
            try:
                inference_time, postprocess_time, visualize_time = self.run_inference(
                    normalized_depth, stamp
                )
                total_time = time.perf_counter() - frame_start
                self.print_time(
                    preprocess_end - frame_start,
                    inference_time,
                    postprocess_time,
                    visualize_time,
                    total_time,
                )
            except Exception as error:
                rospy.logerr_throttle(2.0, "[LatticeFlow] inference failed: %s", error)

    def visualize_trajectories(self, endpoints_w, scores, start_pos, start_vel, stamp) -> None:
        if self.best_traj_pub.get_num_connections() > 0 and self.optimal_poly_x is not None:
            t_values = np.arange(0.0, self.traj_time, self.traj_time / 20.0)
            best = np.stack((self.optimal_poly_x.get_position(t_values), self.optimal_poly_y.get_position(t_values), self.optimal_poly_z.get_position(t_values)), axis=-1)
            header = std_msgs.msg.Header(stamp=stamp if stamp is not None else rospy.Time.now(), frame_id=self.world_frame)
            self.best_traj_pub.publish(point_cloud2.create_cloud_xyz32(header, best.astype(np.float32)))
        if not self.visualize or self.all_trajs_pub.get_num_connections() <= 0:
            return
        t_values = np.arange(0.0, self.traj_time, self.traj_time / 20.0)
        all_x = Polys5Solver(start_pos[0], start_vel[0], self.desire_acc[0], endpoints_w[:, 0, 0] + start_pos[0], endpoints_w[:, 0, 1], endpoints_w[:, 0, 2], self.traj_time)
        all_y = Polys5Solver(start_pos[1], start_vel[1], self.desire_acc[1], endpoints_w[:, 1, 0] + start_pos[1], endpoints_w[:, 1, 1], endpoints_w[:, 1, 2], self.traj_time)
        all_z = Polys5Solver(start_pos[2], start_vel[2], self.desire_acc[2], endpoints_w[:, 2, 0] + start_pos[2], endpoints_w[:, 2, 1], endpoints_w[:, 2, 2], self.traj_time)
        points = np.stack((all_x.get_position(t_values), all_y.get_position(t_values), all_z.get_position(t_values)), axis=-1)
        intensity = np.repeat(scores.reshape(-1), t_values.size)
        points = np.column_stack((points.reshape(-1, 3), intensity))
        fields = [PointField("x", 0, PointField.FLOAT32, 1), PointField("y", 4, PointField.FLOAT32, 1), PointField("z", 8, PointField.FLOAT32, 1), PointField("intensity", 12, PointField.FLOAT32, 1)]
        header = std_msgs.msg.Header(stamp=stamp if stamp is not None else rospy.Time.now(), frame_id=self.world_frame)
        self.all_trajs_pub.publish(point_cloud2.create_cloud(header, fields, points.astype(np.float32)))

    def control_pub(self, _event) -> None:
        if not self.goal_received:
            return
        if self.ctrl_pub is None or self.ctrl_time is None or self.optimal_poly_x is None:
            return
        if self.ctrl_time > self.cur_traj_time:
            return
        if self.arrive and self.last_control_msg is not None:
            self.desire_init = False
            self.last_control_msg.trajectory_flag = self.last_control_msg.TRAJECTORY_STATUS_EMPTY
            self.ctrl_pub.publish(self.last_control_msg)
            return
        with self.lock:
            self.ctrl_time += 0.02
            message = PositionCommand()
            message.header.stamp = rospy.Time.now()
            message.trajectory_flag = message.TRAJECTORY_STATUS_READY
            message.position.x = self.optimal_poly_x.get_position(self.ctrl_time)
            message.position.y = self.optimal_poly_y.get_position(self.ctrl_time)
            message.position.z = self.optimal_poly_z.get_position(self.ctrl_time)
            message.velocity.x = self.optimal_poly_x.get_velocity(self.ctrl_time)
            message.velocity.y = self.optimal_poly_y.get_velocity(self.ctrl_time)
            message.velocity.z = self.optimal_poly_z.get_velocity(self.ctrl_time)
            message.acceleration.x = self.optimal_poly_x.get_acceleration(self.ctrl_time)
            message.acceleration.y = self.optimal_poly_y.get_acceleration(self.ctrl_time)
            message.acceleration.z = self.optimal_poly_z.get_acceleration(self.ctrl_time)
            self.desire_pos = np.asarray([message.position.x, message.position.y, message.position.z])
            self.desire_vel = np.asarray([message.velocity.x, message.velocity.y, message.velocity.z])
            self.desire_acc = np.asarray([message.acceleration.x, message.acceleration.y, message.acceleration.z])
            yaw, yaw_dot = calculate_yaw(self.desire_vel, self.goal - self.desire_pos, self.last_yaw, 0.02)
            self.last_yaw = yaw
            message.yaw = yaw
            message.yaw_dot = yaw_dot
            self.desire_init = True
            self.last_control_msg = message
            self._record_diagnostic("control", expected_z=float(message.position.z))
            # rospy.loginfo_throttle(
            #     DIAGNOSTIC_ROS_THROTTLE_SEC,
            #     "[LatticeFlowCtrl] raw_best_id=%d selected_id=%d endpoint_dz_world=%.3f goal_rel_z_world=%.3f odom_z=%.3f expected_z=%.3f expected_vz=%.3f",
            #     self.last_raw_best_id,
            #     self.last_selected_id,
            #     self.last_endpoint_dz_world,
            #     self.last_goal_rel_z_world,
            #     float(self.odom.pose.pose.position.z),
            #     float(message.position.z),
            #     float(message.velocity.z),
            # )
            self.ctrl_pub.publish(message)

    def warm_up(self) -> None:
        depth = torch.zeros((1, 1, self.height, self.width), dtype=torch.float32, device=self.device)
        obs = torch.zeros((1, 9), dtype=torch.float32, device=self.device)
        if INFERENCE_BACKEND == "torch":
            prepared = self.policy.prepare_observation(obs)
            self.policy(depth, prepared, num_steps=self.nfe)
        else:
            self.policy(depth, obs)
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def shutdown_report(self) -> None:
        rospy.loginfo(
            "[LatticeFlow] selector switches=%d rate=%.4f mean_jump=%.3fm max_jump=%.3fm",
            self.selector.switch_count,
            self.selector.switch_rate,
            self.selector.mean_endpoint_jump,
            self.selector.max_endpoint_jump,
        )
        if self.timing_count > 0:
            count = float(self.timing_count)
            rospy.loginfo(
                "[LatticeFlowTime] final average[%d](ms): preprocess=%.2f inference=%.2f postprocess=%.2f trajectory_visualize=%.2f total=%.2f",
                self.timing_count,
                self.timing_preprocess * 1000.0 / count,
                self.timing_inference * 1000.0 / count,
                self.timing_postprocess * 1000.0 / count,
                self.timing_visualize * 1000.0 / count,
                self.timing_total * 1000.0 / count,
            )
        if self.diagnostic_file is not None:
            with self.diagnostic_lock:
                self.diagnostic_writer = None
                self.diagnostic_file.flush()
                self.diagnostic_file.close()
                self.diagnostic_file = None
            rospy.loginfo("[LatticeFlow] diagnostic CSV saved: %s", self.diagnostic_path)


if __name__ == "__main__":
    LatticeFlowLidarNode()
    rospy.spin()
