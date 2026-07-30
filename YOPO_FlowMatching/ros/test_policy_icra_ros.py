#!/usr/bin/env python3
"""Matched ROS closed-loop evaluator for fair YOPO and LatticeFlow policies."""

from __future__ import annotations

import argparse
import atexit
import json
import sys
import time
from pathlib import Path
from threading import Lock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import open3d as o3d
import rospy
import torch
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image, PointCloud2

from yopo_flow.bootstrap import add_original_yopo_to_path
from yopo_flow.checkpoint import load_policy_checkpoint
from yopo_flow.selection import ContinuityAwareSelector

add_original_yopo_to_path()

from config.config import cfg  # noqa: E402
from control_msg import PositionCommand  # noqa: E402
from policy.primitive import LatticePrimitive  # noqa: E402
from policy.state_transform import StateTransform  # noqa: E402
from policy.yopo_network import YopoNetwork  # noqa: E402
from test_yopo_ros import YopoNet as OriginalYopoNet  # noqa: E402


class ICRAPolicyNode(OriginalYopoNet):
    def __init__(self, settings: dict):
        self.settings = settings
        self.policy_kind = settings["policy"]
        self.checkpoint_path = str(Path(settings["checkpoint"]).resolve())
        self.result_md = Path(settings["result_md"])
        self.result_json = Path(settings["result_json"])
        self.arrival_distance = float(settings["arrival_distance"])
        self.start_wall_time = time.time()
        self.path_length = 0.0
        self.previous_position = None
        self.min_observed_depth = float("inf")
        self.min_geometry_distance = float("inf")
        self.geometry_collision = False
        self.arrival_time = None
        self.report_written = False
        self.speed_samples: list[float] = []
        self.jerk_samples: list[float] = []
        self.previous_command_acceleration = None
        self.selector = ContinuityAwareSelector(
            endpoint_weight=float(settings["selector_endpoint_weight"]),
            hysteresis_margin=float(settings["selector_hysteresis_margin"]),
        )
        self.use_selector = not bool(settings["disable_continuity_selector"])

        self.collision_tree = None
        collision_map = settings.get("collision_map")
        if collision_map:
            pointcloud = o3d.io.read_point_cloud(str(collision_map))
            if len(pointcloud.points) == 0:
                raise ValueError(f"Collision map contains no points: {collision_map}")
            voxel = float(settings["collision_voxel"])
            if voxel > 0.0:
                pointcloud = pointcloud.voxel_down_sample(voxel)
            points = np.asarray(pointcloud.points, dtype=np.float32)
            self.collision_tree = cKDTree(points)
            print(f"Collision map loaded: {len(points)} points from {collision_map}")

        rospy.init_node(f"icra_{self.policy_kind}_planner", anonymous=False)
        cfg["train"] = False
        requested_velocity = settings.get("runtime_velocity")
        if requested_velocity is not None:
            if float(requested_velocity) <= 0.0:
                raise ValueError("runtime_velocity must be positive")
            cfg["velocity"] = float(requested_velocity)
        self.runtime_velocity = float(cfg["velocity"])
        self.height = int(cfg["image_height"])
        self.width = int(cfg["image_width"])
        self.min_dis, self.max_dis = 0.04, 20.0
        self.goal = np.asarray(settings["goal"], dtype=np.float64)
        self.plan_from_reference = bool(settings["plan_from_reference"])
        self.use_trt = False
        self.verbose = bool(settings["verbose"])
        self.visualize = bool(settings["visualize"])
        self.Rotation_bc = R.from_euler(
            "ZYX", [0.0, float(settings["pitch_angle_deg"]), 0.0], degrees=True
        ).as_matrix()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.odom = Odometry()
        self.odom_init = False
        self.last_yaw = 0.0
        self.ctrl_dt = 0.02
        self.ctrl_time = None
        self.desire_init = False
        self.arrive = False
        self.desire_pos = None
        self.desire_vel = None
        self.desire_acc = None
        self.optimal_poly_x = None
        self.optimal_poly_y = None
        self.optimal_poly_z = None
        self.lock = Lock()
        self.last_control_msg = None
        self.state_transform = StateTransform()
        self.lattice_primitive = LatticePrimitive.get_instance()
        self.traj_time = self.lattice_primitive.segment_time
        print(
            "Runtime primitive: "
            f"velocity={self.runtime_velocity:.3f}m/s, "
            f"vel_max={self.lattice_primitive.vel_max:.3f}m/s, "
            f"acc_max={self.lattice_primitive.acc_max:.3f}m/s^2, "
            f"segment_time={self.traj_time:.3f}s"
        )

        self.time_forward = 0.0
        self.time_process = 0.0
        self.time_prepare = 0.0
        self.time_interpolation = 0.0
        self.time_visualize = 0.0
        self.count = 0
        self.depth_fps = 30

        if self.policy_kind == "latticeflow":
            self.policy, self.flow_config, _ = load_policy_checkpoint(
                self.checkpoint_path, self.device
            )
        else:
            self.flow_config = {"evaluation": {"collision_distance": settings["depth_collision_distance"]}}
            state_dict = torch.load(
                self.checkpoint_path, map_location=self.device, weights_only=True
            )
            self.policy = YopoNetwork().to(self.device)
            self.policy.load_state_dict(state_dict)
            self.policy.eval()
        self.policy.eval()
        self.warm_up()

        self.lattice_traj_pub = rospy.Publisher(
            f"/icra_{self.policy_kind}/lattice_trajs_visual", PointCloud2, queue_size=1
        )
        self.best_traj_pub = rospy.Publisher(
            f"/icra_{self.policy_kind}/best_traj_visual", PointCloud2, queue_size=1
        )
        self.all_trajs_pub = rospy.Publisher(
            f"/icra_{self.policy_kind}/trajs_visual", PointCloud2, queue_size=1
        )
        self.ctrl_pub = rospy.Publisher(
            settings["ctrl_topic"], PositionCommand, queue_size=1
        )
        self.odom_sub = rospy.Subscriber(
            settings["odom_topic"], Odometry, self.callback_odometry,
            queue_size=1, tcp_nodelay=True
        )
        self.depth_sub = rospy.Subscriber(
            settings["depth_topic"], Image, self.callback_depth,
            queue_size=1, tcp_nodelay=True
        )
        self.goal_sub = rospy.Subscriber(
            "/move_base_simple/goal", PoseStamped, self.callback_set_goal, queue_size=1
        )
        rospy.on_shutdown(self.write_result_report)
        atexit.register(self.write_result_report)
        rospy.sleep(1.0)
        self.timer_ctrl = rospy.Timer(rospy.Duration(self.ctrl_dt), self.control_pub)
        max_runtime = float(settings["max_runtime"])
        if max_runtime > 0.0:
            self.timeout_timer = rospy.Timer(
                rospy.Duration(max_runtime), self._timeout, oneshot=True
            )
        print(f"ICRA {self.policy_kind} node ready; goal={self.goal.tolist()}")
        rospy.spin()

    def _timeout(self, _event):
        rospy.signal_shutdown("maximum trial runtime reached")

    def callback_set_goal(self, data):
        self.goal = np.asarray(
            [data.pose.position.x, data.pose.position.y, data.pose.position.z],
            dtype=np.float64,
        )
        self.arrive = False
        self.arrival_time = None
        self.selector.reset()
        print(f"New goal: {self.goal.tolist()}")

    def callback_odometry(self, data):
        self.odom = data
        if not self.desire_init:
            self.desire_pos = np.asarray(
                [
                    data.pose.pose.position.x,
                    data.pose.pose.position.y,
                    data.pose.pose.position.z,
                ],
                dtype=np.float64,
            )
            self.desire_vel = np.asarray(
                [
                    data.twist.twist.linear.x,
                    data.twist.twist.linear.y,
                    data.twist.twist.linear.z,
                ],
                dtype=np.float64,
            )
            self.desire_acc = np.zeros(3, dtype=np.float64)
            orientation = data.pose.pose.orientation
            self.last_yaw = R.from_quat(
                [orientation.x, orientation.y, orientation.z, orientation.w]
            ).as_euler("ZYX", degrees=False)[0]
        self.odom_init = True

        position = np.asarray(
            [
                data.pose.pose.position.x,
                data.pose.pose.position.y,
                data.pose.pose.position.z,
            ],
            dtype=np.float64,
        )
        velocity = np.asarray(
            [
                data.twist.twist.linear.x,
                data.twist.twist.linear.y,
                data.twist.twist.linear.z,
            ],
            dtype=np.float64,
        )
        self.speed_samples.append(float(np.linalg.norm(velocity)))
        if self.previous_position is not None:
            self.path_length += float(np.linalg.norm(position - self.previous_position))
        self.previous_position = position

        if self.collision_tree is not None:
            distance = float(self.collision_tree.query(position, k=1, workers=1)[0])
            self.min_geometry_distance = min(self.min_geometry_distance, distance)
            if distance < float(self.settings["vehicle_radius"]):
                self.geometry_collision = True
                if self.settings["stop_on_collision"]:
                    rospy.signal_shutdown("geometric collision")

        if np.linalg.norm(position - self.goal) < self.arrival_distance and not self.arrive:
            self.arrive = True
            self.arrival_time = time.time() - self.start_wall_time
            print(f"Goal reached within {self.arrival_distance:.2f} m")
            if self.settings["stop_on_arrival"]:
                rospy.signal_shutdown("goal reached")

    def callback_depth(self, data):
        if data.encoding == "32FC1":
            depth = np.frombuffer(data.data, dtype=np.float32)
        elif data.encoding == "16UC1":
            depth = np.frombuffer(data.data, dtype=np.uint16).astype(np.float32) / 1000.0
        else:
            depth = np.empty(0, dtype=np.float32)
        valid = depth[np.isfinite(depth) & (depth > self.min_dis)]
        if valid.size:
            self.min_observed_depth = min(self.min_observed_depth, float(valid.min()))
        super().callback_depth(data)

    def process_output(self, endstate_pred, score_pred, return_all_preds=False):
        raw = endstate_pred.reshape(9, self.lattice_primitive.traj_num).T
        scores = score_pred.reshape(self.lattice_primitive.traj_num).astype(np.float64)
        lattice_ids = torch.arange(
            self.lattice_primitive.traj_num - 1, -1, -1
        )
        endpoints = self.state_transform.pred_to_endstate_cpu(raw, lattice_ids)
        if self.use_selector:
            selection = self.selector.select(scores, endpoints)
            action_id = selection.index
            adjusted_scores = selection.adjusted_scores.copy()
            adjusted_scores[action_id] = min(adjusted_scores.min(), scores.min()) - 1e-6
        else:
            action_id = int(np.argmin(scores))
            adjusted_scores = scores

        if return_all_preds:
            return endpoints, adjusted_scores
        return endpoints[action_id][None, :], float(scores[action_id])

    def control_pub(self, timer):
        previous = None
        if self.desire_acc is not None:
            previous = np.asarray(self.desire_acc, dtype=np.float64).copy()
        super().control_pub(timer)
        if previous is not None and self.desire_acc is not None:
            jerk = np.linalg.norm(np.asarray(self.desire_acc) - previous) / self.ctrl_dt
            if np.isfinite(jerk):
                self.jerk_samples.append(float(jerk))

    def write_result_report(self):
        if self.report_written:
            return
        self.report_written = True
        self.result_md.parent.mkdir(parents=True, exist_ok=True)
        self.result_json.parent.mkdir(parents=True, exist_ok=True)
        final_distance = (
            float(np.linalg.norm(self.previous_position - self.goal))
            if self.previous_position is not None else float("nan")
        )
        elapsed = time.time() - self.start_wall_time
        average_forward_ms = 1000.0 * self.time_forward / max(self.count, 1)
        average_callback_ms = 1000.0 * (
            self.time_interpolation + self.time_prepare + self.time_forward
            + self.time_process + self.time_visualize
        ) / max(self.count, 1)
        depth_collision = self.min_observed_depth < float(
            self.settings["depth_collision_distance"]
        )
        payload = {
            "policy": self.policy_kind,
            "checkpoint": self.checkpoint_path,
            "runtime_velocity_mps": self.runtime_velocity,
            "primitive_vel_max_mps": float(self.lattice_primitive.vel_max),
            "primitive_acc_max_mps2": float(self.lattice_primitive.acc_max),
            "trajectory_time_s": float(self.traj_time),
            "goal": self.goal.tolist(),
            "arrival_distance_m": self.arrival_distance,
            "goal_reached": bool(self.arrive),
            "time_to_arrival_s": self.arrival_time,
            "final_distance_m": final_distance,
            "path_length_m": self.path_length,
            "runtime_s": elapsed,
            "replans": self.count,
            "minimum_observed_depth_m": self.min_observed_depth,
            "depth_collision_proxy": bool(depth_collision),
            "minimum_geometry_distance_m": self.min_geometry_distance,
            "geometry_collision": bool(self.geometry_collision),
            "mean_speed_mps": float(np.mean(self.speed_samples)) if self.speed_samples else 0.0,
            "mean_command_jerk_mps3": float(np.mean(self.jerk_samples)) if self.jerk_samples else 0.0,
            "p95_command_jerk_mps3": float(np.quantile(self.jerk_samples, 0.95)) if self.jerk_samples else 0.0,
            "selector_enabled": bool(self.use_selector),
            "lattice_switches": self.selector.switch_count,
            "lattice_switch_rate": self.selector.switch_rate,
            "mean_endpoint_jump_m": self.selector.mean_endpoint_jump,
            "max_endpoint_jump_m": self.selector.max_endpoint_jump,
            "mean_network_forward_ms": average_forward_ms,
            "mean_callback_ms": average_callback_ms,
        }
        self.result_json.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        lines = [
            "# ICRA ROS Closed-loop Trial",
            "",
            f"- Policy: `{self.policy_kind}`",
            f"- Checkpoint: `{self.checkpoint_path}`",
            f"- Runtime velocity: {self.runtime_velocity:.3f} m/s",
            f"- Primitive max acceleration: {self.lattice_primitive.acc_max:.3f} m/s²",
            f"- Trajectory time: {self.traj_time:.3f} s",
            f"- Goal: `{self.goal.tolist()}`",
            f"- Goal reached within {self.arrival_distance:.2f} m: **{self.arrive}**",
            f"- Geometric collision: **{self.geometry_collision}**",
            f"- Depth collision proxy: **{depth_collision}**",
            f"- Final distance: {final_distance:.4f} m",
            f"- Time to arrival: {self.arrival_time if self.arrival_time is not None else float('nan'):.3f} s",
            f"- Path length: {self.path_length:.4f} m",
            f"- Minimum geometry distance: {self.min_geometry_distance:.4f} m",
            f"- Minimum observed depth: {self.min_observed_depth:.4f} m",
            f"- Mean speed: {payload['mean_speed_mps']:.4f} m/s",
            f"- Replans: {self.count}",
            f"- Lattice switches: {self.selector.switch_count}",
            f"- Lattice switch rate: {self.selector.switch_rate:.6f}",
            f"- Mean/max endpoint jump: {self.selector.mean_endpoint_jump:.4f} / {self.selector.max_endpoint_jump:.4f} m",
            f"- Mean/p95 command jerk proxy: {payload['mean_command_jerk_mps3']:.4f} / {payload['p95_command_jerk_mps3']:.4f} m/s^3",
            f"- Mean network forward: {average_forward_ms:.3f} ms",
            f"- Mean full callback: {average_callback_ms:.3f} ms",
            "",
            f"Machine-readable result: `{self.result_json}`",
        ]
        self.result_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["yopo", "latticeflow"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--runtime-velocity",
        type=float,
        default=None,
        help="Deployment velocity in m/s; defaults to the YOPO config value.",
    )
    parser.add_argument("--goal", nargs=3, type=float, default=[30.0, 0.0, 2.0])
    parser.add_argument("--result-md", required=True)
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--arrival-distance", type=float, default=1.0)
    parser.add_argument("--max-runtime", type=float, default=60.0)
    parser.add_argument("--pitch-angle-deg", type=float, default=0.0)
    parser.add_argument("--odom-topic", default="/sim/odom")
    parser.add_argument("--depth-topic", default="/depth_image")
    parser.add_argument("--ctrl-topic", default="/so3_control/pos_cmd")
    parser.add_argument("--plan-from-reference", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-visualize", action="store_true")
    parser.add_argument("--stop-on-arrival", action="store_true")
    parser.add_argument("--stop-on-collision", action="store_true")
    parser.add_argument("--disable-continuity-selector", action="store_true")
    parser.add_argument("--selector-endpoint-weight", type=float, default=0.02)
    parser.add_argument("--selector-hysteresis-margin", type=float, default=0.05)
    parser.add_argument(
        "--collision-map",
        default="/workspace/YOPO/Simulator/src/pointcloud/forest.ply",
    )
    parser.add_argument("--collision-voxel", type=float, default=0.10)
    parser.add_argument("--vehicle-radius", type=float, default=0.30)
    parser.add_argument("--depth-collision-distance", type=float, default=0.60)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ICRAPolicyNode(
        {
            "policy": args.policy,
            "checkpoint": args.checkpoint,
            "runtime_velocity": args.runtime_velocity,
            "goal": args.goal,
            "result_md": args.result_md,
            "result_json": args.result_json,
            "arrival_distance": args.arrival_distance,
            "max_runtime": args.max_runtime,
            "pitch_angle_deg": args.pitch_angle_deg,
            "odom_topic": args.odom_topic,
            "depth_topic": args.depth_topic,
            "ctrl_topic": args.ctrl_topic,
            "plan_from_reference": args.plan_from_reference,
            "verbose": args.verbose,
            "visualize": not args.no_visualize,
            "stop_on_arrival": args.stop_on_arrival,
            "stop_on_collision": args.stop_on_collision,
            "disable_continuity_selector": args.disable_continuity_selector,
            "selector_endpoint_weight": args.selector_endpoint_weight,
            "selector_hysteresis_margin": args.selector_hysteresis_margin,
            "collision_map": args.collision_map,
            "collision_voxel": args.collision_voxel,
            "vehicle_radius": args.vehicle_radius,
            "depth_collision_distance": args.depth_collision_distance,
        }
    )
