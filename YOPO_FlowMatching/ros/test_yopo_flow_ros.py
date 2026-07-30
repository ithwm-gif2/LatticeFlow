#!/usr/bin/env python3
"""ROS closed-loop test using the original YOPO controller interface."""

from __future__ import annotations

import argparse
import atexit
import sys
import time
from pathlib import Path
from threading import Lock

# Running ``python3 ros/test_yopo_flow_ros.py`` makes Python use the ``ros``
# directory as sys.path[0]. Add the project root explicitly so the sibling
# ``yopo_flow`` package is importable without requiring a manual PYTHONPATH.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import rospy
import torch
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image, PointCloud2

from yopo_flow.bootstrap import add_original_yopo_to_path
from yopo_flow.checkpoint import load_policy_checkpoint

add_original_yopo_to_path()

from config.config import cfg  # noqa: E402
from control_msg import PositionCommand  # noqa: E402
from policy.primitive import LatticePrimitive  # noqa: E402
from policy.state_transform import StateTransform  # noqa: E402
from test_yopo_ros import YopoNet as OriginalYopoNet  # noqa: E402


class FlowYopoNet(OriginalYopoNet):
    """Reuse YOPO callbacks/controller while replacing only the neural policy."""

    def __init__(self, config: dict, checkpoint_path: str, result_md: str):
        self.config = config
        self.result_md = Path(result_md)
        self.start_wall_time = time.time()
        self.path_length = 0.0
        self.previous_position = None
        self.min_observed_depth = float("inf")
        self.arrival_time = None
        self.report_written = False

        rospy.init_node("yopo_flow_net", anonymous=False)
        cfg["train"] = False
        self.height = cfg["image_height"]
        self.width = cfg["image_width"]
        self.min_dis, self.max_dis = 0.04, 20.0
        self.goal = np.array(self.config["goal"], dtype=np.float64)
        self.plan_from_reference = self.config["plan_from_reference"]
        self.use_trt = False
        self.verbose = self.config["verbose"]
        self.visualize = self.config["visualize"]
        self.Rotation_bc = R.from_euler(
            "ZYX", [0, self.config["pitch_angle_deg"], 0], degrees=True
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

        self.time_forward = 0.0
        self.time_process = 0.0
        self.time_prepare = 0.0
        self.time_interpolation = 0.0
        self.time_visualize = 0.0
        self.count = 0
        self.depth_fps = 30

        self.policy, self.flow_config, _ = load_policy_checkpoint(
            checkpoint_path, self.device
        )
        self.policy.eval()
        self.warm_up()

        self.lattice_traj_pub = rospy.Publisher(
            "/yopo_flow/lattice_trajs_visual", PointCloud2, queue_size=1
        )
        self.best_traj_pub = rospy.Publisher(
            "/yopo_flow/best_traj_visual", PointCloud2, queue_size=1
        )
        self.all_trajs_pub = rospy.Publisher(
            "/yopo_flow/trajs_visual", PointCloud2, queue_size=1
        )
        self.ctrl_pub = rospy.Publisher(
            self.config["ctrl_topic"], PositionCommand, queue_size=1
        )
        self.odom_sub = rospy.Subscriber(
            self.config["odom_topic"],
            Odometry,
            self.callback_odometry,
            queue_size=1,
            tcp_nodelay=True,
        )
        self.depth_sub = rospy.Subscriber(
            self.config["depth_topic"],
            Image,
            self.callback_depth,
            queue_size=1,
            tcp_nodelay=True,
        )
        self.goal_sub = rospy.Subscriber(
            "/move_base_simple/goal", PoseStamped, self.callback_set_goal, queue_size=1
        )
        rospy.on_shutdown(self.write_result_report)
        atexit.register(self.write_result_report)
        rospy.sleep(1.0)
        self.timer_ctrl = rospy.Timer(rospy.Duration(self.ctrl_dt), self.control_pub)
        print("YOPO Flow Net Node Ready!")
        rospy.spin()

    def callback_odometry(self, data):
        was_arrived = self.arrive
        super().callback_odometry(data)
        position = np.array(
            [
                data.pose.pose.position.x,
                data.pose.pose.position.y,
                data.pose.pose.position.z,
            ],
            dtype=np.float64,
        )
        if self.previous_position is not None:
            self.path_length += float(np.linalg.norm(position - self.previous_position))
        self.previous_position = position
        if self.arrive and not was_arrived:
            self.arrival_time = time.time() - self.start_wall_time
            if self.config.get("stop_on_arrival", False):
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

    def write_result_report(self):
        if self.report_written:
            return
        self.report_written = True
        self.result_md.parent.mkdir(parents=True, exist_ok=True)
        final_position = self.previous_position
        final_distance = (
            float(np.linalg.norm(final_position - self.goal))
            if final_position is not None
            else float("nan")
        )
        elapsed = time.time() - self.start_wall_time
        average_forward_ms = 1000.0 * self.time_forward / max(self.count, 1)
        average_total_ms = 1000.0 * (
            self.time_interpolation
            + self.time_prepare
            + self.time_forward
            + self.time_process
            + self.time_visualize
        ) / max(self.count, 1)
        collision_proxy = self.min_observed_depth < float(
            self.flow_config["evaluation"]["collision_distance"]
        )
        text = f"""# ROS Closed-loop Evaluation Results

- Goal: `{self.goal.tolist()}`
- Goal reached: **{self.arrive}**
- Time to arrival: {self.arrival_time if self.arrival_time is not None else float('nan'):.2f} s
- Final distance to goal: {final_distance:.4f} m
- Travelled path length: {self.path_length:.4f} m
- Runtime: {elapsed:.2f} s
- Network replans: {self.count}
- Minimum observed depth: {self.min_observed_depth:.4f} m
- Collision proxy triggered: **{collision_proxy}**
- Mean network forward time: {average_forward_ms:.3f} ms
- Mean full callback time: {average_total_ms:.3f} ms

## Notes

The collision proxy uses the minimum valid depth observed by the camera. If the simulator exposes a physical collision topic, record that signal alongside this report because a depth-only proxy cannot detect every contact.
"""
        self.result_md.write_text(text, encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--result-md",
        default=str(Path(__file__).resolve().parents[1] / "ROS_RESULTS.md"),
    )
    parser.add_argument("--goal", nargs=3, type=float, default=[50.0, 0.0, 2.0])
    parser.add_argument("--pitch-angle-deg", type=float, default=0.0)
    parser.add_argument("--odom-topic", default="/sim/odom")
    parser.add_argument("--depth-topic", default="/depth_image")
    parser.add_argument("--ctrl-topic", default="/so3_control/pos_cmd")
    parser.add_argument("--plan-from-reference", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-visualize", action="store_true")
    parser.add_argument("--stop-on-arrival", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    settings = {
        "use_tensorrt": False,
        "goal": args.goal,
        "pitch_angle_deg": args.pitch_angle_deg,
        "odom_topic": args.odom_topic,
        "depth_topic": args.depth_topic,
        "ctrl_topic": args.ctrl_topic,
        "plan_from_reference": args.plan_from_reference,
        "verbose": args.verbose,
        "visualize": not args.no_visualize,
        "stop_on_arrival": args.stop_on_arrival,
    }
    FlowYopoNet(settings, args.checkpoint, args.result_md)
