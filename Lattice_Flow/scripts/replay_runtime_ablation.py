#!/usr/bin/env python3
"""Replay recorded LatticeFlow inputs with a chosen deployment velocity.

This is a read-only counterfactual evaluator. It uses the recorded model depth
image (including the deployed virtual-ceiling pipeline), odometry, controller
reference and PX4Ctrl mode, but never initializes a ROS node or publishes
commands.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import rosbag
import torch
from scipy.spatial.transform import Rotation as R

import lattice_flow_lidar_node as runtime


def image_array(message) -> np.ndarray:
    dtype = np.dtype(">f4" if message.is_bigendian else "<f4")
    row_values = message.step // dtype.itemsize
    return np.frombuffer(message.data, dtype=dtype).reshape(
        message.height, row_values
    )[:, : message.width].astype(np.float32)


def nearest_index(times: np.ndarray, query: float) -> int:
    index = int(np.searchsorted(times, query, side="left"))
    index = min(max(index, 0), len(times) - 1)
    left = max(index - 1, 0)
    if abs(query - times[left]) < abs(query - times[index]):
        return left
    return index


def previous_index(times: np.ndarray, query: float) -> int:
    return int(np.searchsorted(times, query, side="right") - 1)


def statistics(values) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
        "min": float(array.min()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--runtime-velocity", type=float, required=True)
    parser.add_argument("--goal-distance", type=float, default=10.0)
    parser.add_argument("--goal-height", type=float, default=1.5)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_recording(bag_path: Path) -> dict:
    depth_frames = []
    odom_time, odom_position, odom_velocity, odom_quaternion = [], [], [], []
    command_time, command_position, command_acceleration = [], [], []
    state_time, state_value = [], []
    topics = [
        "/lattice_flow/depth_front",
        "/ekf_quat/ekf_odom",
        "/setpoints_cmd",
        "/px4ctrl/state",
    ]
    with rosbag.Bag(str(bag_path), "r") as bag:
        for topic, message, stamp in bag.read_messages(topics=topics):
            time_value = stamp.to_sec()
            if topic == "/lattice_flow/depth_front":
                depth_frames.append((time_value, image_array(message)))
            elif topic == "/ekf_quat/ekf_odom":
                odom_time.append(time_value)
                odom_position.append(
                    [
                        message.pose.pose.position.x,
                        message.pose.pose.position.y,
                        message.pose.pose.position.z,
                    ]
                )
                odom_velocity.append(
                    [
                        message.twist.twist.linear.x,
                        message.twist.twist.linear.y,
                        message.twist.twist.linear.z,
                    ]
                )
                odom_quaternion.append(
                    [
                        message.pose.pose.orientation.x,
                        message.pose.pose.orientation.y,
                        message.pose.pose.orientation.z,
                        message.pose.pose.orientation.w,
                    ]
                )
            elif topic == "/setpoints_cmd":
                command_time.append(time_value)
                command_position.append(
                    [message.position.x, message.position.y, message.position.z]
                )
                command_acceleration.append(
                    [
                        message.acceleration.x,
                        message.acceleration.y,
                        message.acceleration.z,
                    ]
                )
            elif topic == "/px4ctrl/state":
                state_time.append(time_value)
                state_value.append(int(message.state))
    if not depth_frames or not odom_time or not state_time:
        raise RuntimeError("Bag is missing depth, odometry, or PX4Ctrl state")
    return {
        "depth": depth_frames,
        "odom_time": np.asarray(odom_time, dtype=np.float64),
        "odom_position": np.asarray(odom_position, dtype=np.float64),
        "odom_velocity": np.asarray(odom_velocity, dtype=np.float64),
        "odom_quaternion": np.asarray(odom_quaternion, dtype=np.float64),
        "command_time": np.asarray(command_time, dtype=np.float64),
        "command_position": np.asarray(command_position, dtype=np.float64),
        "command_acceleration": np.asarray(
            command_acceleration, dtype=np.float64
        ),
        "state_time": np.asarray(state_time, dtype=np.float64),
        "state_value": np.asarray(state_value, dtype=np.int64),
    }


def trajectory_metrics(
    endpoint: np.ndarray,
    start_velocity: np.ndarray,
    start_acceleration: np.ndarray,
    duration: float,
) -> dict[str, float]:
    times = np.linspace(0.0, duration, 101)
    positions, velocities, accelerations = [], [], []
    for axis in range(3):
        polynomial = runtime.Poly5Solver(
            0.0,
            start_velocity[axis],
            start_acceleration[axis],
            endpoint[axis],
            endpoint[3 + axis],
            endpoint[6 + axis],
            duration,
        )
        positions.append(polynomial.get_position(times))
        velocities.append(polynomial.get_velocity(times))
        accelerations.append(polynomial.get_acceleration(times))
    positions = np.stack(positions, axis=1)
    velocities = np.stack(velocities, axis=1)
    accelerations = np.stack(accelerations, axis=1)

    def vector_norm_at(values: np.ndarray, query: float) -> float:
        index = int(np.argmin(np.abs(times - query)))
        return float(np.linalg.norm(values[index]))

    return {
        "endpoint_distance_m": float(np.linalg.norm(endpoint[:3])),
        "endpoint_forward_m": float(endpoint[0]),
        "terminal_speed_mps": float(np.linalg.norm(endpoint[3:6])),
        "terminal_acceleration_mps2": float(np.linalg.norm(endpoint[6:9])),
        "trajectory_length_m": float(
            np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()
        ),
        "vertical_span_m": float(positions[:, 2].max() - positions[:, 2].min()),
        "speed_t0_02_mps": vector_norm_at(velocities, 0.02),
        "speed_t0_10_mps": vector_norm_at(velocities, 0.10),
        "speed_t0_20_mps": vector_norm_at(velocities, 0.20),
        "acceleration_t0_10_mps2": vector_norm_at(accelerations, 0.10),
        "max_speed_mps": float(np.linalg.norm(velocities, axis=1).max()),
        "max_acceleration_mps2": float(
            np.linalg.norm(accelerations, axis=1).max()
        ),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.runtime_velocity <= 0.0 or args.goal_distance <= 0.0:
        raise ValueError("Runtime velocity and goal distance must be positive")
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    metadata_runtime = metadata.get("runtime", {})
    if metadata_runtime and abs(
        float(metadata_runtime["velocity"]) - args.runtime_velocity
    ) > 1.0e-6:
        raise ValueError("Metadata runtime velocity does not match CLI value")

    runtime.yopo_cfg["train"] = False
    runtime.yopo_cfg["velocity"] = float(args.runtime_velocity)
    transform = runtime.LatticeFlowLidarNode._make_state_transform()
    primitive = runtime.LatticeFlowLidarNode._get_primitive()
    policy = runtime.TensorRTPolicy(
        args.metadata,
        args.checkpoint,
        torch.device("cuda"),
        transform,
        runtime.NFE,
    )

    recording = load_recording(args.bag)
    first_depth_time = recording["depth"][0][0]
    first_odom_index = nearest_index(recording["odom_time"], first_depth_time)
    first_rotation = R.from_quat(
        recording["odom_quaternion"][first_odom_index]
    )
    first_yaw = first_rotation.as_euler("ZYX")[0]
    goal_world = recording["odom_position"][first_odom_index].copy()
    goal_world[:2] += args.goal_distance * np.asarray(
        [np.cos(first_yaw), np.sin(first_yaw)]
    )
    goal_world[2] = args.goal_height

    rows = []
    action_counts = Counter()
    lattice_ids = torch.arange(14, -1, -1)
    for stamp, depth_image in recording["depth"]:
        state_index = nearest_index(recording["state_time"], stamp)
        if recording["state_value"][state_index] != 3:
            continue
        odom_index = nearest_index(recording["odom_time"], stamp)
        rotation_world_body = R.from_quat(
            recording["odom_quaternion"][odom_index]
        ).as_matrix()
        command_index = previous_index(recording["command_time"], stamp)
        if command_index >= 0:
            desired_position = recording["command_position"][command_index]
            desired_acceleration = recording["command_acceleration"][command_index]
        else:
            desired_position = recording["odom_position"][odom_index]
            desired_acceleration = np.zeros(3, dtype=np.float64)

        observation = np.concatenate(
            (
                rotation_world_body.T @ recording["odom_velocity"][odom_index],
                rotation_world_body.T @ desired_acceleration,
                rotation_world_body.T @ (goal_world - desired_position),
            )
        ).astype(np.float32)
        depth_tensor = torch.from_numpy(depth_image[None, None]).to("cuda")
        observation_tensor = torch.from_numpy(observation[None]).to("cuda")
        raw, score = policy(depth_tensor, observation_tensor)
        raw_numpy = raw[0].detach().cpu().numpy()
        scores = score[0].detach().cpu().numpy().reshape(-1)
        action = int(np.argmin(scores))
        raw_cells = raw_numpy.reshape(9, 15).T
        endpoints = transform.pred_to_endstate_cpu(raw_cells, lattice_ids)
        metrics = trajectory_metrics(
            endpoints[action],
            observation[:3],
            observation[3:6],
            primitive.segment_time,
        )
        metrics["selected_score"] = float(scores[action])
        rows.append(metrics)
        action_counts[action] += 1

    if not rows:
        raise RuntimeError("No CMD_CTRL depth frames found in bag")
    metric_names = rows[0].keys()
    report = {
        "bag": str(args.bag.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "metadata": str(args.metadata.resolve()),
        "runtime_velocity_mps": float(args.runtime_velocity),
        "primitive": {
            "vel_max_mps": float(primitive.vel_max),
            "acc_max_mps2": float(primitive.acc_max),
            "segment_time_s": float(primitive.segment_time),
        },
        "goal": {
            "distance_m": float(args.goal_distance),
            "height_m": float(args.goal_height),
            "world": goal_world.tolist(),
        },
        "cmd_ctrl_frames": len(rows),
        "selected_action_counts": {
            str(key): value for key, value in sorted(action_counts.items())
        },
        "metrics": {
            name: statistics([row[name] for row in rows])
            for name in metric_names
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
