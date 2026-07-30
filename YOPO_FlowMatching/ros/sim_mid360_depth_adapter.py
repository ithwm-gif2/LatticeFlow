#!/usr/bin/env python3
"""Convert simulator world-frame LiDAR points to the deployed MID-360 image."""

from __future__ import annotations

import argparse
from collections import deque
from threading import Lock

import cv2
import numpy as np
import rospy
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image, PointCloud2


class SimMid360DepthAdapter:
    def __init__(self, args: argparse.Namespace):
        rospy.init_node("sim_mid360_depth_adapter", anonymous=False)
        self.height = int(args.height)
        self.width = int(args.width)
        self.max_depth = float(args.max_depth)
        self.min_depth = float(args.min_depth)
        self.accumulation = max(1, int(args.accumulation))
        self.buffer = deque(maxlen=self.accumulation)
        self.lock = Lock()
        self.odom = None
        self.last_odom_position = None

        vertical_min = float(args.elevation_min_deg)
        vertical_max = float(args.elevation_max_deg)
        self.pitch = np.deg2rad(0.5 * (vertical_min + vertical_max))
        vertical_fov = vertical_max - vertical_min
        self.fx = (self.width / 2.0) / np.tan(
            np.deg2rad(float(args.horizontal_fov_deg)) / 2.0
        )
        self.fy = (self.height / 2.0) / np.tan(np.deg2rad(vertical_fov) / 2.0)
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0

        self.ceiling_enabled = not bool(args.disable_virtual_ceiling)
        self.ceiling_world_z = float(args.virtual_ceiling_world_z)
        self.ceiling_stride = max(1, int(args.virtual_ceiling_stride))
        self.ceiling_min_forward = float(args.virtual_ceiling_min_forward)
        self.fill_iterations = max(0, int(args.fill_iterations))
        self.fill_kernel = int(args.fill_kernel)
        self.fill_neighbors = int(args.fill_neighbors)

        columns = np.arange(self.width, dtype=np.float64)
        rows = np.arange(self.height, dtype=np.float64)
        uu, vv = np.meshgrid(columns, rows)
        forward = np.ones_like(uu)
        left = (self.cx - uu) / self.fx
        up = (self.cy - vv) / self.fy
        sine, cosine = np.sin(self.pitch), np.cos(self.pitch)
        self.rays_body = np.stack(
            (
                cosine * forward - sine * up,
                left,
                sine * forward + cosine * up,
            ),
            axis=-1,
        )
        self.ceiling_sample = np.zeros((self.height, self.width), dtype=bool)
        self.ceiling_sample[:: self.ceiling_stride, :: self.ceiling_stride] = True

        self.depth_pub = rospy.Publisher(args.output_topic, Image, queue_size=1)
        self.raw_pub = rospy.Publisher(args.raw_topic, Image, queue_size=1)
        rospy.Subscriber(args.odom_topic, Odometry, self.odom_callback, queue_size=1)
        rospy.Subscriber(
            args.lidar_topic,
            PointCloud2,
            self.lidar_callback,
            queue_size=1,
            tcp_nodelay=True,
        )
        rospy.loginfo(
            "[SimMID360] lidar=%s output=%s FOV=[%.1f,%.1f] pitch=%.1f ceiling=%s z=%.2f",
            args.lidar_topic,
            args.output_topic,
            vertical_min,
            vertical_max,
            np.rad2deg(self.pitch),
            self.ceiling_enabled,
            self.ceiling_world_z,
        )

    def odom_callback(self, message: Odometry) -> None:
        position = np.asarray(
            [
                message.pose.pose.position.x,
                message.pose.pose.position.y,
                message.pose.pose.position.z,
            ],
            dtype=np.float64,
        )
        if (
            self.last_odom_position is not None
            and np.linalg.norm(position - self.last_odom_position) > 1.0
        ):
            with self.lock:
                self.buffer.clear()
        self.last_odom_position = position
        self.odom = message

    @staticmethod
    def xyz_points(message: PointCloud2) -> np.ndarray:
        points = np.asarray(
            list(
                point_cloud2.read_points(
                    message, field_names=("x", "y", "z"), skip_nans=True
                )
            ),
            dtype=np.float64,
        )
        return points.reshape(-1, 3)

    def pose(self) -> tuple[np.ndarray, np.ndarray]:
        orientation = self.odom.pose.pose.orientation
        rotation = R.from_quat(
            [orientation.x, orientation.y, orientation.z, orientation.w]
        ).as_matrix()
        position = np.asarray(
            [
                self.odom.pose.pose.position.x,
                self.odom.pose.pose.position.y,
                self.odom.pose.pose.position.z,
            ],
            dtype=np.float64,
        )
        return rotation, position

    def project(self, points_body: np.ndarray) -> np.ndarray:
        depth = np.full((self.height, self.width), self.max_depth, np.float32)
        sine, cosine = np.sin(self.pitch), np.cos(self.pitch)
        forward = cosine * points_body[:, 0] + sine * points_body[:, 2]
        left = points_body[:, 1]
        up = -sine * points_body[:, 0] + cosine * points_body[:, 2]
        valid = (
            np.isfinite(points_body).all(axis=1)
            & (forward > self.min_depth)
            & (forward < self.max_depth)
        )
        if not np.any(valid):
            return depth
        forward, left, up = forward[valid], left[valid], up[valid]
        pixel_u = np.rint(self.cx - self.fx * left / forward).astype(np.int32)
        pixel_v = np.rint(self.cy - self.fy * up / forward).astype(np.int32)
        inside = (
            (pixel_u >= 0)
            & (pixel_u < self.width)
            & (pixel_v >= 0)
            & (pixel_v < self.height)
        )
        np.minimum.at(
            depth,
            (pixel_v[inside], pixel_u[inside]),
            forward[inside].astype(np.float32),
        )
        return depth

    def local_fill(self, depth: np.ndarray) -> np.ndarray:
        result = depth.copy()
        if self.fill_iterations <= 0:
            return result
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.fill_kernel, self.fill_kernel)
        )
        for _ in range(self.fill_iterations):
            valid = (
                np.isfinite(result)
                & (result > self.min_depth)
                & (result < self.max_depth * 0.999)
            )
            count = cv2.boxFilter(
                valid.astype(np.float32),
                -1,
                (self.fill_kernel, self.fill_kernel),
                normalize=False,
                borderType=cv2.BORDER_CONSTANT,
            )
            local_minimum = cv2.erode(result, kernel, iterations=1)
            fill = (
                (~valid)
                & (count >= self.fill_neighbors)
                & (local_minimum > self.min_depth)
                & (local_minimum < self.max_depth * 0.999)
            )
            if not np.any(fill):
                break
            result[fill] = local_minimum[fill]
        return result

    def add_ceiling(
        self, depth: np.ndarray, rotation_world_body: np.ndarray, position: np.ndarray
    ) -> np.ndarray:
        if not self.ceiling_enabled:
            return depth
        gap = self.ceiling_world_z - float(position[2])
        if gap <= 0.0:
            return depth
        vertical = (self.rays_body @ rotation_world_body.T)[..., 2]
        positive = vertical > 1.0e-3
        scale = gap / np.where(positive, vertical, np.inf)
        forward = scale * self.rays_body[..., 0]
        valid = (
            positive
            & self.ceiling_sample
            & (scale > self.min_depth)
            & (scale < self.max_depth)
            & (forward > self.ceiling_min_forward)
        )
        result = depth.copy()
        result[valid] = np.minimum(result[valid], scale[valid].astype(np.float32))
        return result

    @staticmethod
    def image_message(depth: np.ndarray, stamp, frame_id: str) -> Image:
        array = np.ascontiguousarray(depth, dtype=np.float32)
        message = Image()
        message.header.stamp = stamp
        message.header.frame_id = frame_id
        message.height, message.width = array.shape
        message.encoding = "32FC1"
        message.is_bigendian = False
        message.step = message.width * 4
        message.data = array.tobytes()
        return message

    def lidar_callback(self, message: PointCloud2) -> None:
        if self.odom is None:
            return
        points_world = self.xyz_points(message)
        if points_world.size == 0:
            return
        with self.lock:
            self.buffer.append(points_world)
            accumulated = np.concatenate(tuple(self.buffer), axis=0)
        rotation, position = self.pose()
        points_body = (accumulated - position) @ rotation
        raw = self.project(points_body)
        model_depth = self.add_ceiling(self.local_fill(raw), rotation, position)
        self.raw_pub.publish(self.image_message(raw, message.header.stamp, "body"))
        self.depth_pub.publish(
            self.image_message(model_depth, message.header.stamp, "body")
        )
        raw_valid = (raw < self.max_depth * 0.999).mean()
        model_valid = (model_depth < self.max_depth * 0.999).mean()
        rospy.loginfo_throttle(
            2.0,
            "[SimMID360] raw_valid=%.1f%% model_nonfar=%.1f%%",
            100.0 * raw_valid,
            100.0 * model_valid,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--odom-topic", default="/sim/odom")
    parser.add_argument("--lidar-topic", default="/lidar_points")
    parser.add_argument("--output-topic", default="/mid360_depth_image")
    parser.add_argument("--raw-topic", default="/mid360_depth_image_raw")
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--horizontal-fov-deg", type=float, default=90.0)
    parser.add_argument("--elevation-min-deg", type=float, default=-7.0)
    parser.add_argument("--elevation-max-deg", type=float, default=52.0)
    parser.add_argument("--min-depth", type=float, default=0.05)
    parser.add_argument("--max-depth", type=float, default=20.0)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--fill-iterations", type=int, default=1)
    parser.add_argument("--fill-kernel", type=int, default=3)
    parser.add_argument("--fill-neighbors", type=int, default=5)
    parser.add_argument("--disable-virtual-ceiling", action="store_true")
    parser.add_argument("--virtual-ceiling-world-z", type=float, default=2.0)
    parser.add_argument("--virtual-ceiling-stride", type=int, default=2)
    parser.add_argument("--virtual-ceiling-min-forward", type=float, default=0.4)
    return parser.parse_args()


if __name__ == "__main__":
    SimMid360DepthAdapter(parse_args())
    rospy.spin()
