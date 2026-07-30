#!/usr/bin/env python3
"""Capture the simulator's published random map as a PLY collision reference."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import rospy
from sensor_msgs.msg import PointCloud2


def pointcloud_xyz(message: PointCloud2) -> np.ndarray:
    fields = {field.name: field for field in message.fields}
    missing = {"x", "y", "z"} - fields.keys()
    if missing:
        raise ValueError(f"PointCloud2 is missing fields: {sorted(missing)}")
    endian = ">" if message.is_bigendian else "<"
    dtype = np.dtype(
        {
            "names": ["x", "y", "z"],
            "formats": [f"{endian}f4", f"{endian}f4", f"{endian}f4"],
            "offsets": [fields[axis].offset for axis in ("x", "y", "z")],
            "itemsize": message.point_step,
        }
    )
    array = np.frombuffer(message.data, dtype=dtype, count=message.width * message.height)
    points = np.column_stack((array["x"], array["y"], array["z"])).astype(np.float32)
    return points[np.isfinite(points).all(axis=1)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/mock_map")
    parser.add_argument("--output", required=True)
    parser.add_argument("--voxel", type=float, default=0.10)
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()

    rospy.init_node("capture_icra_mock_map", anonymous=True)
    message = rospy.wait_for_message(args.topic, PointCloud2, timeout=args.timeout)
    points = pointcloud_xyz(message)
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if args.voxel > 0.0:
        pointcloud = pointcloud.voxel_down_sample(args.voxel)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(output), pointcloud, write_ascii=False):
        raise OSError(f"Failed to write point cloud: {output}")
    print(f"Captured {len(pointcloud.points)} map points to {output}")


if __name__ == "__main__":
    main()
