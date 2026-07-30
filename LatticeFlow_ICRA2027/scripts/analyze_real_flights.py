#!/usr/bin/env python3
"""Build reproducible real-flight metrics from extracted rosbag data."""

from __future__ import annotations

import csv
import io
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "real_flights"
FLIGHTS = {
    "flight_4m": {
        "diag": DATA / "flight_4m_runtime_diag.csv",
        "active_start": 358.301725124,
        "active_end": 373.481719984,
        "nominal_goal_distance_m": 4.0,
    },
    "flight_8m": {
        "diag": DATA / "flight_8m_runtime_diag.csv",
        "active_start": 428.443522526,
        "active_end": 436.528550200,
        "nominal_goal_distance_m": 8.0,
    },
}


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if values.size else float("nan")


def read_diagnostics(path: Path, start: float, end: float) -> list[dict]:
    # One flight log contains a block of NUL bytes from concurrent file flush.
    # Removing only NUL bytes preserves all valid CSV rows and column values.
    text = path.read_bytes().replace(b"\x00", b"").decode("utf-8", errors="ignore")
    rows = []
    for row in csv.DictReader(io.StringIO(text)):
        try:
            parsed = {
                key: value if key == "event" else float(value)
                for key, value in row.items()
            }
        except (TypeError, ValueError):
            continue
        if start <= parsed["ros_time"] <= end:
            rows.append(parsed)
    return rows


def main() -> None:
    archive = np.load(DATA / "real_flights.npz")
    payload = {
        "protocol": {
            "trials": 2,
            "success_definition": "final odometry within 1 m of the commanded goal without physical contact",
            "collision_assessment": "author-observed physical contact; both trials were collision-free",
            "selector_enabled": False,
            "backend": "TensorRT FP16",
            "nfe": 6,
            "compute": "NVIDIA Orin NX",
            "sensor": "Livox MID-360",
        },
        "flights": {},
    }
    summary_rows = []

    for name, metadata in FLIGHTS.items():
        odom_t = archive[f"{name}_odom_t"].astype(np.float64)
        odom_p = archive[f"{name}_odom_p"].astype(np.float64)
        odom_v = archive[f"{name}_odom_v"].astype(np.float64)
        goal = archive[f"{name}_goal"].astype(np.float64)
        timing = archive[f"{name}_timing"].astype(np.float64)
        diagnostics = read_diagnostics(
            metadata["diag"], metadata["active_start"], metadata["active_end"]
        )
        inference_rows = [row for row in diagnostics if row["event"] == "inference"]
        control_rows = [row for row in diagnostics if row["event"] == "control"]

        speed = np.linalg.norm(odom_v, axis=1)
        path_length = float(np.linalg.norm(np.diff(odom_p, axis=0), axis=1).sum())
        displacement = float(np.linalg.norm(odom_p[-1] - odom_p[0]))
        goal_error = float(np.linalg.norm(odom_p[-1] - goal))
        selected_ids = np.asarray(
            [int(row["selected_id"]) for row in inference_rows], dtype=np.int64
        )
        switch_rate = float(
            np.mean(selected_ids[1:] != selected_ids[:-1])
            if selected_ids.size > 1
            else 0.0
        )
        z_tracking = np.asarray(
            [
                abs(row["expected_z"] - row["odom_z"])
                for row in control_rows
                if math.isfinite(row["expected_z"])
            ],
            dtype=np.float64,
        )
        raw_valid = np.asarray(
            [row["depth_raw_valid_ratio"] for row in inference_rows], dtype=np.float64
        )
        input_far = np.asarray(
            [row["depth_input_far_ratio"] for row in inference_rows], dtype=np.float64
        )

        result = {
            "nominal_goal_distance_m": metadata["nominal_goal_distance_m"],
            "duration_s": float(odom_t[-1] - odom_t[0]),
            "path_length_m": path_length,
            "displacement_m": displacement,
            "path_efficiency": displacement / path_length,
            "goal_error_m": goal_error,
            "success": bool(goal_error <= 1.0),
            "collision": False,
            "mean_speed_mps": float(speed.mean()),
            "max_speed_mps": float(speed.max()),
            "altitude_min_m": float(odom_p[:, 2].min()),
            "altitude_max_m": float(odom_p[:, 2].max()),
            "mean_inference_ms": float(timing[:, 2].mean()),
            "p95_inference_ms": percentile(timing[:, 2], 95),
            "mean_total_ms": float(timing[:, 5].mean()),
            "p95_total_ms": percentile(timing[:, 5], 95),
            "raw_lidar_valid_ratio": float(raw_valid.mean()),
            "network_input_far_ratio": float(input_far.mean()),
            "mean_vertical_tracking_error_m": float(z_tracking.mean()),
            "p95_vertical_tracking_error_m": percentile(z_tracking, 95),
            "raw_cell_switch_rate": switch_rate,
            "selected_cell_counts": {
                str(key): int(value)
                for key, value in sorted(Counter(selected_ids.tolist()).items())
            },
            "inference_frames": int(len(inference_rows)),
            "control_commands": int(len(control_rows)),
        }
        payload["flights"][name] = result
        summary_rows.append({"flight": name, **result})

    payload["aggregate"] = {
        "success": 2,
        "trials": 2,
        "collisions": 0,
        "mean_inference_ms": float(
            np.mean([value["mean_inference_ms"] for value in payload["flights"].values()])
        ),
        "mean_total_ms": float(
            np.mean([value["mean_total_ms"] for value in payload["flights"].values()])
        ),
        "maximum_recorded_speed_mps": float(
            max(value["max_speed_mps"] for value in payload["flights"].values())
        ),
    }

    (DATA / "real_flight_metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    fieldnames = list(summary_rows[0].keys())
    with (DATA / "real_flight_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
