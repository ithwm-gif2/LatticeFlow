#!/usr/bin/env python3
"""Aggregate matched ROS closed-loop JSON files without treating frames as samples."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def method_name(payload: dict) -> str:
    if payload["policy"] == "yopo":
        return "YOPO"
    return "LatticeFlow" if payload.get("selector_enabled", False) else "LatticeFlow-no-selector"


def finite(values):
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    successes = np.asarray([r["goal_reached"] for r in rows], dtype=np.float64)
    collisions = np.asarray([r["geometry_collision"] for r in rows], dtype=np.float64)
    continuous = {
        "time_to_arrival_s": [r["time_to_arrival_s"] for r in rows if r["time_to_arrival_s"] is not None],
        "path_length_m": [r["path_length_m"] for r in rows],
        "minimum_geometry_distance_m": [r["minimum_geometry_distance_m"] for r in rows],
        "mean_speed_mps": [r["mean_speed_mps"] for r in rows],
        "lattice_switch_rate": [r["lattice_switch_rate"] for r in rows],
        "mean_endpoint_jump_m": [r["mean_endpoint_jump_m"] for r in rows],
        "mean_command_jerk_mps3": [r["mean_command_jerk_mps3"] for r in rows],
        "mean_network_forward_ms": [r["mean_network_forward_ms"] for r in rows],
    }
    result = {
        "trials": n,
        "successes": int(successes.sum()),
        "success_rate": float(successes.mean()) if n else float("nan"),
        "collisions": int(collisions.sum()),
        "collision_rate": float(collisions.mean()) if n else float("nan"),
    }
    for key, values in continuous.items():
        array = finite(values)
        result[key] = {
            "n": int(array.size),
            "mean": float(array.mean()) if array.size else None,
            "std": float(array.std(ddof=1)) if array.size > 1 else None,
            "median": float(np.median(array)) if array.size else None,
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    grouped: dict[str, list[dict]] = defaultdict(list)
    records = []
    for path in sorted(root.rglob("*.json")):
        if path.name == Path(args.output_json).name:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "policy" not in payload or "goal_reached" not in payload:
            continue
        payload["scenario"] = path.parent.relative_to(root).as_posix()
        payload["source_file"] = str(path)
        grouped[method_name(payload)].append(payload)
        records.append(payload)

    summaries = {name: summarize(rows) for name, rows in grouped.items()}
    output = {"root": str(root), "methods": summaries, "trials": records}
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Matched ROS Closed-loop Summary",
        "",
        "The independent sample is one complete scenario rollout. Failures and timeouts remain in the denominator.",
        "",
        "| Method | Trials | Success | Collision | Time to goal (s) | Min. clearance (m) | Switch rate | Endpoint jump (m) | Jerk proxy (m/s^3) | Forward (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, stats in summaries.items():
        def mean(key):
            value = stats[key]["mean"]
            return "--" if value is None else f"{value:.3f}"

        lines.append(
            f"| {name} | {stats['trials']} | {stats['successes']}/{stats['trials']} "
            f"| {stats['collisions']}/{stats['trials']} | {mean('time_to_arrival_s')} "
            f"| {mean('minimum_geometry_distance_m')} | {mean('lattice_switch_rate')} "
            f"| {mean('mean_endpoint_jump_m')} | {mean('mean_command_jerk_mps3')} "
            f"| {mean('mean_network_forward_ms')} |"
        )
    lines.extend(["", f"Machine-readable results: `{output_json}`", ""])
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines), encoding="utf-8")
    print(output_md)


if __name__ == "__main__":
    main()
