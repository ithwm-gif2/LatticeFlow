#!/usr/bin/env python3
"""Generate the real-flight evidence figure from extracted rosbag data.

Figure contract
---------------
Core conclusion: the same deterministic LatticeFlow policy completed two
collision-free LiDAR-only flights while maintaining bounded altitude and
onboard TensorRT latency.
Evidence: panels a--b show complete world-frame paths over voxelized LiDAR
returns and sampled planned trajectories; panels c--d show altitude and speed
for every odometry sample; panel e decomposes measured runtime.
Archetype: asymmetric mixed-modality quantitative figure. Backend: Python.
The environment display uses the complete extracted 8-cm voxel cloud restricted
to z=0.3--2.5 m so that ground and overhead returns do not obscure obstacles at
flight height. No flight or odometry sample is excluded.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "real_flights"
OUT = ROOT / "paper" / "figures"
ARCHIVE = DATA / "real_flights.npz"
METRICS = DATA / "real_flight_metrics.json"

WIDTH_MM = 183
COLORS = {
    "flight_4m": "#2F6FA3",
    "flight_8m": "#D8843D",
    "cloud": "#AEB4BA",
    "planned": "#76B7B2",
    "command": "#555555",
    "goal": "#3F8F6B",
    "ink": "#222222",
    "preprocess": "#A9C4DB",
    "inference": "#3F78A8",
    "postprocess": "#E3B276",
    "visualization": "#A8CFAF",
    "other": "#D7D7D7",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 6.4,
        "axes.labelsize": 6.4,
        "axes.titlesize": 7.0,
        "xtick.labelsize": 5.8,
        "ytick.labelsize": 5.8,
        "legend.fontsize": 5.5,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "legend.frameon": False,
    }
)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.06, label, transform=ax.transAxes, weight="bold", fontsize=8)


def trajectory_chunks(archive: np.lib.npyio.NpzFile, name: str):
    points = archive[f"{name}_traj_points"]
    offsets = archive[f"{name}_traj_offsets"]
    for start, stop in zip(offsets[:-1], offsets[1:]):
        yield points[start:stop]


def plot_top_down(
    ax: plt.Axes,
    archive: np.lib.npyio.NpzFile,
    metrics: dict,
    name: str,
    title: str,
) -> None:
    cloud = archive[f"{name}_cloud"]
    path = archive[f"{name}_odom_p"]
    goal = archive[f"{name}_goal"]
    near_flight_height = (cloud[:, 2] >= 0.3) & (cloud[:, 2] <= 2.5)
    visible = cloud[near_flight_height]
    ax.scatter(
        visible[:, 0],
        visible[:, 1],
        s=0.45,
        c=COLORS["cloud"],
        alpha=0.38,
        linewidths=0,
        rasterized=True,
        label="LiDAR returns",
    )
    for chunk in trajectory_chunks(archive, name):
        ax.plot(
            chunk[:, 0],
            chunk[:, 1],
            color=COLORS["planned"],
            linewidth=0.65,
            alpha=0.55,
        )
    ax.plot(
        path[:, 0],
        path[:, 1],
        color=COLORS[name],
        linewidth=1.8,
        label="measured path",
        zorder=4,
    )
    ax.scatter(
        path[0, 0], path[0, 1], s=20, c=COLORS["ink"], marker="o", zorder=5
    )
    ax.scatter(
        goal[0],
        goal[1],
        s=48,
        c=COLORS["goal"],
        marker="*",
        edgecolors="white",
        linewidths=0.4,
        zorder=6,
    )
    result = metrics["flights"][name]
    ax.text(
        0.02,
        0.03,
        f"{result['path_length_m']:.2f} m path, {result['duration_s']:.2f} s\n"
        f"goal error {result['goal_error_m']:.2f} m",
        transform=ax.transAxes,
        va="bottom",
        color=COLORS["ink"],
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.5),
    )
    ax.set_title(title)
    ax.set_xlabel("World x (m)")
    ax.set_ylabel("World y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#E8E8E8", linewidth=0.4)


def main() -> None:
    archive = np.load(ARCHIVE)
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, 91 / 25.4))
    grid = fig.add_gridspec(
        2,
        6,
        height_ratios=[1.28, 0.72],
        left=0.055,
        right=0.99,
        top=0.94,
        bottom=0.14,
        wspace=0.95,
        hspace=0.72,
    )

    ax_a = fig.add_subplot(grid[0, 0:3])
    ax_b = fig.add_subplot(grid[0, 3:6])
    plot_top_down(ax_a, archive, metrics, "flight_4m", "4 m goal flight")
    plot_top_down(ax_b, archive, metrics, "flight_8m", "8 m goal flight")
    panel_label(ax_a, "a")
    panel_label(ax_b, "b")

    ax_c = fig.add_subplot(grid[1, 0:2])
    for name, label in [("flight_4m", "4 m"), ("flight_8m", "8 m")]:
        t = archive[f"{name}_odom_t"]
        p = archive[f"{name}_odom_p"]
        ct = archive[f"{name}_cmd_t"]
        cp = archive[f"{name}_cmd_p"]
        ax_c.plot(t, p[:, 2], color=COLORS[name], linewidth=1.25, label=f"{label} measured")
        ax_c.plot(ct, cp[:, 2], color=COLORS[name], linewidth=0.8, linestyle="--", alpha=0.8)
    ax_c.set_xlabel("Time in command mode (s)")
    ax_c.set_ylabel("Altitude (m)")
    ax_c.set_title("Vertical tracking")
    ax_c.grid(color="#E8E8E8", linewidth=0.4)
    ax_c.legend(loc="lower right", handlelength=1.7)
    panel_label(ax_c, "c")

    ax_d = fig.add_subplot(grid[1, 2:4])
    for name, label in [("flight_4m", "4 m"), ("flight_8m", "8 m")]:
        t = archive[f"{name}_odom_t"]
        speed = np.linalg.norm(archive[f"{name}_odom_v"], axis=1)
        ax_d.plot(t, speed, color=COLORS[name], linewidth=1.25, label=label)
    ax_d.set_xlabel("Time in command mode (s)")
    ax_d.set_ylabel("Speed (m s$^{-1}$)")
    ax_d.set_title("Measured speed")
    ax_d.grid(color="#E8E8E8", linewidth=0.4)
    ax_d.legend(loc="upper right")
    panel_label(ax_d, "d")

    ax_e = fig.add_subplot(grid[1, 4:6])
    labels = ["4 m", "8 m"]
    names = ["flight_4m", "flight_8m"]
    components = ["preprocess", "inference", "postprocess", "visualization"]
    bottoms = np.zeros(2)
    for index, component in enumerate(components):
        column = index + 1
        values = np.asarray(
            [archive[f"{name}_timing"][:, column].mean() for name in names]
        )
        ax_e.bar(
            np.arange(2),
            values,
            bottom=bottoms,
            width=0.62,
            color=COLORS[component],
            label=component,
        )
        bottoms += values
    totals = np.asarray(
        [archive[f"{name}_timing"][:, 5].mean() for name in names]
    )
    other = np.maximum(totals - bottoms, 0.0)
    ax_e.bar(
        np.arange(2), other, bottom=bottoms, width=0.62, color=COLORS["other"], label="other"
    )
    for x, total in enumerate(totals):
        ax_e.text(x, total + 0.9, f"{total:.1f}", ha="center", fontsize=5.8)
    ax_e.set_xticks(np.arange(2), labels)
    ax_e.set_ylabel("Runtime (ms)")
    ax_e.set_title("Onboard latency")
    ax_e.set_ylim(0, max(totals) * 1.22)
    ax_e.legend(loc="upper center", bbox_to_anchor=(0.5, -0.34), ncol=3, columnspacing=0.7, handletextpad=0.35)
    panel_label(ax_e, "e")

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig3_real_flights.svg", bbox_inches="tight")
    fig.savefig(OUT / "fig3_real_flights.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig3_real_flights.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / "fig3_real_flights.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Real-flight figure written to {OUT}")


if __name__ == "__main__":
    main()
