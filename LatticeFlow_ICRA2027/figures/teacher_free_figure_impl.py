#!/usr/bin/env python3
"""Generate the teacher-free method and verified four-model result figures.

Figure contract
---------------
Core conclusion: a randomly initialized, teacher-free physical-anchor flow
improves over direct lattice regression, while teacher guidance gives a small
cost advantage and the selector can improve closed-loop continuity.
Evidence: Fig. 1 shows self-improved target construction; Fig. 2a
compares complete-query and per-map cost; Fig. 2b compares safety/sensitivity;
Fig. 2c reports the teacher-free NFE/latency trade-off. No query is filtered.
Archetypes: schematic-led composite and quantitative grid. Backend: Python.
Exports: editable SVG/PDF plus 600-dpi TIFF and PNG preview at 183 mm width.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from make_figures import OUT, arrow, rounded_box, save_pub_py, width_mm


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT.parent / "YOPO_FlowMatching" / "runs"
COMPARISON = (
    RUNS
    / "icra2027_teacher_free_physical_seed0"
    / "teacher_free_comparison"
    / "comparison.json"
)
NFE = RUNS / "icra2027_teacher_free_physical_seed0" / "offline_test" / "nfe.json"
LATENCY = (
    RUNS
    / "icra2027_teacher_free_physical_seed0"
    / "offline_test"
    / "latency.json"
)

COLORS = {
    "direct": "#8A8F98",
    "residual": "#3274A1",
    "physical": "#D98C3F",
    "teacher_free": "#3F8F6B",
    "teacher_free_light": "#A9D4BF",
    "ink": "#222222",
    "muted": "#666666",
    "panel": "#F5F7F8",
}


def make_method_figure() -> None:
    fig, axis = plt.subplots(figsize=(width_mm / 25.4, 55 / 25.4))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    rounded_box(axis, (0.01, 0.12), 0.19, 0.76, "Depth + ego state", "#F4F6F7")
    depth = np.linspace(0.15, 1.0, 80).reshape(8, 10)
    axis.imshow(
        depth,
        extent=(0.035, 0.175, 0.43, 0.73),
        cmap="Blues",
        aspect="auto",
        zorder=2,
    )
    axis.text(0.105, 0.35, r"$D_t,\;v_t,\;a_t,\;g_t$", ha="center", fontsize=7)
    axis.text(
        0.105,
        0.24,
        "randomly initialized\nsingle-frame policy",
        ha="center",
        va="center",
        color=COLORS["muted"],
    )

    rounded_box(axis, (0.235, 0.12), 0.21, 0.76, "Explicit physical sources", "#FBF4EA")
    frustum = Polygon(
        [[0.27, 0.50], [0.415, 0.73], [0.415, 0.27]],
        closed=True,
        facecolor="#F7E7D3",
        edgecolor=COLORS["physical"],
        linewidth=0.8,
    )
    axis.add_patch(frustum)
    xs = np.linspace(0.33, 0.405, 5)
    ys = np.linspace(0.36, 0.64, 3)
    for y in ys:
        axis.scatter(
            xs,
            np.linspace(0.50 + 0.4 * (y - 0.50), y, 5),
            s=10,
            color=COLORS["physical"],
            zorder=3,
        )
    axis.text(
        0.34,
        0.20,
        r"$z_0^\ell=[p_{\rm anc}^\ell/s_p,0,0]$",
        ha="center",
        fontsize=6.2,
    )
    axis.text(0.34, 0.29, "15 distinct terminal positions", ha="center", color=COLORS["muted"])

    rounded_box(axis, (0.48, 0.12), 0.24, 0.76, "Teacher-free target search", "#EFF7F2")
    labels = [
        ("anchor", 0.535, 0.66, COLORS["physical"]),
        ("student", 0.60, 0.57, COLORS["teacher_free"]),
        ("local", 0.655, 0.67, COLORS["residual"]),
    ]
    for text_label, x, y, color in labels:
        axis.scatter([x], [y], s=26, color=color, zorder=3)
        axis.text(x, y + 0.06, text_label, ha="center", fontsize=5.8)
        arrow(axis, (x, y - 0.015), (0.60, 0.45), color=color)
    axis.scatter([0.60], [0.45], s=34, marker="*", color=COLORS["teacher_free"], zorder=4)
    axis.text(0.60, 0.35, r"$z_1^\ell=S\psi_\ell(r_1^\ell)$", ha="center", fontsize=6.2)
    axis.text(
        0.60,
        0.25,
        "ESDF + projected depth\nselect, refine, detach",
        ha="center",
        va="center",
        color=COLORS["muted"],
    )

    rounded_box(axis, (0.755, 0.12), 0.235, 0.76, "Flow + terminal-state decoder", "#EFF5F8")
    for offset, color in [
        (-0.06, COLORS["teacher_free_light"]),
        (0.0, COLORS["teacher_free"]),
        (0.06, COLORS["physical"]),
    ]:
        arrow(
            axis,
            (0.785, 0.52 + offset),
            (0.91, 0.54 + 0.5 * offset),
            color=color,
            connectionstyle=f"arc3,rad={0.25 * offset / 0.06 if offset else 0}",
        )
    axis.text(0.845, 0.69, "$K$ Euler steps", ha="center", fontsize=6.5)
    axis.text(0.87, 0.39, r"$z_K\rightarrow[p_T,v_T,a_T]\rightarrow$ polynomial", ha="center", va="center")
    axis.text(
        0.87,
        0.23,
        "15 scored trajectories +\ncontinuity-aware selection",
        ha="center",
        va="center",
        color=COLORS["muted"],
    )

    arrow(axis, (0.20, 0.50), (0.235, 0.50), color=COLORS["ink"])
    arrow(axis, (0.445, 0.50), (0.48, 0.50), color=COLORS["ink"])
    arrow(axis, (0.72, 0.50), (0.755, 0.50), color=COLORS["ink"])
    axis.text(0.006, 0.96, "a", weight="bold", fontsize=8, va="top")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.98, bottom=0.02)
    save_pub_py(fig, "fig1_method", dpi=600)
    plt.close(fig)


def make_results_figure() -> None:
    comparison = json.loads(COMPARISON.read_text(encoding="utf-8"))
    nfe = json.loads(NFE.read_text(encoding="utf-8"))["metrics"]
    latency = json.loads(LATENCY.read_text(encoding="utf-8"))["summary"]
    summary = comparison["summary"]
    per_map = comparison["per_map"]

    method_keys = ["yopo", "residual", "physical", "teacher_free"]
    method_labels = ["Direct", "R+T", "P+T", "Ours"]
    colors = [
        COLORS["direct"],
        COLORS["residual"],
        COLORS["physical"],
        COLORS["teacher_free"],
    ]

    fig, axes = plt.subplots(1, 3, figsize=(width_mm / 25.4, 57 / 25.4))
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.25, top=0.89, wspace=0.48)

    ax = axes[0]
    means = [summary[f"{key}_selected_cost"]["mean"] for key in method_keys]
    ax.bar(np.arange(4), means, color=colors, width=0.64)
    for map_id in ["8", "9"]:
        points = [per_map[map_id][f"{key}_selected_cost"] for key in method_keys]
        ax.plot(
            np.arange(4),
            points,
            color="#444444",
            linewidth=0.7,
            alpha=0.62,
            marker="o",
            markersize=2.6,
        )
        ax.text(3.22, points[3], f"map {map_id}", va="center", fontsize=5.4, color=COLORS["muted"])
    ax.set_xticks(np.arange(4), method_labels, rotation=16)
    ax.set_ylabel("Selected trajectory cost")
    ax.set_ylim(2.5, 3.35)
    ax.set_xlim(-0.55, 3.55)
    ax.set_title("Held-out trajectory quality")
    ax.text(2.5, 3.27, "teacher-free: −8.31%", ha="center", weight="bold", color=COLORS["teacher_free"], fontsize=5.6)
    ax.text(-0.18, 1.12, "a", transform=ax.transAxes, weight="bold", fontsize=8)

    ax = axes[1]
    metric_keys = ["collision_proxy", "perturb_switch", "endpoint_shift_m"]
    direct = np.array([summary[f"yopo_{key}"]["mean"] for key in metric_keys])
    x = np.arange(3)
    width = 0.19
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width
    for offset, method, label, color in zip(offsets, method_keys, method_labels, colors):
        normalized = 100.0 * np.array(
            [summary[f"{method}_{key}"]["mean"] for key in metric_keys]
        ) / direct
        ax.bar(x + offset, normalized, color=color, width=width, label=label)
    ax.set_xticks(x, ["collision\nproxy", "cell\nswitch", "endpoint\nshift"])
    ax.set_ylabel("Relative to direct regression (%)")
    ax.set_ylim(0, 112)
    ax.set_title("Safety and local sensitivity")
    ax.legend(loc="lower left", ncol=2, columnspacing=0.7, handletextpad=0.35)
    ax.text(-0.18, 1.12, "b", transform=ax.transAxes, weight="bold", fontsize=8)

    ax = axes[2]
    nfe_values = [1, 2, 4, 6, 8]
    x_latency = [latency[f"latticeflow_nfe_{value}"]["mean_ms_per_batch"] for value in nfe_values]
    y_cost = [nfe[str(value)]["selected_cost"]["mean"] for value in nfe_values]
    ax.plot(
        x_latency,
        y_cost,
        color=COLORS["teacher_free"],
        marker="o",
        linewidth=1.2,
        markersize=3.5,
    )
    label_offsets = {1: (0.025, 0.006), 2: (0.025, 0.006), 4: (0.025, 0.008), 6: (-0.06, -0.020), 8: (0.025, 0.009)}
    for value, x_value, y_value in zip(nfe_values, x_latency, y_cost):
        dx, dy = label_offsets[value]
        ax.text(x_value + dx, y_value + dy, str(value), fontsize=5.5, color=COLORS["teacher_free"])
    reference_points = [
        (latency["yopo"]["mean_ms_per_batch"], summary["yopo_selected_cost"]["mean"], "Direct", COLORS["direct"], "s"),
        (1.958, summary["residual_selected_cost"]["mean"], "R+T", COLORS["residual"], "D"),
        (2.714, summary["physical_selected_cost"]["mean"], "P+T", COLORS["physical"], "^"),
    ]
    for x_value, y_value, label, color, marker in reference_points:
        ax.scatter([x_value], [y_value], s=23, color=color, marker=marker, zorder=3)
        ax.text(x_value + 0.035, y_value, label, va="center", fontsize=5.5, color=color)
    ax.set_xlabel("Full-network latency (ms / batch of 16)")
    ax.set_ylabel("Selected trajectory cost")
    ax.set_title("Teacher-free NFE trade-off")
    ax.set_xlim(1.15, 2.9)
    ax.set_ylim(2.83, 3.20)
    ax.text(-0.18, 1.12, "c", transform=ax.transAxes, weight="bold", fontsize=8)

    save_pub_py(fig, "fig2_offline_results", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    make_method_figure()
    make_results_figure()
    print(f"Teacher-free figures written to {OUT}")
