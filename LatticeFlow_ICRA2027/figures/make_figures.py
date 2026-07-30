#!/usr/bin/env python3
"""Generate submission figures from verified physical-anchor experiments.

Figure contract
---------------
Core conclusion: both flow coordinate systems improve over YOPO, while the
explicit physical source matches selected cost without dominating safety.
Evidence: panel a compares held-out cost and maps; panel b compares safety and
local sensitivity; panel c reports the physical-anchor NFE/latency trade-off.
Archetype: schematic-led method figure plus a quantitative three-panel grid.
All panels use the complete 20,000-query result files without filtering.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT.parent / "YOPO_FlowMatching" / "runs"
COMPARISON = (
    RUNS
    / "icra2027_physical_anchor_seed0"
    / "flow_space_comparison"
    / "comparison.json"
)
NFE = RUNS / "icra2027_physical_anchor_seed0" / "offline_test" / "nfe.json"
LATENCY = (
    RUNS / "icra2027_physical_anchor_seed0" / "offline_test" / "latency.json"
)
OUT = ROOT / "paper" / "figures"

width_mm = 183

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 6.5,
        "axes.labelsize": 6.5,
        "axes.titlesize": 7.0,
        "xtick.labelsize": 6.0,
        "ytick.labelsize": 6.0,
        "legend.fontsize": 5.8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "legend.frameon": False,
    }
)

COLORS = {
    "yopo": "#8A8F98",
    "residual": "#3274A1",
    "residual_light": "#9CC7DF",
    "physical": "#D98C3F",
    "physical_light": "#F1C38F",
    "safe": "#3F8F6B",
    "danger": "#C65A5A",
    "ink": "#222222",
    "muted": "#666666",
}


def save_pub_py(fig: plt.Figure, stem: str, dpi: int = 600) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.tiff", dpi=dpi, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")


def rounded_box(axis, xy, width, height, text, facecolor, edgecolor="#D0D4D8"):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        linewidth=0.8,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    axis.add_patch(patch)
    axis.text(
        xy[0] + width / 2,
        xy[1] + height - 0.055,
        text,
        ha="center",
        va="top",
        weight="bold",
        fontsize=7,
        color=COLORS["ink"],
    )
    return patch


def arrow(axis, start, end, color=COLORS["muted"], connectionstyle="arc3"):
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=1.0,
            color=color,
            connectionstyle=connectionstyle,
        )
    )


def make_method_figure() -> None:
    fig, axis = plt.subplots(figsize=(width_mm / 25.4, 55 / 25.4))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    rounded_box(axis, (0.01, 0.12), 0.19, 0.76, "Depth + ego state", "#F4F6F7")
    depth = np.linspace(0.2, 1.0, 80).reshape(8, 10)
    axis.imshow(depth, extent=(0.035, 0.175, 0.43, 0.73), cmap="Blues", aspect="auto")
    axis.text(0.105, 0.35, r"$D_t,\;v_t,\;a_t,\;g_t$", ha="center", fontsize=7)
    axis.text(
        0.105,
        0.24,
        "single-frame\nmap-free deployment",
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

    rounded_box(axis, (0.48, 0.12), 0.24, 0.76, "Cost-refined physical target", "#EFF7F2")
    labels = [
        ("YOPO", 0.535, 0.66, COLORS["yopo"]),
        ("student", 0.60, 0.57, COLORS["residual"]),
        ("local", 0.655, 0.67, COLORS["physical"]),
    ]
    for text_label, x, y, color in labels:
        axis.scatter([x], [y], s=26, color=color, zorder=3)
        axis.text(x, y + 0.06, text_label, ha="center", fontsize=5.8)
        arrow(axis, (x, y - 0.015), (0.60, 0.45), color=color)
    axis.scatter([0.60], [0.45], s=34, marker="*", color=COLORS["safe"], zorder=4)
    axis.text(0.60, 0.35, r"$z_1^\ell=S\psi_\ell(r_1^\ell)$", ha="center", fontsize=6.2)
    axis.text(
        0.60,
        0.25,
        "ESDF + projected depth\nselection and refinement",
        ha="center",
        va="center",
        color=COLORS["muted"],
    )

    rounded_box(axis, (0.755, 0.12), 0.235, 0.76, "Physical flow + YOPO inverse", "#EFF5F8")
    for offset, color in [
        (-0.06, COLORS["residual_light"]),
        (0.0, COLORS["residual"]),
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
    axis.text(0.87, 0.39, r"$z_K\rightarrow r_K\rightarrow$ polynomial", ha="center", va="center")
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
    save_pub_py(fig, "fig1_method")
    plt.close(fig)


def make_results_figure() -> None:
    comparison = json.loads(COMPARISON.read_text(encoding="utf-8"))
    nfe = json.loads(NFE.read_text(encoding="utf-8"))["metrics"]
    latency = json.loads(LATENCY.read_text(encoding="utf-8"))["summary"]
    summary = comparison["summary"]
    per_map = comparison["per_map"]

    fig, axes = plt.subplots(1, 3, figsize=(width_mm / 25.4, 57 / 25.4))
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.25, top=0.89, wspace=0.47)

    # Panel a: complete-query means plus the two held-out map means.
    ax = axes[0]
    methods = ["YOPO", "Residual", "Physical"]
    means = [
        summary["yopo_selected_cost"]["mean"],
        summary["residual_selected_cost"]["mean"],
        summary["physical_selected_cost"]["mean"],
    ]
    colors = [COLORS["yopo"], COLORS["residual"], COLORS["physical"]]
    ax.bar(np.arange(3), means, color=colors, width=0.62)
    for map_id in ["8", "9"]:
        points = [
            per_map[map_id]["yopo_selected_cost"],
            per_map[map_id]["residual_selected_cost"],
            per_map[map_id]["physical_selected_cost"],
        ]
        ax.plot(
            np.arange(3),
            points,
            color="#444444",
            linewidth=0.7,
            alpha=0.65,
            marker="o",
            markersize=2.8,
        )
        ax.text(2.22, points[2], f"map {map_id}", va="center", fontsize=5.5, color=COLORS["muted"])
    ax.set_xticks(np.arange(3), methods, rotation=16)
    ax.set_ylabel("Selected trajectory cost")
    ax.set_ylim(2.5, 3.35)
    ax.set_xlim(-0.55, 2.55)
    ax.set_title("Held-out trajectory quality")
    ax.text(1.50, 3.28, "R vs P: matched", ha="center", weight="bold", color=COLORS["safe"], fontsize=5.8)
    ax.text(-0.18, 1.12, "a", transform=ax.transAxes, weight="bold", fontsize=8)

    # Panel b: three metrics normalized to YOPO, using all 20,000 queries.
    ax = axes[1]
    metric_keys = ["collision_proxy", "perturb_switch", "endpoint_shift_m"]
    yopo = np.array([summary[f"yopo_{key}"]["mean"] for key in metric_keys])
    residual = 100.0 * np.array([summary[f"residual_{key}"]["mean"] for key in metric_keys]) / yopo
    physical = 100.0 * np.array([summary[f"physical_{key}"]["mean"] for key in metric_keys]) / yopo
    x = np.arange(3)
    width = 0.24
    ax.bar(x - width, np.full(3, 100.0), color=COLORS["yopo"], width=width, label="YOPO")
    ax.bar(x, residual, color=COLORS["residual"], width=width, label="Residual")
    ax.bar(x + width, physical, color=COLORS["physical"], width=width, label="Physical")
    ax.set_xticks(x, ["collision\nproxy", "cell\nswitch", "endpoint\nshift"])
    ax.set_ylabel("Relative to YOPO (%)")
    ax.set_ylim(0, 112)
    ax.set_title("Safety and local sensitivity")
    ax.legend(loc="lower left", ncol=1)
    ax.text(-0.18, 1.12, "b", transform=ax.transAxes, weight="bold", fontsize=8)

    # Panel c: physical-anchor NFE curve; residual six-NFE point is contextual.
    ax = axes[2]
    nfe_values = [1, 2, 4, 6, 8]
    x_latency = [latency[f"latticeflow_nfe_{value}"]["mean_ms_per_batch"] for value in nfe_values]
    y_cost = [nfe[str(value)]["selected_cost"]["mean"] for value in nfe_values]
    ax.plot(
        x_latency,
        y_cost,
        color=COLORS["physical"],
        marker="o",
        linewidth=1.2,
        markersize=3.5,
        label="Physical",
    )
    for value, x_value, y_value in zip(nfe_values, x_latency, y_cost):
        ax.text(x_value + 0.025, y_value + 0.004, str(value), fontsize=5.5, color=COLORS["physical"])
    ax.scatter([1.958], [summary["residual_selected_cost"]["mean"]], s=24, color=COLORS["residual"], marker="D", zorder=3)
    ax.text(1.99, summary["residual_selected_cost"]["mean"] - 0.015, "Residual (6)", fontsize=5.5, color=COLORS["residual"])
    yopo_latency = latency["yopo"]["mean_ms_per_batch"]
    yopo_cost = summary["yopo_selected_cost"]["mean"]
    ax.scatter([yopo_latency], [yopo_cost], s=22, color=COLORS["yopo"], marker="s", zorder=3)
    ax.text(yopo_latency + 0.035, yopo_cost, "YOPO", va="center", fontsize=5.8)
    ax.set_xlabel("Full-network latency (ms / batch of 16)")
    ax.set_ylabel("Selected trajectory cost")
    ax.set_title("NFE--latency trade-off")
    ax.set_xlim(1.15, 2.9)
    ax.set_ylim(2.83, 3.20)
    ax.text(-0.18, 1.12, "c", transform=ax.transAxes, weight="bold", fontsize=8)

    save_pub_py(fig, "fig2_offline_results")
    plt.close(fig)


if __name__ == "__main__":
    make_method_figure()
    make_results_figure()
    print(f"Figures written to {OUT}")
