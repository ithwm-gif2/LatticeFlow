#!/usr/bin/env python3
"""Continuously test one YOPO/LatticeFlow checkpoint using RViz goals.

Start the ROS master, controller, simulator, and RViz manually. Then edit only
the USER CONFIGURATION block below and run this file. Standard RViz
``2D Nav Goal`` messages are accepted from ``/move_base_simple/goal``; their
height is replaced with ``RVIZ_GOAL_HEIGHT`` so a 2-D click cannot command the
quadrotor to descend to z=0.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROS_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, ROS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from test_policy_icra_ros_matched import MatchedICRAPolicyNode  # noqa: E402


# =============================================================================
# USER CONFIGURATION -- normally only edit this block
# =============================================================================

# ``latticeflow`` loads residual-source, physical-anchor, teacher-guided, or
# teacher-free Flow checkpoints automatically from the checkpoint config.
# ``yopo`` loads a standard YOPO .pth checkpoint.
MODEL_TYPE = "latticeflow"  # "latticeflow" or "yopo"

CHECKPOINT = (
    PROJECT_ROOT
    / "runs/icra2027_teacher_free_physical_seed0/checkpoints/best.pt"
)

# This should match the init_x/init_y/init_z used when starting the controller.
# Setting it to the current vehicle position makes the vehicle wait for the
# first RViz goal instead of immediately flying elsewhere.
INITIAL_GOAL = [2.0, -2.0, 2.0]

# Standard RViz 2D Nav Goal publishes z=0. Every clicked goal is forced to this
# world-frame flight height.
RVIZ_GOAL_HEIGHT = 2.0

# Recommended for the trained LatticeFlow model. Set False for raw argmin.
ENABLE_CONTINUITY_SELECTOR = True

# True publishes the candidate/best trajectories for RViz visualization.
VISUALIZE_TRAJECTORIES = True

# Optional point cloud used only for geometric collision reporting. Keep None
# unless it exactly matches the map currently loaded by the simulator.
COLLISION_MAP = None

# Result files are written when the node is stopped with Ctrl+C.
RESULT_TAG = "teacher_free_manual_rviz"

# =============================================================================
# ADVANCED SETTINGS -- usually do not need to edit
# =============================================================================

ARRIVAL_DISTANCE = 4.0
MAX_RUNTIME = 0.0  # 0 keeps the node alive for repeated RViz goals.
SELECTOR_ENDPOINT_WEIGHT = 0.02
SELECTOR_HYSTERESIS_MARGIN = 0.05

ODOM_TOPIC = "/sim/odom"
DEPTH_TOPIC = "/depth_image"
CONTROL_TOPIC = "/so3_control/pos_cmd"


class ManualRvizPolicyNode(MatchedICRAPolicyNode):
    """Matched policy node with a fixed altitude for standard RViz 2-D goals."""

    def __init__(self, settings: dict, rviz_goal_height: float):
        self.rviz_goal_height = float(rviz_goal_height)
        super().__init__(settings)

    def callback_set_goal(self, data):
        self.goal = np.asarray(
            [
                data.pose.position.x,
                data.pose.position.y,
                self.rviz_goal_height,
            ],
            dtype=np.float64,
        )
        self.arrive = False
        self.arrival_time = None
        self.selector.reset()
        print(
            "New RViz goal: "
            f"x={self.goal[0]:.2f}, y={self.goal[1]:.2f}, "
            f"z={self.goal[2]:.2f}"
        )


def validate_configuration() -> Path:
    if MODEL_TYPE not in {"latticeflow", "yopo"}:
        raise ValueError(
            f"MODEL_TYPE must be 'latticeflow' or 'yopo', got {MODEL_TYPE!r}"
        )
    checkpoint = Path(CHECKPOINT).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    if len(INITIAL_GOAL) != 3:
        raise ValueError("INITIAL_GOAL must contain [x, y, z]")
    if ARRIVAL_DISTANCE <= 0.0:
        raise ValueError("ARRIVAL_DISTANCE must be positive")
    return checkpoint


def main() -> None:
    checkpoint = validate_configuration()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = PROJECT_ROOT / "runs/manual_rviz"
    result_stem = result_dir / f"{RESULT_TAG}_{stamp}"

    settings = {
        "policy": MODEL_TYPE,
        "checkpoint": str(checkpoint),
        "goal": [float(value) for value in INITIAL_GOAL],
        "result_md": str(result_stem.with_suffix(".md")),
        "result_json": str(result_stem.with_suffix(".json")),
        "arrival_distance": float(ARRIVAL_DISTANCE),
        "max_runtime": float(MAX_RUNTIME),
        "pitch_angle_deg": 0.0,
        "odom_topic": ODOM_TOPIC,
        "depth_topic": DEPTH_TOPIC,
        "ctrl_topic": CONTROL_TOPIC,
        "plan_from_reference": False,
        "verbose": False,
        "visualize": bool(VISUALIZE_TRAJECTORIES),
        "stop_on_arrival": False,
        "stop_on_collision": False,
        "disable_continuity_selector": not bool(ENABLE_CONTINUITY_SELECTOR),
        "selector_endpoint_weight": float(SELECTOR_ENDPOINT_WEIGHT),
        "selector_hysteresis_margin": float(SELECTOR_HYSTERESIS_MARGIN),
        "collision_map": str(Path(COLLISION_MAP).expanduser().resolve())
        if COLLISION_MAP
        else None,
        "collision_voxel": 0.10,
        "vehicle_radius": 0.30,
        "depth_collision_distance": 0.60,
    }

    print("=" * 72)
    print("Manual RViz policy test")
    print(f"Model type       : {MODEL_TYPE}")
    print(f"Checkpoint       : {checkpoint}")
    print(f"Initial goal     : {INITIAL_GOAL}")
    print(f"RViz goal height : {RVIZ_GOAL_HEIGHT:.2f} m")
    print(f"Selector enabled : {ENABLE_CONTINUITY_SELECTOR}")
    print(f"Result Markdown  : {settings['result_md']}")
    print("Use RViz '2D Nav Goal' on /move_base_simple/goal.")
    print("Press Ctrl+C to stop and write the result files.")
    print("=" * 72)

    ManualRvizPolicyNode(settings, RVIZ_GOAL_HEIGHT)


if __name__ == "__main__":
    main()
