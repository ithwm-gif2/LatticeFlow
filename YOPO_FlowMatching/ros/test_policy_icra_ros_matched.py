#!/usr/bin/env python3
"""Matched ROS evaluator with raw-argmin continuity metrics when selection is disabled."""

from __future__ import annotations

import numpy as np
import torch

from test_policy_icra_ros import ICRAPolicyNode, parse_args


class MatchedICRAPolicyNode(ICRAPolicyNode):
    """Always pass the chosen action through the metric tracker.

    The parent evaluator skipped the tracker in its disabled-selector branch,
    making raw argmin switch and endpoint-jump values identically zero.  Here,
    disabled selection is represented by zero endpoint weight and a negative
    hysteresis margin, so the selected action is exactly the raw per-frame
    argmin while all continuity statistics remain observable.
    """

    def process_output(self, endstate_pred, score_pred, return_all_preds=False):
        raw = endstate_pred.reshape(9, self.lattice_primitive.traj_num).T
        scores = score_pred.reshape(self.lattice_primitive.traj_num).astype(np.float64)
        lattice_ids = torch.arange(self.lattice_primitive.traj_num - 1, -1, -1)
        endpoints = self.state_transform.pred_to_endstate_cpu(raw, lattice_ids)
        selection = self.selector.select(scores, endpoints)
        action_id = selection.index
        adjusted_scores = selection.adjusted_scores.copy()
        adjusted_scores[action_id] = min(adjusted_scores.min(), scores.min()) - 1e-6

        if return_all_preds:
            return endpoints, adjusted_scores
        return endpoints[action_id][None, :], float(scores[action_id])


def settings_from_args(args):
    endpoint_weight = args.selector_endpoint_weight
    hysteresis_margin = args.selector_hysteresis_margin
    if args.disable_continuity_selector:
        endpoint_weight = 0.0
        hysteresis_margin = -1.0e12
    return {
        "policy": args.policy,
        "checkpoint": args.checkpoint,
        "runtime_velocity": args.runtime_velocity,
        "goal": args.goal,
        "result_md": args.result_md,
        "result_json": args.result_json,
        "arrival_distance": args.arrival_distance,
        "max_runtime": args.max_runtime,
        "pitch_angle_deg": args.pitch_angle_deg,
        "odom_topic": args.odom_topic,
        "depth_topic": args.depth_topic,
        "ctrl_topic": args.ctrl_topic,
        "plan_from_reference": args.plan_from_reference,
        "verbose": args.verbose,
        "visualize": not args.no_visualize,
        "stop_on_arrival": args.stop_on_arrival,
        "stop_on_collision": args.stop_on_collision,
        "disable_continuity_selector": args.disable_continuity_selector,
        "selector_endpoint_weight": endpoint_weight,
        "selector_hysteresis_margin": hysteresis_margin,
        "collision_map": args.collision_map,
        "collision_voxel": args.collision_voxel,
        "vehicle_radius": args.vehicle_radius,
        "depth_collision_distance": args.depth_collision_distance,
    }


if __name__ == "__main__":
    MatchedICRAPolicyNode(settings_from_args(parse_args()))
