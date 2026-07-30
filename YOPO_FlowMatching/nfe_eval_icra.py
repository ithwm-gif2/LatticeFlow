#!/usr/bin/env python3
"""Evaluate trajectory quality as a function of flow-function evaluations."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from yopo_flow.checkpoint import load_policy_checkpoint
from yopo_flow.config import resolve_device
from yopo_flow.costs import TrajectoryCostEvaluator, gather_lattice
from yopo_flow.dataset import configure_yopo_data_root, dataset_from_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--nfe", nargs="+", type=int, default=[1, 2, 4, 6, 8])
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = raw["config"]
    configure_yopo_data_root(config["data"]["root"])
    device = resolve_device(str(config["runtime"]["device"]))
    model, config, _ = load_policy_checkpoint(str(checkpoint), device)
    evaluator = TrajectoryCostEvaluator(
        config["depth_safety"],
        depth_safety_weight=float(config["loss_weights"]["depth_safety"]),
    ).to(device)
    evaluator.requires_grad_(False)
    loader = DataLoader(
        dataset_from_config(config, "test"),
        batch_size=int(config["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )
    collision_distance = float(config["evaluation"]["collision_distance"])
    values: dict[int, dict[str, list[float]]] = {
        nfe: defaultdict(list) for nfe in args.nfe
    }
    for batch_index, batch in enumerate(tqdm(loader, desc="NFE evaluation", dynamic_ncols=True)):
        if args.max_batches is not None and batch_index >= args.max_batches:
            break
        depth, position, rotation, obs_body, map_id = (
            item.to(device, non_blocking=True) for item in batch
        )
        prepared = model.prepare_observation(obs_body)
        features = model.encode_depth(depth)
        for nfe in args.nfe:
            raw_endstate = model.integrate_from_features(features, prepared, num_steps=nfe)
            score = model.score_from_features(features, prepared, raw_endstate)
            costs = evaluator(
                raw_endstate, depth, position, rotation, obs_body, map_id
            )
            choice = score.flatten(1).argmin(dim=1)
            selected_cost = gather_lattice(costs.total, choice)
            selected_clearance = gather_lattice(costs.min_depth_clearance, choice)
            values[nfe]["selected_cost"].extend(selected_cost.cpu().tolist())
            values[nfe]["oracle_cost"].extend(
                costs.total.flatten(1).amin(dim=1).cpu().tolist()
            )
            values[nfe]["collision_proxy"].extend(
                (selected_clearance < collision_distance).float().cpu().tolist()
            )
            values[nfe]["min_clearance_m"].extend(selected_clearance.cpu().tolist())

    output = {}
    for nfe, metrics in values.items():
        output[str(nfe)] = {}
        for name, items in metrics.items():
            array = np.asarray(items, dtype=np.float64)
            output[str(nfe)][name] = {
                "mean": float(array.mean()),
                "std": float(array.std(ddof=1)),
                "count": int(array.size),
            }
    payload = {
        "checkpoint": str(checkpoint),
        "split": "test",
        "maps": config["data"]["test_maps"],
        "metrics": output,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
