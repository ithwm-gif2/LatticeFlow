#!/usr/bin/env python3
"""Fair full-network latency benchmark for YOPO and LatticeFlow."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from yopo_flow.checkpoint import load_policy_checkpoint
from yopo_flow.config import resolve_device
from yopo_flow.dataset import configure_yopo_data_root, dataset_from_config
from yopo_flow.targets import YOPOTeacher


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batches", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--nfe", nargs="+", type=int, default=[1, 2, 4, 6, 8])
    return parser.parse_args()


def stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms_per_batch": float(array.mean()),
        "std_ms_per_batch": float(array.std(ddof=1)),
        "median_ms_per_batch": float(np.median(array)),
        "p95_ms_per_batch": float(np.quantile(array, 0.95)),
    }


@torch.inference_mode()
def main():
    args = parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = raw["config"]
    configure_yopo_data_root(config["data"]["root"])
    device = resolve_device(str(config["runtime"]["device"]))
    model, config, _ = load_policy_checkpoint(str(checkpoint), device)
    teacher = YOPOTeacher(config["project"]["teacher_checkpoint"], device)
    loader = DataLoader(
        dataset_from_config(config, "test"),
        batch_size=int(config["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )

    timings: dict[str, list[float]] = {f"latticeflow_nfe_{nfe}": [] for nfe in args.nfe}
    timings["yopo"] = []
    for batch_index, batch in enumerate(loader):
        if batch_index >= args.warmup + args.batches:
            break
        depth, _, _, obs_body, _ = (
            item.to(device, non_blocking=True) for item in batch
        )
        student_prepared = model.prepare_observation(obs_body)
        normalized = teacher.state_transform.normalize_obs(obs_body.clone())
        teacher_prepared = teacher.state_transform.prepare_input(normalized)

        for nfe in args.nfe:
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(depth, student_prepared, num_steps=nfe)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000.0
            if batch_index >= args.warmup:
                timings[f"latticeflow_nfe_{nfe}"].append(elapsed)

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        teacher.policy(depth, teacher_prepared)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000.0
        if batch_index >= args.warmup:
            timings["yopo"].append(elapsed)

    batch_size = int(config["evaluation"]["batch_size"])
    summary = {name: stats(values) for name, values in timings.items()}
    for row in summary.values():
        row["mean_ms_per_query"] = row["mean_ms_per_batch"] / batch_size
    payload = {
        "checkpoint": str(checkpoint),
        "device": str(device),
        "batch_size": batch_size,
        "timed_batches": args.batches,
        "warmup_batches": args.warmup,
        "timing_boundary": "depth backbone plus policy head/ODE and score head; state preparation excluded for both policies",
        "summary": summary,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
