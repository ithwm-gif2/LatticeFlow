#!/usr/bin/env python3
"""Train named LatticeFlow ablations under the frozen ICRA protocol."""

from __future__ import annotations

import argparse
from pathlib import Path

from yopo_flow.config import load_config
from yopo_flow.icra_trainer import ICRAFlowTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "configs" / "icra2027.yaml"),
    )
    parser.add_argument(
        "--ablation",
        choices=["teacher_only", "no_consistency", "no_student_exploration", "no_depth_safety"],
        required=True,
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    config["training"]["epochs"] = args.epochs
    config["runtime"]["num_workers"] = args.num_workers
    config["project"]["name"] = f"LatticeFlow_{args.ablation}"
    if args.ablation == "teacher_only":
        config["target_refinement"]["enabled"] = False
    elif args.ablation == "no_consistency":
        config["continuity"]["enabled"] = False
        config["loss_weights"]["local_consistency"] = 0.0
        config["loss_weights"]["lattice_curvature"] = 0.0
    elif args.ablation == "no_student_exploration":
        config["target_refinement"]["include_student_seed"] = False
    elif args.ablation == "no_depth_safety":
        config["loss_weights"]["depth_safety"] = 0.0

    trainer = ICRAFlowTrainer(config, run_dir=args.run_dir)
    print(f"Ablation checkpoint: {trainer.fit()}")


if __name__ == "__main__":
    main()
