#!/usr/bin/env python3
"""Train a leakage-free YOPO baseline for the ICRA evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from yopo_flow.config import load_config
from yopo_flow.yopo_baseline_trainer import YOPOSplitTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "configs" / "icra2027.yaml"),
    )
    parser.add_argument(
        "--run-dir",
        default=str(Path(__file__).parent / "runs" / "icra2027_yopo_seed0"),
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        config["baseline"]["epochs"] = args.epochs
    if args.max_train_batches is not None:
        config["baseline"]["max_train_batches"] = args.max_train_batches
    if args.max_val_batches is not None:
        config["baseline"]["max_val_batches"] = args.max_val_batches
    if args.num_workers is not None:
        config["runtime"]["num_workers"] = args.num_workers
    trainer = YOPOSplitTrainer(config, args.run_dir)
    print(f"Best fair YOPO checkpoint: {trainer.fit()}")


if __name__ == "__main__":
    main()
