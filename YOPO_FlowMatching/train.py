#!/usr/bin/env python3
"""Train the lattice-conditioned flow matching policy."""

from __future__ import annotations

import argparse
from pathlib import Path

from yopo_flow.config import load_config
from yopo_flow.trainer import FlowTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "configs" / "default.yaml"),
    )
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.max_train_batches is not None:
        config["training"]["max_train_batches"] = args.max_train_batches
    if args.max_val_batches is not None:
        config["training"]["max_val_batches"] = args.max_val_batches
    if args.num_workers is not None:
        config["runtime"]["num_workers"] = args.num_workers

    trainer = FlowTrainer(config, run_dir=args.run_dir)
    if args.resume:
        trainer.load_checkpoint(args.resume, resume_optimizer=True)
    checkpoint = trainer.fit()
    print(f"Training finished. Last checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
