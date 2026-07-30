#!/usr/bin/env python3
"""Export LatticeFlow as fixed-shape TensorRT engines for Jetson TensorRT 8.5.

The Jetson torch2trt build cannot reliably convert the whole physical-anchor
policy.  This exporter therefore builds three engines:

    backbone(depth) -> depth_features
    flow(depth_features, prepared_obs, physical_state, time_features) -> velocity
    score(depth_features, prepared_obs, raw_endstate) -> score_logits

The small observation transform, physical-state projection, Euler rollout and
physical-to-YOPO conversion remain as CUDA PyTorch operations in the ROS node.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "checkpoints/best.pt"
DEFAULT_OUTPUT_BASE = ROOT / "engines/lattice_flow_nfe6_fp16"


class ChannelResidualSum(nn.Module):
    """Replace a residual tensor add with concat plus a fixed 1x1 convolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.sum_conv = nn.Conv2d(2 * channels, channels, kernel_size=1, bias=False)
        with torch.no_grad():
            self.sum_conv.weight.zero_()
            indices = torch.arange(channels)
            self.sum_conv.weight[indices, indices, 0, 0] = 1.0
            self.sum_conv.weight[indices, indices + channels, 0, 0] = 1.0
        self.sum_conv.weight.requires_grad_(False)

    def forward(self, residual: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
        return self.sum_conv(torch.cat((residual, identity), dim=1))


class TensorRTBasicBlock(nn.Module):
    """Numerically equivalent ResNet BasicBlock without ``Tensor.__iadd__``."""

    def __init__(self, source: nn.Module):
        super().__init__()
        self.conv1 = copy.deepcopy(source.conv1)
        self.bn1 = copy.deepcopy(source.bn1)
        self.conv2 = copy.deepcopy(source.conv2)
        self.bn2 = copy.deepcopy(source.bn2)
        self.downsample = copy.deepcopy(source.downsample)
        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)
        self.residual_sum = ChannelResidualSum(source.bn2.num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu2(self.residual_sum(out, identity))


class TensorRTBackbone(nn.Module):
    """YOPO ResNet18 backbone with TensorRT-compatible residual blocks."""

    def __init__(self, source_backbone: nn.Module):
        super().__init__()
        source = source_backbone.cnn
        self.conv1 = copy.deepcopy(source.conv1)
        self.bn1 = copy.deepcopy(source.bn1)
        self.relu = nn.ReLU(inplace=False)
        self.layer1 = nn.Sequential(*(TensorRTBasicBlock(block) for block in source.layer1))
        self.layer2 = nn.Sequential(*(TensorRTBasicBlock(block) for block in source.layer2))
        self.layer3 = nn.Sequential(*(TensorRTBasicBlock(block) for block in source.layer3))
        self.layer4 = nn.Sequential(*(TensorRTBasicBlock(block) for block in source.layer4))
        self.output_layer = copy.deepcopy(source.output_layer)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(depth)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.output_layer(x)


class FlowStepWrapper(nn.Module):
    def __init__(self, policy: nn.Module):
        super().__init__()
        self.flow_head = copy.deepcopy(policy.flow_head)

    def forward(
        self,
        depth_features: torch.Tensor,
        prepared_obs: torch.Tensor,
        physical_state: torch.Tensor,
        time_features: torch.Tensor,
        lattice_embedding: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat(
            (
                depth_features,
                prepared_obs,
                physical_state,
                time_features,
                lattice_embedding,
            ),
            dim=1,
        )
        return self.flow_head(features)


class ScoreWrapper(nn.Module):
    def __init__(self, policy: nn.Module):
        super().__init__()
        self.score_head = copy.deepcopy(policy.score_head)

    def forward(
        self,
        depth_features: torch.Tensor,
        prepared_obs: torch.Tensor,
        raw_endstate: torch.Tensor,
        lattice_embedding: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat(
            (
                depth_features,
                prepared_obs,
                raw_endstate,
                lattice_embedding,
            ),
            dim=1,
        )
        return self.score_head(features)


def make_time_features(policy: nn.Module, steps: int, device: torch.device) -> torch.Tensor:
    half = policy.time_embedding.dim // 2
    frequencies = torch.exp(
        -math.log(policy.time_embedding.max_period)
        * torch.arange(half, device=device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    rows = []
    for step in range(steps):
        value = torch.full(
            (1, 1, policy.vertical_num, policy.horizon_num),
            step / float(steps),
            dtype=torch.float32,
            device=device,
        )
        angles = value * frequencies.view(1, half, 1, 1)
        rows.append(torch.cat((angles.sin(), angles.cos()), dim=1))
    return torch.stack(rows, dim=0).contiguous()


def split_forward(
    policy: nn.Module,
    backbone: nn.Module,
    flow: nn.Module,
    score: nn.Module,
    depth: torch.Tensor,
    prepared_obs: torch.Tensor,
    time_features: torch.Tensor,
    lattice_embedding: torch.Tensor,
    steps: int,
):
    depth_features = backbone(depth)
    state = policy.canonical_source(
        depth.shape[0], depth.device, depth.dtype
    ).contiguous()
    dt = 1.0 / float(steps)
    for step in range(steps):
        velocity = flow(
            depth_features,
            prepared_obs,
            state,
            time_features[step],
            lattice_embedding,
        )
        state = policy._project_flow_state(state + dt * velocity)
    raw = policy.flow_state_to_raw(state)
    score_logits = score(
        depth_features, prepared_obs, raw, lattice_embedding
    )
    scores = F.softplus(score_logits.squeeze(1))
    return raw, scores


def benchmark(function, repeat: int) -> float:
    with torch.inference_mode():
        for _ in range(10):
            function()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeat):
            function()
        torch.cuda.synchronize()
    return 1000.0 * (time.perf_counter() - start) / float(repeat)


def register_explicit_batch_group_norm_converter() -> None:
    """Override the broken GroupNorm converter in the Jetson torch2trt fork.

    The bundled converter produces invalid dynamic shape/slice layers on this
    fixed-shape TensorRT 8.5 build.  This replacement uses static explicit-batch
    dimensions and leaves the external library untouched.
    """
    import tensorrt as trt
    from torch2trt import tensorrt_converter

    @tensorrt_converter("torch.nn.functional.group_norm")
    def convert_group_norm(ctx):
        input_tensor = ctx.method_args[0]
        num_groups = int(ctx.method_args[1])
        weight = ctx.method_args[2] if len(ctx.method_args) > 2 else None
        bias = ctx.method_args[3] if len(ctx.method_args) > 3 else None
        eps = float(ctx.method_args[4]) if len(ctx.method_args) > 4 else 1e-5
        output = ctx.method_return

        input_shape = tuple(int(value) for value in input_tensor.shape)
        batch_size = input_shape[0]
        channels = input_shape[1]
        spatial_shape = input_shape[2:]
        split_shape = (
            batch_size,
            num_groups,
            channels // num_groups,
        ) + spatial_shape

        reshape_in = ctx.network.add_shuffle(input_tensor._trt)
        reshape_in.reshape_dims = split_shape
        grouped = reshape_in.get_output(0)

        axes = 0
        for axis in range(2, len(split_shape)):
            axes |= 1 << axis
        mean = ctx.network.add_reduce(
            grouped, trt.ReduceOperation.AVG, axes, True
        ).get_output(0)
        centered = ctx.network.add_elementwise(
            grouped, mean, trt.ElementWiseOperation.SUB
        ).get_output(0)
        squared = ctx.network.add_elementwise(
            centered, centered, trt.ElementWiseOperation.PROD
        ).get_output(0)
        variance = ctx.network.add_reduce(
            squared, trt.ReduceOperation.AVG, axes, True
        ).get_output(0)

        constant_shape = tuple(1 for _ in split_shape)
        eps_constant = ctx.network.add_constant(
            constant_shape, np.full(constant_shape, eps, dtype=np.float32)
        ).get_output(0)
        variance_eps = ctx.network.add_elementwise(
            variance, eps_constant, trt.ElementWiseOperation.SUM
        ).get_output(0)
        std = ctx.network.add_unary(
            variance_eps, trt.UnaryOperation.SQRT
        ).get_output(0)
        normalized = ctx.network.add_elementwise(
            centered, std, trt.ElementWiseOperation.DIV
        ).get_output(0)

        reshape_out = ctx.network.add_shuffle(normalized)
        reshape_out.reshape_dims = input_shape
        result = reshape_out.get_output(0)

        if weight is not None:
            scale = weight.detach().cpu().numpy().astype(np.float32)
        else:
            scale = np.ones(channels, dtype=np.float32)
        if bias is not None:
            shift = bias.detach().cpu().numpy().astype(np.float32)
        else:
            shift = np.zeros(channels, dtype=np.float32)
        power = np.ones(channels, dtype=np.float32)
        result = ctx.network.add_scale_nd(
            result, trt.ScaleMode.CHANNEL, shift, scale, power, 1
        ).get_output(0)
        output._trt = result


def build_engine(name: str, module: nn.Module, inputs, fp16: bool, workspace_mib: int):
    from torch2trt import torch2trt

    build_inputs = [tensor.detach().clone().contiguous() for tensor in inputs]
    print(f"Building {name} TensorRT engine...")
    converted = torch2trt(
        module,
        build_inputs,
        fp16_mode=bool(fp16),
        max_workspace_size=int(workspace_mib) * 1024 * 1024,
    )
    if getattr(converted, "engine", None) is None:
        network = getattr(converted, "network", None)
        if network is not None:
            print(f"TensorRT network diagnostics for failed {name} build:")
            for index in range(network.num_layers):
                if index < 95 or index > 125:
                    continue
                layer = network.get_layer(index)
                input_shapes = []
                for input_index in range(layer.num_inputs):
                    tensor = layer.get_input(input_index)
                    input_shapes.append(None if tensor is None else tuple(tensor.shape))
                output_shapes = []
                for output_index in range(layer.num_outputs):
                    tensor = layer.get_output(output_index)
                    output_shapes.append(None if tensor is None else tuple(tensor.shape))
                print(
                    f"  layer={index} name={layer.name!r} type={layer.type} "
                    f"inputs={input_shapes} outputs={output_shapes}"
                )
        raise RuntimeError(f"TensorRT conversion returned engine=None for {name}")
    return converted


def save_engine(module: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(module.state_dict(), str(path))


def load_engine(path: Path, device: torch.device) -> nn.Module:
    from torch2trt import TRTModule

    module = TRTModule()
    module.load_state_dict(torch.load(str(path), map_location=device))
    return module.to(device).eval()


def error_stats(actual: torch.Tensor, expected: torch.Tensor):
    difference = (actual - expected).abs()
    return float(difference.max().cpu()), float(difference.mean().cpu())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--nfe", type=int, default=6)
    parser.add_argument(
        "--runtime-velocity",
        type=float,
        default=None,
        help=(
            "Deployment velocity in m/s. Defaults to config velocity while "
            "forcing cfg['train']=False before policy construction."
        ),
    )
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--workspace-mib", type=int, default=1024)
    parser.add_argument("--benchmark-repeat", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TensorRT conversion")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.nfe < 1:
        raise ValueError("--nfe must be positive")

    from config.config import cfg as yopo_cfg

    runtime_velocity = float(
        yopo_cfg["velocity"]
        if args.runtime_velocity is None
        else args.runtime_velocity
    )
    if runtime_velocity <= 0.0:
        raise ValueError("--runtime-velocity must be positive")
    yopo_cfg["train"] = False
    yopo_cfg["velocity"] = runtime_velocity

    from yopo_flow.checkpoint import load_policy_checkpoint

    torch.manual_seed(7)
    device = torch.device("cuda")
    policy, config, _ = load_policy_checkpoint(str(args.checkpoint), device)
    if config.get("project", {}).get("policy_variant") != "physical_anchor":
        raise ValueError("Expected a physical_anchor checkpoint")
    policy.eval()
    primitive = policy.state_transform.lattice_primitive
    print(
        "Runtime primitive: "
        f"velocity={runtime_velocity:.3f}m/s, "
        f"vel_max={primitive.vel_max:.3f}m/s, "
        f"acc_max={primitive.acc_max:.3f}m/s^2, "
        f"segment_time={primitive.segment_time:.3f}s"
    )

    register_explicit_batch_group_norm_converter()

    backbone = TensorRTBackbone(policy.image_backbone).to(device).eval()
    flow = FlowStepWrapper(policy).to(device).eval()
    score = ScoreWrapper(policy).to(device).eval()
    time_features = make_time_features(policy, args.nfe, device)
    lattice_embedding = policy.lattice_embedding.detach().clone().contiguous()

    depth = torch.rand((1, 1, 96, 160), dtype=torch.float32, device=device)
    obs = torch.tensor(
        [[2.0, 0.2, -0.1, 0.1, -0.1, 0.05, 4.0, 1.0, 0.5]],
        dtype=torch.float32,
        device=device,
    )
    prepared_obs = policy.prepare_observation(obs)
    with torch.inference_mode():
        original_features = policy.encode_depth(depth)
        compatible_features = backbone(depth)
        backbone_error = error_stats(compatible_features, original_features)
        reference_raw, reference_score = policy(
            depth, prepared_obs, num_steps=args.nfe
        )
        split_raw, split_score = split_forward(
            policy,
            backbone,
            flow,
            score,
            depth,
            prepared_obs,
            time_features,
            lattice_embedding,
            args.nfe,
        )
        split_raw_error = error_stats(split_raw, reference_raw)
        split_score_error = error_stats(split_score, reference_score)
    print("PyTorch wrapper equivalence:")
    print("  backbone max/mean:", backbone_error)
    print("  raw max/mean:", split_raw_error)
    print("  score max/mean:", split_score_error)
    if max(backbone_error[0], split_raw_error[0], split_score_error[0]) > 2e-3:
        raise RuntimeError("TensorRT-compatible PyTorch wrappers are not equivalent")

    state = policy.canonical_source(1, device, torch.float32).contiguous()
    raw_for_score = reference_raw.contiguous()
    fp16 = not args.fp32
    flow_trt = build_engine(
        "flow-step",
        flow,
        (
            original_features,
            prepared_obs,
            state,
            time_features[0],
            lattice_embedding,
        ),
        fp16,
        args.workspace_mib,
    )
    backbone_trt = build_engine(
        "backbone", backbone, (depth,), fp16, args.workspace_mib
    )
    score_trt = build_engine(
        "score",
        score,
        (original_features, prepared_obs, raw_for_score, lattice_embedding),
        fp16,
        args.workspace_mib,
    )

    base = args.output_base.expanduser().resolve()
    paths = {
        "backbone": base.with_name(base.name + "_backbone.trt.pth"),
        "flow": base.with_name(base.name + "_flow.trt.pth"),
        "score": base.with_name(base.name + "_score.trt.pth"),
        "metadata": base.with_name(base.name + "_metadata.json"),
    }
    save_engine(backbone_trt, paths["backbone"])
    save_engine(flow_trt, paths["flow"])
    save_engine(score_trt, paths["score"])

    loaded_backbone = load_engine(paths["backbone"], device)
    loaded_flow = load_engine(paths["flow"], device)
    loaded_score = load_engine(paths["score"], device)
    with torch.inference_mode():
        converted_raw, converted_score = split_forward(
            policy,
            loaded_backbone,
            loaded_flow,
            loaded_score,
            depth,
            prepared_obs,
            time_features,
            lattice_embedding,
            args.nfe,
        )
    raw_error = error_stats(converted_raw, reference_raw)
    score_error = error_stats(converted_score, reference_score)

    torch_latency = benchmark(
        lambda: policy(depth, prepared_obs, num_steps=args.nfe),
        max(1, args.benchmark_repeat),
    )
    trt_latency = benchmark(
        lambda: split_forward(
            policy,
            loaded_backbone,
            loaded_flow,
            loaded_score,
            depth,
            prepared_obs,
            time_features,
            lattice_embedding,
            args.nfe,
        ),
        max(1, args.benchmark_repeat),
    )

    metadata = {
        "mode": "split",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "precision": "fp32" if args.fp32 else "fp16",
        "nfe": args.nfe,
        "runtime": {
            "train": False,
            "velocity": runtime_velocity,
            "primitive_vel_max": float(primitive.vel_max),
            "primitive_acc_max": float(primitive.acc_max),
            "segment_time": float(primitive.segment_time),
        },
        "engines": {name: str(path) for name, path in paths.items() if name != "metadata"},
        "input_shapes": {
            "depth": [1, 1, 96, 160],
            "prepared_obs": [1, 9, 3, 5],
            "physical_state": [1, 9, 3, 5],
            "time_features": [1, policy.time_embedding.dim, 3, 5],
            "raw_endstate": [1, 9, 3, 5],
            "lattice_embedding": list(lattice_embedding.shape),
        },
        "lattice_embedding": lattice_embedding.detach().cpu().tolist(),
        "validation": {
            "raw_max_abs_error": raw_error[0],
            "raw_mean_abs_error": raw_error[1],
            "score_max_abs_error": score_error[0],
            "score_mean_abs_error": score_error[1],
        },
        "latency_ms": {
            "pytorch": torch_latency,
            "tensorrt_split": trt_latency,
        },
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    if raw_error[0] > 0.15 or score_error[0] > 0.15:
        raise RuntimeError("TensorRT numerical error exceeds the deployment limit")
    print("TensorRT split export and reload validation: OK")


if __name__ == "__main__":
    main()
