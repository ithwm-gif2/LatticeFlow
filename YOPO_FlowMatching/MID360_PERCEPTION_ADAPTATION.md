# MID-360 Perception-domain Adaptation Report

## Decision

Perception mismatch does not always require retraining. Deterministic errors in frame transforms,
FOV, depth convention, normalization, hole filling, or virtual-ceiling order should be corrected
first. Here, deterministic fixes alone were insufficient: all 15 trajectories remained outside a
±0.3 m level-flight band on the measured real input, so a new perception-domain training set was
required.

## Real-sensor evidence

A no-control diagnostic node captured 120 synchronized frames from the deployed projection pipeline:

- source cloud: `/cloud_registered`, world frame;
- image: 160×96, MID-360 elevation -7° to +52°, 90° horizontal FOV;
- raw real-return ratio: 10.05% ± 0.20%;
- final model-input far/no-return ratio: 72.39% ± 0.18%;
- small-hole fill: 3×3, at least five valid neighbours, one iteration;
- virtual ceiling: world z=2 m, inserted after real-return filling, stride 2;
- control publisher during capture: none.

The old training camera placed the horizon at row 48, whereas the full MID-360 projection places it
near row 83. Reverting only to the old camera projection increased far pixels to 84.2%. Reusing the
aggressive verified-YOPO inpainting reduced far pixels almost to zero but caused systematic downward
motion. A preprocessing sweep found no stable level-flight region: weak filling climbed, strong
filling descended, and intermediate settings switched between cells.

## Native simulator dataset

The Simulator natively rendered a new independent dataset at camera pitch -22.5° (camera-up is
negative in its convention), corresponding to a +22.5° optical-axis elevation and approximately
-7° to +52° vertical coverage:

- root: `/workspace/YOPO/dataset_mid360`;
- maps: 10;
- images per map: 10,000;
- total images: 100,000;
- stored image: 160×90 16-bit PNG, resized to 160×96 by the loader;
- map split: train 0–6, validation 7, test 8–9;
- size: approximately 2.3 GB.

The generator stores camera pose in CSV. The loader recovers body pose with
`R_WB = R_WC R_BC^{-1}` before sampling state or computing map/ESDF costs. The depth-safety
projector uses the same -22.5° extrinsic.

Each dense simulator frame is converted to the deployment perception domain using a randomly chosen
real raw-return mask, local hole filling, the world-frame ceiling, 20 m normalization, range-dependent
noise, and quantization. Dense depth and ESDF remain privileged training-only signals.

## Training

- configuration: `configs/icra2027_teacher_free_physical_mid360.yaml`;
- run: `runs/icra2027_teacher_free_physical_mid360_seed0`;
- epochs: 20;
- initialization: all network weights random; Flow output zero-initialized at physical anchors;
- YOPO teacher/backbone: not loaded;
- best sparse validation selected cost: 2.873524 at epoch 20;
- checkpoint: `checkpoints/best.pt`.

## Real-depth level-flight regression

The same 120 real depth frames were evaluated with `velocity=[4,0,0]`, zero acceleration, and a
level 4 m forward goal. The metric is the maximum absolute z excursion over the complete 2.5 s
quintic trajectory.

| Metric | Original model | Native MID-360 model |
|---|---:|---:|
| Mean z envelope | 0.583 m | 0.112 m |
| 95th-percentile envelope | 0.630 m | 0.210 m |
| Frames within ±0.3 m | 0/120 | 120/120 |
| Frames climbing above +0.3 m | 120/120 | 0/120 |
| Frames descending below -0.3 m | 0/120 | 0/120 |
| Mean terminal z | +0.583 m | +0.024 m |

## Same-input held-out-map comparison

Both old and new LatticeFlow policies received exactly the same 20,000 sparse MID-360 inputs from
held-out maps 8–9, with identical states and cost evaluation.

| Metric | Original model | Native MID-360 model |
|---|---:|---:|
| Selected cost ↓ | 5.170615 | 2.916884 |
| Oracle cost ↓ | 3.523252 | 2.771550 |
| Mean clearance ↑ | 2.864776 m | 6.981882 m |
| Collision proxy ↓ | 57.245% | 13.425% |

The selected-cost reduction is 43.59%.

## Jetson TensorRT verification

The checkpoint was copied without replacing the deployed `best.pt` and exported under an independent
`runtime4` name. Both engine construction and runtime decoding explicitly set `train=False` and
`velocity=4.0` before creating the lattice primitive.

- checkpoint SHA-256: `9be70cfa9f6b0b6e1d02b5b90dbd5e66d90339e324c22deec90a3ca9457e976b`;
- TensorRT raw max absolute error: 0.002202;
- TensorRT score max absolute error: 0.000376;
- PyTorch latency: 24.05 ms;
- split TensorRT latency: 10.72 ms;
- real 120-frame mean z envelope: 0.1116 m;
- real 120-frame 95th-percentile envelope: 0.2111 m;
- frames within ±0.3 m: 120/120.

The current deployed default checkpoint and engine were not changed. A guarded no-control live node
check and subsequent closed-loop flight are still required before making the new runtime the default.
