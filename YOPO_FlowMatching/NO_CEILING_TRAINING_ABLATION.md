# No-Ceiling Training Ablation

## Purpose

This experiment tests the proposal that the virtual ceiling should be omitted
from training while remaining available in the deployed LiDAR-to-depth
pipeline.  It is a controlled ablation of the native MID-360 teacher-free
physical-anchor model: the dataset, map split, real-return masks, local hole
fill, losses, optimizer, random seed and 20-epoch budget are unchanged.  The
only perception-domain change is
`lidar_domain.virtual_ceiling_enabled: false` during training and validation.

## Training protocol

- Config: `configs/icra2027_teacher_free_physical_mid360_no_ceiling.yaml`
- Dataset: `/workspace/YOPO/dataset_mid360`
- Split: maps 0--6 train, map 7 validation, maps 8--9 test
- Initialization: fully random network weights; zero-initialized Flow output
  at physical lattice anchors
- YOPO teacher/backbone: not loaded
- LiDAR input: real MID-360 return mask, 3x3 local fill with at least five
  neighbours, one iteration, no virtual-ceiling pixels
- Privileged training costs: dense simulator depth, point cloud and ESDF
- Epochs: 20; batch size: 16; seed: 0
- TensorBoard: every-batch loss and domain metrics

Artifacts:

- Run: `runs/icra2027_teacher_free_physical_mid360_no_ceiling_seed0`
- Best checkpoint: `checkpoints/best.pt`
- Best checkpoint SHA-256:
  `851385245f3ab8cc31bf6b1dff4997775fc677b47e4aeccc4b9d86d55c0df282`
- Best validation LiDAR selected cost: `2.877788` at checkpoint epoch 19
- Final epoch checkpoint: `checkpoints/epoch_020.pt`

The ceiling-trained reference reaches `2.873524`; the relative difference is
only about 0.15%, so removing the ceiling does not prevent optimization from
converging on its matched training domain.

## Held-out sparse-LiDAR 2x2 evaluation

The training validation pipeline was rerun on all 20,000 deterministic queries
from held-out maps 8--9.  Cost is always computed against dense privileged map
geometry; only the network perception input changes.

| Training input | Evaluation input | Selected cost down | Oracle cost down | Far ratio |
|---|---|---:|---:|---:|
| Ceiling | No ceiling | 2.989483 | 2.825896 | 90.44% |
| Ceiling | Ceiling | **2.916968** | 2.771743 | 80.85% |
| No ceiling | No ceiling | **2.919972** | 2.770941 | 90.44% |
| No ceiling | Ceiling | 3.292327 | 3.029560 | 80.85% |

The two matched-domain results differ by only 0.10%.  However, injecting the
ceiling at inference into the no-ceiling-trained model increases selected cost
by 12.75%.  This is a perception-domain mismatch, not a failure to fit the
teacher-free trajectory objective.

Raw metrics are stored in
`runs/icra2027_teacher_free_physical_mid360_no_ceiling_seed0/heldout_lidar_ceiling_ablation.json`.

## Recorded real-LiDAR trajectory regression

The same 120 synchronized real MID-360 frames were evaluated without ROS
control publication.  `no ceiling` means real returns plus the deployed local
3x3 fill; `ceiling` is the current deployed depth image including the
world-frame virtual ceiling.  Runtime-4 uses a level 4 m forward goal.

| Model | Input | Mean z envelope | P95 envelope | Within +/-0.3 m | Up >0.3 m | Down <-0.3 m |
|---|---|---:|---:|---:|---:|---:|
| Ceiling-trained | No ceiling | 0.950 m | 1.124 m | 0.0% | 0.0% | 100.0% |
| Ceiling-trained | Ceiling | **0.112 m** | **0.210 m** | **100.0%** | 0.0% | 0.0% |
| No-ceiling-trained | No ceiling | 0.740 m | 0.826 m | 0.0% | 0.0% | 100.0% |
| No-ceiling-trained | Ceiling | 0.479 m | 0.622 m | 3.3% | 96.7% | 0.0% |

At runtime-6 with a 10 m forward goal, the no-ceiling-trained model with the
deployed ceiling input rises above +0.3 m on 120/120 frames (mean envelope
0.768 m).  The ceiling-trained reference has a 0.255 m mean envelope and 65%
of frames remain within +/-0.3 m under the same counterfactual input.

Raw reports:

- `real_lidar_ceiling_ablation_runtime4.json`
- `real_lidar_ceiling_ablation_runtime6_goal10.json`

The reusable read-only evaluator is
`eval_real_lidar_ceiling_ablation.py`; it never initializes ROS or publishes a
control command.

## Decision

Do **not** replace the current flight checkpoint or TensorRT engines with this
no-ceiling-trained model.  It is a valid and useful ablation checkpoint, but
neither tested deployment mode is safe on the recorded real perception domain:
without the ceiling it systematically descends, while adding the ceiling back
at inference produces systematic climb.

If the goal is to reduce ceiling-induced bias while retaining the deployed
ceiling feature, the next defensible experiment is mixed-domain training
(random ceiling dropout or randomized ceiling height) rather than a hard
train-off/test-on split.  The current virtual ceiling remains a depth-image
input; this conclusion does not move it into the trajectory selector or impose
a world-height hard constraint.

