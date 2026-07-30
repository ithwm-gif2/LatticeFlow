# Test Results

本文件记录实际执行过的测试。离线结果与 ROS 闭环结果分开报告，未执行的测试不会写成已通过。

## 1. Static and integration smoke test

- Python syntax compilation: passed.
- Dataset loading: passed; 90,000 training and 10,000 validation samples.
- Student output: `[B, 9, 3, 5]`.
- Score output: `[B, 3, 5]`.
- YOPO teacher checkpoint loading: passed.
- Global ESDF and depth-image cost forward/backward: passed.
- End-to-end gradient propagation: passed.
- Two-sample target refinement example:
  - teacher mean cost: 2.066665;
  - refined target mean cost: 2.012351;
  - target improvement: 0.054314.

## 2. One-batch training-loop smoke test

- Optimizer, checkpoint and TensorBoard writing: passed.
- TensorBoard contains per-batch loss and all cost components.
- After robust cost compression and disabling fp16 by default, gradient norm is finite.
- Example gradient norm: 8.462062 before clipping to the configured maximum.

## 3. Pilot training

- Training batches: 500.
- Validation batches: 100.
- Mean training total loss: 0.4020.
- Mean training student raw trajectory cost: 3.9924.
- Validation selected cost:
  - Flow student: 3.1893;
  - YOPO teacher: 3.0137.
- Interpretation: the flow student approached the teacher after 500 batches but had not yet surpassed it; full training was continued from this checkpoint.

### Continued training

- Total optimizer steps after continuation: 23,000.
- Continued schedule: four complete 5,625-batch epochs after the 500-batch pilot.
- Final full-validation selected cost during training:
  - Flow student: 2.8758;
  - YOPO teacher: 2.9873.
- Final checkpoint: `runs/pilot_500/checkpoints/epoch_005.pt`.
- Validation-best checkpoint: `runs/pilot_500/checkpoints/best.pt`.

## 4. Full offline evaluation

- Checkpoint: `runs/pilot_500/checkpoints/best.pt`.
- Validation samples: 10,000.
- Selected total cost:
  - Flow student: 2.863047;
  - YOPO teacher: 2.970282;
  - relative improvement: **3.610%**.
- Oracle total cost:
  - Flow student: 2.743007;
  - YOPO teacher: 2.764645.
- Mean selected minimum depth clearance:
  - Flow student: 3.198035 m;
  - YOPO teacher: 2.589514 m.
- Collision proxy rate at 0.6 m:
  - Flow student: 18.970%;
  - YOPO teacher: 23.330%.
- Flow path efficiency: 0.987307.
- Flow curvature: 0.000374.
- Mean network inference: 1.435 ms per batch in the offline benchmark.
- Full report: `runs/pilot_500/offline_eval/OFFLINE_RESULTS.md`.

## 5. ROS closed-loop evaluation

- Trials: 3.
- Goal-reaching success: **3/3 (100%)** using YOPO's 5 m arrival radius.
- Collision proxy triggered: **0/3 (0%)** at the 0.6 m depth threshold.
- Mean final distance: 2.4287 m.
- Mean path length: 28.1334 m.
- Mean minimum observed depth: 0.6886 m.
- Mean flow forward time: 2.466 ms.
- Mean full depth callback time: 3.216 ms.
- Aggregate report: `runs/pilot_500/ROS_RESULTS_SUMMARY.md`.
- Individual reports:
  - `runs/pilot_500/ROS_RESULTS.md`;
  - `runs/pilot_500/ROS_RESULTS_goal_25_10.md`;
  - `runs/pilot_500/ROS_RESULTS_goal_25_m10.md`.
- The `--stop-on-arrival` path was also tested with goal `[15, 0, 2]`:
  - arrival time: 4.16 s;
  - final distance: 4.9693 m;
  - minimum observed depth: 0.9629 m;
  - flow forward time: 2.342 ms;
  - report: `runs/pilot_500/ROS_RESULTS_stop_on_arrival.md`.

The simulator does not publish a direct physical collision flag. ROS collision results therefore use the minimum valid camera depth as a proxy and should be interpreted together with trajectory visualization in future large-scale evaluation.
