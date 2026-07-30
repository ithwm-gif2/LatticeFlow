# Physical-Anchor LatticeFlow Experiment

## Question

The residual-source implementation initializes every cell with the same internal vector `r0 = 0`; the physical difference appears only through cell conditioning and the YOPO decoder. This experiment asks whether Flow should instead transport explicitly distinct physical terminal states:

\[
z_0^\ell=[p_{\mathrm{anchor}}^\ell/10,\,0,\,0].
\]

The internal state is normalized body-frame terminal position, velocity, and acceleration. The final state is differentiably inverted to the original YOPO residual, so polynomial reconstruction, score output, ROS topics, and the controller interface are unchanged.

## Controlled design

- Same train/validation/test maps: `0--6 / 7 / 8--9`.
- Same 20 epochs and training seed 0.
- Same frozen fair-YOPO teacher and cost-refined target generator.
- Same ResNet-18, vector field, score head, losses, NFE=6, optimizer, and batch size.
- Same 11,468,858 trainable parameters.
- Only the internal flow coordinates and source are changed.
- The 15 normalized anchor positions have radius 0.5 and are all unique; pairwise distance is 0.147--0.650 (mean 0.337).

Velocity and acceleration use fixed training scales `6*sqrt(3)` and position uses 10 m. A tested YOPO residual--physical--residual round trip had maximum absolute error `4.768e-07`.

## Training

```bash
python3 train_physical_anchor_icra.py \
  --config configs/icra2027_physical_anchor.yaml \
  --run-dir runs/icra2027_physical_anchor_seed0 \
  --epochs 20 --num-workers 4
```

- Best checkpoint: `checkpoints/best.pt`.
- Selected epoch field: 20.
- Best validation selected cost: 2.836877.
- Frozen-teacher validation cost: 3.1163.
- Every optimization batch is available in TensorBoard.

## Held-out offline results

All values use the same 20,000 deterministic queries from maps 8 and 9.

| Method | Selected cost ↓ | Oracle cost ↓ | Clearance (m) ↑ | Collision proxy ↓ | Perturb switch ↓ | Endpoint shift (m) ↓ | Selection regret ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOPO | 3.141475 | 2.794607 | 1.989431 | 0.284800 | 0.045100 | 0.116084 | 0.346867 |
| Residual-source | 2.862222 | 2.756476 | 3.143329 | 0.181450 | 0.033800 | 0.101132 | 0.105746 |
| Physical-anchor | 2.861148 | 2.762867 | 3.038480 | 0.189000 | 0.032200 | 0.094083 | 0.098281 |

Paired physical-anchor minus residual-source differences:

| Metric | Difference | Descriptive 95% CI |
|---|---:|---:|
| Selected cost | -0.001074 | [-0.004499, 0.002307] |
| Oracle cost | +0.006391 | [0.004740, 0.008044] |
| Clearance (m) | -0.104849 | [-0.138914, -0.067859] |
| Collision proxy | +0.007550 | [0.004300, 0.010650] |
| Perturb switch | -0.001600 | [-0.005050, 0.002050] |
| Endpoint shift (m) | -0.007049 | [-0.015860, 0.001276] |
| Score MAE | -0.000427 | [-0.000869, -0.000040] |
| Selection regret | -0.007465 | [-0.010521, -0.004381] |

The bootstrap treats image queries as evaluation samples and is descriptive because queries are nested within only two held-out maps. Per-map selected costs were `2.833811 / 2.888486` for physical anchors and `2.836894 / 2.887550` for residual sources on maps 8/9.

## NFE and latency

| NFE | Selected cost ↓ | Collision proxy ↓ | Full network (ms/batch 16) ↓ |
|---:|---:|---:|---:|
| 1 | 2.965417 | 0.184300 | 1.565 |
| 2 | 2.880482 | 0.185450 | 1.687 |
| 4 | 2.863884 | 0.187850 | 2.016 |
| 6 | 2.861148 | 0.189000 | 2.714 |
| 8 | 2.860512 | 0.189900 | 2.630 |

NFE=4 is a practical quality/latency operating point; NFE=6 remains the matched paper setting.

## Matched ROS closed loop

Simulator seed 3, start `[2,-2,2]`, five fixed goals, 60-s timeout, 1-m arrival radius, and 0.3-m geometry collision radius were used. Each row contains five complete rollouts.

| Method | Success | Collision | Time (s) | Clearance (m) | Switch rate | Endpoint jump (m) | Jerk proxy (m/s³) | Forward (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YOPO raw | 5/5 | 0/5 | 9.477 | 0.918 | 0.144 | 0.239 | 6.102 | 1.166 |
| Residual raw | 5/5 | 0/5 | 8.873 | 1.038 | 0.122 | 0.206 | 5.519 | 2.178 |
| Physical raw | 5/5 | 0/5 | 9.130 | 0.970 | 0.143 | 0.217 | 5.594 | 3.233 |
| Residual + selector | 5/5 | 0/5 | 9.733 | 1.033 | 0.026 | 0.152 | 4.718 | 2.183 |
| Physical + selector | 5/5 | 0/5 | 9.669 | 0.966 | 0.028 | 0.164 | 5.279 | 3.247 |

The selector substantially improves both source variants. The physical source does not outperform the residual source on closed-loop continuity in these five goals.

## Interpretation

The explicit physical source is the cleaner mathematical formulation: each cell starts at a different point in the transported space, and the vector field directly moves physical terminal states. However, the controlled experiment does not support uniform performance improvement. Selected cost is tied and selection regret improves, but safety proxies and ROS smoothness are slightly worse than for residual-source flow.

The defensible conclusion is:

> Explicit physical lattice anchors strengthen the structured-source interpretation, while measured navigation gains are more strongly associated with lattice conditioning, cost-refined targets, and continuity-aware selection than with source coordinates alone.

## Reproduction

```bash
python3 tests/test_physical_anchor.py

python3 compare_flow_spaces_icra.py \
  --residual-checkpoint runs/icra2027_latticeflow_seed0/checkpoints/best.pt \
  --physical-checkpoint runs/icra2027_physical_anchor_seed0/checkpoints/best.pt \
  --output-dir runs/icra2027_physical_anchor_seed0/flow_space_comparison \
  --num-workers 4 --bootstrap-samples 2000

bash ros/run_physical_anchor_ablation.sh
```

Limitations: one training seed, two held-out maps, one ROS map, five goals, static obstacles, no matched direct-regression model, and no real-flight result yet.
