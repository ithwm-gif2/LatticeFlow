# LatticeFlow ICRA 2027 Paper Workspace

Double-anonymous manuscript:

> LatticeFlow: Cost-Refined Flow Matching over Motion Primitives for Depth-Based Quadrotor Navigation

The paper presents a teacher-free physical-anchor flow trained from random weights. The only external environment observation at deployment is depth; goal direction, velocity, and acceleration are proprioceptive/navigation state. Maps, point clouds, dense depth, and ESDF are privileged training signals only.

The text deliberately avoids centering the prior one-stage planner. Its name appears only in the method sentence that declares the inspiration for the spherical lattice; elsewhere the comparison is called `direct lattice regression`.

## Evidence boundary

- Standard-view offline test: all 20,000 deterministic queries from held-out maps 8--9.
- Map split: maps 0--6 training, map 7 validation, maps 8--9 final test.
- Main controlled model: teacher-free physical-anchor flow, seed 0, 20 epochs.
- Diagnostics: teacher-guided residual-coordinate and physical-coordinate flows.
- ROS: one random forest (seed 3), one fixed start, five matched goals, raw and selector variants.
- LiDAR domain model: independently retrained same architecture using native MID-360 FOV and 120 recorded raw-return masks.
- Real flight: two independent bags, both entering the 1-m goal region without observed contact; no real baseline and no population success-rate claim.

## Verified headline results

- Held-out selected cost: 3.1415 direct regression to 2.8805 LatticeFlow (8.31% lower).
- Held-out 0.6-m collision proxy: 28.48% to 18.49%.
- Matched ROS: 5/5 success and 0/5 geometric collision; selector switch rate 0.145 to 0.024.
- Sparse MID-360 held-out input: standard-view model 5.1706 cost / 57.25% proxy; sensor-matched model 2.9169 / 13.43%.
- Recorded level-flight regression: mean vertical envelope 0.583 m to 0.112 m; 120/120 sensor-matched frames within +/-0.3 m.
- Real flights: 2/2 descriptive successes, 0/2 observed collisions, maximum measured speed 2.68 m/s.
- Orin NX bag timing: 20.8 ms mean inference and 43.3 ms mean total planning time.

## Source evidence

- Core training/evaluation: `../YOPO_FlowMatching/TEACHER_FREE_EXPERIMENT.md`
- Four-model comparison: `../YOPO_FlowMatching/runs/icra2027_teacher_free_physical_seed0/teacher_free_comparison/COMPARISON.md`
- ROS summary: `../YOPO_FlowMatching/runs/icra_ros_teacher_free/ROS_SUMMARY.md`
- MID-360 adaptation: `../YOPO_FlowMatching/MID360_PERCEPTION_ADAPTATION.md`
- Ceiling ablation: `../YOPO_FlowMatching/NO_CEILING_TRAINING_ABLATION.md`
- Extracted bag data: `data/real_flights/real_flights.npz`
- Reproducible bag metrics: `scripts/analyze_real_flights.py`
- Machine-readable summary: `data/real_flights/real_flight_metrics.json`

## Figures

```bash
cd /home/hwm/CF_YOPO/LatticeFlow_ICRA2027/figures
python3 make_teacher_free_figures.py
```

The scripts export editable SVG/PDF, 600-dpi TIFF, and PNG previews into `paper/figures`. The real-flight source passes the Nature Figure static preflight with 14 PASS, 0 WARN, and 0 FAIL findings.

## Build

```bash
cd /home/hwm/CF_YOPO/LatticeFlow_ICRA2027/paper
/home/hwm/miniconda3/bin/conda run -n paper \
  tectonic main.tex --keep-logs --keep-intermediates
```

Before submission, confirm the final ICRA 2027 page, anonymity, video, and generative-AI disclosure rules. The current scientific limitations are one training seed, two held-out maps, one closed-loop forest, five simulated goals, two real flights, no dynamic obstacles, and no same-backbone direct regressor trained on the identical self-refined targets.
