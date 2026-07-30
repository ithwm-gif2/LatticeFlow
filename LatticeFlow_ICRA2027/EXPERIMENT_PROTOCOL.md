# Frozen ICRA 2027 Experiment Protocol

## Dataset and leakage control

- Dataset root: `/home/hwm/A2A_Flow_Matching/YOPO/dataset`.
- Training maps: 0--6 (70,000 depth images).
- Validation map: 7 (10,000 depth images).
- Held-out test maps: 8--9 (20,000 depth images).
- Validation and test velocity, acceleration and goal states are deterministically generated per image.
- The fair YOPO teacher/baseline is retrained only on maps 0--6. The original checkpoint trained on all ten maps is excluded from held-out-map claims.

## Training

- Fair YOPO baseline: 20 epochs, batch size 16, AdamW, learning rate $1.5\times10^{-4}$.
- LatticeFlow: 20 epochs, batch size 16, AdamW, learning rate $1.5\times10^{-4}$.
- Main model seed: 0. Additional seeds are required for the final paper if compute time permits.
- TensorBoard records every optimization batch.
- Model selection uses the selected trajectory cost on validation map 7.

## Offline endpoints

- Selected and oracle trajectory cost.
- Smoothness, acceleration, obstacle, goal and depth-safety cost components.
- Minimum projected depth clearance and collision proxy rate.
- Score mean absolute error and selection regret.
- Perturbation-induced lattice-cell switch rate and physical endpoint shift.
- Flow path efficiency, flow curvature and inference latency.
- Results are reported overall and separately for maps 8 and 9.

## Closed-loop endpoints

The independent unit is one complete rollout defined by map, start, goal and simulator seed.

- Primary: success within 1 m before timeout; geometric collision rate.
- Secondary: time to goal, path length, minimum obstacle clearance, mean speed and replanning latency.
- Continuity: lattice-cell switches per second, mean/max terminal-position jump, commanded acceleration variation and jerk proxy.
- Every method must use identical scenarios and controller parameters.
- The depth-only collision proxy is secondary once geometric pose-to-map collision checking is available.

## Baselines and ablations

1. Fair YOPO regression baseline.
2. LatticeFlow without cost refinement (teacher target only).
3. LatticeFlow without local consistency and lattice-curvature regularization.
4. LatticeFlow without continuity-aware selection.
5. LatticeFlow with NFE $\in\{1,2,4,6,8\}$.
6. Same-backbone direct regression to the cost-refined target, if compute permits.

## Statistical reporting

- Do not treat replanning frames as independent closed-loop samples.
- Report trial counts, raw success/collision proportions and 95% bootstrap confidence intervals over trials.
- For continuous rollout metrics, report mean, standard deviation, median and paired trial differences.
- Offline image-query intervals are descriptive; per-map means must also be shown because queries are nested within maps.
- No post-hoc trial exclusion. Simulator failures, timeouts and crashes remain in the denominator and are labelled by cause.

## Real-world placeholder

The anonymous paper reserves a subsection and table for author-run real-quadrotor experiments. No value is entered until the physical trials are completed. Required fields are environment, platform, depth sensor, onboard computer, commanded speed, number of trials, success, collision, minimum clearance, latency and continuity metrics.

