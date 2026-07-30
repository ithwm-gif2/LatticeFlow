# Real-Quadrotor Completion Protocol

This file is the author-facing checklist for replacing the anonymous manuscript's real-world `TBD` fields. Simulation numbers must never be copied into the physical-flight table.

## Fixed deployment boundary

- Network inputs: one forward depth frame, body-frame goal direction, velocity, and acceleration.
- No map, point cloud, ESDF, image history, or action history at inference.
- The terminal-state output, fifth-order polynomial reconstruction, and SO(3) control interface remain identical for YOPO and LatticeFlow.

## Hardware fields to report

- Airframe dimensions and mass.
- Depth-camera model, resolution, rate, usable range, and mounting pitch.
- State-estimation source and update rate.
- Onboard computer, GPU/accelerator, power mode, and software precision.
- Control, replanning, and depth-frame rates.
- Commanded speed and acceleration limits.
- Environment dimensions, obstacle type, lighting, and whether motion capture is used only for evaluation or also for state estimation.

## Matched trial definition

One independent sample is a complete start--goal flight. YOPO, LatticeFlow with raw argmin, and LatticeFlow with continuity-aware selection must use the same starts, goals, speed limits, controller gains, safety pilot rules, and timeout. Do not count depth frames or replans as independent samples.

Required outcomes per trial:

- success within 1 m before timeout;
- physical collision or safety-pilot intervention;
- time to goal and path length;
- minimum geometric clearance, when motion-capture/map ground truth is available;
- mean speed;
- lattice switches per second;
- mean and maximum terminal-position jump;
- commanded-acceleration variation or jerk proxy;
- mean network-forward and full planning callback latency.

Failures, timeouts, process crashes, and safety interventions remain in the denominator and receive an explicit failure label.

## Suggested CSV schema

```text
trial_id,method,start_x,start_y,start_z,goal_x,goal_y,goal_z,
speed_limit,success,collision,safety_intervention,timeout,
time_to_goal_s,path_length_m,min_clearance_m,mean_speed_mps,
lattice_switches,switch_rate_hz,mean_endpoint_jump_m,max_endpoint_jump_m,
mean_jerk_mps3,p95_jerk_mps3,mean_forward_ms,mean_callback_ms,failure_cause
```

## Manuscript completion rule

Report raw counts together with proportions. For continuous metrics, report mean, standard deviation, median, and paired trial differences. If the physical trial count is too small for a stable confidence interval, state the count and avoid significance language.
