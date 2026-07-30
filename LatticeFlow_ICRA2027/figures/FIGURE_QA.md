# Figure QA Notes

## Figure 1: method schematic

- Core conclusion: LatticeFlow begins from physically distinct motion-primitive anchors, constructs teacher-free targets with privileged trajectory costs, and integrates a deterministic conditional flow.
- Evidence chain: depth/state input -> physical source lattice -> anchor/student/local candidate search -> monotonic cost refinement -> terminal-state decoder and trajectory selection.
- Archetype: schematic-led composite; no quantitative observations.

## Figure 2: held-out offline evidence

- Core conclusion: LatticeFlow improves selected trajectory quality and safety/local-sensitivity proxies over direct lattice regression on both held-out maps; four to six NFE capture nearly all of the gain.
- Panel a: overall selected cost plus both per-map means.
- Panel b: collision proxy, perturbation switch, and endpoint shift normalized to direct regression.
- Panel c: full-network latency versus selected cost for NFE 1, 2, 4, 6, and 8, with teacher-guided diagnostics shown for context.
- Data: all 20,000 fixed queries; no query filtering.

## Figure 3: real-flight evidence

- Core conclusion: a sensor-matched LatticeFlow instance completes two collision-free LiDAR-only flights with bounded altitude and onboard latency.
- Panels a--b: all command-mode odometry over accumulated 8-cm voxelized LiDAR returns and sampled planned trajectories.
- Display restriction: cloud points are shown only for world $z=0.3$--2.5 m to suppress ground/overhead clutter; no flight or odometry sample is excluded.
- Panels c--d: altitude and speed for every command-mode odometry sample.
- Panel e: measured mean preprocessing, inference, postprocessing, visualization, and residual timing for both bags.
- Statistical boundary: two trials are descriptive feasibility evidence, not a success-rate estimate.
- Static preflight: 14 PASS, 0 WARN, 0 FAIL.

## Export contract

- Final width: 183 mm.
- Backend: Python/Matplotlib only.
- Editable SVG/PDF text, PDF font type 42.
- PNG preview at 300 dpi and TIFF at 600 dpi.
- Restrained blue/orange/gray palette; no rainbow mapping or red/green-only encoding.
