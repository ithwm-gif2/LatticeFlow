# LatticeFlow Research Workspace

This private research repository contains the source code used for the
LatticeFlow teacher-free physical-anchor flow-matching project.

## Repository layout

- `YOPO_FlowMatching/`: training, offline evaluation, ROS simulation, target
  refinement, physical-anchor flow, and MID-360 perception-domain adaptation.
- `Lattice_Flow/`: standalone real-vehicle LiDAR-to-depth inference and ROS
  deployment code.
- `LatticeFlow_ICRA2027/`: anonymous LaTeX manuscript, figure-generation code,
  aggregate tables, and the compiled paper.
- `docker_yopo/`: source snapshot copied from the modified Docker workspace,
  containing the planner interface plus the simulator and controller source.

## Deliberately excluded

The repository does not contain training datasets, generated MID-360 images,
point-cloud maps, rosbags, checkpoints, TensorRT/ONNX engines, TensorBoard
runs, raw real-LiDAR diagnostics, ROS build/devel trees, caches, or LaTeX
intermediate files. See `RELEASE_MANIFEST.md` for the complete boundary.

Expected external paths in the original Docker workflow include:

```text
/workspace/YOPO/dataset
/workspace/YOPO/dataset_mid360
```

Trained weights and real-flight bags must be distributed separately through an
artifact store if release is later approved.

## Main entry points

```text
YOPO_FlowMatching/train_teacher_free_physical_icra.py
YOPO_FlowMatching/offline_eval_icra.py
YOPO_FlowMatching/ros/test_yopo_flow_ros.py
Lattice_Flow/lattice_flow_lidar_node.py
LatticeFlow_ICRA2027/paper/main.tex
```

The code still contains machine-specific example paths from the experimental
workspace. Configure dataset, checkpoint, ROS workspace, and deployment paths
before running it on another machine.

## Third-party notice

The Docker source snapshot is derived from the public upstream project listed
in `THIRD_PARTY.md`. Keep this repository private until redistribution rights,
copyright notices, and the final publication-release plan have been reviewed.
