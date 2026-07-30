# LatticeFlow Jetson LiDAR Deployment

This directory contains the teacher-free physical-anchor checkpoint and a
ROS1 inference node for the Jetson. M-Detector is intentionally not used.

## Inputs and outputs

- LiDAR: `/cloud_registered` (`sensor_msgs/PointCloud2`, frame `world`)
- odometry: `/ekf_quat/ekf_odom` (`nav_msgs/Odometry`, frame `world`)
- RViz goal: `/move_base_simple/goal`
- controller: `/setpoints_cmd` (`quadrotor_msgs/PositionCommand`)
- model input: one normalized `96x160` forward depth image plus body-frame
  velocity, acceleration, and goal vector

The point cloud is transformed into the current body frame and rasterized with
a selectable perspective model. The default `mid360_fov_aligned` mode assumes
a flat MID-360 installation with elevation coverage `-7` to `+52` degrees, so
the virtual optical axis is pitched up by `22.5` degrees. The alternative
`training_camera` mode uses the original horizontal `90x60` degree camera.
Changing `LIDAR_MOUNT_PITCH_DEG` shifts the MID-360 coverage while leaving the
policy trajectory frame unchanged.

LiDAR no-return pixels remain at the simulator's 20 m maximum depth. Only a
locally surrounded small hole is filled (`3x3`, at least five valid neighbors);
global full-image inpainting is not used. The synthetic ceiling is added after
real-LiDAR hole filling. It defaults to an absolute world plane at `z=2.0 m`;
setting `VIRTUAL_CEILING_MODE="body"` instead uses a plane 2.5 m above the
vehicle. The ceiling ray stride remains configurable and defaults to 2.

## Start PyTorch inference

Start the existing LIO/controller process manually and verify that the LiDAR and odometry topics are available. M-Detector is not needed. The policy node does not import any files from the LIO/controller project.

Then source the system ROS environment and start this node:

```bash
source /opt/ros/noetic/setup.bash

cd /home/nv/Lattice_Flow
python3 lattice_flow_lidar_node.py
```

The user configuration is at the top of `lattice_flow_lidar_node.py`. The
default checkpoint is `checkpoints/best.pt`; edit `VIRTUAL_CEILING_ENABLED`,
`VIRTUAL_CEILING_MODE`, `VIRTUAL_CEILING_WORLD_Z`,
`VIRTUAL_CEILING_BODY_HEIGHT`, `LIDAR_PROJECTION_MODE`,
`LIDAR_MOUNT_PITCH_DEG`, `NFE`, or topic constants there when needed.
`GOAL_HEIGHT <= 0` means an RViz 2D goal holds the altitude measured when the
goal is received; a positive value selects an absolute world-frame height.
By default, the first valid odometry message creates an automatic goal 4 m
ahead of the startup yaw at world z=1.5 m. Inference and control therefore do
not wait for an RViz goal; a later RViz goal replaces the automatic goal.

Runtime diagnostics are written to `logs/runtime_diag_YYYYMMDD_HHMMSS.csv`.
Every inference/control row records the raw-best and selected lattice IDs,
body/world endpoint Z, relative goal Z, odometry Z, published expected Z, and
depth valid/far ratios. The same values are throttled to the ROS console as
`[LatticeFlowDiag]` and `[LatticeFlowCtrl]` messages.

## Debug topics

```bash
rostopic hz /cloud_registered
rostopic hz /ekf_quat/ekf_odom
rostopic hz /setpoints_cmd
rostopic hz /lattice_flow/depth_front
```

Published visualization topics:

- `/lattice_flow/depth_front`: normalized network input;
- `/lattice_flow/depth_front_raw`: sparse projected depth before local filling;
- `/lattice_flow/depth_cloud_world`: projected depth pixels in world frame;
- `/lattice_flow/best_traj_visual`: selected trajectory;
- `/lattice_flow/trajs_visual`: all 15 lattice trajectories.

## TensorRT

The validated Jetson TensorRT 8.5 exporter uses three fixed-shape FP16 engines
for the backbone, one reusable Flow step, and the score head:

```bash
cd /home/nv/Lattice_Flow
python3 export_lattice_flow_trt_split.py \
  --checkpoint /home/nv/Lattice_Flow/checkpoints/best.pt \
  --output-base /home/nv/Lattice_Flow/engines/lattice_flow_nfe6_fp16 \
  --runtime-velocity 4.0 \
  --nfe 6 \
  --workspace-mib 1024
```

`--runtime-velocity` is applied before the physical primitive and checkpoint
model are constructed. The generated metadata records the actual velocity,
acceleration and trajectory duration. For the unchanged native MID-360
checkpoint, the validated runtime-6 export is:

```bash
python3 export_lattice_flow_trt_split.py \
  --checkpoint checkpoints/best_mid360_native.pt \
  --output-base engines/lattice_flow_mid360_native_runtime6_nfe6_fp16 \
  --runtime-velocity 6.0 \
  --nfe 6 --workspace-mib 1024
```

The ROS node's `cfg["velocity"]` and selected metadata must use the same
runtime velocity. Merely renaming or swapping an engine is not sufficient.

`lattice_flow_lidar_node.py` defaults to `INFERENCE_BACKEND = "tensorrt"` and
loads `engines/lattice_flow_nfe6_fp16_metadata.json`. Metadata stores the source
checkpoint SHA-256 and the node rejects an engine whose hash or NFE does not
match `checkpoints/best.pt`. On the deployment Jetson, split TensorRT latency was
about 12.47 ms versus 24.87 ms for PyTorch; runtime validation gave maximum raw
and score errors of 0.00164 and 0.00030, respectively.
