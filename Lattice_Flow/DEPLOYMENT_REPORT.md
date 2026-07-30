# Jetson deployment report

Date: 2026-07-26

## Synchronized package

Remote directory:

```text
/home/nv/Lattice_Flow
```

Checkpoint SHA-256:

```text
4c5bdd75213b64df74d21c5bbb56c9852e51125880546418e889931039d2d48b
```

The package vendors the required YOPO lattice, state-transform, polynomial solver, and controller helpers inside this directory.
No Python module is imported from another project, and M-Detector is not used. The generated `quadrotor_msgs` Python package is also included locally.

## Verified

- Jetson model loading on CUDA: passed;
- checkpoint variant: `physical_anchor`;
- output: `[1, 9, 3, 5]` residuals and `[1, 3, 5]` scores;
- synthetic LiDAR projection: `96x160`, finite output;
- virtual ceiling: selectable absolute-world or body-relative ray-plane intersection, added after real-depth hole filling;
- ROS node registration/startup without sensors: passed;
- Python 3.8 bytecode compilation on Jetson: passed;
- standalone runtime audit: system ROS + project working directory only;
- vendored `quadrotor_msgs` import and CUDA checkpoint loading: passed;
- ROS lifecycle test: node remained alive until the 18 s test timeout;
- ROS processes from debugging were cleaned after testing.

## TensorRT status

The physical-anchor checkpoint at `/home/nv/Lattice_Flow/checkpoints/best.pt`
was converted successfully with `export_lattice_flow_trt_split.py`. The source
checkpoint SHA-256 is recorded in metadata and checked again at node startup.
The deployment uses three FP16 engines: backbone, reusable Flow step, and score
head. The small physical-state projection and YOPO residual inverse remain CUDA
PyTorch operations.

Validated runtime comparison on the Jetson:

- PyTorch model latency: 24.87 ms;
- split TensorRT latency: 12.47 ms;
- maximum raw output error: 0.00164;
- maximum score error: 0.00030.

`lattice_flow_lidar_node.py` now defaults to `INFERENCE_BACKEND = "tensorrt"`.

## Automatic startup goal

The first valid odometry message creates a goal 4 m ahead of the vehicle's
startup yaw at absolute world z=1.5 m. This marks the goal as active, so policy
inference and `/setpoints_cmd` publication begin without waiting for RViz. A
later RViz goal replaces the automatic goal.

## Runtime command

```bash
source /opt/ros/noetic/setup.bash
cd /home/nv/Lattice_Flow
python3 lattice_flow_lidar_node.py
```
