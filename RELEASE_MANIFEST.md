# Clean Release Manifest

Snapshot prepared on 2026-07-29 from:

- host workspace: `/home/hwm/CF_YOPO`;
- Docker workspace: `/workspace/YOPO` in container `yopo_fm`.

## Included

- LatticeFlow training, evaluation, ROS, and deployment source;
- configuration files and reproducibility documentation;
- anonymous manuscript source, vector/PDF figures, and compiled PDF;
- modified Docker planner, simulator source, and controller source.

## Excluded

- `/workspace/YOPO/dataset`, `dataset_mid360`, and pilot datasets;
- all host `runs/` directories and TensorBoard event files;
- `*.pt`, `*.pth`, `*.ckpt`, `*.onnx`, and TensorRT engines;
- rosbags and extracted raw real-flight arrays;
- real-LiDAR mask banks and diagnostic captures;
- Docker `YOPO/saved`, simulator point-cloud maps, and generation logs;
- ROS `build/`, `devel/`, and generated workspaces;
- Python caches, temporary files, and LaTeX intermediates;
- the complete A2A third-party repository and the local paper-PDF library.

The `.gitignore` repeats these exclusions to reduce the chance of accidentally
adding data or binary artifacts in future commits.
