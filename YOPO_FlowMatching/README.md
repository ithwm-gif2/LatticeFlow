# LatticeFlow：无教师物理锚点 Flow Matching

本项目训练一个保持 YOPO 输出与控制接口的深度避障模型，但主模型不加载 YOPO 教师、不加载 YOPO 或 ImageNet 预训练 backbone，所有神经网络权重均随机初始化。部署输入只有单帧深度图、目标方向、速度和加速度；点云、ESDF 和深度投影代价仅用于训练期目标搜索与离线评价。

教师引导版本的历史说明保存在 `archive/README.teacher_guided.md`。当前论文主模型为 teacher-free physical-anchor LatticeFlow。

## 1. 核心训练思路

YOPO 的 `3×5` lattice 为 15 个 cell 提供不同的物理终端位置。第 `l` 个 Flow source 为：

```text
z0_l = [p_anchor_l / position_scale, 0, 0]
```

Flow 在归一化物理终端状态 `[position, velocity, acceleration]` 中积分，结束后通过可微逆变换恢复 YOPO 的 9 维 residual，再沿用原五次多项式轨迹和控制接口。

训练时，每个 cell 的 `x1` 不由教师给出，而由以下候选产生：

1. 物理 anchor 对应的零 residual；
2. 当前学生模型的 detached 输出；
3. 两个 anchor 局部随机扰动；
4. 一个学生输出局部随机扰动。

候选使用 YOPO 轨迹代价、训练期 ESDF 障碍物代价和深度投影安全代价逐 cell 选优。随后执行两步归一化代价梯度下降；只有真实未压缩代价下降时才接受更新。最终目标被 detach，用于 conditional flow matching。该过程允许学生逐步替代初始 anchor，并不需要教师示范。

为确保随机初始化初期稳定，Flow 输出层置零，因此初始 ODE 严格保持 15 个物理 anchor。ResNet-18 仍是网络结构的一部分，但使用随机权重。

## 2. 数据与划分

数据目录：

```text
/home/hwm/A2A_Flow_Matching/YOPO/dataset
```

- 训练：maps 0--6，共 70,000 张；
- 验证：map 7，共 10,000 张；
- 测试：maps 8--9，共 20,000 张。

配置文件为 `configs/icra2027_teacher_free_physical.yaml`。配置中的 `teacher_checkpoint` 仅供现有离线评测器加载公平 YOPO 基线，`TeacherFreePhysicalFlowTrainer` 从不实例化或加载该模型。

## 3. 训练与 TensorBoard

```bash
docker exec -it yopo_fm bash
cd /home/hwm/CF_YOPO/YOPO_FlowMatching
export PYTHONPATH=.

python3 train_teacher_free_physical_icra.py \
  --config configs/icra2027_teacher_free_physical.yaml \
  --run-dir runs/icra2027_teacher_free_physical_seed0 \
  --epochs 20 --num-workers 4
```

每个 batch 都记录 total、flow、endpoint、trajectory、score、local consistency、lattice curvature、梯度范数、目标来源比例、目标改进量和梯度接受率：

```bash
tensorboard \
  --logdir /home/hwm/CF_YOPO/YOPO_FlowMatching/runs/icra2027_teacher_free_physical_seed0/tensorboard \
  --bind_all --port 6006
```

已训练 checkpoint：

```text
runs/icra2027_teacher_free_physical_seed0/checkpoints/best.pt
```

最佳 map-7 validation cost 为 `2.854378`（epoch 17/18 附近达到，checkpoint 记录 epoch 19 的最佳状态信息）。

## 4. MID-360 原生视场与真实无回波训练

真机 MID-360 平置安装时覆盖约 `-7°~+52°`，对应虚拟相机中心仰角
`+22.5°`、垂直 FOV `59°`。旧模型使用以机体前方为中心的约 `60°`
深度，因此仅在端侧调整投影会产生明显的像素行语义偏移。项目使用 Simulator
原生重采该视场，而不是对旧 PNG 做二维平移或拉伸。

完整数据集位于：

```text
/workspace/YOPO/dataset_mid360
```

它包含 10 张地图、每图 10000 帧，共 100000 张 `160×90` 16-bit 深度图；
加载时仍转换为网络输入 `1×96×160`。生成独立数据集（默认拒绝覆盖已有目录）：

```bash
cd /home/hwm/CF_YOPO/YOPO_FlowMatching
bash scripts/generate_mid360_dataset.sh
```

Simulator 的 pose CSV 保存渲染相机姿态。`data.camera_pitch_deg: -22.5` 使加载器按
`R_WB = R_WC R_BC^{-1}` 恢复机体姿态，确保状态输入、地图/ESDF 代价和 ROS 控制接口
仍处于机体前向坐标。训练期 depth-safety 也使用相同外参投影。

仅匹配 FOV 仍不能模拟真实雷达采样。训练输入进一步使用真机采集的 120 帧 raw-return
mask，并严格复现端侧顺序：raw mask → 3×3 小孔洞填充（至少 5 个邻居）→ 世界系
`z=2 m` 虚拟天花板（stride 2）→ 20 m 归一化与量程噪声。地图、ESDF 和原生稠密
深度仅用于训练期 privileged cost；部署仍只输入雷达投影深度和无人机状态。

正式训练保持随机初始化、无 YOPO 教师和无预训练 backbone：

```bash
docker exec -it yopo_fm bash
cd /home/hwm/CF_YOPO/YOPO_FlowMatching
export YOPO_ORIGINAL_ROOT=/workspace/YOPO/YOPO
export PYTHONPATH=.

python3 train_teacher_free_physical_icra.py \
  --config configs/icra2027_teacher_free_physical_mid360.yaml \
  --run-dir runs/icra2027_teacher_free_physical_mid360_seed0 \
  --epochs 20 --num-workers 4
```

TensorBoard 目录为
`runs/icra2027_teacher_free_physical_mid360_seed0/tensorboard`。除原训练指标外，
每个 batch 记录 `domain/input_far_ratio`、`domain/output_far_ratio` 和
`domain/retained_nonfar_ratio`；最佳模型按 `val_lidar/student_selected_cost` 保存。

最终 checkpoint：

```text
runs/icra2027_teacher_free_physical_mid360_seed0/checkpoints/best.pt
```

最佳稀疏验证 cost 为 `2.873524`。在 120 帧真机固定水平状态上，完整轨迹平均
z 包络从旧模型的 `0.583 m` 降到 `0.112 m`，120/120 帧均位于 ±0.3 m。
在相同的 20000 个 held-out MID-360 稀疏输入上，新模型相对旧模型将 selected cost
从 `5.170615` 降到 `2.916884`，collision proxy 从 `57.245%` 降到 `13.425%`。
完整诊断见 `MID360_PERCEPTION_ADAPTATION.md`。

Jetson 独立导出文件为
`engines/lattice_flow_mid360_native_runtime4_nfe6_fp16_metadata.json`，split TensorRT
延迟约 `10.72 ms`。该引擎必须在创建 primitive 之前同时设置
`cfg["train"]=False` 和 `cfg["velocity"]=4.0`；不能只替换 checkpoint 或 metadata。

### Runtime 6 / 10 m 目标评估

保持 native MID-360 checkpoint 不变的 runtime4/runtime6 实飞 bag 重放、模拟
MID-360 五方向闭环结果及风险分析见：

```text
runs/runtime6_goal10_evaluation/RESULTS.md
```

Docker 中复现实验：

```bash
cd /home/hwm/CF_YOPO/YOPO_FlowMatching
bash ros/run_mid360_runtime_goal10_ablation.sh
```

脚本在测试期间临时将模拟 LiDAR 调整为 `-7°~52°`，结束后自动恢复原配置；
虚拟天花板仍作为深度图输入，未改成轨迹层约束。
当前真机默认 `best.pt` 和旧引擎尚未被覆盖。

## 5. 离线评测

```bash
python3 offline_eval_icra.py \
  --checkpoint runs/icra2027_teacher_free_physical_seed0/checkpoints/best.pt \
  --output-dir runs/icra2027_teacher_free_physical_seed0/offline_test \
  --split test --num-workers 4 --visualizations 0

python3 compare_teacher_free_icra.py \
  --residual-checkpoint runs/icra2027_latticeflow_seed0/checkpoints/best.pt \
  --physical-checkpoint runs/icra2027_physical_anchor_seed0/checkpoints/best.pt \
  --teacher-free-checkpoint runs/icra2027_teacher_free_physical_seed0/checkpoints/best.pt \
  --output-dir runs/icra2027_teacher_free_physical_seed0/teacher_free_comparison \
  --num-workers 4 --bootstrap-samples 2000

python3 nfe_eval_icra.py \
  --checkpoint runs/icra2027_teacher_free_physical_seed0/checkpoints/best.pt \
  --output runs/icra2027_teacher_free_physical_seed0/offline_test/nfe.json \
  --nfe 1 2 4 6 8

python3 latency_eval_icra.py \
  --checkpoint runs/icra2027_teacher_free_physical_seed0/checkpoints/best.pt \
  --output runs/icra2027_teacher_free_physical_seed0/offline_test/latency.json \
  --nfe 1 2 4 6 8
```

20,000 个固定测试查询的主要结果：

| 方法 | Selected cost ↓ | Clearance ↑ | Collision proxy ↓ | Switch ↓ |
|---|---:|---:|---:|---:|
| YOPO | 3.1415 | 1.9894 m | 28.48% | 4.51% |
| Residual + teacher | 2.8622 | 3.1433 m | 18.15% | 3.38% |
| Physical + teacher | 2.8611 | 3.0385 m | 18.90% | 3.22% |
| Physical teacher-free | 2.8805 | 3.0421 m | 18.49% | 3.38% |

无教师模型相对公平 YOPO 将 selected cost 降低 8.308%，collision proxy 相对降低 35.1%。相对 physical + teacher，cost 高 0.0193，但 collision proxy 低 0.41 个百分点；因此教师对 cost/endpoint 拟合仍有小幅帮助，但不是主要安全增益的必要条件。

## 6. ROS 闭环实验

自动运行同一 seed-3 地图、同一起点、五个目标的 raw/selector 对比：

```bash
cd /home/hwm/CF_YOPO/YOPO_FlowMatching
bash ros/run_teacher_free_ablation.sh
```

结果写入 `runs/icra_ros_teacher_free/ROS_SUMMARY.md`。raw 和 selector 均为 5/5 成功、0/5 碰撞。continuity-aware selector 将 switch 从 0.145 降至 0.024、endpoint jump 从 0.271 m 降至 0.164 m、jerk proxy 从 6.45 降至 5.56 m/s³。这说明确定性 Flow 并不会自动阻止不同 lattice cell 间的离散切换，闭环 selector 仍然必要。

## 7. 代码入口

- `yopo_flow/self_targets.py`：无教师候选选择和单调代价细化；
- `yopo_flow/teacher_free_trainer.py`：随机初始化训练器与逐 batch TensorBoard；
- `train_teacher_free_physical_icra.py`：训练入口；
- `compare_teacher_free_icra.py`：四模型同查询配对对比；
- `tests/test_teacher_free_physical.py`：初始 anchor、单调目标和配置边界测试；
- `ros/run_teacher_free_ablation.sh`：固定协议闭环实验；
- `TEACHER_FREE_EXPERIMENT.md`：完整实验记录与论文结论边界。

论文工作区位于 `../LatticeFlow_ICRA2027`。

## 8. 无天花板训练消融

已完成一版保持 MID-360 数据、真实回波 mask、局部填充、随机初始化、损失、
seed 0 和 20 epoch 预算不变，仅关闭训练/验证深度图中虚拟天花板的模型：

```text
configs/icra2027_teacher_free_physical_mid360_no_ceiling.yaml
runs/icra2027_teacher_free_physical_mid360_no_ceiling_seed0/checkpoints/best.pt
```

最佳 map-7 LiDAR validation cost 为 `2.877788`，与有天花板训练模型的
`2.873524` 接近。但 2x2 held-out 测试表明：新模型在匹配的无天花板输入上
cost 为 `2.919972`，若推理重新注入天花板则恶化到 `3.292327`。在 120 帧
真实雷达深度上，推理注入天花板后 96.7% 的 runtime-4 轨迹上冲超过 0.3 m；
runtime-6/10 m 目标下为 120/120 帧上冲。因此该权重只保留作消融实验，未替换
真机 checkpoint 或 TensorRT engine。

完整协议、结果与部署结论见 `NO_CEILING_TRAINING_ABLATION.md`。
