# YOPO Flow Matching

本项目将 YOPO 的视锥 lattice 与 A2A 的条件 flow matching 思路结合，用单帧深度图、无人机当前速度、加速度和目标方向生成 15 条候选避障轨迹。模型保持原 YOPO 的输出与控制接口，可以继续使用其五次多项式轨迹生成器和 SO3 控制器。

代码持久化在宿主机：

```text
/home/hwm/CF_YOPO/YOPO_FlowMatching
```

训练和测试在 `yopo_fm` Docker 容器中运行，原 YOPO 源码、教师权重和数据集仍从 `/workspace/YOPO` 读取。

## 1. 为什么不能只用代价直接定义普通 flow matching

标准 conditional flow matching 需要明确的目标样本 `x1`。碰撞距离、平滑度和目标引导等指标只能判断轨迹好坏，本身并不是 `x1`。如果直接将这些指标作用于 ODE 终点，可以训练一个 cost-guided neural ODE，但不再具有标准 flow matching 的速度监督，容易发生模式坍缩、轨迹集中到少数 lattice 或训练不稳定。

本项目使用“教师初始化 + 代价改进”的方式把指标转化为明确的 `x1`：

1. 冻结的 YOPO 教师输出初始末端状态。
2. 将当前 flow 学生输出加入候选集合，使学生已经探索到的更优轨迹能够反过来成为新目标。
3. 在教师附近采样随机候选。
4. 根据 YOPO 全局 ESDF 代价和当前深度图投影安全代价，为每个 lattice 独立选择最佳候选。
5. 对选中末端状态执行少量可微代价下降，得到细化后的 `x1`。
6. 用该 `x1` 训练从 canonical lattice `x0` 出发的条件速度场。

因此，教师只提供初始搜索中心，不是模型性能上限。学生候选、随机探索和可微代价下降共同构成 policy-improvement 循环。

## 2. Lattice 与输出定义

沿用 YOPO 的 `vertical_num=3`、`horizon_num=5`，共 15 个 lattice cell。每个 cell 的网络原始状态为：

```text
[delta_yaw, delta_pitch, radial_offset,
 terminal_vx, terminal_vy, terminal_vz,
 terminal_ax, terminal_ay, terminal_az]
```

所有维度位于 YOPO 的归一化空间 `[-1, 1]`。零向量对应 canonical lattice：角度无偏移、终点距离为 `radio_range`、末端速度和加速度为零。因此：

```text
x0 = 每个 lattice cell 的九维零偏移
x1 = 教师/学生探索后经轨迹代价细化的九维末端状态
```

推理时对条件速度场执行默认 6 步 Euler 积分，然后预测每条轨迹的单调对数代价分数。选取最低分 cell 后，继续使用 YOPO 的 `pred_to_endstate` 和五次多项式求出期望位置、速度、加速度。

## 3. 网络条件

模型输入包括：

- 16 位单通道深度图，读取后归一化到 `[0, 1]` 并缩放为 `96×160`；
- 当前速度 `v_xyz`；
- 当前加速度 `a_xyz`；
- 机体系目标方向 `g_xyz`。

深度图经与 YOPO 相同的单通道 ResNet-18 编码为 `3×5` 特征网格，并使用 YOPO 教师的图像 backbone 初始化。状态通过原 YOPO 的 primitive-frame 变换对齐到每个 lattice cell。

速度网络在每个 lattice cell 上共享参数，条件为：

```text
depth feature + primitive-frame state + x_t + time embedding + lattice embedding
```

## 4. 损失函数

每个 batch 优化：

```text
L = wf * L_flow
  + we * L_endpoint
  + wt * L_trajectory
  + ws * L_score
```

- `L_flow`：`v_theta(x_t,t,c)` 与 `x1-x0` 的均方误差。
- `L_endpoint`：ODE 积分终点与细化 `x1` 的均方误差。
- `L_trajectory`：积分终点生成的实际五次多项式轨迹代价。
- `L_score`：预测分数与轨迹总代价的 Smooth-L1 损失。

轨迹代价包括：

- 五次多项式 jerk 平滑代价；
- 加速度积分代价；
- 目标方向引导代价；
- 基于完整地图点云 ESDF 的碰撞代价；
- 基于当前深度图的局部安全代价。

深度安全代价将轨迹点投影回深度图。YOPO 相机坐标系为 `+x` 向前、`+y` 向左、`+z` 向上：

```text
u = cx - fx * y / x
v = cy - fy * z / x
clearance = observed_depth(u,v) - x
```

当 clearance 小于安全距离时施加平滑 Softplus 惩罚。ESDF 的指数碰撞代价偶尔会产生极大梯度，训练目标和 score 标签使用单调的 `log(1+cost)` 压缩；候选优劣排序不受影响。

## 5. 数据划分

按照原 YOPO 的方式，对每张地图内部使用固定随机种子执行 90%/10% 图像划分：

- 训练集：90,000 张图像；
- 验证集：10,000 张图像；
- 10 张地图均同时出现在训练集与验证集。

训练输入为深度图和随机采样的速度、加速度、目标方向。相机 pose 和地图 pointcloud 只用于构建训练/评估代价，不是部署时的网络输入。

## 6. 目录结构

```text
YOPO_FlowMatching/
├── configs/default.yaml          # 模型、损失和训练配置
├── yopo_flow/
│   ├── model.py                  # lattice-conditioned flow 网络
│   ├── costs.py                  # ESDF 与深度图轨迹代价
│   ├── targets.py                # 教师、探索和 x1 细化
│   ├── trainer.py                # 训练、验证、TensorBoard、checkpoint
│   └── checkpoint.py             # 测试时加载模型
├── train.py                      # 训练入口
├── offline_eval.py               # 离线评估和轨迹投影图
├── ros/test_yopo_flow_ros.py     # ROS 闭环入口
├── tests/smoke_test.py           # 单 batch 全链路测试
└── TEST_RESULTS.md               # 已执行测试记录
```

## 7. 环境与快速检查

代码依赖当前 `yopo_fm` 容器中的 PyTorch、OpenCV、Open3D、SciPy、TensorBoard、ROS 和原 YOPO。flow matching 公式在本项目内实现，不依赖容器中缺失的 `torchcfm`。

```bash
docker exec -it yopo_fm bash
cd /home/hwm/CF_YOPO/YOPO_FlowMatching
export PYTHONPATH=.
python3 tests/smoke_test.py --batch-size 2
```

## 8. 训练

完整训练：

```bash
docker exec -it yopo_fm bash
cd /home/hwm/CF_YOPO/YOPO_FlowMatching
export PYTHONPATH=.
python3 train.py --config configs/default.yaml
```

限制 batch 数的调试训练：

```bash
python3 train.py \
  --epochs 1 \
  --max-train-batches 10 \
  --max-val-batches 10 \
  --num-workers 0 \
  --run-dir runs/debug
```

续训：

```bash
python3 train.py \
  --epochs 50 \
  --run-dir runs/experiment_name \
  --resume runs/experiment_name/checkpoints/epoch_005.pt
```

checkpoint 保存完整模型、优化器、epoch、global step 和配置。默认每 5 个 epoch 保存一次，并额外维护验证集最优的 `best.pt`。

## 9. TensorBoard

```bash
tensorboard \
  --logdir /home/hwm/CF_YOPO/YOPO_FlowMatching/runs \
  --bind_all \
  --port 6006
```

每个训练 batch 都记录：

- `train/total_loss`、`flow_loss`、`endpoint_loss`、`trajectory_loss`、`score_loss`；
- 学习率和梯度范数；
- student、teacher 和 refined target 的总成本；
- smooth、safety、guidance、acceleration、depth safety 分项；
- 当前深度图轨迹最小 clearance；
- refined target 相对 YOPO 教师的改进量。

## 10. 离线评估

```bash
python3 offline_eval.py \
  --checkpoint runs/experiment_name/checkpoints/best.pt \
  --output-dir runs/experiment_name/offline_eval
```

输出：

- `OFFLINE_RESULTS.md`：学生和 YOPO 教师的选中代价、oracle 代价、clearance、碰撞代理率、score regret、flow 曲率和推理时间；
- `metrics.json`：完整统计量；
- `visualizations/*.png`：深度图上的学生/教师轨迹投影。

离线低代价不等价于闭环飞行成功，因此必须继续进行 ROS 测试。

## 11. ROS 闭环测试

在容器中打开三个终端。

终端 1，启动飞行器和控制器：

```bash
cd /workspace/YOPO/Controller
source /opt/ros/noetic/setup.bash
source devel/setup.bash
roslaunch so3_quadrotor_simulator simulator_attitude_control.launch
```

终端 2，启动 CUDA 深度仿真器：

```bash
cd /workspace/YOPO/Simulator
source /opt/ros/noetic/setup.bash
source devel/setup.bash
rosrun sensor_simulator sensor_simulator_cuda
```

终端 3，启动 flow policy：

```bash
source /opt/ros/noetic/setup.bash
source /workspace/YOPO/Controller/devel/setup.bash
source /workspace/YOPO/Simulator/devel/setup.bash
cd /home/hwm/CF_YOPO/YOPO_FlowMatching
export PYTHONPATH=.:$PYTHONPATH
python3 ros/test_yopo_flow_ros.py \
  --checkpoint runs/experiment_name/checkpoints/best.pt \
  --goal 50 0 2 \
  --stop-on-arrival \
  --result-md runs/experiment_name/ROS_RESULTS.md
```

节点退出时会记录目标到达状态、到达时间、最终目标距离、路径长度、运行时间、重规划次数、最小观测深度、碰撞代理和推理耗时。当前模拟器没有直接暴露物理碰撞 topic，因此报告中的 collision proxy 需要与飞行轨迹和仿真状态共同判断。

## 12. 重要限制

- 当前数据由独立随机相机位姿组成，不是时序轨迹，因此这里实现的是 lattice-to-trajectory flow，不是原 A2A 的 history-action-to-future-action flow。
- 训练使用完整地图 ESDF，部署只使用深度图和自身状态；两者存在信息差，ROS 闭环是最终判断标准。
- 相同地图同时出现在训练和验证集，验证结果主要衡量图像/状态泛化，不代表新地图泛化。
- 6 步 flow 推理比原 YOPO 单次 head 更慢，应在闭环测试中关注控制频率。
