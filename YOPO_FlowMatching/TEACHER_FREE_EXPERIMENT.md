# Teacher-Free Physical-Anchor LatticeFlow 实验记录

## 结论

用户提出的方案是可行的：以 15 个 lattice 物理端点作为确定性 `x0`，不使用教师模型或预训练 backbone，通过学生自身 detached 输出、局部随机搜索和 privileged trajectory cost 构造 `x1`，可以从随机初始化训练出有效避障策略。

当前证据支持“无教师方案保留了绝大多数相对 YOPO 的质量与安全增益”，但不支持“无教师全面优于教师引导模型”。教师引导 physical flow 的 selected cost 仍低 0.0193；无教师模型的 collision proxy 则低 0.41 个百分点。闭环中 continuity-aware selector 是必要组成部分。

## 实现边界

- 部署输入：单帧深度、目标方向、速度、加速度；
- 训练 privileged 信息：pose、点云/ESDF、深度投影轨迹安全代价；
- YOPO 教师：训练期间不实例化、不加载；
- backbone：ResNet-18 结构随机初始化，不加载 YOPO 或 ImageNet 权重；
- Flow source：15 个不同的物理终端位置，末端速度和加速度为零；
- 初始行为：Flow 最后一层置零，初始积分严格返回 anchor；
- target search：anchor、detached student、2 个 anchor noise、1 个 student noise；
- refinement：2 步、逐 cell 梯度归一化、只有未压缩代价下降才接受；
- 输出：YOPO 9 维 residual、trajectory score、原五次多项式与 ROS 控制接口。

## 训练

- 配置：`configs/icra2027_teacher_free_physical.yaml`；
- seed：0；
- maps：train 0--6、validation 7、test 8--9；
- epochs：20；batch size：16；AdamW；learning rate：`1.5e-4`；
- NFE：6；
- best validation cost：`2.8543775381`；
- checkpoint：`runs/icra2027_teacher_free_physical_seed0/checkpoints/best.pt`；
- TensorBoard：`runs/icra2027_teacher_free_physical_seed0/tensorboard`。

训练后期抽样日志中 detached student 已成为主要 target seed，且代价梯度更新仍保持较高接受率。这表明训练从初始 anchor 阶段转入了学生自改进阶段，而不是一直拟合固定 anchor。

## Held-out 离线结果

测试集包含 maps 8--9 的全部 20,000 个固定查询。所有方法使用相同输入和相同扰动。

| 方法 | Selected cost ↓ | Oracle ↓ | Clearance ↑ | Collision proxy ↓ | Switch ↓ | Shift ↓ | Regret ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOPO | 3.141475 | 2.794607 | 1.989431 | 0.284800 | 0.045100 | 0.116084 | 0.346867 |
| Residual + teacher | 2.862222 | 2.756476 | 3.143329 | 0.181450 | 0.033800 | 0.101132 | 0.105746 |
| Physical + teacher | 2.861148 | 2.762867 | 3.038480 | 0.189000 | 0.032200 | 0.094083 | 0.098281 |
| Physical teacher-free | 2.880468 | 2.773581 | 3.042132 | 0.184900 | 0.033800 | 0.106659 | 0.106887 |

配对差值（无教师减 physical + teacher）：

- selected cost：`+0.019319`，描述性 95% CI `[0.015560, 0.023105]`；
- oracle cost：`+0.010714`，`[0.008620, 0.012750]`；
- clearance：`+0.003652 m`，区间跨零；
- collision proxy：`-0.004100`，`[-0.007600, -0.000750]`；
- endpoint shift：`+0.012576 m`，`[0.003785, 0.021417]`；
- selection regret：`+0.008606`。

这些 query-level bootstrap 区间只作描述，因为样本嵌套在两张地图内；环境层独立样本量只有 2，不能将其解释为跨环境显著性检验。

## NFE 与时延

| NFE | Cost ↓ | Collision proxy ↓ | ms / batch of 16 ↓ |
|---:|---:|---:|---:|
| 1 | 2.951922 | 0.185900 | 1.539667 |
| 2 | 2.891635 | **0.182800** | 1.645472 |
| 4 | 2.882329 | 0.184150 | 2.013537 |
| 6 | **2.880468** | 0.184900 | 2.615915 |
| 8 | 2.880531 | 0.185200 | 2.645954 |

四步相对六步 cost 仅高约 0.065%，实际部署可以根据算力选择 NFE 2--4；论文主结果仍使用预先固定的 NFE 6。

## ROS 闭环结果

同一随机森林 seed 3、起点 `[2,-2,2]`、五个固定目标，每个完整 rollout 为一个独立观测。

| 模式 | 成功 | 碰撞 | 时间 | Clearance | Switch | Jump | Jerk | Net |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Teacher-free raw | 5/5 | 0/5 | 8.812 s | 0.952 m | 0.145 | 0.271 m | 6.45 m/s³ | 3.26 ms |
| Teacher-free selector | 5/5 | 0/5 | 9.620 s | 0.989 m | 0.024 | 0.164 m | 5.56 m/s³ | 3.24 ms |

selector 相对 YOPO raw 将 switch 降低 83.5%、jump 降低 31.3%、jerk 降低 9.0%。但是 residual + teacher selector 的 jump `0.152 m` 和 jerk `4.72 m/s³` 更低，因此当前无教师模型不是所有闭环平滑指标最优。

## 统计与论文表述边界

- 训练 seed 只有 1 个；
- held-out 环境只有 2 张地图；
- ROS 只有 1 张地图、每配置 5 个目标；
- 没有动态障碍实验；
- 没有 matched direct-regression control；
- 真机结果尚未填写。

因此论文使用“可行、保留主要增益、教师非必要但仍有小幅拟合优势”等表述，不使用“普遍显著优于”“已证明真实环境鲁棒”等结论。

## 结果文件

- 单模型离线：`runs/icra2027_teacher_free_physical_seed0/offline_test/OFFLINE_RESULTS.md`；
- 四模型配对：`runs/icra2027_teacher_free_physical_seed0/teacher_free_comparison/COMPARISON.md`；
- NFE：`runs/icra2027_teacher_free_physical_seed0/offline_test/nfe.json`；
- 时延：`runs/icra2027_teacher_free_physical_seed0/offline_test/latency.json`；
- ROS：`runs/icra_ros_teacher_free/ROS_SUMMARY.md`；
- 论文 PDF：`../LatticeFlow_ICRA2027/paper/main.pdf`。
