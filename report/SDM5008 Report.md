## 项目简介

本项目构建于 NVIDIA Isaac Lab 仿真平台，旨在为逐际动力（LimX Dynamics）的 TRON1 双足机器人提供一套高效的强化学习运动控制框架。项目核心贡献包括：

1. **高鲁棒性运动控制**：实现了 TRON1 机器人在平坦地面及非结构化复杂地形下的稳定行走与精准速度跟随能力。
2. **前沿算法复现与集成**：通过 IssacSim 仿真在 TRON1 平台上成功复现并部署了 [**HIM (Hybrid Internal Model)**](https://arxiv.org/abs/2411.14386) [^1]与 [**PIM (Perceptive Internal Model)**](https://arxiv.org/abs/2312.11460) [^2]等先进强化学习算法。
3. **系统性性能评估**：通过多维度的对比实验，详细分析了 HIM、PIM 与基准 **Encoder-MLP** 算法在不同任务场景下的性能差异，为双足机器人的算法选型提供了详尽的数据支持。

项目核心代码基于逐际动力的 [**TRON1 强化学习开源仓库**](https://github.com/limxdynamics/tron1-rl-isaaclab) 完成。

**关键词：**Isaac Lab, TRON1, Bipedal Locomotion, Reinforcement Learning, PPO, HIM, PIM, Robust Control.

---

## 1. 框架理解与架构总览 (Code Review & Architecture)

本项目基于 NVIDIA Isaac Lab 的 `ManagerBasedRLEnv` 构建，采用了高度模块化的配置驱动设计。整个强化学习环境被解耦为四个核心管理器：场景（Scene）、观测（Observation）、奖励（Reward）和动作（Action）。

### 1.1 Scene Configuration (场景配置)

**文件**: `limx_base_env_cfg.py`, `pointfoot_cfg.py`, `terrains_cfg.py`

场景配置模块负责构建物理仿真世界。

- **机器人资产**:

  - 使用 `ArticulationCfg` 加载 USD 文件 (`PF_TRON1A.usd`)。

  - **关节属性**:

    | **关节名称 (Joint Name)** | **阻尼 (Damping)** | **刚度 (Stiffness)** | **初始位置 (Initial Pos)** | **备注**       |
    | ------------------------- | ------------------ | -------------------- | -------------------------- | -------------- |
    | `abad_L_Joint`            | 2.5                | 40.0                 | 0.0                        | 左侧髋外展关节 |
    | `abad_R_Joint`            | 2.5                | 40.0                 | 0.0                        | 右侧髋外展关节 |
    | `hip_L_Joint`             | 2.5                | 40.0                 | 0.0                        | 左侧髋关节     |
    | `hip_R_Joint`             | 2.5                | 40.0                 | 0.0                        | 右侧髋关节     |
    | `knee_L_Joint`            | 2.5                | 40.0                 | 0.0                        | 左侧膝关节     |
    | `knee_R_Joint`            | 2.5                | 40.0                 | 0.0                        | 右侧膝关节     |
    | `foot_L_Joint`            | N/A*               | N/A*                 | 0.0                        | -              |
    | `foot_R_Joint`            | N/A*               | N/A*                 | 0.0                        | -              |

  - **物理属性**:

    | **属性名称 (Property Name)** | **值 (Value)** | **描述**                       |
    | ---------------------------- | -------------- | ------------------------------ |
    | `enabled_self_collisions`    | `True`         | 启用自碰撞检测，防止机器人穿模 |
    | `linear_damping`             | `0.0`          | 线性阻尼系数                   |
    | `angular_damping`            | `0.0`          | 角度阻尼系数                   |
    | `max_linear_velocity`        | `1000.0`       | 最大线速度限制                 |
    | `max_angular_velocity`       | `1000.0`       | 最大角速度限制                 |
    | `solver_position_iter`       | `4`            | 位置求解器迭代次数             |
    | `solver_velocity_iter`       | `4`            | 速度求解器迭代次数             |

- **地形**:

  - 通过 `TerrainImporterCfg` 导入地形。
  - 支持多种地形生成器：`BLIND_ROUGH_TERRAINS_CFG` (波浪、格子、粗糙噪声) 和 `STAIRS_TERRAINS_CFG` (金字塔楼梯)。

- **传感器**:

  - **Contact Sensor**: 定义在 `PFSceneCfg` 中，用于检测足部接触力，采样频率跟随物理仿真步长。
  - **Ray Caster (Height Scanner)**: 仅在 HIM/PIM 配置中启用，用于扫描地形高度，作为 Critic 的特权输入。

### 1.2 Action Manager (动作管理器)

**文件**: `limx_base_env_cfg.py`, `pointfoot_cfg.py`

动作管理器定义了策略网络输出到物理执行器的映射路径。

- **控制模式**: **关节位置控制 (Joint Position Control)**。
- **动作变换**: $q_{target} = q_{default} + \text{scale} \times a_{network}$。其中 `scale=0.25`，将神经网络输出映射到合理的物理角度范围。
- **执行器 (Actuator)**:
  - 使用 `RandomLaggyActuatorCfg` 封装了带有随机延迟的 PD 控制器。
  - **PD 公式**: $\tau = K_p (q_{target} - q) - K_d \dot{q}$
  - **Sim-to-Real 优化**:
    - **参数**: $K_p=40.0, K_d=2.5$。
    - **随机延迟**: 引入 `max_lag=3` (仿真步) 的随机延迟，模拟真实硬件的通信滞后，增强策略鲁棒性。

### 1.3 Observation Manager (观测管理器)

**文件**: `limx_base_env_cfg.py`, `observations.py`

观测管理器构建状态空间，通过 **Actor-Critic 非对称观测** 设计解决部分可观测问题（POMDP）。

| **观测项名称**     | **归属组别**    | **计算逻辑/公式**                                            | **功能与意义**                                               |
| ------------------ | --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **base_ang_vel**   | Policy & Critic | $\mathbf{\omega}_{base} + \mathcal{N}(0, 0.05)$              | 感知机身旋转速率，用于维持姿态平衡。Policy 端注入噪声模拟 IMU 误差。 |
| **proj_gravity**   | Policy & Critic | $\mathbf{R}_{base}^T \cdot \mathbf{g}_{world} + \mathcal{N}(0, 0.025)$ | 重力向量在基座坐标系的投影。是机器人感知自身倾斜角度（Roll/Pitch）的核心依据。 |
| **joint_pos**      | Policy & Critic | $(q - q_{default}) + \mathcal{N}(0, 0.01)$                   | 关节相对位置。感知腿部姿态和伸展程度。                       |
| **joint_vel**      | Policy & Critic | $\dot{q} + \mathcal{N}(0, 0.01)$                             | 关节速度。用于感知运动趋势和阻尼控制。                       |
| **last_action**    | Policy & Critic | $a_{t-1}$                                                    | 上一时刻的动作。提供时序信息，帮助网络推断系统延迟和动力学响应。 |
| **gait_command**   | Policy & Critic | $[f, \phi_{off}, T_{dur}]$                                   | 输入的步态指令（频率、相位偏移、占空比），告知机器人当前应执行何种步态。 |
| **base_lin_vel**   | **Critic Only** | $\mathbf{v}_{base}$ (True State)                             | 基座线速度（无噪声）。帮助 Critic 准确估计价值，Policy 无法直接获取。 |
| **heights**        | **Critic Only** | `RayCaster` 扫描结果                                         | 地形高度图。**HIM/PIM 的核心**，使 Critic 能“看见”楼梯，从而指导盲视 Actor 抬脚。 |
| **contact_forces** | **Critic Only** | $F_{foot}$                                                   | 足端接触力。感知触地状态。                                   |
| **robot_mass**     | **Critic Only** | $m_{robot}$                                                  | 机器人质量。用于隐式系统辨识，适应负载变化。                 |

### 1.4 Reward Manager (奖励管理器)

**文件**: `limx_base_env_cfg.py`, `rewards.py`

奖励函数通过 **Tracking (追踪)**、**Regularization (正则化)** 和 **Gait (步态)** 三类项塑造行为。

| **奖励名称**             | **计算公式 (Code Implementation)**                     | **物理含义与功能**                                           |
| ------------------------ | ------------------------------------------------------ | ------------------------------------------------------------ |
| **rew_lin_vel_xy**       | $e^{-\|v_{xy} - v_{xy}^{cmd}\|^2/\sigma^2}$            | **核心任务**：鼓励机器人精准跟随用户输入的 XY 线速度指令。   |
| **rew_ang_vel_z**        | $e^{-(\omega_z - \omega_z^{cmd})^2 / \sigma^2}$        | **核心任务**：鼓励机器人精准跟随转向（Yaw）指令。            |
| **pen_flat_orientation** | $\|\mathbf{g}_{proj} - [0, 0, -1]\|$ (L2 Norm)         | **姿态约束**：惩罚重力投影偏离 Z 轴，强制躯干保持水平。      |
| **pen_lin_vel_z**        | $v_z^2$ (Squared L2)                                   | **稳定性**：惩罚基座在 Z 轴的运动，抑制跳跃和颠簸。          |
| **pen_joint_accel**      | $\|\ddot{q}\|$                                         | **平滑性**：惩罚关节加速度，减少电机高频震荡和磨损。         |
| **pen_action_rate**      | $\|a_t - a_{t-1}\|$                                    | **平滑性**：惩罚动作的一阶差分，鼓励控制信号连续平滑。       |
| **test_gait_reward**     | $r_{force} + r_{vel}$ (基于相位的混合高斯核)           | **步态塑形**：强制足端在 *Stance* 相触地受力，在 *Swing* 相抬起运动。 |
| **rew_feet_clearance**   | $\sum (h_{foot} - h_{target})^2 \cdot v_{xy}$ (摆动相) | **越障能力**：在摆动相奖励足端抬高到指定高度，防止踢到台阶边缘。 |
| **foot_landing_vel**     | $\sum v_{z, impact}^2$ (仅在即将触地时)                | **柔顺性**：惩罚触地瞬间的 Z 轴速度，鼓励轻柔着陆，减少冲击。 |

------

## 2. 算法对比分析：Encoder-MLP vs HIM vs PIM

本部分详细对比了代码中实现的三种不同训练配置。它们主要在**观测空间结构**和**奖励函数权重**上有所不同，分别对应不同的研究阶段或方法论。

### 2.1 算法变体定义

1. **Encoder-MLP**: 基础盲视策略，通过本体感知信息估计机器人基座线速度。机器人没有外部感知能力，仅靠本体感觉（IMU、关节）行走。

   <img src="images/SDM5008 Report/image-20260105171415652.png" alt="image-20260105171415652" style="zoom:25%;" />

2. **HIM (Hybrid Internal Model)** : HIM 训练一个基于本体感觉历史的**内部模型 (Internal Model/Estimator)**，通过监督学习显式地**预测**隐式特权信息，使得 Policy 不仅仅是被 Critic “指导”，而是自身具备了从本体感觉中**推理**环境和自身状态的能力。

   <img src="images/SDM5008 Report/image-20260105170815865.png" alt="image-20260105170815865" style="zoom:70%;" />

3. **PIM (Perceptive Internal Model)**: 在 HIM 的架构上引入了视觉（高程图）编码器，构建了一个**多模态的内部模型**。与简单的叠加输入不同，PIM 利用视觉信息来**修正和增强**对环境状态的估计（即构建包含几何信息的内部表征），从而让机器人能够**主动规划落点**以应对盲视无法处理的**剧烈地形变化**（如陡峭楼梯或断崖）。

   <img src="images/SDM5008 Report/image-20260105170901472.png" alt="image-20260105170901472" style="zoom:58%;" />

### 2.2 观测空间与算法架构对比

| **特性**        | **Base (Blind Flat/Rough)**               | **HIM (Blind Stairs)**                                    | **PIM (Blind Stairs)**                               |
| --------------- | ----------------------------------------- | --------------------------------------------------------- | ---------------------------------------------------- |
| **Policy 输入** | **纯本体感觉** (关节位置/速度, IMU, 指令) | **纯本体感觉** (同 Base)                                  | **纯本体感觉** (同 Base)                             |
| **Critic 输入** | 本体感觉 + 特权物理信息 (摩擦力, 质量等)  | 本体感觉 + 特权物理信息 + **地形高度扫描 (Heights)**      | 本体感觉 + 特权物理信息 + **地形高度扫描 (Heights)** |
| **地形感知**    | 无 (`height_scanner = None`)              | **开启** (仅 Critic 可见)                                 | **开启** (仅 Critic/Perceptive 可见)                 |
| **感知配置**    | N/A                                       | `observations.critic.heights`                             | `observations.perceptive.heights`                    |
| **核心逻辑**    | 学习基础运动，无法应对突变地形            | 利用特权高度信息辅助 Critic 估值，训练盲视 Actor 应对楼梯 | 类似的架构，可能配合特定的 Encoder 或 Estimator 模块 |

### 2.3 奖励函数深度对比 (Reward Shaping Analysis)

下表总结了三种配置在 `limx_pointfoot_env_cfg.py` 中的具体权重差异。HIM 和 PIM 的配置主要通过 `__post_init__` 方法覆盖 Base 的默认值。

| **奖励项 (Reward Term)** | **功能描述**           | **Base (Default)**     | **HIM Config**                   | **PIM Config**                   | **分析与解读**                                               |
| ------------------------ | ---------------------- | ---------------------- | -------------------------------- | -------------------------------- | ------------------------------------------------------------ |
| **rew_lin_vel_xy**       | XY线速度追踪           | 3.0 (std $\sqrt{0.2}$) | **1.0** (std 0.25)               | **1.0** (std 0.25)               | HIM/PIM 降低了速度追踪的绝对权重，避免过拟合速度而忽略地形稳定性。 |
| **rew_ang_vel_z**        | Z角速度追踪            | 1.5                    | **0.5**                          | **0.5**                          | 同上，降低转向权值。                                         |
| **pen_lin_vel_z**        | Z轴速度惩罚 (跳跃抑制) | -0.5                   | **-2.0**                         | **-2.0**                         | HIM/PIM 大幅增加了对机身垂直晃动的惩罚，要求在楼梯上行走更加平稳。 |
| **pen_flat_orientation** | 姿态惩罚 (保持水平)    | -10.0                  | **-2.0**                         | **-0.2**                         | **关键差异**：Base 强制水平；HIM 允许少量倾斜以适应坡度；**PIM 极大放宽了姿态约束**，允许机器人大幅度俯仰以攀爬更难的地形。 |
| **pen_base_height**      | 基座高度维持           | -20.0                  | **-1.0**                         | **-1.0**                         | 在崎岖地形（楼梯）上，绝对高度难以保持，因此大幅降低了此惩罚权重。 |
| **rew_feet_clearance**   | 足部抬高奖励           | N/A (Default 0)        | **0.2**                          | **0.5**                          | **新增项**：HIM/PIM 必须奖励抬脚（Clearance），否则会被楼梯绊倒。PIM 比 HIM 更鼓励高抬腿。 |
| **test_gait_reward**     | 步态约束奖励           | 1.0                    | **0.5**                          | **0.4** (或移除)                 | 在复杂地形上，严格的强制步态可能适得其反，因此降低了步态约束的权重。 |
| **pen_feet_distance**    | 双脚距离惩罚           | -10.0                  | **-40.0**                        | **-40.0**                        | 大幅增加惩罚，防止在楼梯上双脚打架或劈叉。                   |
| **移除的项**             | 精简奖励函数           | N/A                    | `feet_regulation`, `landing_vel` | `feet_regulation`, `landing_vel` | HIM/PIM 移除了针对平地优化的着陆速度和足部调节规则，依靠物理接触自然演化。 |

---

## 3. 平地速度跟随 (Flat Ground Velocity Tracking)

### 3.1 实验设置与算法配置 (Experimental Setup)

本实验采用基础的 **Encoder-MLP** 算法在盲视下的策略，旨在验证 TRON1 机器人在平坦地面上的全向移动能力与姿态稳定性。

| **配置项 (Configuration)** | **参数设定 (Settings)**                                      | **说明 (Description)**                                       |
| -------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **算法架构**               | Base Blind (Encoder-MLP)                                     | 仅使用本体感觉 (Proprioception)，无视觉输入                  |
| **地形环境**               | Flat Plane                                                   | 无障碍物的无限延伸平坦地面                                   |
| **观测空间**               | $\mathbf{o}_t \in \mathbb{R}^{48}$                           | 包含 $\omega_{base}, \mathbf{g}_{proj}, q, \dot{q}, a_{t-1}$ 及步态指令 |
| **指令范围**               | $v_x \in [-1.5, 1.5]$ m/s $v_y \in [-1.0, 1.0]$ m/s $\omega_z \in [-0.5, 0.5]$ rad/s | 随机指令每 5 秒重采样一次                                    |

**核心奖励函数配置 (Reward Function Setup):**

根据 `limx_base_env_cfg.py`，关键奖励项权重如下，高权重的速度追踪与严格的姿态惩罚是本实验成功的关键：
$$
r_{total} = w_{lin} \cdot r_{lin\_vel} + w_{ang} \cdot r_{ang\_vel} + w_{orient} \cdot r_{orientation} + \dots
$$

| **奖励项 (Reward Term)** | **权重 (Weight)** | **核心公式 (Formula)**                        | **功能**               |
| ------------------------ | ----------------- | --------------------------------------------- | ---------------------- |
| `rew_lin_vel_xy`         | **3.0**           | $\exp(-\|v_{xy} - v_{xy}^{cmd}\|^2 / 0.25)$   | 精准追踪平面线速度     |
| `rew_ang_vel_z`          | **1.5**           | $\exp(-(\omega_z - \omega_z^{cmd})^2 / 0.25)$ | 精准追踪转向角速度     |
| `pen_flat_orientation`   | **-10.0**         | $\|\mathbf{g}_{proj} - [0, 0, -1]\|$          | **强约束**基座保持水平 |
| `pen_base_height`        | **-20.0**         | $(h_{base} - 0.68)^2$                         | 锁定基座高度在 0.68m   |

### 3.2 考核指标定义 (Evaluation Metrics)

为量化评估模型性能，我们定义以下指标：

**1. 速度追踪误差 (Velocity Tracking Error, MSE):**
$$
MSE_{v} = \frac{1}{T} \sum_{t=1}^{T} \| \mathbf{v}_t^{meas} - \mathbf{v}_t^{cmd} \|^2
$$
**2. 姿态稳定性 (Attitude Stability):**

通过 Roll 和 Pitch 角的极差震荡幅度：
$$
R_{\phi} = \phi_{\max}-\phi_{\min}, \quad R_{\theta} = \theta_{\max}-\theta_{\min}
$$
**3. 存活率 (Survival Rate):**
$$
R_{survival} = \frac{N_{success}}{N_{total}} \times 100\%
$$
*(其中 $N_{success}$ 为未发生摔倒或基座触地的 Episode 数量)*

### 3.3 实验结果展示 (Experimental Results)

#### 3.3.1 速度响应曲线 (Velocity Response)

**图 1: 线速度 ($v_x, v_y$) 与角速度 ($\omega_z$) 追踪性能**

<img src="images/SDM5008 Report/flat_1_velocity_tracking.png" alt="flat_1_velocity_tracking" style="zoom:23%;" />

**图 2: 线速度 ($v_x, v_y$) 与角速度 ($\omega_z$) 跟踪误差分布**

<img src="images/SDM5008 Report/flat_2_error_distribution.png" alt="flat_2_error_distribution" style="zoom:25%;" />

**数据分析:**

**速度跟踪性能 (Velocity Tracking)**

从时域波形图来看，控制器对**阶跃信号（Step Input）**展现了良好的动态响应能力：

- **响应速度**：$v_x$（纵向）、$v_y$（横向）和 $\omega_z$（转向）均能迅速响应指令变化，上升/下降沿陡峭，无显著的延迟或超调。
- **稳定性**：在指令保持阶段，机器人实际速度围绕期望值波动，但整体均值稳定，未出现发散或明显的稳态误差漂移。
- **耦合影响**：$v_x$ 的大幅突变未对 $v_y$ 和 $\omega_z$ 造成显著的干扰，说明各自由度间的解耦控制较为理想。

**误差统计分布 (Error Distribution)**

误差分布图进一步量化了跟踪精度，三轴误差均呈现标准的**高斯分布（正态分布）**特性：

- **无系统偏差 (Unbiased)**：三个维度的误差均值 ($\mu$) 极低（$v_x: 0.028, v_y: 0.035, \omega_z: -0.034$），几乎为零，说明模型不存在系统性的“跑偏”问题。
- **精度分析**：
  - **$v_y$ (横向)** 表现最佳，具有最小的均方误差 (MSE: 6.4e-02) 和标准差 ($\sigma=0.250$)，说明横向控制最为收敛。
  - **$v_x$ (纵向)** 的波动范围稍大 ($\sigma=0.392$)，MSE (1.5e-01) 最高，这通常是因为纵向运动幅度大且受腿部摆动对地冲击影响最直接。
  - **$\omega_z$ (转向)** 标准差 ($\sigma=0.315$) 居中，控制较为平稳。

该双足机器人的运动控制策略表现出**“响应快、精度高、无偏置”**的特性。尽管存在典型的高频震荡噪声，但误差被有效限制在合理的高斯分布范围内，能够精确地执行复杂的速度切换指令。

#### 3.3.2 姿态稳定性分析 (Attitude Stability)

**图 3: 基座姿态角 (Roll & Pitch) 随时间变化**

<img src="images/SDM5008 Report/flat_3_oscillation.png" alt="flat_3_oscillation" style="zoom:23%;" />

**数据分析：**

**振荡幅度与有界性 (Amplitude & Boundedness)**

整体来看，机器人表现出了**“动态稳定但伴随高频抖动”**的特性。虽然存在显著的姿态波动，但并未发生发散（倒地），表明控制策略具有鲁棒性。

- **Pitch (俯仰角)**：波动范围较大（Range: 0.431 rad，约 $24.7^\circ$）。这是由急加减速引起的惯性效应。曲线中明显的尖峰（如 $t=50s$ 附近）对应了**速度指令的大幅切换**，说明机器人在应对纵向冲击时会产生较大的前后晃动。
- **Roll (横滚角)**：波动范围相对较小（Range: 0.354 rad，约 $20.3^\circ$）。考虑到双足机器人的横向平衡通常较难维持，这一波动幅度表明机器人在高速运动中左右摇摆较为剧烈，呈现出类似“踏步调整”的策略来维持平衡。

**稳态偏置 (Steady-State Bias)**

观察零基准线（Zero Ref），两个自由度均存在微小的非零均值偏置：

- **Roll**：呈现持续的负值偏置（均值 $< 0$），这意味着机器人的躯干在运动过程中长期向一侧（左或右，取决于坐标系定义）轻微倾斜。
- **Pitch**：主体位于正值区间（均值 $> 0$），表明机器人在运动时保持着轻微的**“前倾”姿态**。这对于高速行走/奔跑是合理的，有助于质心前移以辅助加速。

**收敛能力 (Convergence)**

- **快速恢复**：尽管在 $t=5s, 37s, 50s$ 等时刻出现了大幅度的姿态突变（对应速度阶跃），曲线总是能迅速回调至均值附近。这种**强回复力**说明RL策略学习到了有效的姿态恢复机制，能够抵抗剧烈的加减速扰动。

该双足机器人在执行速度跟踪任务时，**牺牲了一定的姿态平滑度以换取高动态响应能力**。面对指令速度的大幅切换，能够快速调整自身姿态，执行速度跟随任务。

### 3.4 结论 (Conclusion)

**数据统计:**

| **指标 (Metric)**                   | **平均误差/标准差 (Mean/Std)** |
| ----------------------------------- | ------------------------------ |
| **线速度追踪 MSE** ($v_x$)          | **0.15** $(m/s)^2$             |
| **线速度追踪 MSE** ($v_y$)          | **0.064** $(m/s)^2$            |
| **角速度追踪 MSE** ($\omega_z$)     | **0.1** $(rad/s)^2$            |
| **Roll 震荡幅度** ($R_\phi$)        | **0.354** rad                  |
| **Pitch 震荡幅度** ($R_\theta$)     | **0.431** rad                  |
| **存活率** (about 1 min continuous) | **100%**                       |

实验表明，基于 **PF Base Blind Flat** 配置的策略网络：

1. **响应精准**：在 $3.0$ 的高奖励权重驱动下，实现了低延迟的速度指令响应。机器人能够迅速调整自身姿态跟随指令速度。
2. **姿态平稳**：`-10.0` 的姿态惩罚项有效地抑制了机身晃动，使 Roll/Pitch 大部分时间维持在 $\pm 0.05$ rad 安全范围内。
3. **鲁棒性**：在持续 1 分钟的随机变速测试中，机器人未发生摔倒，验证了该盲视基准算法在平坦地形上的可靠性。



## 参考文献 (References)

[^1]:Long, J., Ren, J., Shi, M., Wang, Z., Huang, T., Luo, P., & Pang, J. (2024). **Learning Humanoid Locomotion with Perceptive Internal Model**. *arXiv preprint arXiv:2411.14386*. https://arxiv.org/abs/2411.14386
[^2]:Long, J., Wang, Z., Li, Q., Gao, J., Cao, L., & Pang, J. (2024). **Hybrid Internal Model: Learning Agile Legged Locomotion with Simulated Robot Response**. *The Twelfth International Conference on Learning Representations (ICLR)*. https://arxiv.org/abs/2312.11460