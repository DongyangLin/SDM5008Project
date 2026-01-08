# **SDM5008 课程期末项目报告**

---

**课程名称：** SDM5008 – 高级机器人控制  
**项目标题：** 基于强化学习的点足双足机器人运动控制（Isaac Lab 平台）  
**实验平台：** NVIDIA Isaac Lab（Isaac Sim 4.5.0 + Isaac Lab 2.1.0）

**组员信息：**  
*   **姓名：** `Zikun Zhuang` `Zhiyu Wang`  
*   **学号：** `12532840` `12532838`  
*   **邮箱：** `[12532840@mail.sustech.edu.cn]` `[12532838@mail.sustech.edu.cn]`

**提交日期：** `[2026 年 1 月 8 日]`

---

## 项目简介

本项目构建于 NVIDIA Isaac Lab 仿真平台，旨在为逐际动力（LimX Dynamics）的 TRON1 双足机器人提供一套高效的强化学习运动控制框架。项目核心贡献包括：

1. **高鲁棒性运动控制**：实现了 TRON1 机器人在平坦地面及非结构化复杂地形下的稳定行走与精准速度跟随能力。
2. **前沿算法复现与集成**：通过 IssacSim 仿真在 TRON1 平台上成功复现并部署了 [**HIM (Hybrid Internal Model)**](https://arxiv.org/abs/2411.14386) [^1]与 [**PIM (Perceptive Internal Model)**](https://arxiv.org/abs/2312.11460) [^2]等先进强化学习算法。
3. **系统性性能评估**：通过多维度的对比实验，详细分析了 HIM、PIM 与基准 **Encoder-MLP** 算法在不同任务场景下的性能差异，为双足机器人的算法选型提供了详尽的数据支持。

项目核心代码基于逐际动力的 [**TRON1 强化学习开源仓库**](https://github.com/limxdynamics/tron1-rl-isaaclab) 完成。

**关键词:** Isaac Lab, TRON1, Bipedal Locomotion, Reinforcement Learning, PPO, HIM, PIM, Robust Control.

---

## 1. 框架理解与架构总览

本项目基于 NVIDIA Isaac Lab 的 `ManagerBasedRLEnv` 构建，采用了高度模块化的配置驱动设计。整个强化学习环境被解耦为四个核心管理器：场景（Scene）、观测（Observation）、奖励（Reward）和动作（Action）。

### 1.1 场景配置

**相关文件**：`limx_base_env_cfg.py`, `pointfoot_cfg.py`, `terrains_cfg.py`

将机器人资产、地形、光照与传感器等物理实体组织为一个可交互的仿真场景，并为后续的观测、奖励与控制模块提供统一的物理基础。本项目中，场景配置主要通过 `PFSceneCfg` 类实现。

#### 1.1.1 机器人 USD 资产配置

机器人本体的物理与几何属性在 `exts/bipedal_locomotion/bipedal_locomotion/assets/config/pointfoot_cfg.py` 中定义，对应配置对象为 `POINTFOOT_CFG`。

- **USD 资产加载**：代码通过 `sim_utils.UsdFileCfg` 加载位于 `../usd/PF_TRON1A/PF_TRON1A.usd` 文件。USD 文件作为统一的几何、关节拓扑与碰撞体描述，是 Isaac Sim / Isaac Lab 中机器人几何与物理模型的基础。
- **刚体与物理求解属性**：通过 `RigidBodyPropertiesCfg` 显式配置刚体属性和物理仿真参数，如启用自碰撞（`enabled_self_collisions=True`），防止双足运动时发生穿模；设置求解器迭代次数，提高高频接触下的数值稳定性；限制最大线速度与角速度，避免仿真数值发散等。
- **初始状态**：`init_state` 定义了机器人出生时的基座初始位姿和各关节的默认角度（`joint_pos`）。该初始化状态作为环境 reset 时的标准起点，确保不同 episode 之间具有一致的初始分布。

#### 1.1.2 交互式场景装配

在 `limx_base_env_cfg.py` 中，`PFSceneCfg` 类继承自 `InteractiveSceneCfg`，负责将机器人资产与环境要素集成到统一的仿真场景中。其结构清晰地体现了 `SceneCfg` 的三类核心组成：

- **地形配置**：`terrain = TerrainImporterCfg(...)`
  
  - 配置可采用平面地形（plane）；也支持多种地形生成器，如 `BLIND_ROUGH_TERRAINS_CFG`（波浪、格子、粗糙）和 `STAIRS_TERRAINS_CFG`（金字塔楼梯）等。
  - 通过 `RigidBodyMaterialCfg` 定义地面物理材质。

- **光照配置**：`light = AssetBaseCfg(...)`
  
  - 使用 `DomeLightCfg` 创建全局天空光。光照也仅影响可视化，不影响物理仿真与学习过程。

- **机器人实例化**：`robot：ArticulationCfg = MISSING`
  
  - `PFSceneCfg` 并不直接绑定具体机器人配置，具体的 `ArticulationCfg`（如 `POINTFOOT_CFG`）在子类或环境配置中注入，让 `SceneCfg` 与具体机器人解耦，提高了代码复用性。
  - 机器人关节的 PD 参数（刚度、阻尼、初始角度）在 `ArticulationCfg` 中统一管理：
    - **关节属性**
      
      | **关节名称** | **阻尼** | **刚度** | **初始位置** | **备注** |
      | -------------- | ---- | ---- | --- | ------------- |
      | `abad_L_Joint` | 2.5  | 40.0 | 0.0 | 左侧髋外展关节 |
      | `abad_R_Joint` | 2.5  | 40.0 | 0.0 | 右侧髋外展关节 |
      | `hip_L_Joint`  | 2.5  | 40.0 | 0.0 | 左侧髋关节     |
      | `hip_R_Joint`  | 2.5  | 40.0 | 0.0 | 右侧髋关节     |
      | `knee_L_Joint` | 2.5  | 40.0 | 0.0 | 左侧膝关节     |
      | `knee_R_Joint` | 2.5  | 40.0 | 0.0 | 右侧膝关节     |
      | `foot_L_Joint` | N/A* | N/A* | 0.0 | -             |
      | `foot_R_Joint` | N/A* | N/A* | 0.0 | -             |
      
    - **物理属性**
      | **属性名称** | **值 ** | **描述** |
      | ------------------------- | -------- | --------------------------- |
      | `enabled_self_collisions` | True   | 启用自碰撞检测，防止机器人穿模 |
      | `linear_damping`          | 0.0    | 线性阻尼系数                 |
      | `angular_damping`         | 0.0    | 角度阻尼系数                 |
      | `max_linear_velocity`     | 1000.0 | 最大线速度限制               |
      | `max_angular_velocity`    | 1000.0 | 最大角速度限制               |
      | `solver_position_iter`    | 4      | 位置求解器迭代次数           |
      | `solver_velocity_iter`    | 4      | 速度求解器迭代次数           |

#### 1.1.3 传感器配置

场景配置项还负责声明环境级传感器，供观测管理器使用。

- **接触力传感器**：`contact_forces = ContactSensorCfg(...)`
  用于检测足部接触力与空中时间，采样频率跟随物理仿真步长，是步态奖励设计（如接触时序）足部状态观测的重要数据来源。

- **高度扫描传感器**：`height_scanner：RayCasterCfg = MISSING`
  在 `PFSceneCfg` 中仅声明接口，在 HIM / PIM 等子类配置中具体实例化。在 HIM 中作为 `Critic` 的特权信息；在 PIM 中作为 `Perceptive_Observation` 感知观测输入。

### 1.2 动作管理器

**相关文件**：`limx_base_env_cfg.py`, `pointfoot_cfg.py`

动作管理器定义了策略输出与仿真器执行器之间的接口，将策略网络输出的抽象动作转换为物理可执行的控制指令，并通过执行器作用于机器人关节。

#### 1.2.1 动作空间定义

动作空间在 `ActionsCfg` 中通过 `JointPositionActionCfg` 定义。

- **控制模式**：**关节位置控制**。
    策略网络直接输出关节位置的无量纲动作向量，不直接输出力矩或速度。

- **受控关节**：6 个主动关节。
    动作仅作用于 2 个髋外展、2 个髋关节与 2 个膝关节。足端关节不参与主动控制，仅作为被动接触刚体。

#### 1.2.2 动作语义映射

策略输出并不直接作为绝对关节角，而是通过偏移形式映射为目标位置：
$$
q_{target} = q_{default} + \text{scale} \times a_{network}
$$
其中， $q_{\text{default}}$ 是在 `ArticulationCfg` 中定义的默认关节角；$a_{\text{policy}}$ 是策略网络输出；动作缩放因子 $\alpha = 0.25$ 将神经网络输出映射到合理的物理角度范围。
这样可以限制单步动作幅度，防止策略在早期训练中产生不合理的大角度跃迁，使学习过程围绕一个物理可行的名义姿态展开，并提高训练收敛速度和仿真稳定性。

#### 1.2.3 执行器模型

- **执行器类型**:
  - 实际的关节驱动由 `pointfoot_cfg.py` 中的 `actuators` 配置完成，使用 `RandomLaggyActuatorCfg`封装了带有随机延迟的 PD 控制器。
  - 属于隐式执行器：策略不直接感知力矩，力矩由物理引擎根据 PD 误差自动计算。

- **PD 控制律**：
  - 在每个仿真步中，根据配置中的刚度（$K_p$）和阻尼（$K_d$）参数，物理引擎计算最终施加的力矩 $\tau$ 为：
  $$
  \tau = K_p (q_{target} - q_{current}) - K_d \dot{q}_{current}
  $$
  该 PD 参数在场景配置项中统一定义。

- **Sim-to-Real 随机延迟**：
  
  - `RandomLaggyActuatorCfg` 在标准 PD 控制的基础上，执行器引入了 `max_lag=3` (仿真步) 的随机延迟，模拟真实硬件的通信滞后，增强策略鲁棒性。

### 1.3 观测管理器

**相关文件**：`limx_base_env_cfg.py`, `observations.py`

观测管理器负责从仿真世界中提取、加工并组织观测信息，构建适用于强化学习的状态空间，同时通过噪声与信息不对称模拟真实世界感知条件。
本项目采用了分组观测 + Actor–Critic 非对称观测的设计，以应对部分可观测马尔可夫决策过程和 sim-to-real 差距。

观测配置由 `ObservationsCfg` 统一管理，并划分为多个 `ObsGroup`：
| **观测组** | **使用对象** | **主要作用** |
| --- | --- | --- |
| `PolicyCfg`	| Actor | 提供可现实获取的、带噪声的感知信息。 |
| `CriticCfg`	| Critic | 提供无噪声、含特权信息的完整系统状态信息。 |
| `HistoryObsCfg`	| Actor	| 记录历史感知信息，供策略网络使用。 |
| `CommandsObsCfg`	| Actor / Critic | 注入高层任务或速度指令。 |

明确区分了 “策略在真实系统中能看到什么” 与 “训练时价值函数可以利用什么”。

#### 1.3.1 策略观测组

`PolicyCfg` 定义了输入给 Actor 网络的观测向量，模拟真实机器人可获取的传感器信息。

- **观测内容组成**
  这些观测共同构成了闭环控制所需的最小充分信息集。
  - 基座状态（IMU 类信息）
    - `base_ang_vel`：基座角速度
    - `proj_gravity`：重力向量在基座坐标系下的投影（姿态感知核心）
  - 关节状态（编码器信息）
    - `joint_pos`：相对默认姿态的关节位置
    - `joint_vel`：关节速度
  - 步态与历史动作
    - `last_action`：上一时刻动作
    - `gait_phase`：当前步态相位
    - `gait_command`：步态指令（频率、相位偏移、占空比）

- **噪声注入与数值处理流程**
  为了缩小 Sim-to-Real 的差距，Policy 观测显式引入了噪声注入和归一化：
  * **噪声注入**：使用 `GaussianNoise` 为观测添加高斯白噪声。例如，`base_ang_vel`（基座角速度）添加了均值为 0、标准差为 0.05 的噪声。
  * **处理流程**：原始物理数据 $\rightarrow$ 添加噪声 $\rightarrow$ 裁剪 $\rightarrow$ 缩放$\rightarrow$ 神经网络输入。

#### 1.3.2 历史观测组

`HistoryObsCfg` 在 `PolicyCfg` 的基础上引入多步历史观测，记录了`history_length = 10` 仿真步的 `PolicyCfg` 数据。为策略提供隐式系统状态（如执行器延迟、接触滞后），弥补单步观测无法完全描述系统动力学的问题。在 HIM / PIM 算法中有重要应用。

#### 1.3.4 价值观测组

`CriticCfg` 定义了输入给价值网络的观测，是一个仅在训练阶段可用的特权空间。

- **观测内容组成**
  
  - **无噪声真实状态**：不添加噪声的 Ground Truth 信息（`enable_corruption=False`），有助于稳定价值函数学习。
    `base_lin_vel`、`base_ang_vel`、`joint_pos`、`joint_vel`、`robot_pos`、`robot_vel`、`robot_base_pose`

  - **环境与系统特权信息**：策略网络无法获取的额外信息。
    - 地形感知 
      地形高度扫描 `heights` 由 `RayCaster` 生成，是 HIM / PIM 架构的关键，使 Critic “看见”台阶和障碍。
    - 接触与动力学信息
      `robot_feet_contact_force`、`robot_joint_torque`、`robot_joint_acc`
    - 系统参数
      `robot_mass`、`robot_inertia`、`robot_joint_stiffness`、`robot_joint_damping`、材质属性（摩擦系数等）

- **意义与作用**
  - 帮助 Critic 准确估计价值函数；
  - 通过优势函数间接指导 Actor；
  - 实现 “盲视 Actor + 全知 Critic” 的训练范式。

#### 1.3.5 命令观测组

`CommandsObsCfg` 用于注入高层任务指令。
- `velocity_commands`：目标线速度 / 角速度
  该观测使策略学习成为条件策略，支持多任务或多速度目标训练。

重要观测项列表如下：

| **观测项名称** | **可见观测组** | **计算逻辑/公式** | **功能与意义** |
| ------------- | ----------- | ---------------- | -------------- |
| **base_ang_vel**   | Policy & Critic | $\mathbf{\omega}_{base} + \mathcal{N}(0, 0.05)$ | 感知机身旋转速率，用于维持姿态平衡。Policy 端注入噪声模拟 IMU 误差。 |
| **proj_gravity**   | Policy & Critic | $\mathbf{R}_{base}^T \cdot \mathbf{g}_{world} + \mathcal{N}(0, 0.025)$ | 重力向量在基座坐标系的投影。是机器人感知自身倾斜角度（Roll/Pitch）的核心依据。 |
| **joint_pos**      | Policy & Critic | $(q - q_{default}) + \mathcal{N}(0, 0.01)$ | 关节相对位置。感知腿部姿态和伸展程度。                       |
| **joint_vel**      | Policy & Critic | $\dot{q} + \mathcal{N}(0, 0.01)$ | 关节速度。用于感知运动趋势和阻尼控制。                       |
| **last_action**    | Policy & Critic | $a_{t-1}$ | 上一时刻的动作。提供时序信息，帮助网络推断系统延迟和动力学响应。 |
| **gait_command**   | Policy & Critic | $[f, \phi_{off}, T_{dur}]$ | 输入的步态指令（频率、相位偏移、占空比），告知机器人当前应执行何种步态。 |
| **base_lin_vel**   | **Critic** | $\mathbf{v}_{base}$ | 基座线速度（无噪声）。帮助 Critic 准确估计价值，Policy 无法直接获取。 |
| **heights**        | **Critic** | $\text{RayCaster}$ | 地形高度图。HIM / PIM 的核心，使 Critic 能“看见”楼梯，从而指导盲视 Actor 抬脚。 |
| **contact_forces** | **Critic** | $F_{foot}$ | 足端接触力。感知触地状态。|
| **robot_mass**     | **Critic** | $m_{robot}$ | 机器人质量。用于隐式系统辨识，适应负载变化。                 |

### 1.4 奖励管理器

**相关文件**：`limx_base_env_cfg.py`, `rewards.py`

奖励管理器通过 `RewardsCfg` 类定义了强化学习的目标函数结构。奖励函数的设计直接决定了机器人学习到的运动风格、稳定性和能耗特性。本项目的奖励函数采用多项加权求和的形式：
$$
R_t = \sum_i w_i r_i (s_t, a_t)
$$
其中每一项 `RewTerm` 对应一种物理或行为约束，权重 `weight` 决定该约束在训练中的相对重要性。

- **奖励项分类**
  从功能上看，奖励项可以清晰地分为四大类：
  | 奖励类别 | 设计目的 |
  | --- | --- |
  | **存活与基本稳定** | 确保机器人首先“不倒、不炸”。 |
  | **指令追踪** | 学会服从速度与步态指令。 |
  | **物理正则化** | 抑制非物理、不平滑或高能耗行为。 |
  | **步态塑形** | 诱导合理的摆动 / 支撑相行为。 |

- **权重分布**:
  每个奖励项都有一个 `weight` 参数。
  
  * 正权重（如 `keep_balance` 的 `1.0`）表示奖励，促进该行为的发生。
  * 负权重（如 `pen_joint_torque` 的 `-0.00008`）表示惩罚，抑制该行为。
  * 权重的绝对值大小决定了该项在总奖励中的主导地位。从权重分布可以看出明确的训练优先级：
    - 大权重负项：确保稳定与安全。
    - 中等正项：学会服从指令。
    - 小权重正则项：优化运动质量。

这种设计体现了典型的 “从可行 $→$ 稳定 $→$ 高性能” 的训练路径。

#### 1.4.1 存活与稳定性奖励

- **存活奖励**
  `keep_balance = RewTerm(func=mdp.stay_alive, weight=1.0)`
  只要仿真未终止，智能体即可获得正奖励；为训练提供最基本的正反馈，防止策略在早期陷入“自毁式探索”。

- **姿态与高度约束**（强约束项）
  具有极大的负权重，在训练初期起到“硬约束”的作用，明确禁止倒地、塌腰等失败状态。
  - `pen_base_height (weight = -20.0)`
    惩罚基座高度偏离目标值（0.68 m）。
  - `pen_flat_orientation (weight = -10.0)`
    惩罚躯干倾斜，强制身体保持水平。

#### 1.4.2 指令追踪奖励

指令追踪是本任务的核心目标，对应机器人对用户输入速度的响应能力。

- **线速度追踪**
  `rew_lin_vel_xy (weight = 3.0)`
  使用高斯核函数鼓励机器人在 XY 平面内精准跟随期望速度：
  $$
  r_{vel} = \exp\left(-\frac{\|v_{xy} - v_{xy}^{cmd}\|^2}{\sigma^2}\right)
  $$
  使用指数核函数，使小误差区域梯度更平滑。

- **角速度追踪**
  `rew_ang_vel_z (weight = 1.5)`
  使用高斯核函数控制机器人 Yaw 转向行为：
  $$
  r_{ang} = \exp\left(-\frac{\|w_{z} - w_{z}^{cmd}\|^2}{\sigma^2}\right)
  $$
  权重小于线速度，避免频繁急转导致不稳定。

#### 1.4.3 物理正则化与能耗约束

这类奖励项的目标不是“完成任务”，而是约束如何完成任务。

- **运动平滑性**
  减少控制抖动，使策略更适合真实执行器。
  - `pen_action_rate`：抑制动作突变。
  - `pen_action_smoothness`：惩罚二阶动作差分，减少控制器的抖动。
  - `pen_joint_accel`：限制关节高频加速度，鼓励平滑且节能的运动。

- **能耗与机械负载**
  促使策略形成节能、机械友好的运动模式。
  - `pen_joint_torque`：限制力矩幅值
  - `pen_joint_powers`：惩罚功率消耗
  - `pen_joint_vel_l2`：防止关节高速甩动

- **非期望接触与关节限制**
  - `pen_undesired_contacts`：防止非足部（髋、膝、机身）与地面发生接触。
  - `pen_joint_pos_limits`：避免关节越界，保证运动的可执行性。

#### 1.4.4 步态塑形奖励

- **基于相位的步态奖励**
  - `test_gait_reward` 奖励结合接触力与足端速度，并与 `gait_command` 中的相位信息关联。使策略不需要显式建模步态机，也能学习出周期性、可解释的行走模式。
    - 支撑相：奖励稳定接触力；
    - 摆动相：奖励足端抬起与前摆。
  
- **落脚与抬脚细化约束**
  这些奖励用于提升落脚柔顺性与越障鲁棒性。
  - `foot_landing_vel`：惩罚触地瞬间的垂直速度，减少冲击。
  - `pen_feet_distance`：避免双脚过近导致自碰撞。
  - `pen_feet_regulation`：结合基座高度约束足端空间分布。

重要奖励项列表如下：

| **奖励名称** | **计算公式** | **物理含义与功能** |
| ----------- | ------------------------ | ----------------------- |
| **rew_lin_vel_xy**       | $e^{-\|v_{xy} - v_{xy}^{cmd}\|^2/\sigma^2}$ | **核心任务**：鼓励机器人精准跟随用户输入的 XY 线速度指令。|
| **rew_ang_vel_z**        | $e^{-(\omega_z - \omega_z^{cmd})^2 / \sigma^2}$ | **核心任务**：鼓励机器人精准跟随转向（Yaw）指令。|
| **pen_flat_orientation** | $\|\mathbf{g}_{proj} - [0, 0, -1]^T\|$ | **姿态约束**：惩罚重力投影偏离 Z 轴，强制躯干保持水平。|
| **pen_lin_vel_z**        | $v_z^2$ | **稳定性**：惩罚基座在 Z 轴的运动，抑制跳跃和颠簸。|
| **pen_joint_accel**      | $\|\ddot{q}\|$ | **平滑性**：惩罚关节加速度，减少电机高频震荡和磨损。|
| **pen_action_rate**      | $\|a_t - a_{t-1}\|$ | **平滑性**：惩罚动作的一阶差分，鼓励控制信号连续平滑。|
| **test_gait_reward**     | $r_{force} + r_{vel}$ | **步态塑形**：强制足端在 *Stance* 相触地受力，在 *Swing* 相抬起运动。|
| **rew_feet_clearance**   | $\sum (h_{foot} - h_{target})^2 \cdot v_{xy}$ | **越障能力**：在摆动相奖励足端抬高到指定高度，防止踢到台阶边缘。|
| **foot_landing_vel**     | $\sum v_{z, impact}^2$ | **柔顺性**：惩罚触地瞬间的 Z 轴速度，鼓励轻柔着陆，减少冲击。|

---

## 2. 算法对比分析：Encoder-MLP vs HIM vs PIM

本部分对代码中实现的三种训练配置（Encoder-MLP、HIM 与 PIM）进行了系统对比。这三种方法的核心差异并不在于控制形式本身，而在于 Policy 对环境与自身状态的建模方式。

Encoder-MLP 直接在纯本体感知条件下进行策略学习，完全依赖动力学反馈进行隐式补偿；HIM 通过引入基于历史本体感知的内部模型，使 Policy 能够推理不可观测的隐式状态；PIM 在此基础上进一步融合外部感知信息，构建包含地形几何结构的多模态状态表征。

为适配不同的状态建模范式，三种方法在观测空间结构与奖励函数权重上进行了相应调整。他们围绕同一目标展开：在受限或不完整感知条件下，使 Policy 获得足够的信息以稳定应对复杂地形，并提升整体鲁棒性与泛化能力。

### 2.1 算法变体定义

1. **Encoder-MLP**
    Encoder-MLP 是最基础的盲视策略，依赖本体感知信息估计机器人基座线速度。机器人没有外部感知能力，仅靠本体感知（IMU、关节）行走。
    
    <img src="images/Structure_encoder_mlp.png" alt="Structure_encoder_mlp" style="zoom:29%;" />

  - 可观测信息：关节位置 / 速度、IMU 信息（姿态、角速度）、速度指令等本体感知量。
  - 缺失信息：无任何显式环境感知（如地形高度、台阶结构）。
  - 主要能力：平地或弱起伏地形；地形变化缓慢、可由动力学反馈“滞后补偿”的场景。

2. **HIM (Hybrid Internal Model)** 
    HIM 训练一个基于本体感知历史的**内部模型 (Internal Model/Estimator)**，通过**对比学习**显式地**预测**隐式特权信息，使得 Policy 不仅仅是被 Critic “指导”，而是自身具备了从本体感知中**推理**环境和自身状态的能力。
    
    <img src="images/Structure_him.png" alt="Structure_him" style="zoom:80%;" />

  - Actor：仍然是盲视的，不直接接收地形高度信息；
  - Critic：拥有特权信息（如地形高度扫描），从而提供更准确的价值估计；
  - Internal Model：作为中介，将 Critic 的“知识”蒸馏到 Actor 的内部状态表示中。

3. **PIM (Perceptive Internal Model)**
    在 HIM 的 Policy 观测量基础上，PIM 的 Policy 观测器引入了外部感知，即视觉（高程图）编码器，构建了一个**多模态的内部模型**。PIM 利用雷达扫描信息来**修正和增强**对环境状态的估计（即构建包含环境信息的内部表征），从而让机器人能够**主动规划落点**以应对盲视无法处理的**剧烈地形变化**（如陡峭楼梯或断崖）。
    
    <img src="images/Structure_pim.png" alt="Structure_pim" style="zoom:65%;" />

  - Actor：Policy 观测中新增 perceptive 观测，同时接收本体与环境感知信息，使 Actor 能够在决策阶段显式利用地形几何结构；
  - Critic：拥有 perceptive 观测以及其他特权信息；
  - Internal Model：作为中介，将 Critic 的“知识”蒸馏到 Actor 的内部状态表示中。

### 2.2 观测空间与算法架构对比

HIM 和 PIM 的观测量配置分别在代码文件`limx_him_base_env_cfg.py` 和 `limx_pim_base_env_cfg.py`中。

| **特性** | **Base (Blind Flat/Rough)** | **HIM (Blind Stairs)** | **PIM (Blind Stairs)** |
| -------- | ----------- | ----------------------- | ---------- |
| **Policy 输入** | **本体感知** | **本体感知** | **本体感知 + 环境感知（地形高度扫描）** |
| **Critic 输入** | 本体感知 + 特权物理信息 | 本体感知 + 特权物理信息 + **地形高度扫描** | 本体感知 + 特权物理信息 + **地形高度扫描** |
| **地形感知**    | 无 | **开启** (仅 Critic 可见) | **开启** |
| **感知配置**    | 无 | `observations.critic.heights` | `observations.perceptive.heights` |
| **核心逻辑**    | 学习基础运动，对复杂地形适应能力差。 | 利用特权高度信息辅助 Critic 估值，结合隐式状态估计预测当前环境，训练盲视 Actor 应对复杂地形。 | 使用高程图（外部感知）和本体的感知信息结合，增强对地形的预测和自身状态估计能力，以应对复杂地形。 |

三种方法的本质差异不在于控制器形式，而在于状态信息如何被获取、推理和使用。

### 2.3 奖励函数对比

为了匹配不同的信息结构，三种算法在奖励权重设计上也存在系统性的调整。下表总结了 `limx_pointfoot_env_cfg.py` 中的具体权重的主要差异。

| **奖励项 (Reward Term)** | **功能描述** | **Base (Default)**| **HIM Weight** | **PIM Weight** | **分析与解读** |
| ----- | --- | --- | --- | --- | ------------ |
| **rew_lin_vel_xy**       | XY线速度追踪 | 3.0 | **1.0** | **1.0** | HIM/PIM 降低了速度追踪的绝对权重，避免过拟合速度而忽略地形稳定性。 |
| **rew_ang_vel_z**        | 转向追踪     | 1.5 | **0.5** | **0.5** | 同上，降低转向权值。|
| **pen_lin_vel_z**        | 垂直振动抑制 | -0.5 | **-2.0** | **-2.0** | HIM/PIM 大幅增加了对机身垂直晃动的惩罚，要求在楼梯上行走更加平稳。 |
| **pen_flat_orientation** | 姿态水平     | -10.0 | **-2.0** | **-0.2** | PIM 放宽了姿态约束，高程图信息使机器人能够更加自然适应复杂地形，无需大惩罚。 |
| **pen_base_height**      | 基座高度     | -20.0 | **-1.0** | **-1.0** | 降低惩罚权重，使机器人在高度变化剧烈的地形中能够有效调整姿态。 |
| **rew_feet_clearance**   | 抬脚高度     | 0 | **0.2** | **0.5** | HIM 和 PIM均奖励抬脚高度，以应对复杂的地形。 |
| **test_gait_reward**     | 步态一致性   | 1.0 | **0.5** | **0.4** | 在复杂地形上，严格的强制步态可能适得其反，因此降低了步态约束的权重。 |
| **pen_feet_distance**    | 双脚间距     | -10.0 | **-40.0** | **-40.0** | 增大惩罚，防止机器人因为复杂地形而损失步态风格。 |
| **pen_feet_regulation**   | 精简奖励     | -0.1 | **0** | **0** | HIM/PIM 移除了针对平地优化的足部调节规则，依靠状态预测算法自动调节。 |
| **foot_landing_vel**      | 着陆速度     | -0.5 | **0** | **0** | 同上，移除了针对平地优化的着陆速度惩罚。 |

- Base 策略通过“强奖励约束”维持行为；
- HIM 通过“状态推理”减弱对奖励塑形的依赖；
- PIM 则进一步将稳定性来源从奖励转移到感知 + 表征能力本身。



---

## 3. 平地速度跟随

### 3.1 实验设置与算法配置

本实验阶段的主要任务是实现四足机器人在平坦地面上的稳定运动控制。具体要求策略网络能够根据输入的指令 $\mathbf{c} = [v_x^{cmd}, v_y^{cmd}, \omega_z^{cmd}]$，精准地控制机器人的线速度和角速度，同时保持基座姿态（Roll / Pitch）的平稳，避免在高速运动或转向时发生跌倒。

在该任务中，环境被配置为 `PFBlindFlatEnvCfg`。与复杂地形任务不同，采用 **PF Base Blind Flat (Encoder-MLP)** 策略，旨在验证 TRON1 机器人在平坦地面上的全向移动能力与姿态稳定性。

| **配置项** | **参数设定** | **说明** |
| --------------- | ----------------------- | ------------------------------- |
| **算法架构** | Base Blind <br> (Encoder-MLP) | 仅观测本体感知信息 。 |
| **地形环境** | Flat Plane | 无障碍物的无限延伸平坦地面。 |
| **观测空间** | $\mathbf{o}_t \in \mathbb{R}^{48}$ | 包含 $\omega_{base}, \mathbf{g}_{proj}, q, \dot{q}, a_{t-1}$ 及步态指令。 |
| **指令范围** | $v_x \in [-1.5, 1.5]$ m/s <br> $v_y \in [-1.0, 1.0]$ m/s <br> $\omega_z \in [-0.5, 0.5]$ rad/s | 随机指令每 5 秒重采样一次。 |

### 3.2 思路与方法

#### 3.2.1 观测空间设计

由于采用了“盲视”设置，观测空间被精简为纯本体感知信息。观测向量 $\mathbf{o}_t$ 包含以下关键分量：

- **指令**：目标速度 $[v_x, v_y, \omega_z]$。
- **本体状态**：基座角速度、重力向量在基座坐标系下的投影（反映姿态）、关节位置及速度。
- **历史信息**：上一时刻的动作输出，用于捕捉时序特征。

#### 3.2.2 奖励函数塑造

在 `limx_base_env_cfg.py` 中设计了复合奖励函数。总奖励 $r_t$ 由追踪奖励、正则化惩罚和步态约束组成。

* **速度追踪**

  最小化均方误差 (MSE)，采用指数核函数形式的奖励，在误差接近 0 时提供密集的梯度信号：

  $$
  r_{vel} = \alpha_1 \exp\left(-\frac{\|v_{xy} - v_{xy}^{cmd}\|^2}{\sigma_v^2}\right) + \alpha_2 \exp\left(-\frac{(\omega_z - \omega_z^{cmd})^2}{\sigma_\omega^2}\right)
  $$
  在配置中，`rew_lin_vel_xy` 权重设为 **3.0**，`rew_ang_vel_z` 权重设为 **1.5**。高权重的设定使速度跟踪作为所有奖励的主导项。

* **姿态稳定性与存活**

  为了降低 Roll/Pitch 的震荡幅度并防止摔倒，引入了以下关键项：

  - **基座姿态惩罚 (`pen_flat_orientation`)**：权重 **-10.0**。该项惩罚重力向量在基座 XY 平面上的分量，约束机器人保持躯干水平。
  - **角速度正则化 (`pen_ang_vel_xy`)**：权重 **-0.05**。抑制非指令方向（Roll/Pitch方向）的角速度，减少躯干晃动。
  - **存活奖励 (`keep_balance`)**：权重 **1.0**。只要机器人未触发终止条件（如基座接触地面），每一步都会获得正向奖励，鼓励长时运行。

* **动作平滑与步态约束**

  - **步态奖励 (`test_gait_reward`)**：权重 **1.0**。通过 `GaitReward` 函数，让机器人学习特定的接触相和摆动相时序，避免生成不自然步态（滑步或跳跃），间接提高了行走的稳定性。
  - **平滑性惩罚**：包括 `pen_joint_accel` (关节加速度)、`pen_action_rate` (动作变化率) 和 `pen_joint_powers` (功率)。

#### 3.2.3 域随机化

为了增强策略的鲁棒性，使策略具备一定的泛化能力，在 `EventsCfg` 中配置随机化事件以增加适当扰动：

- **推力扰动 (`push_robot`)**：以 **0.002** 的概率（每步）在基座上施加瞬时推力（XY方向最大 $± 500N$）。这迫使 Policy 学习在受到外力冲击后快速调整足端落点以恢复平衡（Push Recovery）。
- **动力学参数随机化**:
  - **质量 (`add_base_mass`)**：基座质量在 $[-1.0, 3.0] kg$ 范围内变化，模拟不同负载情况。
  - **摩擦力 (`robot_physics_material`)**：地面摩擦系数在 $[0.4, 1.2]$ 间变化，确保机器人在不同表面（从滑到涩）都能稳定行走。
  - **关节刚度与阻尼**：模拟电机特性的不确定性。

### 3.3 实验结果展示

**图 1：速度跟随任务仿真截图**

<div style="display:flex; gap:10px; justify-content:center;">
  <img src="images/Screenshot_flat_ground_velocity_tracking1.png" style="zoom:55%;">
  <img src="images/Screenshot_flat_ground_velocity_tracking2.png" style="zoom:55%;">
  <img src="images/Screenshot_flat_ground_velocity_tracking3.png" style="zoom:55%;">
</div>


使用上述配置训练3000轮后的表现如下。

#### 3.3.1 速度响应曲线

**图 2：线速度 ($v_x, v_y$) 与角速度 ($\omega_z$) 追踪性能**

<img src="images/flat_1_velocity_tracking.png" alt="flat_1_velocity_tracking" style="zoom:15%;" />

**图 3：线速度 ($v_x, v_y$) 与角速度 ($\omega_z$) 跟踪误差分布**

<img src="images/flat_2_error_distribution.png" alt="flat_2_error_distribution" style="zoom:17%;" />

**数据分析:**

* **速度跟踪性能**

  从时域波形图来看，控制器对 **阶跃信号** 展现了良好的动态响应能力：

  - **响应速度**：$v_x$（纵向）、$v_y$（横向）和 $\omega_z$（转向）均能迅速响应指令变化，上升/下降沿陡峭，无显著的延迟或超调。
  - **稳定性**：在指令保持阶段，机器人实际速度围绕期望值波动，但整体均值稳定，未出现发散或明显的稳态误差漂移。
  - **耦合影响**：$v_x$ 的大幅突变未对 $v_y$ 和 $\omega_z$ 造成显著的干扰，说明各自由度间的解耦控制较为理想。

* **误差统计分布**

  误差分布图进一步量化了跟踪精度，三轴误差均呈现标准的**高斯分布（正态分布）**特性：

  - **无系统偏差**：三个维度的误差均值 ($\mu$) 极低（$v_x:0.028, v_y:0.035, \omega_z:-0.034$），几乎为零，说明模型不存在系统性的“跑偏”问题。
  - **精度分析**：
    - **$v_y$ (横向)** 表现最佳，具有最小的均方误差 (MSE：6.4e-02) 和标准差 ($\sigma=0.250$)，说明横向控制最为收敛。
    - **$v_x$ (纵向)** 的波动范围稍大 ($\sigma=0.392$)，MSE (1.5e-01) 最高，这通常是因为纵向运动幅度大且受腿部摆动对地冲击影响最直接。
    - **$\omega_z$ (转向)** 标准差 ($\sigma=0.315$) 居中，控制较为平稳。

该双足机器人的运动控制策略表现出**响应快、精度高、无偏置**的特性。尽管存在部分高频震荡噪声，但误差被有效限制在合理的高斯分布范围内，能够精确地执行复杂的速度切换指令。

#### 3.3.2 姿态稳定性分析

**图 4：基座姿态角 (Roll & Pitch) 随时间变化**

<img src="images/flat_3_oscillation.png" alt="flat_3_oscillation" style="zoom:17%;" />

**数据分析：**

* **振荡幅度与有界性**

  整体来看，机器人表现出了**动态稳定但伴随高频抖动**的特性。虽然存在姿态波动，但并未发生发散（倒地），表明控制策略具有鲁棒性。

  - **俯仰角**：波动范围较大（范围：0.431 rad，约 $24.7^\circ$）。这是由急加减速引起的惯性效应。曲线中明显的尖峰（如 $t=50s$ 附近）对应了**速度指令的大幅切换**，说明机器人在应对纵向冲击时会产生较大的前后晃动。
  - **横滚角**：波动范围相对较小（范围：0.354 rad，约 $20.3^\circ$）。考虑到双足机器人的横向平衡通常较难维持，这一波动幅度表明机器人在高速运动中左右摇摆较为剧烈，呈现出“踏步调整”的策略来维持平衡。

* **稳态偏置**

  观察零基准线，两个自由度均存在微小的非零均值偏置：

  - **横滚角**：呈现靠近0的微小负偏置，机器人能够基本保持基座水平。
  - **俯仰角**：主体位于正值区间，同样靠近0参考线。表明机器人在运动时保持着轻微的**前倾姿态**。这对于较高速度的运动是合理的，有助于质心前移以辅助加速。

* **收敛能力**

  - **快速恢复**：尽管在 $t=5s, 37s, 50s$ 等时刻出现了大幅度的姿态突变（对应速度阶跃），曲线总是能迅速回调至均值附近。这种**强回复力**说明RL策略学习到了有效的姿态恢复机制，能够抵抗剧烈的加减速扰动。

  该双足机器人在执行速度跟踪任务时，**在确保姿态尽量平稳的同时保持高动态响应能力**。面对指令速度的大幅切换，能够快速调整自身姿态，执行速度跟随任务。

**数据统计:**

| **指标 (Metric)**                   | **数值**            |
| ----------------------------------- | ------------------- |
| **线速度追踪 MSE** ($v_x$)          | **0.15** $(m/s)^2$  |
| **线速度追踪 MSE** ($v_y$)          | **0.064** $(m/s)^2$ |
| **角速度追踪 MSE** ($\omega_z$)     | **0.1** $(rad/s)^2$ |
| **Roll 震荡幅度** ($R_\phi$)        | **0.354** rad       |
| **Pitch 震荡幅度** ($R_\theta$)     | **0.431** rad       |
| **存活率** (about 1 min continuous) | **100%**            |

基于 `PFBlindFlatEnvCfg` 的配置，通过高权重的速度追踪奖励配合严格的姿态惩罚，以及推力扰动训练，我们成功训练出了一个在平坦地面上具备高性能速度跟随能力且鲁棒的盲视行走策略。该策略不仅满足了基本的移动需求，还在抗干扰和姿态平稳性上达到了预期指标。


---

## 4. 抗干扰鲁棒性测试

### 4.1 测试任务描述

本测试旨在评估 Policy 在平地行走过程中应对突发外部干扰的稳定性。在实际部署中，机器人可能会遭遇碰撞、地面突然滑动或被推挤等情况。为了验证控制策略的鲁棒性，我们在仿真环境中利用域随机化技术，向机器人的基座施加不可预测的瞬时推力，观察其是否能够保持平衡并快速恢复到正常的行走步态。

### 4.2 实验设置

实验基于 `PFBlindFlatEnvCfg` 环境配置，主要通过 `EventsCfg` 中的 `push_robot` 事件来实现干扰施加。

- 干扰注入机制:

  使用了 `mdp.apply_external_force_torque_stochastic` 函数。该函数会在仿真过程中以一定的概率随机采样力和力矩，并直接作用于机器人的刚体上。

- 参数配置:

  根据 `limx_base_env_cfg.py` 的定义，干扰参数如下：

  - **施力对象**：机器人的基座 (`base_Link`)。
  - **力的大小**：在 $x$ 和 $y$ 轴方向上，力的大小在 $[-70.0, 70.0]$ N 范围内均匀采样。
  - **力矩大小：在 $x$ 和 $y$ 轴方向上，力矩在 $[-5.0, 5.0]$ N·m 范围内均匀采样。
  - **触发概率**：每步触发概率为 0.3，模拟随机间断性的外力冲击。

### 4.3 考核指标

为了量化抗干扰能力，我们定义了以下两个核心指标：

1. 最大承受冲量:

   机器人能够承受且不发生跌倒（即未触发 `base_contact` 终止条件）的最大水平推力冲量。这反映了系统的稳定裕度。

2. 步态恢复速度:

   定义为从受到干扰时刻 $t_{impact}$ 开始，到机器人的基座线速度误差回归到稳态基准范围（例如 $\pm 5\%$ 误差带）所需的时间。恢复时间越短，说明 Policy 的动态调整能力越强。

### 4.4 结果与分析

**图 5：抗干扰测试仿真截图 (Encoder-MLP vs HIM vs PIM)**

<div style="display:flex; gap:10px; justify-content:center;">
  <img src="images/Screenshot_flat_ground_disturbance_rejection1.png" style="zoom:50%;">
  <img src="images/Screenshot_him_disturbance_rejection2.png" style="zoom:50%;">
  <img src="images/Screenshot_pim_disturbance_rejection3.png" style="zoom:50%;">
</div>

**图 6：步态相位差及机器人所受外力随时间变化曲线：**

<img src="images/SDM5008 Report/flat_4_gait_phase_with_force.png" alt="flat_4_gait_phase_with_force" style="zoom:20%;" />

**图 7：随机推力影响下基座姿态角随时间变化**

<img src="images/SDM5008 Report/flat_3_oscillation.png" alt="flat_3_oscillation" style="zoom:20%;" />

**数据分析：**

**干扰力与冲量分析**

- **最大瞬时推力**：
  - 最大推力峰值为 **67.2 N**。
  - 结合您提到的“约 80N”观察值，推测在数据采样间隙或未展示的行中可能存在接近 **80 N** 的峰值。我们以 **67.2 N ~ 80 N** 作为本次考核的极限压力值。
- **干扰持续时间**：
  - 干扰持续时间与控制频率相同，**$\Delta t=0.2s$**。
- **最大冲量**：
  - 峰值冲量：$I \approx 67.2 \, \text{N} \times 0.2 \, \text{s} = \mathbf{13.4 \, \text{Ns}}$。
  - **评价**：对于中型机器人，承受 13-16 Ns 的侧向/后向冲量属于**高难度**测试，相当于受到一个成年人中等力量的快速推搡。

**响应与恢复过程**

- **速度响应**：
  - **受击瞬间**：机器人受到推力后，实际速度 `act_vx` 从 $0.5 \, \text{m/s}$ 迅速飙升至峰值 **0.96 m/s** (`t_step` 49)。这表明机器人被推着向前“冲”了一段距离。
  - **调整过程**：随后 Policy 采取了急刹车策略，速度在 `t_step` 99 骤降至 **0.05 m/s**（几乎停顿），这是一种典型的“踉跄后重整步伐”的策略。
  - **恢复稳态**：速度在 `t_step` 130 左右重新稳定在 **0.50 m/s**。
  - **恢复用时**：从干扰结束 (`t_step` 48) 到速度回归 (`t_step` 130)，耗时约 **1.64 秒** ($82 \text{ steps} \times 0.02 \text{s}$)。
- **姿态稳定性**：
  - **姿态角**：在整个剧烈变速过程中（0.96 m/s -> 0.05 m/s），`abs_roll` 保持在 $\approx 3.1$ ($180^\circ$)，`abs_pitch` 保持在 $0.05 \text{ rad}$ ($3^\circ$) 以内。
  - **结论**：尽管速度波动剧烈，但机身姿态始终保持水平，**未发生翻滚或倾倒**。

**步态相位分析**

- **相位连续性**：在外力介入区域（**图 6** 中剧烈波动段），步态相位曲线虽然频率发生短暂变化（为了快速调整落足点），但**没有出现相位错乱或信号丢失**。这证明 Policy 具备**基于相位的抗扰动能力**，通过调节步频来吸收冲击能量。

| **考核维度**     | **测量数据/表现** | **分析**                                                     |
| ---------------- | ----------------- | ------------------------------------------------------------ |
| **最大承受推力** | **67.2 N**        | 能够承受显著的外部冲击（~16 Ns 冲量），远超一般行走测试的微扰标准。 |
| **抗摔倒能力**   | **未摔倒**        | 在速度剧烈波动（0.05~0.96 m/s）的情况下，IMU 姿态数据未见发散，机身极其稳定。 |
| **步态恢复速度** | **~1.6 秒**       | 恢复时间在 1.5-2.0秒之间。机器人采用“前冲-急停-调整”的策略，虽然未能“瞬间”恢复（<1s），但这种策略对于大推力来说更加稳健安全。 |


---

## 5. 复杂地形适应

### 5.1 测试任务描述

本测试旨在评估所训练 Policy 在非平坦复杂地形条件下的行走稳定性与环境适应能力。相较于平地行走，复杂地形（如台阶、坡道和不规则高度扰动）会对机器人的足端落点选择、躯干姿态控制以及周期性步态协调提出更高要求。

在真实场景中，双足机器人不可避免地需要应对地面高度突变、局部坡度变化以及感知不完全等问题。因此，本实验通过在仿真环境中引入参数化随机地形，系统性地考察策略在地形变化条件下是否能够保持持续行走、不发生跌倒，并维持合理的运动效率和姿态稳定性。

### 5.2 实验设置

实验基于点足机器人构建，并在 PF 配置下切换不同复杂地形模式。地形由 `terrains_cfg.py` 统一管理，通过高度场方式在仿真中生成。

- **环境配置：**

  使用环境配置文件：`limx_base_env_cfg.py`，`terrains_cfg.py`
  启用地形感知但不提供显式地形高度输入，主要考察策略的内在动力学鲁棒性。

- **地形类型：**
  
  - **台阶地形**
    地面高度以固定步长周期性变化，用于考察策略对高度突变的适应能力。
  - **坡道地形**
    包含正坡与负坡，主要测试机器人对重力分量变化的补偿能力。
  - **障碍物地形**
    地形中分布有离散的几何障碍（凸起或凹陷），其高度和间距在一定范围内随机采样，用于测试机器人足端落点选择能力与避障稳定性。
  - **随机起伏地形**
    地形高度在给定范围内随机扰动，用于模拟自然环境中的不规则地面。

- **地形参数随机化：**
  - 台阶高度、坡度角度、地形粗糙度在每个轮次重置时随机采样；
  - 地形参数分布与训练阶段保持一致，以评估策略的泛化能力。

- **终止条件：**
  
  - 机器人基座与地面发生非期望接触；
  - 基座姿态角超过安全阈值；
  - 机器人速度长期偏离指令值。

### 5.3 考核指标

- **指令速度跟踪情况**
  
  分析机器人在复杂地形中对指令速度的跟随情况。
  
- **最大可通过地形难度** 
  对不同地形逐步提高难度参数（如台阶高度、坡度角、地形起伏幅值），记录机器人仍能保持稳定行走的最大参数值，用以衡量策略的适应上限。

- **运动连续性**
  通过分析足端接触序列与步态周期变化，评估策略是否出现明显的停顿、拖步或非周期性异常动作。

### 5.4 结果与分析

**图 8：HIM 复杂地形下速度跟随仿真截图**

<div style="display: flex; flex-direction: column; align-items: center; gap: 10px;">
  <div style="display: flex; gap: 10px; justify-content: center;">
    <img src="images/Screenshot_him_downstairs.png" style="zoom:33%;">
    <img src="images/Screenshot_him_upstairs.png" style="zoom:33%;">
    <img src="images/Screenshot_him_slopes.png" style="zoom:33%;">
  </div>

  <div style="display: flex; gap: 10px; justify-content: center;">
    <img src="images/Screenshot_him_obstacles.png" style="zoom:50%;">
    <img src="images/Screenshot_him_rough.png" style="zoom:50%;">
  </div>

</div>

**图 9：HIM 台阶地形下速度跟踪曲线**

<img src="images/him_downstairs_1_velocity_tracking.png" alt="him_downstairs_1_velocity_tracking" style="zoom:16%;" />

**图 10：PIM 复杂地形下速度跟随仿真截图**

<div style="display: flex; flex-direction: column; align-items: center; gap: 10px;">
  <div style="display: flex; gap: 10px; justify-content: center;">
    <img src="images/Screenshot_pim_downstairs.png" style="zoom:35%;">
    <img src="images/Screenshot_pim_upstairs.png" style="zoom:35%;">
    <img src="images/Screenshot_pim_slopes.png" style="zoom:35%;">
  </div>

  <div style="display: flex; gap: 10px; justify-content: center;">
    <img src="images/Screenshot_pim_obstacles.png" style="zoom:35%;">
    <img src="images/Screenshot_pim_rough.png" style="zoom:35%;">
    <img src="images/Screenshot_pim_edge.png" style="zoom:35%;">
  </div>

</div>

**图 11：PIM 台阶地形下速度跟踪曲线**

<img src="images/pim_stairs_1_velocity_tracking.png" alt="pim_stairs_1_velocity_tracking" style="zoom:17%;" />

**数据分析：**

**指令速度跟随情况**

- HIM 和 PIM 在复杂地形中都展现出良好的指令速度跟随效果（以楼梯地形为例）。

  | 算法    | 指标                       | 数值                 |
  | ------- | -------------------------- | -------------------- |
  | **HIM** | **线速度追踪 MSE** ($v_x$) | 0.032 $(m/s)^2$      |
  |         | **线速度追踪 MSE** ($v_y$) | 0.018$(m/s)^2$       |
  |         | **角速度追踪 MSE** ($w_z$) | 0.11$(rad/s)^2$      |
  | **PIM** | **线速度追踪 MSE** ($v_x$) | **0.029** $(m/s)^2$  |
  |         | **线速度追踪 MSE** ($v_y$) | **0.0062**$(m/s)^2$  |
  |         | **角速度追踪 MSE** ($w_z$) | **0.052**$(rad/s)^2$ |

- 从数据中可以发现，**PIM** 的各方向速度跟踪误差均小于 HIM, 展现了**带感知**运动控制算法对**复杂地形**的强大**适应能力**。
- 同样作为盲视的运动控制策略，即使在楼梯地形，**HIM** 的**线速度跟踪**误差仍然远**优于** Encoder-MLP 算法。以$v_x$的跟踪为例，提升高达**78.67%**。

**最大可通过地形难度**

使用训练最终阶段的地形难度收敛值最为最大可通过地形难度指标。

- HIM：收敛至3.8左右。

  | **子地形类型**           | **参数项** | **范围配置**  | **收敛数值**         |
  | ------------------------ | ---------- | ------------- | -------------------- |
  | **楼梯 (Stairs)**        | 台阶高度   | 0.05 ~ 0.20 m | **11.33 cm**         |
  | **粗糙地形 (Rough)**     | 噪声高度   | 0.01 ~ 0.06 m | **3.11 cm**          |
  | **平滑斜坡 (Slope)**     | 坡度       | 0.0 ~ 0.7     | **0.295 (约 16.5°)** |
  | **离散障碍 (Obstacles)** | 障碍高度   | 0.05 ~ 0.15 m | **9.22 cm**          |

- PIM：收敛至5.2左右。

  | **子地形类型**           | **参数项** | **范围配置**  | **结果数值**         |
  | ------------------------ | ---------- | ------------- | -------------------- |
  | **楼梯 (Stairs)**        | 台阶高度   | 0.05 ~ 0.20 m | **13.67 cm**         |
  | **粗糙地形 (Rough)**     | 噪声高度   | 0.01 ~ 0.06 m | **3.89 cm**          |
  | **平滑斜坡 (Slope)**     | 坡度       | 0.0 ~ 0.7     | **0.405 (约 22.0°)** |
  | **离散障碍 (Obstacles)** | 障碍高度   | 0.05 ~ 0.15 m | **10.78 cm**         |

**动作连续性**

- HIM 和 PIM 均未发现明显的磕碰和动作失调。

**边缘检测**

- 由于 HIM 为盲视策略，无法检测地形边缘，会跟随指令速度掉落边缘。
- **PIM** 发挥了**传感器**的优势，能够**检测到边缘**并且抗拒指令速度，控制自身**不掉落**。



---

## 6. 开源项目发布

为验证本文所实现算法的可复现性、工程完整性与可扩展性，并推动相关研究工作的复用与交流，本文将基于 Isaac Lab 实现的多种步态控制算法（Encoder-MLP / HIM / PIM）整理并发布为一个标准的开源项目。项目涵盖从环境配置、训练脚本到可视化演示的完整流程，符合主流机器人学习开源项目的组织规范。

### 6.1 项目链接

- GitHub 仓库：
  https://github.com/DongyangLin/SDM5008Project.git

- 演示视频 / GIF：
  Gif 见项目 [README.md](../README.md) ，完整演示视频见 Blackboard。

### 6.2 项目使用

[README.md](../README.md) 中包含项目简介、环境配置指南、训练与测试指令示例、Demo 视频、代码主要框架等。

---

## 附录

**Blind Flat 训练曲线**

![flat_training_episode_length](images/flat_training_episode_length.png)

![flat_training_mean_reward](images/flat_training_mean_reward.png)

**HIM 训练曲线**

![him_training_episode_length](images/him_training_episode_length.png)

![him_training_mean_reward](images/him_training_mean_reward.png)

![him_training_terrain_level](images/him_training_terrain_level.png)

**PIM 训练曲线**

![pim_training_episode_length](images/pim_training_episode_length.png)

![pim_training_mean_reward](images/pim_training_mean_reward.png)

![pim_training_cmd_level](images/pim_training_cmd_level.png)

![pim_training_terrain_level](images/pim_training_terrain_level.png)

[^1]:Long, J., Ren, J., Shi, M., Wang, Z., Huang, T., Luo, P., & Pang, J. (2024). **Learning Humanoid Locomotion with Perceptive Internal Model**. *arXiv preprint arXiv:2411.14386*. https://arxiv.org/abs/2411.14386
[^2]:Long, J., Wang, Z., Li, Q., Gao, J., Cao, L., & Pang, J. (2024). **Hybrid Internal Model：Learning Agile Legged Locomotion with Simulated Robot Response**. *The Twelfth International Conference on Learning Representations (ICLR)*. https://arxiv.org/abs/2312.11460
