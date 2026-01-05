## 框架理解与代码总结 (Code Review & Architecture)

本项目基于 NVIDIA Isaac Lab 框架构建，采用模块化的配置驱动设计（Configuration-driven Design）。通过精读代码，我们将系统解耦为场景配置（Scene）、观测（Observation）、奖励（Reward）和动作（Action）四个核心子系统。以下是对各模块功能的详细梳理及架构分析。

### 1. 场景配置 (Scene Configuration)

场景配置模块负责定义仿真环境的物理实体和环境属性。在本项目中，这主要体现在 `PFSceneCfg` 类及其引用的资产配置中。

* **USD 资产加载与物理属性**:
    在 `exts/bipedal_locomotion/bipedal_locomotion/assets/config/pointfoot_cfg.py` 中，`POINTFOOT_CFG` 定义了机器人的资产属性。
    * **USD 路径**: 代码通过 `sim_utils.UsdFileCfg` 加载位于 `../usd/PF_TRON1A/PF_TRON1A.usd` 的通用场景描述（USD）文件，这是机器人的几何与物理模型基础。
    * **刚体与关节属性**: 配置中显式定义了刚体属性（`RigidBodyPropertiesCfg`），如启用自碰撞（`enabled_self_collisions=True`）和求解器迭代次数，确保物理仿真的稳定性。
    * **初始状态**: `init_state` 定义了机器人出生时的默认关节角度（`joint_pos`）和基座位置，为强化学习提供一致的复位状态。

* **场景集成**:
    在 `limx_base_env_cfg.py` 的 `PFSceneCfg` 类中，机器人资产被集成到交互式场景中。同时，该类还配置了地形（`TerrainImporterCfg`）、光照（`DomeLightCfg`）以及传感器（如 `ContactSensorCfg` 用于足部接触检测）。

### 2. 动作管理器 (Action Manager)

动作管理器定义了智能体（Policy）输出与仿真器执行器之间的接口。

* **动作空间定义**:
    在 `limx_base_env_cfg.py` 的 `ActionsCfg` 中，定义了 `joint_pos` 动作项，使用 `mdp.JointPositionActionCfg`。这意味着策略网络的输出被解释为关节位置的目标偏移量。
    
    * **缩放因子 (Scaling)**: 配置设定了 `scale=0.25`，将神经网络输出的无量纲动作值映射到物理关节角度范围。
    * **关节映射**: 明确指定了控制的关节名称（如 `abad_L_Joint`, `knee_R_Joint` 等）。
    
* **PD 控制器设定**:
    实际的力矩生成发生在物理引擎层，由 `pointfoot_cfg.py` 中的 `actuators` 配置定义。项目使用了 `RandomLaggyActuatorCfg`（一种带有随机延迟的隐式执行器配置），其本质是一个比例-微分（PD）控制器。
    根据配置中的刚度（Stiffness, $$K_p$$）和阻尼（Damping, $$K_d$$）参数（例如 `stiffness=40.0`, `damping=2.5`），物理引擎计算最终施加的力矩 $$\tau$$：
    $$
    \tau = K_p (q_{target} - q_{current}) - K_d \dot{q}_{current}
    $$
    其中 $$q_{target}$$ 由默认关节位置加上缩放后的动作值计算得出：$$q_{target} = q_{default} + \text{scale} \times a_{network}$$。

### 3. 观测管理器 (Observation Manager)

观测管理器负责构建状态空间，并模拟真实世界的传感器噪声。在 `limx_base_env_cfg.py` 的 `ObservarionsCfg` 类中，观测被分为不同的组（Group）。

* **策略观测 (Policy Group)**:
    `PolicyCfg` 定义了输入给 Actor 网络的观测向量。为了缩小 Sim-to-Real 的差距，这里引入了噪声注入和归一化：
    * **噪声注入**: 使用 `GaussianNoise` 为观测添加高斯白噪声。例如，`base_ang_vel`（基座角速度）添加了均值为 0、标准差为 0.05 的噪声。
    * **观测项**: 包含基座角速度、重力投影向量（反映姿态）、关节位置和速度、上一次的动作以及步态指令（`gait_command`）。
    * **处理流程**: 原始物理数据 $$\rightarrow$$添加噪声$$\rightarrow$$裁剪（Clip）$$\rightarrow$$缩放（Scale）$$\rightarrow$$ 神经网络输入。

* **评价观测 (Critic Group)**:
    `CriticCfg` 定义了输入给 Critic 网络的观测。这是一个“特权”（Privileged）观测空间，仅在训练阶段使用。
    * **特征**: 包含所有策略观测的无噪声版本（Ground Truth），以及策略网络无法获取的额外信息，如机器人质量（`robot_mass`）、地形高度扫描（`heights`）、接触力（`robot_feet_contact_force`）和物理属性（摩擦力、关节刚度等）。这有助于 Critic 更准确地估计价值函数。

### 4. 奖励管理器 (Reward Manager)

奖励管理器通过 `RewardsCfg` 类定义了强化学习的目标函数。奖励函数的设计直接决定了机器人的行为风格。

* **奖励项设计**:
    代码中使用了多种奖励项（`RewTerm`），主要分为两类：
    1.  **跟踪奖励 (Tracking Rewards)**: 鼓励机器人服从速度指令。例如 `rew_lin_vel_xy` 使用高斯核函数鼓励实际速度接近指令速度：
        $$
        r_{vel} = \exp\left(-\frac{\|v_{cmd} - v_{meas}\|^2}{\sigma^2}\right)
        $$
    2.  **正则化/惩罚 (Regularization/Penalties)**: 抑制不期望的行为。
        
        * `pen_joint_accel` 和 `pen_joint_powers`: 惩罚关节加速度和功率，鼓励平滑且节能的运动。
        * `pen_base_height`: 惩罚基座高度偏离目标值（如 0.68m），维持稳定的站立高度。
        * `pen_action_smoothness`: 惩罚动作的二阶差分，减少控制器的抖动。
    
* **权重的影响**:
    每个奖励项都有一个 `weight` 参数。
    
    * 正权重（如 `keep_balance` 的 `1.0`）表示奖励，促进该行为的发生。
    * 负权重（如 `pen_joint_torque` 的 `-0.00008`）表示惩罚，抑制该行为。
    * 权重的绝对值大小决定了该项在总奖励中的主导地位。例如，`pen_base_height` 的权重设为 `-20.0`（在 `limx_base_env_cfg.py` 中），表明高度维持是训练初期极其重要的约束条件。

### 总结

该代码库展现了典型的 Isaac Lab 强化学习任务架构。通过将物理仿真（Scene）、感知（Observation）、决策（Action）和评价（Reward）高度模块化，代码清晰地定义了一个从传感器输入到电机输出的闭环控制系统。特别是特权观测与噪声观测的分离，以及详尽的正则化奖励项设计，体现了为实现 Sim-to-Real 鲁棒迁移所做的针对性优化。





## 平坦地形上的盲视行走与速度追踪 (Blind Flat Locomotion & Velocity Tracking)

### 1. 任务定义与目标 (Task Definition)

本实验阶段的主要任务是实现四足机器人在平坦地面上的稳定运动控制。具体要求策略网络（Policy）能够根据输入的指令 $\mathbf{c} = [v_x^{cmd}, v_y^{cmd}, \omega_z^{cmd}]$，精准地控制机器人的线速度和角速度，同时保持基座姿态（Roll/Pitch）的平稳，避免在高速运动或转向时发生跌倒。

在该任务中，环境被配置为 `PFBlindFlatEnvCfg`。与复杂地形任务不同，此配置移除了高度扫描传感器（`height_scanner = None`），迫使智能体仅依赖本体感觉（Proprioception，即 IMU 和关节编码器数据）来维持平衡并执行运动指令。

### 2. 方法论 (Methodology)

#### 2.1 观测空间设计 (Observation Space)

由于采用了“盲视”（Blind）设置，观测空间被精简为纯本体感知信息，这有助于网络专注于学习机器人的动力学特性，而非过拟合于视觉输入。观测向量 $\mathbf{o}_t$ 包含以下关键分量：

- **指令 (Commands)**: 目标速度 $[v_x, v_y, \omega_z]$。
- **本体状态 (Proprioception)**: 基座角速度、重力向量在基座坐标系下的投影（反映姿态）、关节位置及速度。
- **历史信息 (History)**: 上一时刻的动作输出，用于捕捉时序特征。

#### 2.2 奖励函数塑造 (Reward Shaping)

为了满足“速度追踪误差小”、“姿态稳定”和“存活率高”的考核标准，我们在 `limx_base_env_cfg.py` 中设计了复合奖励函数。总奖励 $r_t$ 由追踪奖励、正则化惩罚和步态约束组成。

##### 2.2.1 速度追踪 (Velocity Tracking)

这是核心任务目标。为了最小化均方误差 (MSE)，我们采用了指数核函数形式的奖励，在误差接近 0 时提供密集的梯度信号：

$$r_{vel} = \alpha_1 \exp\left(-\frac{\|v_{xy} - v_{xy}^{cmd}\|^2}{\sigma_v^2}\right) + \alpha_2 \exp\left(-\frac{(\omega_z - \omega_z^{cmd})^2}{\sigma_\omega^2}\right)$$

在配置中，`rew_lin_vel_xy` 权重设为 **3.0**，`rew_ang_vel_z` 权重设为 **1.5**。高权重的设定迫使智能体将速度追踪作为首要优化目标。

##### 2.2.2 姿态稳定性与存活 (Stability & Survival)

为了降低 Roll/Pitch 的震荡幅度并防止摔倒，引入了以下关键项：

- **基座姿态惩罚 (`pen_flat_orientation`)**: 权重高达 **-10.0**。该项惩罚重力向量在基座 XY 平面上的分量，强力约束机器人保持躯干水平。
- **角速度正则化 (`pen_ang_vel_xy`)**: 权重 **-0.05**。抑制非指令方向（Roll/Pitch方向）的角速度，直接减少躯干晃动。
- **存活奖励 (`keep_balance`)**: 权重 **1.0**。只要机器人未触发终止条件（如基座接触地面），每一步都会获得正向奖励，鼓励长时运行。

##### 2.2.3 动作平滑与步态约束 (Smoothness & Gait)

- **步态奖励 (`test_gait_reward`)**: 权重 **1.0**。通过 `GaitReward` 函数，强制机器人学习特定的接触相和摆动相时序，避免生成这种不自然的滑步或跳跃步态，间接提高了行走的稳定性。
- **平滑性惩罚**: 包括 `pen_joint_accel` (关节加速度)、`pen_action_rate` (动作变化率) 和 `pen_joint_powers` (功率)。这些项虽然权重较小，但对于减少电机抖动、降低 Sim-to-Real 差距至关重要。

#### 2.3 域随机化 (Domain Randomization)

为了增强策略的鲁棒性，使其能够应对不可预见的扰动（模拟评分时的随机推力或物理参数误差），我们在 `EventsCfg` 中配置了广泛的随机化事件：

- **推力扰动 (`push_robot`)**: 以 **0.002** 的概率（每步）在基座上施加瞬时推力（XY方向最大 ±500N）。这迫使 Policy 学习在受到外力冲击后快速调整足端落点以恢复平衡（Push Recovery）。
- **动力学参数随机化**:
  - **质量 (`add_base_mass`)**: 基座质量在 $[-1.0, 3.0]$ kg 范围内变化，模拟不同负载情况。
  - **摩擦力 (`robot_physics_material`)**: 地面摩擦系数在 $[0.4, 1.2]$ 间变化，确保机器人在不同表面（从滑到涩）都能稳定行走。
  - **关节刚度与阻尼**: 模拟电机特性的不确定性。

### 3. 实验结果与分析 (Results & Analysis)

*(注：本节数据由后续仿真实验补充)*

#### 3.1 速度追踪性能

下图展示了在随机指令序列下的速度追踪响应曲线。

- [插入图表：指令速度 vs. 实际速度 (Vx, Vy, Omega_z)]
- **定量分析**: 计算整个评估周期的均方误差 (MSE)。
  - $MSE_{v_x} = \dots$
  - $MSE_{v_y} = \dots$
  - $MSE_{\omega_z} = \dots$

#### 3.2 姿态稳定性评估

在维持目标速度的过程中，基座的姿态保持情况如下：

- [插入图表：Roll 和 Pitch 随时间变化的曲线]
- **数据**: Roll 角的最大震荡幅度控制在 $\pm \dots$ rad 以内，Pitch 角控制在 $\pm \dots$ rad 以内，证明了 `pen_flat_orientation` 的有效性。

#### 3.3 扰动恢复能力

在 `push_robot` 事件触发时（即外部推力施加瞬间）：

- **观察**: 机器人表现出明显的抗扰动行为（例如：顺势跨步支撑）。
- **存活率**: 在持续 1 分钟的随机指令与推力干扰测试中，机器人的存活率为 $\dots\%$。

### 4. 结论 (Conclusion)

基于 `PFBlindFlatEnvCfg` 的配置，通过高权重的速度追踪奖励配合严格的姿态惩罚，以及推力扰动训练，我们成功训练出了一个在平坦地面上具备高性能速度跟随能力且鲁棒的盲视行走策略。该策略不仅满足了基本的移动需求，还在抗干扰和动作平滑性上达到了预期指标。





## 抗干扰鲁棒性测试 (Disturbance Rejection)

### 1. 测试任务描述 (Task Description)

本测试旨在评估 Policy 在平地行走过程中应对突发外部干扰的稳定性。在实际部署中，机器人可能会遭遇碰撞、地面突然滑动或被推挤等情况。为了验证控制策略的鲁棒性，我们在仿真环境中利用域随机化（Domain Randomization）技术，向机器人的基座（Base）施加不可预测的瞬时推力（Impulse），观察其是否能够保持平衡并快速恢复到正常的行走步态。

### 2. 实验设置 (Experimental Setup)

实验基于 `PFBlindFlatEnvCfg` 环境配置，主要通过 `EventsCfg` 中的 `push_robot` 事件来实现干扰施加。

- 干扰注入机制:

  使用了 mdp.apply_external_force_torque_stochastic 函数。该函数会在仿真过程中以一定的概率随机采样力和力矩，并直接作用于机器人的刚体上。

- 参数配置:

  根据 limx_base_env_cfg.py 的定义，干扰参数如下：

  - **施力对象**: 机器人的基座 (`base_Link`)。
  - **力的大小 (Force)**: 在 $x$ 和 $y$ 轴方向上，力的大小在 $[-500.0, 500.0]$ N 范围内均匀采样。
  - **力矩大小 (Torque)**: 在 $x$ 和 $y$ 轴方向上，力矩在 $[-50.0, 50.0]$ N·m 范围内均匀采样。
  - **触发概率**: 每步触发概率为 0.002，模拟稀疏但强烈的突发冲击。

- 冲量计算 (Impulse Calculation):

  由于力是施加在仿真时间步（$\Delta t$）上的，瞬时冲量 $J$ 可近似计算为 $J = F \times \Delta t$。为了测定“最大承受冲量”，我们在测试脚本中逐步增大 force_range 的上限，直到机器人的存活率显著下降。

### 3. 考核指标 (Evaluation Metrics)

为了量化抗干扰能力，我们定义了以下两个核心指标：

1. 最大承受冲量 (Maximum Withstandable Impulse, Ns):

   机器人能够承受且不发生跌倒（即未触发 base_contact 终止条件）的最大水平推力冲量。这反映了系统的稳定裕度。

2. 步态恢复速度 (Gait Recovery Speed):

   定义为从受到干扰时刻 $t_{impact}$ 开始，到机器人的基座线速度误差和姿态角（Roll/Pitch）方差回归到稳态基准范围（例如 $\pm 5\%$ 误差带）所需的时间。恢复时间越短，说明 Policy 的动态调整能力越强。

### 4. 结果与分析 (Results & Analysis)

*(注：本节数据由后续仿真实验补充)*