# **双足机器人强化学习运动控制 / Bipedal Robot RL Locomotion Control with Isaac Lab**

---

## **概述 / Overview**

该仓库基于 [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab) 仿真平台，对双足点足机器人（Point-foot Biped）[limxdynamics TRON1](https://www.limxdynamics.com/en/tron1) 进行基于强化学习的运动控制。
我们系统性地完成了从环境配置、奖励函数设计、策略训练到鲁棒性测试的完整流程，在双足机器人运动控制上运用了 HIM（Hybrid Internal Model）与 PIM（Perceptive Internal Model）算法，并对比和评估了两种策略架构在双足行走任务中的表现差异。

项目目标包括：
- 在平地环境中实现稳定、可控的双足行走；
- 实现对期望速度指令的精准跟踪；
- 通过随机外力扰动测试策略的抗干扰与恢复能力；
- 在复杂地形中测试策略的适应能力，如平地、上下楼梯、斜坡、障碍物、粗糙地面等；
- 对多种运动控制算法进行定性与定量比较（HIM 和 PIM）。


**关键词 / Keywords:** Isaac Lab, TRON1，Bipedal Locomotion, Reinforcement Learning, PPO, HIM, PIM, Robust Control



---

## 1. 项目文件结构 / Project File Structure

```
SDM5008Project/                         # 项目根目录
├── README.md                           # 项目说明文档方
├── exts/
│   └── bipedal_locomotion/             # 主目录
│       ├── bipedal_locomotion/
│       │   ├── assets/                 # 机器人与环境相关资源
│       │   │   ├── config/             # 机器人结构与物理仿真参数配置
│       │   │   │   ├── pointfoot_cfg.py    # 点足机器人（Point-Foot）
│       │   │   │   ├── solefoot_cfg.py     # 平足机器人（Sole-Foot）
│       │   │   │   └── wheelfoot_cfg.py    # 轮足机器人（Wheel-Foot）
│       │   │   └── usd/                # USD 格式的机器人模型与场景资源
│       │   ├── tasks/                  # 任务定义模块
│       │   │   └── locomotion/          # 行走/运动任务
│       │   │       ├── agents/          # 强化学习智能体配置
│       │   │       │   └── limx_rsl_rl_ppo_cfg.py
│       │   │       │       # 基于 RSL-RL 的 PPO 算法超参数配置文件
│       │   │       ├── cfg/             # 环境与任务配置
│       │   │       │   ├── PF/           # Point-Foot 机器人任务配置
│       │   │       │   │   ├── limx_base_env_cfg.py    # 基础行走环境配置（观测、动作空间等）
│       │   │       │   │   ├── limx_him_base_env_cfg.py    # HIM（Hybrid Internal Model）
│       │   │       │   │   ├── limx_pim_base_env_cfg.py    # PIM（Pure Internal Model）
│       │   │       │   │   └── terrains_cfg.py         # 地形参数配置（平地、台阶等）
│       │   │       │   ├── SF/           # Sole-Foot 机器人任务配置
│       │   │       │   └── WF/           # Wheel-Foot 机器人任务配置
│       │   │       ├── mdp/             # MDP（马尔可夫决策过程）定义
│       │   │       │   ├── commands/         # 速度、方向等高层指令定义
│       │   │       │   ├── curriculums.py    # 课程学习策略（逐步增加任务难度）
│       │   │       │   ├── events.py         # 随机扰动与事件触发（推搡、随机重置等）
│       │   │       │   ├── observations.py   # 观测量定义（状态、历史信息等）
│       │   │       │   └── rewards.py        # 奖励函数设计（稳定性、能耗、跟踪误差）
│       │   │       └── robots/           # 机器人环境封装
│       │   │           ├── __init__.py
│       │   │           ├── limx_pointfoot_env_cfg.py    # 点足机器人环境定义
│       │   │           ├── limx_solefoot_env_cfg.py     # 平足机器人环境定义
│       │   │           └── limx_wheelfoot_env_cfg.py    # 轮足机器人环境定义
│       │   └── utils/                   # 工具函数
│       │       ├── random_lag_actuator.py    # 随机执行器时滞建模，提高策略鲁棒性
│       │       └── wrappers/
│       │           └── rsl_rl/
│       │               ├── him_exporter.py   # HIM 模型导出工具
│       │               ├── pim_exporter.py   # PIM 模型导出工具
│       │               └── rl_mlp_cfg.py     # 多层感知机（MLP）网络结构配置
│       ├── config/                      # 扩展级别的全局配置
│       ├── docs/                        # 项目文档（设计说明、实验说明等）
│       └── setup.py                     # Isaac Lab 扩展安装脚本
├── logs/                               # 训练与测试日志
│   └── rsl_rl/
│       ├── pf_him_stair/                # 点足 + HIM + 复杂地形实验日志
│       ├── pf_pim_stair/                # 点足 + PIM + 复杂地形实验日志
│       └── pf_tron_1a_flat/             # 点足 + Tron 模型 + 平地实验日志
├── rsl_rl/                             # 强化学习算法实现（基于 RSL-RL）
│   ├── licenses/
│   ├── rsl_rl/
│   │   ├── algorithm/                  # 强化学习算法
│   │   │   ├── him_ppo.py               # HIM-PPO 算法实现
│   │   │   ├── pim_ppo.py               # PIM-PPO 算法实现
│   │   │   └── ppo.py                   # 标准 PPO 算法基类
│   │   ├── env/
│   │   │   └── vec_env.py               # 向量化环境接口
│   │   ├── modules/                    # 网络与模型模块
│   │   │   ├── actor_critic.py          # Actor-Critic 基础结构
│   │   │   ├── him_actor_critic.py      # HIM Actor-Critic 网络
│   │   │   ├── him_estimator.py         # HIM 状态估计器
│   │   │   ├── mlp_encoder.py           # MLP 编码器
│   │   │   ├── pim_actor_critic.py      # PIM Actor-Critic 网络
│   │   │   └── pim_estimator.py         # PIM 状态估计器
│   │   ├── runner/                     # 训练流程控制
│   │   │   ├── him_on_policy_runner.py  # HIM 在线策略训练流程
│   │   │   ├── on_policy_runner.py      # 通用 PPO 训练流程
│   │   │   └── pim_on_policy_runner.py  # PIM 在线策略训练流程
│   │   ├── storage/                    # 轨迹与数据存储
│   │   │   ├── him_rollout_storage.py   # HIM 数据缓存
│   │   │   ├── pim_rollout_storage.py   # PIM 数据缓存
│   │   │   └── rollout_storage.py       # 通用 rollout 缓存
│   │   └── utils/
│   └── setup.py                         # RSL-RL 包安装脚本
└── scripts/                            # 实验运行脚本
    └── rsl_rl/
        ├── cli_args.py                 # 命令行参数定义
        ├── play.py                     # 使用训练好的模型进行测试
        ├── play_him.py                 # HIM 模型测试脚本
        ├── play_pim.py                 # PIM 模型测试脚本
        ├── plot_data.py                # 训练曲线与指标可视化
        ├── train.py                    # 模型训练主脚本
        └── verify_exported_model.py    # 导出模型正确性验证
```


---

## **2. 环境配置与安装 / Environment Setup & Installation**

请参考 IsaacLab 官方文档：[IsaacLab Installation](https://isaac-sim.github.io/IsaacLab/v2.1.0/source/setup/installation/pip_installation.html)

- 新建 Conda 环境
  ```bash
    conda create -n env_isaaclab python=3.10
    conda activate env_isaaclab 
  ```
- 安装 PyTorch
  ```bash
    pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
  ```
- 安装 IsaacSim
  ```bash
    pip install --upgrade pip
    pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
  ```
- 安装 IsaacLab
  - 克隆 IsaacLab 仓库
    ```bash
      git clone -b v2.1.0 https://github.com/isaac-sim/IsaacLab.git
    ```
  - 安装依赖
    ```bash
      sudo apt install cmake build-essential
    ```
  - 安装 IsaacLab
    ```bash
      cd IsaacLab
      ./isaaclab.sh --install
    ```
- 克隆并安装仓库
  请将仓库克隆到 `IsaacLab` 文件夹之外！
  ```bash
    git clone https://github.com/DongyangLin/SDM5008Project.git
    cd [repository name]
    python -m pip install -e exts/bipedal_locomotion
  ```
  为了使用 `MLP` 分支，需要卸载原生 `rsl_rl` 库，并安装新的 `rsl_rl` 库：
  ```bash
    pip uninstall rsl-rl-lib
    cd bipedal_locomotion_isaaclab/rsl_rl
    python -m pip install -e .
  ```

---

## **3. 训练与测试 / Training & Evaluation**

### 3.1 训练机器人 / Training

- Encoder-MLP 算法
  使用 `scripts/rsl_rl/train.py` 脚本直接训练机器人，指定任务：
  Use the `scripts/rsl_rl/train.py` script to train the robot directly, specifying the task:
  ```bash
    python scripts/rsl_rl/train.py --task=Isaac-Limx-PF-Blind-Flat-v0 --headless
  ```

  以下参数可用于自定义训练：
  The following arguments can be used to customize the training:
    * --headless: 以无渲染模式运行仿真 / Run the simulation in headless mode
    * --num_envs: 要运行的并行环境数量 / Number of parallel environments to run
    * --max_iterations: 最大训练迭代次数 / Maximum number of training iterations
    * --save_interval: 保存模型的间隔 / Interval to save the model
    * --task: 选择任务名称 / Name of the task
    * --seed: 随机数生成器的种子 / Seed for the random number generator
    * --checkpoint_path: 训练起始点模型的相对路径 / Relative path to checkpoint file
  
- HIM 算法
  在 `scripts/rsl_rl/train.py` 中注释掉 `runner = OnPolicyRunner(...)`(131行) 并选择 `runner = HIMOnPolicyRunner(...)`(134行) 取消注释，使用 `scripts/rsl_rl/train.py` 脚本训练：
  ```bash
    python scripts/rsl_rl/train.py --task=Isaac-Limx-PF-Stair-HIM-v0 --headless
  ```

- PIM 算法
  在 `scripts/rsl_rl/train.py` 中注释掉 `runner = OnPolicyRunner(...)`(131行) 并选择 `runner = PIMOnPolicyRunner(...)`(137行) 取消注释，使用 `scripts/rsl_rl/train.py` 脚本训练：
  ```bash
    python scripts/rsl_rl/train.py --task=Isaac-Limx-PF-Stair-PIM-v0 --headless
  ```

### 3.2 运行训练好的模型 / Playing a trained model

- 运行训练好的 Encoder-MLP 模型：
  To play a trained Encoder-MLP model:
  ```bash
    python scripts/rsl_rl/play.py --task=Isaac-Limx-PF-Blind-Flat-Play-v0 --num_envs=50 --checkpoint_path=logs/rsl_rl/pf_tron_1a_flat/2025-12-15_16-38-07/model_3000.pt
  ```
  以下参数可用于自定义运行：
  The following arguments can be used to customize the playing:
    * --task: 选择任务名称 / Name of the task
    * --num_envs: 要运行的并行环境数量 / Number of parallel environments to run
    * --headless: 以无头模式运行仿真 / Run the simulation in headless mode
    * --checkpoint_path: 要加载的模型路径 / Path to the checkpoint to load

- 运行训练好的 HIM 模型：
  To play a trained HIM model:
  ```bash
    python scripts/rsl_rl/play_him.py --task=Isaac-Limx-PF-Stair-HIM-Play-v0 --num_envs=50 --checkpoint_path=logs/rsl_rl/pf_him_stair/2025-12-16_Stable_Phase_3/model_5000.pt
  ```

- 运行训练好的 PIM 模型：
  To play a trained PIM model:
  ```bash
    python scripts/rsl_rl/play_pim.py --task=Isaac-Limx-PF-Stair-PIM-Play-v0 --num_envs=50 --checkpoint_path=logs/rsl_rl/pf_pim_stair/2025-12-17_09-56-22/model_11000.pt
  ```

---

## **4. 视频演示 / Video Demonstration**

### 4.1 速度跟随 / Velocity Tracking

- **点足盲目平地 / Pointfoot Blind Flat**:
  ![Flat Velocity Tracking](./media/Flat_ground_velocity_tracking.gif)

### 4.2 抗扰测试 / Disturbance Rejection

- **HIM 抗扰测试 / HIM Disturbance Rejection**
  ![HIM Disturbance Rejection](./media/HIM_disturbance_rejection.gif)

- **PIM 抗扰测试 / PIM Disturbance Rejection**
  ![PIM Disturbance Rejection](./media/PIM_disturbance_rejection.gif)

### 4.3 地形适应 / Terrain Traversal

- **HIM 地形适应 / HIM Terrain Traversal**
  | ![HIM Stairs](./media/HIM_stairs.gif) | ![HIM Slopes](./media/HIM_slopes.gif) |
  | :---: | :---: |
  | **![HIM Obstacles](./media/HIM_obstacles.gif)** | **![HIM Rough](./media/HIM_rough.gif)** |

- **PIM 地形适应 / PIM Terrain Traversal**
  | ![PIM Stairs](./media/PIM_stairs.gif) | ![PIM Slopes](./media/PIM_slopes.gif) |
  | :---: | :---: |
  | **![PIM Obstacles](./media/PIM_obstacles.gif)** | **![PIM Rough](./media/PIM_rough.gif)** |
  | **![PIM Obstacles](./media/PIM_edge.gif)** |



## 致谢 / Acknowledgements

本项目使用以下开源库：
This project uses the following open-source libraries:
- [IsaacLabExtensionTemplate](https://github.com/isaac-sim/IsaacLabExtensionTemplate)
- [bipedal_locomotion_isaaclab](https://github.com/Andy-xiong6/bipedal_locomotion_isaaclab)
- [tron1-rl-isaaclab](https://github.com/limxdynamics/tron1-rl-isaaclab.git)
- [HIMLoco](https://github.com/InternRobotics/HIMLoco.git)

**贡献者 / Contributors:**
- Zikun Zhuang
- Zhiyu Wang

