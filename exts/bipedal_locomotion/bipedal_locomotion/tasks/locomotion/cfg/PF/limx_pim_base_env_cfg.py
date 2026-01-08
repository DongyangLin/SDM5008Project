import math

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as GaussianNoise
from bipedal_locomotion.assets.config.pointfoot_cfg import POINTFOOT_CFG
from bipedal_locomotion.tasks.locomotion.cfg.PF.limx_base_env_cfg import PFSceneCfg, ActionsCfg, TerminationsCfg, EventsCfg
from bipedal_locomotion.tasks.locomotion import mdp


##############
# MDP设置 / MDP Settings
##############


@configclass
class CommandCfg:
    """命令规范配置类 / Command specifications configuration class"""

    # 步态命令配置 / Gait command configuration
    gait_command = mdp.UniformGaitCommandCfg(
        resampling_time_range=(5.0, 5.0),  # 命令重采样时间范围 (固定5秒) / Command resampling time range (fixed 5s)
        debug_vis=False,                    # 不显示调试可视化 / No debug visualization
        ranges=mdp.UniformGaitCommandCfg.Ranges(
            frequencies=(2.0, 2.5),       # 步态频率范围 [Hz] / Gait frequency range [Hz]
            offsets=(0.5, 0.5),           # 相位偏移范围 [0-1] / Phase offset range [0-1]
            durations=(0.5, 0.5),         # 接触持续时间范围 [0-1] / Contact duration range [0-1]
            # swing_height=(0.1, 0.2)     # 摆动高度范围 [m] / Swing height range [m]
        ),
    )
    
    # 基座速度命令配置 / Base velocity command configuration
    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(0.0, 5.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        # heading_control_stiffness = 1.0,
        debug_vis=False,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.5, 0.5)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(    # 限制范围 / Limit ranges
            lin_vel_x=(-1.5, 1.5), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-0.5, 0.5)
            # lin_vel_x=(0.7, 0.7), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
        ),
    )


@configclass
class ObservarionsCfg:
    """观测规范配置类 / Observation specifications configuration class"""
    
    @configclass
    class PIMCriticCfg(ObsGroup):
        """PIM评论家观测组配置类 / PIM critic observation group configuration class"""

        # 本体感知相关观测 / Proprioception related observations
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, 
            params={"command_name": "base_velocity"}
        )
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        proj_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        last_action = ObsTerm(func=mdp.last_action)
        
        # 机器人基座线速度（真值） / Robot base linear velocity (ground truth)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel) 

        # 环境感知相关观测 / Environment perception related observations
        robot_mass = ObsTerm(func=mdp.robot_mass)                      # 机器人质量观测 / Robot mass observation
        robot_joint_stiffness = ObsTerm(func=mdp.robot_joint_stiffness)    # 机器人关节刚度观测 / Robot joint stiffness observation
        robot_joint_damping = ObsTerm(func=mdp.robot_joint_damping)        # 机器人关节阻尼观测 / Robot joint damping observation
        robot_base_pose = ObsTerm(func=mdp.robot_base_pose)            # 机器人基座位姿观测 / Robot base pose observation
        robot_feet_contact_force = ObsTerm(                            # 机器人足部接触力观测 / Robot feet contact force observation
            func=mdp.robot_feet_contact_force_current,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot_[LR]_Link")}
        )
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    
    @configclass
    class PIMHistoryObsCfg(ObsGroup):
        """PIM历史观测组配置类  / PIM history observation group configuration class"""

        # 速度命令观测 / Velocity command observation
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, 
            params={"command_name": "base_velocity"}
        )
        
        # 本体感知观测 / Proprioception observations
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, 
            noise=GaussianNoise(mean=0.0, std=0.05),
            clip=(-100.0, 100.0),
            scale=0.25
        )
        proj_gravity = ObsTerm(
            func=mdp.projected_gravity, 
            noise=GaussianNoise(mean=0.0, std=0.025),
            clip=(-100.0, 100.0),
            scale=1.0
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, 
            noise=GaussianNoise(mean=0.0, std=0.01),
            clip=(-100.0, 100.0),
            scale=1.0
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel, 
            noise=GaussianNoise(mean=0.0, std=0.01),
            clip=(-100.0, 100.0),
            scale=0.05
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5  # 必须与 algorithm.obs_history_len 一致
            self.flatten_history_dim = False
    
    @configclass
    class PIMHeightScanObsCfg(ObsGroup):
        """高度扫描观测组配置 - 包含来自高度扫描传感器的信息 / Height scan observation group - includes information from height scanner sensor"""
        
        # 高度扫描观测 / Height scan observation
        heights = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")}
        )
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 1
            self.flatten_history_dim = False

    # PIM 观测组实例化 / PIM observation group instantiation
    critic: PIMCriticCfg = PIMCriticCfg()
    policy: PIMHistoryObsCfg = PIMHistoryObsCfg()
    perceptive: PIMHeightScanObsCfg = PIMHeightScanObsCfg()


@configclass
class RewardsCfg:
    """奖励项配置类 - 定义强化学习的奖励函数 / Reward terms configuration class - defines RL reward functions"""

    # 终止相关奖励 / Termination-related rewards
    keep_balance = RewTerm(
        func=mdp.stay_alive,    # 保持存活奖励 / Stay alive reward
        weight=1.0              # 奖励权重 / Reward weight
    )

    # 速度跟踪奖励 / Velocity tracking rewards
    rew_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=3.0, params={"command_name": "base_velocity", "std": math.sqrt(0.2)}
    )
    rew_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=1.5, params={"command_name": "base_velocity", "std": math.sqrt(0.2)}
    )

    # 调节相关奖励 / Regulation-related rewards
    pen_base_height = RewTerm(
        func=mdp.base_com_height,                   # 基座高度惩罚 / Base height penalty
        params={"target_height": 0.78},            # 目标高度 78cm / Target height 78cm
        weight=-20.0,                               # 负权重表示惩罚 / Negative weight indicates penalty
    )
    
    # 关节相关惩罚 / Joint-related penalties
    pen_lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    pen_ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    pen_joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-0.00008)
    pen_joint_accel = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-07)
    pen_action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.03)
    pen_joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    pen_joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-1e-03)
    pen_joint_powers = RewTerm(func=mdp.joint_powers_l1, weight=-5e-04)
    
    # 足部和接触相关惩罚 / Foot and contact-related penalties
    pen_undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,                # 不期望接触惩罚 / Undesired contacts penalty
        weight=-0.5,
        params={
            # 监控非足部的接触 / Monitor non-foot contacts
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["abad_.*", "hip_.*", "knee_.*", "base_Link"]),
            "threshold": 10.0,                      # 接触力阈值 / Contact force threshold
        },
    )

    # 足部和姿态相关惩罚 / Foot and posture-related penalties
    pen_action_smoothness = RewTerm(
        func=mdp.ActionSmoothnessPenalty,           # 动作平滑性惩罚 / Action smoothness penalty
        weight=-0.04
    )

    pen_flat_orientation = RewTerm(
        func=mdp.flat_orientation_l2,               # 平坦朝向L2惩罚 / Flat orientation L2 penalty
        weight=-10.0
    )

    pen_feet_distance = RewTerm(
        func=mdp.feet_distance,                     # 足部距离惩罚 / Foot distance penalty
        weight=-10,
        params={
            "min_feet_distance": 0.12,            # 最小足部距离 / Minimum foot distance
            "feet_links_name": ["foot_[RL]_Link"]  # 足部连杆名称 / Foot link names
        }
    )
    
    pen_feet_regulation = RewTerm(
        func=mdp.feet_regulation,                   # 足部调节惩罚 / Foot regulation penalty
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["foot_[RL]_Link"]),
            "base_height_target": 0.65,            # 基座目标高度 / Base target height
            "foot_radius": 0.03                    # 足部半径 / Foot radius
        },
    )

    foot_landing_vel = RewTerm(
        func=mdp.foot_landing_vel,                  # 足部着陆速度惩罚 / Foot landing velocity penalty
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["foot_[RL]_Link"]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["foot_[RL]_Link"]),
            "foot_radius": 0.03,
            "about_landing_threshold": 0.08         # 即将着陆阈值 / About to land threshold
        },
    )
    
    # 步态奖励 / Gait reward
    test_gait_reward = RewTerm(
        func=mdp.GaitReward,                        # 步态奖励函数 / Gait reward function
        weight=1.0,
        params={
            "tracking_contacts_shaped_force": -2.0,    # 接触力跟踪形状参数 / Contact force tracking shaping
            "tracking_contacts_shaped_vel": -2.0,      # 接触速度跟踪形状参数 / Contact velocity tracking shaping
            "gait_force_sigma": 25.0,                  # 步态力标准差 / Gait force sigma
            "gait_vel_sigma": 0.25,                    # 步态速度标准差 / Gait velocity sigma
            "kappa_gait_probs": 0.05,                  # 步态概率参数 / Gait probability parameter
            "command_name": "gait_command",            # 命令名称 / Command name
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="foot_.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names="foot_.*"),
        },
    )

    # 足部离地高度奖励 / Foot clearance reward
    rew_feet_clearance = RewTerm(
        func=mdp.foot_clearance_reward, 
        weight=0.5, 
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": [0.10, 0.20], # p_z_target approx
            "asset_cfg": SceneEntityCfg("robot", body_names=".*foot_[LR]_Link"), # Regex for feet
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "foot_radius": 0.03,
        }
    )


@configclass
class CurriculumCfg:
    """课程学习配置类 / Curriculum learning configuration class"""

    # 地形难度课程 / Terrain difficulty curriculum
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    
    # 线速度命令水平课程 / Linear velocity command level curriculum
    lin_vel_cmd_levels = CurrTerm(func=mdp.lin_vel_cmd_levels)


########################
# 环境定义 / Environment Definition
########################


@configclass
class PFPIMBaseEnvCfg(ManagerBasedRLEnvCfg):
    """测试环境配置类 / Test environment configuration class"""

    # 场景设置 / Scene settings
    scene: PFSceneCfg = PFSceneCfg(num_envs=4096, env_spacing=2.5)

    # 基本设置 / Basic settings
    observations: ObservarionsCfg = ObservarionsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandCfg = CommandCfg()

    # MDP设置 / MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """后初始化配置 / Post-initialization configuration"""
        
        self.decimation = 4                         # 控制频率降采样 (50Hz -> 12.5Hz) / Control frequency downsampling
        self.episode_length_s = 20.0                # 每个episode长度20秒 / Episode length 20 seconds
        self.sim.render_interval = 2 * self.decimation  # 渲染间隔 / Rendering interval
        
        # 仿真设置 / Simulation settings
        self.sim.dt = 0.005                        # 仿真时间步 5ms / Simulation timestep 5ms
        self.seed = 42                             # 随机种子 / Random seed
        
        # 更新传感器更新周期 / Update sensor update periods
        # 基于最小更新周期(物理更新周期)来同步所有传感器 / Sync all sensors based on smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        
        self.scene.robot = POINTFOOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.joint_pos = {
            "abad_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "knee_R_Joint": 0.0,
        }

        # 调整基座质量随机化参数 / Adjust base mass randomization parameters
        self.events.add_base_mass.params["asset_cfg"].body_names = "base_Link"
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 2.0)

        # 设置基座接触终止条件 / Set base contact termination condition
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base_Link"
        
        # 更新视口相机设置 / Update viewport camera settings
        self.viewer.origin_type = "env"       # 相机跟随环境 / Camera follows environment


@configclass
class PFPIMBaseEnvCfg_PLAY(PFPIMBaseEnvCfg):
    """双足机器人基础测试环境配置 - 用于策略评估 / Base play environment configuration - for policy evaluation"""

    def __post_init__(self):
        super().__post_init__()

        # 选取较小环境便于观测 / Select smaller environment for easier observation
        self.scene.num_envs = 32
        self.episode_length_s = 100.0

        # 禁用策略评估时的随机化 / Disable randomization for play
        self.observations.policy.enable_corruption = True

        # 移除随机推力事件 / Remove random pushing event
        self.events.push_robot = None

        # 移除质量随机化事件 / Remove mass randomization event
        self.events.add_base_mass = None
        
        # 移除速度难度课程 / Remove velocity difficulty curriculum
        self.curriculum.lin_vel_cmd_levels=None
