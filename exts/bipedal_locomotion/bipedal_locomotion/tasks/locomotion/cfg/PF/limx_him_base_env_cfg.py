import math
from dataclasses import MISSING

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as GaussianNoise
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as UniformNoise
from bipedal_locomotion.assets.config.pointfoot_cfg import POINTFOOT_CFG
from bipedal_locomotion.tasks.locomotion.cfg.PF.limx_base_env_cfg import PFSceneCfg, CommandCfg, ActionsCfg, EventsCfg, TerminationsCfg
from bipedal_locomotion.tasks.locomotion import mdp

##############
# MDP设置 / MDP Settings
##############

@configclass
class ObservarionsCfg:
    """观测规范配置类 / Observation specifications configuration class"""
    
    @configclass
    class HIMCriticCfg(ObsGroup):
        # --- Part 1: 完全复制 HistoryObsCfg 的内容 (对应 N) ---
        # 必须包含 Commands，以此保证维度对齐
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, 
            params={"command_name": "base_velocity"}
        )
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        proj_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        last_action = ObsTerm(func=mdp.last_action)
        gait_phase = ObsTerm(func=mdp.get_gait_phase)
        gait_command = ObsTerm(func=mdp.get_gait_command, params={"command_name": "gait_command"})
        
        # --- Part 2: 紧接着必须是 GT Linear Velocity (对应切片 N:N+3) ---
        # 这是 Estimator 训练显式速度估计的 Ground Truth
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel) 

        # --- Part 3: 其他特权信息 (Privileged Info) ---
        # 这里的顺序不敏感，只要在 vel 后面即可
        heights = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")}
        )
        robot_mass = ObsTerm(func=mdp.robot_mass)
        robot_joint_stiffness = ObsTerm(func=mdp.robot_joint_stiffness)
        robot_joint_damping = ObsTerm(func=mdp.robot_joint_damping)
        robot_base_pose = ObsTerm(func=mdp.robot_base_pose)
        robot_feet_contact_force = ObsTerm(
            func=mdp.robot_feet_contact_force_current,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot_[LR]_Link")}
        )
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    
    @configclass
    class HIMHistoryObsCfg(ObsGroup):
        # 1. Commands (必须在最前面！占用索引 0-2)
        # 对应代码切片 [:, 3:N+3] 中的 "3" 是为了跳过这部分
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, 
            params={"command_name": "base_velocity"}
        )
        
        # 2. Proprioception (本体感知)
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
        gait_phase = ObsTerm(func=mdp.get_gait_phase)
        gait_command = ObsTerm(func=mdp.get_gait_command, params={"command_name": "gait_command"})
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5  # 必须与 algorithm.obs_history_len 一致
            self.flatten_history_dim = False
    
    # HIM:
    critic: HIMCriticCfg = HIMCriticCfg()
    policy: HIMHistoryObsCfg = HIMHistoryObsCfg()


@configclass
class RewardsCfg:
    """奖励项配置类 - 定义强化学习的奖励函数 / Reward terms configuration class - defines RL reward functions"""

    # 终止相关奖励 / Termination-related rewards
    keep_balance = RewTerm(
        func=mdp.stay_alive,    # 保持存活奖励 / Stay alive reward
        weight=1.0              # 奖励权重 / Reward weight
    )

    # tracking related rewards
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
    
    pen_undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,                # 不期望接触惩罚 / Undesired contacts penalty
        weight=-0.5,
        params={
            # 监控非足部的接触 / Monitor non-foot contacts
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["abad_.*", "hip_.*", "knee_.*", "base_Link"]),
            "threshold": 10.0,                      # 接触力阈值 / Contact force threshold
        },
    )

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


@configclass
class CurriculumCfg:
    """课程学习配置类 / Curriculum learning configuration class"""

    # 地形难度课程 / Terrain difficulty curriculum
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    
    lin_vel_cmd_levels = CurrTerm(func=mdp.lin_vel_cmd_levels)


########################
# 环境定义 / Environment Definition
########################


@configclass
class PFHIMBaseEnvCfg(ManagerBasedRLEnvCfg):
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
        self.episode_length_s = 20.0               # 每个episode长度20秒 / Episode length 20 seconds
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
        self.viewer.origin_type = "env"  # 相机跟随环境 / Camera follows environment
        
@configclass
class PFHIMBaseEnvCfg_PLAY(PFHIMBaseEnvCfg):
    """双足机器人基础测试环境配置 - 用于策略评估 / Base play environment configuration - for policy evaluation"""
    def __post_init__(self):
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 32
        
        self.episode_length_s = 100.0

        # disable randomization for play
        self.observations.policy.enable_corruption = True
        # remove random pushing event
        self.events.push_robot = None
        # remove random base mass addition event
        self.events.add_base_mass = None
        
        self.curriculum.lin_vel_cmd_levels=None
