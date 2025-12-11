import math

from isaaclab.utils import configclass

from bipedal_locomotion.assets.config.pointfoot_cfg import POINTFOOT_CFG
from bipedal_locomotion.tasks.locomotion.cfg.PF.limx_base_env_cfg import PFEnvCfg
from bipedal_locomotion.tasks.locomotion.cfg.PF.limx_pim_base_env_cfg import PFPIMBasedEnvCfg, PFPIMBaseEnvCfg_PLAY
from bipedal_locomotion.tasks.locomotion.cfg.PF.terrains_cfg import (
    BLIND_ROUGH_TERRAINS_CFG,
    BLIND_ROUGH_TERRAINS_PLAY_CFG,
    STAIRS_TERRAINS_CFG,
    STAIRS_TERRAINS_PLAY_CFG,
    HIM_TERRAINS_CFG,
    HIM_PLAY_TERRAINS_CFG,
)

from isaaclab.sensors import RayCasterCfg, patterns, OffsetCfg
from bipedal_locomotion.tasks.locomotion import mdp
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as GaussianNoise
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm


######################
# 双足机器人基础环境 / Pointfoot Base Environment
######################


@configclass
class PFBaseEnvCfg(PFEnvCfg):
    """双足机器人基础环境配置 - 所有变体的共同基础 / Base environment configuration for pointfoot robot - common foundation for all variants"""
    def __post_init__(self):
        super().__post_init__()

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
class PFBaseEnvCfg_PLAY(PFBaseEnvCfg):
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


############################
# 双足机器人盲视平地环境 / Pointfoot Blind Flat Environment
############################


@configclass
class PFBlindFlatEnvCfg(PFBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = None
        self.observations.policy.heights = None
        self.observations.critic.heights = None

        self.curriculum.terrain_levels = None


@configclass
class PFBlindFlatEnvCfg_PLAY(PFBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.height_scanner = None
        self.observations.policy.heights = None
        self.observations.critic.heights = None

        self.curriculum.terrain_levels = None


#############################
# 双足机器人盲视粗糙环境 / Pointfoot Blind Rough Environment
#############################


@configclass
class PFBlindRoughEnvCfg(PFBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = None
        self.observations.policy.heights = None
        self.observations.critic.heights = None

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = BLIND_ROUGH_TERRAINS_CFG


@configclass
class PFBlindRoughEnvCfg_PLAY(PFBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.height_scanner = None
        self.observations.policy.heights = None
        self.observations.critic.heights = None

        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = BLIND_ROUGH_TERRAINS_PLAY_CFG


##############################
# 双足机器人盲视楼梯环境 / Pointfoot Blind Stairs Environment
##############################


@configclass
class PFBlindStairEnvCfg(PFBaseEnvCfg):
    """盲视楼梯环境配置 - 专门训练爬楼梯能力 / Blind stairs environment configuration - specialized for stair climbing training"""
    
    def __post_init__(self):
        """后初始化 - 配置楼梯训练环境 / Post-initialization - configure stairs training environment"""
        super().__post_init__()
        
        # 移除视觉组件 / Remove vision components
        self.scene.height_scanner = None
        self.observations.policy.heights = None
        self.observations.critic.heights = None

        # 调整速度命令范围以适应楼梯环境 / Adjust velocity command ranges for stairs environment
        self.commands.base_velocity.ranges.lin_vel_x = (0.2, 0.6)      # 前进速度：0.5-1.0 m/s / Forward velocity: 0.5-1.0 m/s
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)     # 横向速度：0（仅直行）/ Lateral velocity: 0 (straight only)
        self.commands.base_velocity.ranges.ang_vel_z = (-math.pi / 6, math.pi / 6)  # 转向：±30度 / Turning: ±30 degrees

        # 调整奖励权重以适应楼梯爬升 / Adjust reward weights for stair climbing
        self.rewards.rew_lin_vel_xy.weight = 2.0          # 增加线速度跟踪奖励 / Increase linear velocity tracking reward
        self.rewards.rew_ang_vel_z.weight = 1.5           # 增加角速度跟踪奖励 / Increase angular velocity tracking reward
        self.rewards.pen_lin_vel_z.weight = -0.1 # -1.0 ！！！！！！！！！！！
        # self.rewards.pen_lin_vel_z.weight = -1.0          # 增加Z方向速度惩罚 / Increase Z velocity penalty
        self.rewards.pen_ang_vel_xy.weight = -0.05        # XY角速度惩罚 / XY angular velocity penalty
        self.rewards.pen_action_rate.weight = -0.01       # 动作变化率惩罚 / Action rate penalty
        self.rewards.pen_flat_orientation.weight = -2.5   # 姿态保持惩罚 / Orientation keeping penalty
        self.rewards.pen_undesired_contacts.weight = -1.0 # 不期望接触惩罚 / Undesired contact penalty

        # 设置楼梯地形 / Set up stairs terrain
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = STAIRS_TERRAINS_CFG

@configclass
class PFBlindStairEnvCfg_PLAY(PFBaseEnvCfg_PLAY):
    """盲视楼梯测试环境配置 / Blind stairs play environment configuration"""
    
    def __post_init__(self):
        """后初始化 - 配置楼梯测试环境 / Post-initialization - configure stairs testing environment"""
        super().__post_init__()
        
        # 移除视觉组件 / Remove vision components
        self.scene.height_scanner = None
        self.observations.policy.heights = None
        self.observations.critic.heights = None

        # 设置测试专用的速度命令 / Set testing-specific velocity commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)    # 固定前进速度范围 / Fixed forward velocity range
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)   # 无横向移动 / No lateral movement
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)   # 无转向 / No turning

        # 固定重置姿态（无偏航角变化）/ Fixed reset pose (no yaw variation)
        self.events.reset_robot_base.params["pose_range"]["yaw"] = (-0.0, 0.0)

        # 设置测试楼梯地形 / Set up testing stairs terrain
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        # 设置中等难度的楼梯测试环境 / Set medium difficulty stairs testing environment
        self.scene.terrain.terrain_generator = STAIRS_TERRAINS_PLAY_CFG.replace(difficulty_range=(0.5, 0.5))


#############################
# 带高度扫描的双足机器人楼梯环境 / Pointfoot Stairs Environment with Height Scanning
#############################

@configclass
class PFStairEnvCfgv1(PFBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_Link",
            attach_yaw_only=True,
            offset = OffsetCfg(pos=(0, 0, 20.0)),
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.0, 0.5]), #TODO: adjust size to fit real robot
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.observations.policy.heights = ObsTerm(func=mdp.height_scan,
            params = {"sensor_cfg": SceneEntityCfg("height_scanner"),
                      "offset":0.78}, # the defualt height of robot is 0.78m
                    noise=GaussianNoise(mean=0.0, std=0.01),
                    clip = (-2.0, 2.0),
        )
        self.observations.critic.heights = ObsTerm(func=mdp.height_scan,
            params = {"sensor_cfg": SceneEntityCfg("height_scanner"),
                      "offset":0.78}, # the defualt height of robot is 0.78m
            clip = (-2.0, 2.0),
        )
        
        # 调整速度命令范围以适应楼梯环境 / Adjust velocity command ranges for stairs environment
        self.commands.base_velocity.ranges.lin_vel_x = (-0.2, 1.0)      
        self.commands.base_velocity.ranges.lin_vel_y = (-0.2, 0.2)     
        self.commands.base_velocity.ranges.ang_vel_z = (-math.pi / 6, math.pi / 6) 
        
        self.rewards.pen_base_height.params["sensor_cfg"] = SceneEntityCfg("height_scanner") # add height scanner to fit stairs height
        
        self.rewards.pen_lin_vel_z.weight = -0.1 # allow the robot to climb up and down
        
        self.rewards.pen_feet_distance.weight = -100.0
        
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = STAIRS_TERRAINS_CFG


@configclass
class PFStairEnvCfgv1_PLAY(PFBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_Link",
            attach_yaw_only=True,
            offset = OffsetCfg(pos=(0, 0, 20.0)),
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.0, 0.5]), #TODO: adjust size to fit real robot
            debug_vis=True,
            mesh_prim_paths=["/World/ground"],
        )
        self.observations.policy.heights = ObsTerm(func=mdp.height_scan,
            params = {"sensor_cfg": SceneEntityCfg("height_scanner"),
                      "offset":0.78}, # the defualt height of robot is 0.78m
                    noise=GaussianNoise(mean=0.0, std=0.01),
                    clip = (-2.0, 2.0),
        )
        self.observations.critic.heights = ObsTerm(func=mdp.height_scan,
            params = {"sensor_cfg": SceneEntityCfg("height_scanner"),
                      "offset":0.78}, # the defualt height of robot is 0.78m
            clip = (-2.0, 2.0),
        )
        
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        
        self.commands.base_velocity.ranges.lin_vel_x = (0.4, 0.5)      
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)     
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0) 

        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = STAIRS_TERRAINS_PLAY_CFG.replace(difficulty_range=(0.5, 0.5))
        

##############################
# HIM双足机器人盲视楼梯环境 / Pointfoot Blind Stairs Environment
##############################      

@configclass
class PFHIMEnvCfg(PFBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        # ... (Terrain, Height Scanner, Observations, Commands setup remains the same) ...
        # 1. 地形与课程设置
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = HIM_TERRAINS_CFG

        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_Link",
            attach_yaw_only=True,
            offset = OffsetCfg(pos=(0, 0, 20.0)),
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.0, 1.0]), 
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.observations.policy.heights = None
        self.observations.critic.heights = ObsTerm(func=mdp.height_scan,
            params = {"sensor_cfg": SceneEntityCfg("height_scanner"),
                      "offset":0.68}, 
            clip = (-2.0, 2.0),
        )
        
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        
        self.commands.base_velocity.limit_ranges.lin_vel_x = (-0.2, 1.0)      
        self.commands.base_velocity.limit_ranges.lin_vel_y = (-0.2, 0.2)     
        self.commands.base_velocity.limit_ranges.ang_vel_z = (-math.pi / 6, math.pi / 6)
        self.commands.base_velocity.ranges.lin_vel_x=(-0.5, 0.5) # higher vel at beginning
        # self.commands.gait_command = None
        
        self.curriculum.terrain_levels.func = mdp.terrain_levels_vel_constrained

        # =========================================================================
        # REWARD MODIFICATIONS (Strictly following HIM Table 5)
        # =========================================================================

        # 1. Linear velocity tracking 
        # Eq: exp(-|v_cmd - v_xy|^2 / sigma)
        # Weight: 1.0
        self.rewards.rew_lin_vel_xy.weight = 1.0
        self.rewards.rew_lin_vel_xy.params["std"] = 0.25 # sigma = 0.25

        # 2. Angular velocity tracking 
        # Eq: exp(-|w_cmd - w_yaw|^2 / sigma)
        # Weight: 0.5
        self.rewards.rew_ang_vel_z.weight = 0.5
        self.rewards.rew_ang_vel_z.params["std"] = 0.25 # sigma = 0.25

        # 3. Linear velocity (z) 
        # Eq: v_z^2
        # Weight: -2.0
        self.rewards.pen_lin_vel_z.weight = -2.0

        # 4. Angular velocity (xy) 
        # Eq: |w_xy|^2
        # Weight: -0.05
        self.rewards.pen_ang_vel_xy.weight = -0.05

        # 5. Orientation 
        # Eq: |g_proj|^2 (approx via flat_orientation_l2)
        # Weight: -0.2
        self.rewards.pen_flat_orientation.weight = -0.2

        # 6. Joint accelerations 
        # Eq: |theta_ddot|^2
        # Weight: -2.5e-7
        self.rewards.pen_joint_accel.weight = -2.5e-7

        # 7. Joint power 
        # Eq: |tau * theta_dot|
        # Weight: -2e-5
        self.rewards.pen_joint_powers.weight = -2e-5

        # 8. Body height 
        # Eq: (h_target - h)^2
        # Weight: -1.0
        self.rewards.pen_base_height.weight = -1.0
        # Important: HIM targets robot base height relative to ground. 
        # Ensure target matches your robot. 
        # Using height scanner to compute height relative to terrain.
        self.rewards.pen_base_height.params["sensor_cfg"] = SceneEntityCfg("height_scanner")
        self.rewards.pen_base_height.params["target_height"] = 0.68

        # 9. Foot clearance 
        # Eq: sum((p_z_target - p_z)^2 * v_xy)
        # Weight: -0.01
        self.rewards.pen_feet_clearance = RewTerm(
            func=mdp.feet_clearance_him, 
            weight=-0.01, 
            params={
                "target_height": 0.15, # p_z_target approx
                "asset_cfg": SceneEntityCfg("robot", body_names=".*foot_[LR]_Link") # Regex for feet
            }
        )

        # 10. Action rate 
        # Eq: |a_t - a_{t-1}|^2
        # Weight: -0.01
        self.rewards.pen_action_rate.weight = -0.01

        # 11. Smoothness 
        # Eq: |a_t - 2a_{t-1} + a_{t-2}|^2
        # Weight: -0.01
        self.rewards.pen_action_smoothness.weight = -0.01

        # =========================================================================
        # REMOVE NON-HIM REWARDS
        # =========================================================================
        # HIM does not use these specific regularization terms
        self.rewards.pen_feet_regulation = None
        self.rewards.foot_landing_vel = None
        # self.rewards.test_gait_reward = None
        # self.rewards.pen_feet_distance = None # Not in Table 5
        self.rewards.pen_feet_distance.weight=-50.0
        self.rewards.pen_undesired_contacts = None # Not in Table 5 (though often kept for safety, strictly HIM doesn't list it)
        self.rewards.pen_joint_pos_limits = None # Not in Table 5
        self.rewards.pen_joint_vel_l2 = None # Not in Table 5 (covered by power/smoothness)
        self.rewards.pen_joint_torque = None # Not in Table 5 (covered by power)
        
        # debug_vis
        self.commands.base_velocity.debug_vis=False
        
        
@configclass
class PFHIMPlayEnvCfg(PFBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_Link",
            attach_yaw_only=True,
            offset = OffsetCfg(pos=(0, 0, 20.0)),
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.0, 1.0]), #TODO: adjust size to fit real robot
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.observations.policy.heights = None
        self.observations.critic.heights = ObsTerm(func=mdp.height_scan,
            params = {"sensor_cfg": SceneEntityCfg("height_scanner"),
                      "offset":0.78}, # the defualt height of robot is 0.78m
            clip = (-2.0, 2.0),
        )
        
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        
        self.commands.base_velocity.ranges.lin_vel_x = (0.4, 0.5)      
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)     
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.debug_vis = True
        
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = HIM_PLAY_TERRAINS_CFG


#############################
# PIM带高度扫描的双足机器人楼梯环境 / Pointfoot Stairs Environment with Height Scanning
#############################

@configclass
class PFPIMEnvCfg(PFPIMBasedEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        # ... (Terrain, Height Scanner, Observations, Commands setup remains the same) ...
        # 1. 地形与课程设置
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = HIM_TERRAINS_CFG

        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_Link",
            attach_yaw_only=True,
            offset = OffsetCfg(pos=(0, 0, 20.0)),
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.7, 1.1]), 
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.observations.policy.heights = None
        self.observations.critic.heights = ObsTerm(func=mdp.height_scan,
            params = {"sensor_cfg": SceneEntityCfg("height_scanner"),
                      "offset":0.78}, 
            clip = (-2.0, 2.0),
        )
        
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        
        self.commands.base_velocity.limit_ranges.lin_vel_x = (-0.2, 1.0)      
        self.commands.base_velocity.limit_ranges.lin_vel_y = (-0.2, 0.2)     
        self.commands.base_velocity.limit_ranges.ang_vel_z = (-math.pi / 6, math.pi / 6)
        self.commands.gait_command = None

        # =========================================================================
        # REWARD MODIFICATIONS (Strictly following HIM Table 5)
        # =========================================================================

        # 1. Linear velocity tracking 
        # Eq: exp(-|v_cmd - v_xy|^2 / sigma)
        # Weight: 1.0
        self.rewards.rew_lin_vel_xy.weight = 1.0
        self.rewards.rew_lin_vel_xy.params["std"] = 0.25 # sigma = 0.25

        # 2. Angular velocity tracking 
        # Eq: exp(-|w_cmd - w_yaw|^2 / sigma)
        # Weight: 0.5
        self.rewards.rew_ang_vel_z.weight = 0.5
        self.rewards.rew_ang_vel_z.params["std"] = 0.25 # sigma = 0.25

        # 3. Linear velocity (z) 
        # Eq: v_z^2
        # Weight: -2.0
        self.rewards.pen_lin_vel_z.weight = -2.0

        # 4. Angular velocity (xy) 
        # Eq: |w_xy|^2
        # Weight: -0.05
        self.rewards.pen_ang_vel_xy.weight = -0.05

        # 5. Orientation 
        # Eq: |g_proj|^2 (approx via flat_orientation_l2)
        # Weight: -0.2
        self.rewards.pen_flat_orientation.weight = -0.2

        # 6. Joint accelerations 
        # Eq: |theta_ddot|^2
        # Weight: -2.5e-7
        self.rewards.pen_joint_accel.weight = -2.5e-7

        # 7. Joint power 
        # Eq: |tau * theta_dot|
        # Weight: -2e-5
        self.rewards.pen_joint_powers.weight = -2e-5

        # 8. Body height 
        # Eq: (h_target - h)^2
        # Weight: -1.0
        self.rewards.pen_base_height.weight = -1.0
        # Important: HIM targets robot base height relative to ground. 
        # Ensure target matches your robot. 
        # Using height scanner to compute height relative to terrain.
        self.rewards.pen_base_height.params["sensor_cfg"] = SceneEntityCfg("height_scanner")
        self.rewards.pen_base_height.params["target_height"] = 0.78

        # 9. Foot clearance 
        # Eq: sum((p_z_target - p_z)^2 * v_xy)
        # Weight: -0.01
        self.rewards.pen_feet_clearance = RewTerm(
            func=mdp.feet_clearance_him, 
            weight=-0.01, 
            params={
                "target_height": 0.15, # p_z_target approx
                "asset_cfg": SceneEntityCfg("robot", body_names=".*foot_[LR]_Link") # Regex for feet
            }
        )

        # 10. Action rate 
        # Eq: |a_t - a_{t-1}|^2
        # Weight: -0.01
        self.rewards.pen_action_rate.weight = -0.01

        # 11. Smoothness 
        # Eq: |a_t - 2a_{t-1} + a_{t-2}|^2
        # Weight: -0.01
        self.rewards.pen_action_smoothness.weight = -0.01

        # =========================================================================
        # REMOVE NON-HIM REWARDS
        # =========================================================================
        # HIM does not use these specific regularization terms
        self.rewards.pen_feet_regulation = None
        self.rewards.foot_landing_vel = None
        self.rewards.test_gait_reward = None
        # self.rewards.pen_feet_distance = None # Not in Table 5
        self.rewards.pen_feet_distance.weight=-5.0
        self.rewards.pen_undesired_contacts = None # Not in Table 5 (though often kept for safety, strictly HIM doesn't list it)
        self.rewards.pen_joint_pos_limits = None # Not in Table 5
        self.rewards.pen_joint_vel_l2 = None # Not in Table 5 (covered by power/smoothness)
        self.rewards.pen_joint_torque = None # Not in Table 5 (covered by power)
        
        # debug_vis
        self.commands.base_velocity.debug_vis=False
        
        
@configclass
class PFPIMPlayEnvCfg(PFPIMBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_Link",
            attach_yaw_only=True,
            offset = OffsetCfg(pos=(0, 0, 20.0)),
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.7, 1.1]), #TODO: adjust size to fit real robot
            debug_vis=True,
            mesh_prim_paths=["/World/ground"],
        )
        self.observations.policy.heights = None
        self.observations.critic.heights = ObsTerm(func=mdp.height_scan,
            params = {"sensor_cfg": SceneEntityCfg("height_scanner"),
                      "offset":0.78}, # the defualt height of robot is 0.78m
            clip = (-2.0, 2.0),
        )
        
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        
        self.commands.base_velocity.ranges.lin_vel_x = (0.4, 0.5)      
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)     
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = HIM_PLAY_TERRAINS_CFG

