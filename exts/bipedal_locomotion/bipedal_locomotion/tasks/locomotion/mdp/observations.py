from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.sensors import Camera, ContactSensor, Imu, RayCaster, RayCasterCamera, TiledCamera

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def robot_joint_torque(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """机器人关节力矩 / Joint torque of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.applied_torque.to(device)


def robot_joint_acc(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """机器人关节加速度 / Joint acceleration of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.joint_acc.to(device)


def robot_feet_contact_force(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg):
    """机器人脚部接触力 / Contact force of the robot feet"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    contact_force_tensor = contact_sensor.data.net_forces_w_history.to(device)
    return contact_force_tensor.view(contact_force_tensor.shape[0], -1)

def robot_feet_contact_force_current(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg):
    """机器人脚部接触力（仅当前帧） / Contact force of the robot feet (Current Frame Only)"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 使用 net_forces_w 而不是 net_forces_w_history
    contact_force_tensor = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids] 
    
    # 形状: (num_envs, num_feet, 3) -> (num_envs, num_feet * 3) = 12
    return contact_force_tensor.view(contact_force_tensor.shape[0], -1)

def robot_mass(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """机器人的质量 / Mass of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.default_mass.to(device)


def robot_inertia(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """机器人的惯量 / Inertia of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    inertia_tensor = asset.data.default_inertia.to(device)
    return inertia_tensor.view(inertia_tensor.shape[0], -1)


def robot_joint_pos(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """机器人的关节位置 / Joint positions of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.default_joint_pos.to(device)


def robot_joint_stiffness(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """机器人的关节刚度 / Joint stiffness of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.default_joint_stiffness.to(device)


def robot_joint_damping(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """机器人的关节阻尼 / Joint damping of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.default_joint_damping.to(device)


def robot_pos(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """机器人的位置 / Position of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.root_pos_w.to(device)


def robot_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """机器人的速度 / Velocity of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.root_vel_w.to(device)


def robot_material_properties(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """机器人的材料属性 / Material properties of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    material_tensor = asset.root_physx_view.get_material_properties().to(device)
    return material_tensor.view(material_tensor.shape[0], -1)


def robot_center_of_mass(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """机器人的质心 / Center of mass of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    com_tensor = asset.root_physx_view.get_coms().clone().to(device)
    return com_tensor.view(com_tensor.shape[0], -1)


def robot_contact_force(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """机器人的接触力 / The contact forces of the robot."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    body_contact_force = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    return body_contact_force.reshape(body_contact_force.shape[0], -1)


def get_gait_phase(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    获取当前步态相位作为观测值。
    Get the current gait phase as observation.

    步态相位由 [sin(phase), cos(phase)] 表示，以确保连续性。
    通过计算当前的 episode 长度和步态频率来确定相位。
    The gait phase is represented by [sin(phase), cos(phase)] to ensure continuity.
    The phase is calculated based on the episode length and gait frequency.

    返回值 / Returns:
        torch.Tensor: 步态观测 / The gait phase observation. 形状 / Shape: (num_envs, 2).
    """
    # 检查 episode_length_buf 是否可用 / Check if episode_length_buf is available
    if not hasattr(env, "episode_length_buf"):
        return torch.zeros(env.num_envs, 2, device=env.device)

    # 从命令管理器获取步态命令 / Get the gait command from command manager
    command_term = env.command_manager.get_term("gait_command")
    
    # 计算基于 episode 长度的步态索引 / Calculate gait indices based on episode length
    gait_indices = torch.remainder(env.episode_length_buf * env.step_dt * command_term.command[:, 0], 1.0)
    
    # 重塑 gait_indices 为 (num_envs, 1) 形状 / Reshape  gait_indices to shape (num_envs, 1)
    gait_indices = gait_indices.unsqueeze(-1)

    # 转换为 sin/cos 表示 / Convert to sin/cos representation
    sin_phase = torch.sin(2 * torch.pi * gait_indices)
    cos_phase = torch.cos(2 * torch.pi * gait_indices)

    return torch.cat([sin_phase, cos_phase], dim=-1)


def get_gait_command(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """
    获取当前步态命令参数作为观测值。
    Get the current gait command parameters as observation.

    返回值 / Returns:
        torch.Tensor: 步态命令参数 [频率, 偏移, 持续时间] / The gait command parameters [frequency, offset, duration].
                      形状 / Shape: (num_envs, 3).
    """
    return env.command_manager.get_command(command_name)


def robot_base_pose(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """机器人基座的位姿 / Pose of the robot base"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.root_pos_w.to(device)

def feet_lin_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """资产根部坐标系中的线速度 / Root linear velocity in the asset's root frame."""
    # 提取使用的数量（以启用类型提示）/ Extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.body_lin_vel_w[:, asset_cfg.body_ids].flatten(start_dim=1)

def generated_commands(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """给定名称的命令管理器中命令项生成的命令 / The generated command from command term in the command manager with the given name."""
    return env.command_manager.get_command(command_name)

def joint_pos_rel_exclude_wheel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                                wheel_joints_name: list[str] = ["wheel_[RL]_Joint"] 
                                ) -> torch.Tensor:
    """
    资产相对于默认关节位置的关节位置。
    Joint positions of the asset relative to the default joint positions, excluding the specified wheel joints.

    注意：只有在 :attr:`asset_cfg.joint_ids` 中配置的关节才会返回它们的位置。
    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # 提取使用的数量（以启用类型提示）/ Extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    wheel_joints_idx = asset.find_joints(wheel_joints_name)[0]
    all_joints_idx = range(asset.num_joints)
    pos_idx_exclude_wheel = [i for i in all_joints_idx if i not in wheel_joints_idx]
    return asset.data.joint_pos[:, pos_idx_exclude_wheel] - asset.data.default_joint_pos[:, pos_idx_exclude_wheel]
