from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any
from isaaclab.terrains import TerrainImporter
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def modify_event_parameter(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    param_name: str,
    value: Any | SceneEntityCfg,
    num_steps: int,
) -> torch.Tensor:
    """Curriculum that modifies a parameter of an event at a given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the event term.
        param_name: The name of the event term parameter.
        value: The new value for the event term parameter.
        num_steps: The number of steps after which the change should be applied.

    Returns:
        torch.Tensor: Whether the parameter has already been modified or not.
    """
    if env.common_step_counter > num_steps:
        # obtain term settings
        term_cfg = env.event_manager.get_term_cfg(term_name)
        # update term settings
        term_cfg.params[param_name] = value
        env.event_manager.set_term_cfg(term_name, term_cfg)
        return torch.ones(1)
    return torch.zeros(1)


def disable_termination(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    num_steps: int,
) -> torch.Tensor:
    """Curriculum that modifies the push velocity range at a given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the termination term.
        num_steps: The number of steps after which the change should be applied.

    Returns:
        torch.Tensor: Whether the parameter has already been modified or not.
    """
    env.command_manager.num_envs
    if env.common_step_counter > num_steps:
        # obtain term settings
        term_cfg = env.termination_manager.get_term_cfg(term_name)
        # Remove term settings
        term_cfg.params = dict()
        term_cfg.func = lambda env: torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        env.termination_manager.set_term_cfg(term_name, term_cfg)
        return torch.ones(1)
    return torch.zeros(1)


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "rew_lin_vel_xy",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.2, 0.2], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)

def terrain_levels_vel_constrained(
    env: ManagerBasedRLEnv, 
    env_ids: Sequence[int], 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reward_term_name: str = "rew_lin_vel_xy"
) -> torch.Tensor:
    """
    基于行走距离和群体速度跟踪精度的地形课程。
    
    升级条件 (move_up):
    1. 个体条件：机器人行走的距离超过地形块长度的一半。
    2. 群体条件（参考 lin_vel_cmd_levels）：该批次环境的平均速度跟踪奖励超过最大值的 80%。
       这意味着只有当整体策略在当前难度下表现稳定时，才允许个体尝试更难的地形。
    
    降级条件 (move_down):
    1. 个体条件：机器人行走的距离小于指令速度理论距离的 50%。
    """
    # 提取对象
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    
    # 1. 计算行走距离 (个体指标)
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    
    # 2. 原始升级条件：距离达标
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    
    # 3. [修改] 速度奖励约束 (采用 lin_vel_cmd_levels 的逻辑)
    try:
        # 获取奖励配置
        reward_term = env.reward_manager.get_term_cfg(reward_term_name)
        
        # 计算该批次环境的平均每步奖励 (与 lin_vel_cmd_levels 逻辑一致)
        # mean( sum(r) ) / max_time
        avg_reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s
        
        # 判定标准：整体表现 > 80% 满分
        # 注意：这里是一个标量布尔值
        tracking_pass = avg_reward > reward_term.weight * 0.8
        
        # 组合条件：个体走得远 AND 整体策略跟踪好
        # 如果 tracking_pass 为 False，所有环境的 move_up 都会被置为 False
        if not tracking_pass:
            move_up[:] = False
            
    except KeyError:
        print(f"[Warning] terrain_levels_vel_constrained: Reward term '{reward_term_name}' not found. Skipping tracking constraint.")
    except Exception as e:
        print(f"[Error] terrain_levels_vel_constrained: {e}")

    # 4. 降级条件 (个体指标)
    target_dist = torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s
    move_down = distance < target_dist * 0.5
    
    # 互斥处理
    move_down *= ~move_up
    
    # 5. 执行更新
    terrain.update_env_origins(env_ids, move_up, move_down)
    
    # 返回平均地形等级
    return torch.mean(terrain.terrain_levels.float())