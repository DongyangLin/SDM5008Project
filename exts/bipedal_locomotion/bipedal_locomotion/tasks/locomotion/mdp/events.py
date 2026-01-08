from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
import isaaclab.sim as sim_utils
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

PUSH_ARROW_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "arrow_x": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
            scale=(1.0, 0.5, 0.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        )
    }
)

def prepare_quantity_for_tron(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_radius = 0.127,
):
    """为TRON机器人准备数量参数 / Prepare quantity parameters for TRON robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    env._foot_radius = foot_radius


def apply_external_force_torque_stochastic(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    force_range: dict[str, tuple[float, float]],
    torque_range: dict[str, tuple[float, float]],
    probability: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """
    随机对指定刚体施加外力与力矩，并进行可视化。
    Apply stochastic external forces and torques with visualization support.
    """

    # 获取受控刚体 / Get the controlled asset
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # 清除旧力 / Clear old forces
    asset._external_force_b *= 0
    asset._external_torque_b *= 0

    # 解析环境ID / Resolve environment IDs
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # 随机选择施加力的环境 / Randomly select environments to apply forces
    random_values = torch.rand(env_ids.shape, device=env_ids.device)
    mask = random_values < probability
    masked_env_ids = env_ids[mask]

    # 可视化初始化 / Visualization initialization
    if not hasattr(apply_external_force_torque_stochastic, "markers"):
        cfg = PUSH_ARROW_MARKER_CFG.copy()
        cfg.prim_path = "/World/Visuals/PushForceArrows"
        apply_external_force_torque_stochastic.markers = VisualizationMarkers(cfg)

    viz_markers = apply_external_force_torque_stochastic.markers

    device = asset.device
    num_envs = env.scene.num_envs

    # 准备 Marker 参数 / Prepare Marker parameters
    marker_pos = asset.data.root_pos_w.clone()
    marker_rot = torch.zeros(num_envs, 4, device=device)
    marker_rot[:, 0] = 1.0
    marker_scale = torch.zeros(num_envs, 3, device=device)  # 默认隐藏

    if len(masked_env_ids) > 0:
        # 解析目标刚体索引 / Resolve target body indices
        if isinstance(asset_cfg.body_ids, list):
            target_body_indices = asset_cfg.body_ids
            num_bodies = len(asset_cfg.body_ids)
        elif isinstance(asset_cfg.body_ids, int):
            target_body_indices = [asset_cfg.body_ids]
            num_bodies = 1
        elif isinstance(asset_cfg.body_ids, slice):
            # 解析切片 / Resolve slice
            start = asset_cfg.body_ids.start if asset_cfg.body_ids.start else 0
            stop = (
                asset_cfg.body_ids.stop if asset_cfg.body_ids.stop else asset.num_bodies
            )
            step = asset_cfg.body_ids.step if asset_cfg.body_ids.step else 1
            target_body_indices = list(range(start, stop, step))
            num_bodies = len(target_body_indices)
        else:
            # 默认选择所有刚体 / Default to all bodies
            target_body_indices = list(range(asset.num_bodies))
            num_bodies = asset.num_bodies

        # 采样维度 / Sampling size
        size = (len(masked_env_ids), num_bodies, 3)

        # 采样力和力矩 / Sample forces and torques
        f_list = [force_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
        f_range = torch.tensor(f_list, device=device)
        forces = math_utils.sample_uniform(f_range[:, 0], f_range[:, 1], size, device)

        t_list = [torque_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
        t_range = torch.tensor(t_list, device=device)
        torques = math_utils.sample_uniform(t_range[:, 0], t_range[:, 1], size, device)

        # 应用力和力矩 / Apply forces and torques
        asset.set_external_force_and_torque(
            forces, torques, env_ids=masked_env_ids, body_ids=asset_cfg.body_ids
        )

        # 可视化力箭头 / Visualize force arrows
        primary_body_idx = target_body_indices[0]
        target_body_pos = asset.data.body_pos_w[masked_env_ids, primary_body_idx, :]
        marker_pos[masked_env_ids] = target_body_pos

        # 计算箭头方向 / Compute arrow direction
        applied_forces = forces[:, 0, :]
        
        force_mag = torch.norm(applied_forces, dim=-1, keepdim=True)   # 计算四元数表示的方向 / Compute orientation in quaternion
        yaw = torch.atan2(applied_forces[:, 1], applied_forces[:, 0])
        pitch = torch.zeros_like(yaw)
        roll = torch.zeros_like(yaw)
        quat = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
        marker_rot[masked_env_ids] = quat   # 设置 Marker 旋转 / Set Marker rotation

        # 计算箭头长度和粗细 / Compute arrow length and thickness
        scale_factor = 0.025
        arrow_len = force_mag * scale_factor
        arrow_len = torch.clamp(arrow_len, min=0.5, max=10.0)   # 限制箭头长度 / Limit arrow length

        thickness = 0.2
        marker_scale[masked_env_ids, 0] = arrow_len.squeeze()
        marker_scale[masked_env_ids, 1] = thickness
        marker_scale[masked_env_ids, 2] = thickness

    # 提交可视化 / Submit visualization
    viz_markers.visualize(
        translations=marker_pos, orientations=marker_rot, scales=marker_scale
    )


def randomize_rigid_body_mass_inertia(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    mass_inertia_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """通过添加、缩放或设置随机值来随机化物体的惯量. 
    Randomize the inertia of the bodies by adding, scaling, or setting random values.

    该函数允许随机化资产物体的质量。函数从给定的分布参数中采样随机值，并根据操作将值添加、缩放或设置到物理仿真中。
    This function allows randomizing the mass of the bodies of the asset. The function samples random values from the
    given distribution parameters and adds, scales, or sets the values into the physics simulation based on the operation.

    Tips:
        该函数使用CPU张量来分配物体质量。建议仅在环境初始化期间使用此函数。
        This function uses CPU tensors to assign the body masses. It is recommended to use this function
        only during the initialization of the environment.
    """
    # 提取使用的量（以启用类型提示） / Extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # 解析环境索引 / Resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # 解析物体索引 / Resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # 获取物体当前的惯量 (num_assets, num_bodies) / Get the current inertias of the bodies (num_assets, num_bodies)
    inertias = asset.root_physx_view.get_inertias().clone()
    masses = asset.root_physx_view.get_masses().clone()

    masses = _randomize_prop_by_op(
        masses, mass_inertia_distribution_params, env_ids, body_ids, operation=operation, distribution=distribution
    )
    scale = masses / asset.root_physx_view.get_masses()
    inertias *= scale.unsqueeze(-1)

    asset.root_physx_view.set_masses(masses, env_ids)
    asset.root_physx_view.set_inertias(inertias, env_ids)


def randomize_rigid_body_coms(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    com_distribution_params: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    通过为每个维度添加、缩放或设置随机值来随机化物体的重心 (COM)
    Randomize the center of mass (COM) of the bodies by adding, scaling, or setting random values for each dimension.
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    coms = asset.root_physx_view.get_coms().clone()

    # 对每个维度分别应用随机化 / Apply randomization to each dimension separately
    for dim in range(3):  # 0=x, 1=y, 2=z
        coms[..., dim] = _randomize_prop_by_op(
            coms[..., dim],
            com_distribution_params[dim],
            env_ids,
            body_ids,
            operation=operation,
            distribution=distribution,
        )

    asset.root_physx_view.set_coms(coms, env_ids)


"""
Internal helper functions.
"""

def _randomize_prop_by_op(
    data: torch.Tensor,
    distribution_parameters: tuple[float | torch.Tensor, float | torch.Tensor],
    dim_0_ids: torch.Tensor | None,
    dim_1_ids: torch.Tensor | slice,
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"],
) -> torch.Tensor:
    """根据给定的操作和分布执行数据随机化 / Perform data randomization based on the given operation and distribution.

    Args:
        data: 要随机化的数据张量。形状为 (dim_0, dim_1) / The data tensor to be randomized. Shape is (dim_0, dim_1).
        distribution_parameters: 用于采样值的分布参数 / The parameters for the distribution to sample values from.
        dim_0_ids: 要随机化的第一维索引 / The indices of the first dimension to randomize.
        dim_1_ids: 要随机化的第二维索引 / The indices of the second dimension to randomize.
        operation: 对数据执行的操作。选项：'add', 'scale', 'abs' / The operation to perform on the data. Options: 'add', 'scale', 'abs'.
        distribution: 采样随机值的分布。选项：'uniform', 'log_uniform', 'gaussian' / The distribution to sample the random values from. Options: 'uniform', 'log_uniform', 'gaussian'.

    Returns:
        随机化后的数据张量。形状为 (dim_0, dim_1) / The data tensor after randomization. Shape is (dim_0, dim_1).

    Raises:
        NotImplementedError: 如果操作或分布不受支持 / If the operation or distribution is not supported.
    """
    # 解析形状 / Resolve shape
    # -- dim 0
    if dim_0_ids is None:
        n_dim_0 = data.shape[0]
        dim_0_ids = slice(None)
    else:
        n_dim_0 = len(dim_0_ids)
        if not isinstance(dim_1_ids, slice):
            dim_0_ids = dim_0_ids[:, None]
    # -- dim 1
    if isinstance(dim_1_ids, slice):
        n_dim_1 = data.shape[1]
    else:
        n_dim_1 = len(dim_1_ids)

    # resolve the distribution
    if distribution == "uniform":
        dist_fn = math_utils.sample_uniform
    elif distribution == "log_uniform":
        dist_fn = math_utils.sample_log_uniform
    elif distribution == "gaussian":
        dist_fn = math_utils.sample_gaussian
    else:
        raise NotImplementedError(
            f"Unknown distribution: '{distribution}' for joint properties randomization."
            " Please use 'uniform', 'log_uniform', 'gaussian'."
        )
    # perform the operation
    if operation == "add":
        data[dim_0_ids, dim_1_ids] += dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "scale":
        data[dim_0_ids, dim_1_ids] *= dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "abs":
        data[dim_0_ids, dim_1_ids] = dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    else:
        raise NotImplementedError(
            f"Unknown operation: '{operation}' for property randomization. Please use 'add', 'scale', or 'abs'."
        )
    return data
