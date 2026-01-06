from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

# --- [关键修改] 使用 isaaclab 开头的路径 ---
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

# def apply_external_force_torque_stochastic(
#     env: ManagerBasedEnv,
#     env_ids: torch.Tensor,
#     force_range: dict[str, tuple[float, float]],
#     torque_range: dict[str, tuple[float, float]],
#     probability: float,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
# ):
#     """随机施加外部力和力矩 / Randomize the external forces and torques applied to the bodies.

#     该函数创建从给定范围采样的随机力和力矩集合。力和力矩的数量等于物体数量乘以环境数量。
#     力和力矩通过调用``asset.set_external_force_and_torque``应用到物体上。
#     只有当在环境中调用``asset.write_data_to_sim()``时，力和力矩才会被应用。

#     This function creates a set of random forces and torques sampled from the given ranges. The number of forces
#     and torques is equal to the number of bodies times the number of environments. The forces and torques are
#     applied to the bodies by calling ``asset.set_external_force_and_torque``. The forces and torques are only
#     applied when ``asset.write_data_to_sim()`` is called in the environment.
#     """
#     # extract the used quantities (to enable type-hinting)
#     asset: RigidObject | Articulation = env.scene[asset_cfg.name]
#     # clear the existing forces and torques
#     asset._external_force_b *= 0
#     asset._external_torque_b *= 0

#     # resolve environment ids
#     if env_ids is None:
#         env_ids = torch.arange(env.scene.num_envs, device=asset.device)

#     random_values = torch.rand(env_ids.shape, device=env_ids.device)
#     mask = random_values < probability
#     masked_env_ids = env_ids[mask]

#     if len(masked_env_ids) == 0:
#         return

#     # resolve number of bodies
#     num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else asset.num_bodies

#     # sample random forces and torques
#     size = (len(masked_env_ids), num_bodies, 3)
#     force_range_list = [force_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
#     force_range = torch.tensor(force_range_list, device=asset.device)
#     forces = math_utils.sample_uniform(force_range[:, 0], force_range[:, 1], size, asset.device)
#     torque_range_list = [torque_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
#     torque_range = torch.tensor(torque_range_list, device=asset.device)
#     torques = math_utils.sample_uniform(torque_range[:, 0], torque_range[:, 1], size, asset.device)
#     # set the forces and torques into the buffers
#     # note: these are only applied when you call: `asset.write_data_to_sim()`
#     asset.set_external_force_and_torque(forces, torques, env_ids=masked_env_ids, body_ids=asset_cfg.body_ids)


def apply_external_force_torque_stochastic(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    force_range: dict[str, tuple[float, float]],
    torque_range: dict[str, tuple[float, float]],
    probability: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """
    随机施加外部力并可视化的修正版函数 (v2.1.0+)。
    已调整：箭头长度增加5倍，起点精确定位到受力刚体中心。
    """

    # 1. 获取 Asset
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # 2. 清除旧力
    asset._external_force_b *= 0
    asset._external_torque_b *= 0

    # 3. 解析 env_ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # 4. 随机采样
    random_values = torch.rand(env_ids.shape, device=env_ids.device)
    mask = random_values < probability
    masked_env_ids = env_ids[mask]

    # =========================================================================
    # [可视化初始化]
    # =========================================================================
    if not hasattr(apply_external_force_torque_stochastic, "markers"):
        cfg = PUSH_ARROW_MARKER_CFG.copy()
        cfg.prim_path = "/World/Visuals/PushForceArrows"
        apply_external_force_torque_stochastic.markers = VisualizationMarkers(cfg)

    viz_markers = apply_external_force_torque_stochastic.markers

    # =========================================================================
    # 数据计算与应用
    # =========================================================================
    device = asset.device
    num_envs = env.scene.num_envs

    # --- [修改点 1] 初始化位置逻辑优化 ---
    # 我们不再默认使用 root_pos，稍后会根据 body index 覆盖它
    # 先初始化为 root 以防万一，但 scale 默认为 0 (隐藏)
    marker_pos = asset.data.root_pos_w.clone()
    marker_rot = torch.zeros(num_envs, 4, device=device)
    marker_rot[:, 0] = 1.0
    marker_scale = torch.zeros(num_envs, 3, device=device)  # 默认隐藏

    if len(masked_env_ids) > 0:
        # 解析受力刚体索引
        # asset_cfg.body_ids 存储的是需要施加力的刚体索引列表
        if isinstance(asset_cfg.body_ids, list):
            target_body_indices = asset_cfg.body_ids
            num_bodies = len(asset_cfg.body_ids)
        elif isinstance(asset_cfg.body_ids, int):
            target_body_indices = [asset_cfg.body_ids]
            num_bodies = 1
        elif isinstance(asset_cfg.body_ids, slice):
            # 处理切片情况，通常是所有 bodies
            # 这里需要展开 range，否则后面无法用来索引 body_pos_w
            start = asset_cfg.body_ids.start if asset_cfg.body_ids.start else 0
            stop = (
                asset_cfg.body_ids.stop if asset_cfg.body_ids.stop else asset.num_bodies
            )
            step = asset_cfg.body_ids.step if asset_cfg.body_ids.step else 1
            target_body_indices = list(range(start, stop, step))
            num_bodies = len(target_body_indices)
        else:
            # 默认为全部
            target_body_indices = list(range(asset.num_bodies))
            num_bodies = asset.num_bodies

        # 准备采样大小
        size = (len(masked_env_ids), num_bodies, 3)

        # --- 采样力与力矩 ---
        f_list = [force_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
        f_range = torch.tensor(f_list, device=device)
        forces = math_utils.sample_uniform(f_range[:, 0], f_range[:, 1], size, device)

        t_list = [torque_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
        t_range = torch.tensor(t_list, device=device)
        torques = math_utils.sample_uniform(t_range[:, 0], t_range[:, 1], size, device)

        # --- 应用力 ---
        asset.set_external_force_and_torque(
            forces, torques, env_ids=masked_env_ids, body_ids=asset_cfg.body_ids
        )

        # --- [修改点 2] 准确获取作用点 ---
        # 假设我们主要关注第一个受力刚体（通常是 base_Link）来画箭头
        # 如果是对多个刚体施力，这里只画第一个，避免箭头太乱
        primary_body_idx = target_body_indices[0]

        # 获取该刚体的世界坐标 (N, NumBodies, 3) -> (N, 3)
        # 注意：masked_env_ids 是被推的环境索引
        # asset.data.body_pos_w 包含了所有环境所有刚体的位置

        # 提取被推环境的、特定刚体的位置
        target_body_pos = asset.data.body_pos_w[masked_env_ids, primary_body_idx, :]

        # 更新 Marker 位置：直接等于刚体中心位置，不加任何偏移
        marker_pos[masked_env_ids] = target_body_pos

        # --- [修改点 3] 长度放大与旋转 ---
        # 提取施加在这个刚体上的力
        applied_forces = forces[:, 0, :]

        force_mag = torch.norm(applied_forces, dim=-1, keepdim=True)
        yaw = torch.atan2(applied_forces[:, 1], applied_forces[:, 0])
        pitch = torch.zeros_like(yaw)
        roll = torch.zeros_like(yaw)
        quat = math_utils.quat_from_euler_xyz(roll, pitch, yaw)

        marker_rot[masked_env_ids] = quat

        # 缩放逻辑修改：
        # 原来 scale_factor = 0.02 (50N -> 1m)
        # 现在 scale_factor = 0.10 (50N -> 5m) -> 放大5倍
        scale_factor = 0.025
        arrow_len = force_mag * scale_factor

        # 放宽最大长度限制，防止被截断
        # min=0.5 保证小力也能看见，max=10.0 保证大力足够长
        arrow_len = torch.clamp(arrow_len, min=0.5, max=10.0)

        # 增加粗度，避免长了之后看起来像针
        thickness = 0.2

        marker_scale[masked_env_ids, 0] = arrow_len.squeeze()
        marker_scale[masked_env_ids, 1] = thickness
        marker_scale[masked_env_ids, 2] = thickness

    # 提交可视化
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
    """通过添加、缩放或设置随机值来随机化物体的惯量 / Randomize the inertia of the bodies by adding, scaling, or setting random values.

    该函数允许随机化资产物体的质量。函数从给定的分布参数中采样随机值，并根据操作将值添加、缩放或设置到物理仿真中。
    
    This function allows randomizing the mass of the bodies of the asset. The function samples random values from the
    given distribution parameters and adds, scales, or sets the values into the physics simulation based on the operation.

    .. tip::
        该函数使用CPU张量来分配物体质量。建议仅在环境初始化期间使用此函数。
        This function uses CPU tensors to assign the body masses. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # get the current inertias of the bodies (num_assets, num_bodies)
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
    """通过为每个维度添加、缩放或设置随机值来随机化物体的重心（COM）
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

    # Apply randomization to each dimension separately
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
    # resolve shape
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
