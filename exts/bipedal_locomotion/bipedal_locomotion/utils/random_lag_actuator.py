import torch
from isaaclab.actuators import ImplicitActuator, ImplicitActuatorCfg
from isaaclab.utils import ArticulationActions
from isaaclab.utils import configclass

# =========================================================================
# 1. 先定义 Actuator 类 (First define the Class)
# =========================================================================
class RandomLaggyActuator(ImplicitActuator):
    """
    带有随机时延的隐式执行器。
    """
    # 这里 cfg 的类型提示使用父类 ImplicitActuatorCfg，避免找不到 RandomLaggyActuatorCfg 的报错
    def __init__(self, cfg: ImplicitActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        
        # 安全获取参数，提供默认值以防万一
        self.max_lag = getattr(cfg, "max_lag", 0)
        self.randomize_lag = getattr(cfg, "randomize_lag", False)
        
        self._action_buffer = None
        self._lag_indices = None
        self._reset_mask = None

    def reset(self, env_ids: torch.Tensor):
        super().reset(env_ids)
        if self._reset_mask is not None:
            self._reset_mask[env_ids] = True
        if self._lag_indices is not None and self.randomize_lag:
            self._lag_indices[env_ids] = torch.randint(
                0, self.max_lag + 1, (len(env_ids),), device=self._device
            )

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        
        # 1. 懒加载初始化
        if self._action_buffer is None:
            self._device = joint_pos.device
            num_envs, num_joints = joint_pos.shape
            
            # 初始化缓冲区 [N, Lag+1, J, 3]
            self._action_buffer = torch.zeros(
                num_envs, self.max_lag + 1, num_joints, 3, 
                device=self._device, dtype=joint_pos.dtype
            )
            self._reset_mask = torch.ones(num_envs, dtype=torch.bool, device=self._device)
            
            if self.randomize_lag:
                self._lag_indices = torch.randint(0, self.max_lag + 1, (num_envs,), device=self._device)
            else:
                self._lag_indices = torch.zeros(num_envs, dtype=torch.long, device=self._device)

        # 2. 预处理指令
        current_cmd = torch.stack([
            control_action.joint_positions,
            control_action.joint_velocities,
            control_action.joint_efforts
        ], dim=-1)

        # 3. 处理 Reset 防暴冲
        if self._reset_mask.any():
            env_ids = self._reset_mask.nonzero(as_tuple=False).squeeze(-1)
            self._action_buffer[env_ids] = current_cmd[env_ids].unsqueeze(1)
            self._reset_mask[env_ids] = False

        # 4. 滚动并插入
        self._action_buffer = torch.roll(self._action_buffer, shifts=1, dims=1)
        self._action_buffer[:, 0] = current_cmd

        # 5. 取出延迟指令
        indices = self._lag_indices.view(-1, 1, 1, 1).expand(-1, 1, joint_pos.shape[1], 3)
        delayed_cmd = torch.gather(self._action_buffer, 1, indices).squeeze(1)

        # 6. 写回并计算
        control_action.joint_positions = delayed_cmd[..., 0]
        control_action.joint_velocities = delayed_cmd[..., 1]
        control_action.joint_efforts = delayed_cmd[..., 2]

        return super().compute(control_action, joint_pos, joint_vel)

# =========================================================================
# 2. 后定义 Config 类 (Then define the Config)
# =========================================================================
@configclass
class RandomLaggyActuatorCfg(ImplicitActuatorCfg):
    """Configuration for the random laggy actuator model."""
    
    # 直接赋值！不再是 None，也不需要后续绑定
    class_type = RandomLaggyActuator 
    
    max_lag: int = 0
    randomize_lag: bool = False