import torch
import torch.nn as nn
import os
import copy

class PIMPolicyExporter(nn.Module):
    """
    专门用于导出 PIMActorCritic 的包装器。
    功能：
    1. 接收 history 观测输入
    2. (可选) 执行 Observation Normalization
    3. 调用 Estimator
    4. 调用 Actor 输出动作
    """
    def __init__(self, actor_critic, normalizer=None):
        super().__init__()
        # 1. 提取核心模块 (深拷贝防止影响原模型)
        self.estimator = copy.deepcopy(actor_critic.estimator)
        self.actor = copy.deepcopy(actor_critic.actor)
        self.dim_nonperceptive_obs = actor_critic.dim_nonperceptive_obs
        self.dim_perceptive_obs = actor_critic.dim_perceptive_obs
        
        # 2. 处理归一化 (Fusion)
        self.has_normalizer = False
        if normalizer is not None:
            self.has_normalizer = True
            # 从 RunningMeanStd 中提取 mean 和 var
            # 注意：这里假设 normalizer.running_ms.mean 的维度与 obs_history 一致 (e.g., 225)
            # 如果你的 normalizer 维度是单步 (e.g., 45)，需要在这里做 repeat 操作
            self.register_buffer('obs_mean', normalizer.running_ms.mean.clone())
            self.register_buffer('obs_var', normalizer.running_ms.var.clone())
            print(f"[PIMExporter] Fused Normalization: Mean shape {self.obs_mean.shape}, Var shape {self.obs_var.shape}")
        else:
            print("[PIMExporter] WARNING: No normalizer provided. The exported model expects NORMALIZED inputs.")

    def forward(self, obs_history, obs_perceptive):
        # 1. 归一化处理 (In-graph Normalization)
        # 公式: (x - mean) / (std + eps)
        obs_history = obs_history
        if self.has_normalizer:
            obs_history = (obs_history - self.obs_mean) / (torch.sqrt(self.obs_var) + 1e-4)

        # 2. Estimator 推理
        # 对应 PIMActorCritic.act_inference 中的逻辑
        vel, latent = self.estimator(obs_history, obs_perceptive)

        # 3. 拼接 Actor 输入
        # obs_history[:, -self.num_one_step_obs:] 取的是最新的那一帧观测
        obs_current = obs_history[:, -self.dim_nonperceptive_obs:]
        actor_input = torch.cat((obs_current, obs_perceptive, vel, latent), dim=-1)

        # 4. Actor 推理
        actions = self.actor(actor_input)
        
        return actions

def export_pim_actor_critic_as_onnx(
    actor_critic,
    path,
    name="pim_actor_critic",
    input_history_dim=None,  # 变成可选参数
    input_perceptive_dim=None,
    normalizer=None
):
    """
    导出 PIM 策略为 ONNX 格式 (支持自动推断输入维度)。
    """
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, name + ".onnx")

    # =====================================================
    # 自动推断 Input Dim 的逻辑
    # =====================================================
    if input_history_dim is None:
        try:
            # 优先尝试方法 1: 直接读取 ActorCritic 的属性
            if hasattr(actor_critic, "dim_nonperceptive_obs") and hasattr(actor_critic, "history_length"):
                input_history_dim = actor_critic.history_length * actor_critic.dim_nonperceptive_obs
                print(f"[Auto-Detect] Found input_history_dim from actor_critic properties: {input_history_dim}")

            # 备选方法 2: 检查 Estimator Encoder 的第一层线性层
            elif hasattr(actor_critic, "estimator"):
                # encoder 是 nn.Sequential，第0层通常是 Linear
                input_history_dim = actor_critic.estimator.history_length * actor_critic.estimator.dim_nonperceptive_obs
                print(f"[Auto-Detect] Found input_history_dim from estimator properties: {input_history_dim}")
            
            if input_history_dim is None:
                raise ValueError("Could not auto-detect input_history_dim.")
                
        except Exception as e:
            print(f"Error auto-detecting input_history_dim: {e}")
            print("Please provide input_history_dim manually.")
            return
        
    if input_perceptive_dim is None:
        try:
            # 优先尝试方法 1: 直接读取 ActorCritic 的属性
            if hasattr(actor_critic, "dim_nonperceptive_obs"):
                input_perceptive_dim = actor_critic.dim_perceptive_obs
                print(f"[Auto-Detect] Found input_perceptive_dim from actor_critic properties: {input_perceptive_dim}")

            # 备选方法 2: 检查 Estimator Encoder 的第一层线性层
            elif hasattr(actor_critic, "estimator"):
                # encoder 是 nn.Sequential，第0层通常是 Linear
                input_perceptive_dim = actor_critic.estimator.dim_perceptive_obs
                print(f"[Auto-Detect] Found input_perceptive_dim from estimator properties: {input_perceptive_dim}")

            if input_perceptive_dim is None:
                raise ValueError("Could not auto-detect input_perceptive_dim.")
                
        except Exception as e:
            print(f"Error auto-detecting input_perceptive_dim: {e}")
            print("Please provide input_perceptive_dim manually.")
            return
        
    # =====================================================

    # 1. 准备模型 (CPU)
    actor_critic_cpu = actor_critic
    if next(actor_critic.parameters()).is_cuda:
        actor_critic_cpu = copy.deepcopy(actor_critic).cpu()
    
    normalizer_cpu = normalizer
    if normalizer is not None and hasattr(normalizer, 'running_ms'):
        if normalizer.running_ms.mean.is_cuda:
            normalizer_cpu = copy.deepcopy(normalizer)
            normalizer_cpu.running_ms.mean = normalizer_cpu.running_ms.mean.cpu()
            normalizer_cpu.running_ms.var = normalizer_cpu.running_ms.var.cpu()

    # 实例化导出包装器
    export_model = PIMPolicyExporter(actor_critic_cpu, normalizer_cpu)
    export_model.eval()

    # 2. 创建 Dummy Input
    print(f"Generating dummy history_obs with shape: (1, {input_history_dim})")
    dummy_history = torch.randn(1, input_history_dim)
    print(f"Generating dummy perceptive_obs with shape: (1, {input_perceptive_dim})")
    dummy_perceptive = torch.randn(1, input_perceptive_dim)

    # 3. 导出
    torch.onnx.export(
        export_model,
        (dummy_history, dummy_perceptive),
        file_path,
        verbose=False, # 关掉 verbose 稍微清爽点
        input_names=["obs_history", "obs_perceptive"],
        output_names=["actions"],
        export_params=True,
        opset_version=13,
        do_constant_folding=True
    )
    
    print(f"✅ Successfully exported PIM Policy to: {file_path}")
    
def export_pim_actor_critic_as_jit(
    actor_critic,
    path,
    name="pim_actor_critic",
    input_history_dim=None,
    input_perceptive_dim=None,
    normalizer=None
):
    """
    导出 PIM 策略为 TorchScript (JIT) 格式 (.pt)。
    使用 torch.jit.trace 进行追踪。
    """
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, name + ".pt")

    # =====================================================
    # 1. 自动推断 Input Dim (与 ONNX 逻辑一致)
    # =====================================================
    if input_history_dim is None:
        try:
            # 优先尝试方法 1: 直接读取 ActorCritic 的属性
            if hasattr(actor_critic, "dim_nonperceptive_obs") and hasattr(actor_critic, "history_length"):
                input_history_dim = actor_critic.history_length * actor_critic.dim_nonperceptive_obs
                print(f"[Auto-Detect] Found input_history_dim from actor_critic properties: {input_history_dim}")

            # 备选方法 2: 检查 Estimator Encoder 的第一层线性层
            elif hasattr(actor_critic, "estimator"):
                # encoder 是 nn.Sequential，第0层通常是 Linear
                input_history_dim = actor_critic.estimator.history_length * actor_critic.estimator.dim_nonperceptive_obs
                print(f"[Auto-Detect] Found input_history_dim from estimator properties: {input_history_dim}")
            
            if input_history_dim is None:
                raise ValueError("Could not auto-detect input_history_dim.")
                
        except Exception as e:
            print(f"Error auto-detecting input_history_dim: {e}")
            print("Please provide input_history_dim manually.")
            return
        
    if input_perceptive_dim is None:
        try:
            # 优先尝试方法 1: 直接读取 ActorCritic 的属性
            if hasattr(actor_critic, "dim_nonperceptive_obs"):
                input_perceptive_dim = actor_critic.dim_perceptive_obs
                print(f"[Auto-Detect] Found input_perceptive_dim from actor_critic properties: {input_perceptive_dim}")

            # 备选方法 2: 检查 Estimator Encoder 的第一层线性层
            elif hasattr(actor_critic, "estimator"):
                # encoder 是 nn.Sequential，第0层通常是 Linear
                input_perceptive_dim = actor_critic.estimator.dim_perceptive_obs
                print(f"[Auto-Detect] Found input_perceptive_dim from estimator properties: {input_perceptive_dim}")

            if input_perceptive_dim is None:
                raise ValueError("Could not auto-detect input_perceptive_dim.")
                
        except Exception as e:
            print(f"Error auto-detecting input_perceptive_dim: {e}")
            print("Please provide input_perceptive_dim manually.")
            return

    # =====================================================
    # 2. 准备模型 (转移到 CPU 并去除梯度)
    # =====================================================
    # 必须确保所有子模块都在 CPU 上
    actor_critic_cpu = actor_critic
    if next(actor_critic.parameters()).is_cuda:
        actor_critic_cpu = copy.deepcopy(actor_critic).cpu()
    
    normalizer_cpu = normalizer
    if normalizer is not None and hasattr(normalizer, 'running_ms'):
        if normalizer.running_ms.mean.is_cuda:
            normalizer_cpu = copy.deepcopy(normalizer)
            normalizer_cpu.running_ms.mean = normalizer_cpu.running_ms.mean.cpu()
            normalizer_cpu.running_ms.var = normalizer_cpu.running_ms.var.cpu()

    # 使用我们之前定义的包装器 (PIMPolicyExporter)
    # 确保你已经运行了定义 PIMPolicyExporter 类的代码
    trace_model = PIMPolicyExporter(actor_critic_cpu, normalizer_cpu)
    trace_model.eval()

    # =====================================================
    # 3. 创建 Dummy Input 并执行 Tracing
    # =====================================================
    print(f"Generating dummy history_obs with shape: (1, {input_history_dim})")
    dummy_history = torch.randn(1, input_history_dim)
    print(f"Generating dummy perceptive_obs with shape: (1, {input_perceptive_dim})")
    dummy_perceptive = torch.randn(1, input_perceptive_dim)

    print(f"Tracing model...")
    # 使用 torch.jit.trace
    # strict=False 允许一些非 Tensor 的操作（通常对 MLP 没影响，但更安全）
    traced_script_module = torch.jit.trace(trace_model, (dummy_history, dummy_perceptive), strict=False)

    # =====================================================
    # 4. 保存模型
    # =====================================================
    traced_script_module.save(file_path)
    
    print(f"✅ Successfully exported PIM Policy to JIT: {file_path}")
    
    # =====================================================
    # 5. 验证 (可选)
    # =====================================================
    try:
        print("Verifying exported model...")
        loaded_model = torch.jit.load(file_path)
        with torch.no_grad():
            output_original = trace_model((dummy_history, dummy_perceptive))
            output_jit = loaded_model((dummy_history, dummy_perceptive))

        # 检查误差
        diff = torch.max(torch.abs(output_original - output_jit)).item()
        print(f"Verification Max Diff: {diff:.6f}")
        if diff < 1e-5:
            print("Verification Passed! Output matches.")
        else:
            print("WARNING: Verification output mismatch!")
    except Exception as e:
        print(f"Verification failed: {e}")

# ==========================================
# 使用示例 (假设你在 train.py 或 play.py 中)
# ==========================================
# if __name__ == "__main__":
    # 假设 runner 是你的 PPO Runner
    # actor_critic = runner.alg.actor_critic
    # obs_normalizer = runner.obs_normalizer  # 获取归一化器
    
    # 假设历史输入维度是 135 (5帧历史 * 27维观测)，观测维度是 96
    # export_pim_actor_critic_as_onnx(
    #     actor_critic,
    #     path="exported_models",
    #     name="him_policy",
    #     input_history_dim=135,
    #     input_perceptive_dim=96,
    #     normalizer=obs_normalizer
    # )