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
        # 提取核心模块 (深拷贝防止影响原模型) / Extract core modules (deep copy to avoid affecting original model)
        self.estimator = copy.deepcopy(actor_critic.estimator)
        self.actor = copy.deepcopy(actor_critic.actor)
        self.dim_nonperceptive_obs = actor_critic.dim_nonperceptive_obs
        self.dim_perceptive_obs = actor_critic.dim_perceptive_obs
        
        # 归一化 / Normalization (Fusion)
        self.has_normalizer = False
        if normalizer is not None:
            self.has_normalizer = True
            self.register_buffer('obs_mean', normalizer.running_ms.mean.clone())
            self.register_buffer('obs_var', normalizer.running_ms.var.clone())
            print(f"[PIMExporter] Fused Normalization: Mean shape {self.obs_mean.shape}, Var shape {self.obs_var.shape}")
        else:
            print("[PIMExporter] WARNING: No normalizer provided. The exported model expects NORMALIZED inputs.")

    def forward(self, obs_history, obs_perceptive):
        # 归一化 / In-graph Normalization
        obs_history = obs_history
        if self.has_normalizer:
            obs_history = (obs_history - self.obs_mean) / (torch.sqrt(self.obs_var) + 1e-4)

        # Estimator 推理 / Estimator Inference
        vel, latent = self.estimator(obs_history, obs_perceptive)

        # 拼接 Actor 输入 / Concatenate Actor input
        obs_current = obs_history[:, -self.dim_nonperceptive_obs:]   # 取 obs_history 最新的那一帧观测 / Get the latest one-step observation from obs_history
        actor_input = torch.cat((obs_current, obs_perceptive, vel, latent), dim=-1)

        # Actor 推理 / Actor Inference
        actions = self.actor(actor_input)
        
        return actions

def export_pim_actor_critic_as_onnx(
    actor_critic,
    path,
    name="pim_actor_critic",
    input_history_dim=None,
    input_perceptive_dim=None,
    normalizer=None
):
    """
    导出 PIM 策略为 ONNX 格式 (支持自动推断输入维度)。
    Export PIM policy to ONNX format (supports automatic input dimension inference).
    """
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, name + ".onnx")

    # =====================================================
    # 1. 自动推断输入维度 / Auto-detect Input Dim
    # =====================================================
    if input_history_dim is None:
        try:
            # 方法 1: 直接读取 ActorCritic 的属性 / Preferentially try Method 1: Directly read ActorCritic properties
            if hasattr(actor_critic, "dim_nonperceptive_obs") and hasattr(actor_critic, "history_length"):
                input_history_dim = actor_critic.history_length * actor_critic.dim_nonperceptive_obs
                print(f"[Auto-Detect] Found input_history_dim from actor_critic properties: {input_history_dim}")

            # 方法 2: 检查 Estimator Encoder 的第一层线性层 / Method 2: Check Estimator Encoder's first linear layer
            elif hasattr(actor_critic, "estimator"):
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
            # 方法 1: 直接读取 ActorCritic 的属性 / Preferentially try Method 1: Directly read ActorCritic properties
            if hasattr(actor_critic, "dim_nonperceptive_obs"):
                input_perceptive_dim = actor_critic.dim_perceptive_obs
                print(f"[Auto-Detect] Found input_perceptive_dim from actor_critic properties: {input_perceptive_dim}")

            # 方法 2: 检查 Estimator Encoder 的第一层线性层 / Method 2: Check Estimator Encoder's first linear layer
            elif hasattr(actor_critic, "estimator"):
                input_perceptive_dim = actor_critic.estimator.dim_perceptive_obs
                print(f"[Auto-Detect] Found input_perceptive_dim from estimator properties: {input_perceptive_dim}")

            if input_perceptive_dim is None:
                raise ValueError("Could not auto-detect input_perceptive_dim.")
                
        except Exception as e:
            print(f"Error auto-detecting input_perceptive_dim: {e}")
            print("Please provide input_perceptive_dim manually.")
            return
        
    # =====================================================
    # 2. 准备模型 (转移到 CPU) / Prepare model (move to CPU)
    # =====================================================
    actor_critic_cpu = actor_critic
    if next(actor_critic.parameters()).is_cuda:
        actor_critic_cpu = copy.deepcopy(actor_critic).cpu()
    
    normalizer_cpu = normalizer
    if normalizer is not None and hasattr(normalizer, 'running_ms'):
        if normalizer.running_ms.mean.is_cuda:
            normalizer_cpu = copy.deepcopy(normalizer)
            normalizer_cpu.running_ms.mean = normalizer_cpu.running_ms.mean.cpu()
            normalizer_cpu.running_ms.var = normalizer_cpu.running_ms.var.cpu()

    # 实例化导出包装器 / Instantiate Export Wrapper
    export_model = PIMPolicyExporter(actor_critic_cpu, normalizer_cpu)
    export_model.eval()

    # =====================================================
    # 3. 创建 Dummy Input / Create Dummy Input
    # =====================================================
    print(f"Generating dummy history_obs with shape: (1, {input_history_dim})")
    dummy_history = torch.randn(1, input_history_dim)
    print(f"Generating dummy perceptive_obs with shape: (1, {input_perceptive_dim})")
    dummy_perceptive = torch.randn(1, input_perceptive_dim)

    # =====================================================
    # 4. 导出 ONNX 模型 / Export ONNX model
    # =====================================================
    torch.onnx.export(
        export_model,
        (dummy_history, dummy_perceptive),
        file_path,
        verbose=False,
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
    Export PIM policy to TorchScript (JIT) format (.pt).
    Uses torch.jit.trace for tracing.
    """
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, name + ".pt")
    
    # =====================================================
    # 1. 自动推断输入维度 / Auto-detect Input Dim
    # =====================================================
    if input_history_dim is None:
        try:
            # 方法 1: 直接读取 ActorCritic 的属性 / Preferentially try Method 1: Directly read ActorCritic properties
            if hasattr(actor_critic, "dim_nonperceptive_obs") and hasattr(actor_critic, "history_length"):
                input_history_dim = actor_critic.history_length * actor_critic.dim_nonperceptive_obs
                print(f"[Auto-Detect] Found input_history_dim from actor_critic properties: {input_history_dim}")

            # 方法 2: 检查 Estimator Encoder 的第一层线性层 / Alternatively try Method 2: Check the first linear layer of Estimator Encoder
            elif hasattr(actor_critic, "estimator"):
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
            # 方法 1: 直接读取 ActorCritic 的属性 / Preferentially try Method 1: Directly read ActorCritic properties
            if hasattr(actor_critic, "dim_nonperceptive_obs"):
                input_perceptive_dim = actor_critic.dim_perceptive_obs
                print(f"[Auto-Detect] Found input_perceptive_dim from actor_critic properties: {input_perceptive_dim}")

            # 方法 2: 检查 Estimator Encoder 的第一层线性层 / Alternatively try Method 2: Check the first linear layer of Estimator Encoder
            elif hasattr(actor_critic, "estimator"):
                input_perceptive_dim = actor_critic.estimator.dim_perceptive_obs
                print(f"[Auto-Detect] Found input_perceptive_dim from estimator properties: {input_perceptive_dim}")

            if input_perceptive_dim is None:
                raise ValueError("Could not auto-detect input_perceptive_dim.")
                
        except Exception as e:
            print(f"Error auto-detecting input_perceptive_dim: {e}")
            print("Please provide input_perceptive_dim manually.")
            return

    # =====================================================
    # 2. 准备模型 (转移到 CPU 并去除梯度) / Prepare model (move to CPU and remove gradients)
    # =====================================================
    # 必须确保所有子模块都在 CPU 上 / Must ensure all sub-modules are on CPU
    actor_critic_cpu = actor_critic
    if next(actor_critic.parameters()).is_cuda:
        actor_critic_cpu = copy.deepcopy(actor_critic).cpu()
    
    normalizer_cpu = normalizer
    if normalizer is not None and hasattr(normalizer, 'running_ms'):
        if normalizer.running_ms.mean.is_cuda:
            normalizer_cpu = copy.deepcopy(normalizer)
            normalizer_cpu.running_ms.mean = normalizer_cpu.running_ms.mean.cpu()
            normalizer_cpu.running_ms.var = normalizer_cpu.running_ms.var.cpu()

    # 使用之前定义的包装器 (PIMPolicyExporter) / Use the previously defined wrapper (PIMPolicyExporter)
    trace_model = PIMPolicyExporter(actor_critic_cpu, normalizer_cpu)
    trace_model.eval()

    # =====================================================
    # 3. 创建 Dummy Input 并执行追踪 / Create Dummy Input and Perform Tracing
    # =====================================================
    print(f"Generating dummy history_obs with shape: (1, {input_history_dim})")
    dummy_history = torch.randn(1, input_history_dim)
    print(f"Generating dummy perceptive_obs with shape: (1, {input_perceptive_dim})")
    dummy_perceptive = torch.randn(1, input_perceptive_dim)

    print(f"Tracing model...")
    traced_script_module = torch.jit.trace(trace_model, (dummy_history, dummy_perceptive), strict=False)

    # =====================================================
    # 4. 保存模型 / Save the model
    # =====================================================
    traced_script_module.save(file_path)
    print(f"✅ Successfully exported PIM Policy to JIT: {file_path}")
    
    # =====================================================
    # 5. 验证 (可选) / Verification (optional)
    # =====================================================
    try:
        print("Verifying exported model...")
        loaded_model = torch.jit.load(file_path)
        with torch.no_grad():
            output_original = trace_model((dummy_history, dummy_perceptive))
            output_jit = loaded_model((dummy_history, dummy_perceptive))

        # 检查误差 / Check for discrepancies
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