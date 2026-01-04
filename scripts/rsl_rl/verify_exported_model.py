import torch
import onnxruntime as ort
import numpy as np
import os

def compare_him_models(onnx_path, pt_path, input_dim):
    print("=" * 60)
    print(f"🔍 Starting Model Verification")
    print(f"   ONNX Path: {onnx_path}")
    print(f"   JIT Path:  {pt_path}")
    print(f"   Input Dim: {input_dim}")
    print("=" * 60)

    # 1. 检查文件是否存在
    if not os.path.exists(onnx_path) or not os.path.exists(pt_path):
        print("❌ Error: One of the model files does not exist.")
        return

    # 2. 生成虚拟输入 (Dummy Input)
    # 使用相同的随机种子，或者直接生成一次数据传给两者
    # Shape: (1, input_dim) -> Batch size = 1
    dummy_input_tensor = torch.randn(1, input_dim, dtype=torch.float32)
    dummy_input_numpy = dummy_input_tensor.detach().cpu().numpy()

    print(f"🎲 Generated dummy input with shape: {dummy_input_tensor.shape}")

    # ---------------------------------------------------------
    # 3. 运行 JIT (.pt) 模型
    # ---------------------------------------------------------
    try:
        print(f"⚡ Loading JIT model...")
        jit_model = torch.jit.load(pt_path)
        jit_model.eval() # 切换到评估模式
        
        with torch.no_grad():
            jit_output_tensor = jit_model(dummy_input_tensor)
            
        jit_output = jit_output_tensor.detach().cpu().numpy()
        print(f"✅ JIT Inference Successful. Output shape: {jit_output.shape}")
        
    except Exception as e:
        print(f"❌ JIT Execution Failed: {e}")
        return

    # ---------------------------------------------------------
    # 4. 运行 ONNX 模型
    # ---------------------------------------------------------
    try:
        print(f"📦 Loading ONNX model...")
        # 使用 CPU 运行以确保与 PyTorch CPU 对齐
        ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        
        # 获取输入层名称 (通常我们在导出时命名为 "obs_history")
        input_name = ort_session.get_inputs()[0].name
        output_name = ort_session.get_outputs()[0].name
        
        print(f"   ONNX Input Name: {input_name}")
        print(f"   ONNX Output Name: {output_name}")

        # 运行推理
        ort_inputs = {input_name: dummy_input_numpy}
        ort_outs = ort_session.run(None, ort_inputs)
        
        onnx_output = ort_outs[0]
        print(f"✅ ONNX Inference Successful. Output shape: {onnx_output.shape}")

    except Exception as e:
        print(f"❌ ONNX Execution Failed: {e}")
        return

    # ---------------------------------------------------------
    # 5. 比较结果
    # ---------------------------------------------------------
    print("-" * 60)
    print("📊 Comparison Results:")
    
    # 计算误差
    diff = np.abs(jit_output - onnx_output)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"   Max Difference:  {max_diff:.8f}")
    print(f"   Mean Difference: {mean_diff:.8f}")
    
    print("-" * 60)
    
    # 打印前几个数值供肉眼检查
    print("👀 Visual Check (First 5 output values):")
    print(f"   JIT:  {jit_output[0][:5]}")
    print(f"   ONNX: {onnx_output[0][:5]}")
    
    print("-" * 60)

    # 判定通过标准 (通常误差在 1e-5 以内是可以接受的，取决于浮点精度)
    if max_diff < 1e-4:
        print("🎉 SUCCESS: Models match! The outputs are consistent.")
    else:
        print("⚠️  WARNING: High output difference detected!")
        print("   Possible causes: precision mismatch (fp32 vs fp16), non-deterministic ops, or normalization mismatch.")


def compare_pim_models(onnx_path, pt_path, input_history_dim, input_perceptive_dim):
    print("=" * 60)
    print(f"🔍 Starting Model Verification")
    print(f"   ONNX Path: {onnx_path}")
    print(f"   JIT Path:  {pt_path}")
    print(f"   Input History Dim: {input_history_dim}")
    print(f"   Input Perceptive Dim: {input_perceptive_dim}")
    print("=" * 60)

    # 1. 检查文件是否存在
    if not os.path.exists(onnx_path) or not os.path.exists(pt_path):
        print("❌ Error: One of the model files does not exist.")
        return

    # 2. 生成虚拟输入 (Dummy Input)
    # 使用相同的随机种子，或者直接生成一次数据传给两者
    # Shape: (1, input_dim) -> Batch size = 1
    dummy_input_history = torch.randn(1, input_history_dim).detach().cpu().numpy()
    dummy_input_perceptive = torch.randn(1, input_perceptive_dim).detach().cpu().numpy()
    print(f"🎲 Generated dummy history input with shape: {dummy_input_history.shape}")
    print(f"🎲 Generated dummy perceptive input with shape: {dummy_input_perceptive.shape}")

    # ---------------------------------------------------------
    # 3. 运行 JIT (.pt) 模型
    # ---------------------------------------------------------
    try:
        print(f"⚡ Loading JIT model...")
        jit_model = torch.jit.load(pt_path)
        jit_model.eval() # 切换到评估模式
        
        with torch.no_grad():
            jit_output_tensor = jit_model((dummy_input_history, dummy_input_perceptive))

        jit_output = jit_output_tensor.detach().cpu().numpy()
        print(f"✅ JIT Inference Successful. Output shape: {jit_output.shape}")
        
    except Exception as e:
        print(f"❌ JIT Execution Failed: {e}")
        return

    # ---------------------------------------------------------
    # 4. 运行 ONNX 模型
    # ---------------------------------------------------------
    try:
        print(f"📦 Loading ONNX model...")
        # 使用 CPU 运行以确保与 PyTorch CPU 对齐
        ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        
        # 获取输入层名称 (通常我们在导出时命名为 "obs_history""obs_perceptive")
        input_name_history = ort_session.get_inputs()[0].name
        input_name_perceptive = ort_session.get_inputs()[1].name
        output_name = ort_session.get_outputs()[0].name

        print(f"   ONNX Input Name (History): {input_name_history}")
        print(f"   ONNX Input Name (Perceptive): {input_name_perceptive}")
        print(f"   ONNX Output Name: {output_name}")

        # 运行推理
        ort_inputs = {input_name_history: dummy_input_history, input_name_perceptive: dummy_input_perceptive}
        ort_outs = ort_session.run(None, ort_inputs)
        
        onnx_output = ort_outs[0]
        print(f"✅ ONNX Inference Successful. Output shape: {onnx_output.shape}")

    except Exception as e:
        print(f"❌ ONNX Execution Failed: {e}")
        return

    # ---------------------------------------------------------
    # 5. 比较结果
    # ---------------------------------------------------------
    print("-" * 60)
    print("📊 Comparison Results:")
    
    # 计算误差
    diff = np.abs(jit_output - onnx_output)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"   Max Difference:  {max_diff:.8f}")
    print(f"   Mean Difference: {mean_diff:.8f}")
    
    print("-" * 60)
    
    # 打印前几个数值供肉眼检查
    print("👀 Visual Check (First 5 output values):")
    print(f"   JIT:  {jit_output[0][:5]}")
    print(f"   ONNX: {onnx_output[0][:5]}")
    
    print("-" * 60)

    # 判定通过标准 (通常误差在 1e-5 以内是可以接受的，取决于浮点精度)
    if max_diff < 1e-4:
        print("🎉 SUCCESS: Models match! The outputs are consistent.")
    else:
        print("⚠️  WARNING: High output difference detected!")
        print("   Possible causes: precision mismatch (fp32 vs fp16), non-deterministic ops, or normalization mismatch.")


if __name__ == "__main__":
    # ================= 配置区 =================
    # 修改为你实际导出的路径和维度
    # HIM
    # MODEL_DIR = "/home/user/SDM5008/limxtron1lab-main/logs/rsl_rl/pf_him_stair/2025-12-16_Stable_Phase_3/exported"
    # ONNX_NAME = "him_actor_critic.onnx"
    # PT_NAME = "him_actor_critic.pt"
    
    # INPUT_DIM = 165 

    # onnx_file = os.path.join(MODEL_DIR, ONNX_NAME)
    # pt_file = os.path.join(MODEL_DIR, PT_NAME)

    # compare_him_models(onnx_file, pt_file, INPUT_DIM)

    # ==========================================
    # PIM
    MODEL_DIR = "/home/sustech/wzy/SDM5008/SDM5008Project/logs/rsl_rl/pf_pim_stair/2025-12-17_09-56-22/exported"
    ONNX_NAME = "pim_actor_critic.onnx"
    PT_NAME = "pim_actor_critic.pt"

    INPUT_HISTORY_DIM = 135  # 5帧历史 * 27维观测
    INPUT_PERCEPTIVE_DIM = 96  # 感知观测维度

    onnx_file = os.path.join(MODEL_DIR, ONNX_NAME)
    pt_file = os.path.join(MODEL_DIR, PT_NAME)

    compare_pim_models(onnx_file, pt_file, INPUT_HISTORY_DIM, INPUT_PERCEPTIVE_DIM)