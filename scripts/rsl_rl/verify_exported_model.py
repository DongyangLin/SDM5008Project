import torch
import onnxruntime as ort
import numpy as np
import os

def print_shape_info(name, inputs, outputs):
    """辅助函数：美化打印 Shape 信息"""
    print(f"   [{name}] Input Shapes:")
    for i, inp in enumerate(inputs):
        # 兼容处理：如果是Tensor/Array直接打shape，如果是meta info则取属性
        shape = inp.shape if hasattr(inp, 'shape') else inp
        print(f"      - Input {i}: {shape}")
    
    print(f"   [{name}] Output Shapes:")
    for i, out in enumerate(outputs):
        shape = out.shape if hasattr(out, 'shape') else out
        print(f"      - Output {i}: {shape}")

def compare_models(onnx_path, pt_path, input_dim):
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
    # Shape: (1, input_dim) -> Batch size = 1
    dummy_input_tensor = torch.randn(1, input_dim, dtype=torch.float32)
    dummy_input_numpy = dummy_input_tensor.detach().cpu().numpy()

    print(f"🎲 Generated dummy input with shape: {dummy_input_tensor.shape}")
    print("-" * 60)

    # ---------------------------------------------------------
    # 3. 运行 JIT (.pt) 模型
    # ---------------------------------------------------------
    jit_output = None
    try:
        print(f"⚡ Loading JIT model...")
        jit_model = torch.jit.load(pt_path)
        jit_model.eval() 
        
        # 运行推理
        with torch.no_grad():
            jit_output_tensor = jit_model(dummy_input_tensor)
            
        # [新增] 打印 JIT 形状信息
        # JIT 模型的输入输出形状通常需要通过运行一次来动态获取，或者检查 graph
        # 这里我们直接打印实际运行时的 tensor shape
        output_tensors = jit_output_tensor if isinstance(jit_output_tensor, tuple) else (jit_output_tensor,)
        print_shape_info("JIT (Runtime)", [dummy_input_tensor], output_tensors)

        # 处理输出 (支持多输出)
        if isinstance(jit_output_tensor, tuple):
             jit_output = [t.detach().cpu().numpy() for t in jit_output_tensor]
             print(f"✅ JIT Inference Successful. (Multi-output detected)")
        else:
             jit_output = [jit_output_tensor.detach().cpu().numpy()]
             print(f"✅ JIT Inference Successful.")
        
    except Exception as e:
        print(f"❌ JIT Execution Failed: {e}")
        return

    print("-" * 60)

    # ---------------------------------------------------------
    # 4. 运行 ONNX 模型
    # ---------------------------------------------------------
    onnx_output = None
    try:
        print(f"📦 Loading ONNX model...")
        ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        
        # [新增] 打印 ONNX 元数据中的静态形状信息
        print(f"   [ONNX Metadata] Model Inputs:")
        for meta in ort_session.get_inputs():
            print(f"      - Name: {meta.name}, Shape: {meta.shape}, Type: {meta.type}")
        
        print(f"   [ONNX Metadata] Model Outputs:")
        for meta in ort_session.get_outputs():
            print(f"      - Name: {meta.name}, Shape: {meta.shape}, Type: {meta.type}")

        # 获取输入层名称
        input_name = ort_session.get_inputs()[0].name
        input_shapes = [ort_session.get_inputs()[i].shape for i in range(ort_session.get_inputs().__len__())]
        print("ONNX input shape:", input_shapes)
        
        # 运行推理
        ort_inputs = {input_name: dummy_input_numpy}
        ort_outs = ort_session.run(None, ort_inputs)
        
        onnx_output = ort_outs # 这是一个 list
        
        # [新增] 打印 ONNX 运行时形状
        print_shape_info("ONNX (Runtime)", [dummy_input_numpy], onnx_output)
        print(f"✅ ONNX Inference Successful.")

    except Exception as e:
        print(f"❌ ONNX Execution Failed: {e}")
        return

    # ---------------------------------------------------------
    # 5. 比较结果
    # ---------------------------------------------------------
    print("=" * 60)
    print("📊 Comparison Results:")
    
    # 检查输出数量是否一致
    if len(jit_output) != len(onnx_output):
        print(f"❌ Error: Output count mismatch! JIT: {len(jit_output)}, ONNX: {len(onnx_output)}")
        return

    # 逐个输出比较
    for i in range(len(jit_output)):
        jit_out = jit_output[i]
        onnx_out = onnx_output[i]
        
        # 展平以便比较 (防止 shape 维度只有 1 的差异，例如 (1, 12) vs (12,))
        diff = np.abs(jit_out.flatten() - onnx_out.flatten())
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        print(f"   [Output {i}] Shape Check: JIT {jit_out.shape} vs ONNX {onnx_out.shape}")
        print(f"   [Output {i}] Max Diff:    {max_diff:.8f}")
        print(f"   [Output {i}] Mean Diff:   {mean_diff:.8f}")
        
        if max_diff < 1e-4:
            print(f"   ✅ Output {i} MATCHED!")
        else:
            print(f"   ⚠️ Output {i} MISMATCH!")

    print("-" * 60)
    print("👀 Visual Check (First 5 values of Output 0):")
    print(f"   JIT:  {jit_output[0].flatten()[:5]}")
    print(f"   ONNX: {onnx_output[0].flatten()[:5]}")
    print("=" * 60)

if __name__ == "__main__":
    # ================= 配置区 =================
    MODEL_DIR = "/home/user/SDM5008/limxtron1lab-main/logs/rsl_rl/pf_him_stair/2025-12-16_Stable_Phase_3/exported"
    ONNX_NAME = "him_actor_critic.onnx"
    PT_NAME = "him_actor_critic.pt"
    
    INPUT_DIM = 165 
    # ==========================================

    onnx_file = os.path.join(MODEL_DIR, ONNX_NAME)
    pt_file = os.path.join(MODEL_DIR, PT_NAME)

    compare_models(onnx_file, pt_file, INPUT_DIM)