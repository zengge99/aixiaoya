import torch
import torch.nn.functional as F
import os
import sys
import numpy as np
import onnxruntime as ort
from main import NERModel, BERT_LOCAL_FOLDER, MODEL_WEIGHTS_PATH, MAX_LEN, NUM_LABELS

ONNX_PATH = "movie_ner_bert.onnx"

def export_and_verify():
    # --- 1. 加载模型 ---
    print(f"正在从 {BERT_LOCAL_FOLDER} 加载架构...")

    model = NERModel(BERT_LOCAL_FOLDER)
    
    print(f"正在加载权重: {MODEL_WEIGHTS_PATH}")
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location='cpu'))
    model.eval()

    # --- 2. 准备虚拟输入 (Dummy Inputs) ---
    # BERT 通常需要两个输入：IDs 和 Mask
    dummy_input_ids = torch.randint(0, 20000, (1, MAX_LEN), dtype=torch.long)
    dummy_mask = torch.ones(1, MAX_LEN, dtype=torch.long)
    
    # 组合为元组
    dummy_inputs = (dummy_input_ids, dummy_mask)

    # --- 3. 导出 ONNX ---
    # 定义输入输出名称
    input_names = ["input_ids", "attention_mask"]
    output_names = ["logits"]
    
    # 设置动态轴：支持不同的 batch_size 和 序列长度
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
        "logits": {0: "batch_size", 1: "seq_len"}
    }

    print(f"正在导出 ONNX 模型 (Opset 18)...")
    torch.onnx.export(
        model, 
        dummy_inputs, 
        ONNX_PATH, 
        opset_version=18,
        input_names=input_names, 
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True
    )
    print(f"✅ 导出成功: {ONNX_PATH}")

    # --- 4. 推理对比 ---
    print("\n运行推理验证...")
    with torch.no_grad():
        # 获取 PyTorch 输出 (并转为概率，方便观察)
        torch_logits = model(dummy_input_ids, dummy_mask)
        torch_output = F.softmax(torch_logits, dim=-1).numpy()

    # 启动 ONNX Runtime
    ort_session = ort.InferenceSession(ONNX_PATH)
    
    # 准备 ONNX 输入字典
    ort_inputs = {
        "input_ids": dummy_input_ids.numpy(),
        "attention_mask": dummy_mask.numpy()
    }
    
    # 获取 ONNX 输出
    ort_logits = ort_session.run(None, ort_inputs)[0]
    ort_output = torch.softmax(torch.from_numpy(ort_logits), dim=-1).numpy()

    # --- 5. 详细数值打印 ---
    print("\n" + "="*60)
    print(f"{'模型节点':<15} | {'数值示例 (首个 Token 的 3 类概率)':<40}")
    print("-"*60)
    # 打印第一个 batch, 第一个 token 的概率分布 [P(O), P(B), P(I)]
    print(f"{'PyTorch Probs':<15} | {torch_output[0, 0, :]}")
    print(f"{'ONNX Probs':<15} | {ort_output[0, 0, :]}")
    print("-"*60)

    # --- 6. 精度量化对比 ---
    abs_diff = np.abs(torch_output - ort_output)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)
    
    print(f"最大绝对误差: {max_diff:.2e}")
    print(f"平均绝对误差: {mean_diff:.2e}")
    
    if np.allclose(torch_output, ort_output, atol=1e-5):
        print("✅ 结论：精度匹配 (Tolerance < 1e-5)")
    else:
        print("⚠️ 结论：存在微小差异（可能是浮点精度或 Opset 算子实现差异）")
    print("="*60 + "\n")

if __name__ == "__main__":
    # 确保 main.py 中的相关变量和类在此作用域内可用
    export_and_verify()