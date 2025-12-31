import torch
import pickle
import os
import numpy as np
import onnxruntime as ort
from main_opt import Extractor, MODEL_PATH, VOCAB_PATH, MAX_LEN, EMBED_DIM, HIDDEN_DIM

# 配置保持一致
MODEL_PATH = "movie_model.pth"
VOCAB_PATH = "vocab.pkl"
ONNX_PATH = "movie_extractor.onnx"
MAX_LEN = 300

def export_and_verify():
    # 1. 加载模型
    with open(VOCAB_PATH, 'rb') as f:
        char_to_idx = pickle.load(f)
    
    model = Extractor(len(char_to_idx), embed_dim=64, hidden_dim=128)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()

    # 2. 导出 ONNX
    dummy_input = torch.randint(0, len(char_to_idx), (1, MAX_LEN), dtype=torch.long)
    torch.onnx.export(model, dummy_input, ONNX_PATH, opset_version=15,
                      input_names=["input_ids"], output_names=["probs"],
                      dynamic_axes={"input_ids": {0: "batch_size"}})

    # 3. 推理对比
    with torch.no_grad():
        torch_output = model(dummy_input).numpy()

    ort_session = ort.InferenceSession(ONNX_PATH)
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_output = ort_session.run(None, ort_inputs)[0]

    # --- 4. 详细数值打印 ---
    print("\n" + "="*50)
    print(f"{'模型节点':<15} | {'数值示例 (前5个元素)':<35}")
    print("-"*50)
    # 打印前5个概率值进行肉眼对比
    print(f"{'PyTorch Probs':<15} | {torch_output.flatten()[:5]}")
    print(f"{'ONNX Probs':<15} | {ort_output.flatten()[:5]}")
    print("-"*50)

    # 5. 精度量化对比
    abs_diff = np.abs(torch_output - ort_output)
    max_diff = np.max(abs_diff)
    
    print(f"最大绝对误差: {max_diff:.2e}")
    
    # 使用 np.allclose 进行自动化判定 (默认阈值 rtol=1e-05, atol=1e-08)
    if np.allclose(torch_output, ort_output, atol=1e-5):
        print("✅ 结论：精度完全匹配 (Tolerance < 1e-5)")
    else:
        print("❌ 结论：精度存在显著差异，请检查模型结构")
    print("="*50 + "\n")

if __name__ == "__main__":
    export_and_verify()

