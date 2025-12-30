import torch
import pickle
import os
from main import Extractor  # 替换为你的 main.py 文件名

# 全局配置（和 main.py 保持一致）
MODEL_PATH = "movie_model.pth"
VOCAB_PATH = "vocab.pkl"
MAX_LEN = 300
EMBED_DIM = 64
HIDDEN_DIM = 128

def export_onnx():
    # 加载词表和 PyTorch 模型
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VOCAB_PATH):
        print("请先训练模型生成 movie_model.pth 和 vocab.pkl")
        return

    with open(VOCAB_PATH, 'rb') as f:
        char_to_idx = pickle.load(f)
    
    model = Extractor(len(char_to_idx), embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()

    # 构造 dummy input（和实际输入形状一致）
    dummy_input = torch.zeros(1, MAX_LEN, dtype=torch.long)  # [batch_size, seq_len]

    # 导出 ONNX 模型
    onnx_path = "movie_extractor.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input_ids"],
        output_names=["probs"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "probs": {0: "batch_size", 1: "seq_len"}
        },
        verbose=False
    )
    print(f"✅ ONNX 模型已导出至: {onnx_path}")
    print(f"📌 需和 vocab.pkl 一起分发")

if __name__ == "__main__":
    export_onnx()
