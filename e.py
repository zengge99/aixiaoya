import torch
import pickle
import os
from main import Extractor  # 保持和你的main.py关联

# 全局配置（和 main.py 严格一致）
MODEL_PATH = "movie_model.pth"
VOCAB_PATH = "vocab.pkl"
MAX_LEN = 300
EMBED_DIM = 64
HIDDEN_DIM = 128
ONNX_PATH = "movie_extractor.onnx"  # 区分修复版模型

def export_onnx():
    # 1. 前置检查
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 未找到模型文件：{MODEL_PATH}（请先训练模型）")
        return
    if not os.path.exists(VOCAB_PATH):
        print(f"❌ 未找到词表文件：{VOCAB_PATH}（请先训练模型）")
        return

    # 2. 加载词表和模型（严格匹配训练逻辑）
    with open(VOCAB_PATH, 'rb') as f:
        char_to_idx = pickle.load(f)
    
    model = Extractor(len(char_to_idx), embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()  # 关键1：强制禁用Dropout/BatchNorm

    # 3. 构造dummy input（固定batch_size=1，避免动态维度问题）
    dummy_input = torch.zeros(1, MAX_LEN, dtype=torch.long)  # [1, MAX_LEN]

    # 4. 导出ONNX（核心修复配置）
    print("📌 开始导出ONNX模型（修复GRU/Attention兼容问题）...")
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        opset_version=15,  # 关键2：升级到opset15，完美兼容GRU/LayerNorm
        input_names=["input_ids"],
        output_names=["probs"],
        dynamic_axes=None,  # 关键3：关闭动态batch，避免GRU算子近似实现
        training=torch.onnx.TrainingMode.EVAL,  # 关键4：强制推理模式
        do_constant_folding=True,  # 优化静态算子，提升精度
        keep_initializers_as_inputs=False,  # 减少冗余节点
        verbose=False
    )

    # 5. 优化ONNX模型（精简冗余算子，进一步提升精度）
    try:
        from onnxsim import simplify
        import onnx
        # 加载并精简模型
        onnx_model = onnx.load(ONNX_PATH)
        simplified_model, check = simplify(onnx_model)
        assert check, "ONNX模型精简后验证失败"
        onnx.save(simplified_model, ONNX_PATH)
        print(f"✅ ONNX模型精简完成")
    except ImportError:
        print("⚠️ 未安装onnx-simplifier（建议执行：pip install onnx-simplifier），跳过模型精简")
    except Exception as e:
        print(f"⚠️ 模型精简失败：{e}（不影响基础推理）")

    print(f"✅ 修复版ONNX模型已导出至: {ONNX_PATH}")
    print(f"📌 部署时请将推理代码中的模型路径改为：{ONNX_PATH}")

if __name__ == "__main__":
    export_onnx()