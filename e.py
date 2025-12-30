import torch
from torch.onnx import export_options

# 1. 定义动态维度
# 假设 batch 大小和序列长度都是动态的
batch_dim = torch.export.Dim("batch", min=1, max=1024)
# 虽然你最大长是300，但定义为动态可以让模型更通用
seq_dim = torch.export.Dim("sequence", min=1, max=512) 

# 2. 准备模型和输入
model.eval()
dummy_input = torch.zeros(1, 300, dtype=torch.long)

# 3. 使用 dynamo_export (新版)
export_output = torch.onnx.dynamo_export(
    model,
    dummy_input,
    # 这里的 dynamic_shapes 是核心
    dynamic_shapes={
        "input_ids": {0: batch_dim, 1: seq_dim}
    }
)

# 4. 保存模型
export_output.save("movie_extractor_v2.onnx")