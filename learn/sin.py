import torch
import torch.nn as nn
import numpy as np

# 1. 生成sin函数的采样数据（模拟你给AI的“未知数据”）
# 随机生成5000个x值，范围在[0, 2π]
x = np.random.uniform(0, 2*np.pi, 5000)
# 计算对应的y=sin(x)
y = np.sin(x)

# 转换为torch张量（AI能处理的格式）
x_tensor = torch.tensor(x, dtype=torch.float32).reshape(-1, 1)
y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# 2. 定义简单的神经网络（只有1层隐藏层，符合万能逼近定理）
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 输入层(1个特征：x) → 隐藏层(32个神经元) → 输出层(1个值：y)
        self.layers = nn.Sequential(
            nn.Linear(1, 32),  # 线性变换
            nn.ReLU(),         # 激活函数（让网络能拟合非线性关系）
            nn.Linear(32, 1)   # 输出层
        )
    
    def forward(self, x):
        return self.layers(x)

# 3. 训练模型（拟合过程）
model = SimpleNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # 优化器（调整参数）
loss_fn = nn.MSELoss()  # 损失函数（衡量预测误差）

epochs = 1000  # 训练轮数
print("===== 开始训练模型 =====")
for epoch in range(epochs):
    model.train()
    y_pred = model(x_tensor)  # 预测值
    loss = loss_fn(y_pred, y_tensor)  # 计算损失
    
    optimizer.zero_grad()  # 清空梯度
    loss.backward()        # 反向传播（计算参数调整方向）
    optimizer.step()       # 调整参数
    
    # 每200轮打印一次损失（看误差是否变小）
    if (epoch + 1) % 200 == 0:
        print(f"训练轮数：{epoch+1}/{epochs} | 当前损失值：{loss.item():.6f}")

# 4. 纯命令行验证拟合效果（无可视化）
model.eval()
print("\n===== 拟合效果验证（数值对比） =====")

# 生成10个均匀分布的测试点（覆盖0到2π），方便对比
test_x_list = np.linspace(0, 2*np.pi, 10)
# 遍历每个测试点，打印真实值和拟合值
for idx, test_x in enumerate(test_x_list):
    # 计算真实sin值
    true_y = np.sin(test_x)
    # 计算AI拟合值
    test_x_tensor = torch.tensor([[test_x]], dtype=torch.float32)
    pred_y = model(test_x_tensor).detach().numpy()[0][0]
    # 计算单个点的误差
    error = abs(pred_y - true_y)
    # 格式化打印，保留6位小数，直观对比
    print(f"测试点{idx+1} | x = {test_x:.4f} | 真实sin(x) = {true_y:.6f} | AI拟合值 = {pred_y:.6f} | 误差 = {error:.6f}")

# 5. 统计整体拟合误差（量化效果）
print("\n===== 整体拟合误差统计 =====")
# 生成更多测试点做统计，结果更准确
x_test = np.linspace(0, 2*np.pi, 1000)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32).reshape(-1, 1)
y_pred_all = model(x_test_tensor).detach().numpy().reshape(-1)
y_true_all = np.sin(x_test)

# 计算关键误差指标
mean_error = np.mean(np.abs(y_pred_all - y_true_all))  # 平均绝对误差
max_error = np.max(np.abs(y_pred_all - y_true_all))    # 最大绝对误差
mse_error = np.mean((y_pred_all - y_true_all)**2)      # 均方误差（和训练时的损失一致）

print(f"平均绝对误差：{mean_error:.6f}")
print(f"最大绝对误差：{max_error:.6f}")
print(f"均方误差（MSE）：{mse_error:.6f}")
print("\n结论：误差越小，说明AI拟合sin函数的效果越好（通常平均误差<0.05即拟合效果优秀）")