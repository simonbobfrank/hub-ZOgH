import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import torch
import torch.nn as nn
import numpy as np



# 生成训练用的数据和标签
def generate_data(n_samples, dim):
    # 生成数据 X
    X = torch.randn(n_samples, dim)

    # 生成标签 y
    y = torch.argmax(X, dim=1)

    # 返回数据 X 和标签 y
    return X, y


# 向量维度 = 5，同时也是分类的类别数量（0、1、2、3、4 共5类）
dim = 5
# 训练集样本数量：1000个
n_train = 1000
# 测试集样本数量：200个
n_test = 200

# 调用上面写的函数，生成训练数据和标签
X_train, y_train = generate_data(n_train, dim)
# 生成测试数据和标签
X_test, y_test = generate_data(n_test, dim)


class SimpleNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()

        # 第一层全连接层（线性层）
        self.fc1 = nn.Linear(input_dim, 16)
        # ReLU激活函数
        self.relu = nn.ReLU()

        # 第二层全连接层
        self.fc2 = nn.Linear(16, num_classes)

    # 前向传播函数
    def forward(self, x):
        x = self.fc1(x)
        # 再经过ReLU激活
        x = self.relu(x)
        # 最后经过第二层全连接，输出结果
        x = self.fc2(x)
        # 返回输出
        return x

# 创建模型实例
model = SimpleNet(input_dim=dim, num_classes=dim)


# 定义损失函数：交叉熵损失
criterion = nn.CrossEntropyLoss()

# 定义优化器：Adam（最常用的优化器）
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练轮数
epochs = 50


# 打印提示文字
print("开始训练...")

# 循环训练 epochs 轮（50次）
for epoch in range(epochs):
    # 把模型设为训练模式（固定写法）
    model.train()

    # 前向传播：把训练数据输入模型，得到输出
    outputs = model(X_train)

    # 计算损失：预测值和真实标签差多少
    # 损失越小，模型越准
    loss = criterion(outputs, y_train)

    # 清空上一步的梯度（必须写，否则梯度会累加）
    optimizer.zero_grad()

    # 反向传播：计算每个参数的梯度
    loss.backward()

    # 更新参数：根据梯度调整模型权重
    optimizer.step()

    # 每训练10轮，打印一次结果
    if (epoch+1) % 10 == 0:
        # 把模型设为评估模式（不计算梯度，速度更快）
        model.eval()

        # with torch.no_grad()：不计算梯度，节省内存
        with torch.no_grad():
            # 把测试集输入模型，得到输出
            test_out = model(X_test)
            # 对每一行找最大值下标 → 预测标签
            pred = torch.argmax(test_out, dim=1)
            # 计算准确率：预测正确的数量 / 总数量
            acc = (pred == y_test).float().mean()

        # 打印：轮数 | 损失 | 测试集准确率
        print(f"Epoch {epoch+1:2d} | Loss: {loss.item():.3f} | Test Acc: {acc:.2%}")




# 换行打印
print("\n测试单个样本：")

# 生成1个5维的随机向量
x = torch.randn(1, dim)
print("输入向量：", x.numpy().round(2))

# 真实标签：找向量里最大值的下标
true_label = torch.argmax(x).item()
print("真实类别：", true_label)

# 不计算梯度
with torch.no_grad():
    # 把向量输入模型得到输出
    out = model(x)
    # 模型预测的标签
    pred_label = torch.argmax(out).item()

# 打印预测结果
print("模型预测：", pred_label)
# 打印是否预测正确
print("预测正确：", pred_label == true_label)