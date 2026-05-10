import os
# 解决可能的多线程冲突问题（保留你的环境配置）
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import torch
import torch.nn as nn
import torch.optim as optim
import random

# ===================== 1. 任务定义 & 数据生成 =====================
# 任务：5个汉字的句子，判断 "你" 在第几个位置（0~4，5分类）
VOCAB = ["我", "他", "她", "它", "们", "这", "那", "好", "坏", "是"]  # 常用字
TARGET_CHAR = "你"  # 目标字
SENTENCE_LEN = 5    # 固定5个字
NUM_CLASSES = 5     # 分类数：0,1,2,3,4 对应位置

# 随机生成一条样本：5个字，必须包含且只包含1个"你"
def generate_sample():
    sentence = [random.choice(VOCAB) for _ in range(SENTENCE_LEN)]
    target_pos = random.randint(0, SENTENCE_LEN - 1)  # 随机"你"的位置
    sentence[target_pos] = TARGET_CHAR
    return sentence, target_pos

# 构建词汇表（映射：字 -> 数字）
vocab_set = VOCAB + [TARGET_CHAR]
word2idx = {word: i for i, word in enumerate(vocab_set)}
VOCAB_SIZE = len(word2idx)

# 批量生成数据
def generate_data(num_samples):
    data, labels = [], []
    for _ in range(num_samples):
        sent, label = generate_sample()
        # 文字转数字索引
        sent_idx = [word2idx[w] for w in sent]
        data.append(sent_idx)
        labels.append(label)
    return torch.LongTensor(data), torch.LongTensor(labels)

# 生成训练集 & 测试集
train_data, train_labels = generate_data(8000)
test_data, test_labels = generate_data(2000)

# ===================== 2. 模型定义：RNN & LSTM =====================
# 通用循环神经网络模型
class TextClassifier(nn.Module):
    def __init__(self, model_type="lstm", vocab_size=VOCAB_SIZE, embed_dim=16, hidden_size=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # 词嵌入层

        # 选择 RNN 或 LSTM
        if model_type == "rnn":
            self.rnn = nn.RNN(embed_dim, hidden_size, batch_first=True)
        elif model_type == "lstm":
            self.rnn = nn.LSTM(embed_dim, hidden_size, batch_first=True)

        # 分类头
        self.fc = nn.Linear(hidden_size, NUM_CLASSES)

    def forward(self, x):
        # x: [batch, seq_len]
        x = self.embedding(x)  # [batch, seq_len, embed_dim]
        out, _ = self.rnn(x)   # out: [batch, seq_len, hidden_size]
        # 取最后一个时间步输出做分类
        last_out = out[:, -1, :]
        logits = self.fc(last_out)
        return logits

# ===================== 3. 训练配置 =====================
BATCH_SIZE = 64
EPOCHS = 10
LR = 0.001

# 选择模型："rnn" 或 "lstm"
MODEL_TYPE = "lstm"
model = TextClassifier(model_type=MODEL_TYPE)

# 损失函数 & 优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# 数据加载器
train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ===================== 4. 开始训练（去掉tqdm，用简单打印） =====================
print(f"===== 开始训练 {MODEL_TYPE.upper()} 模型 =====")
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    # 去掉tqdm，直接遍历数据加载器
    for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 每10个batch打印一次进度（替代tqdm的进度提示）
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"\nEpoch {epoch+1} 结束，平均损失: {avg_loss:.4f}\n")

# ===================== 5. 测试模型 =====================
def test_model(model, test_data, test_labels):
    model.eval()
    with torch.no_grad():
        outputs = model(test_data)
        pred = torch.argmax(outputs, dim=1)
        acc = (pred == test_labels).sum().item() / len(test_labels)
    return acc

test_acc = test_model(model, test_data, test_labels)
print(f"===== 测试集准确率: {test_acc:.4f} =====")

# ===================== 6. 手动测试（直观体验） =====================
def predict(sentence):
    # 输入必须是5个汉字
    assert len(sentence) == 5
    sent_idx = [word2idx.get(w, 0) for w in sentence]
    x = torch.LongTensor([sent_idx])
    model.eval()
    with torch.no_grad():
        out = model(x)
        pos = torch.argmax(out).item()
    print(f"输入文本：{''.join(sentence)}")
    print(f"模型预测：'你' 在第 {pos} 位（真实位置：{sentence.index('你')}）\n")

# 测试几个例子
predict(["我", "你", "他", "她", "它"])
predict(["这", "好", "你", "坏", "那"])
predict(["们", "是", "我", "你", "她"])