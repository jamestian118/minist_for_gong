import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm

# 定义参数 分别是批次大小、迭代次数、学习率
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
RATIO = 0.1

train_set = MNIST(root='./data', train=True, download=True, transform=ToTensor())
# 对原始数据集进行随机抽样
subset_size = int(len(train_set) * RATIO)
train_subset, _ = torch.utils.data.random_split(train_set, [subset_size, len(train_set) - subset_size])

# 使用新的数据集进行训练
train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)

test_set = MNIST(root='./data', train=False, download=True, transform=ToTensor())
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)



# 定义模型
class Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_heads):
        super().__init__()
        # 输入嵌入层，将输入数据映射到指定的隐藏层维度
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        # 初始化transformer编码层，d_model为隐藏层维度，nhead为多头注意力头数
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        # 将多个transformer编码层连接起来，形成一个完整的编码器
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 输出层，将隐藏层映射为输出维度
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 通过输入嵌入层将输入数据映射到隐藏层
        x = self.input_embedding(x)
        # 在第1维增加一维，batch_size=1
        x = x.unsqueeze(1)
        # 调换第0维和第1维，使得序列长度为第0维
        x = x.permute(1, 0, 2)
        # 将输入序列通过transformer编码层进行特征提取，得到上下文感知的特征向量
        x = self.transformer_encoder(x)
        # 取最后一帧输出作为整个序列的表示，即最终的特征向量
        x = x[-1, :, :]
        # 将特征向量通过输出层映射到指定的输出维度
        x = self.output_layer(x)
        return x


# 初始化模型、损失函数和优化器
model = Transformer(input_dim=28 * 28, hidden_dim=512, output_dim=10, num_layers=4, num_heads=8).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 初始化训练损失列表和训练精度列表
train_losses = []
train_accs = []

# 训练模型
for epoch in range(EPOCHS):
    running_loss = 0.0
    correct = 0
    total = 0

    for data, target in tqdm(train_loader):
        data = data.to(DEVICE)
        target = target.to(DEVICE)

        optimizer.zero_grad()

        output = model(data.view(-1, 28 * 28))
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # 统计训练损失
        running_loss += loss.item() * data.size(0)

        # 统计训练精度
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    epoch_loss = running_loss / len(train_set)
    epoch_acc = 100.0 * correct / total

    # 将训练损失和训练精度添加到列表中
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)

    print(f'Epoch {epoch + 1}/{EPOCHS} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.2f}%')

# 测试模型
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for data, target in test_loader:
        data = data.to(DEVICE)
        target = target.to(DEVICE)

        output = model(data.view(-1, 28 * 28))
        _, predicted = torch.max(output.data, 1)

        total += target.size(0)
        correct += (predicted == target).sum().item()

accuracy = 100.0 * correct / total
print(f'Test Accuracy: {accuracy:.2f}%')

x = np.array(range(1, epochs + 1))
# 绘制训练损失和训练精度曲线
plt.figure(figsize=(10, 5))
plt.plot(x, train_losses, label='Training Loss')
plt.plot(x, train_accs, label='Training Accuracy')
plt.scatter(x, train_losses, marker='o')
plt.scatter(x, train_accs, marker='o')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss/Accuracy')
plt.title('Training Loss and Accuracy')
plt.show()
