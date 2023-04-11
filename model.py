# 建立全连接神经网络模型
import torch
import torch.nn as nn
import torch.nn.functional as F

# fcn
class FullyConnectedNN(nn.Module):  # 含有两个隐藏层，一个输出层
    def __init__(self, input_size, hidden_size1,hidden_size2, num_classes):
        super(FullyConnectedNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 原始特征表示为一个28*28=784的一维向量
        x = F.relu(self.fc1(x))  # 使用relu函数激活，就是所谓活化函数
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# cnn
class CNN(nn.Module):  # 含有两个卷积层、两个最大池化层和两个全连接层
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):  # 卷积层和全连接层使用relu函数激活
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
