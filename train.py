
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from model import FullyConnectedNN
from model import CNN
from data_loader import load_mnist_data
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, train_loader, val_loader, unlabeled_loader, optimizer, criterion, epochs, device):
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        total = 0
        correct = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        train_accuracy = correct / total
        val_accuracy = evaluate(model, val_loader, device, return_accuracy=True)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss / (batch_idx + 1)}, '
              f'Train Accuracy: {train_accuracy:.2%}, Val Accuracy: {val_accuracy:.2%}')

def evaluate(model, data_loader, device, return_accuracy=False):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = correct / total
    if return_accuracy:
        return accuracy
    else:
        print(f'Accuracy: {accuracy:.2%}')

def train_and_evaluate(epochs, batch_size, train_size, val_split, learning_rate, device):
    train_accuracy_history = []
    val_accuracy_history = []
    x = np.array(range(0, epochs))
    
    train_loader, val_loader, test_loader, unlabeled_loader = load_mnist_data(batch_size, train_size, val_split)

    model = CNN(num_classes=10).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
    criterion = torch.nn.CrossEntropyLoss()  # 输出仍然采用交叉熵损失函数，不过在内部使用了softmax函数

    print(f"Training with {train_size * 100}% of the training data")
    train_loader, val_loader, test_loader, unlabeled_loader = load_mnist_data(batch_size, train_size, val_split)
    # train过程
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        total = 0
        correct = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        train_accuracy = correct / total
        train_accuracy_history.append(train_accuracy)
        val_accuracy = evaluate(model, val_loader, device, return_accuracy=True)
        val_accuracy_history.append(val_accuracy)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss / (batch_idx + 1)}, '
              f'Train Accuracy: {train_accuracy:.2%}, Val Accuracy: {val_accuracy:.2%}')

    # train(model, train_loader, val_loader, unlabeled_loader, optimizer, criterion, epochs, device)
    evaluate(model, test_loader, device)

# def train_and_evaluate(epochs, batch_size, train_size, val_split, hidden_size, learning_rate):
#     train_loader, val_loader, test_loader = load_mnist_data(batch_size, train_size, val_split)
#     x = np.array(range(0, epochs))

#     input_size = 28 * 28  # minist图像大小为28*28像素，同时也是输入层神经元数量
#     num_classes = 10  # minist有10个类别，同时也是输出层的神经元数量
#     # 创建FCN模型
#     model = FullyConnectedNN(input_size, hidden_size, num_classes).to(device)
#     criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#     train_accuracy_history = []
#     val_accuracy_history = []

#     for epoch in range(epochs):
#         model.train()
#         for images, labels in train_loader:
#             images, labels = images.to(device), labels.to(device)

#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#         model.eval()
#         correct_train, total_train = 0, 0
#         with torch.no_grad():
#             for images, labels in train_loader:
#                 images, labels = images.to(device), labels.to(device)
#                 outputs = model(images)
#                 _, predicted = torch.max(outputs.data, 1)
#                 total_train += labels.size(0)
#                 correct_train += (predicted == labels).sum().item()
#         train_accuracy = correct_train / total_train
#         train_accuracy_history.append(train_accuracy)

#         correct_val, total_val = 0, 0
#         with torch.no_grad():
#             for images, labels in val_loader:
#                 images, labels = images.to(device), labels.to(device)
#                 outputs = model(images)
#                 _, predicted = torch.max(outputs.data, 1)
#                 total_val += labels.size(0)
#                 correct_val += (predicted == labels).sum().item()
#         val_accuracy = correct_val / total_val
#         val_accuracy_history.append(val_accuracy)

#         print(f"Epoch [{epoch+1}/{epochs}], Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}")

#     # 对测试集进行测试
#     correct_test, total_test = 0, 0
#     with torch.no_grad():
#         for images, labels in test_loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total_test += labels.size(0)
#             correct_test += (predicted == labels).sum().item()
#     test_accuracy = correct_test / total_test
#     print(f"Test Accuracy: {test_accuracy:.4f}")

    # 历次准确率绘制
    plt.plot(train_accuracy_history, label="Train Accuracy")
    plt.plot(val_accuracy_history, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.scatter(x, train_accuracy_history, marker='o')
    plt.scatter(x, val_accuracy_history, marker='o')
    plt.show()

# # 训练并进行模型评价
# epochs = 20
# batch_size = 64
# train_size = [0.1, 0.3, 0.5, 0.7, 1.0]  # Train size就是第一问训练集大小的改变
# val_split = 0.1  # 从样本中抽取10%作为验证集
# hidden_size = 128  # 隐藏层神经元数量
# learning_rate = 0.001  # 学习率

# for size in train_size:
#     print(f"Training with {size * 100}% of the training data")
#     train_and_evaluate(epochs, batch_size, size, val_split, hidden_size, learning_rate)

# 参数设置，就在其中
train_and_evaluate(epochs=10, batch_size=64, train_size=0.1, val_split=0.1, learning_rate=0.001, device=device)
