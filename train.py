import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from model import FullyConnectedNN
from model import CNN
from data_loader import load_mnist_data
import numpy as np
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_accuracy = 0


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
        global test_accuracy
        test_accuracy = accuracy
        print(f'Test Accuracy: {accuracy:.2%}')


# CNN
def train_and_evaluate_CNN(epochs, batch_size, train_size, val_split, learning_rate, device):
    train_accuracy_history = []  # 记录训练准确率
    val_accuracy_history = []  # 记录验证准确率
    x = np.array(range(1, epochs+1))
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
    evaluate(model, test_loader, device)

    # 历次准确率绘制
    plt.plot(x,train_accuracy_history, label="Train Accuracy")
    plt.plot(x,val_accuracy_history, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.scatter(x, train_accuracy_history, marker='o')
    plt.scatter(x, val_accuracy_history, marker='o')
    plt.scatter(1, test_accuracy, marker='o')
    plt.show()

# def train_and_evaluate_CNN(epochs, batch_size, train_size, val_split, learning_rate, device):
#     train_accuracy_history = []
#     val_accuracy_history = []
#     x = np.array(range(1, epochs+1))
#
#     model = CNN(num_classes=10).to(device)
#
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     criterion = torch.nn.CrossEntropyLoss()
#
#     print(f"Training with {train_size * 100}% of the training data")
#     train_loader, val_loader, test_loader, unlabeled_loader = load_mnist_data(batch_size, train_size, val_split)
#
#     for epoch in range(epochs):
#         running_loss = 0.0
#         total = 0
#         correct = 0
#
#         # labeled data
#         model.train()
#         for batch_idx, (data, target) in enumerate(train_loader):
#             data, target = data.to(device), target.to(device)
#             optimizer.zero_grad()
#             output = model(data)
#             loss = criterion(output, target)
#             loss.backward()
#             optimizer.step()
#
#             running_loss += loss.item()
#             _, predicted = torch.max(output.data, 1)
#             total += target.size(0)
#             correct += (predicted == target).sum().item()
#
#         train_accuracy = correct / total
#         train_accuracy_history.append(train_accuracy)
#
#         val_accuracy = evaluate(model, val_loader, device, return_accuracy=True)
#         val_accuracy_history.append(val_accuracy)
#
#         # unlabeled data
#         model.train()
#         for batch_idx, (data, _) in enumerate(unlabeled_loader):
#             data = data.to(device)
#             optimizer.zero_grad()
#             output = model(data)
#             loss = -torch.mean(torch.sum(F.log_softmax(output, dim=1) * torch.softmax(output, dim=1), dim=1))
#             loss.backward()
#             optimizer.step()
#
#         print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss / (batch_idx + 1)}, Train Accuracy: {train_accuracy:.2%}, Val Accuracy: {val_accuracy:.2%}')
#
#     evaluate(model, test_loader, device)
#     plt.plot(x,train_accuracy_history, label="Train Accuracy")
#     plt.plot(x,val_accuracy_history, label="Val Accuracy")
#     plt.xlabel("Epoch")
#     plt.ylabel("Accuracy")
#     plt.legend()
#     plt.scatter(x, train_accuracy_history, marker='o')
#     plt.scatter(x, val_accuracy_history, marker='o')
#     plt.scatter(1, test_accuracy, marker='o')
#     plt.show()

# FCNN


def train_and_evaluate_FCNN(epochs, batch_size, train_size, val_split, hidden_size1, hidden_size2, learning_rate):
    train_loader, val_loader, test_loader = load_mnist_data(batch_size, train_size, val_split)
    x = np.array(range(1, epochs+1))

    input_size = 28 * 28  # minist图像大小为28*28像素，同时也是输入层神经元数量
    num_classes = 10  # minist有10个类别，同时也是输出层的神经元数量
    # 创建FCN模型
    model = FullyConnectedNN(input_size, hidden_size1,hidden_size2, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam优化算法

    train_accuracy_history = []  # 记录训练准确率
    val_accuracy_history = []  # 记录验证准确率

    # epoch的解释：一个 epoch 表示对整个数据集进行一次完整的训练。
    # 通常情况下，一个 epoch 的迭代次数等于数据集的大小除以批次大小。
    # 例如，如果数据集包含 1000 个样本，批次大小为 10，则一个 epoch 的迭代次数为 100
    for epoch in range(epochs):  # 对每一个epoch：
        model.train()  # model.train()表示将该model设置为训练模式。一般在开始新epoch训练时，我们会首先执行该命令
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()  # 清空过往梯度
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()  # 将损失loss 向输入侧进行反向传播
            optimizer.step()  # 参数更新

        model.eval()  # 评估模式
        # 训练集
        correct_train, total_train = 0, 0
        with torch.no_grad():  # 表明当前计算不需要反向传播，使用之后，强制后边的内容不进行计算图的构建（不算梯度）
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)  # 值所对应的index就对应着相应的类别class，我们只关心预测的类别是什么
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
        train_accuracy = correct_train / total_train
        train_accuracy_history.append(train_accuracy)

        # 验证集
        correct_val, total_val = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        val_accuracy = correct_val / total_val
        val_accuracy_history.append(val_accuracy)

        print(f"Epoch [{epoch + 1}/{epochs}], Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # 对测试集进行测试
    correct_test, total_test = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
    test_accuracy = correct_test / total_test
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # 历次准确率绘制

    plt.plot(x,train_accuracy_history, label="Train Accuracy")
    plt.plot(x,val_accuracy_history, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.scatter(x, train_accuracy_history, marker='o')
    plt.scatter(x, val_accuracy_history, marker='o')
    plt.scatter(1, test_accuracy, marker='o')
    plt.show()


# FCNN模型参数设置
# epochs = 10
# batch_size = 64
# train_size = 0.3  # 训练集大小的改变
# val_split = 0.1  # 从新训练集中抽取10%作为验证集
# hidden_size1 = 256  # 隐藏层神经元数量
# hidden_size2 = 128
# learning_rate = 0.001  # 学习率
#
#
# print(f"Training with {train_size * 100}% of the training data")
# train_and_evaluate_FCNN(epochs, batch_size, train_size, val_split, hidden_size1,hidden_size2, learning_rate)

# CNN模型参数设置
train_and_evaluate_CNN(epochs=10, batch_size=64, train_size=0.3, val_split=0.1, learning_rate=0.001, device=device)
