import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import os


'''****************************数据集处理和划分*********************************'''
# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载数据集
train_dataset = datasets.ImageFolder(root='Covid19_dataset/train', transform=transform)
test_dataset = datasets.ImageFolder(root='Covid19_dataset/test', transform=transform)
# 获取类别名称
class_names = train_dataset.classes
print("类别名称:", class_names)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# 加载模型
def load_model(model, filepath='model/model_epoch_20.pth'):
    model.load_state_dict(torch.load(filepath))
    model.eval()
    return model

# 显示图片的函数
def show_image(img, label, pred_label, class_names):
    # 反归一化
    img = img.permute(1, 2, 0)  # 将形状从 (C, H, W) 转换为 (H, W, C)
    img = img.numpy()
    img = np.clip(img, 0, 1)  # 将像素值限制在[0, 1]范围内

    # 反归一化
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean  # 反归一化
    img = np.clip(img, 0, 1)  # 保证像素值在[0, 1]范围内

    # 显示图像
    plt.imshow(img)
    plt.title(f"True: {class_names[label]} | Pred: {class_names[pred_label]}")
    plt.axis('off')
    plt.show()

# 模型训练
def train_model(model, train_loader, criterion, optimizer, num_epochs=20, save_every_epoch=True):
    train_losses = []
    train_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

        # 每个 epoch 后保存模型（可选）
        if save_every_epoch:
            model_dir = 'resnet18_model'
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            model_path = os.path.join(model_dir, f'model_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), model_path)
            print(f"Epoch {epoch + 1} 模型已保存！路径：{model_path}")

    return train_losses, train_accuracies

# 评估模型
def evaluate_model(model, test_loader, class_names):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

            # 显示第一张图片及其预测结果
            show_image(inputs[0].cpu(), labels[0].cpu().item(), predicted[0].cpu().item(), class_names)

    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test F1 Score: {f1:.4f}")

    return accuracy, recall, precision, f1


if __name__ == '__main__':
    # 训练和评估模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('run in: ', device)
    model = models.resnet18(pretrained=True)  # 加载预训练模型
    # 修改最后的全连接层以适应我们数据集的类别数
    model.fc = nn.Linear(model.fc.in_features, len(class_names))  # 用我们数据集的类别数替代1000
    # 将模型移到GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses, train_accuracies = train_model(model, train_loader, criterion, optimizer, num_epochs=50, save_every_epoch=True)
    # 绘制训练损失和准确率曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Training Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.ylim(0, max(train_losses) + 0.1)  # 设置y轴范围，从0到损失的最大值加一点空间

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Training Accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    plt.ylim(0.3, 1.1)  # 准确率的范围通常是0到1

    plt.show()


    # 加载并测试模型
    model = load_model(model, filepath='resnet18_model/model_epoch_50.pth')
    evaluate_model(model, test_loader, test_dataset.classes)
