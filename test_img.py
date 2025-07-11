import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms, models
from PIL import Image
import os

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# 加载模型
def load_model(model, filepath='resnet34_model/model_epoch_50.pth'):
    model.load_state_dict(torch.load(filepath))
    model.eval()  # 设置模型为评估模式
    return model


# 显示图片（显示原图）
def show_image(img, label, pred_label, class_names):
    # 这里不进行反归一化，直接显示原图
    img = np.array(img)  # 将PIL图像转化为numpy数组
    plt.imshow(img)
    plt.title(f"True: {class_names[label]} | Pred: {class_names[pred_label]}")
    plt.axis('off')
    plt.show()


# 加载单张图片并进行推理
def predict_image(model, image_path, label, class_names):
    # 加载图像并预处理（不做Normalize）
    img = Image.open(image_path).convert("RGB")  # 加载并转换为RGB图像
    img_transformed = transform(img).unsqueeze(0).to(device)  # 转换为Tensor并增加批次维度

    # 前向传播
    output = model(img_transformed)
    _, predicted = torch.max(output, 1)  # 获取预测标签

    # 显示图片和预测结果
    show_image(img, label=label, pred_label=predicted.cpu().item(), class_names=class_names)


if __name__ == '__main__':
    # 设置测试图像路径
    image_path = 'Covid19_dataset/test/Normal/0103.jpeg'  # 替换为你想测试的图片路径
    label = 1  # 0，1，2 代表 ['Covid', 'Normal', 'Viral Pneumonia'] 根据自己加载的图片是哪一类型来确定

    # 获取类别名称（假设你知道类别数或从训练数据中获取）
    class_names = ['Covid', 'Normal', 'Viral Pneumonia']  # 替换为你的类别名称列表

    # 加载模型
    model = models.resnet34(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))  # 修改最后的全连接层
    model = model.to(device)

    # 加载训练好的模型
    model = load_model(model, filepath='resnet34_model/model_epoch_50.pth')

    # 预测并显示单张图像的结果
    predict_image(model, image_path, label, class_names)


