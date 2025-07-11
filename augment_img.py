import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# 定义数据增强变换
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(30),      # 随机旋转30度
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),  # 随机改变亮度、对比度、饱和度、色调
    transforms.CenterCrop(224),  # 随机裁剪并调整为224x224
    transforms.ToTensor(),              # 转换为Tensor
])

# 读取图像
img = Image.open("Covid19_dataset/train/Covid/01.jpeg")

# 应用数据增强
augmented_img = transform(img)

# 展示增强后的图像
plt.imshow(augmented_img.permute(1, 2, 0))  # 将tensor从C×H×W转换为H×W×C
plt.show()
