"""苹果病害分类训练脚本（优化版）
- 显式类别映射，与推理端 class_mapping.json 保持一致
- 数据增强（旋转/翻转/亮度/缩放），提升泛化
- 类别不平衡处理（class_weight + 加权采样）
- 训练结束后保存类别映射，推理端直接读取
"""
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

# ============ 配置 ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(os.path.dirname(BASE_DIR), "Apple")  # 含 train/ test/
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
MAPPING_PATH = os.path.join(BASE_DIR, "class_mapping.json")
MODEL_SAVE_PATH = os.path.join(os.path.dirname(BASE_DIR), "best_apple_model2.pth")

EPOCHS = 40
BATCH_SIZE = 16
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============ 类别映射（与推理端一致，ASCII 排序） ============
CLASS_NAMES = ["Scab", "healthy", "rust"]
class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}

# ============ 数据预处理 ============
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============ 数据集 ============
# 用自定义映射，避免 ImageFolder 按字母序的隐式顺序
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)

# 强制映射为我们的顺序（防止文件夹名变化导致错位）
assert train_dataset.classes == CLASS_NAMES, f"训练类别 {train_dataset.classes} 与期望 {CLASS_NAMES} 不一致"
train_dataset.class_to_idx = class_to_idx
test_dataset.class_to_idx = class_to_idx

print(f"类别映射: {class_to_idx}")
print(f"训练集: {len(train_dataset)} 张 | 测试集: {len(test_dataset)} 张")

# ============ 类别不平衡处理 ============
targets = [train_dataset.targets[i] for i in range(len(train_dataset))]
class_counts = np.bincount(targets, minlength=len(CLASS_NAMES))
print(f"各类别样本数: {dict(zip(CLASS_NAMES, class_counts.tolist()))}")

class_weights = 1.0 / (class_counts + 1e-8)
class_weights = class_weights / class_weights.sum() * len(CLASS_NAMES)  # 归一化
sample_weights = class_weights[targets]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ============ 模型 ============
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
for param in model.parameters():
    param.requires_grad = False

num_classes = len(CLASS_NAMES)
model.fc = nn.Linear(2048, num_classes)
model = model.to(DEVICE)

# 损失函数带类别权重（缓解不平衡）
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(DEVICE))
optimizer = optim.Adam(model.fc.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

# ============ 训练 ============
best_test_acc = 0.0
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}, 测试精度: {test_acc:.2f}%")
    scheduler.step()

    if test_acc > best_test_acc:
        best_test_acc = test_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"  ✅ 保存最佳模型 ({test_acc:.2f}%) → {MODEL_SAVE_PATH}")

# 保存类别映射（推理端读取，保证一致）
with open(MAPPING_PATH, "w", encoding="utf-8") as f:
    json.dump({"class_to_idx": class_to_idx,
               "idx_to_class": {str(v): k for k, v in class_to_idx.items()}}, f, ensure_ascii=False, indent=2)
print(f"✅ 类别映射已保存 → {MAPPING_PATH}")
print(f"最佳测试精度: {best_test_acc:.2f}%")
