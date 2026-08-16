import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import os

# 1. 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 2. 数据预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. 加载数据（不再划分验证集，直接把文件夹当成训练集）
train_dir = "../Apple/train"  # 这里放 70% 的图片
test_dir = "../Apple/test"  # 这里放 30% 的图片

train_dataset = datasets.ImageFolder(train_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

print(f"训练集类别: {train_dataset.class_to_idx}")  # {'healthy': 0, 'rust': 1, 'scab': 2}
print(f"训练集图片数: {len(train_dataset)}")
print(f"测试集图片数: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# 4. 加载预训练模型
model = models.resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

num_classes = len(train_dataset.classes)
model.fc = nn.Linear(2048, num_classes)
model = model.to(device)

# 5. 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

# 6. 训练
epochs = 30
best_test_acc = 0.0

for epoch in range(epochs):
    # 训练阶段
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # 测试阶段（用测试集做评估）
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100 * correct / total
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}, 测试精度: {test_acc:.2f}%,LR: {current_lr:.6f}")
    scheduler.step()
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        torch.save(model.state_dict(), "best_apple_model2.pth")
        print(f"✅ 保存最佳模型 (测试精度: {test_acc:.2f}%)")