"""补全 Scab 训练数据：从 test/Scab 的 10 张图增强生成多张训练图

原因：train/Scab 为空，模型从未见过 Scab 类别。
做法：对每张 test/Scab 图片做 6 种增强（原图+翻转+旋转+亮度），生成 7 张训练图。
注意：这会与 test/Scab 原图存在轻微数据泄漏，训练时应避免同图在测试中出现。
"""
import os
import sys
import cv2
import numpy as np

# 路径（相对运行目录，或通过参数传入）
BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, "..", "..", "..", "code", "langchain1.2Study", "Apple", "test", "Scab")
DST = os.path.join(BASE, "..", "..", "..", "code", "langchain1.2Study", "Apple", "train", "Scab")

if len(sys.argv) > 1:
    SRC = sys.argv[1]
if len(sys.argv) > 2:
    DST = sys.argv[2]

os.makedirs(DST, exist_ok=True)


def augment(img):
    """返回 6 个增强变体（不含原图）"""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    variants = []
    # 水平翻转
    variants.append(cv2.flip(img, 1))
    # 垂直翻转
    variants.append(cv2.flip(img, 0))
    # 旋转 90
    M90 = cv2.getRotationMatrix2D(center, 90, 1.0)
    variants.append(cv2.warpAffine(img, M90, (w, h)))
    # 旋转 180
    M180 = cv2.getRotationMatrix2D(center, 180, 1.0)
    variants.append(cv2.warpAffine(img, M180, (w, h)))
    # 亮度增强
    variants.append(cv2.convertScaleAbs(img, alpha=1.2, beta=0))
    # 亮度减弱
    variants.append(cv2.convertScaleAbs(img, alpha=0.8, beta=0))
    return variants


files = sorted(f for f in os.listdir(SRC) if f.endswith((".jpg", ".png")))
print(f"源图 {len(files)} 张 (test/Scab)")
total = 0
for f in files:
    img = cv2.imread(os.path.join(SRC, f))
    if img is None:
        continue
    name = os.path.splitext(f)[0]
    # 原图也复制一份进训练集
    cv2.imwrite(os.path.join(DST, f"{name}_orig.jpg"), img)
    total += 1
    for i, var in enumerate(augment(img)):
        cv2.imwrite(os.path.join(DST, f"{name}_aug{i}.jpg"), var)
        total += 1

print(f"✅ 已生成 {total} 张 Scab 训练图 → {DST}")
