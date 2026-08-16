import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import os

# ============================================================
# 1. 设备配置
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ============================================================
# 2. 加载模型（和训练时一致：ResNet50）
# ============================================================
CLASS_NAMES = ['Scab','healthy', 'rust', ]  # 顺序必须和训练时一致

# 加载和训练时相同的模型结构
model = models.resnet50(weights=None)  # 不加载预训练权重
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(CLASS_NAMES))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "best_apple_model.pth")

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"✅ 模型加载成功: {model_path}")
else:
    print(f"❌ 模型文件不存在: {model_path}")
    print("请先训练并保存模型！")
    exit()

# 图像预处理（和训练时一致）
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============================================================
# 3. FastAPI 服务
# ============================================================
app = FastAPI(title="苹果病害分类服务")

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/predict_apple")
async def predict_apple(file: UploadFile = File(...)):
    """苹果病害分类接口：返回类别和置信度"""
    try:
        # 1. 读取图片
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')

        # 2. 预处理
        img_tensor = transform(img).unsqueeze(0).to(device)

        # 3. 推理
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        class_name = CLASS_NAMES[predicted.item()]
        confidence_score = confidence.item()

        return {
            "status": "success",
            "class": class_name,
            "class_index": predicted.item(),
            "confidence": round(confidence_score, 4),
            "all_probs": {
                CLASS_NAMES[i]: round(probabilities[0][i].item(), 4)
                for i in range(len(CLASS_NAMES))
            }
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

# ============================================================
# 4. 启动服务
# ============================================================
# 在终端运行：
# uvicorn deploy_pest:app --host 0.0.0.0 --port 8001 --reload