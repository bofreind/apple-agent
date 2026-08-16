"""苹果病害分类服务（优化版）
- 从 class_mapping.json 读取类别映射，与训练端一致
- 置信度阈值：低于阈值返回"不确定"
- 统一日志输出
"""
import os
import json
import logging
import io

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from fastapi import FastAPI, UploadFile, File
from PIL import Image

import config

# ============ 日志 ============
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("apple_pest")

# ============ 类别映射 ============
with open(config.CLASS_MAPPING_PATH, encoding="utf-8") as f:
    MAPPING = json.load(f)
CLASS_NAMES = list(MAPPING["class_to_idx"].keys())  # ['Scab','healthy','rust']
idx_to_class = {int(k): v for k, v in MAPPING["idx_to_class"].items()}

# ============ 设备 & 模型 ============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("使用设备: %s", device)

model = models.resnet50(weights=None)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(CLASS_NAMES))

if os.path.exists(config.MODEL_PATH):
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    logger.info("模型加载成功: %s (类别: %s)", config.MODEL_PATH, CLASS_NAMES)
else:
    logger.error("模型文件不存在: %s", config.MODEL_PATH)
    raise FileNotFoundError(config.MODEL_PATH)

# ============ 预处理（与训练端 test_transform 一致） ============
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(config.INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============ FastAPI 服务 ============
app = FastAPI(title="苹果病害分类服务")


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/predict_apple")
async def predict_apple(file: UploadFile = File(...)):
    """苹果病害分类接口：返回类别和置信度"""
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        class_name = idx_to_class[predicted.item()]
        confidence_score = confidence.item()
        logger.info("识别: %s (%.4f)", class_name, confidence_score)

        # 置信度阈值：过低返回"不确定"
        if confidence_score < config.CONFIDENCE_THRESHOLD:
            return {
                "status": "success",
                "class": "unknown",
                "class_index": -1,
                "confidence": round(confidence_score, 4),
                "all_probs": {CLASS_NAMES[i]: round(probabilities[0][i].item(), 4)
                              for i in range(len(CLASS_NAMES))},
                "message": f"置信度不足 ({confidence_score:.2f} < {config.CONFIDENCE_THRESHOLD})，无法确定类别"
            }

        return {
            "status": "success",
            "class": class_name,
            "class_index": predicted.item(),
            "confidence": round(confidence_score, 4),
            "all_probs": {CLASS_NAMES[i]: round(probabilities[0][i].item(), 4)
                          for i in range(len(CLASS_NAMES))}
        }

    except Exception as e:
        logger.exception("推理异常")
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.PEST_SERVICE_HOST, port=config.PEST_SERVICE_PORT)
