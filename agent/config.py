"""集中配置文件：所有路径、端口、模型统一从这里读"""
import os

# ============ 路径 ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLASS_MAPPING_PATH = os.path.join(BASE_DIR, "class_mapping.json")

# 模型文件路径：优先环境变量 APPLE_MODEL_PATH，否则默认放 BASE_DIR 同级
# （模型文件不入 git，用环境变量或本地放置）
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(BASE_DIR), "best_apple_model2.pth")
MODEL_PATH = os.getenv("APPLE_MODEL_PATH", DEFAULT_MODEL_PATH)

# ============ 服务 ============
PEST_SERVICE_HOST = "0.0.0.0"
PEST_SERVICE_PORT = 8000
AGENT_SERVICE_PORT = 8002
PEST_SERVICE_URL = f"http://localhost:{PEST_SERVICE_PORT}/predict_apple"

# ============ 推理 ============
CONFIDENCE_THRESHOLD = 0.6   # 低于此置信度返回"不确定"
INPUT_SIZE = 224             # ResNet50 输入尺寸

# ============ Milvus ============
MILVUS_URI = "http://localhost:19530"
DB_NAME = "rag_tutorial"
COLLECTION_NAME = "Appledocs"
EMBED_DIM = 1024

# ============ 模型 ============
EMBED_MODEL_NAME = "qwen3.7-text-embedding"
LLM_MODEL_NAME = "qwen3.7-plus"
