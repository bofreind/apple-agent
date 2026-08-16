from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings
from langchain.tools import tool
from pymilvus import MilvusClient
import requests
import os
import tempfile
import base64
from dotenv import load_dotenv

load_dotenv(override=True)

# =========================
# 1. 配置
# =========================
MILVUS_URI = "http://localhost:19530"
DB_NAME = "rag_tutorial"
COLLECTION_NAME = "Appledocs"
EMBED_MODEL_NAME = "qwen3.7-text-embedding"
EMBED_DIM = 1024
PEST_SERVICE_URL = "http://localhost:8000/predict_apple"

# =========================
# 2. 初始化 Milvus
# =========================
client = MilvusClient(MILVUS_URI)
client.use_database(db_name=DB_NAME)

# =========================
# 3. 嵌入模型
# =========================
embedding_model = DashScopeEmbeddings(model=EMBED_MODEL_NAME)

# =========================
# 4. 大模型
# =========================
model = ChatOpenAI(
    model="qwen3.7-plus",
    api_key=os.getenv("QIANWEN_API_KEY"),
    base_url=os.getenv("QIANWEN_API_URL")
)


# =========================
# 5. 调用病虫害识别服务（直接 HTTP 调用）
# =========================
def call_pest_service(image_path: str) -> dict:
    """直接调用病虫害识别服务，返回完整结果"""
    try:
        with open(image_path, "rb") as f:
            response = requests.post(PEST_SERVICE_URL, files={"file": f}, timeout=10)

            print(f"🔍 原始响应: {response.text}")
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                return {
                    "success": True,
                    "class": data.get("class"),
                    "class_index": data.get("class_index"),
                    "confidence": data.get("confidence"),
                    "all_probs": data.get("all_probs", {})
                }
            return {"success": False, "error": data.get("message", "识别异常")}
        return {"success": False, "error": f"HTTP失败：{response.status_code}"}
    except Exception as e:
        return {"success": False, "error": f"识别失败：{str(e)}"}


@tool
def identify_apple_disease(image_path: str) -> str:
    """识别苹果叶片病害：输入图片路径，返回健康、锈病或疮痂病。"""
    result = call_pest_service(image_path)
    if result.get("success"):
        return f"识别结果：{result['class']}，置信度：{result['confidence']:.2%}"
    return f"识别失败：{result.get('error', '未知错误')}"


# =========================
# 6. 创建 Agent
# =========================
agent = create_agent(
    model=model,
    tools=[identify_apple_disease],
    system_prompt=(
        "你是一个农业问答助手。"
        "如果用户上传了图片，调用 identify_apple_disease 工具识别病害。"
        "回答要专业、准确。"
    ),
)


# =========================
# 7. 检索函数
# =========================
def retrieve(query: str, limit: int = 3):
    query_vector = embedding_model.embed_query(query)
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        limit=limit,
        output_fields=["chunk_id", "text", "source"]
    )
    return results[0]


# =========================
# 8. FastAPI 服务
# =========================
app = FastAPI(title="农业智能体服务")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源（开发环境）
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有 HTTP 方法
    allow_headers=["*"],  # 允许所有请求头
)

class ChatRequest(BaseModel):
    question: str
    image_base64:  Optional[str] = None


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    智能体问答接口
    返回格式：class, confidence, all_probs, answer
    """
    try:
        class_result = None
        image_bytes = None

        # 如果有图片，先识别病害
        if request.image_base64:
            image_bytes = base64.b64decode(request.image_base64)
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                tmp_file.write(image_bytes)
                tmp_path = tmp_file.name

            # 🔥 直接调用病虫害识别服务
            class_result = call_pest_service(tmp_path)
            print(f"🔍 分类结果: {class_result}")  # 终端打印调试
            print(f"🔍 all_probs: {class_result.get('all_probs', {})}")  # 添加这行
            if class_result.get("success"):
                result_text = f"【病害识别结果】\n类别：{class_result['class']}，置信度：{class_result['confidence']:.2%}\n\n"
            else:
                result_text = f"【病害识别失败】\n{class_result.get('error', '未知错误')}\n\n"
        else:
            result_text = ""

        # 检索知识库
        his = retrieve(request.question, limit=3)
        knowledge_block = "【知识库检索结果】\n"
        for i, item in enumerate(his, 1):
            text = item["entity"]["text"]
            source = item["entity"].get("source", "unknown")
            score = item["distance"]
            knowledge_block += f"[{i}] 来源：{source}，相关度：{score:.4f}\n{text}\n\n"

        # 组合 Prompt
        user_prompt = f"""
用户问题：{request.question}

{result_text}
{knowledge_block}

请根据以上信息回答用户问题。
- 优先参考病害识别结果。
- 结合知识库中的农业知识。
"""

        # 调用 Agent
        result = agent.invoke({
            "messages": [{"role": "user", "content": user_prompt}]
        })

        final_message = result["messages"][-1]
        answer = final_message.content

        # ============================================================
        # 🔥 返回兼容前端的格式
        # ============================================================
        response_data = {
            "status": "success",
            "answer": answer,
            "metadata": {
                "has_image": image_bytes is not None
            }
        }

        # 如果有分类结果，添加到返回数据中
        if class_result and class_result.get("success"):
            response_data["class"] = class_result.get("class")
            response_data["class_index"] = class_result.get("class_index")
            response_data["confidence"] = class_result.get("confidence")
            response_data["all_probs"] = class_result.get("all_probs", {})
        else:
            response_data["class"] = "unknown"
            response_data["class_index"] = -1
            response_data["confidence"] = 0.0
            response_data["all_probs"] = {}

        return response_data

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)