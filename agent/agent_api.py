"""苹果病害智能体服务（优化版）
- 初始化改为 FastAPI lifespan（启动时才连 Milvus/模型，挂掉也能起服务）
- 调用识别服务带超时 + 重试
- 配置从 config.py 统一读取
"""
from typing import Optional
from contextlib import asynccontextmanager

import requests
import os
import tempfile
import base64
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

import config

# ============ 日志 ============
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("apple-agent")

load_dotenv(override=True)

# 全局对象（lifespan 中初始化）
milvus_client = None
embedding_model = None
llm_model = None
agent = None


# ============ 识别服务调用（带重试） ============
def call_pest_service(image_path: str, retries: int = 2) -> dict:
    """调用病虫害识别服务，失败自动重试"""
    last_err = None
    for attempt in range(retries + 1):
        try:
            with open(image_path, "rb") as f:
                response = requests.post(config.PEST_SERVICE_URL, files={"file": f}, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success":
                    return {
                        "success": True,
                        "class": data.get("class"),
                        "class_index": data.get("class_index"),
                        "confidence": data.get("confidence"),
                        "all_probs": data.get("all_probs", {}),
                        "message": data.get("message"),
                    }
                return {"success": False, "error": data.get("message", "识别异常")}
            last_err = f"HTTP失败：{response.status_code}"
        except Exception as e:
            last_err = f"识别失败：{str(e)}"
        logger.warning("识别服务调用失败(第%d次): %s", attempt + 1, last_err)
    return {"success": False, "error": last_err}


# ============ 工具 ============
def identify_apple_disease(image_path: str) -> str:
    """识别苹果叶片病害：输入图片路径，返回健康、锈病或黑星病。"""
    result = call_pest_service(image_path)
    if result.get("success"):
        if result.get("class") == "unknown":
            return f"识别结果不确定：{result.get('message', '置信度不足')}"
        return f"识别结果：{result['class']}，置信度：{result['confidence']:.2%}"
    return f"识别失败：{result.get('error', '未知错误')}"


# ============ 知识库检索 ============
def retrieve(query: str, limit: int = 3):
    query_vector = embedding_model.embed_query(query)
    results = milvus_client.search(
        collection_name=config.COLLECTION_NAME,
        data=[query_vector],
        limit=limit,
        output_fields=["chunk_id", "text", "source"],
        timeout=10,
    )
    return results[0]


# ============ FastAPI 生命周期 ============
@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时初始化 Milvus / 嵌入模型 / LLM / Agent"""
    global milvus_client, embedding_model, llm_model, agent
    logger.info("正在初始化智能体服务...")

    # 1. Milvus
    from pymilvus import MilvusClient
    try:
        milvus_client = MilvusClient(config.MILVUS_URI)
        milvus_client.use_database(db_name=config.DB_NAME)
        logger.info("Milvus 连接成功: %s/%s", config.MILVUS_URI, config.COLLECTION_NAME)
    except Exception as e:
        logger.error("Milvus 连接失败: %s（问答仍可用，但知识库检索不可用）", e)
        milvus_client = None

    # 2. 嵌入模型
    from langchain_community.embeddings import DashScopeEmbeddings
    embedding_model = DashScopeEmbeddings(model=config.EMBED_MODEL_NAME)

    # 3. LLM
    from langchain_openai import ChatOpenAI
    llm_model = ChatOpenAI(
        model=config.LLM_MODEL_NAME,
        api_key=os.getenv("QIANWEN_API_KEY"),
        base_url=os.getenv("QIANWEN_API_URL"),
    )

    # 4. Agent
    from langchain.agents import create_agent
    from langchain.tools import tool
    agent = create_agent(
        model=llm_model,
        tools=[identify_apple_disease],
        system_prompt=(
            "你是一个农业问答助手。"
            "如果用户上传了图片，调用 identify_apple_disease 工具识别病害。"
            "回答要专业、准确。"
        ),
    )
    logger.info("智能体服务初始化完成")
    yield
    logger.info("智能体服务关闭")


app = FastAPI(title="农业智能体服务", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str
    image_base64: Optional[str] = None


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "milvus_ok": milvus_client is not None,
    }


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    智能体问答接口
    返回格式：class, confidence, all_probs, answer
    """
    try:
        class_result = None
        image_bytes = None
        result_text = ""

        # 有图片则先识别病害
        if request.image_base64:
            image_bytes = base64.b64decode(request.image_base64)
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                tmp_file.write(image_bytes)
                tmp_path = tmp_file.name

            class_result = call_pest_service(tmp_path)
            logger.info("分类结果: %s", class_result)
            if class_result.get("success"):
                if class_result.get("class") == "unknown":
                    result_text = f"【病害识别结果】\n{class_result.get('message', '置信度不足')}\n\n"
                else:
                    result_text = (f"【病害识别结果】\n类别：{class_result['class']}，"
                                   f"置信度：{class_result['confidence']:.2%}\n\n")
            else:
                result_text = f"【病害识别失败】\n{class_result.get('error', '未知错误')}\n\n"

        # 知识库检索（Milvus 不可用时降级为跳过）
        knowledge_block = ""
        if milvus_client is not None and embedding_model is not None:
            try:
                his = retrieve(request.question, limit=3)
                knowledge_block = "【知识库检索结果】\n"
                for i, item in enumerate(his, 1):
                    text = item["entity"]["text"]
                    source = item["entity"].get("source", "unknown")
                    score = item["distance"]
                    knowledge_block += f"[{i}] 来源：{source}，相关度：{score:.4f}\n{text}\n\n"
            except Exception as e:
                logger.warning("知识库检索失败: %s", e)
                knowledge_block = ""

        # 组合 Prompt
        user_prompt = f"""
用户问题：{request.question}

{result_text}
{knowledge_block}

请根据以上信息回答用户问题。
- 优先参考病害识别结果。
- 结合知识库中的农业知识。
"""

        result = agent.invoke({
            "messages": [{"role": "user", "content": user_prompt}]
        })
        final_message = result["messages"][-1]
        answer = final_message.content

        response_data = {
            "status": "success",
            "answer": answer,
            "metadata": {"has_image": image_bytes is not None}
        }

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
        logger.exception("chat 接口异常")
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=config.AGENT_SERVICE_PORT)
