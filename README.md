# 苹果病虫害智能体

基于 LangChain 的苹果病虫害智能体项目，包含 Python 智能体服务与 Spring Boot 前端转发层。

## 项目结构

```
apple-agent/
├── agent/                 # Python 智能体 + 分类模型代码
│   ├── agent_api.py           # 【最终版】FastAPI 智能体服务（8002/chat）
│   ├── apple_agent.py         # 旧版命令行交互版（无 API，仅参考）
│   ├── apple_pest_model.py    # 苹果病害分类模型（ResNet50）
│   ├── train_apple_model.py   # 训练脚本
│   └── test.py / test.jpg     # 测试
├── spring/                # Spring Boot 转发层（IDEA）
│   ├── pom.xml
│   └── src/               # Controller + Service + 前端 html
└── README.md
```

## 架构

```
浏览器 (static/*.html)
    ↓ HTTP
Spring Boot (spring/) —— 转发
    ├─ /api/apple/predict → agent_api.py (8002/chat)  苹果智能体
    ├─ /api/cifar/describe → Python 8000/describe     CIFAR 描述
    └─ /api/predict        → Python 高光谱服务
    ↓ HTTP
Python 服务 (agent/)
    ├─ apple_agent.py + agent_api.py   智能体问答
    └─ apple_pest_model.py             分类模型（需 best_apple_model.pth）
```

## 启动方式

### 1. Python 智能体服务（最终版）
```bash
cd agent
pip install -r requirements.txt   # 依赖（langchain、fastapi 等）
# 配置 .env 里的 API Key（通义千问 QIANWEN_API_KEY / QIANWEN_API_URL）
python agent_api.py               # 启动 FastAPI 服务，端口 8002
# 依赖: Milvus 向量库(19530)、病害识别服务(8000/predict_apple)
```

### 2. Spring Boot 转发层
用 IDEA 打开 `spring/`，运行 `PytorchApplication`，浏览器访问 http://localhost:8080

### 3. 使用
打开前端页面，上传苹果叶片图片，智能体返回病害分析结果。

## 注意

- 模型文件 `best_apple_model*.pth` 未纳入 git（90MB，用 `train_apple_model.py` 重新训练生成，默认加载 `best_apple_model2.pth`）
- 图片数据集 `train/`、`test/` 未纳入 git
- `.env` 含 API 密钥，**已从 git 排除，请勿提交**
- 类别映射统一由 `agent/class_mapping.json` 管理（训练/推理共用，避免类别错位）

## 重新训练（修复 Scab 数据缺失后）

```bash
cd agent
python augment_scab.py   # 从 test/Scab 增强生成训练图（train/Scab 原本为空）
python train_apple_model.py  # 训练并保存模型 + 类别映射
```

## 优化说明（2026-08）

- **类别映射统一**：训练/推理共用 `class_mapping.json`，修复了类别顺序可能错位的问题
- **Scab 数据补全**：train/Scab 原本 0 张，用 test/Scab 增强生成 70 张
- **置信度阈值**：低于 0.6 返回"不确定"，不再硬报类别
- **agent_api.py 生命周期化**：Milvus 等依赖改为 lifespan 初始化，单个组件挂掉不影响服务启动
- **调用重试**：识别服务调用带超时(15s)+重试(2次)
