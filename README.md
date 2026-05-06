# StoryRag v2.0 — 具角色与记忆能力的智能 RAG 系统

基于 DeepSeek LLM + 混合检索 + 角色管理 + 三层记忆，实现「能记住用户、保持人设」的助理型 RAG。

---

## 核心能力

| 能力 | 说明 |
|------|------|
| **混合检索** | 向量检索 + 关键词检索 + Reranker 精排，多语言 PDF 语义问答 |
| **稳定角色** | 结构化角色配置 + 多层注入 + 防漂移，不依赖单条 prompt |
| **三层记忆** | 短期（会话上下文）+ 长期（用户偏好/事实）+ 时间线（事件回溯） |
| **LangGraph Agent** | ReAct 模式自主调用工具，单次检索即返回带来源的回答 |

---

## 快速开始

### 1. 环境要求

- Python 3.10+
- CUDA GPU（可选，加速向量嵌入）
- 8GB+ RAM / 10GB+ 磁盘

### 2. 安装

```bash
git clone <repository-url>
cd Yezuoju-main

python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # Linux/Mac

pip install -r requirements.txt
```

### 3. 配置

```bash
cp env.example .env
```

编辑 `.env`，至少填写：

```env
DEEPSEEK_API_KEY=sk-your-key-here
VECTOR_DB_PATH=src/data/vector_database
```

### 4. 准备数据

将 PDF 放入 `src/data/source/Chinese/`（支持中/英/日）。

```bash
python src/data_processing/process_pipeline.py
```

### 5. 启动

```bash
# 调试模式（交互对话）
python debug_tool.py

# 或启动 API 服务
cd backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

访问 http://localhost:8000/docs 查看 API 文档。

---

## 角色系统

### 设计原则

角色控制**不依赖单条 prompt**，而是四层注入机制：

```
第一层：RoleProfile 结构化数据   →  身份、职责、风格、边界全部字段化
第二层：build_system_prompt_segment() → 动态拼入 Agent 系统消息
第三层：每 5 轮注入 reinforcement  → 防止长对话中角色漂移
第四层：tone_variants + signature_phrases → 场景语气变体
```

### 内置角色

| role_id | 名称 | 风格 |
|---------|------|------|
| `humorous_butler` | 幽默的男管家 | 幽默庄重，英式管家（默认） |
| `scholarly_assistant` | 严谨的学术助手 | 理性严谨，学术规范 |
| `storyteller` | 博学的说书人 | 生动传神，叙事风格 |

### 使用

```python
from src.agents.deepseek_agent import create_deepseek_agent

agent = create_deepseek_agent(
    vector_db_path="src/data/vector_database",
    role_id="humorous_butler"     # 指定角色
)
```

**API 端点**：

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/v1/chat/roles` | 获取可用角色列表 |
| `GET` | `/api/v1/chat/role/current` | 查看当前角色配置 |
| `POST` | `/api/v1/chat/role/switch` | 切换角色 `{"role_id":"storyteller"}` |

---

## 记忆系统

### 三层架构

| 层级 | 存储 | 生命周期 | 用途 |
|------|------|----------|------|
| **短期记忆** | 内存 LRU (50条) | 会话内 | 当前对话的关键上下文 |
| **长期记忆** | 磁盘 JSON | 跨会话 | 用户偏好、稳定事实（自动去重） |
| **时间线记忆** | 磁盘 JSON | 跨会话 | 重要事件 + 时间戳，支持按日期回溯 |

### 自动工作流

```
用户消息 → record_user_message(短期)
         → search_all(检索相关历史)
         → build_full_context(构建推理上下文)
         → 注入 Agent 消息队列

会话结束 → sync_from_session(短期→长期提升)
         → 清空短期记忆
```

### 手动记忆操作

```python
from src.core.memory_manager import get_memory_manager

mm = get_memory_manager()

# 保存用户事实
await mm.record_user_fact("用户老家在南京", importance=3)

# 保存用户偏好
await mm.record_user_preference("用户喜欢简洁回答", importance=4)

# 记录事件
await mm.record_event("用户查询了《呐喊》", "询问了狂人日记的内容")
```

**API 端点**：

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/v1/chat/memory/query` | 搜索记忆（跨层级） |
| `POST` | `/api/v1/chat/memory/long-term` | 保存长期记忆 |
| `POST` | `/api/v1/chat/memory/event` | 保存时间线事件 |
| `GET` | `/api/v1/chat/memory/timeline` | 获取时间线（按天数） |
| `DELETE` | `/api/v1/chat/memory/{id}` | 删除记忆 |
| `GET` | `/api/v1/chat/memory/stats` | 记忆统计 |

---

## 检索工具

| 工具 | 说明 |
|------|------|
| `smart_retrieval` | 智能混合检索（向量+关键词+Reranker） |
| `metadata_filter_retrieval` | 元数据过滤直接召回（不排序，快速定位特定章节） |

---

## 项目结构

```
Yezuoju-main/
├── src/                          # 核心引擎
│   ├── core/                     # 角色 & 记忆（v2.0 新增）
│   │   ├── role_profile.py      # 结构化角色配置
│   │   ├── role_manager.py      # 角色注入与一致性管理
│   │   └── memory_manager.py    # 三层记忆系统
│   ├── models/                   # 数据模型与配置
│   ├── rag/                      # 检索引擎（混合检索/Reranker）
│   ├── agents/                   # LangGraph Agent
│   ├── tools/                    # LangChain 工具封装
│   ├── data_processing/          # PDF 处理流水线
│   └── data/                     # 数据存储
│       ├── source/               # 原始 PDF（按语言分类）
│       ├── chunks/               # 文本块 JSON
│       ├── sessions/             # 会话数据
│       ├── memory/               # 长期记忆 & 时间线
│       ├── roles/                # 自定义角色 JSON 配置
│       └── vector_database/      # Chroma 向量数据库
│
├── backend/                      # FastAPI 后端
│   └── app/
│       ├── api/chat_router.py   # 对话 + 记忆 + 角色 API
│       ├── core/                 # 会话管理 / 配置 / Token 估算
│       └── agents/               # Agent 服务层
│
├── frontend/                     # React 前端
├── debug_tool.py                 # 交互式调试工具
├── env.example                   # 环境变量模板
└── requirements.txt
```

---

## Python SDK 示例

```python
from src.core.memory_manager import get_memory_manager
from src.agents.deepseek_agent import create_deepseek_agent

# 初始化记忆（可预填充长期记忆）
mm = get_memory_manager()
await mm.record_user_preference("用户偏好详细的历史背景分析", importance=4)

# 创建带角色的 Agent
agent = create_deepseek_agent(
    vector_db_path="src/data/vector_database",
    role_id="humorous_butler",
    memory_manager=mm
)

# 对话
response = agent.chat("朱元璋的家庭背景如何？", session_id="demo-001")
print(response.answer)
print(f"来源: {len(response.sources)} 篇文档")
```

---

## 常见问题

| 问题 | 解决 |
|------|------|
| `DEEPSEEK_API_KEY` 未设置 | 编辑 `.env` 或 `set DEEPSEEK_API_KEY=sk-xxx` |
| 向量数据库不存在 | 运行 `python src/data_processing/process_pipeline.py` |
| Embedding 模型首次加载慢 | 正常现象，后续使用缓存 |
| CUDA 不可用 | 安装 CUDA 版 PyTorch：`pip install torch --index-url https://download.pytorch.org/whl/cu124` |
| "未知章节"显示 | 确保数据生成脚本使用新字段名 (`source`/`chapter`/`start_page`/`end_page`) |

---

## 技术栈

- **LLM**：DeepSeek Chat API
- **Agent**：LangGraph ReAct
- **向量库**：ChromaDB
- **嵌入**：BAAI/bge-m3 / multilingual-e5-large
- **后端**：FastAPI + Uvicorn
- **前端**：React
- **分词**：jieba + tiktoken

---

## 许可证

MIT License
