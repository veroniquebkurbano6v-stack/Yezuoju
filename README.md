# StoryRag v2.0 — PDF 智能检索系统

基于检索增强生成（RAG）技术的智能问答系统，包含完整的前后端架构：
- **数据处理层** (`src/`): PDF解析、向量化、索引构建
- **后端服务层** (`backend/`): FastAPI + ChromaDB + DeepSeek LLM
- **前端界面层** (`frontend/`): React + Vite + Tailwind CSS

=====================================

## 快速开始

### 1. 环境准备

#### 1.1 复制环境配置
复制环境示例为 `.env` 并填写秘密/覆盖项：
- 复制：`cp env.example .env`（在 Windows 上手动复制 `env.example` 到 `.env` 并编辑）

#### 1.2 创建并激活虚拟环境

```bash
python -m venv .venv
source .venv/bin/activate       # macOS / Linux
.venv\Scripts\activate        # Windows (PowerShell)
pip install -r requirements.txt
```

### 2. 数据准备

选择以下任一方式准备数据：

#### 2.1 完整流程（推荐）
生成 JSON（identify_title）并导入到 Chroma：

```bash
python src/process_pipeline.py --mode full
```

#### 2.2 导入现有 JSON
如果已生成 JSON 文件，可直接导入到 Chroma：

```bash
python -m src.embedding_vector --ingest-json-dir src/data/pages_title --db-path src/data/vector_database
```

### 3. 启动服务

#### 3.1 启动后端服务

```bash
# 在新终端窗口中
python -m backend.app.main
```
后端将在 `http://localhost:8000/` 启动，提供 API 服务。

#### 3.2 启动前端界面

```bash
# 在新终端窗口中，进入前端目录
cd frontend
npm install  # 首次运行需要
npm run dev
```
前端将在 `http://localhost:5173/` 启动，提供Web界面。

### 4. 开始使用

- 打开浏览器访问 **http://localhost:5173/**
- 使用检索面板进行直接查询
- 创建对话进行多轮对话交互
- 享受完整的 RAG 检索增强体验！

**注意**：确保同时启动后端和前端服务，前端依赖后端API才能正常工作。

### 5. 交互式代理（可选）

如果需要使用命令行交互：

```bash
python run_agent.py
```

## 配置说明

配置文件：使用 `.env` 管理运行配置与密钥（项目根）。已提供 `env.example`（复制 `env.example` 到 `.env` 后修改）。

主要可配置项（在 `.env` 中）：
- `VECTOR_DB_PATH`：ChromaDB 存储路径（默认 `src/data/vector_database`）。
- `EMBEDDING_MODEL`：嵌入模型名称（默认 `intfloat/multilingual-e5-large`）。
- `EMB_CACHE_DIR`：预计算的 sections embeddings 与缓存路径（默认 `src/data/pages_title`）。
- `FILE_MATCH_THRESHOLD` / `SECTION_MATCH_THRESHOLD`：智能检索中文件名与章节标题匹配阈值（可调）。
- `DEEPSEEK_API_KEY`（可选）：DeepSeek LLM 服务密钥。
- `OMP_NUM_THREADS` / `TOKENIZERS_PARALLELISM`：并行相关环境变量建议。

示例 `.env`（将 `env.example` 复制为 `.env` 或创建 `.env` 并粘贴以下内容）：

```env
VECTOR_DB_PATH=src/data/vector_database
EMBEDDING_MODEL=intfloat/multilingual-e5-large
EMB_CACHE_DIR=src/data/emb_cache

# DeepSeek API (optional)
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1

# Smart retrieval thresholds (can be tuned)
FILE_MATCH_THRESHOLD=0.7
SECTION_MATCH_THRESHOLD=0.75

# Performance / env flags
OMP_NUM_THREADS=1
TOKENIZERS_PARALLELISM=false
```

## 核心功能

### 混合检索引擎
结合向量检索和关键词检索，返回最相关的 60 个文本块，支持：
- 智能文件名和章节标题匹配
- 基于相似度的精确检索
- 可选的重排序优化

### 智能代理系统
集成 DeepSeek 大语言模型，基于检索结果生成准确回答，支持：
- 多轮对话记忆
- 结构化回答格式
- 基于历史记录的智能回退
- 友好的信息不足提示

### 现代化Web界面
React + Tailwind CSS 构建的响应式前端，提供：
- 直观的检索面板
- 对话管理功能
- 实时引用展示
- 支持桌面和移动设备

### 对话记忆管理
基于 LangChain Checkpointer 的持久化多对话支持，包含：
- 对话创建与删除
- 历史记录保存
- 多对话切换

## 目录结构

```
StoryRag/
├── run_agent.py              # 系统统一入口脚本
├── .env                     # 环境变量配置
├── requirements.txt         # Python 依赖
├── src/                     # 数据处理层
│   ├── mixed_retrieval.py   # 混合检索引擎
│   ├── deepseek_agent.py    # DeepSeek 检索代理
│   ├── embedding_vector.py  # 向量数据库模块
│   ├── identify_title.py    # 标题识别模块
│   ├── process_pipeline.py  # 数据处理流水线
│   └── data/
│       ├── source/          # 原始 PDF 文件
│       ├── pages_title/     # JSON 元数据
│       └── vector_database/ # ChromaDB 向量数据库
├── backend/                 # 后端服务层
│   └── app/
│       ├── main.py          # FastAPI 应用入口
│       ├── api/             # API 路由
│       └── core/            # 核心服务（对话管理等）
├── frontend/                # 前端界面层
│   ├── src/
│   │   ├── App.jsx          # 主应用组件
│   │   ├── components/      # React 组件
│   │   └── index.css        # Tailwind CSS 样式
│   ├── package.json         # 前端依赖
│   └── vite.config.js       # Vite 配置
└── dialog_checkpoints/      # 对话持久化存储
```

## 技术架构

### 检索流程

1. **用户查询** → `run_agent.py` 或 Web API
2. **参数提取** → 解析查询和过滤条件
3. **智能检索** → `smart_retrieval_impl` 执行检索：
   - 文件名和章节标题匹配
   - 向量检索
   - 可选重排序
4. **结果处理** → 格式化检索结果
5. **生成回答** → 基于检索结果和历史记录生成回答
6. **返回结果** → 包含回答和引用信息

### 核心技术栈

**后端技术栈**：
- **Web框架**：FastAPI
- **向量数据库**：ChromaDB
- **嵌入模型**：multilingual-e5-large
- **大语言模型**：DeepSeek
- **对话管理**：LangChain Checkpointer
- **开发框架**：Python, LangChain

**前端技术栈**：
- **UI框架**：React 18
- **构建工具**：Vite
- **样式框架**：Tailwind CSS v4
- **HTTP客户端**：Fetch API

## 后端服务

后端为纯净服务，**启动时仅加载已构建的向量库与可选的 LLM agent，不做数据处理**。推荐部署在单独进程/容器中。

### 关键环境变量

- `VECTOR_DB_PATH`：向量数据库路径（示例：`src/data/vector_database`）
- `EMBEDDING_MODEL`：嵌入模型（仅用于必要时初始化检索器）
- `DEEPSEEK_API_KEY`：若设置则自动初始化 DeepSeek agent（可选）

### 可用 API

- `POST /api/query`：JSON 请求体 `{ "question": "...", "top_k": 10 }`，返回检索结果
- `POST /api/dialogs/{id}/chat`：对话交互接口
- `GET /api/dialogs`：获取对话列表
- `POST /api/dialogs`：创建新对话
- `DELETE /api/dialogs/{id}`：删除对话

## 前端界面

现代化Web界面，提供直观的检索和对话交互体验。基于 React + Vite + Tailwind CSS 构建。

### 前端功能

- **检索面板**：直接查询 `/api/query` 接口，展示带引用的检索结果
- **对话管理**：创建/删除对话，从后端同步对话列表
- **聊天界面**：与 `/api/dialogs/{id}/chat` 集成，支持上下文对话
- **引用展示**：显示检索结果的来源信息
- **响应式设计**：支持桌面和移动设备

## 执行模式

### 交互式模式

```bash
python run_agent.py
```

### 直接查询模式

```bash
python run_agent.py --query "你的查询内容"
```

### 批量查询模式

```bash
python run_agent.py --file queries.txt
```

## 常见问题解答（FAQ）

### Q: 安装依赖时遇到权限问题怎么办？
A: 在命令前添加 `sudo`（Linux/macOS）或使用管理员权限运行 PowerShell（Windows）。

### Q: 后端无法启动，提示端口被占用？
A: 检查端口 8000 是否被其他程序占用，或修改 FastAPI 配置使用其他端口。

### Q: 前端无法连接到后端？
A: 确保后端服务已启动，且前端配置中的 API 地址正确。检查浏览器控制台的网络请求错误。

### Q: 检索结果不准确怎么办？
A: 尝试调整 `.env` 中的 `FILE_MATCH_THRESHOLD` 和 `SECTION_MATCH_THRESHOLD` 阈值，或重新生成向量数据库。

### Q: 如何添加新的 PDF 文件？
A: 将 PDF 文件放入 `src/data/source/` 目录，然后运行 `python src/process_pipeline.py --mode full` 重新生成向量数据库。

## 贡献指南

1. ** Fork 仓库**：在 GitHub 上 Fork 本项目
2. **创建分支**：基于 `main` 分支创建特性分支
3. **提交更改**：提交代码更改，确保遵循项目代码风格
4. **运行测试**：确保所有测试通过
5. **创建 Pull Request**：提交 PR，描述更改内容和动机

## 更新日志

### v2.0.0
- 重构为完整的前后端架构
- 集成 DeepSeek LLM 智能代理
- 实现基于对话历史的智能回答
- 优化检索结果展示
- 支持多轮对话管理

### v1.0.0
- 初始版本
- 基本的 PDF 解析和向量检索功能
- 简单的命令行界面

## 开发者说明

- 使用 `.env` 与 `requirements.txt` 管理环境和依赖（禁止使用 CI workflow 来写入运行时机密）。
- 系统会在启动时尝试加载 `src/data/pages_title/sections_embeddings.npz`（若存在）以加速智能检索；否则会在第一次运行时计算并缓存该文件。
- 若需调试或调整阈值，可直接在 `.env` 中修改 `FILE_MATCH_THRESHOLD` 与 `SECTION_MATCH_THRESHOLD`。

## 安全与备份

- 已将原 `.github/workflows/ci.yml` 备份到 `src/archive/`，并从工作流目录中删除以避免 CI 在无意中使用 workflow 配置来注入环境变量。

## 许可证

MIT License