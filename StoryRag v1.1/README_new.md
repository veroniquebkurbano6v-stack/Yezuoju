# StoryRag - 三语童话检索增强生成系统

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 项目简介

StoryRag 是一个支持中文、日文和英文童话故事的检索增强生成（RAG）系统。它允许用户针对童话故事提出问题，系统会从故事库中检索相关内容并生成带有引用的回答。

### 主要功能

- 🌍 **多语言支持**：支持中文、英文和日文童话故事
- 📚 **双模式查询**：支持单本书籍查询和全库查询
- 🔍 **智能引用**：每个回答都附带准确的引用来源
- 🤖 **无需OpenAI**：使用DeepSeek模型，无需OpenAI API
- 🖥️ **双接口**：提供命令行界面和Web界面

## 目录结构

```
StoryRag v1.1/
├── .env                 # 环境变量配置文件
├── api.py               # FastAPI服务器实现
├── llm.py               # DeepSeek API客户端
├── main.py              # 命令行入口
├── rag.py               # RAG核心实现
├── requirements.txt     # 项目依赖
├── frontend/            # Web前端文件
└── Ragdate/             # 数据目录
    ├── Chinese/         # 中文童话故事
    ├── English/         # 英文童话故事
    └── Japanse/         # 日文童话故事
```

## 安装指南

### 环境要求

- Python 3.10 或更高版本
- pip 包管理器

### 安装步骤

1. **克隆项目**
   ```bash
   git clone <repository-url>
   cd "StoryRag v1.1"
   ```

2. **创建虚拟环境**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

4. **配置环境变量**

   创建 `.env` 文件（如果不存在）并添加以下内容：
   ```
   DEEPSEEK_API_KEY=你的DeepSeek API密钥
   DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
   DEEPSEEK_MODEL=deepseek-chat
   DEEPSEEK_TIMEOUT=30

   # 数据与检索设置（可选）
   DATA_DIR=./Ragdate
   LANGS=Chinese,English,Japanse
   DEFAULT_LANGUAGE=Chinese
   CHUNK_SIZE=700
   CHUNK_OVERLAP=120
   TOP_K=4
   ```

   **注意**：必须设置 `DEEPSEEK_API_KEY`，其他参数有默认值。

5. **准备数据**

   将童话故事文本文件（.txt格式）放入对应语言文件夹：
   - 中文故事：`Ragdate/Chinese/`
   - 英文故事：`Ragdate/English/`
   - 日文故事：`Ragdate/Japanse/`

## 使用指南

### 命令行界面

1. **全库模式（默认）**
   ```bash
   python main.py "请用中文总结故事主旨并标注出处"
   ```

2. **指定语言**
   ```bash
   python main.py --language English "What is the moral of the tale?"
   ```

3. **单书模式**
   ```bash
   python main.py --mode single --book xiaowangzi.txt "主人公是谁？"
   ```

### Web界面

1. **启动服务**
   ```bash
   uvicorn api:app --reload --port 8000
   ```

2. **访问界面**

   打开浏览器访问 `http://localhost:8000`

   - 在Web界面中选择语言（或使用自动检测）
   - 选择查询模式（全库或单书）
   - 如选择单书模式，从下拉列表中选择书籍
   - 输入问题并提交

## API接口

### 健康检查
```http
GET /api/health
```

### 获取可用书籍列表
```http
GET /api/books
```

### 提交查询
```http
POST /api/query
Content-Type: application/json

{
  "query": "用户问题",
  "language": "Chinese|English|Japanse|null",
  "mode": "library|single",
  "book": "书籍名称（当mode=single时）"
}
```

## 工作原理

1. **文本分块**：将童话故事文本按指定大小（CHUNK_SIZE）和重叠（CHUNK_OVERLAP）分割成块
2. **向量索引**：使用TF-IDF和句子转换器创建文本块的向量表示
3. **检索**：根据用户问题检索最相关的文本块（TOP_K个）
4. **生成回答**：将检索到的文本块和问题一起发送给DeepSeek模型，生成带有引用的回答
5. **引用标注**：在回答中标注引用来源，格式为 `[语言/文件#chunk]`

## 常见问题解答

### Q: 如何添加新的童话故事？
A: 只需将.txt格式的童话故事文件放入对应语言的文件夹（`Ragdate/Chinese/`、`Ragdate/English/`或`Ragdate/Japanse/`）中。

### Q: 支持哪些文本格式？
A: 目前仅支持.txt格式。如需使用其他格式，请先将其转换为.txt文件。

### Q: 如何调整检索效果？
A: 可以通过修改.env文件中的以下参数：
- `CHUNK_SIZE`：文本块大小（默认700）
- `CHUNK_OVERLAP`：文本块重叠大小（默认120）
- `TOP_K`：检索的文本块数量（默认4）

### Q: 如何更换语言模型？
A: 目前系统仅支持DeepSeek模型。如需使用其他模型，需要修改llm.py文件中的实现。

## 许可证

本项目采用 MIT 许可证。详情请参阅 LICENSE 文件。

## 贡献指南

欢迎提交问题报告和功能请求！如果您想贡献代码，请：

1. Fork 本项目
2. 创建您的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开一个 Pull Request



感谢使用 StoryRag！