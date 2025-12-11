# StoryRag (三语童话 RAG MVP)

目标：中/日/英各至少 1 本童话，支持单书/全库模式，每次回答带引用片段。模型使用 DeepSeek（无需 OpenAI）。

## 准备
1) Python 3.10+，创建虚拟环境并安装依赖：
```
pip install -r requirements.txt
```

2) 复制 `env.example` 为 `.env`，填入你的 DeepSeek Key：
```
cp env.example .env
```
必须设置：
- `DEEPSEEK_API_KEY`：你的 DeepSeek key  
- 可选：`CHUNK_SIZE`、`CHUNK_OVERLAP`、`TOP_K`、`DATA_DIR` 等

3) 数据放置：`Ragdate/<语言>/文件`，默认语言文件夹：`Chinese`、`English`、`Japanse`。**仅支持 .txt 文件**（不再支持 PDF）。

## 运行 CLI（自动识别语言）
- 全库模式（默认）：  
`python main.py "请用中文总结故事主旨并标注出处"`

- 指定语言：  
`python main.py --language English "What is the moral of the tale?"`

- 单书模式：  
`python main.py --mode single --book xiaowangzi.pdf "主人公是谁？"`

命令行会输出模型回复及引用的片段标识 `[语言/文件#chunk]`。

## 运行 API + 前端
1) 启动服务（默认 8000 端口）：
```
uvicorn api:app --reload --port 8000
```
2) 打开浏览器访问 `http://localhost:8000`，填写语言/模式/问题后提交，会调用 `/api/query` 并展示模型回复及引用。
   - 语言默认自动检测；如需单书模式，请先选择语言并选择下拉列表中的书目。

## 设计要点（对应图片需求）
- 向量检索：TF-IDF + 可配置分片（chunk_size/overlap）
- 单书 / 全库模式选择，语言可切换
- 引用：检索结果附带 `[language/file#chunk]`，回答时要求模型引用
- API/LLM：DeepSeek Chat（读取 `.env`），不依赖 OpenAI

