# DeepSeek Agent 使用指南

## 📖 概述

`deepseek_agent.py` 是一个基于 DeepSeek 模型的智能检索代理，它使用 LangChain Agents 框架，可以自主调用封装好的检索工具来回答用户问题。

## 🎯 核心特性

### 1. **Agent 架构**
- 使用 LangChain 的 `create_openai_tools_agent` 创建 OpenAI Functions Agent
- 通过 `AgentExecutor` 执行工具调用和推理
- 支持多轮对话和历史记录

### 2. **集成的工具**
Agent 自动集成以下检索工具：

| 工具名称 | 功能描述 |
|---------|---------|
| `query_classifier` | 问题分类工具，分析用户问题类型并提取关键词（推荐首先使用） |
| `smart_retrieval` | 智能检索工具，根据问题类型自动选择最优策略（必须传递 query_type 和 keywords！） |

### 3. **系统提示词**
Agent 内置系统提示词，指导其：
- 仔细分析用户问题，选择合适的检索工具
- 如果第一次检索结果不理想，可以尝试其他工具
- 基于检索结果给出准确、有条理的回答
- 如果检索结果为空，诚实地告诉用户没有找到相关信息

## 💻 使用方法

### 方式一：直接实例化

```python
from src.deepseek_agent import DeepSeekRetrievalAgent

# 创建 Agent 实例
agent = DeepSeekRetrievalAgent(
    vector_db_path="src/data/chunks",  # 向量数据库路径
    api_key="your_deepseek_api_key",   # DeepSeek API 密钥（或使用环境变量 DEEPSEEK_API_KEY）
    base_url="https://api.deepseek.com/v1"  # API 基础 URL
)

# 对话
response = agent.chat(
    user_input="朱元璋是怎么从贫农变成皇帝的？",
    chat_history=[]  # 可选的对话历史记录
)

if response["success"]:
    print(f"回答：{response['answer']}")
else:
    print(f"错误：{response['error']}")
```

### 方式二：使用工厂函数

```python
from src.deepseek_agent import create_deepseek_agent

# 使用工厂函数创建 Agent
agent = create_deepseek_agent(
    vector_db_path="src/data/chunks",
    api_key=None,  # None 表示使用环境变量 DEEPSEEK_API_KEY
    embedding_model="intfloat/multilingual-e5-large"
)

# 简单查询（不使用 Agent，直接调用工具）
result = agent.simple_query("明朝建立的过程是怎样的？")
print(result)
```

### 方式三：复用已初始化的工具实例

```python
from src.langchain_retrieval_tools import SmartRetrievalTool, QueryClassifierTool
from src.deepseek_agent import DeepSeekRetrievalAgent

# 先创建工具实例
tools_instance = YourToolsClass()  # 假设你有自己的工具封装类

# 创建 Agent，复用工具实例
agent = DeepSeekRetrievalAgent(
    vector_db_path="src/data/chunks",
    api_key="your_api_key",
    tools_instance=tools_instance  # 传入已初始化的工具实例
)
```

## 📝 API 接口

### `chat(user_input, chat_history)`

与 Agent 进行对话。

**参数：**
- `user_input` (str): 用户输入的问题
- `chat_history` (List[Dict[str, str]], optional): 对话历史记录，格式为 `[{"role": "user/assistant", "content": "内容"}]`

**返回值：**
```python
{
    "success": True,
    "user_input": "用户问题",
    "answer": "Agent 的回答",
    "chat_history": [...],  # 对话历史
    "intermediate_steps": [...]  # Agent 调用工具的中间步骤
}
```

### `simple_query(query)`

简单查询接口，直接使用智能检索工具（不经过 Agent）。

**参数：**
- `query` (str): 用户查询

**返回值：**
- `str`: 检索结果的 JSON 字符串

## 🔧 配置选项

### 环境变量

在 `.env` 文件中配置：

```bash
# DeepSeek API 配置
DEEPSEEK_API_KEY=sk-your-api-key-here

# 向量数据库路径
VECTOR_DB_PATH=src/data/chunks

# 嵌入模型
EMBEDDING_MODEL=intfloat/multilingual-e5-large
```

### 初始化参数

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `vector_db_path` | str | - | 向量数据库路径 |
| `api_key` | str | None | DeepSeek API 密钥（可从环境变量读取） |
| `base_url` | str | `https://api.deepseek.com/v1` | API 基础 URL |
| `embedding_model` | str | `intfloat/multilingual-e5-large` | 嵌入模型名称 |
| `tools_instance` | object | None | 已初始化的工具实例（可选） |

## 🌟 使用示例

### 单轮对话

```python
agent = DeepSeekRetrievalAgent(vector_db_path="src/data/chunks")

response = agent.chat("朱元璋是哪一年出生的？")
print(response["answer"])
```

### 多轮对话

```python
chat_history = []

# 第一轮
response1 = agent.chat("洪武元年是公元多少年？", chat_history)
print(f"Q1: {response1['answer']}")
chat_history.append({"role": "user", "content": "洪武元年是公元多少年？"})
chat_history.append({"role": "assistant", "content": response1['answer']})

# 第二轮（基于上一轮）
response2 = agent.chat("那这一年发生了什么大事？", chat_history)
print(f"Q2: {response2['answer']}")
```

### 查看 Agent 的工具调用过程

```python
response = agent.chat("鄱阳湖之战中陈友谅是怎么死的？")

# 查看中间步骤
for step in response.get("intermediate_steps", []):
    print(f"调用的工具：{step['tool']}")
    print(f"工具输入：{step['input']}")
    print(f"工具输出：{step['output'][:200]}...")
```

## ⚙️ Agent 工作原理

1. **接收用户输入** → Agent 分析问题
2. **选择工具** → 根据问题类型选择合适的检索工具
3. **执行检索** → 调用工具获取检索结果
4. **推理回答** → 基于检索结果生成答案
5. **返回结果** → 将答案返回给用户

整个过程完全自主，Agent 会根据需要多次调用不同工具，直到找到满意的答案。

## 🛠️ 调试技巧

### 启用详细日志

```python
import logging
logging.basicConfig(level=logging.INFO)

agent = DeepSeekRetrievalAgent(...)
```

### 查看工具调用详情

```python
response = agent.chat("你的问题")

# 打印中间步骤
import json
for i, step in enumerate(response.get("intermediate_steps", []), 1):
    print(f"\n=== 步骤 {i} ===")
    print(f"工具：{step.get('tool', 'N/A')}")
    print(f"输入：{json.dumps(step.get('input', {}), ensure_ascii=False)}")
    print(f"输出：{step.get('output', '')[:300]}...")
```

## 📋 最佳实践

1. **优先使用 smart_retrieval** - Agent 会自动选择最优策略
2. **提供对话历史** - 多轮对话时传入 `chat_history` 提升上下文理解
3. **监控中间步骤** - 通过 `intermediate_steps` 了解 Agent 的推理过程
4. **合理设置 top_k** - 根据需求调整检索结果数量（默认 5）

## ❓ 常见问题

### Q: Agent 为什么不调用工具？
A: 检查以下几点：
- 确保 `api_key` 正确
- 确保向量数据库路径存在
- 查看日志中的错误信息
- 尝试使用 `simple_query` 验证工具是否正常工作

### Q: 如何优化 Agent 的回答质量？
A: 
- 提供更具体的问题描述
- 添加对话历史帮助理解上下文
- 调整系统提示词中的指导语
- 增加 `max_iterations` 允许更多推理轮次

### Q: 能否自定义工具？
A: 可以！在初始化时传入自定义的 `tools` 列表即可。

---

**📌 注意**：本脚本依赖于 `langchain_retrieval_tools.py` 中封装的检索工具，请确保该模块已正确初始化并可用。
