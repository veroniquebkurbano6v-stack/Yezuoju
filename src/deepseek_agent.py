"""
DeepSeek 智能检索代理系统
集成 DeepSeek 模型和 LangChain 工具，实现自主检索和问答
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool, render_text_description
from langgraph.prebuilt import create_react_agent

logger = logging.getLogger(__name__)


class DeepSeekRetrievalAgent:
    """DeepSeek 智能检索代理"""

    def __init__(self, vector_db_path: str, api_key: str = None, base_url: str = None,
                 embedding_model: str = "intfloat/multilingual-e5-large", tools_instance=None):
        """
        初始化 DeepSeek 检索代理

        Args:
            vector_db_path: 向量数据库路径
            api_key: DeepSeek API 密钥
            base_url: DeepSeek API 基础 URL
            embedding_model: 嵌入模型名称
            tools_instance: 已初始化的 langchain tools 实例（可选，避免重复初始化）
        """
        self.vector_db_path = vector_db_path
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = base_url or "https://api.deepseek.com/v1"
        self.embedding_model = embedding_model

        if not self.api_key:
            raise ValueError("DeepSeek API 密钥未设置。请设置 DEEPSEEK_API_KEY 环境变量或直接传入 api_key 参数。")

        # 初始化 DeepSeek 模型
        self.llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=0.1,
            max_tokens=2000
        )

        logger.info("DeepSeek 模型初始化完成")

        # 初始化工具
        if tools_instance is not None:
            self.tools_instance = tools_instance
            self.tools = tools_instance.tools
            logger.info("复用已初始化的 langchain tools")
        else:
            # 添加 src 目录到 Python 路径
            import sys
            from pathlib import Path
            src_path = Path(__file__).resolve().parent
            if str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))
            
            logger.info("[DeepSeekAgent] 开始创建 LangChain 工具...")
            try:
                from langchain_retrieval_tools import SmartRetrievalTool
                logger.info("[DeepSeekAgent] 导入成功，开始创建工具实例...")
                
                # 创建工具实例
                logger.info("[DeepSeekAgent] 创建 SmartRetrievalTool...")
                self.smart_retrieval = SmartRetrievalTool()
                logger.info("[DeepSeekAgent] SmartRetrievalTool 创建成功")
                
                self.tools = [
                    self.smart_retrieval,   # 只保留智能检索工具
                ]
                logger.info(f"[DeepSeekAgent] LangChain 工具初始化完成，共 {len(self.tools)} 个工具")
                self.tools_instance = None
            except Exception as e:
                logger.error(f"[DeepSeekAgent] ❌ 工具创建失败：{e}", exc_info=True)
                raise

        logger.info(f"LangChain 工具初始化完成，共 {len(self.tools)} 个工具")

        # 创建 Agent - 使用优化的系统消息（减少 API 调用次数）
        self.system_message = """你是一个专业的文档检索助手。

【重要】为了快速响应用户，你必须遵循以下原则：

## 🚀 核心工作流（一步检索）

**首选方案：直接调用 smart_retrieval 工具，并传入分析结果**

smart_retrieval 是一个智能检索工具，它会根据你提供的信息执行检索。

✅ **推荐调用方式**（只需 1 次工具调用）：
```
smart_retrieval(query="用户问题", keywords=["分析出的关键词"])
```

### 🔍 你需要在调用前完成分析：

**1. 自动分析问题类型**（4 类之一）：
   - 故事梗概：询问书籍、章节的主要内容、情节概要等
   - 事实依据：询问历史事实、时间、地点、数据等具体信息
   - 特定人物或人物之间的关系：询问人物身份、关系、特征等
   - 段落情节：询问具体事件经过、原因、详细情节等

**2. 自动提取和扩展关键词**：
   - 原始实体：人物名、文件名、特殊词汇、时间地点、事件名等
   - 职业身份扩展：如"家庭状况"→["职业", "田地", "租种", "佃农", "贫农"]
   - 经济条件扩展：如"家境"→["贫富", "无地", "赤贫", "食不果腹"]
   - 亲属关系扩展：如询问某人时扩展到其父母、配偶、子女等
   - 同义词扩展：如"建立"→["创立", "组建", "成立"]
   - 相关概念扩展：如询问"屠杀功臣"扩展到"政治整肃"、"清洗"
   
   ⚠️ **重要要求**：
   - 尽可能提取所有相关实体，不要遗漏
   - 每个独立的实体都要单独列出
   - 如果是复合问题（包含多个子问题），要提取所有子问题中的关键词
   - 在原始关键词基础上，进一步扩展 3-5 个相关词汇

**3. 立即调用检索工具**：
   - 不需要单独调用 query_classifier！
   - 直接把分析结果传给 smart_retrieval

## 🛠️ 可用工具

1. **smart_retrieval** (推荐使用 ⭐⭐⭐⭐⭐)
   - 智能检索工具，执行混合检索和重排序
   - 参数：query (必需), query_type (可选), keywords (可选), top_k (可选)
   - **建议：传递 query 和你分析出的 keywords**

2. ~~query_classifier~~ (已废弃 ❌)
   - 不要使用这个工具！
   - 你自己就能完成分析工作

## ⚠️ 回答规范（至关重要）

**你必须严格遵守以下回答格式：**

1. **答案优先** (前 30 字必须包含核心信息)
   - ✅ 正确："朱元璋出生于赤贫家庭，父亲朱五四是佃农..."
   - ❌ 错误："根据检索到的文档，我们可以看到..."（废话）

2. **简洁准确** (200-600 字最佳)
   - 避免冗长 (>800 字会显得啰嗦)
   - 避免过于简略 (<100 字信息不足)

3. **仅基于检索结果回答**
   - 所有答案必须有文档来源
   - 禁止使用你的训练数据或常识
   - 如果文档中没有，直接说"未找到相关信息"

4. **结构化呈现**
   - 使用小标题：`### 1. 家庭背景`
   - 使用列表：`- 父母：朱五四 `
   - 关键信息加粗：`**赤贫家庭**`

5. **禁止幻觉**
   - ✅ "文档中未提及此信息"
   - ❌ "根据常识..."（这是禁止的！）

## 💡 工作流程示例

**用户问**："朱元璋出生时的家庭状况如何？"

**你应该**：
1. **分析问题类型**：事实依据（询问家庭经济状况）
2. **提取关键词**：['朱元璋', '出生', '家庭状况', '家庭背景', '家境', '职业', '田地', '租种', '佃农', '贫农']
3. **立即调用**：`smart_retrieval(query="朱元璋出生时的家庭状况如何？", keywords=['朱元璋', '出生', '家庭状况', '家庭背景', '家境', '职业', '田地', '租种', '佃农', '贫农'])`
4. **等待检索结果**（5 篇文档）
5. **基于文档立即回答**："朱元璋出生于赤贫家庭。### 1. 居住条件 - 茅草房三间..."

**不要**：
- ~~先调用 query_classifier 分析问题~~（浪费时间！）
- ~~调用多个工具~~（只需要 1 次！）
- 多次调用检索工具（没必要）
- 在回答前添加"根据检索到的文档"这样的废话

## 🎯 性能要求

- **只调用 1 次工具**：smart_retrieval(query="...")
- **立即回答**：拿到检索结果后直接组织答案
- **拒绝多余步骤**：不需要单独分析问题

记住：用户需要快速、准确的答案，不需要看到你的分析过程！**一次调用，立即回答！**"""

        # 使用 LangGraph 创建 ReAct Agent
        self.agent_executor = create_react_agent(self.llm, self.tools)

        logger.info("DeepSeek Agent 创建完成")

    def chat(self, user_input: str, chat_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        与 DeepSeek 检索代理对话

        Args:
            user_input: 用户输入的问题
            chat_history: 对话历史记录

        Returns:
            包含回答和工具调用信息的字典
        """
        try:
            logger.info(f"[DeepSeekAgent.chat] 用户输入：'{user_input}'")
            logger.info(f"[DeepSeekAgent.chat] 聊天历史长度：{len(chat_history) if chat_history else 0}")

            if not self.tools:
                return {
                    "success": False,
                    "user_input": user_input,
                    "error": "工具未初始化",
                    "chat_history": chat_history or []
                }

            # 使用 Agent Executor 执行
            logger.info("[DeepSeekAgent] 调用 Agent Executor...")
            
            # 构建消息列表
            from langchain_core.messages import HumanMessage, SystemMessage
            
            messages = []
            
            # 添加系统消息
            if self.system_message:
                messages.append(SystemMessage(content=self.system_message))
            
            # 添加对话历史
            if chat_history and len(chat_history) > 0:
                for msg in chat_history[-10:]:  # 只使用最近 10 轮对话
                    if msg.get("role") == "user":
                        messages.append(HumanMessage(content=msg["content"]))
                    elif msg.get("role") == "assistant":
                        from langchain_core.messages import AIMessage
                        messages.append(AIMessage(content=msg["content"]))
            
            # 添加当前用户输入
            messages.append(HumanMessage(content=user_input))
            
            logger.info(f"[DeepSeekAgent] 消息列表长度：{len(messages)}")

            # 执行 Agent
            response = self.agent_executor.invoke({"messages": messages})
            
            logger.info(f"[DeepSeekAgent] Agent 执行完成")
            
            # 从响应中提取最后的 AI 消息和工具调用信息
            final_answer = ""
            retrieved_docs = []
            tool_calls_info = []
            
            if isinstance(response, dict):
                output_messages = response.get("messages", [])
                if output_messages:
                    # 遍历所有消息，提取工具调用信息
                    for msg in output_messages:
                        # 检查是否是工具调用消息
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tc in msg.tool_calls:
                                tool_call_info = {
                                    "name": tc.get("name", ""),
                                    "args": tc.get("args", {}),
                                    "id": tc.get("id", "")
                                }
                                tool_calls_info.append(tool_call_info)
                                logger.info(f"[DeepSeekAgent] 🛠️ 工具调用：{tool_call_info['name']}")
                                logger.info(f"[DeepSeekAgent]    参数：{tool_call_info['args']}")
                        
                        # 获取最后一条 AI 消息
                        if hasattr(msg, "content"):
                            # 检查是否是 AIMessage（不是 ToolMessage）
                            from langchain_core.messages import AIMessage
                            if isinstance(msg, AIMessage):
                                final_answer = msg.content
                    
                    # 如果没有找到 AIMessage，尝试获取最后一条非工具消息
                    if not final_answer and output_messages:
                        last_message = output_messages[-1]
                        if hasattr(last_message, "content"):
                            final_answer = last_message.content
                        elif isinstance(last_message, str):
                            final_answer = last_message
                    
                    logger.info(f"[DeepSeekAgent] 输出：{final_answer[:200]}...")
                    logger.info(f"[DeepSeekAgent] 工具调用次数：{len(tool_calls_info)}")
            
            # 如果没有回答，返回错误
            if not final_answer:
                logger.warning("[DeepSeekAgent] Agent 返回空回答")
                final_answer = "抱歉，我暂时无法回答您的问题。请尝试换一种问法或提供更详细的信息。"
            
            # 从工具调用信息中提取检索结果
            for tool_call in tool_calls_info:
                tool_name = tool_call.get("name", "")
                tool_args = tool_call.get("args", {})
                
                # 如果是检索工具调用，记录相关信息
                if tool_name == "smart_retrieval":
                    logger.info(f"[DeepSeekAgent] 检测到检索工具调用：{tool_name}")
            
            # 从全局变量中获取最后一次检索结果
            try:
                from langchain_retrieval_tools import get_last_retrieval_result
                retrieved_docs_raw = get_last_retrieval_result()
                
                # 转换为前端期望的格式
                for doc in retrieved_docs_raw:
                    retrieved_docs.append({
                        "rank": doc.get('rank', 0),
                        "document": doc.get('document', ''),
                        "metadata": doc.get('metadata', {}),
                        "score": doc.get('score', 0)
                    })
                
                logger.info(f"[DeepSeekAgent] 从全局缓存中获取到 {len(retrieved_docs)} 篇文档")
            except Exception as e:
                logger.warning(f"[DeepSeekAgent] 获取全局检索结果失败：{e}")
            
            logger.info(f"[DeepSeekAgent] 检索到 {len(retrieved_docs)} 篇文档")

            # 返回结果
            return {
                "success": True,
                "user_input": user_input,
                "answer": final_answer,
                "chat_history": chat_history or [],
                "retrieved_docs": retrieved_docs,  # 添加检索到的文档列表
                "tool_calls": tool_calls_info,  # 添加工具调用信息
                "intermediate_steps": []  # LangGraph 不直接提供 intermediate_steps
            }

        except Exception as e:
            error_msg = f"DeepSeek 检索代理执行失败：{str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "user_input": user_input,
                "error": error_msg,
                "chat_history": chat_history or []
            }

    def simple_query(self, query: str) -> str:
        """
        简单查询接口（直接使用智能检索工具）

        Args:
            query: 用户查询

        Returns:
            检索结果的文本
        """
        try:
            logger.info(f"[DeepSeekAgent.simple_query] 执行简单查询：{query}")
            
            # 直接使用智能检索工具
            result = self.smart_retrieval._run(query=query, top_k=5)
            
            logger.info(f"[DeepSeekAgent.simple_query] 查询完成，结果长度：{len(result)}")
            return result
        except Exception as e:
            logger.error(f"简单查询失败：{e}")
            return f"查询失败：{str(e)}"


def create_deepseek_agent(vector_db_path: str, api_key: str = None, base_url: str = None,
                         embedding_model: str = "intfloat/multilingual-e5-large", tools_instance=None) -> DeepSeekRetrievalAgent:
    """
    创建 DeepSeek 检索代理实例的工厂函数

    Args:
        vector_db_path: 向量数据库路径
        api_key: DeepSeek API 密钥
        base_url: DeepSeek API 基础 URL
        embedding_model: 嵌入模型名称
        tools_instance: 已初始化的 langchain tools 实例

    Returns:
        DeepSeekRetrievalAgent 实例
    """
    return DeepSeekRetrievalAgent(
        vector_db_path=vector_db_path,
        api_key=api_key,
        base_url=base_url,
        embedding_model=embedding_model,
        tools_instance=tools_instance
    )
