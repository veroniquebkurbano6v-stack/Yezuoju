"""
DeepSeek 智能检索代理系统 - StoryRag v2.0
集成 DeepSeek 模型、LangChain 工具、角色管理与三层记忆系统
"""

import os
import json
import asyncio
import logging
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool, render_text_description
from langgraph.prebuilt import create_react_agent

from src.models import Document, RAGResponse, UsageSummary
from src.core.role_manager import RoleManager, get_role_manager
from src.core.memory_manager import MemoryManager, get_memory_manager
from src.core.ollama_client import get_ollama_enhancer

logger = logging.getLogger(__name__)


def _safe_async_run(coro):
    """
    安全地在同步上下文中运行异步协程

    兼容：
    - 无事件循环时：直接 asyncio.run()
    - 已有事件循环（如 FastAPI）：在线程中运行
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, coro)
        try:
            return future.result(timeout=10)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise TimeoutError("协程执行超时")


class DeepSeekRetrievalAgent:
    """
    DeepSeek 智能检索代理（角色 + 记忆增强版）

    在原有 RAG 能力基础上增加：
    - 稳定角色行为：结构化角色配置 + 多层注入 + 防漂移机制
    - 三层记忆系统：短期记忆 + 长期记忆 + 时间线记忆
    """

    def __init__(self, vector_db_path: str, api_key: str = None, base_url: str = None,
                 embedding_model: str = "intfloat/multilingual-e5-large", tools_instance=None,
                 role_id: str = None, memory_manager: MemoryManager = None):
        """
        初始化 DeepSeek 检索代理

        Args:
            vector_db_path: 向量数据库路径
            api_key: DeepSeek API 密钥
            base_url: DeepSeek API 基础 URL
            embedding_model: 嵌入模型名称
            tools_instance: 已初始化的 langchain tools 实例
            role_id: 角色标识符，如 'humorous_butler'
            memory_manager: 记忆管理器实例
        """
        self.vector_db_path = vector_db_path
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = base_url or "https://api.deepseek.com/v1"
        self.embedding_model = embedding_model

        if not self.api_key:
            raise ValueError("DeepSeek API 密钥未设置。请设置 DEEPSEEK_API_KEY 环境变量或直接传入 api_key 参数。")

        self.llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=0.1,
            max_tokens=2000
        )

        logger.info("DeepSeek 模型初始化完成")

        self.role_manager = get_role_manager(role_id) if role_id else get_role_manager()
        self.memory_manager = memory_manager or get_memory_manager()

        logger.info(f"[DeepSeekAgent] 角色：{self.role_manager.profile.display_name}")
        logger.info("[DeepSeekAgent] 三层记忆系统已绑定")

        # 初始化工具
        if tools_instance is not None:
            # 支持两种格式：对象（有 .tools 属性）或列表
            if isinstance(tools_instance, list):
                # 如果直接是列表，直接使用
                self.tools = tools_instance
                self.tools_instance = None
                logger.info(f"复用已初始化的 langchain tools（列表格式），共 {len(self.tools)} 个工具")
            elif hasattr(tools_instance, 'tools'):
                # 如果是对象，访问其 .tools 属性
                self.tools_instance = tools_instance
                self.tools = tools_instance.tools
                logger.info("复用已初始化的 langchain tools（对象格式）")
            else:
                raise ValueError(f"Invalid tools_instance type: {type(tools_instance)}")
        else:
            # 添加 src 目录到 Python 路径
            import sys
            from pathlib import Path
            src_path = Path(__file__).resolve().parents[1]
            if str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))
            
            logger.info("[DeepSeekAgent] 开始创建 LangChain 工具...")
            try:
                from src.tools.langchain_retrieval_tools import SmartRetrievalTool
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

        self._build_system_message()

        self.agent_executor = create_react_agent(self.llm, self.tools)

        logger.info("DeepSeek Agent 创建完成")

    def _build_system_message(self):
        """动态构建系统消息（角色配置 + 检索任务指令）"""
        role_segment = self.role_manager.get_role_prompt_segment()

        retrieval_segment = """## 检索任务指令

【重要】为了快速响应用户，你必须遵循以下原则：

### 核心工作流（一步检索）

**首选方案：直接调用 smart_retrieval 工具，并传入分析结果**

smart_retrieval 是一个智能检索工具，它会根据你提供的信息执行检索。

✅ **推荐调用方式**（只需 1 次工具调用）：
```
smart_retrieval(query="用户问题", keywords=["分析出的关键词"])
```

### 你需要在调用前完成分析：

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

**3. 提取元数据过滤条件（如果用户明确指定了文档）**：
   - 当用户提到具体的书名、文件名、章节名时，提取为 metadata_filter
   - 常见场景：
     * "狂人日记讲了什么" → metadata_filter={"pdf_filename": "鲁迅短篇小说集：呐喊.pdf"}
     * "安徒生童话中的丑小鸭" → metadata_filter={"pdf_filename": "安徒生童话.pdf"}
     * "洪武年间的事件" → metadata_filter={"pdf_filename": "洪武：朱元璋的成与败.pdf"}
   - 注意：只有当用户明确提到某本书时才使用，否则不传此参数
   - 可用的文件名（从上下文或工具返回中获取）：
     * "鲁迅短篇小说集：呐喊.pdf"
     * "安徒生童话.pdf"
     * "洪武：朱元璋的成与败.pdf"
     * "英文原版. Spider-Man： Homecoming 蜘蛛侠：英雄归来(电影同名小说).pdf"
     * "日语化物语.pdf"

**4. 立即调用检索工具**：
   - 不需要单独调用 query_classifier！
   - 直接把分析结果传给 smart_retrieval

### 可用工具

1. **smart_retrieval** (推荐使用)
   - 智能检索工具，执行混合检索和重排序
   - 参数：query (必需), query_type (可选), keywords (可选), top_k (可选), metadata_filter (可选)
   - 示例：
     ```
     smart_retrieval(
         query="狂人日记讲了什么",
         keywords=["狂人日记", "内容", "故事", "情节"],
         query_type="故事梗概",
         metadata_filter={"pdf_filename": "鲁迅短篇小说集：呐喊.pdf"}
     )
     ```

### 回答规范（至关重要）

**你必须严格遵守以下回答格式：**

1. **答案优先** (前 30 字必须包含核心信息)
   - ✅ 正确："朱元璋出生于赤贫家庭，父亲朱五四是佃农..."
   - ❌ 错误："根据检索到的文档，我们可以看到..."

2. **简洁准确** (200-600 字最佳)

3. **仅基于检索结果回答**
   - 所有答案必须有文档来源
   - 如果文档中没有，直接说"未找到相关信息"

4. **结构化呈现**
   - 使用小标题：`### 1. 家庭背景`
   - 使用列表：`- 父母：朱五四`
   - 关键信息加粗：`**赤贫家庭**`

5. **结合记忆回答**
   - 如果上下文中包含"长期记忆"或"时间线"信息，请结合这些信息回答
   - 在引用用户历史信息时，使用自然的过渡，如"根据您之前提到的..."

### 角色一致性要求

1. 回复中保持角色语气和风格一致
2. 在消息上下文中如果出现角色的「标志性用语」，可自然地融入
3. 如果上下文中出现角色强化提示，严格按照提示调整语气
4. **严禁**使用"作为AI"、"根据我的训练数据"等暴露AI身份的表述"""

        self.system_message = f"{role_segment}\n\n---\n\n{retrieval_segment}"

        logger.info("DeepSeek Agent 创建完成")

    def chat(self, user_input: str, chat_history: List[Dict[str, str]] = None, session_id: str = "") -> RAGResponse:
        """

        Args:
            user_input: 用户输入的问题
            chat_history: 对话历史记录
            session_id: 会话 ID

        Returns:
            RAGResponse 对象
        """
        import asyncio
        try:
            self._build_system_message()
            logger.info(f"[DeepSeekAgent.chat] 用户输入：'{user_input}'")

            if not self.tools:
                return RAGResponse(answer="工具未初始化", sources=[], confidence=0.0, session_id=session_id)

            logger.info("[DeepSeekAgent] 调用 Agent Executor...")
            from langchain_core.messages import HumanMessage, SystemMessage

            messages = []

            if self.system_message:
                messages.append(SystemMessage(content=self.system_message))

            # 注入角色强化片段
            reinforcement = self.role_manager.on_turn_start()
            if reinforcement:
                messages.append(HumanMessage(content=f"[角色提示] {reinforcement}"))

            # 注入三层记忆上下文
            memory_context = _safe_async_run(self.memory_manager.build_full_context(current_query=user_input))

            if memory_context:
                logger.info(f"[DeepSeekAgent] 记忆上下文已注入，长度：{len(memory_context)} 字符")
                messages.append(SystemMessage(content=memory_context))
            else:
                logger.info(f"[DeepSeekAgent] 无记忆上下文可注入")

            if chat_history and len(chat_history) > 0:
                for msg in chat_history:
                    if msg.get("role") == "user":
                        messages.append(HumanMessage(content=msg["content"]))
                    elif msg.get("role") == "assistant":
                        from langchain_core.messages import AIMessage
                        messages.append(AIMessage(content=msg["content"]))

            messages.append(HumanMessage(content=user_input))

            logger.info(f"[DeepSeekAgent] 消息列表长度：{len(messages)}")

            response = self.agent_executor.invoke({"messages": messages})

            logger.info(f"[DeepSeekAgent] Agent 执行完成")

            final_answer = ""
            retrieved_docs = []
            tool_calls_info = []

            if isinstance(response, dict):
                output_messages = response.get("messages", [])
                if output_messages:
                    for msg in output_messages:
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tc in msg.tool_calls:
                                tool_calls_info.append({
                                    "name": tc.get("name", ""),
                                    "args": tc.get("args", {}),
                                    "id": tc.get("id", "")
                                })

                        from langchain_core.messages import AIMessage
                        if isinstance(msg, AIMessage):
                            final_answer = msg.content

                    if not final_answer and output_messages:
                        last_message = output_messages[-1]
                        if hasattr(last_message, "content"):
                            final_answer = last_message.content
                        elif isinstance(last_message, str):
                            final_answer = last_message

            if not final_answer:
                logger.warning("[DeepSeekAgent] Agent 返回空回答")
                final_answer = "抱歉，我暂时无法回答您的问题。请尝试换一种问法或提供更详细的信息。"

            source_documents = []
            try:
                from src.tools.langchain_retrieval_tools import get_last_retrieval_result
                retrieved_docs_raw = get_last_retrieval_result()

                for doc_data in retrieved_docs_raw:
                    doc = Document(
                        content=doc_data.get('document', ''),
                        metadata=doc_data.get('metadata', {}),
                        score=doc_data.get('final_score', doc_data.get('score', 0.0)),
                        doc_id=doc_data.get('doc_id', '')
                    )
                    source_documents.append(doc)

                logger.info(f"[DeepSeekAgent] 从全局缓存中获取到 {len(source_documents)} 篇文档")
            except Exception as e:
                logger.warning(f"[DeepSeekAgent] 获取全局检索结果失败：{e}")

            confidence = 0.0
            if source_documents:
                max_score = max(doc.score for doc in source_documents) if source_documents else 0
                doc_count_factor = min(len(source_documents) / 5.0, 1.0)
                confidence = max_score * 0.7 + doc_count_factor * 0.3

            # 角色包装
            final_answer = self.role_manager.wrap_answer(final_answer)

            # Ollama 角色强化（本地模型二次润色，增强管家等角色设定）
            try:
                ollama_enhancer = get_ollama_enhancer()
                if ollama_enhancer.is_available():
                    logger.info(f"[DeepSeekAgent] 调用 Ollama 角色强化，当前角色：{self.role_manager.profile.display_name}")
                    final_answer = ollama_enhancer.enhance_response(
                        final_answer,
                        self.role_manager.role_id
                    )
                else:
                    logger.info("[DeepSeekAgent] Ollama 不可用，跳过角色强化")
            except Exception as e:
                logger.warning(f"[DeepSeekAgent] Ollama 角色强化异常：{e}")

            # 角色漂移检测
            if self.role_manager.detect_role_drift(final_answer):
                logger.warning("[DeepSeekAgent] 检测到角色漂移，在回答末尾添加角色锚定")
                final_answer += f"\n\n*——{self.role_manager.profile.display_name} 敬上*"

            # 记录到三层记忆
            try:
                _safe_async_run(self.memory_manager.record_user_message(user_input, session_id=session_id, importance=2))
                _safe_async_run(self.memory_manager.record_assistant_response(final_answer, session_id=session_id, importance=2))
            except Exception as e:
                logger.warning(f"[DeepSeekAgent] 记忆记录失败：{e}")

            return RAGResponse(
                answer=final_answer,
                sources=source_documents,
                confidence=confidence,
                session_id=session_id,
                usage_tokens={"input": 0, "output": 0}
            )

        except Exception as e:
            error_msg = f"DeepSeek 检索代理执行失败：{str(e)}"
            logger.error(error_msg)
            return RAGResponse(
                answer=f"抱歉，处理您的请求时出现错误：{error_msg}",
                sources=[],
                confidence=0.0,
                session_id=session_id
            )

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
                         embedding_model: str = "intfloat/multilingual-e5-large", tools_instance=None,
                         role_id: str = None, memory_manager=None) -> DeepSeekRetrievalAgent:
    """


    Args:
        vector_db_path: 向量数据库路径
        api_key: DeepSeek API 密钥
        base_url: DeepSeek API 基础 URL
        embedding_model: 嵌入模型名称
        tools_instance: 已初始化的 langchain tools 实例
        role_id: 角色标识符
        memory_manager: 记忆管理器实例

    Returns:
        DeepSeekRetrievalAgent 实例
    """
    return DeepSeekRetrievalAgent(
        vector_db_path=vector_db_path,
        api_key=api_key,
        base_url=base_url,
        embedding_model=embedding_model,
        tools_instance=tools_instance,
        role_id=role_id,
        memory_manager=memory_manager,
    )
