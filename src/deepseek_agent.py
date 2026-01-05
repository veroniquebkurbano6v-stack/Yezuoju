"""
DeepSeek智能检索代理系统
集成DeepSeek模型和LangChain工具，实现自主检索和问答
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_tools import get_langchain_tools

logger = logging.getLogger(__name__)

TOOL_PARAM_PROMPT = """你是一个专业的文档检索助手。

用户查询：{query}

请从用户查询中提取以下信息并以JSON格式返回：
- query: 用户的查询内容（必须原样提取，不能修改）
- filter_pdf: 用户指定的PDF文件名（如果没有指定则为None）

要求：
1. query 必须完全等于用户查询，不能修改、添加或删除任何内容
2. 如果用户没有指定PDF文件，filter_pdf 必须是 null
3. 只返回JSON对象，不要有任何其他内容

JSON格式：
{{"query": "用户查询内容", "filter_pdf": "文件名.pdf或null"}}
"""


class DeepSeekRetrievalAgent:
    """DeepSeek智能检索代理"""

    def __init__(self, vector_db_path: str, api_key: str = None, base_url: str = None,
                 embedding_model: str = "intfloat/multilingual-e5-large", tools_instance=None):
        """
        初始化DeepSeek检索代理

        Args:
            vector_db_path: 向量数据库路径
            api_key: DeepSeek API密钥
            base_url: DeepSeek API基础URL
            embedding_model: 嵌入模型名称
            tools_instance: 已初始化的 langchain tools 实例（可选，避免重复初始化）
        """
        self.vector_db_path = vector_db_path
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = base_url or "https://api.deepseek.com/v1"
        self.embedding_model = embedding_model

        if not self.api_key:
            raise ValueError("DeepSeek API密钥未设置。请设置DEEPSEEK_API_KEY环境变量或直接传入api_key参数。")

        self.llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=0.1,
            max_tokens=2000
        )

        logger.info("DeepSeek模型初始化完成")

        if tools_instance is not None:
            self.tools = tools_instance.tools
            self.smart_retrieval = tools_instance.smart_retrieval
            self.smart_retrieval_impl = tools_instance.smart_retrieval_impl
            logger.info("复用已初始化的 langchain tools")
        else:
            tools_instance = get_langchain_tools(self.vector_db_path, self.embedding_model)
            self.tools = tools_instance.tools
            self.smart_retrieval = tools_instance.smart_retrieval
            self.smart_retrieval_impl = tools_instance.smart_retrieval_impl

        self.param_parser = JsonOutputParser()
        self.param_prompt = ChatPromptTemplate.from_messages([
            ("human", TOOL_PARAM_PROMPT)
        ])

        logger.info("DeepSeek检索代理初始化完成")

    def _extract_params(self, query: str) -> Dict[str, Any]:
        """从用户查询中提取工具参数"""
        try:
            chain = self.param_prompt | self.llm | self.param_parser
            result = chain.invoke({"query": query})
            logger.info(f"[DeepSeekAgent] 提取参数: {result}")
            return result
        except Exception as e:
            logger.warning(f"[DeepSeekAgent] 参数提取失败，使用原始查询: {e}")
            return {"query": query, "filter_pdf": None}

    def chat(self, user_input: str, chat_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        与DeepSeek检索代理对话

        Args:
            user_input: 用户输入的问题
            chat_history: 对话历史记录（仅用于最终回答，不影响检索）

        Returns:
            包含回答和工具调用信息的字典
        """
        try:
            logger.info(f"[DeepSeekAgent.chat] 用户输入: '{user_input}'")
            logger.info(f"[DeepSeekAgent.chat] 聊天历史长度: {len(chat_history) if chat_history else 0}")

            if not self.tools or not self.smart_retrieval:
                return {
                    "success": False,
                    "user_input": user_input,
                    "error": "工具未初始化",
                    "chat_history": chat_history or []
                }

            params = self._extract_params(user_input)
            query = params.get("query", user_input)
            filter_pdf = params.get("filter_pdf")

            # 初始化最终答案变量
            final_answer = None
            answer_source = "none"

            logger.info(f"[DeepSeekAgent] 执行检索: query='{query}', filter_pdf={filter_pdf}")

            import json
            query_request_json = json.dumps({"query": query, "filter_pdf": filter_pdf}, ensure_ascii=False)

            # 尝试使用 invoke 方法调用工具，如果失败则直接调用函数
            try:
                tool_result = self.smart_retrieval.invoke({"query_request_json": query_request_json, "top_k": 60})
            except Exception:
                # 如果是普通函数，调用原始函数
                tool_result = self.smart_retrieval(query_request_json, top_k=60)

            # 获取原始检索结果用于引用展示框
            try:
                raw_results = self.smart_retrieval_impl(query, top_k=10, filter_conditions=None, filter_pdf=filter_pdf)
                retrieved_docs = []
                for r in raw_results:
                    retrieved_docs.append({
                        "pdf_filename": r.pdf_filename or "未知文件",
                        "section_title": r.section_title or "未知章节",
                        "page_number": int(r.page_number) if r.page_number is not None else 0,
                        "score": float(r.score) if r.score is not None else 0,
                        "text": r.text[:200] + "..." if r.text and len(r.text) > 200 else r.text,
                        "matched": getattr(r, 'matched', True)
                    })
            except Exception as e:
                logger.warning(f"获取原始检索结果失败: {e}")
                retrieved_docs = []

            # 调试日志：记录检索结果状态
            logger.info(f"[DeepSeekAgent] retrieved_docs 数量: {len(retrieved_docs)}")
            if retrieved_docs:
                logger.info(f"[DeepSeekAgent] 第一个结果: {retrieved_docs[0].get('pdf_filename', 'N/A')} / {retrieved_docs[0].get('section_title', 'N/A')}")

            # 判断是否有检索结果
            has_retrieved = len(retrieved_docs) > 0
            has_matched = any(doc.get('matched', True) for doc in retrieved_docs) if retrieved_docs else False

            logger.info(f"[DeepSeekAgent] has_retrieved: {has_retrieved}, has_matched: {has_matched}")

            logger.info(f"[DeepSeekAgent] 进入主流程（检索回答）")

            # 如果没有检索结果，尝试基于对话历史回答
            if not has_retrieved or not has_matched:
                logger.info(f"[DeepSeekAgent] 进入历史回答分支")
                logger.info(f"[DeepSeekAgent] 无检索结果，尝试基于历史记录回答，历史长度: {len(chat_history) if chat_history else 0}")

                if chat_history and len(chat_history) > 0:
                    # 构建基于历史记录的回答提示
                    history_text = ""
                    for msg in chat_history[-10:]:  # 只使用最近10轮对话
                        role = msg.get("role", "user")
                        content = msg.get("content", "")
                        history_text += f"- {role}: {content[:200]}\n" if len(content) > 200 else f"- {role}: {content}\n"

                    # 检查最近一条助手回答是否包含具体引用信息
                    last_assistant_msg = None
                    for msg in reversed(chat_history):
                        if msg.get("role") == "assistant":
                            last_assistant_msg = msg
                            break

                    history_based_prompt = f"""基于以下对话历史记录回答用户当前问题：

【对话历史】
{history_text}

【当前问题】
{user_input}

请根据上述对话历史回答当前问题。如果历史记录中包含相关信息，请整理后回答；特别注意：如果历史记录中的assistant回答包含具体引用（如页码、章节），请直接使用这些信息进行扩展回答。

要求：
- 如果历史记录中有相关信息，采用结构化格式回答
- 优先使用历史记录中已有的具体引用和论据进行扩展
- 如果历史记录中没有相关信息，直接说明无法回答并引导用户提供更多信息
- 不要编造信息"""

                    try:
                        history_response = self.llm.invoke(history_based_prompt)
                        final_answer = history_response.content if hasattr(history_response, 'content') else str(history_response)
                        answer_source = "history"
                    except Exception as e:
                        logger.warning(f"基于历史记录回答失败: {e}")
                        final_answer = None
                        answer_source = "none"
                else:
                    final_answer = None
                    answer_source = "none"

            logger.info(f"[DeepSeekAgent] 历史分支处理完成，final_answer是否为空: {final_answer is None}, answer_source: {answer_source}")

            # 始终构建历史文本（无论检索是否成功，都参考历史对话）
            history_text = ""
            if chat_history and len(chat_history) > 0:
                for msg in chat_history[-10:]:  # 使用最近10轮对话
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    history_text += f"- {role}: {content[:200]}\n" if len(content) > 200 else f"- {role}: {content}\n"

            # 基于检索结果回答（始终考虑历史记录）
            if final_answer is None:
                # 调用DeepSeek模型基于检索结果回答用户问题
                analysis_prompt = f"""用户问题：{user_input}

检索结果：
{tool_result}

【对话历史】
{history_text if history_text else "无历史记录"}

请基于以上检索结果和对话历史回答用户的问题。如果当前检索结果不足以回答，可以结合历史对话中的相关信息进行补充。

要求：
- 如果历史记录中有相关信息，优先使用并注明"根据之前的对话"
- 采用结构化格式回答
- 在回答末尾注明主要参考来源

请直接给出结构清晰的答案。"""

                try:
                    # 使用DeepSeek模型分析并回答问题
                    logger.info(f"[DeepSeekAgent] 准备调用LLM...")
                    analysis_response = self.llm.invoke(analysis_prompt)
                    logger.info(f"[DeepSeekAgent] LLM调用完成")
                    final_answer = analysis_response.content if hasattr(analysis_response, 'content') else str(analysis_response)
                    answer_source = "retrieval"

                    logger.info(f"[DeepSeekAgent] LLM调用成功，final_answer长度={len(final_answer)}")
                except Exception as e:
                    logger.warning(f"DeepSeek模型分析失败，返回原始检索结果: {e}")
                    final_answer = tool_result
                    answer_source = "raw_results"

            # 如果 final_answer 仍然为空，返回提示信息
            if final_answer is None:
                logger.warning(f"[DeepSeekAgent] final_answer为空，返回提示信息")
                final_answer = """当前对话信息不足以回答您的问题。请结合知识库中的文件名或章节标题进行更精确的询问。

例如：
- 可以提到具体的文件名称（如"安徒生童话.pdf"）
- 可以提到具体的章节标题（如"第二章：责任与生活的抉择"）
- 可以描述您想了解的内容主题

这样可以帮助我从知识库中准确检索到相关信息。"""
                answer_source = "no_info"

            # 最终返回结果
            return {
                "success": True,
                "user_input": user_input,
                "answer": final_answer,
                "retrieved_docs": retrieved_docs,
                "chat_history": chat_history or [],
                "answer_source": answer_source
            }

        except Exception as e:
            error_msg = f"DeepSeek检索代理执行失败: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "user_input": user_input,
                "error": error_msg,
                "chat_history": chat_history or []
            }

    def simple_query(self, query: str) -> str:
        """
        简单查询接口

        Args:
            query: 用户查询

        Returns:
            检索结果的文本
        """
        try:
            params = self._extract_params(query)
            
            # 使用工具函数获取格式化结果
            import json
            query_request_json = json.dumps({"query": params["query"], "filter_pdf": params["filter_pdf"]}, ensure_ascii=False)
            tool_result = self.smart_retrieval(query_request_json, top_k=60)
            
            return tool_result
        except Exception as e:
            logger.error(f"简单查询失败: {e}")
            return f"查询失败：{str(e)}"


def create_deepseek_agent(vector_db_path: str, api_key: str = None, base_url: str = None,
                         embedding_model: str = "intfloat/multilingual-e5-large", tools_instance=None) -> DeepSeekRetrievalAgent:
    """
    创建DeepSeek检索代理实例的工厂函数

    Args:
        vector_db_path: 向量数据库路径
        api_key: DeepSeek API密钥
        base_url: DeepSeek API基础URL
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
