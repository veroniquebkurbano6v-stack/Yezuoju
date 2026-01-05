"""
对话管理API路由模块
提供对话创建、管理、历史记录和AI聊天功能
"""
# FastAPI核心组件导入
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel  # 数据验证模型
from typing import Any, Dict, List  # 类型提示
import json  # JSON数据处理
import logging  # 日志记录

# 项目内部模块导入
from app.core.checkpointer_manager import dialog_manager, get_dialog_memory_dependency  # 对话管理器
from deepseek_agent import create_deepseek_agent  # DeepSeek AI代理创建函数

# 配置日志记录器
logger = logging.getLogger(__name__)

# 创建对话管理路由器
# prefix: 所有路由添加/api/dialogs前缀
# tags: 在API文档中分组显示，便于管理
router = APIRouter(prefix="/api/dialogs", tags=["dialogs"])  # 对话路由


# 请求数据模型定义
class ChatRequest(BaseModel):
    """聊天请求数据模型"""
    question: str  # 用户问题内容
    pdf_filename: str = None  # 可选的PDF文件过滤参数，用于指定检索特定文档


class TitleUpdateRequest(BaseModel):
    """对话标题更新请求数据模型"""
    title: str  # 新的对话标题


# 路由定义部分

@router.get("/")
def list_dialogs():
    """
    获取所有对话列表及其元信息
    返回系统中所有对话的基本信息，包括对话ID和详细元数据
    """
    dialogs = dialog_manager.list_dialogs()  # 获取所有对话ID列表
    dialog_info = []
    # 遍历每个对话ID，获取详细的元信息
    for dialog_id in dialogs:
        info = dialog_manager.get_dialog_info(dialog_id)
        dialog_info.append(info)
    return {"dialogs": dialogs, "dialog_info": dialog_info}


@router.post("/", status_code=201)
def create_dialog():
    """
    创建新对话
    在系统中创建一个新的对话会话，返回唯一标识符
    """
    new_id = dialog_manager.create_dialog()  # 创建新对话，获取对话ID
    return {"dialog_id": new_id, "message": "对话创建成功"}


@router.delete("/{dialog_id}")
def delete_dialog(dialog_id: str):
    """
    删除指定对话
    根据对话ID删除整个对话及其所有历史记录
    """
    try:
        dialog_manager.delete_dialog(dialog_id)  # 删除指定对话
        return {"message": f"对话 {dialog_id} 已删除", "success": True}
    except ValueError as e:
        # 如果对话ID不存在，抛出400错误
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{dialog_id}/history")
def get_dialog_history(dialog_id: str):
    """
    获取对话历史记录
    根据对话ID获取完整的对话历史，包括用户消息和AI回复
    """
    memory = dialog_manager.get_memory(dialog_id)  # 获取对话的记忆存储
    
    if memory is None:
        # 如果对话不存在，返回404错误
        raise HTTPException(status_code=404, detail="对话不存在")
    
    chat_history = []
    # 新的 Checkpointer 实现返回统一的字典格式
    if isinstance(memory, dict) and "messages" in memory:
        chat_history = memory.get("messages", [])
    else:
        chat_history = []
    
    return {"dialog_id": dialog_id, "history": chat_history}


@router.put("/{dialog_id}/title")
def update_dialog_title(dialog_id: str, request: TitleUpdateRequest):
    """
    更新对话标题
    为指定对话设置新的标题，便于用户识别和管理
    """
    try:
        dialog_manager.update_dialog_title(dialog_id, request.title)  # 更新对话标题
        return {"message": "标题更新成功", "success": True}
    except ValueError as e:
        # 如果对话ID不存在，返回404错误
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{dialog_id}/chat")
def chat(
    request: ChatRequest,
    dialog_id: str,
    memory = Depends(get_dialog_memory_dependency),
    fastapi_request: Request = None,
):
    """
    核心聊天接口：处理用户对话请求
    将请求转发给 DeepSeekRetrievalAgent（若可用），并把对话记忆持久化到 Checkpointer
    支持PDF文件过滤和多种回退机制
    """
    # 获取FastAPI应用实例，用于访问全局状态
    app = fastapi_request.app if fastapi_request is not None else None

    # 获取已初始化的 agent 和 vector system（在 backend.app.main 的 startup 中创建）
    agent = app.state.agent if app and hasattr(app.state, "agent") else None
    vector_system = app.state.vector_system if app and hasattr(app.state, "vector_system") else None

    # 如果 agent 不存在但环境中有 API KEY，则尝试临时创建一个
    # 这提供了在主启动失败时的备用初始化机制
    if agent is None:
        import os
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if api_key:
            try:
                agent = create_deepseek_agent(
                    os.getenv("VECTOR_DB_PATH", "src/data/vector_database"), 
                    api_key=api_key, 
                    base_url=os.getenv("DEEPSEEK_BASE_URL")
                )
                logger.info("[dialogs.chat] 临时创建 DeepSeek agent 成功")
            except Exception as e:
                logger.warning(f"[dialogs.chat] 临时创建 agent 失败: {e}")
                agent = None

    # 构建 chat_history 列表供 agent 使用（兼容不同 memory 实现）
    chat_history: List[Dict[str, str]] = []
    
    # 兼容两种记忆存储格式：字典格式和LangChain格式
    if isinstance(memory, dict) and "messages" in memory:
        # 字典格式：直接使用messages列表
        chat_history = memory.get("messages", [])
        logger.debug("使用字典格式的记忆存储")
    else:
        # 其他格式的记忆存储，chat_history 保持为空列表
        logger.debug("使用其他格式的记忆存储")

    # 根据可用组件选择处理策略
    if agent is not None:
        # 策略1：使用 DeepSeek Agent 进行智能对话
        try:
            query_text = request.question
            filter_pdf = request.pdf_filename  # 从前端请求获取PDF过滤参数

            # 记录查询信息，用于调试和监控
            if filter_pdf:
                logger.info(f"[dialogs.chat] 带PDF过滤查询: '{query_text}', 文件: {filter_pdf}")
            else:
                logger.info(f"[dialogs.chat] 简单查询: '{query_text}'")

            # 直接传递原始查询，让 Agent 内部处理参数提取和工具调用
            # Agent 会根据查询内容决定是否使用文档检索工具
            result = agent.chat(query_text, chat_history=chat_history)
            logger.info("[dialogs.chat] Agent 处理完成")
            
        except Exception as e:
            # Agent 调用失败，返回500错误
            logger.error(f"[dialogs.chat] Agent 调用失败: {e}")
            raise HTTPException(status_code=500, detail=f"Agent 调用失败: {e}")
    else:
        # 策略2：回退到简单的向量检索（当 Agent 不可用时）
        logger.info("[dialogs.chat] 使用向量检索回退策略")
        docs = []
        if vector_system is not None:
            try:
                # 执行向量搜索，返回最相关的文档片段
                results = vector_system.search(request.question, top_k=4, use_reranker=False)
                for r in results:
                    docs.append({
                        "text": r.text, 
                        "section_title": r.section_title, 
                        "page_number": r.page_number
                    })
                logger.info(f"[dialogs.chat] 检索到 {len(docs)} 个相关文档片段")
            except Exception as e:
                logger.error(f"[dialogs.chat] 向量检索失败: {e}")
                raise HTTPException(status_code=500, detail=f"检索失败: {e}")
        else:
            logger.warning("[dialogs.chat] 向量检索系统也不可用")
        
        # 构建回退响应
        result = {
            "success": True,
            "user_input": request.question,
            "answer": "（DeepSeek agent 未初始化，返回检索摘要）",
            "tool_calls": [],
            "retrieved_docs": docs
        }

    # 将对话保存回 memory（兼容不同的记忆存储实现）
    try:
        # 策略1：如果 memory 支持 save_context（LangChain记忆），使用它保存交互
        if hasattr(memory, "save_context"):
            memory.save_context({"input": request.question}, {"output": result.get("answer", "")})
            logger.debug("使用 LangChain save_context 保存对话")
        # 策略2：回退到字典格式的 append 操作
        elif isinstance(memory, dict) and "messages" in memory:
            dialog_manager.save_context(dialog_id, request.question, result.get("answer", ""))
            logger.debug("使用 Checkpointer save_context 保存对话")
    except Exception as e:
        # 记忆保存失败不应该阻塞响应，仅记录日志
        logger.warning(f"[dialogs.chat] 对话记忆保存失败: {e}")
    
    # 更新对话元数据（消息计数等）
    try:
        dialog_manager.increment_message_count(dialog_id)
        logger.debug(f"[dialogs.chat] 更新对话 {dialog_id} 的消息计数")
    except Exception as e:
        logger.warning(f"[dialogs.chat] 更新对话元数据失败: {e}")
    
    # 返回响应，包含对话ID和AI回复结果
    return {"dialog_id": dialog_id, **result}


