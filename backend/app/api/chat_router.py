"""
聊天API路由
处理与聊天相关的所有HTTP请求
"""

import os
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse

from app.core.models import (
    ChatRequest, ChatResponse, ChatHistoryResponse, 
    Message, Reference, ClearChatRequest, ClearChatResponse
)
from app.agents.agent_service import DeepSeekAgentService

# 初始化日志
logger = logging.getLogger(__name__)

# 创建路由实例
chat_router = APIRouter()

# 对话历史存储（内存存储，生产环境建议使用数据库）
chat_histories: Dict[str, List[Message]] = {}

# 智能体服务实例（单例模式）
agent_service = None

def get_agent_service() -> DeepSeekAgentService:
    """获取智能体服务实例"""
    global agent_service
    if agent_service is None:
        try:
            from app.core.config import settings
            agent_service = DeepSeekAgentService(
                vector_db_path=settings.VECTOR_DB_PATH,
                api_key=settings.DEEPSEEK_API_KEY,
                base_url=settings.DEEPSEEK_BASE_URL,
                embedding_model=settings.EMBEDDING_MODEL
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"智能体服务初始化失败: {str(e)}"
            )
    return agent_service

@chat_router.get("/pdfs", response_model=List[Dict], summary="获取PDF文件列表，按语言分组")
async def get_pdfs():
    """
    获取src/data/source目录下的PDF文件列表，按语言分组
    
    Returns:
        按语言分组的PDF文件列表
    """
    try:
        # 计算source目录路径
        current_file = os.path.abspath(__file__)
        api_dir = os.path.dirname(current_file)
        app_dir = os.path.dirname(api_dir)
        backend_dir = os.path.dirname(app_dir)
        project_root = os.path.dirname(backend_dir)
        source_dir = os.path.join(project_root, "src", "data", "source")
        
        if not os.path.exists(source_dir):
            return []
        
        language_groups = {}
        
        # 遍历语言子目录
        for lang_dir in os.listdir(source_dir):
            lang_path = os.path.join(source_dir, lang_dir)
            if not os.path.isdir(lang_path):
                continue
            
            # 获取该语言目录下的所有PDF文件
            pdf_files = [f for f in os.listdir(lang_path) if f.lower().endswith('.pdf')]
            
            if pdf_files:
                # 使用目录名作为语言标识
                language_groups[lang_dir] = pdf_files
        
        result = [{"language": lang, "files": files} for lang, files in language_groups.items()]
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取PDF文件列表失败: {str(e)}"
        )

@chat_router.post("/query", response_model=ChatResponse, summary="发送查询，获取智能体回复")
async def send_query(
    request: ChatRequest,
    agent_service: DeepSeekAgentService = Depends(get_agent_service) # 从依赖注入获取智能体服务实例
):
    """
    发送用户查询，获取智能体的回复
    
    Args:
        request: 包含查询内容和可选对话ID的请求对象
    
    Returns:
        包含智能体回答、引用来源和对话ID的响应对象
    """
    try:
        logger.info(f"[send_query] ====== 开始处理查询 ======")
        logger.info(f"[send_query] 收到的原始 request.query: '{request.query}'")
        logger.info(f"[send_query] conversation_id: {request.conversation_id}")
        
        # 生成或使用现有的对话ID
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        # 获取当前对话历史
        chat_history = chat_histories.get(conversation_id, [])
        
        # 直接使用原始查询，让 Agent 自动从查询中提取参数
        # Agent 现在会从 "[仅在文件：xxx.pdf]" 格式中自动提取 filter_pdf
        original_query = request.query
        logger.info(f"[send_query] 使用原始查询: '{original_query}'")
        
        # 调用智能体服务，让 Agent 自动处理参数提取
        result = agent_service.query(
            query=original_query,
            conversation_id=conversation_id,
            chat_history=chat_history
        )
        
        # 格式化引用来源
        references = []
        if result.get("tool_calls"):
            # 从工具调用结果中提取引用
            for tool_call in result["tool_calls"]:
                if tool_call.get("output"):
                    for doc in tool_call["output"]:
                        references.append(Reference(
                            text_preview=doc.text[:150] + "..." if len(doc.text) > 150 else doc.text,
                            section_title=doc.section_title,
                            page_number=doc.page_number,
                            score=doc.score,
                            pdf_filename=doc.pdf_filename if hasattr(doc, "pdf_filename") else "未知文件"
                        ))
        
        # 限制最多5个引用
        references = references[:5]
        
        # 生成时间戳
        timestamp = datetime.now().isoformat()
        
        # 构建响应
        response = ChatResponse(
            answer=result["answer"],
            references=references,
            conversation_id=conversation_id,
            timestamp=timestamp,
            mode="precise"  # 目前默认使用精确模式
        )
        
        # 更新对话历史
        if conversation_id not in chat_histories:
            chat_histories[conversation_id] = []
        
        # 添加用户消息
        chat_histories[conversation_id].append(Message(
            role="user",
            content=request.query,
            timestamp=timestamp
        ))
        
        # 添加助手消息
        chat_histories[conversation_id].append(Message(
            role="assistant",
            content=result["answer"],
            timestamp=timestamp,
            references=references
        ))
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"处理查询失败: {str(e)}"
        )

@chat_router.get("/history", response_model=ChatHistoryResponse, summary="获取对话历史")
async def get_history(
    conversation_id: Optional[str] = None
):
    """
    获取指定对话ID的对话历史
    
    Args:
        conversation_id: 对话ID，可选参数
    
    Returns:
        包含对话历史消息的响应对象
    """
    if not conversation_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="必须提供对话ID"
        )
    
    if conversation_id not in chat_histories:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="对话历史不存在"
        )
    
    return ChatHistoryResponse(
        messages=chat_histories[conversation_id],
        conversation_id=conversation_id,
        total_messages=len(chat_histories[conversation_id])
    )

@chat_router.post("/clear", response_model=ClearChatResponse, summary="清空对话历史")
async def clear_chat(
    request: ClearChatRequest
):
    """
    清空指定对话ID的对话历史
    
    Args:
        request: 包含可选对话ID的请求对象
    
    Returns:
        清空操作的结果响应
    """
    if not request.conversation_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="必须提供对话ID"
        )
    
    if request.conversation_id not in chat_histories:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="对话历史不存在"
        )
    
    # 清空对话历史
    del chat_histories[request.conversation_id]
    
    return ClearChatResponse(
        success=True,
        message="对话历史已清空",
        conversation_id=request.conversation_id
    )

@chat_router.get("/conversations", summary="获取所有对话ID列表")
async def get_conversations():
    """
    获取所有已存在的对话ID列表
    
    Returns:
        包含所有对话ID的响应对象
    """
    return {
        "conversation_ids": list(chat_histories.keys()),
        "total_conversations": len(chat_histories)
    }
