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
from app.core.session_manager import session_manager as enterprise_session_manager

# 初始化日志
logger = logging.getLogger(__name__)

# 创建路由实例
chat_router = APIRouter()

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
        request: 包含查询内容和可选对话 ID 的请求对象
    
    Returns:
        包含智能体回答、引用来源和对话 ID 的响应对象（包含 RAG 评测所需字段）
    """
    import time
    start_time = time.time()
    
    try:
        logger.info(f"[send_query] ====== 开始处理查询 ======")
        logger.info(f"[send_query] 收到的原始 request.query: '{request.query}'")
        logger.info(f"[send_query] conversation_id: {request.conversation_id}")
        
        # 生成或使用现有的对话 ID
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        # 直接使用原始查询，让 Agent 自动从查询中提取参数
        # Agent 现在会从 "[仅在文件：xxx.pdf]" 格式中自动提取 filter_pdf
        original_query = request.query
        logger.info(f"[send_query] 使用原始查询：'{original_query}'")
        
        # 获取会话历史（使用企业级会话管理器的智能压缩上下文）
        chat_history = await enterprise_session_manager.get_context_for_agent(
            conversation_id, original_query
        )
        # 转换为字典格式供 Agent 使用
        chat_history_dicts = [{"role": msg["role"], "content": msg["content"]} for msg in chat_history]
        
        # 调用智能体服务，让 Agent 自动处理参数提取
        result = agent_service.query(
            query=original_query,
            conversation_id=conversation_id,
            chat_history=chat_history_dicts
        )
        
        # 计算延迟
        latency_ms = (time.time() - start_time) * 1000
        
        # 格式化引用来源
        references = []
        retrieved_docs = []
        citations = []
        context_chunks = []
        
        # 优先从 result 中获取 retrieved_docs（Agent 直接返回的检索结果）
        if result.get("retrieved_docs"):
            logger.info(f"[send_query] Agent 返回了 {len(result['retrieved_docs'])} 篇检索文档")
            for doc in result["retrieved_docs"]:
                doc_meta = doc.get('metadata', {})
                ref = Reference(
                    text_preview=doc.get('document', '')[:150] + "..." if len(doc.get('document', '')) > 150 else doc.get('document', ''),
                    section_title=doc_meta.get('chapter', doc_meta.get('section_title', '未知章节')),
                    page_number=doc_meta.get('start_page', doc_meta.get('page_number', 0)),
                    score=doc.get('score', 0),
                    pdf_filename=doc_meta.get('source', doc_meta.get('pdf_filename', '未知文件'))
                )
                references.append(ref)
                
                # 构建检索到的文档列表（用于评测）
                doc_text = f"[{doc_meta.get('chapter', '')}] {doc.get('document', '')}"
                retrieved_docs.append(doc_text)

                # 构建引用列表
                citations.append(doc_meta.get('chapter', ''))
                
                # 构建上下文
                context_chunks.append(doc.get('document', ''))
        elif result.get("tool_calls"):
            # 兼容旧版本：从工具调用结果中提取引用
            logger.info(f"[send_query] 从 tool_calls 中提取引用信息")
            for tool_call in result["tool_calls"]:
                if tool_call.get("output"):
                    for doc in tool_call["output"]:
                        # 构建引用对象
                        ref = Reference(
                            text_preview=doc.text[:150] + "..." if len(doc.text) > 150 else doc.text,
                            section_title=doc.section_title,
                            page_number=doc.page_number,
                            score=doc.score,
                            pdf_filename=doc.pdf_filename if hasattr(doc, "pdf_filename") else "未知文件"
                        )
                        references.append(ref)
                        
                        # 构建检索到的文档列表（用于评测）
                        doc_text = f"[{doc.section_title}] {doc.text}"
                        retrieved_docs.append(doc_text)
                        
                        # 构建引用列表
                        citations.append(doc.section_title)
                        
                        # 构建上下文
                        context_chunks.append(doc.text)
        
        # 限制最多 5 个引用
        references = references[:5]
        retrieved_docs = retrieved_docs[:10]  # 最多 10 个检索文档
        citations = citations[:5]  # 最多 5 个引用
        
        # 构建完整上下文
        context = ".".join(context_chunks)
        
        # 估算 Token 消耗（简单估算：中文字符数 / 2）
        input_tokens = len(original_query) + len(context)
        output_tokens = len(result.get("answer", ""))
        tokens = {
            "input": input_tokens // 2,
            "output": output_tokens // 2,
            "total": (input_tokens + output_tokens) // 2
        }
        
        # 生成时间戳
        timestamp = datetime.now().isoformat()
        
        # 构建响应（包含所有评测所需字段）
        response = ChatResponse(
            answer=result["answer"],
            references=references,
            conversation_id=conversation_id,
            timestamp=timestamp,
            mode="precise",  # 目前默认使用精确模式
            retrieved_docs=retrieved_docs,
            citations=citations,
            context=context,
            latency_ms=latency_ms,
            tokens=tokens,
            success=result.get("success", True)
        )
        
        # 更新对话历史（使用企业级会话管理器）
        await enterprise_session_manager.add_message(
            session_id=conversation_id,
            role="user",
            content=request.query,
            metadata={"timestamp": timestamp}
        )
        
        await enterprise_session_manager.add_message(
            session_id=conversation_id,
            role="assistant",
            content=result["answer"],
            metadata={
                "timestamp": timestamp,
                "references": [ref.dict() for ref in references]
            }
        )
        
        return response
        
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"处理查询失败：{str(e)}"
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
        包含对话历史消息的响应对象（如果对话不存在则返回空列表）
    """
    if not conversation_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="必须提供对话ID"
        )
    
    # 🔥 使用企业级会话管理器获取消息（如果会话不存在会自动创建）
    messages = await enterprise_session_manager.get_messages(conversation_id, limit=100)
    
    # 🔥 如果消息为空，返回空列表而不是 404（新对话的正常情况）
    if not messages:
        logger.info(f"📝 新对话或无历史：{conversation_id}")
        return ChatHistoryResponse(
            messages=[],
            conversation_id=conversation_id,
            total_messages=0
        )
    
    # 转换为 Message 对象
    message_objects = [
        Message(
            role=msg["role"],
            content=msg["content"],
            timestamp=msg.get("timestamp", "")
        )
        for msg in messages
    ]
    
    return ChatHistoryResponse(
        messages=message_objects,
        conversation_id=conversation_id,
        total_messages=len(message_objects)
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
    
    # 使用企业级会话管理器删除会话
    success = await enterprise_session_manager.delete_session(request.conversation_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="对话历史不存在"
        )
    
    return ClearChatResponse(
        success=True,
        message="对话历史已清空",
        conversation_id=request.conversation_id
    )

@chat_router.get("/conversations", summary="获取所有对话ID列表")
async def get_conversations():
    """
    获取所有已存在的对话ID列表（含消息计数）
    
    Returns:
        包含所有对话ID和详细信息的响应对象
    """
    sessions = await enterprise_session_manager.list_sessions()
    
    return {
        "conversation_ids": [s["session_id"] for s in sessions],
        "conversations": [
            {
                "session_id": s["session_id"],
                "message_count": s.get("message_count", 0),
                "last_accessed": s.get("last_accessed", ""),
                "created_at": s.get("created_at", ""),
            }
            for s in sessions
        ],
        "total_conversations": len(sessions)
    }


# ============================================================
# 记忆系统 API
# ============================================================

from pydantic import BaseModel, Field


class MemoryQueryRequest(BaseModel):
    query: str = Field(..., description="搜索关键词")
    layer: str = Field(default="all", description="记忆层级：all / short_term / long_term / timeline")
    limit: int = Field(default=10, ge=1, le=50, description="最大结果数")


class LongTermMemorySaveRequest(BaseModel):
    content: str = Field(..., description="记忆内容")
    category: str = Field(default="fact", description="分类：preference / fact / user_info")
    importance: int = Field(default=3, ge=1, le=5, description="重要性")
    session_id: str = Field(default="", description="关联会话ID")


class TimelineEventSaveRequest(BaseModel):
    title: str = Field(..., description="事件标题")
    description: str = Field(default="", description="事件描述")
    event_type: str = Field(default="user_action", description="事件类型")
    importance: int = Field(default=2, ge=1, le=5, description="重要性")
    session_id: str = Field(default="", description="关联会话ID")


class RoleSwitchRequest(BaseModel):
    role_id: str = Field(default="humorous_butler", description="角色标识符")


class MemoryDeleteRequest(BaseModel):
    memory_id: str = Field(..., description="记忆ID")
    layer: str = Field(default="long_term", description="记忆层级")


@chat_router.post("/memory/query", summary="搜索记忆")
async def query_memory(request: MemoryQueryRequest):
    """
    跨所有记忆层级搜索

    Args:
        request: 包含查询关键词和层级的请求对象
    """
    try:
        from src.core.memory_manager import get_memory_manager
        mm = get_memory_manager()

        if request.layer == "short_term":
            results = await mm.short_term.search(request.query, limit=request.limit)
            return {"results": [e.to_dict() for e in results], "layer": "short_term", "total": len(results)}
        elif request.layer == "long_term":
            results = await mm.long_term.search(request.query, limit=request.limit)
            return {"results": [e.to_dict() for e in results], "layer": "long_term", "total": len(results)}
        elif request.layer == "timeline":
            results = await mm.timeline.search(request.query, limit=request.limit)
            return {"results": [e.to_dict() for e in results], "layer": "timeline", "total": len(results)}
        else:
            all_results = await mm.search_all(request.query)
            return all_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"记忆查询失败：{str(e)}")


@chat_router.post("/memory/long-term", summary="保存长期记忆")
async def save_long_term_memory(request: LongTermMemorySaveRequest):
    """
    手动保存一条长期记忆

    Args:
        request: 包含记忆内容的请求对象
    """
    try:
        from src.core.memory_manager import get_memory_manager
        mm = get_memory_manager()
        entry = await mm.long_term.add(
            content=request.content,
            category=request.category,
            importance=request.importance,
            session_id=request.session_id,
        )
        return {"success": True, "memory": entry.to_dict()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"保存记忆失败：{str(e)}")


@chat_router.post("/memory/event", summary="保存时间线事件")
async def save_timeline_event(request: TimelineEventSaveRequest):
    """
    手动保存一个时间线事件

    Args:
        request: 包含事件信息的请求对象
    """
    try:
        from src.core.memory_manager import get_memory_manager
        mm = get_memory_manager()
        event = await mm.timeline.add_event(
            title=request.title,
            description=request.description,
            event_type=request.event_type,
            importance=request.importance,
            session_id=request.session_id,
        )
        return {"success": True, "event": event.to_dict()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"保存事件失败：{str(e)}")


@chat_router.get("/memory/timeline", summary="获取时间线")
async def get_timeline(days: int = 30, limit: int = 20):
    """
    获取最近 N 天的时间线事件

    Args:
        days: 最近多少天
        limit: 最大事件数
    """
    try:
        from src.core.memory_manager import get_memory_manager
        mm = get_memory_manager()
        events = await mm.timeline.get_recent(days=days, limit=limit)
        return {"events": [e.to_dict() for e in events], "total": len(events)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取时间线失败：{str(e)}")


@chat_router.delete("/memory/{memory_id}", summary="删除记忆")
async def delete_memory(memory_id: str, layer: str = "long_term"):
    """
    删除指定记忆条目

    Args:
        memory_id: 记忆ID
        layer: 记忆层级
    """
    try:
        from src.core.memory_manager import get_memory_manager
        mm = get_memory_manager()
        if layer == "long_term":
            await mm.long_term.delete(memory_id)
        else:
            raise HTTPException(status_code=400, detail=f"不支持删除 {layer} 层级的记忆")
        return {"success": True, "deleted_id": memory_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除记忆失败：{str(e)}")


@chat_router.get("/memory/stats", summary="获取记忆统计")
async def get_memory_stats():
    """获取各层级记忆数量统计（含长期记忆详情和压缩状态）"""
    try:
        from src.core.memory_manager import get_memory_manager
        mm = get_memory_manager()
        lt_items = [e.to_dict() for e in mm.long_term._entries.values()]

        compression_status = {}
        try:
            from app.core.session_manager import session_manager
            async with session_manager._cache_lock:
                keys = list(session_manager._memory_cache.keys())
            if keys:
                session_data = await session_manager.get_session(keys[0])
                if session_data:
                    compression_status = {
                        "compressed": session_data.get("_compressed_at") is not None,
                        "compressed_at": session_data.get("_compressed_at"),
                        "message_count": session_data.get("message_count", 0),
                    }
        except Exception:
            pass

        return {
            "short_term": {"message_count": mm.short_term.size},
            "long_term": {"total": mm.long_term.size, "items": lt_items},
            "timeline": {"total": mm.timeline.size},
            "compression": compression_status,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取统计失败：{str(e)}")


@chat_router.get("/session/{session_id}/compression-status", summary="查询会话压缩状态")
async def get_session_compression_status(session_id: str):
    """查询指定会话的压缩状态和消息统计"""
    try:
        from app.core.session_manager import session_manager
        session_data = await session_manager.get_session(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="会话不存在")

        messages = session_data.get("messages", [])
        compressed_msgs = [m for m in messages if m.get("metadata", {}).get("compressed")]
        normal_msgs = [m for m in messages if not m.get("metadata", {}).get("compressed")]

        return {
            "session_id": session_id,
            "total_messages_stored": len(messages),
            "normal_messages": len(normal_msgs),
            "compressed_summary_messages": len(compressed_msgs),
            "original_messages_in_summary": sum(
                m.get("metadata", {}).get("original_count", 0) for m in compressed_msgs
            ),
            "compressed_at": session_data.get("_compressed_at"),
            "compression_threshold": session_manager.compression_threshold,
            "compression_keep_recent": session_manager.compression_keep_recent,
            "compression_enabled": session_manager.compression_enabled,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询压缩状态失败：{str(e)}")


# ============================================================
# 角色系统 API
# ============================================================

@chat_router.get("/roles", summary="获取可用角色列表")
async def get_roles():
    """获取所有可用角色"""
    try:
        from src.core.role_profile import list_available_roles
        return {"roles": list_available_roles()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取角色列表失败：{str(e)}")


@chat_router.post("/role/switch", summary="切换角色")
async def switch_role(request: RoleSwitchRequest):
    """
    切换 Agent 的角色

    Args:
        request: 包含目标角色ID的请求对象
    """
    try:
        from src.core.role_manager import get_role_manager
        rm = get_role_manager()
        rm.switch_role(request.role_id)
        return {
            "success": True,
            "current_role": rm.profile.display_name,
            "role_id": rm.role_id,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"切换角色失败：{str(e)}")


@chat_router.get("/role/current", summary="获取当前角色信息")
async def get_current_role():
    """获取当前 Agent 的角色配置"""
    try:
        from src.core.role_manager import get_role_manager
        rm = get_role_manager()
        return {
            "role_id": rm.role_id,
            "profile": rm.profile.to_dict(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取角色信息失败：{str(e)}")
