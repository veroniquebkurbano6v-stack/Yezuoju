"""
数据模型定义
使用Pydantic定义请求和响应的数据结构
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """聊天请求模型"""
    query: str = Field(..., min_length=1, max_length=1000, description="用户查询内容")
    conversation_id: Optional[str] = Field(None, description="对话ID，用于多会话支持")


class Reference(BaseModel):
    """引用来源模型"""
    text_preview: str = Field(..., description="文本预览（前150字符）")
    section_title: str = Field(..., description="章节标题")
    page_number: int = Field(..., description="页码")
    score: float = Field(..., description="相关性分数")
    pdf_filename: str = Field(..., description="来源文件名")


class ChatResponse(BaseModel):
    """聊天响应模型"""
    answer: str = Field(..., description="智能体回答")
    references: List[Reference] = Field(..., description="前5个引用文本块")
    conversation_id: str = Field(..., description="对话ID")
    timestamp: str = Field(..., description="响应时间戳")
    mode: str = Field(..., description="响应模式：precise或general")


class Message(BaseModel):
    """消息模型"""
    role: str = Field(..., description="角色：user或assistant")
    content: str = Field(..., description="消息内容")
    timestamp: str = Field(..., description="消息时间戳")
    references: Optional[List[Reference]] = Field(None, description="引用来源，仅assistant角色有")


class ChatHistoryResponse(BaseModel):
    """聊天历史响应模型"""
    messages: List[Message] = Field(..., description="对话历史消息列表")
    conversation_id: str = Field(..., description="对话ID")
    total_messages: int = Field(..., description="消息总数")


class ClearChatRequest(BaseModel):
    """清空聊天请求模型"""
    conversation_id: Optional[str] = Field(None, description="对话ID，用于指定清空哪个对话")


class ClearChatResponse(BaseModel):
    """清空聊天响应模型"""
    success: bool = Field(..., description="清空是否成功")
    message: str = Field(..., description="响应消息")
    conversation_id: Optional[str] = Field(None, description="清空的对话ID")
