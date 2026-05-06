"""
RAG 系统核心数据模型

职责：
1. 定义不可变数据结构
2. 提供实用方法
3. 确保类型安全
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class Document:
    """
    文档片段
    
    代表从 PDF 中提取的一个文本块及其元数据
    """
    content: str                                    # 文本内容
    metadata: dict[str, Any] = field(default_factory=dict)  # 元数据（页码、章节等）
    score: float = 0.0                              # 相关性分数
    doc_id: str = ""                                # 文档唯一标识
    
    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            "content": self.content,
            "metadata": self.metadata,
            "score": self.score,
            "doc_id": self.doc_id
        }


@dataclass(frozen=True)
class Query:
    """
    用户查询
    
    不可变设计：防止查询参数被意外修改
    """
    text: str                                       # 查询文本
    top_k: int = 10                                 # 返回结果数量
    filters: dict[str, Any] = field(default_factory=dict)  # 过滤条件
    strategy: str = "auto"                          # 检索策略：auto/vector/keyword/chapter


@dataclass(frozen=True)
class RetrievalResult:
    """
    检索结果
    
    包含查询和相关文档列表
    """
    query: str                                      # 原始查询
    documents: list[Document] = field(default_factory=list)  # 相关文档
    total_found: int = 0                            # 找到的总数
    strategy_used: str = ""                         # 使用的检索策略
    
    def summary(self) -> str:
        """生成可读的摘要"""
        return f"Found {self.total_found} documents using {self.strategy_used}"


@dataclass(frozen=True)
class ChatMessage:
    """
    聊天消息
    
    用于会话管理
    """
    role: str                                       # "user" 或 "assistant"
    content: str                                    # 消息内容
    metadata: dict[str, Any] = field(default_factory=dict)  # 额外元数据
    timestamp: str = ""                             # 时间戳
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "role": self.role,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }


@dataclass(frozen=True)
class RAGResponse:
    """
    RAG 响应
    
    包含答案、来源和置信度
    """
    answer: str                                     # AI 生成的答案
    sources: list[Document] = field(default_factory=list)  # 引用来源
    confidence: float = 0.0                         # 置信度 (0-1)
    session_id: str = ""                            # 会话 ID
    usage_tokens: dict[str, int] = field(default_factory=lambda: {"input": 0, "output": 0})
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "answer": self.answer,
            "sources": [doc.to_dict() for doc in self.sources],
            "confidence": self.confidence,
            "session_id": self.session_id,
            "usage_tokens": self.usage_tokens
        }


@dataclass(frozen=True)
class SessionMetadata:
    """
    会话元数据
    
    用于会话管理和追踪
    """
    session_id: str                                 # 会话唯一标识
    user_id: str                                    # 用户 ID
    created_at: str                                 # 创建时间
    last_accessed: str                              # 最后访问时间
    message_count: int = 0                          # 消息数量
    title: str = ""                                 # 会话标题
    metadata: dict[str, Any] = field(default_factory=dict)  # 额外元数据


@dataclass
class UsageSummary:
    """
    Token 用量统计
    
    可变设计：需要累加统计
    """
    input_tokens: int = 0
    output_tokens: int = 0
    
    def add_turn(self, prompt: str, output: str) -> 'UsageSummary':
        """
        添加一轮对话的用量
        
        Args:
            prompt: 用户输入
            output: 系统输出
        
        Returns:
            新的 UsageSummary 对象（原对象不变）
        """
        # 简化估算：按空格分词
        return UsageSummary(
            input_tokens=self.input_tokens + len(prompt.split()),
            output_tokens=self.output_tokens + len(output.split()),
        )
    
    def total_tokens(self) -> int:
        """总 token 数"""
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class ToolExecution:
    """
    工具执行结果
    
    用于追踪工具调用
    """
    tool_name: str                                  # 工具名称
    success: bool                                   # 是否成功
    result: Any = None                              # 执行结果
    error: str = ""                                 # 错误信息
    execution_time_ms: float = 0.0                  # 执行时间（毫秒）


@dataclass(frozen=True)
class ChapterInfo:
    """
    章节信息
    
    用于章节级检索
    """
    chapter_title: str                              # 章节标题
    page_start: int                                 # 起始页码
    page_end: int                                   # 结束页码
    subtitle: str = ""                              # 副标题（可选）
    content_preview: str = ""                       # 内容预览
