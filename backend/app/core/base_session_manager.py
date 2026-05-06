"""
会话管理器抽象基类 - StoryRag v2.0
定义统一接口，便于切换不同实现（本地文件/Redis/其他存储）
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Any


class BaseSessionManager(ABC):
    """
    会话管理器抽象基类
    
    所有会话管理器实现都必须继承此类并实现所有抽象方法。
    """
    
    @abstractmethod
    async def create_session(
        self, 
        session_id: str, 
        user_id: Optional[str] = None, 
        metadata: Optional[Dict] = None
    ) -> str:
        """创建新会话"""
        pass
    
    @abstractmethod
    async def add_message(
        self, 
        session_id: str, 
        role: str, 
        content: str, 
        metadata: Optional[Dict] = None
    ) -> bool:
        """添加消息到会话"""
        pass
    
    @abstractmethod
    async def get_session(self, session_id: str) -> Optional[Dict]:
        """获取完整会话数据"""
        pass
    
    @abstractmethod
    async def get_messages(self, session_id: str, limit: int = 50) -> List[Dict]:
        """获取最近 N 条消息"""
        pass
    
    @abstractmethod
    async def get_context_for_agent(self, session_id: str, current_query: str) -> List[Dict]:
        """为 Agent 构建优化的上下文"""
        pass
    
    @abstractmethod
    async def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        pass
    
    @abstractmethod
    async def list_sessions(self, user_id: Optional[str] = None) -> List[Dict]:
        """列出所有会话（可按用户过滤）"""
        pass
    
    @abstractmethod
    async def cleanup_expired_sessions(self) -> int:
        """清理过期会话"""
        pass
