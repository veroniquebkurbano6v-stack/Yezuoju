"""
企业级会话管理器 - StoryRag v2.0
提供多层存储、智能上下文管理、TTL 过期策略

优化特性：
1. 统一接口抽象（继承 BaseSessionManager）
2. LRU 内存缓存限制
3. 并发安全保护（asyncio.Lock）
4. 准确的 Token 计数（使用 tiktoken）
5. 配置中心化管理
"""

import os
import json
import asyncio
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
from pathlib import Path
from collections import OrderedDict
import aiofiles
import aiofiles.os

# 导入依赖
from .base_session_manager import BaseSessionManager
from .config import session_config
from .token_estimator import TokenEstimator

try:
    from src.utils.logger import get_logger
except ImportError:
    import logging
    def get_logger(name):
        return logging.getLogger(name)

logger = get_logger(__name__)


class AsyncRLock:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._owner = None
        self._count = 0

    async def __aenter__(self):
        current = asyncio.current_task()
        if self._owner == current:
            self._count += 1
            return self
        await self._lock.acquire()
        self._owner = current
        self._count = 1
        return self

    async def __aexit__(self, *args):
        current = asyncio.current_task()
        if self._owner != current:
            raise RuntimeError("Cannot release un-acquired lock")
        self._count -= 1
        if self._count == 0:
            self._owner = None
            self._lock.release()


class EnterpriseSessionManager(BaseSessionManager):
    """
    企业级会话管理器（优化版）
    
    特性：
    1. 多层存储架构（LRU 内存缓存 + 磁盘持久化）
    2. 智能消息截断（基于准确的 token 计数）
    3. TTL 自动过期
    4. 异步 IO 操作
    5. 并发安全保护
    6. Agent 友好的上下文提取
    """
    
    def __init__(
        self, 
        storage_path: str = None,
        max_messages_per_session: int = None,
        max_tokens_for_agent: int = None,
        ttl_days: int = None,
        max_cache_size: int = None
    ):
        # 使用配置中心的值，如果没有传入参数
        self.storage_path = Path(storage_path or session_config.session_data_path).resolve()
        self.max_messages = max_messages_per_session or session_config.max_messages_per_session
        self.max_tokens = max_tokens_for_agent or session_config.max_tokens_for_agent
        self.ttl = timedelta(days=ttl_days or session_config.ttl_days)
        self.max_cache_size = max_cache_size or session_config.max_cache_size
        
        self.compression_enabled = os.getenv("SESSION_COMPRESSION_ENABLED", "true").lower() == "true"
        self.compression_threshold = int(os.getenv("SESSION_COMPRESSION_THRESHOLD", "20"))
        self.compression_keep_recent = int(os.getenv("SESSION_COMPRESSION_KEEP_RECENT", "8"))
        
        # LRU 内存缓存层（使用 OrderedDict 实现）
        self._memory_cache: OrderedDict[str, Dict] = OrderedDict()
        self._metadata_cache: OrderedDict[str, Dict] = OrderedDict()
        
        # 并发安全锁
        self._cache_lock = AsyncRLock()
        
        # Token 估算器
        self._token_estimator = TokenEstimator()
        
        # 初始化存储目录
        self._init_storage()
        
        logger.info(f"✅ EnterpriseSessionManager 初始化完成")
        logger.info(f"   存储路径：{self.storage_path}")
        logger.info(f"   最大消息数：{self.max_messages}")
        logger.info(f"   Agent 最大 Token: {self.max_tokens}")
        logger.info(f"   TTL: {self.ttl.days} 天")
        logger.info(f"   最大缓存大小: {self.max_cache_size}")
    
    def _init_storage(self):
        """初始化存储目录"""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        (self.storage_path / "sessions").mkdir(exist_ok=True)
        (self.storage_path / "metadata").mkdir(exist_ok=True)
        (self.storage_path / "archive").mkdir(exist_ok=True)
        self._restore_sessions_from_disk()

    def _restore_sessions_from_disk(self):
        metadata_dir = self.storage_path / "metadata"
        if not metadata_dir.exists():
            return
        restored = 0
        for meta_file in metadata_dir.glob("*.json"):
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                session_id = metadata.get("session_id")
                if session_id:
                    self._lru_cache_set(session_id, metadata, self._metadata_cache)
                    restored += 1
            except Exception:
                pass
        if restored > 0:
            logger.info(f"📂 从磁盘恢复了 {restored} 个会话元数据")
    
    def _lru_cache_get(self, key: str, cache: OrderedDict) -> Optional[Dict]:
        """从 LRU 缓存获取数据并更新访问顺序"""
        if key not in cache:
            return None
        
        # 移动到末尾表示最近访问
        cache.move_to_end(key)
        return cache[key]
    
    def _lru_cache_set(self, key: str, value: Dict, cache: OrderedDict):
        """设置 LRU 缓存，超出限制时删除最久未访问的"""
        if key in cache:
            cache.move_to_end(key)
        else:
            # 如果缓存已满，删除最久未访问的（第一个）
            if len(cache) >= self.max_cache_size:
                cache.popitem(last=False)
        
        cache[key] = value
    
    async def create_session(self, session_id: str, user_id: Optional[str] = None, metadata: Optional[Dict] = None) -> str:
        """创建新会话"""
        now = datetime.now()
        
        session_data = {
            "session_id": session_id,
            "user_id": user_id,
            "metadata": metadata or {},
            "created_at": now.isoformat(),
            "last_accessed": now.isoformat(),
            "message_count": 0,
            "messages": []
        }
        
        # 写入内存缓存（线程安全）
        async with self._cache_lock:
            self._lru_cache_set(session_id, session_data, self._memory_cache)
            self._lru_cache_set(session_id, {
                "session_id": session_id,
                "user_id": user_id,
                "created_at": now.isoformat(),
                "last_accessed": now.isoformat(),
                "message_count": 0
            }, self._metadata_cache)
        
        # 异步持久化到磁盘
        await self._persist_session(session_id)
        
        logger.info(f"📝 创建会话：{session_id}")
        return session_id
    
    async def add_message(
        self, 
        session_id: str, 
        role: str, 
        content: str, 
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        添加消息到会话（如果会话不存在则自动创建）
        """
        # 检查会话是否存在，不存在则创建
        async with self._cache_lock:
            if session_id not in self._memory_cache:
                # 尝试从磁盘加载
                loaded = await self._load_session(session_id)
                if not loaded:
                    # 会话不存在，自动创建
                    logger.info(f"📝 会话不存在，自动创建：{session_id}")
                    await self.create_session(session_id)
        
        session = self._memory_cache[session_id]
        
        # 构建消息对象（使用准确的 Token 计数）
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
            "token_count": self._estimate_tokens(content)
        }
        
        # 添加到消息列表
        session["messages"].append(message)
        session["message_count"] += 1
        session["last_accessed"] = datetime.now().isoformat()
        
        # 限制消息数量（滑动窗口）
        if len(session["messages"]) > self.max_messages:
            session["messages"] = session["messages"][-self.max_messages:]
        
        # 自动压缩检测：超过阈值时触发历史压缩
        if self.compression_enabled and len(session["messages"]) >= self.compression_threshold:
            compressed_count = await self._auto_compress_history(session_id)
            if compressed_count > 0:
                logger.info(f"📦 会话 {session_id} 自动压缩完成，压缩 {compressed_count} 条旧消息")
        
        # 更新元数据缓存
        async with self._cache_lock:
            self._metadata_cache[session_id]["message_count"] = session["message_count"]
            self._metadata_cache[session_id]["last_accessed"] = session["last_accessed"]
        
        # 异步持久化
        asyncio.create_task(self._persist_session(session_id))
        
        return True
    
    async def get_session(self, session_id: str) -> Optional[Dict]:
        """获取完整会话数据"""
        async with self._cache_lock:
            session = self._lru_cache_get(session_id, self._memory_cache)
        
        if session:
            session["last_accessed"] = datetime.now().isoformat()
            return session
        
        # 从磁盘加载
        return await self._load_session(session_id)
    
    async def get_messages(self, session_id: str, limit: int = 50) -> List[Dict]:
        """获取最近 N 条消息"""
        session = await self.get_session(session_id)
        if not session:
            return []
        
        messages = session.get("messages", [])
        return messages[-limit:]
    
    async def get_context_for_agent(
        self, 
        session_id: str, 
        current_query: str
    ) -> List[Dict]:
        """
        为 Agent 构建优化的上下文
        
        策略：
        1. 始终保留最近 3 轮对话
        2. 根据 token 限制动态调整
        3. 优先保留有工具调用的消息
        4. 压缩超长消息
        """
        session = await self.get_session(session_id)
        if not session:
            return [{"role": "user", "content": current_query}]
        
        all_messages = session.get("messages", [])
        context_messages = []
        current_tokens = 0
        
        # 逆向遍历（从新到旧）
        for i, msg in enumerate(reversed(all_messages)):
            msg_tokens = msg.get("token_count", self._estimate_tokens(msg["content"]))
            
            # 始终保留最近 3 条
            if i < 3:
                context_messages.insert(0, msg)
                current_tokens += msg_tokens
                continue
            
            # 检查 token 限制
            if current_tokens + msg_tokens > self.max_tokens:
                break
            
            # 优先保留有工具调用的消息
            if msg.get("metadata", {}).get("tool_calls") or msg.get("metadata", {}).get("references"):
                context_messages.insert(0, msg)
                current_tokens += msg_tokens
            else:
                # 普通消息：压缩处理
                compressed_msg = self._compress_message(msg, self.max_tokens - current_tokens)
                if compressed_msg:
                    context_messages.insert(0, compressed_msg)
                    current_tokens += compressed_msg.get("token_count", msg_tokens)
        
        # 添加当前查询
        context_messages.append({"role": "user", "content": current_query})
        
        logger.debug(f"🤖 Agent 上下文：{len(context_messages)} 条消息，约 {current_tokens} tokens")
        return context_messages
    
    async def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        # 清理内存缓存（线程安全）
        deleted_from_cache = False
        async with self._cache_lock:
            if session_id in self._memory_cache:
                del self._memory_cache[session_id]
                deleted_from_cache = True
            if session_id in self._metadata_cache:
                del self._metadata_cache[session_id]
        
        # 删除磁盘文件
        session_file = self.storage_path / "sessions" / f"{session_id}.json"
        metadata_file = self.storage_path / "metadata" / f"{session_id}.json"
        
        try:
            files_deleted = False
            if session_file.exists():
                await aiofiles.os.remove(session_file)
                files_deleted = True
            if metadata_file.exists():
                await aiofiles.os.remove(metadata_file)
                files_deleted = True
            
            if deleted_from_cache or files_deleted:
                logger.info(f"🗑️ 已删除会话：{session_id}")
                return True
            else:
                logger.warning(f"⚠️ 尝试删除不存在的会话：{session_id}")
                return False
        except Exception as e:
            logger.error(f"删除会话失败：{e}")
            return False
    
    async def list_sessions(self, user_id: Optional[str] = None) -> List[Dict]:
        """列出所有会话（可按用户过滤）"""
        sessions_info = []
        
        async with self._cache_lock:
            # 遍历元数据缓存
            for session_id, metadata in self._metadata_cache.items():
                if user_id and metadata.get("user_id") != user_id:
                    continue
                sessions_info.append(metadata)
        
        # 按最后访问时间排序
        sessions_info.sort(key=lambda x: x.get("last_accessed", ""), reverse=True)
        
        return sessions_info
    
    async def cleanup_expired_sessions(self) -> int:
        """清理过期会话"""
        now = datetime.now()
        expired_count = 0
        
        # 获取所有会话 ID（需要在锁外处理，避免长时间持有锁）
        async with self._cache_lock:
            session_ids = list(self._metadata_cache.keys())
        
        for session_id in session_ids:
            metadata = self._metadata_cache.get(session_id, {})
            last_accessed = metadata.get("last_accessed")
            
            if last_accessed:
                last_time = datetime.fromisoformat(last_accessed)
                age = now - last_time
                
                if age > self.ttl:
                    await self.delete_session(session_id)
                    expired_count += 1
        
        logger.info(f"🧹 清理了 {expired_count} 个过期会话")
        return expired_count
    
    async def _persist_session(self, session_id: str):
        """持久化会话到磁盘"""
        async with self._cache_lock:
            if session_id not in self._memory_cache:
                return
            
            session = self._memory_cache[session_id]
            metadata = self._metadata_cache.get(session_id, {})
        
        try:
            # 保存会话数据
            session_file = self.storage_path / "sessions" / f"{session_id}.json"
            async with aiofiles.open(session_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(session, ensure_ascii=False, indent=2))
            
            # 保存元数据
            metadata_file = self.storage_path / "metadata" / f"{session_id}.json"
            async with aiofiles.open(metadata_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(metadata, ensure_ascii=False, indent=2))
            
            logger.debug(f"💾 已持久化会话：{session_id}")
        except Exception as e:
            logger.error(f"持久化会话失败：{e}")
    
    async def _load_session(self, session_id: str) -> Optional[Dict]:
        """从磁盘加载会话"""
        session_file = self.storage_path / "sessions" / f"{session_id}.json"
        
        if not session_file.exists():
            return None
        
        try:
            async with aiofiles.open(session_file, 'r', encoding='utf-8') as f:
                content = await f.read()
                session = json.loads(content)
            
            # 加载到 LRU 内存缓存（线程安全）
            async with self._cache_lock:
                self._lru_cache_set(session_id, session, self._memory_cache)
                
                # 同时加载元数据
                metadata_file = self.storage_path / "metadata" / f"{session_id}.json"
                if metadata_file.exists():
                    async with aiofiles.open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.loads(await f.read())
                    self._lru_cache_set(session_id, metadata, self._metadata_cache)
            
            logger.debug(f"📥 已加载会话：{session_id}")
            return session
        except Exception as e:
            logger.error(f"加载会话失败：{e}")
            return None
    
    def _estimate_tokens(self, text: str) -> int:
        """估算文本的 token 数（使用 tiktoken）"""
        return self._token_estimator.estimate_tokens(text)

    async def _auto_compress_history(self, session_id: str) -> int:
        """
        自动压缩过长会话历史

        策略：
        1. 保留最近 self.compression_keep_recent 条消息不变
        2. 将更早的消息按轮次对（user+assistant）合并为摘要
        3. 摘要作为 system 消息插入历史前面

        Returns:
            被压缩的消息数量
        """
        session = self._memory_cache.get(session_id)
        if not session:
            return 0

        messages = session.get("messages", [])
        total = len(messages)

        if total < self.compression_threshold:
            return 0

        keep_recent = self.compression_keep_recent
        to_compress = messages[:-keep_recent] if keep_recent > 0 else messages
        to_keep = messages[-keep_recent:] if keep_recent > 0 else []

        if not to_compress:
            return 0

        summary_parts = []
        for msg in to_compress:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            short_content = content[:120] + "..." if len(content) > 120 else content
            prefix = "用户" if role == "user" else "助手"
            summary_parts.append(f"[{prefix}]: {short_content}")

        summary_text = "【历史对话摘要】\n" + "\n".join(summary_parts)

        compressed_msg = {
            "role": "system",
            "content": summary_text,
            "timestamp": datetime.now().isoformat(),
            "metadata": {"compressed": True, "original_count": len(to_compress)},
            "token_count": self._estimate_tokens(summary_text),
        }

        session["messages"] = [compressed_msg] + to_keep
        session["message_count"] = len(session["messages"])
        session["_compressed_at"] = datetime.now().isoformat()

        asyncio.create_task(self._persist_session(session_id))

        return len(to_compress)
    
    def _compress_message(self, message: Dict, remaining_tokens: int) -> Optional[Dict]:
        """压缩消息以适应 token 预算"""
        content = message["content"]
        estimated_tokens = self._estimate_tokens(content)
        
        if estimated_tokens <= remaining_tokens:
            return message
        
        # 使用 TokenEstimator 进行准确截断
        compressed_content = self._token_estimator.truncate_to_tokens(content, remaining_tokens - 3)
        
        compressed_msg = message.copy()
        compressed_msg["content"] = compressed_content
        compressed_msg["token_count"] = self._estimate_tokens(compressed_content)
        compressed_msg["_compressed"] = True
        
        return compressed_msg


# 全局单例
session_manager = EnterpriseSessionManager()