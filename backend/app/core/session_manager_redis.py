"""
基于 Redis 的企业级会话管理器 - StoryRag v2.0

特性：
1. 微秒级访问延迟
2. 分布式共享
3. 原生 TTL 过期
4. 丰富的数据结构支持
5. AOF/RDB 持久化
"""
import redis.asyncio as redis
import json
import logging
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta

try:
    from src.utils.logger import get_logger
except ImportError:
    def get_logger(name):
        return logging.getLogger(name)

logger = get_logger(__name__)


class RedisSessionManager:
    """
    Redis 会话管理器
    
    Redis 数据结构设计：
    - Hash: session:{session_id} → 会话元数据
    - List: session:{session_id}:messages → 消息列表（倒序）
    - Set: user_sessions:{user_id} → 用户会话索引
    - Sorted Set: global_timeline → 全局时间线（用于分页查询）
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        password: Optional[str] = None,
        db: int = 0,
        ttl_days: int = 30,
        max_messages: int = 100
    ):
        self.redis_url = redis_url
        self.password = password
        self.db = db
        self.ttl = timedelta(days=ttl_days)
        self.max_messages = max_messages
        self.redis_client: Optional[redis.Redis] = None
        
        logger.info(f"🔴 RedisSessionManager 初始化")
        logger.info(f"   Redis URL: {redis_url}")
        logger.info(f"   TTL: {ttl_days} 天")
        logger.info(f"   最大消息数：{max_messages}")
    
    async def connect(self):
        """连接到 Redis"""
        if not self.redis_client:
            try:
                self.redis_client = await redis.from_url(
                    self.redis_url,
                    password=self.password,
                    db=self.db,
                    encoding="utf-8",
                    decode_responses=True
                )
                
                # 测试连接
                await self.redis_client.ping()
                logger.info("✅ Redis 连接成功")
                
                # 获取 Redis 信息
                info = await self.redis_client.info('server')
                logger.info(f"   Redis 版本：{info.get('redis_version', 'unknown')}")
                logger.info(f"   模式：{info.get('redis_mode', 'standalone')}")
                
            except Exception as e:
                logger.error(f"❌ Redis 连接失败：{e}")
                raise
    
    async def close(self):
        """关闭 Redis 连接"""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("🔴 Redis 连接已关闭")
    
    async def create_session(
        self, 
        session_id: str, 
        user_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """创建会话"""
        now = datetime.now().isoformat()
        
        session_key = f"session:{session_id}"
        session_data = {
            "session_id": session_id,
            "user_id": user_id or "anonymous",
            "metadata": json.dumps(metadata or {}),
            "created_at": now,
            "last_accessed": now,
            "message_count": "0"
        }
        
        # 写入 Hash（原子操作）
        await self.redis_client.hset(session_key, mapping=session_data)
        
        # 设置 TTL（自动过期）
        await self.redis_client.expire(session_key, int(self.ttl.total_seconds()))
        
        # 更新用户会话索引
        if user_id:
            user_index_key = f"user_sessions:{user_id}"
            await self.redis_client.sadd(user_index_key, session_id)
            await self.redis_client.expire(user_index_key, int(self.ttl.total_seconds()))
        
        # 添加到全局时间线（用于分页查询）
        timeline_key = "global_timeline"
        await self.redis_client.zadd(timeline_key, {session_id: datetime.now().timestamp()})
        
        logger.info(f"📝 创建会话：{session_id}")
        return session_id
    
    async def add_message(
        self, 
        session_id: str, 
        role: str, 
        content: str,
        metadata: Optional[Dict] = None
    ) -> bool:
        """添加消息到会话（如果会话不存在则自动创建）"""
        session_key = f"session:{session_id}"
        messages_key = f"{session_key}:messages"
        
        # 🔥 检查会话是否存在
        exists = await self.redis_client.exists(session_key)
        if not exists:
            # 🔥 会话不存在，自动创建
            logger.info(f"📝 会话不存在，自动创建：{session_id}")
            await self.create_session(session_id)
        
        # 构建消息对象
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": json.dumps(metadata or {}),
            "token_count": str(self._estimate_tokens(content))
        }
        
        # 左侧推送（保持顺序：最新的在前面）
        await self.redis_client.lpush(messages_key, json.dumps(message, ensure_ascii=False))
        
        # 限制消息数量（trimming）
        if self.max_messages > 0:
            await self.redis_client.ltrim(messages_key, 0, self.max_messages - 1)
        
        # 更新最后访问时间
        await self.redis_client.hset(
            session_key,
            "last_accessed",
            datetime.now().isoformat()
        )
        
        # 增加消息计数
        await self.redis_client.hincrby(session_key, "message_count", 1)
        
        logger.debug(f"💾 添加消息到 {session_id}")
        return True
    
    async def get_context_for_agent(
        self, 
        session_id: str, 
        current_query: str
    ) -> List[Dict]:
        """为 Agent 构建优化的上下文"""
        messages_key = f"session:{session_id}:messages"
        
        # 获取所有消息（List 是倒序存储：0=最新）
        raw_messages = await self.redis_client.lrange(messages_key, 0, -1)
        
        if not raw_messages:
            return [{"role": "user", "content": current_query}]
        
        # 解析消息
        messages = [json.loads(msg) for msg in raw_messages]
        
        # 构建上下文
        context = []
        token_count = 0
        max_tokens = 4096
        min_recent_turns = 3
        
        for i, msg in enumerate(messages):
            msg_tokens = int(msg.get("token_count", 50))
            
            # 始终保留最近 3 条
            if i < min_recent_turns:
                context.insert(0, msg)
                token_count += msg_tokens
                continue
            
            # 检查 token 限制
            if token_count + msg_tokens > max_tokens:
                break
            
            # 优先保留有工具调用的消息
            metadata = json.loads(msg.get("metadata", "{}"))
            has_tool_calls = metadata.get("tool_calls")
            has_references = metadata.get("references")
            
            if has_tool_calls or has_references:
                context.insert(0, msg)
                token_count += msg_tokens
            else:
                # 普通消息：检查是否需要压缩
                remaining_tokens = max_tokens - token_count
                if msg_tokens > remaining_tokens:
                    compressed_content = msg["content"][:int(len(msg["content"]) * (remaining_tokens / msg_tokens))]
                    if len(compressed_content) > 50:
                        compressed_content = compressed_content[:50] + "..."
                    
                    msg["content"] = compressed_content
                    msg["token_count"] = str(self._estimate_tokens(compressed_content))
                
                context.insert(0, msg)
                token_count += int(msg["token_count"])
        
        # 添加当前查询
        context.append({"role": "user", "content": current_query})
        
        logger.debug(f"🤖 Agent 上下文：{len(context)} 条消息，约 {token_count} tokens")
        return context
    
    async def get_messages(
        self, 
        session_id: str, 
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict]:
        """分页获取消息列表"""
        messages_key = f"session:{session_id}:messages"
        
        start = offset
        end = offset + limit - 1
        
        raw_messages = await self.redis_client.lrange(messages_key, start, end)
        messages = [json.loads(msg) for msg in reversed(raw_messages)]
        
        return messages
    
    async def list_sessions(self, user_id: Optional[str] = None) -> List[Dict]:
        """列出所有会话"""
        pattern = "session:*"
        sessions = []
        
        async for key in self.redis_client.scan_iter(pattern, count=100):
            if ":messages" in key:
                continue
            
            session_data = await self.redis_client.hgetall(key)
            
            if session_data:
                if "metadata" in session_data:
                    try:
                        session_data["metadata"] = json.loads(session_data["metadata"])
                    except:
                        pass
                
                if user_id and session_data.get("user_id") != user_id:
                    continue
                
                sessions.append(session_data)
        
        sessions.sort(key=lambda x: x.get("last_accessed", ""), reverse=True)
        return sessions
    
    async def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        session_key = f"session:{session_id}"
        messages_key = f"{session_key}:messages"
        
        session_data = await self.redis_client.hgetall(session_key)
        user_id = session_data.get("user_id") if session_data else None
        
        deleted = await self.redis_client.delete(session_key, messages_key)
        
        if user_id:
            user_index_key = f"user_sessions:{user_id}"
            await self.redis_client.srem(user_index_key, session_id)
        
        timeline_key = "global_timeline"
        await self.redis_client.zrem(timeline_key, session_id)
        
        if deleted > 0:
            logger.info(f"🗑️ 已删除会话：{session_id}")
            return True
        
        logger.warning(f"⚠️ 尝试删除不存在的会话：{session_id}")
        return False
    
    async def cleanup_expired_sessions(self) -> int:
        """清理过期会话 - Redis 会自动处理 TTL 过期的 key"""
        logger.info("🧹 Redis 会自动清理过期会话，无需手动操作")
        return 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        info = await self.redis_client.info('stats')
        memory_info = await self.redis_client.info('memory')
        
        session_count = 0
        async for _ in self.redis_client.scan_iter("session:*", count=100):
            session_count += 1
        
        return {
            "total_sessions": session_count,
            "redis_version": info.get("redis_version", "unknown"),
            "connected_clients": info.get("connected_clients", 0),
            "used_memory_human": memory_info.get("used_memory_human", "unknown"),
            "used_memory_peak_human": memory_info.get("used_memory_peak_human", "unknown"),
        }
    
    def _estimate_tokens(self, text: str) -> int:
        """估算文本的 token 数"""
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        english_chars = sum(1 for c in text if c.isascii() and c.isalpha())
        return int(chinese_chars * 1.5 + english_chars * 0.25) + 10


# 全局单例
redis_session_manager = RedisSessionManager()
