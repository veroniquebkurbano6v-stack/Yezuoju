"""
三层记忆管理器 - StoryRag v2.0

三类记忆：

1. 短期记忆（ShortTermMemory）
   - 保存当前会话中的关键上下文
   - 滑动窗口 + 关键信息提取
   - 内存存储，随会话结束而清理

2. 长期记忆（LongTermMemory）
   - 保存经过筛选的用户稳定偏好或长期事实
   - 持久化到磁盘
   - 支持跨会话检索

3. 时间线记忆（TimelineMemory）
   - 对重要事件记录时间戳
   - 支持按时间顺序回溯和检索
   - 持久化到磁盘
"""

from __future__ import annotations

import os
import json
import asyncio
import logging
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import OrderedDict
import aiofiles
import aiofiles.os

logger = logging.getLogger(__name__)


# ============================================================
# 数据模型
# ============================================================

@dataclass
class MemoryEntry:
    """记忆条目"""

    memory_id: str
    """记忆唯一标识"""

    memory_type: str
    """记忆类型：short_term / long_term / timeline"""

    content: str
    """记忆内容"""

    category: str = "general"
    """分类：preference / fact / event / context / user_info"""

    importance: int = 1
    """重要性 (1-5)，1 最低，5 最高"""

    created_at: str = ""
    """创建时间"""

    last_accessed: str = ""
    """最后访问时间"""

    access_count: int = 0
    """访问次数"""

    session_id: str = ""
    """关联的会话 ID"""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """扩展元数据"""

    def __post_init__(self):
        now = datetime.now().isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.last_accessed:
            self.last_accessed = now

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        return cls(**data)


@dataclass
class TimelineEvent:
    """时间线事件"""

    event_id: str
    """事件唯一标识"""

    title: str
    """事件标题"""

    description: str
    """事件描述"""

    timestamp: str
    """事件发生时间"""

    event_type: str = "user_action"
    """事件类型：user_action / system_event / important_fact / milestone"""

    participants: List[str] = field(default_factory=list)
    """参与者/关联实体"""

    session_id: str = ""
    """关联的会话 ID"""

    importance: int = 1
    """重要性 (1-5)"""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """扩展元数据"""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TimelineEvent":
        return cls(**data)


# ============================================================
# 短期记忆
# ============================================================

class ShortTermMemory:
    """
    短期记忆

    在当前会话生命周期内保存关键上下文。
    使用滑动窗口 + LRU 淘汰策略。
    """

    def __init__(self, max_entries: int = 50):
        self.max_entries = max_entries
        self._entries: OrderedDict[str, MemoryEntry] = OrderedDict()
        self._lock = asyncio.Lock()

    async def add(self, content: str, category: str = "context",
                  importance: int = 1, session_id: str = "",
                  metadata: Dict = None) -> MemoryEntry:
        """
        添加短期记忆条目

        Args:
            content: 记忆内容
            category: 分类
            importance: 重要性
            session_id: 会话 ID
            metadata: 扩展元数据

        Returns:
            MemoryEntry 对象
        """
        import uuid

        entry = MemoryEntry(
            memory_id=str(uuid.uuid4())[:8],
            memory_type="short_term",
            content=content,
            category=category,
            importance=importance,
            session_id=session_id,
            metadata=metadata or {},
        )

        async with self._lock:
            self._entries[entry.memory_id] = entry
            self._entries.move_to_end(entry.memory_id)

            if len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

        logger.debug(f"[ShortTermMemory] 添加记忆：{content[:50]}... (id={entry.memory_id})")
        return entry

    async def get_all(self, limit: int = 20) -> List[MemoryEntry]:
        """获取最近的短期记忆条目"""
        async with self._lock:
            entries = list(self._entries.values())[-limit:]
            for e in entries:
                e.access_count += 1
                e.last_accessed = datetime.now().isoformat()
            return entries

    async def search(self, query: str, limit: int = 5) -> List[MemoryEntry]:
        """
        简单关键词搜索短期记忆

        Args:
            query: 搜索关键词
            limit: 最大返回数量

        Returns:
            匹配的 MemoryEntry 列表
        """
        results = []
        async with self._lock:
            for entry in reversed(list(self._entries.values())):
                if query.lower() in entry.content.lower():
                    results.append(entry)
                    entry.access_count += 1
                    if len(results) >= limit:
                        break
        return results

    async def get_context_prompt(self) -> str:
        """
        生成短期记忆的上下文提示文本

        用于注入到 Agent 的消息中。
        """
        entries = await self.get_all(limit=10)
        if not entries:
            return ""

        lines = ["## 当前会话关键上下文（短期记忆）"]
        for e in entries:
            lines.append(f"- [{e.category}] {e.content}")
        return "\n".join(lines)

    async def clear(self):
        """清空短期记忆"""
        async with self._lock:
            self._entries.clear()

    @property
    def size(self) -> int:
        return len(self._entries)


# ============================================================
# 长期记忆
# ============================================================

class LongTermMemory:
    """
    长期记忆

    保存用户的稳定偏好和长期事实，跨会话持久化。
    自动识别可提升为长期记忆的重要信息。
    """

    def __init__(self, storage_path: str = None):
        if storage_path is None:
            storage_path = str(
                Path(__file__).resolve().parents[2] / "src" / "data" / "memory"
            )
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._entries: Dict[str, MemoryEntry] = {}
        self._lock = asyncio.Lock()
        self._load_from_disk()

    def _get_file_path(self, user_id: str = "default") -> Path:
        return self.storage_path / f"long_term_{user_id}.json"

    def _load_from_disk(self):
        """从磁盘加载长期记忆"""
        file_path = self._get_file_path()
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        entry = MemoryEntry.from_dict(item)
                        self._entries[entry.memory_id] = entry
                logger.info(f"[LongTermMemory] 从磁盘加载 {len(self._entries)} 条长期记忆")
            except Exception as e:
                logger.error(f"[LongTermMemory] 加载失败：{e}")
                self._entries = {}

    async def _save_to_disk(self):
        """保存长期记忆到磁盘"""
        async with self._lock:
            data = [e.to_dict() for e in self._entries.values()]
            file_path = self._get_file_path()
            try:
                async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
                    await f.write(json.dumps(data, ensure_ascii=False, indent=2))
            except Exception as e:
                logger.error(f"[LongTermMemory] 保存失败：{e}")

    async def add(self, content: str, category: str = "preference",
                  importance: int = 1, session_id: str = "",
                  metadata: Dict = None) -> MemoryEntry:
        """
        添加长期记忆条目（自动去重）

        Args:
            content: 记忆内容
            category: 分类（preference / fact / user_info）
            importance: 重要性 (1-5)
            session_id: 关联会话 ID
            metadata: 扩展元数据
        """
        import uuid

        entry = MemoryEntry(
            memory_id=str(uuid.uuid4())[:8],
            memory_type="long_term",
            content=content,
            category=category,
            importance=importance,
            session_id=session_id,
            metadata=metadata or {},
        )

        async with self._lock:
            for existing in self._entries.values():
                if existing.content.strip() == content.strip():
                    existing.importance = max(existing.importance, importance)
                    existing.last_accessed = datetime.now().isoformat()
                    break
            else:
                self._entries[entry.memory_id] = entry

        await self._save_to_disk()
        logger.info(f"[LongTermMemory] 添加长期记忆：{content[:50]}...")
        return entry

    async def update(self, memory_id: str, **kwargs):
        """更新长期记忆条目"""
        async with self._lock:
            if memory_id in self._entries:
                entry = self._entries[memory_id]
                for key, value in kwargs.items():
                    if hasattr(entry, key):
                        setattr(entry, key, value)
                entry.last_accessed = datetime.now().isoformat()
        await self._save_to_disk()

    async def delete(self, memory_id: str):
        """删除长期记忆条目"""
        async with self._lock:
            if memory_id in self._entries:
                del self._entries[memory_id]
        await self._save_to_disk()
        logger.info(f"[LongTermMemory] 删除记忆：{memory_id}")

    async def get_all(self, limit: int = 50) -> List[MemoryEntry]:
        """获取所有长期记忆条目（按重要性排序）"""
        async with self._lock:
            entries = sorted(
                self._entries.values(),
                key=lambda e: (e.importance, e.created_at),
                reverse=True,
            )
            return entries[:limit]

    async def search(self, query: str, limit: int = 5) -> List[MemoryEntry]:
        """
        搜索长期记忆

        Args:
            query: 搜索关键词
            limit: 最大返回数量
        """
        results = []
        needs_save = False
        async with self._lock:
            for entry in self._entries.values():
                if query.lower() in entry.content.lower():
                    results.append(entry)
                    entry.access_count += 1
                    needs_save = True
            results.sort(key=lambda e: e.importance, reverse=True)
        if needs_save and results:
            await self._save_to_disk()
        return results[:limit]

    async def get_context_prompt(self, current_query: str = "") -> str:
        """
        生成长期记忆的上下文提示文本

        Args:
            current_query: 当前查询，用于检索相关记忆
        """
        if current_query:
            entries = await self.search(current_query, limit=5)
        else:
            entries = await self.get_all(limit=10)

        if not entries:
            return ""

        lines = ["## 关于用户的长期记忆"]
        for e in entries:
            lines.append(f"- [{e.category}] {e.content} (重要性:{e.importance})")
        return "\n".join(lines)

    async def should_promote(self, content: str, turn_count: int,
                             repeated_count: int = 0) -> bool:
        """
        判断是否应将内容提升为长期记忆

        规则：
        - 被多次提及的主题
        - 明确表达偏好的内容
        - 重要的个人事实
        - 重要性评分 >= 4

        Args:
            content: 内容
            turn_count: 当前会话轮数
            repeated_count: 被重复提及的次数
        """
        preference_markers = ["喜欢", "偏好", "习惯", "总是", "经常", "不喜欢", "讨厌"]
        fact_markers = ["我是", "我在", "我的", "我住在", "我的工作是"]

        has_preference = any(m in content for m in preference_markers)
        has_fact = any(m in content for m in fact_markers)
        is_repeated = repeated_count >= 2

        return has_preference or has_fact or is_repeated

    async def sync_from_session(self, session_memory: ShortTermMemory):
        """
        从短期记忆中提取可提升为长期记忆的条目

        Args:
            session_memory: 短期记忆实例
        """
        entries = await session_memory.get_all(limit=20)
        promoted = 0
        for entry in entries:
            if entry.importance >= 3 or await self.should_promote(
                entry.content, turn_count=0, repeated_count=entry.access_count
            ):
                exists = await self.search(entry.content, limit=1)
                if not exists:
                    await self.add(
                        content=entry.content,
                        category=entry.category,
                        importance=entry.importance,
                        session_id=entry.session_id,
                        metadata=entry.metadata,
                    )
                    promoted += 1
        if promoted > 0:
            logger.info(f"[LongTermMemory] 从短期记忆中提升了 {promoted} 条记忆")

    @property
    def size(self) -> int:
        return len(self._entries)


# ============================================================
# 时间线记忆
# ============================================================

class TimelineMemory:
    """
    时间线记忆

    记录重要事件的时间戳，支持按时间顺序回溯。
    """

    def __init__(self, storage_path: str = None):
        if storage_path is None:
            storage_path = str(
                Path(__file__).resolve().parents[2] / "src" / "data" / "memory"
            )
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._events: Dict[str, TimelineEvent] = {}
        self._lock = asyncio.Lock()
        self._load_from_disk()

    def _get_file_path(self, user_id: str = "default") -> Path:
        return self.storage_path / f"timeline_{user_id}.json"

    def _load_from_disk(self):
        file_path = self._get_file_path()
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        event = TimelineEvent.from_dict(item)
                        self._events[event.event_id] = event
                logger.info(f"[TimelineMemory] 从磁盘加载 {len(self._events)} 个时间线事件")
            except Exception as e:
                logger.error(f"[TimelineMemory] 加载失败：{e}")
                self._events = {}

    async def _save_to_disk(self):
        async with self._lock:
            data = [e.to_dict() for e in self._events.values()]
            file_path = self._get_file_path()
            try:
                async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
                    await f.write(json.dumps(data, ensure_ascii=False, indent=2))
            except Exception as e:
                logger.error(f"[TimelineMemory] 保存失败：{e}")

    async def add_event(self, title: str, description: str = "",
                        event_type: str = "user_action",
                        participants: List[str] = None,
                        session_id: str = "",
                        importance: int = 1,
                        timestamp: str = None,
                        metadata: Dict = None) -> TimelineEvent:
        """
        添加时间线事件

        Args:
            title: 事件标题
            description: 事件描述
            event_type: 事件类型
            participants: 参与者列表
            session_id: 会话 ID
            importance: 重要性
            timestamp: 时间戳（不提供则使用当前时间）
            metadata: 扩展元数据

        Returns:
            TimelineEvent 对象
        """
        import uuid

        event = TimelineEvent(
            event_id=str(uuid.uuid4())[:8],
            title=title,
            description=description,
            timestamp=timestamp or datetime.now().isoformat(),
            event_type=event_type,
            participants=participants or [],
            session_id=session_id,
            importance=importance,
            metadata=metadata or {},
        )

        async with self._lock:
            self._events[event.event_id] = event

        await self._save_to_disk()
        logger.info(f"[TimelineMemory] 添加事件：{title}")
        return event

    async def get_events_by_time(self, start_time: str = None,
                                 end_time: str = None,
                                 event_type: str = None,
                                 limit: int = 50) -> List[TimelineEvent]:
        """
        按时间范围检索事件

        Args:
            start_time: 起始时间（ISO格式）
            end_time: 结束时间（ISO格式）
            event_type: 事件类型过滤
            limit: 最大数量
        """
        async with self._lock:
            events = list(self._events.values())

            if start_time:
                events = [e for e in events if e.timestamp >= start_time]
            if end_time:
                events = [e for e in events if e.timestamp <= end_time]
            if event_type:
                events = [e for e in events if e.event_type == event_type]

            events.sort(key=lambda e: e.timestamp)
            return events[:limit]

    async def get_recent(self, days: int = 30, limit: int = 20) -> List[TimelineEvent]:
        """获取最近 N 天的事件"""
        start_time = (datetime.now() - timedelta(days=days)).isoformat()
        return await self.get_events_by_time(start_time=start_time, limit=limit)

    async def search(self, query: str, limit: int = 10) -> List[TimelineEvent]:
        """
        搜索时间线事件

        Args:
            query: 搜索关键词
            limit: 最大返回数量
        """
        results = []
        async with self._lock:
            for event in self._events.values():
                if (query.lower() in event.title.lower() or
                        query.lower() in event.description.lower()):
                    results.append(event)
            results.sort(key=lambda e: e.timestamp)
        return results[:limit]

    async def get_timeline_prompt(self, days: int = 30) -> str:
        """
        生成时间线上下文提示

        Args:
            days: 查看最近多少天的事件
        """
        events = await self.get_recent(days=days, limit=10)
        if not events:
            return ""

        lines = [f"## 近期重要事件（最近{days}天）"]
        for e in events:
            dt = datetime.fromisoformat(e.timestamp).strftime("%m-%d %H:%M")
            lines.append(f"- [{dt}] {e.title}" +
                         (f"：{e.description[:60]}" if e.description else ""))
        return "\n".join(lines)

    @property
    def size(self) -> int:
        return len(self._events)


# ============================================================
# 统一记忆管理器
# ============================================================

class MemoryManager:
    """
    统一记忆管理器

    协调三层记忆的读写和生命周期管理。
    对外提供统一的记忆接口。
    """

    def __init__(self, storage_path: str = None):
        self.short_term = ShortTermMemory(max_entries=50)
        self.long_term = LongTermMemory(storage_path=storage_path)
        self.timeline = TimelineMemory(storage_path=storage_path)
        logger.info("[MemoryManager] 三层记忆系统初始化完成")

    async def record_user_message(self, message: str, session_id: str = "",
                                  importance: int = 1):
        await self.short_term.add(
            content=f"用户说：{message[:200]}",
            category="context",
            importance=importance,
            session_id=session_id,
        )
        preference_markers = ["喜欢", "偏好", "习惯", "总是", "经常", "不喜欢", "讨厌"]
        fact_markers = ["我是", "我在", "我的", "我住在", "我的工作是"]
        is_preference = any(m in message for m in preference_markers)
        is_fact = any(m in message for m in fact_markers)
        if is_preference or is_fact:
            category = "preference" if is_preference else "fact"
            await self.long_term.add(
                content=message,
                category=category,
                importance=max(importance, 4),
                session_id=session_id,
            )
            logger.info(f"[MemoryManager] 检测到{category}，已实时写入长期记忆：{message[:60]}...")

    async def record_assistant_response(self, response: str, session_id: str = "",
                                        importance: int = 1):
        """
        记录助手回复到短期记忆

        Args:
            response: 助手回复
            session_id: 会话 ID
            importance: 重要性
        """
        await self.short_term.add(
            content=f"助手回答涉及：{response[:150]}",
            category="context",
            importance=importance,
            session_id=session_id,
        )

    async def record_user_fact(self, fact: str, session_id: str = "",
                               importance: int = 3):
        """
        记录用户事实到长期记忆

        Args:
            fact: 用户事实
            session_id: 会话 ID
            importance: 重要性
        """
        await self.long_term.add(
            content=fact,
            category="fact",
            importance=importance,
            session_id=session_id,
        )

    async def record_user_preference(self, preference: str, session_id: str = "",
                                     importance: int = 4):
        """
        记录用户偏好到长期记忆

        Args:
            preference: 用户偏好
            session_id: 会话 ID
            importance: 重要性
        """
        await self.long_term.add(
            content=preference,
            category="preference",
            importance=importance,
            session_id=session_id,
        )

    async def record_event(self, title: str, description: str = "",
                           importance: int = 2, session_id: str = ""):
        """
        记录事件到时间线

        Args:
            title: 事件标题
            description: 事件描述
            importance: 重要性
            session_id: 会话 ID
        """
        await self.timeline.add_event(
            title=title,
            description=description,
            importance=importance,
            session_id=session_id,
        )

    async def build_full_context(self, current_query: str = "",
                                 user_name: str = "") -> str:
        """
        构建完整的三层记忆上下文

        用于注入到 Agent 的消息队列中。

        Args:
            current_query: 当前用户查询
            user_name: 用户名称

        Returns:
            格式化的记忆上下文文本
        """
        parts = []

        short_term_prompt = await self.short_term.get_context_prompt()
        if short_term_prompt:
            parts.append(short_term_prompt)

        long_term_prompt = await self.long_term.get_context_prompt(current_query)
        if long_term_prompt:
            parts.append(long_term_prompt)

        timeline_prompt = await self.timeline.get_timeline_prompt()
        if timeline_prompt:
            parts.append(timeline_prompt)

        return "\n\n".join(parts) if parts else ""

    async def cleanup_session(self, session_id: str = ""):
        """
        会话结束时清理短期记忆，并同步长期记忆

        Args:
            session_id: 会话 ID
        """
        await self.long_term.sync_from_session(self.short_term)
        await self.short_term.clear()
        logger.info(f"[MemoryManager] 会话 {session_id} 记忆已清理")

    async def search_all(self, query: str) -> Dict[str, Any]:
        """
        跨所有记忆层搜索

        Args:
            query: 搜索关键词

        Returns:
            包含所有层级搜索结果的字典
        """
        return {
            "short_term": [e.to_dict() for e in await self.short_term.search(query)],
            "long_term": [e.to_dict() for e in await self.long_term.search(query)],
            "timeline": [e.to_dict() for e in await self.timeline.search(query)],
        }


# ============================================================
# 全局单例
# ============================================================

_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """获取统一记忆管理器实例"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager
