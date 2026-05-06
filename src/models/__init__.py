"""
RAG 系统核心数据模型和配置管理

职责：
1. 定义不可变数据结构
2. 提供配置管理系统
3. 确保类型安全
"""
from __future__ import annotations

# 导出数据类
from .models import (
    Document,
    Query,
    RetrievalResult,
    ChatMessage,
    RAGResponse,
    SessionMetadata,
    UsageSummary,
    ToolExecution,
    ChapterInfo
)

# 导出配置管理（向后兼容，实际使用 src.config）
try:
    from .config import RAGConfig, config, get_config, get_deepseek_config, get_vector_db_config, get_redis_config
except ImportError:
    # 如果旧配置文件不存在，从新的统一配置导入
    from src.config import Settings as RAGConfig
    from src.config import settings as config
    from src.config import get_settings as get_config
    from src.config import get_deepseek_config, get_vector_db_config, get_redis_config

# 导出配置加载器
from .config_loader import ConfigLoader, config_loader, get_config_loader, get_commands, get_tools, get_command, get_tool

__all__ = [
    # 数据类
    'Document',
    'Query',
    'RetrievalResult',
    'ChatMessage',
    'RAGResponse',
    'SessionMetadata',
    'UsageSummary',
    'ToolExecution',
    'ChapterInfo',
    # 配置管理
    'RAGConfig',
    'config',
    'get_config',
    'get_deepseek_config',
    'get_vector_db_config',
    'get_redis_config',
    # 配置加载器
    'ConfigLoader',
    'config_loader',
    'get_config_loader',
    'get_commands',
    'get_tools',
    'get_command',
    'get_tool'
]
