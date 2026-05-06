"""
配置中心 - StoryRag v2.0
使用 Pydantic Settings 进行配置管理，支持环境变量覆盖
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class SessionManagerConfig(BaseSettings):
    """会话管理器配置"""
    
    # 消息限制
    max_messages_per_session: int = 100
    max_tokens_for_agent: int = 4096
    
    # 缓存配置
    max_cache_size: int = 100
    ttl_days: int = 30
    
    # 存储路径
    session_data_path: str = "../src/data/sessions"
    
    # Redis 配置
    redis_url: str = "redis://localhost:6379"
    redis_password: Optional[str] = None
    redis_db: int = 0
    
    # Token 估算模型
    token_model_name: str = "gpt-3.5-turbo"
    
    model_config = SettingsConfigDict(env_prefix="SESSION_")


class RAGConfig(BaseSettings):
    """RAG 系统配置"""

    # 检索配置
    default_top_k: int = 60
    default_candidate_top_k: int = 120
    hybrid_alpha: float = 0.7

    # 模型配置
    embedding_model: str = "intfloat/multilingual-e5-large"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # 路径配置
    vector_db_path: str = "src/data/vector_database"
    session_data_path: str = "../src/data/sessions"

    model_config = SettingsConfigDict(env_prefix="RAG_")


class AppSettings(BaseSettings):
    """应用全局配置（统一入口，向后兼容 v1.0）"""

    # DeepSeek API
    DEEPSEEK_API_KEY: Optional[str] = None
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com/v1"
    DEEPSEEK_MODEL: str = "deepseek-chat"

    # 向量数据库
    VECTOR_DB_PATH: str = "../src/data/vector_database"
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-large"

    # 角色系统（v2.0）
    DEFAULT_ROLE_ID: str = "humorous_butler"

    model_config = SettingsConfigDict(env_file="../../.env", extra="allow")

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        if hasattr(rag_config, name):
            return getattr(rag_config, name)
        if hasattr(session_config, name):
            return getattr(session_config, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


# 全局配置实例
session_config = SessionManagerConfig()
rag_config = RAGConfig()
settings = AppSettings()
