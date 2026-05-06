"""
RAG 系统配置管理

职责：
1. 集中管理所有配置
2. 从环境变量读取
3. 提供类型安全的配置访问
"""
from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache


class RAGConfig:
    """
    RAG 系统配置
    
    使用类变量存储配置，支持从环境变量读取
    """
    
    # ==================== DeepSeek 配置 ====================
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    
    # ==================== 向量数据库配置 ====================
    VECTOR_DB_PATH: str = os.getenv(
        "VECTOR_DB_PATH", 
        str(Path(__file__).resolve().parents[1] / "data" / "vector_database")
    )
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    
    # ==================== Redis 配置（可选）====================
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")
    REDIS_TTL_DAYS: int = int(os.getenv("REDIS_TTL_DAYS", "30"))
    
    # ==================== 会话管理配置 ====================
    SESSION_MAX_MESSAGES: int = int(os.getenv("SESSION_MAX_MESSAGES", "100"))
    SESSION_TTL_DAYS: int = int(os.getenv("SESSION_TTL_DAYS", "30"))
    
    # ==================== 检索配置 ====================
    DEFAULT_TOP_K: int = 10
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    MMR_LAMBDA: float = 0.5  # MMR 多样性参数
    
    # ==================== 缓存配置 ====================
    CACHE_ENABLED: bool = os.getenv("CACHE_ENABLED", "true").lower() == "true"
    CACHE_DIR: str = os.getenv("CACHE_DIR", "./cache")
    
    # ==================== 日志配置 ====================
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @classmethod
    def validate(cls) -> list[str]:
        """
        验证配置完整性
        
        Returns:
            错误信息列表，空列表表示配置有效
        """
        errors = []
        
        if not cls.DEEPSEEK_API_KEY:
            errors.append("DEEPSEEK_API_KEY 未设置")
        
        if not Path(cls.VECTOR_DB_PATH).exists():
            errors.append(f"向量数据库路径不存在: {cls.VECTOR_DB_PATH}")
        
        return errors
    
    @classmethod
    def summary(cls) -> str:
        """生成配置摘要"""
        return f"""
RAG Config Summary:
  DeepSeek API: {'✓' if cls.DEEPSEEK_API_KEY else '✗'}
  Vector DB: {cls.VECTOR_DB_PATH}
  Embedding Model: {cls.EMBEDDING_MODEL}
  Redis: {'✓' if cls.REDIS_PASSWORD else '✗ (using default)'}
  Cache: {'✓' if cls.CACHE_ENABLED else '✗'}
        """.strip()


# 全局配置实例
config = RAGConfig()


@lru_cache(maxsize=1)
def get_config() -> RAGConfig:
    """
    获取配置实例（带缓存）
    
    Returns:
        RAGConfig 实例
    """
    return config


# 便捷函数
def get_deepseek_config() -> dict:
    """获取 DeepSeek 配置"""
    return {
        "api_key": config.DEEPSEEK_API_KEY,
        "base_url": config.DEEPSEEK_BASE_URL,
        "model": config.DEEPSEEK_MODEL
    }


def get_vector_db_config() -> dict:
    """获取向量数据库配置"""
    return {
        "db_path": config.VECTOR_DB_PATH,
        "embedding_model": config.EMBEDDING_MODEL
    }


def get_redis_config() -> dict:
    """获取 Redis 配置"""
    return {
        "url": config.REDIS_URL,
        "password": config.REDIS_PASSWORD if config.REDIS_PASSWORD else None,
        "ttl_days": config.REDIS_TTL_DAYS
    }
