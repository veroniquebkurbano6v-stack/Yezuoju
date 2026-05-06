"""
统一配置管理系统 - StoryRag v2.0

职责：
1. 集中管理所有应用配置
2. 从环境变量和 .env 文件读取
3. 提供类型安全的配置访问
4. 支持开发和生产环境

使用方式：
    from src.config import settings
    
    api_key = settings.DEEPSEEK_API_KEY
    db_path = settings.VECTOR_DB_PATH
"""
import os
from pathlib import Path
from typing import Optional, List
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """
    应用全局配置
    
    优先级：环境变量 > .env 文件 > 默认值
    """
    
    model_config = {"extra": "allow"}
    
    # ==================== 应用基本信息 ====================
    APP_NAME: str = "StoryRag v2.0"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False
    
    # ==================== DeepSeek LLM 配置 ====================
    DEEPSEEK_API_KEY: Optional[str] = None
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com/v1"
    DEEPSEEK_MODEL: str = "deepseek-chat"
    
    # ==================== 向量数据库配置 ====================
    VECTOR_DB_PATH: str = ""  # 将在 model_post_init 中设置默认值
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    
    # ==================== Redis 配置（可选）====================
    REDIS_URL: str = "redis://localhost:6379"
    REDIS_PASSWORD: Optional[str] = None
    REDIS_TTL_DAYS: int = 30
    CACHE_ENABLED: bool = True
    
    # ==================== 会话管理配置 ====================
    SESSION_MAX_MESSAGES: int = 100
    SESSION_TTL_DAYS: int = 30
    SESSION_STORAGE_PATH: str = "./src/data/sessions"
    
    # ==================== 检索配置 ====================
    DEFAULT_TOP_K: int = 10
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    MMR_LAMBDA: float = 0.5  # MMR 多样性参数
    
    # ==================== 缓存配置 ====================
    CACHE_DIR: str = "./cache"
    
    # ==================== 日志配置 ====================
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # ==================== API 配置 ====================
    MAX_QUERY_LENGTH: int = 1000
    MAX_RESPONSE_LENGTH: int = 2000
    
    # ==================== CORS 配置 ====================
    FRONTEND_URL: str = "http://localhost:5173"
    CORS_ORIGINS: List[str] = []  # 将在 model_post_init 中设置
    
    def model_post_init(self, __context):
        """模型初始化后的处理"""
        # 设置 VECTOR_DB_PATH 默认值
        if not self.VECTOR_DB_PATH:
            project_root = Path(__file__).resolve().parents[1]
            self.VECTOR_DB_PATH = str(project_root / "data" / "vector_database")
        
        # 设置 CORS_ORIGINS
        if not self.CORS_ORIGINS:
            self.CORS_ORIGINS = [self.FRONTEND_URL]
        
        # DEBUG 模式下的特殊配置
        if self.DEBUG:
            self.LOG_LEVEL = "DEBUG"
    
    def validate(self) -> list[str]:
        """
        验证配置完整性
        
        Returns:
            错误信息列表，空列表表示配置有效
        """
        errors = []
        
        if not self.DEEPSEEK_API_KEY:
            errors.append("⚠️  DEEPSEEK_API_KEY 未设置，请在 .env 文件中配置")
        
        vector_db_path = Path(self.VECTOR_DB_PATH)
        if not vector_db_path.exists():
            errors.append(f"⚠️  向量数据库路径不存在: {self.VECTOR_DB_PATH}")
            errors.append("   请运行: python src/data_processing/process_pipeline.py")
        
        return errors
    
    def summary(self) -> str:
        """生成配置摘要"""
        return f"""
╔══════════════════════════════════════════╗
║     StoryRag v2.0 配置摘要              ║
╠══════════════════════════════════════════╣
║  DeepSeek API: {'✓' if self.DEEPSEEK_API_KEY else '✗'}
║  Vector DB: {self.VECTOR_DB_PATH[:50]}...
║  Embedding: {self.EMBEDDING_MODEL}
║  Redis: {'✓' if self.REDIS_PASSWORD else '✗ (using default)'}
║  Cache: {'✓' if self.CACHE_ENABLED else '✗'}
║  Debug: {'✓' if self.DEBUG else '✗'}
╚══════════════════════════════════════════╝
        """.strip()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    获取配置实例（单例模式，带缓存）
    
    Returns:
        Settings 实例
    """
    # 确定 .env 文件路径
    # parents[1] 从 src/config.py 向上一级到达项目根目录
    env_path = Path(__file__).resolve().parents[1] / ".env"
    
    if env_path.exists():
        settings = Settings(_env_file=str(env_path))
    else:
        settings = Settings()
    
    # 验证配置
    errors = settings.validate()
    if errors:
        print("\n".join(errors))
    
    return settings


# 全局配置实例（延迟加载）
_settings_instance = None


def get_config() -> Settings:
    """
    获取配置实例（兼容旧代码）
    
    Returns:
        Settings 实例
    """
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = get_settings()
    return _settings_instance


# 便捷函数（保持向后兼容）
def get_deepseek_config() -> dict:
    """获取 DeepSeek 配置"""
    config = get_config()
    return {
        "api_key": config.DEEPSEEK_API_KEY,
        "base_url": config.DEEPSEEK_BASE_URL,
        "model": config.DEEPSEEK_MODEL
    }


def get_vector_db_config() -> dict:
    """获取向量数据库配置"""
    config = get_config()
    return {
        "db_path": config.VECTOR_DB_PATH,
        "embedding_model": config.EMBEDDING_MODEL
    }


def get_redis_config() -> dict:
    """获取 Redis 配置"""
    config = get_config()
    return {
        "url": config.REDIS_URL,
        "password": config.REDIS_PASSWORD if config.REDIS_PASSWORD else None,
        "ttl_days": config.REDIS_TTL_DAYS
    }


# 导出全局配置实例（用于直接导入）
settings = get_config()


if __name__ == "__main__":
    """测试配置加载"""
    print(settings.summary())
    print(f"\n完整配置：")
    for key, value in settings.model_dump().items():
        # 隐藏敏感信息
        if "KEY" in key or "PASSWORD" in key:
            value = "***" if value else None
        print(f"  {key}: {value}")
