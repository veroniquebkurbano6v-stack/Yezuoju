"""
应用核心配置文件
管理环境变量和应用配置
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """应用配置类"""
    # 基本配置
    APP_NAME: str = "PDF智能检索系统"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
    
    # DeepSeek配置
    DEEPSEEK_API_KEY: Optional[str] = os.getenv("DEEPSEEK_API_KEY")
    DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    
    # 向量数据库配置
    VECTOR_DB_PATH: str = os.getenv("VECTOR_DB_PATH", "../src/data/vector_database")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
    
    # 应用配置
    MAX_QUERY_LENGTH: int = 1000
    MAX_RESPONSE_LENGTH: int = 2000
    DEFAULT_TOP_K: int = 60
    
    # 前端配置
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:3000")
    
    # CORS配置
    CORS_ORIGINS: list = [FRONTEND_URL]
    
    class Config:
        env_file = "../../.env"
        case_sensitive = True


# 创建全局配置实例
settings = Settings()
