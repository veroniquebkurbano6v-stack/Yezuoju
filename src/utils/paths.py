"""
项目路径管理工具

职责：
1. 统一管理项目路径
2. 避免硬编码路径
3. 提供便捷的路径访问函数

使用方式：
    from src.utils.paths import get_project_root, get_data_dir
    
    project_root = get_project_root()
    data_dir = get_data_dir()
"""
from pathlib import Path
from functools import lru_cache


@lru_cache(maxsize=1)
def get_project_root() -> Path:
    """
    获取项目根目录
    
    Returns:
        项目根目录的绝对路径
    """
    # 此文件位于 src/utils/paths.py
    # parents[2] 即为项目根目录
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def get_src_dir() -> Path:
    """获取 src 目录"""
    return get_project_root() / "src"


@lru_cache(maxsize=1)
def get_data_dir() -> Path:
    """获取 data 目录"""
    return get_src_dir() / "data"


@lru_cache(maxsize=1)
def get_source_dir() -> Path:
    """获取 PDF 源文件目录"""
    return get_data_dir() / "source"


@lru_cache(maxsize=1)
def get_vector_db_dir() -> Path:
    """获取向量数据库目录"""
    return get_data_dir() / "vector_database"


@lru_cache(maxsize=1)
def get_chunks_dir() -> Path:
    """获取文本块目录"""
    return get_data_dir() / "chunks"


@lru_cache(maxsize=1)
def get_titles_dir() -> Path:
    """获取标题 JSON 目录"""
    return get_data_dir() / "pages_title"


@lru_cache(maxsize=1)
def get_backend_dir() -> Path:
    """获取 backend 目录"""
    return get_project_root() / "backend"


@lru_cache(maxsize=1)
def get_frontend_dir() -> Path:
    """获取 frontend 目录"""
    return get_project_root() / "frontend"


@lru_cache(maxsize=1)
def get_cache_dir() -> Path:
    """获取缓存目录"""
    cache_dir = get_project_root() / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@lru_cache(maxsize=1)
def get_session_dir() -> Path:
    """获取会话数据目录"""
    session_dir = get_project_root() / "src" / "data" / "sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def ensure_directory(path: Path) -> Path:
    """
    确保目录存在，如果不存在则创建
    
    Args:
        path: 目录路径
        
    Returns:
        目录路径
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


# 导出所有路径函数
__all__ = [
    'get_project_root',
    'get_src_dir',
    'get_data_dir',
    'get_source_dir',
    'get_vector_db_dir',
    'get_chunks_dir',
    'get_titles_dir',
    'get_backend_dir',
    'get_frontend_dir',
    'get_cache_dir',
    'get_session_dir',
    'ensure_directory'
]


if __name__ == "__main__":
    """测试路径函数"""
    print("项目路径信息：")
    print(f"  项目根目录: {get_project_root()}")
    print(f"  src 目录: {get_src_dir()}")
    print(f"  data 目录: {get_data_dir()}")
    print(f"  PDF 源文件: {get_source_dir()}")
    print(f"  向量数据库: {get_vector_db_dir()}")
    print(f"  文本块: {get_chunks_dir()}")
    print(f"  标题 JSON: {get_titles_dir()}")
    print(f"  backend: {get_backend_dir()}")
    print(f"  frontend: {get_frontend_dir()}")
    print(f"  cache: {get_cache_dir()}")
    print(f"  session_data: {get_session_dir()}")
