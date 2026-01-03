"""
工具函数模块
提供通用的辅助功能
"""

import os
import time
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

def get_current_timestamp(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    获取当前时间戳
    
    Args:
        format: 时间戳格式，默认"%Y-%m-%d %H:%M:%S"
    
    Returns:
        格式化的时间戳字符串
    """
    return datetime.now().strftime(format)

def validate_api_key(api_key: str) -> bool:
    """
    验证API密钥格式是否正确
    
    Args:
        api_key: 待验证的API密钥
    
    Returns:
        布尔值，表示API密钥格式是否正确
    """
    if not api_key:
        return False
    
    # DeepSeek API密钥格式：sk-开头，后跟32个字符
    return api_key.startswith("sk-") and len(api_key) == 36

def ensure_directory_exists(directory_path: str) -> bool:
    """
    确保目录存在，如果不存在则创建
    
    Args:
        directory_path: 目录路径
    
    Returns:
        布尔值，表示目录是否存在或创建成功
    """
    try:
        os.makedirs(directory_path, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"创建目录失败: {directory_path}, 错误: {str(e)}")
        return False

def format_error_message(error: Exception, include_traceback: bool = False) -> str:
    """
    格式化错误信息
    
    Args:
        error: 异常对象
        include_traceback: 是否包含堆栈跟踪
    
    Returns:
        格式化的错误信息字符串
    """
    if include_traceback:
        import traceback
        return f"{str(error)}\n\n堆栈跟踪:\n{traceback.format_exc()}"
    return str(error)

def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    截断字符串到指定长度
    
    Args:
        text: 待截断的字符串
        max_length: 最大长度
        suffix: 截断后添加的后缀
    
    Returns:
        截断后的字符串
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def extract_file_extension(filename: str) -> str:
    """
    提取文件名的扩展名
    
    Args:
        filename: 文件名
    
    Returns:
        文件扩展名（不含点号）
    """
    return os.path.splitext(filename)[1][1:].lower()

def is_pdf_file(filename: str) -> bool:
    """
    判断是否为PDF文件
    
    Args:
        filename: 文件名
    
    Returns:
        布尔值，表示是否为PDF文件
    """
    return extract_file_extension(filename) == "pdf"

def calculate_similarity(text1: str, text2: str) -> float:
    """
    计算两个文本的相似度（简单实现）
    
    Args:
        text1: 第一个文本
        text2: 第二个文本
    
    Returns:
        相似度分数，范围0-1
    """
    # 简单的Jaccard相似度实现
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 and not words2:
        return 1.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union)

def retry_on_failure(max_retries: int = 3, delay_seconds: float = 1.0, 
                     retry_exceptions: tuple = (Exception,)):
    """
    重试装饰器
    
    Args:
        max_retries: 最大重试次数
        delay_seconds: 重试间隔（秒）
        retry_exceptions: 需要重试的异常类型
    
    Returns:
        装饰器函数
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except retry_exceptions as e:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(f"函数 {func.__name__} 执行失败，{attempt + 1}/{max_retries} 重试... 错误: {str(e)}")
                    time.sleep(delay_seconds * (2 ** attempt))  # 指数退避
        return wrapper
    return decorator

def filter_dict_by_keys(dictionary: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    """
    根据指定的键过滤字典
    
    Args:
        dictionary: 待过滤的字典
        keys: 需要保留的键列表
    
    Returns:
        过滤后的字典
    """
    return {key: dictionary[key] for key in keys if key in dictionary}

def safe_get(dictionary: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    安全地获取字典中的值
    
    Args:
        dictionary: 字典对象
        key: 键名
        default: 默认值
    
    Returns:
        字典中的值或默认值
    """
    return dictionary.get(key, default)

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    安全除法，避免除以零
    
    Args:
        numerator: 分子
        denominator: 分母
        default: 分母为零时的默认值
    
    Returns:
        除法结果或默认值
    """
    if denominator == 0:
        return default
    return numerator / denominator

def convert_size(size_bytes: int) -> str:
    """
    将字节大小转换为人类可读的格式
    
    Args:
        size_bytes: 字节大小
    
    Returns:
        人类可读的大小字符串
    """
    import math
    if size_bytes == 0:
        return "0B"
    
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    
    return f"{s} {size_name[i]}"

def get_file_size(file_path: str) -> Optional[int]:
    """
    获取文件大小（字节）
    
    Args:
        file_path: 文件路径
    
    Returns:
        文件大小（字节）或None（如果文件不存在或发生错误）
    """
    try:
        if os.path.exists(file_path):
            return os.path.getsize(file_path)
        return None
    except Exception as e:
        logger.error(f"获取文件大小失败: {file_path}, 错误: {str(e)}")
        return None
