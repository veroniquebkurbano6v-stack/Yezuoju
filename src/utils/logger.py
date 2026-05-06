"""
统一日志配置

职责：
1. 统一管理日志配置
2. 支持不同级别的日志输出
3. 提供便捷的日志获取函数

使用方式：
    from src.utils.logger import get_logger
    
    logger = get_logger(__name__)
    logger.info("这是一条信息")
"""
import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_format: Optional[str] = None,
    log_file: Optional[str] = None
) -> None:
    """
    配置全局日志系统
    
    Args:
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: 日志格式字符串
        log_file: 日志文件路径（可选）
    """
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # 清除现有的 handler（避免重复）
    root_logger.handlers.clear()
    
    # 控制台 handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # 文件 handler（如果指定）
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # 文件记录更详细的日志
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # 抑制第三方库的过多日志
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    获取命名日志记录器
    
    Args:
        name: 日志记录器名称（通常使用 __name__）
        
    Returns:
        Logger 实例
    """
    return logging.getLogger(name)


# 自动配置日志（导入时执行）
from src.config import settings
setup_logging(
    level=settings.LOG_LEVEL,
    log_format=settings.LOG_FORMAT
)


if __name__ == "__main__":
    """测试日志配置"""
    logger = get_logger(__name__)
    
    logger.debug("这是一条 DEBUG 日志")
    logger.info("这是一条 INFO 日志")
    logger.warning("这是一条 WARNING 日志")
    logger.error("这是一条 ERROR 日志")
    logger.critical("这是一条 CRITICAL 日志")
