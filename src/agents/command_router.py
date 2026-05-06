"""
命令路由器 - 基于配置的意图识别和路由

职责：
1. 根据用户问题类型路由到对应命令
2. 从配置文件加载命令定义
3. 提供可扩展的命令注册机制
"""
from __future__ import annotations

import logging
from typing import Optional, Dict, Any
from src.models.config_loader import get_config_loader

logger = logging.getLogger(__name__)


class CommandRouter:
    """
    命令路由器
    
    根据用户查询的意图，路由到合适的命令处理器
    """
    
    def __init__(self):
        """初始化命令路由器"""
        self.config_loader = get_config_loader()
        
        logger.info(f"CommandRouter 初始化完成")
    
    def get_command_config(self, command_name: str) -> Optional[Dict[str, Any]]:
        """
        获取指定命令的配置
        
        Args:
            command_name: 命令名称（如 "ask", "search", "summarize"）
            
        Returns:
            命令配置字典，未找到返回 None
        """
        return self.config_loader.get_command(command_name)
    
    def get_all_commands(self) -> list[Dict[str, Any]]:
        """获取所有可用的命令配置"""
        return self.config_loader.get_enabled_commands()
    
    def get_command_tools(self, command_name: str) -> list[str]:
        """
        获取指定命令所需的工具列表
        
        Args:
            command_name: 命令名称
            
        Returns:
            工具名称列表
        """
        command = self.config_loader.get_command(command_name)
        if command:
            return command.get("tools", [])
        return []


# 全局单例
command_router = CommandRouter()


def get_command_router() -> CommandRouter:
    """获取命令路由器实例"""
    return command_router
