"""
配置数据加载器

职责：
1. 从 JSON 文件加载命令和工具配置
2. 提供类型安全的访问接口
3. 支持动态重载
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from functools import lru_cache


class ConfigLoader:
    """配置数据加载器"""
    
    def __init__(self, reference_data_dir: str | Path = None):
        """
        初始化配置加载器
        
        Args:
            reference_data_dir: 参考数据目录路径
        """
        if reference_data_dir is None:
            # 默认路径：src/reference_data/
            self.base_dir = Path(__file__).resolve().parents[1] / "reference_data"
        else:
            self.base_dir = Path(reference_data_dir)
        
        self._commands_cache: list[dict] | None = None
        self._tools_cache: list[dict] | None = None
    
    def load_commands(self) -> list[dict[str, Any]]:
        """
        加载命令配置
        
        Returns:
            命令配置列表
        """
        if self._commands_cache is not None:
            return self._commands_cache
        
        commands_file = self.base_dir / "commands.json"
        if not commands_file.exists():
            raise FileNotFoundError(f"Commands config not found: {commands_file}")
        
        with open(commands_file, 'r', encoding='utf-8') as f:
            self._commands_cache = json.load(f)
        
        return self._commands_cache
    
    def load_tools(self) -> list[dict[str, Any]]:
        """
        加载工具配置
        
        Returns:
            工具配置列表
        """
        if self._tools_cache is not None:
            return self._tools_cache
        
        tools_file = self.base_dir / "tools.json"
        if not tools_file.exists():
            raise FileNotFoundError(f"Tools config not found: {tools_file}")
        
        with open(tools_file, 'r', encoding='utf-8') as f:
            self._tools_cache = json.load(f)
        
        return self._tools_cache
    
    def get_command(self, name: str) -> dict[str, Any] | None:
        """
        获取指定命令配置
        
        Args:
            name: 命令名称
        
        Returns:
            命令配置字典，未找到返回 None
        """
        commands = self.load_commands()
        for cmd in commands:
            if cmd.get("name") == name:
                return cmd
        return None
    
    def get_tool(self, name: str) -> dict[str, Any] | None:
        """
        获取指定工具配置
        
        Args:
            name: 工具名称
        
        Returns:
            工具配置字典，未找到返回 None
        """
        tools = self.load_tools()
        for tool in tools:
            if tool.get("name") == name:
                return tool
        return None
    
    def get_enabled_commands(self) -> list[dict[str, Any]]:
        """获取所有启用的命令"""
        commands = self.load_commands()
        return [cmd for cmd in commands if cmd.get("enabled", True)]
    
    def get_enabled_tools(self) -> list[dict[str, Any]]:
        """获取所有启用的工具"""
        tools = self.load_tools()
        return [tool for tool in tools if tool.get("enabled", True)]
    
    def reload(self):
        """重新加载所有配置（清除缓存）"""
        self._commands_cache = None
        self._tools_cache = None


# 全局配置加载器实例
config_loader = ConfigLoader()


@lru_cache(maxsize=1)
def get_config_loader() -> ConfigLoader:
    """
    获取配置加载器实例（带缓存）
    
    Returns:
        ConfigLoader 实例
    """
    return config_loader


# 便捷函数
def get_commands() -> list[dict]:
    """获取所有命令配置"""
    return config_loader.load_commands()


def get_tools() -> list[dict]:
    """获取所有工具配置"""
    return config_loader.load_tools()


def get_command(name: str) -> dict | None:
    """获取指定命令"""
    return config_loader.get_command(name)


def get_tool(name: str) -> dict | None:
    """获取指定工具"""
    return config_loader.get_tool(name)
