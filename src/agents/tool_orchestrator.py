"""
工具编排器 - 基于配置的工具链管理

职责：
1. 根据命令需求动态加载工具
2. 管理工具的启用/禁用状态
3. 提供工具元数据查询
"""
from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional
from src.models.config_loader import get_config_loader

logger = logging.getLogger(__name__)


class ToolOrchestrator:
    """
    工具编排器
    
    根据命令配置，动态组装所需的工具链
    """
    
    def __init__(self):
        """初始化工具编排器"""
        self.config_loader = get_config_loader()
        self._tool_registry: Dict[str, Any] = {}
        
        logger.info(f"ToolOrchestrator 初始化完成")
    
    def get_tools_for_command(self, command_name: str) -> List[Dict[str, Any]]:
        """
        获取指定命令所需的工具配置列表
        
        Args:
            command_name: 命令名称（如 "ask", "search", "summarize"）
            
        Returns:
            工具配置列表
        """
        # 获取命令配置
        command = self.config_loader.get_command(command_name)
        if not command:
            logger.warning(f"未找到命令: {command_name}")
            return []
        
        # 获取该命令需要的工具名称列表
        required_tool_names = command.get("tools", [])
        
        # 加载对应的工具配置
        tools_config = []
        for tool_name in required_tool_names:
            tool_config = self.config_loader.get_tool(tool_name)
            if tool_config and tool_config.get("enabled", True):
                tools_config.append(tool_config)
            else:
                logger.warning(f"工具 '{tool_name}' 不存在或已禁用")
        
        logger.info(f"命令 '{command_name}' 需要 {len(tools_config)} 个工具: {[t['name'] for t in tools_config]}")
        return tools_config
    
    def get_all_enabled_tools(self) -> List[Dict[str, Any]]:
        """获取所有启用的工具配置"""
        return self.config_loader.get_enabled_tools()
    
    def get_tool_metadata(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        获取工具的元数据
        
        Args:
            tool_name: 工具名称
            
        Returns:
            工具配置字典，未找到返回 None
        """
        return self.config_loader.get_tool(tool_name)
    
    def validate_tool_chain(self, command_name: str) -> Dict[str, Any]:
        """
        验证命令的工具链是否完整
        
        Args:
            command_name: 命令名称
            
        Returns:
            验证结果，包含是否有效和缺失的工具列表
        """
        command = self.config_loader.get_command(command_name)
        if not command:
            return {
                "valid": False,
                "error": f"命令 '{command_name}' 不存在",
                "missing_tools": []
            }
        
        required_tools = command.get("tools", [])
        missing_tools = []
        
        for tool_name in required_tools:
            tool = self.config_loader.get_tool(tool_name)
            if not tool or not tool.get("enabled", True):
                missing_tools.append(tool_name)
        
        return {
            "valid": len(missing_tools) == 0,
            "command": command_name,
            "required_tools": required_tools,
            "missing_tools": missing_tools,
            "available_tools": [t for t in required_tools if t not in missing_tools]
        }
    
    def register_tool_instance(self, tool_name: str, tool_instance: Any):
        """
        注册工具实例（用于将 LangChain Tool 对象与配置关联）
        
        Args:
            tool_name: 工具名称（与 tools.json 中的 name 对应）
            tool_instance: LangChain Tool 实例
        """
        self._tool_registry[tool_name] = tool_instance
        logger.info(f"注册工具实例: {tool_name}")
    
    def get_tool_instance(self, tool_name: str) -> Optional[Any]:
        """
        获取已注册的工具实例
        
        Args:
            tool_name: 工具名称
            
        Returns:
            LangChain Tool 实例，未找到返回 None
        """
        return self._tool_registry.get(tool_name)
    
    def build_tool_chain(self, command_name: str) -> List[Any]:
        """
        构建完整的工具链（返回 LangChain Tool 实例列表）
        
        Args:
            command_name: 命令名称
            
        Returns:
            LangChain Tool 实例列表
        """
        tools_config = self.get_tools_for_command(command_name)
        tool_instances = []
        
        for tool_config in tools_config:
            tool_name = tool_config["name"]
            tool_instance = self.get_tool_instance(tool_name)
            if tool_instance:
                tool_instances.append(tool_instance)
            else:
                logger.warning(f"工具 '{tool_name}' 未注册实例，跳过")
        
        logger.info(f"为命令 '{command_name}' 构建了 {len(tool_instances)} 个工具的链条")
        return tool_instances


# 全局单例
tool_orchestrator = ToolOrchestrator()


def get_tool_orchestrator() -> ToolOrchestrator:
    """获取工具编排器实例"""
    return tool_orchestrator
