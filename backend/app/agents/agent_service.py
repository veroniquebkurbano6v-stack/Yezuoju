"""
智能体服务层
封装DeepSeekRetrievalAgent，提供更简洁的API接口
"""

import os
import sys
import logging
from typing import Dict, Any, Optional

# 添加src目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..", "src")))

from deepseek_agent import DeepSeekRetrievalAgent

logger = logging.getLogger(__name__)

class DeepSeekAgentService:
    """
    DeepSeek智能体服务
    封装DeepSeekRetrievalAgent，提供更简洁的API接口
    """
    
    def __init__(self, vector_db_path: str, api_key: str = None, base_url: str = None, 
                 embedding_model: str = "intfloat/multilingual-e5-large"):
        """
        初始化智能体服务
        
        Args:
            vector_db_path: 向量数据库路径
            api_key: DeepSeek API密钥
            base_url: DeepSeek API基础URL
            embedding_model: 嵌入模型名称
        """
        self.vector_db_path = vector_db_path
        self.api_key = api_key
        self.base_url = base_url
        self.embedding_model = embedding_model
        
        self.agent = None
        
        # 初始化智能体
        self._initialize_agent()
    
    def _initialize_agent(self):
        """
        初始化DeepSeek检索代理
        """
        try:
            logger.info("正在初始化DeepSeek智能体服务...")
            
            self.agent = DeepSeekRetrievalAgent(
                vector_db_path=self.vector_db_path,
                api_key=self.api_key,
                base_url=self.base_url,
                embedding_model=self.embedding_model
            )
            
            logger.info("DeepSeek智能体服务初始化完成")
            
        except Exception as e:
            error_msg = f"DeepSeek智能体服务初始化失败: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
    
    def query(self, query: str, conversation_id: Optional[str] = None, 
              chat_history: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        发送查询，获取智能体回复
        
        Args:
            query: 用户查询内容
            conversation_id: 对话ID
            chat_history: 对话历史
        
        Returns:
            包含智能体回复的字典
        """
        try:
            logger.info(f"[AgentService.query] 接收查询请求")
            logger.info(f"[AgentService.query] 原始查询: '{query}'")
            logger.info(f"[AgentService.query] conversation_id: '{conversation_id}'")
            
            # 调用DeepSeekRetrievalAgent的chat方法
            result = self.agent.chat(user_input=query, chat_history=chat_history)
            
            if not result.get("success"):
                logger.error(f"智能体查询失败: {result.get('error')}")
                raise Exception(result.get("error", "智能体查询失败"))
            
            logger.info(f"智能体查询成功: answer='{result['answer'][:50]}...'")
            
            return result
            
        except Exception as e:
            error_msg = f"智能体服务查询失败: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
    
    def get_available_tools(self) -> Dict[str, Any]:
        """
        获取可用的工具列表
        
        Returns:
            包含可用工具信息的字典
        """
        try:
            return {
                "tools": self.agent.get_available_tools()
            }
        except Exception as e:
            logger.error(f"获取可用工具失败: {str(e)}")
            return {
                "tools": [],
                "error": str(e)
            }
    
    def test(self) -> Dict[str, Any]:
        """
        测试智能体服务是否正常工作
        
        Returns:
            包含测试结果的字典
        """
        try:
            logger.info("正在测试智能体服务...")
            
            # 测试基本查询
            test_query = "测试智能体服务是否正常工作"
            result = self.query(query=test_query)
            
            # 测试工具功能
            tools_test = self.agent.test_tools()
            
            return {
                "success": True,
                "query_test": {
                    "query": test_query,
                    "answer": result.get("answer", "").strip()[:100] + "..."
                },
                "tools_test": tools_test
            }
            
        except Exception as e:
            logger.error(f"智能体服务测试失败: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def health_check(self) -> Dict[str, Any]:
        """
        健康检查
        
        Returns:
            包含健康状态的字典
        """
        try:
            return {
                "status": "healthy",
                "agent_initialized": self.agent is not None
            }
        except Exception as e:
            logger.error(f"健康检查失败: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }


# 单例模式实现
_agent_service_instance = None

def get_agent_service_instance(vector_db_path: str, api_key: str = None, 
                              base_url: str = None, embedding_model: str = None) -> DeepSeekAgentService:
    """
    获取智能体服务单例实例
    
    Args:
        vector_db_path: 向量数据库路径
        api_key: DeepSeek API密钥
        base_url: DeepSeek API基础URL
        embedding_model: 嵌入模型名称
    
    Returns:
        DeepSeekAgentService实例
    """
    global _agent_service_instance
    
    if _agent_service_instance is None:
        _agent_service_instance = DeepSeekAgentService(
            vector_db_path=vector_db_path,
            api_key=api_key,
            base_url=base_url,
            embedding_model=embedding_model
        )
    
    return _agent_service_instance
