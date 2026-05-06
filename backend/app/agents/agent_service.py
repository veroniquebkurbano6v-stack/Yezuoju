"""
智能体服务层
封装DeepSeekRetrievalAgent，提供更简洁的API接口
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.agents.deepseek_agent import DeepSeekRetrievalAgent
from src.models import RAGResponse

logger = logging.getLogger(__name__)

class DeepSeekAgentService:
    """
    DeepSeek智能体服务
    封装DeepSeekRetrievalAgent，提供更简洁的API接口
    """

    def __init__(self, vector_db_path: str, api_key: str = None, base_url: str = None, 
                 embedding_model: str = "intfloat/multilingual-e5-large",
                 role_id: str = None):
        """


        Args:
            vector_db_path: 向量数据库路径
            api_key: DeepSeek API密钥
            base_url: DeepSeek API基础URL
            embedding_model: 嵌入模型名称
            role_id: 角色标识符，如 'humorous_butler'

        Raises:
            ValueError: 当 role_id 不在有效角色列表中时
        """
        self._validate_role_id(role_id)

        self.vector_db_path = vector_db_path
        self.api_key = api_key
        self.base_url = base_url
        self.embedding_model = embedding_model
        self.role_id = role_id
        
        self.agent = None
        
        self._initialize_agent()

    @staticmethod
    def _validate_role_id(role_id):
        if role_id is None:
            return
        from src.core.role_profile import get_role, BUILT_IN_ROLES
        profile = get_role(role_id)
        if profile.role_id != role_id:
            raise ValueError(
                f"无效的角色ID: '{role_id}'。"
                f"可用内置角色: {', '.join(sorted(BUILT_IN_ROLES.keys()))}"
            )

    def _initialize_agent(self):
        """初始化DeepSeek检索代理"""
        try:
            logger.info("正在初始化DeepSeek智能体服务...")
            
            self.agent = DeepSeekRetrievalAgent(
                vector_db_path=self.vector_db_path,
                api_key=self.api_key,
                base_url=self.base_url,
                embedding_model=self.embedding_model,
                role_id=self.role_id,
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
            包含智能体回复的字典（兼容旧接口）
        """
        try:
            logger.info(f"[AgentService.query] 接收查询请求")
            logger.info(f"[AgentService.query] 原始查询: '{query}'")
            logger.info(f"[AgentService.query] conversation_id: '{conversation_id}'")
            
            # 调用DeepSeekRetrievalAgent的chat方法，返回 RAGResponse 对象
            rag_response: RAGResponse = self.agent.chat(
                user_input=query, 
                chat_history=chat_history,
                session_id=conversation_id or ""
            )
            
            logger.info(f"智能体查询成功: answer='{rag_response.answer[:50]}...'")
            
            # 转换为字典格式（保持向后兼容）
            return {
                "success": True,
                "answer": rag_response.answer,
                "retrieved_docs": [
                    {
                        "rank": idx + 1,
                        "document": doc.content,
                        "metadata": doc.metadata,
                        "score": doc.score
                    }
                    for idx, doc in enumerate(rag_response.sources)
                ],
                "confidence": rag_response.confidence,
                "session_id": rag_response.session_id,
                "usage_tokens": rag_response.usage_tokens,
                "chat_history": chat_history or []
            }
            
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
