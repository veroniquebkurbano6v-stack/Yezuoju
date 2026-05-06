#!/usr/bin/env python3
"""
StoryRag v2.0 后端主应用入口
提供文档检索、向量搜索和AI对话功能的REST API服务
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量配置（显式指定项目根目录的 .env 文件）
from src.utils.paths import get_project_root
project_root = get_project_root()
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

# 确保项目src和backend目录可被Python导入
from src.utils.paths import get_project_root, get_backend_dir
project_root = get_project_root()
backend_root = get_backend_dir()
sys.path.insert(0, str(project_root.joinpath("src")))  # 添加src到Python路径
sys.path.insert(0, str(backend_root))  # 添加backend到Python路径

# FastAPI相关导入
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import logging

# 配置日志记录器
import sys
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# 核心功能模块导入
from src.rag.mixed_retrieval import VectorRetriever  # 向量检索器

# Redis Session Manager（可选，用于企业级缓存）
try:
    from backend.app.core.session_manager_redis import redis_session_manager
    REDIS_AVAILABLE = True
    logger.info("✅ Redis Session Manager 可用")
except ImportError:
    redis_session_manager = None
    REDIS_AVAILABLE = False
    logger.warning("⚠️ Redis Session Manager 未安装，使用文件存储版本")

# API 路由导入 - 使用相对导入
from app.api.dialogs import router as dialogs_router  # 对话相关路由
from app.api.chat_router import chat_router  # 聊天相关路由（PDF选择和查询功能）

# 请求数据模型定义
class QueryRequest(BaseModel):
    """查询请求数据模型"""
    question: str  # 用户问题
    top_k: int = 10  # 返回结果数量，默认为10


def create_app() -> FastAPI:
    """
    创建并配置 FastAPI 应用程序
    包含路由、中间件、事件处理器和依赖注入配置
    """
    sys.stderr.write("[DEBUG] 开始执行 create_app()\n")
    sys.stderr.flush()
    # 创建 FastAPI 应用实例，设置应用标题
    app = FastAPI(title="StoryRag Backend - Retrieval API")
    
    # 配置 CORS 中间件，允许跨域访问
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 允许所有域名访问
        allow_credentials=True,  # 允许携带凭据（cookies、认证头等）
        allow_methods=["*"],  # 允许所有 HTTP 方法
        allow_headers=["*"],  # 允许所有 HTTP 头
    )

    # 注册 API 路由 - 先注册 API 路由，再挂载静态文件
    # 避免静态文件路由覆盖 API 路由的问题
    sys.stderr.write(f"[DEBUG] 注册 dialogs_router，前缀：/api/dialogs，路由数：{len(dialogs_router.routes)}\n")
    sys.stderr.flush()
    logger.info(f"注册 dialogs_router，前缀：/api/dialogs，路由数：{len(dialogs_router.routes)}")
    app.include_router(dialogs_router, prefix="/api/dialogs")  # 注册对话相关路由，添加/api/dialogs 前缀
    sys.stderr.write(f"[DEBUG] 注册 chat_router，前缀：/api/chat，路由数：{len(chat_router.routes)}\n")
    sys.stderr.flush()
    logger.info(f"注册 chat_router，前缀：/api/chat，路由数：{len(chat_router.routes)}")
    app.include_router(chat_router, prefix="/api/chat")  # 注册聊天路由，添加/api/chat 前缀

    # 应用启动事件处理器 - 初始化所有重量级组件
    @app.on_event("startup")
    async def startup_event():
        """
        应用启动时的初始化事件
        包含向量检索系统、LangChain 工具、DeepSeek 代理等组件的初始化
        """
        logger.info("="*60)
        logger.info("开始执行应用启动初始化...")
        logger.info("="*60)
        
        # 从环境变量读取配置，设置默认值
        from src.utils.paths import get_vector_db_dir
        db_path = os.getenv("VECTOR_DB_PATH")
        if not db_path:
            # 如果没有设置环境变量，使用项目根目录的相对路径
            db_path = str(get_vector_db_dir())
        else:
            # 如果是相对路径，转换为绝对路径
            db_path = os.path.abspath(db_path)
        
        embedding_model = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
        logger.info(f"启动向量检索系统，db_path={db_path} embedding_model={embedding_model}")
            
        # 初始化向量检索器
        try:
            logger.info("开始初始化 VectorRetriever...")
            # 创建向量检索器实例
            app.state.vector_retriever = VectorRetriever(db_path)
            logger.info("✅ VectorRetriever 初始化成功")
        except Exception as e:
            logger.error(f"❌ 无法初始化 VectorRetriever: {e}", exc_info=True)
            app.state.vector_retriever = None  # 设置为 None 以避免后续使用时出错

        # 初始化 LangChain 工具（使用现有工具类）
        try:
            logger.info("开始初始化 LangChain tools...")
            from src.tools.langchain_retrieval_tools import SmartRetrievalTool
            from langchain_core.tools import Tool
                    
            # 创建工具实例
            logger.info("创建 SmartRetrievalTool 实例...")
            smart_retrieval = SmartRetrievalTool()
            logger.info("✅ SmartRetrievalTool 创建成功")
                    
            app.state.langchain_tools = [
                Tool(
                    name="smart_retrieval",
                    func=smart_retrieval._run,
                    description="智能检索工具，根据问题类型自动选择最优检索策略"
                )
            ]
            logger.info(f"✅ LangChain tools 已初始化，共 {len(app.state.langchain_tools)} 个工具")
        except Exception as e:
            logger.error(f"❌ LangChain tools 初始化失败：{e}", exc_info=True)
            app.state.langchain_tools = None
                
        # 预初始化 DeepSeek 智能体服务（唯一入口）
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if api_key:
            try:
                logger.info("开始预初始化 DeepSeek 智能体服务...")
                from app.agents.agent_service import DeepSeekAgentService
                from app.core.config import settings
                                
                # 创建智能体服务实例并存储到 app.state
                app.state.agent_service = DeepSeekAgentService(
                    vector_db_path=settings.VECTOR_DB_PATH,
                    api_key=settings.DEEPSEEK_API_KEY,
                    base_url=settings.DEEPSEEK_BASE_URL,
                    embedding_model=settings.EMBEDDING_MODEL,
                    role_id=os.getenv("DEFAULT_ROLE_ID", "humorous_butler"),
                )
                logger.info("✅ DeepSeek 智能体服务预初始化完成")
            except Exception as e:
                logger.error(f"❌ DeepSeek 智能体服务初始化失败：{e}", exc_info=True)
                app.state.agent_service = None
        else:
            logger.warning("⚠️  未配置 DEEPSEEK_API_KEY，跳过智能体服务初始化")
            app.state.agent_service = None
        
        # 初始化 Redis Session Manager（如果可用）
        global REDIS_AVAILABLE
        if REDIS_AVAILABLE:
            try:
                logger.info("开始初始化 Redis Session Manager...")
                await redis_session_manager.connect()
                stats = await redis_session_manager.get_stats()
                app.state.redis_session_manager = redis_session_manager
                logger.info(f"✅ Redis Session Manager 已连接 - 版本：{stats['redis_version']}, 内存：{stats['used_memory_human']}")
            except Exception as e:
                logger.error(f"❌ Redis Session Manager 连接失败：{e}", exc_info=True)
                app.state.redis_session_manager = None
                REDIS_AVAILABLE = False
        else:
            logger.info("使用文件存储版 Session Manager")
            app.state.redis_session_manager = None

    # 应用关闭事件处理器
    @app.on_event("shutdown")
    async def shutdown_event():
        """应用停止时的清理事件"""
        logger.info("应用停止，释放资源")
        
        # 关闭 Redis 连接
        if REDIS_AVAILABLE and app.state.redis_session_manager:
            try:
                await app.state.redis_session_manager.close()
                logger.info("✅ Redis Session Manager 已关闭")
            except Exception as e:
                logger.error(f"❌ 关闭 Redis Session Manager 失败：{e}")

    # 依赖注入提供者函数
    def get_vector_retriever():
        """
        向量检索器依赖注入函数
        返回已初始化的 vector_retriever（如果存在），避免重复初始化
        """
        return app.state.vector_retriever

    def get_agent():
        """
        DeepSeek 代理依赖注入函数
        如果代理未初始化或不存在则返回 None，避免 AttributeError
        """
        return getattr(app.state, "agent", None)
        
    def get_redis_session_manager():
        """
        Redis Session Manager 依赖注入函数
        如果 Redis 未初始化则返回 None
        """
        return getattr(app.state, "redis_session_manager", None) if REDIS_AVAILABLE else None

    # 核心查询 API 端点 - 已移至 chat_router 中实现
    # @app.post("/api/query")
    # async def query(req: QueryRequest, vector_retriever = Depends(get_vector_retriever), agent = Depends(get_agent)):
    #     """
    #     文档检索查询接口
    #     接收用户问题，返回相关的文档片段和检索结果
    #     """
    #     # 检查向量检索器是否已正确初始化
    #     if vector_retriever is None:
    #         return {"error": "Vector retriever not initialized"}
    #    
    #     # TODO: 使用 vector_retriever 实现查询逻辑
    #     # 目前返回一个占位响应
    #     return {
    #         "question": req.question,
    #         "results": [],
    #         "message": "Query endpoint needs implementation with vector_retriever"
    #     }

    # 挂载前端静态文件 - 最后执行，避免覆盖API路由
    from src.utils.paths import get_frontend_dir
    frontend_path = get_frontend_dir() / "dist"
    if frontend_path.exists():
        # 挂载静态文件，启用HTML模式，支持SPA路由
        app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="static")
    else:
        logger.warning(f"前端静态文件目录不存在: {frontend_path}")

    return app


# 创建FastAPI应用实例
app = create_app()

# 主程序入口点
if __name__ == "__main__":
    import uvicorn
    
    # 使用uvicorn启动ASGI服务器
    # 直接传入app实例，而不是模块路径，这样更直接和高效
    uvicorn.run(
        app, 
        host="0.0.0.0",  # 监听所有网络接口
        port=int(os.getenv("PORT", 8000))  # 从环境变量读取端口，默认8000
    )


