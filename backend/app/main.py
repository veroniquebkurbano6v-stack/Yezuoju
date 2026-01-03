#!/usr/bin/env python3
"""
StoryRag v2.0 后端主应用入口
提供文档检索、向量搜索和AI对话功能的REST API服务
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量配置
load_dotenv()

# 确保项目src和backend目录可被Python导入
project_root = Path(__file__).resolve().parents[2]  # 项目根目录: StoryRag v2.0
backend_root = Path(__file__).resolve().parent.parent  # backend目录
sys.path.insert(0, str(project_root.joinpath("src")))  # 添加src到Python路径
sys.path.insert(0, str(backend_root))  # 添加backend到Python路径

# FastAPI相关导入
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import logging

# 配置日志记录器
logger = logging.getLogger(__name__)

# 核心功能模块导入
from embedding_vector import VectorSearchSystem  # 向量检索系统
from deepseek_agent import create_deepseek_agent  # DeepSeek AI代理

# API路由导入 - 使用相对导入
from app.api.dialogs import router as dialogs_router  # 对话相关路由
from app.api.chat_router import chat_router  # 聊天相关路由（PDF选择和查询功能）

# 请求数据模型定义
class QueryRequest(BaseModel):
    """查询请求数据模型"""
    question: str  # 用户问题
    top_k: int = 10  # 返回结果数量，默认为10


def create_app() -> FastAPI:
    """
    创建并配置FastAPI应用程序
    包含路由、中间件、事件处理器和依赖注入配置
    """
    # 创建FastAPI应用实例，设置应用标题
    app = FastAPI(title="StoryRag Backend - Retrieval API")
    
    # 配置CORS中间件，允许跨域访问
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 允许所有域名访问
        allow_credentials=True,  # 允许携带凭据（cookies、认证头等）
        allow_methods=["*"],  # 允许所有HTTP方法
        allow_headers=["*"],  # 允许所有HTTP头
    )

    # 注册API路由 - 先注册API路由，再挂载静态文件
    # 避免静态文件路由覆盖API路由的问题
    app.include_router(dialogs_router)  # 注册对话相关路由
    app.include_router(chat_router, prefix="/api")  # 注册聊天路由，添加/api前缀

    # 应用启动事件处理器 - 初始化所有重量级组件
    @app.on_event("startup")
    async def startup_event():
        """
        应用启动时的初始化事件
        包含向量检索系统、LangChain工具、DeepSeek代理等组件的初始化
        """
        # 从环境变量读取配置，设置默认值
        db_path = os.getenv("VECTOR_DB_PATH", "src/data/vector_database")
        embedding_model = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
        logger.info(f"启动向量检索系统，db_path={db_path} embedding_model={embedding_model}")
        
        # 初始化向量检索系统
        try:
            # 创建向量检索系统实例
            app.state.vector_system = VectorSearchSystem(db_path)
            # 初始化检索器（加载必要的组件：嵌入模型、Reranker、元数据索引等）
            app.state.vector_system.initialize_retriever()
            logger.info("向量检索系统初始化成功")
        except Exception as e:
            logger.error(f"无法初始化向量检索系统: {e}")
            app.state.vector_system = None  # 设置为None以避免后续使用时出错

        # 初始化LangChain工具（复用VectorSearchSystem已初始化的组件，避免重复加载）
        try:
            from langchain_tools import get_langchain_tools
            # 复用已初始化的组件，避免重复创建EmbedddingModel和RerankerModel
            vector_system = app.state.vector_system
            if vector_system is not None:
                app.state.langchain_tools = get_langchain_tools(
                    db_path,
                    embedding_model,
                    vector_db=vector_system.vector_db,
                    embedding_model=vector_system.embedding_model,
                    reranker_model=vector_system.reranker_model
                )
            else:
                # 如果VectorSearchSystem初始化失败，创建独立的LangChain工具
                app.state.langchain_tools = get_langchain_tools(db_path, embedding_model)
            logger.info("LangChain tools 已初始化")
        except Exception as e:
            logger.error(f"LangChain tools 初始化失败: {e}")
            app.state.langchain_tools = None

        # 初始化DeepSeek代理（可选功能，需要API密钥）
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if api_key:
            try:
                app.state.agent = create_deepseek_agent(
                    db_path,
                    api_key=api_key,
                    base_url=os.getenv("DEEPSEEK_BASE_URL"),
                    tools_instance=app.state.langchain_tools
                )
                logger.info("DeepSeek agent 已初始化")
            except Exception as e:
                logger.warning(f"DeepSeek agent 初始化失败: {e}")
                app.state.agent = None
        else:
            logger.info("未配置DEEPSEEK_API_KEY，跳过DeepSeek agent初始化")
            app.state.agent = None

    # 应用关闭事件处理器
    @app.on_event("shutdown")
    async def shutdown_event():
        """应用停止时的清理事件"""
        logger.info("应用停止，释放资源")

    # 依赖注入提供者函数
    def get_vector_system():
        """
        向量检索系统依赖注入函数
        返回已初始化的vector system（如果存在），避免重复初始化
        """
        return app.state.vector_system

    def get_agent():
        """
        DeepSeek代理依赖注入函数
        如果代理未初始化或不存在则返回None，避免AttributeError
        """
        return getattr(app.state, "agent", None)

    # 核心查询API端点
    @app.post("/api/query")
    async def query(req: QueryRequest, vector_system = Depends(get_vector_system), agent = Depends(get_agent)):
        """
        文档检索查询接口
        接收用户问题，返回相关的文档片段和检索结果
        """
        # 检查向量检索系统是否已正确初始化
        if vector_system is None:
            return {"error": "Vector system not initialized"}

        # 执行向量搜索，使用Reranker进行重排序
        results = vector_system.search(req.question, top_k=req.top_k, use_reranker=True)
        
        # 将结果格式化为简单的字典格式，便于前端使用
        out = []
        for r in results:
            out.append({
                "text": r.text,  # 文档片段内容
                "section_title": r.section_title,  # 章节标题
                "page_number": r.page_number,  # 页码
                "score": float(r.score),  # 相似度分数
                "pdf_filename": r.pdf_filename,  # PDF文件名
                "chunk_id": r.chunk_id  # 文档块ID
            })

        return {
            "query": req.question,  # 原问题
            "results": out,  # 检索结果列表
            "count": len(out)  # 结果数量
        }

    # 挂载前端静态文件 - 最后执行，避免覆盖API路由
    frontend_path = Path(__file__).resolve().parents[2] / "frontend" / "dist"
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


