#!/usr/bin/env python3
"""
FastAPI后端服务入口文件
PDF智能检索系统API服务
"""

import os
import sys
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="PDF智能检索系统 API",
    description="基于RAG技术的PDF智能检索前后端系统",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 配置CORS
origins = [
    "http://localhost:3000",  # 前端开发服务器地址
    "http://localhost:5173",  # Vite默认地址
    "http://localhost:5174",  # Vite备用地址
    "http://localhost:8080",  # 可能的前端生产服务器地址
    os.getenv("FRONTEND_URL", "http://localhost:3000"),  # 从环境变量获取前端地址
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 健康检查端点
@app.get("/api/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy", "message": "PDF智能检索系统API运行正常"}

# 系统状态检查端点
@app.get("/api/system/status")
async def system_status():
    """系统状态检查端点"""
    return {
        "status": "running",
        "version": "2.0.0",
        "service": "PDF智能检索系统",
        "timestamp": os.environ.get("TIMESTAMP", "N/A")
    }

# 导入路由
from app.api.chat_router import chat_router

# 注册路由
app.include_router(chat_router, prefix="/api/chat", tags=["chat"])

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    logger.info(f"启动FastAPI服务，地址: {host}:{port}")
    uvicorn.run(app, host=host, port=port)
