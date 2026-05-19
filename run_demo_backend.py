#!/usr/bin/env python3
"""
StoryRag 简化后端 - 用于云端演示
"""
import os
import sys
from pathlib import Path
from datetime import datetime

# 设置项目路径
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / "backend"))
sys.path.insert(0, str(project_root / "src"))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# 从 env.example 读取默认配置
if not os.path.exists(".env"):
    print("📝 创建默认 .env 文件...")
    with open("env.example", "r") as f:
        with open(".env", "w") as f2:
            f2.write(f.read())

# 读取环境变量
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173")

# 创建应用
app = FastAPI(title="StoryRag Backend (Demo Mode)")

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS.split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
)

# 健康检查
@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "backend": "running",
            "vector_db": "demo_mode",
            "agent_service": "demo_mode",
            "redis_session_manager": "demo_mode"
        }
    }

# 测试端点
@app.get("/api/chat/pdfs", tags=["PDFs"])
async def get_pdfs():
    """返回演示用的 PDF 列表"""
    return [
        {
            "language": "Chinese",
            "files": ["安徒生童话.pdf", "洪武：朱元璋的成与败.pdf", "鲁迅短篇小说集：呐喊.pdf"]
        },
        {
            "language": "English",
            "files": ["Spider-Man：Homecoming.pdf"]
        },
        {
            "language": "Japanese",
            "files": ["日语化物语.pdf"]
        }
    ]

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "StoryRag Backend is running!",
        "docs": "/docs"
    }

# 挂载前端文件（如果存在）
frontend_dist = project_root / "frontend" / "dist"
if frontend_dist.exists():
    print(f"📂 挂载前端静态文件: {frontend_dist}")
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("🚀 StoryRag 后端服务启动")
    print("=" * 60)
    print(f"📍 项目根目录: {project_root}")
    print(f"✅ API 地址: http://0.0.0.0:8000")
    print(f"📚 API 文档: http://0.0.0.0:8000/docs")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
