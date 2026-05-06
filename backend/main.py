#!/usr/bin/env python3
"""
FastAPI后端服务入口文件
PDF智能检索系统API服务
"""

import os
import sys
import logging

os.environ.setdefault("PYTHONIOENCODING", "utf-8")


class SafeStreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            if hasattr(stream, 'encoding') and stream.encoding:
                msg = msg.encode(stream.encoding, errors='replace').decode(stream.encoding)
            stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            pass


_root_logger = logging.getLogger()
_root_logger.handlers.clear()
_root_logger.addHandler(SafeStreamHandler(sys.stdout))
_root_logger.setLevel(logging.INFO)

logger = logging.getLogger(__name__)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(
    title="PDF智能检索系统 API",
    description="基于RAG技术的PDF智能检索前后端系统",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:8080",
    os.getenv("FRONTEND_URL", "http://localhost:3000"),
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "message": "PDF智能检索系统API运行正常"}


@app.get("/")
async def root():
    return RedirectResponse(url="http://localhost:5173/")

@app.get("/api/system/status")
async def system_status():
    return {
        "status": "running",
        "version": "2.0.0",
        "service": "PDF智能检索系统",
        "timestamp": os.environ.get("TIMESTAMP", "N/A")
    }

from app.api.chat_router import chat_router

app.include_router(chat_router, prefix="/api/chat", tags=["chat"])

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    logger.info(f"启动FastAPI服务，地址: {host}:{port}")
    logger.info(f"  API 文档:   http://localhost:{port}/docs")
    logger.info(f"  健康检查:   http://localhost:{port}/api/health")
    logger.info(f"  前端页面:   http://localhost:5173/")
    uvicorn.run(app, host=host, port=port)
