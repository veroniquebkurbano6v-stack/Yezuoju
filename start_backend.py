#!/usr/bin/env python3
"""
StoryRag 后端服务启动脚本
"""
import sys
from pathlib import Path

# 设置项目路径
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "backend"))

# 导入并启动应用
from app.main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
