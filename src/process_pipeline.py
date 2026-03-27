#!/usr/bin/env python3
"""
统一处理流水线：pdf -> json -> vectors
支持三种模式：json-only / vector-only / full

功能特色：
- 智能缓存：自动跳过已处理的 PDF 文件
- 强制重建：支持 --force 参数清空向量数据库
- 详细日志：实时显示处理进度和统计信息
- 错误处理：完善的异常捕获和友好的错误提示

处理流程：
1. new_identify_title.py - 识别 PDF 章节标题，生成标题 JSON
2. concatenate_text_blocks.py - 拼接文本块，生成初始 chunks（含摘要）
3. ingest_embeddings.py - 为 chunks 生成嵌入向量
4. new_embedding_vector.py - 将 chunks 导入向量数据库
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import subprocess
from datetime import datetime

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# === 路径配置 ===
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = SRC_DIR / "data"
SOURCE_DIR = DATA_DIR / "source"  # PDF 源文件
TITLES_DIR = DATA_DIR / "pages_title"  # 标题 JSON 输出
CHUNKS_DIR = DATA_DIR / "chunks"  # 文本块 JSON
VECTOR_DB_DIR = DATA_DIR / "vector_database"  # 向量数据库

# === 脚本路径 ===
SCRIPTS = {
    "identify_title": SRC_DIR / "new_identify_title.py",
    "concatenate": SRC_DIR / "concatenate_text_blocks.py",
    "embeddings": SRC_DIR / "ingest_embeddings.py",
    "vector_db": SRC_DIR / "new_embedding_vector.py"
}


class ProcessingPipeline:
    """PDF 处理流水线"""
    
    def __init__(self, mode: str = "full", force: bool = False, book_name: Optional[str] = None):
        """
        初始化流水线
        
        Args:
            mode: 处理模式 ("full", "json-only", "vector-only")
            force: 是否强制覆盖已有数据
            book_name: 指定处理的书籍名称（None 表示处理所有）
        """
        self.mode = mode
        self.force = force
        self.book_name = book_name
        
        # 验证脚本存在
        self._validate_scripts()
        
        logger.info(f"🚀 PDF 处理流水线初始化完成")
        logger.info(f"   模式：{mode}")
        logger.info(f"   强制模式：{force}")
        if book_name:
            logger.info(f"   指定书籍：《{book_name}》")
    
    def _validate_scripts(self):
        """验证所有必需的脚本是否存在"""
        missing_scripts = []
        for name, script_path in SCRIPTS.items():
            if not script_path.exists():
                missing_scripts.append(f"{name}: {script_path}")
        
        if missing_scripts:
            raise FileNotFoundError(
                f"缺少必需的脚本：\n" +
                "\n".join([f"  - {s}" for s in missing_scripts]) +
                "\n请确保所有脚本都在 src/ 目录下"
            )
    
    def run_step(self, step_name: str, script_path: Path, extra_args: List[str] = None) -> bool:
        """
        运行单个处理步骤
        
        Args:
            step_name: 步骤名称
            script_path: 脚本路径
            extra_args: 额外的命令行参数
            
        Returns:
            bool: 是否成功
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"⚙️  开始执行：{step_name}")
        logger.info(f"{'='*60}")
        
        # 构建命令
        cmd = [sys.executable, str(script_path)]
        
        if extra_args:
            cmd.extend(extra_args)
        
        logger.info(f"📋 执行命令：{' '.join(cmd)}")
        
        try:
            # 运行脚本
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=False,
                text=True
            )
            
            logger.info(f"✅ {step_name} 执行成功")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ {step_name} 执行失败：{e}")
            return False
        except Exception as e:
            logger.error(f"❌ {step_name} 发生异常：{e}")
            return False
    
    def run_full_pipeline(self) -> bool:
        """
        运行完整流水线
        
        Returns:
            bool: 是否成功
        """
        logger.info("\n" + "="*60)
        logger.info("🎯 开始完整处理流程")
        logger.info("="*60)
        start_time = datetime.now()
        
        # 步骤 1: 识别标题
        if not self.run_step(
            "步骤 1/4: 识别 PDF 章节标题",
            SCRIPTS["identify_title"],
            ["--book", self.book_name] if self.book_name else []
        ):
            return False
        
        # 步骤 2: 拼接文本块
        if not self.run_step(
            "步骤 2/4: 拼接文本块并生成摘要",
            SCRIPTS["concatenate"],
            ["--book", self.book_name] if self.book_name else []
        ):
            return False
        
        # 步骤 3: 生成嵌入向量
        if not self.run_step(
            "步骤 3/4: 生成嵌入向量",
            SCRIPTS["embeddings"],
            ["--book", self.book_name] if self.book_name else []
        ):
            return False
        
        # 步骤 4: 导入向量数据库
        if not self.run_step(
            "步骤 4/4: 导入向量数据库",
            SCRIPTS["vector_db"],
            [
                "--input-dir", str(CHUNKS_DIR),
                "--db-path", str(VECTOR_DB_DIR),
                "--book", self.book_name
            ] if self.book_name else [
                "--input-dir", str(CHUNKS_DIR),
                "--db-path", str(VECTOR_DB_DIR)
            ]
        ):
            return False
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("\n" + "="*60)
        logger.info(f"🎉 完整处理流程执行成功！")
        logger.info(f"⏱️  总耗时：{duration}")
        logger.info(f"="*60)
        
        return True
    
    def run_json_only(self) -> bool:
        """
        仅运行 JSON 生成流程（前两步）
        
        Returns:
            bool: 是否成功
        """
        logger.info("\n" + "="*60)
        logger.info("📝 开始 JSON 生成流程")
        logger.info("="*60)
        start_time = datetime.now()
        
        # 步骤 1: 识别标题
        if not self.run_step(
            "步骤 1/2: 识别 PDF 章节标题",
            SCRIPTS["identify_title"],
            ["--book", self.book_name] if self.book_name else []
        ):
            return False
        
        # 步骤 2: 拼接文本块
        if not self.run_step(
            "步骤 2/2: 拼接文本块并生成摘要",
            SCRIPTS["concatenate"],
            ["--book", self.book_name] if self.book_name else []
        ):
            return False
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("\n" + "="*60)
        logger.info(f"🎉 JSON 生成流程执行成功！")
        logger.info(f"⏱️  总耗时：{duration}")
        logger.info(f"📂 输出目录：{CHUNKS_DIR}")
        logger.info(f"="*60)
        
        return True
    
    def run_vector_only(self) -> bool:
        """
        仅运行向量化流程（后两步）
        
        Returns:
            bool: 是否成功
        """
        logger.info("\n" + "="*60)
        logger.info("🔢 开始向量化流程")
        logger.info("="*60)
        start_time = datetime.now()
        
        # 步骤 1: 生成嵌入向量
        if not self.run_step(
            "步骤 1/2: 生成嵌入向量",
            SCRIPTS["embeddings"],
            [
                "--book", self.book_name,
                "--force" if self.force else []
            ] if self.book_name else [
                "--force" if self.force else []
            ]
        ):
            return False
        
        # 步骤 2: 导入向量数据库
        if not self.run_step(
            "步骤 2/2: 导入向量数据库",
            SCRIPTS["vector_db"],
            [
                "--input-dir", str(CHUNKS_DIR),
                "--db-path", str(VECTOR_DB_DIR),
                "--book", self.book_name,
                "--force" if self.force else [],
                "--clear-db" if self.force else []
            ] if self.book_name else [
                "--input-dir", str(CHUNKS_DIR),
                "--db-path", str(VECTOR_DB_DIR),
                "--force" if self.force else [],
                "--clear-db" if self.force else []
            ]
        ):
            return False
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("\n" + "="*60)
        logger.info(f"🎉 向量化流程执行成功！")
        logger.info(f"⏱️  总耗时：{duration}")
        logger.info(f"💾 向量数据库：{VECTOR_DB_DIR}")
        logger.info(f"="*60)
        
        return True
    
    def run(self) -> bool:
        """
        根据模式运行对应的流程
        
        Returns:
            bool: 是否成功
        """
        if self.mode == "full":
            return self.run_full_pipeline()
        elif self.mode == "json-only":
            return self.run_json_only()
        elif self.mode == "vector-only":
            return self.run_vector_only()
        else:
            logger.error(f"未知模式：{self.mode}")
            return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="PDF 处理流水线 - 一键完成从 PDF 到向量数据库的全流程",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 完整处理所有 PDF
  python src/process_pipeline.py
  
  # 仅生成 JSON（标题识别 + 文本拼接）
  python src/process_pipeline.py --mode json-only
  
  # 仅向量化（嵌入向量 + 数据库导入）
  python src/process_pipeline.py --mode vector-only
  
  # 处理单本书籍
  python src/process_pipeline.py --book "洪武：朱元璋的成与败"
  
  # 强制覆盖所有已有数据
  python src/process_pipeline.py --force
  
  # 组合使用
  python src/process_pipeline.py --mode json-only --book "安徒生童话"
"""
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "json-only", "vector-only"],
        help="处理模式（默认：full）"
    )
    
    parser.add_argument(
        "--book",
        type=str,
        help="指定要处理的书籍名称（不含.pdf 扩展名）"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制覆盖已有数据（危险操作）"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅显示计划，不实际执行"
    )
    
    args = parser.parse_args()
    
    # 打印欢迎信息
    print("="*60)
    print("🚀 PDF 处理流水线")
    print("="*60)
    print(f"📂 项目根目录：{PROJECT_ROOT}")
    print(f"📁 PDF 源目录：{SOURCE_DIR}")
    print(f"📄 标题输出：{TITLES_DIR}")
    print(f"📊 文本块：{CHUNKS_DIR}")
    print(f"💾 向量数据库：{VECTOR_DB_DIR}")
    print("="*60)
    
    # 显示处理计划
    print(f"\n📋 处理计划:")
    print(f"   模式：{args.mode}")
    print(f"   强制模式：{args.force}")
    if args.book:
        print(f"   指定书籍：《{args.book}》")
    
    if args.dry_run:
        print("\n⚠️  [干跑模式] 不执行任何操作")
        return
    
    # 创建并运行流水线
    pipeline = ProcessingPipeline(
        mode=args.mode,
        force=args.force,
        book_name=args.book
    )
    
    success = pipeline.run()
    
    # 最终总结
    print("\n" + "="*60)
    if success:
        print("✅ 处理成功！")
    else:
        print("❌ 处理失败，请检查错误日志")
    print("="*60)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
