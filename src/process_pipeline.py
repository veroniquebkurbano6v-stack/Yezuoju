#!/usr/bin/env python3
"""
统一处理流水线：pdf -> json -> vectors
支持三种模式：json-only / vector-only / full

功能特性：
- 智能缓存：自动跳过已处理的 PDF 文件
- 强制重建：支持 --force 参数清空向量数据库
- 详细日志：实时显示处理进度和统计信息
- 错误处理：完善的异常捕获和友好的错误提示
"""
import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 添加项目根目录到 sys.path，解决模块导入问题
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def reset_chroma_database(db_path: str):
    """
    重置 ChromaDB 数据库（清空所有数据）
    
    Args:
        db_path: 数据库路径
    """
    try:
        import chromadb
        from chromadb.config import Settings
        
        logger.info(f"正在重置 ChromaDB 数据库：{db_path}")
        client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(allow_reset=True, anonymized_telemetry=False)
        )
        client.reset()  # 重置整个数据库
        logger.info("✅ ChromaDB 数据库已重置")
    except Exception as e:
        logger.error(f"❌ 重置数据库失败：{e}")
        raise

def count_json_files(json_dir: str) -> int:
    """统计 JSON 文件数量"""
    return len(list(Path(json_dir).glob("**/*.json")))

def count_pdf_files(source_dir: str) -> int:
    """统计 PDF 文件数量"""
    return len(list(Path(source_dir).rglob("*.pdf")))

def main():
    parser = argparse.ArgumentParser(
        description="PDF 处理流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 仅生成 JSON（跳过已缓存的文件）
  python process_pipeline.py --mode json-only
  
  # 仅向量化（从 JSON 生成向量索引）
  python process_pipeline.py --mode vector-only
  
  # 完整流程（JSON + 向量）
  python process_pipeline.py --mode full
  
  # 强制重建（先清空向量数据库）
  python process_pipeline.py --mode full --force
  
  # 指定自定义路径
  python process_pipeline.py --mode full --source-dir ./my_pdfs --db-path ./my_db
        """
    )
    
    # 使用相对于项目根目录的绝对路径
    project_root = Path(__file__).parent.parent
    
    parser.add_argument(
        "--mode", 
        choices=["json-only", "vector-only", "full"], 
        default="json-only",
        help="处理模式：json-only（仅生成 JSON）/ vector-only（仅向量化）/ full（完整流程）"
    )
    parser.add_argument(
        "--source-dir", 
        default=str(project_root / "src" / "data" / "source"), 
        help=f"PDF 源目录（默认：{project_root / 'src' / 'data' / 'source'}）"
    )
    parser.add_argument(
        "--json-dir", 
        default=str(project_root / "src" / "data" / "pages_title"), 
        help=f"JSON 输出/输入目录（默认：{project_root / 'src' / 'data' / 'pages_title'}）"
    )
    parser.add_argument(
        "--db-path", 
        default="src/data/vector_database", 
        help="ChromaDB 存储路径（默认：src/data/vector_database）"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=64, 
        help="向量化时的批处理大小（默认：64）"
    )
    parser.add_argument(
        "--embedding-model", 
        default="intfloat/multilingual-e5-large", 
        help="嵌入模型名称（默认：intfloat/multilingual-e5-large）"
    )
    parser.add_argument(
        "--force", 
        action="store_true", 
        help="强制重建：先清空向量数据库（慎用！会删除所有已有数据）"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="预演模式：只显示将要执行的操作，不实际处理"
    )
    
    args = parser.parse_args()
    mode = args.mode
    
    # 打印配置信息
    logger.info("=" * 80)
    logger.info("📋 PDF 处理流水线配置")
    logger.info("=" * 80)
    logger.info(f"运行模式：{mode}")
    logger.info(f"PDF 源目录：{args.source_dir}")
    logger.info(f"JSON 目录：{args.json_dir}")
    logger.info(f"向量数据库路径：{args.db_path}")
    logger.info(f"批次大小：{args.batch_size}")
    logger.info(f"嵌入模型：{args.embedding_model}")
    logger.info(f"强制重建：{'是' if args.force else '否'}")
    logger.info(f"预演模式：{'是' if args.dry_run else '否'}")
    logger.info("=" * 80)
    
    # 预演模式：只显示配置
    if args.dry_run:
        logger.info("✅ 预演模式：配置检查完成，未执行任何实际操作")
        return
    
    total_start_time = time.time()
    stats = {
        "json_files_before": 0,
        "json_files_after": 0,
        "pdf_count": 0,
        "success": True,
        "errors": []
    }
    
    try:
        # 统计初始状态
        stats["pdf_count"] = count_pdf_files(args.source_dir)
        stats["json_files_before"] = count_json_files(args.json_dir)
        
        logger.info(f"📊 初始状态：{stats['pdf_count']} 个 PDF 文件，{stats['json_files_before']} 个 JSON 文件")
        logger.info("")
        
        # ========== 第一阶段：生成 JSON ==========
        if mode in ("json-only", "full"):
            logger.info("🚀 第一阶段：生成 JSON 文件")
            logger.info("-" * 80)
            stage_start = time.time()
            
            try:
                import identify_title
                identify_title.main()
                stage_elapsed = time.time() - stage_start
                logger.info(f"✅ JSON 生成完成（耗时：{stage_elapsed:.2f}秒）")
            except Exception as e:
                error_msg = f"生成 JSON 失败：{e}"
                logger.error(f"❌ {error_msg}")
                stats["errors"].append(error_msg)
                stats["success"] = False
                if mode == "json-only":
                    raise  # json-only 模式下直接抛出异常
            
            logger.info("")
            stats["json_files_after"] = count_json_files(args.json_dir)
            new_json_count = stats["json_files_after"] - stats["json_files_before"]
            logger.info(f"📈 JSON 文件变化：{stats['json_files_before']} → {stats['json_files_after']} (+{new_json_count})")
            logger.info("")
        
        # ========== 第二阶段：向量化 ==========
        if mode in ("vector-only", "full"):
            logger.info("🚀 第二阶段：向量化处理")
            logger.info("-" * 80)
            stage_start = time.time()
            
            # 强制重建：清空向量数据库
            if args.force:
                logger.warning("⚠️  检测到 --force 参数，将清空向量数据库...")
                reset_chroma_database(args.db_path)
                logger.info("")
            
            # 检查 JSON 文件是否存在
            json_count = count_json_files(args.json_dir)
            if json_count == 0:
                error_msg = f"在 {args.json_dir} 中未找到 JSON 文件，请先运行 json-only 模式"
                logger.error(f"❌ {error_msg}")
                stats["errors"].append(error_msg)
                stats["success"] = False
                raise ValueError(error_msg)
            
            logger.info(f"发现 {json_count} 个 JSON 文件")
            logger.info(f"开始将 JSON 导入到 ChromaDB：{args.json_dir} -> {args.db_path}")
            logger.info(f"使用嵌入模型：{args.embedding_model}")
            logger.info(f"批次大小：{args.batch_size}")
            logger.info("")
            
            try:
                from embedding_vector import ingest_json_directory_to_chroma
                ingest_json_directory_to_chroma(
                    args.json_dir, 
                    db_path=args.db_path, 
                    batch_size=args.batch_size, 
                    model_name=args.embedding_model
                )
                stage_elapsed = time.time() - stage_start
                logger.info(f"✅ JSON 导入完成（耗时：{stage_elapsed:.2f}秒）")
            except Exception as e:
                error_msg = f"导入 JSON 到 ChromaDB 失败：{e}"
                logger.error(f"❌ {error_msg}")
                stats["errors"].append(error_msg)
                stats["success"] = False
                raise  # 重新抛出异常以便在完整模式下继续执行
            
            logger.info("")
        
        # ========== 汇总统计 ==========
        total_elapsed = time.time() - total_start_time
        logger.info("=" * 80)
        logger.info("📊 处理完成汇总")
        logger.info("=" * 80)
        logger.info(f"总耗时：{total_elapsed:.2f}秒")
        logger.info(f"PDF 文件数：{stats['pdf_count']}")
        logger.info(f"JSON 文件数：{stats['json_files_before']} → {stats['json_files_after']}")
        
        if stats["success"]:
            logger.info("✅ 全部处理成功！")
        else:
            logger.warning("⚠️  部分操作失败:")
            for error in stats["errors"]:
                logger.warning(f"   - {error}")
        
        logger.info("=" * 80)
        
        # 如果有错误且不是完整模式，退出码设为 1
        if not stats["success"] and mode != "full":
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.error("\n❌ 用户中断操作")
        sys.exit(1)
    except Exception as e:
        total_elapsed = time.time() - total_start_time
        logger.error(f"\n❌ 处理失败（总耗时：{total_elapsed:.2f}秒）：{e}")
        sys.exit(1)

if __name__ == "__main__":
    main()


