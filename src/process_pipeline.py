#!/usr/bin/env python3
"""
统一处理流水线：pdf -> json -> vectors
支持三种模式：json-only / vector-only / full
"""
import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(description="PDF 处理流水线")
    parser.add_argument("--mode", choices=["json-only", "vector-only", "full"], default="json-only")
    parser.add_argument("--source-dir", default="src/data/source", help="PDF 源目录")
    parser.add_argument("--json-dir", default="src/data/pages_title", help="JSON 输出/输入目录")
    parser.add_argument("--db-path", default="src/data/vector_database", help="ChromaDB 存储路径")
    parser.add_argument("--batch-size", type=int, default=64, help="向量化时的批处理大小")
    parser.add_argument("--embedding-model", default="intfloat/multilingual-e5-large", help="嵌入模型名称")
    args = parser.parse_args()

    mode = args.mode

    if mode in ("json-only", "full"):
        # 调用 identify_title.main 来生成 JSON（会跳过已有缓存）
        try:
            import identify_title
            logger.info("开始生成 JSON（identify_title）...")
            # identify_title.main 会扫描 src/data/source 并写入 src/data/pages_title
            identify_title.main()
            logger.info("JSON 生成完成")
        except Exception as e:
            logger.error(f"生成 JSON 失败: {e}")
            if mode == "json-only":
                return

    if mode in ("vector-only", "full"):
        try:
            from embedding_vector import ingest_json_directory_to_chroma
            json_dir = args.json_dir
            db_path = args.db_path
            logger.info(f"开始将 JSON 导入到 ChromaDB：{json_dir} -> {db_path}")
            logger.info(f"使用嵌入模型: {args.embedding_model}")
            ingest_json_directory_to_chroma(json_dir, db_path=db_path, batch_size=args.batch_size, model_name=args.embedding_model)
            logger.info("JSON 导入完成")
        except Exception as e:
            logger.error(f"导入 JSON 到 ChromaDB 失败: {e}")

if __name__ == "__main__":
    main()


