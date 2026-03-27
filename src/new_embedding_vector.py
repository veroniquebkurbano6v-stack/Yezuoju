#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
向量数据库导入脚本（精简版）

读取 concatenate_text_blocks.py 生成的 chunks JSON 文件，
将已有的 embedding 向量和元数据直接存入 ChromaDB 向量数据库。

JSON 格式要求：
[
    {
        "id": "安徒生童话_打火匣_1_8_0",
        "embedding": [0.1, 0.2, ...],  # 已有向量
        "document": "摘要文本",
        "metadata": {
            "source": "安徒生童话.pdf",
            "chapter": "打火匣",
            "start_page": 1,
            "end_page": 8,
            "full_text": "原文"
        }
    }
]
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)





class VectorDatabase:
    """ChromaDB 向量数据库封装"""
    
    def __init__(self, db_path: str = "src/data/vector_database"):
        """
        初始化向量数据库
        
        Args:
            db_path: 数据库存储路径
        """
        import chromadb
        from chromadb.config import Settings
        
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=chromadb.Settings(allow_reset=True, anonymized_telemetry=False)
        )
        
        self.collection = self.client.get_or_create_collection(
            name="document_chunks",
            metadata={"hnsw:space": "cosine"}  # 余弦相似度
        )
        
        logger.info(f"💾 ChromaDB 数据库初始化完成：{db_path}")
    
    def store_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 1000):
        """
        批量存储文档块到 ChromaDB
        
        Args:
            chunks: 文档块列表，每个包含 id, embedding, document, metadata
            batch_size: 批次大小
        """
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        for chunk in chunks:
            chunk_id = chunk.get("id")
            embedding = chunk.get("embedding")
            document = chunk.get("document")
            metadata = chunk.get("metadata", {})
            
            if not chunk_id or not embedding or not document:
                logger.warning(f"跳过无效数据块：{chunk}")
                continue
            
            ids.append(chunk_id)
            embeddings.append(embedding)
            documents.append(document)
            
            # 构建元数据
            meta = {
                "source": metadata.get("source", ""),
                "chapter": metadata.get("chapter", ""),
                "start_page": metadata.get("start_page", 0),
                "end_page": metadata.get("end_page", 0),
                "chunk_id": chunk_id
            }
            
            # 添加全文预览（如果不太长）
            full_text = metadata.get("full_text", "")
            if full_text and len(full_text) < 1000:
                meta["preview"] = full_text[:500]
            
            metadatas.append(meta)
        
        if not ids:
            logger.warning("没有有效的数据块可存储")
            return
        
        # 分批存储
        total_stored = 0
        for i in range(0, len(ids), batch_size):
            end_idx = min(i + batch_size, len(ids))
            
            self.collection.add(
                ids=ids[i:end_idx],
                embeddings=embeddings[i:end_idx],
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx]
            )
            total_stored += (end_idx - i)
            logger.debug(f"存储批次 {i//batch_size + 1}: {end_idx - i} 条记录")
        
        logger.info(f"✅ 成功存储 {total_stored}/{len(ids)} 个数据块到向量数据库")
    
    def get_existing_ids(self) -> set:
        """获取数据库中已有的 ID 集合（用于去重）"""
        existing_ids = set()
        limit = 10000
        offset = 0
        
        while True:
            results = self.collection.get(limit=limit, offset=offset)
            if not results["ids"]:
                break
            
            existing_ids.update(results["ids"])
            offset += limit
            
            # 防止无限循环
            if offset > 1000000:
                break
        
        logger.info(f"数据库中已有 {len(existing_ids)} 条记录")
        return existing_ids
    
    def clear_database(self):
        """清空数据库"""
        results = self.collection.get()
        if results["ids"]:
            self.collection.delete(ids=results["ids"])
        logger.info("🗑️ 数据库已清空")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        results = self.collection.get(include=["metadatas"])
        
        total_count = len(results["ids"])
        sources = set()
        chapters = set()
        
        for meta in results.get("metadatas", []):
            if isinstance(meta, dict):
                if meta.get("source"):
                    sources.add(meta["source"])
                if meta.get("chapter"):
                    chapters.add(meta["chapter"])
        
        return {
            "total_chunks": total_count,
            "total_sources": len(sources),
            "total_chapters": len(chapters),
            "sources": list(sources)[:10],
            "chapters": list(chapters)[:20]
        }


def ingest_json_file(json_path: Path, vector_db: VectorDatabase, force_mode: bool = False) -> Dict[str, int]:
    """
    处理单个 JSON 文件并导入向量数据库
    
    Args:
        json_path: JSON 文件路径
        vector_db: 向量数据库
        force_mode: 强制覆盖模式
        
    Returns:
        统计信息字典
    """
    stats = {"processed": 0, "skipped": 0, "failed": 0}
    
    # 读取 JSON 文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        logger.warning(f"{json_path} 格式错误：期望数组")
        return stats
    
    logger.info(f"📄 读取到 {len(data)} 条记录")
    
    # 获取已有 ID（用于去重）
    if not force_mode:
        existing_ids = vector_db.get_existing_ids()
    else:
        existing_ids = set()
        logger.info("⚡ [强制模式] 忽略已存在记录")
    
    # 分离需要处理和跳过的记录
    chunks_to_store = []
    
    for chunk in data:
        chunk_id = chunk.get("id")
        
        # 检查是否已存在
        if chunk_id in existing_ids and not force_mode:
            stats["skipped"] += 1
            continue
        
        # 验证数据完整性
        if not chunk.get("embedding") or not chunk.get("document"):
            logger.warning(f"跳过无效数据块：{chunk_id}")
            stats["failed"] += 1
            continue
        
        chunks_to_store.append(chunk)
    
    if not chunks_to_store:
        logger.info("⏭️ 所有记录都已存在，跳过")
        return stats
    
    # 批量存储到数据库
    logger.info(f"💾 开始存入向量数据库...")
    vector_db.store_chunks(chunks_to_store, batch_size=1000)
    
    stats["processed"] = len(chunks_to_store)
    logger.info(f"✅ 完成：处理 {stats['processed']} 条，跳过 {stats['skipped']} 条")
    
    return stats


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='向量数据库导入工具（精简版）')
    parser.add_argument('--input-dir', type=str, default='src/data/chunks',
                       help='输入 JSON 目录（默认：src/data/chunks）')
    parser.add_argument('--db-path', type=str, default='src/data/vector_database',
                       help='ChromaDB 存储路径')
    parser.add_argument('--force', action='store_true',
                       help='强制覆盖已存在的记录')
    parser.add_argument('--clear-db', action='store_true',
                       help='先清空数据库再导入')
    parser.add_argument('--show-stats', action='store_true',
                       help='显示数据库统计信息')
    parser.add_argument('--book', type=str,
                       help='指定要处理的书籍（不含.json 扩展名）')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 向量数据库导入工具（精简版）")
    print("=" * 60)
    
    # 验证输入目录
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"❌ 输入目录不存在：{input_dir}")
        return
    
    # 收集 JSON 文件
    if args.book:
        json_files = [input_dir / f"{args.book}_chunks.json"]
        if not json_files[0].exists():
            logger.error(f"❌ 文件不存在：{json_files[0]}")
            return
    else:
        json_files = list(input_dir.glob("*.json"))
    
    if not json_files:
        logger.warning(f"⚠️ 未找到任何 JSON 文件")
        return
    
    logger.info(f"📚 找到 {len(json_files)} 个 JSON 文件")
    for jf in json_files:
        logger.info(f"   - {jf.name}")
    
    # 初始化向量数据库
    logger.info("\n🔧 初始化向量数据库...")
    vector_db = VectorDatabase(db_path=args.db_path)
    
    # 清空数据库（如果需要）
    if args.clear_db:
        vector_db.clear_database()
    
    # 处理每个文件
    total_stats = {"processed": 0, "skipped": 0, "failed": 0}
    
    for json_file in json_files:
        logger.info(f"\n{'='*60}")
        logger.info(f"📖 开始处理：{json_file.name}")
        logger.info(f"{'='*60}")
        
        stats = ingest_json_file(
            json_file,
            vector_db,
            force_mode=args.force
        )
        
        for key in total_stats:
            total_stats[key] += stats[key]
    
    # 显示汇总
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 处理完成汇总:")
    logger.info(f"{'='*60}")
    logger.info(f"   ✅ 成功处理：{total_stats['processed']} 条")
    logger.info(f"   ⏭️  跳过：{total_stats['skipped']} 条")
    logger.info(f"   ❌ 失败：{total_stats['failed']} 条")
    
    # 显示数据库统计
    if args.show_stats:
        logger.info(f"\n{'='*60}")
        db_stats = vector_db.get_stats()
        logger.info(f"📈 数据库统计:")
        logger.info(f"{'='*60}")
        for key, value in db_stats.items():
            logger.info(f"   {key}: {value}")
    
    logger.info(f"\n✅ 全部处理完成！")


if __name__ == "__main__":
    main()