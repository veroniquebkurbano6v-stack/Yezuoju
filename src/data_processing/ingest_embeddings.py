#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
嵌入向量生成脚本（就地修改版）

读取src/data/chunks中的 JSON 数据，使用嵌入模型生成向量，
并直接修改原 JSON 文件的 embedding 字段。
"""

import sys
import os
import json
from pathlib import Path
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 设置 transformers 安全加载环境变量（解决 PyTorch 2.6+ 的安全检查）
os.environ['TRANSFORMERS_NO_LOAD_CHECK'] = '1'

# === 配置参数 ===
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
CHUNKS_DIR = Path("src/data/chunks")


class EmbeddingGenerator:
    """嵌入向量生成器（支持批量处理和缓存）"""
    
    def __init__(self, model_name: str):
        """
        初始化嵌入模型
        
        Args:
            model_name: 模型名称
        """
        print(f"🤖 加载嵌入模型：{model_name}")
        
        # 检查 CUDA 是否可用
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
                gpu_name = torch.cuda.get_device_name(0)
                print(f"  ✓ 使用 GPU: {gpu_name}")
            else:
                device = "cpu"
                print(f"  ⚠ CUDA 不可用，使用 CPU")
        except ImportError:
            device = "cpu"
            print(f"  ⚠ PyTorch 未安装，使用 CPU")
        
        # 加载模型
        self.model = SentenceTransformer(model_name, device=device)
        print(f"  ✓ 模型加载完成")
    
    def generate_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        批量生成嵌入向量
        
        Args:
            texts: 文本列表
            batch_size: 批次大小
            
        Returns:
            嵌入向量列表（每个向量是浮点数列表）
        """
        if not texts:
            return []
        
        embeddings = []
        total = len(texts)
        
        print(f"\n🔮 开始生成嵌入向量...")
        print(f"   总文本数：{total}")
        print(f"   批次大小：{batch_size}")
        
        for i in range(0, total, batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # 批量编码
            batch_embeddings = self.model.encode(
                batch_texts,
                normalize_embeddings=True,  # 归一化（提高检索精度）
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            # 转换为列表
            embeddings.extend(batch_embeddings.tolist())
            
            # 显示进度
            processed = min(i + batch_size, total)
            progress = (processed / total) * 100
            print(f"   进度：{processed}/{total} ({progress:.1f}%)")
        
        return embeddings
def has_embedding(chunk: Dict[str, Any]) -> bool:
    """检查文本块是否已有嵌入向量"""
    embedding = chunk.get('embedding')
    if not embedding:
        return False
    if isinstance(embedding, list) and len(embedding) > 0:
        return True
    return False


def process_json_file(json_path: Path, embedding_gen: EmbeddingGenerator, force_mode: bool = False) -> bool:
    """
    处理单个 JSON 文件（就地修改）
    
    Args:
        json_path: JSON 文件路径
        embedding_gen: 嵌入生成器
        force_mode: 是否强制覆盖已有向量
        
    Returns:
        bool: 是否成功处理
    """
    book_name = json_path.stem.replace('_chunks', '')
    
    print(f"\n{'='*60}")
    print(f"📖 开始处理：《{book_name}》")
    print(f"   文件：{json_path.name}")
    print(f"{'='*60}")
    
    # 读取 JSON
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
    except Exception as e:
        print(f"❌ 读取文件失败：{e}")
        return False
    
    print(f"✓ 读取到 {len(chunks)} 个文本块")
    
    # 统计需要处理的文本块
    chunks_to_process = []
    already_processed = 0
    
    for chunk in chunks:
        if has_embedding(chunk) and not force_mode:
            already_processed += 1
        else:
            chunks_to_process.append(chunk)
    
    if already_processed > 0 and not force_mode:
        print(f"⏭️  检测到 {already_processed} 条已有向量，将跳过")
    
    if not chunks_to_process:
        print(f"✅ 《{book_name}》已全部处理，无需操作")
        return True
    
    print(f"📝 待处理：{len(chunks_to_process)} 条")
    
    # 提取所有 document 字段
    texts = [chunk['document'] for chunk in chunks_to_process]
    
    # 生成嵌入向量
    embeddings = embedding_gen.generate_batch(texts, batch_size=32)
    
    # 填充embedding 字段
    print(f"\n💾 填充embedding 字段...")
    for i, chunk in enumerate(chunks_to_process):
        chunk['embedding'] = embeddings[i]
    
    # 🔥 写回原文件（就地修改）
    print(f"\n💾 保存修改到原文件...")
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 《{book_name}》处理完成！")
        print(f"   新增向量：{len(chunks_to_process)} 条")
        print(f"   总计向量：{len(chunks)} 条")
        return True
        
    except Exception as e:
        print(f"❌ 保存文件失败：{e}")
        return False


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='嵌入向量生成脚本（就地修改）')
    parser.add_argument('--book', type=str, help='指定要处理的书籍（不含.json 扩展名）')
    parser.add_argument('--force', action='store_true', help='强制重新生成（覆盖已有向量）')
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小（默认 32）')
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 嵌入向量生成脚本（就地修改版）")
    print("=" * 60)
    
    # 检查 chunks 目录
    if not CHUNKS_DIR.exists():
        print(f"❌ 目录不存在：{CHUNKS_DIR}")
        print(f"   请先运行：python src/concatenate_text_blocks.py")
        return
    
    # 收集 JSON 文件
    if args.book:
        # 处理指定的书籍
        target_file = CHUNKS_DIR / f"{args.book}_chunks.json"
        if not target_file.exists():
            print(f"❌ 文件不存在：{target_file}")
            return
        json_files = [target_file]
    else:
        # 处理所有 JSON 文件
        json_files = list(CHUNKS_DIR.glob('*_chunks.json'))
    
    if not json_files:
        print("❌ 未找到任何 *_chunks.json 文件")
        return
    
    print(f"\n📚 找到 {len(json_files)} 个 JSON 文件:")
    for i, json_file in enumerate(json_files, 1):
        book_name = json_file.stem.replace('_chunks', '')
        print(f"  {i}. 《{book_name}》")
    
    # 确认处理模式
    if args.force:
        print(f"\n⚡ [强制模式] 将覆盖所有已有向量")
    
    # 初始化嵌入模型
    embedding_gen = EmbeddingGenerator(EMBEDDING_MODEL)
    
    # 处理每个文件
    success_count = 0
    for json_file in json_files:
        success = process_json_file(json_file, embedding_gen, args.force)
        if success:
            success_count += 1
    
    # 显示统计
    print("\n" + "=" * 60)
    print("📊 处理完成统计:")
    print("=" * 60)
    print(f"   总文件数：{len(json_files)}")
    print(f"   成功：{success_count}")
    print(f"   失败：{len(json_files) - success_count}")
    print(f"   输出目录：{CHUNKS_DIR}")
    print("=" * 60)
    
    if args.book:
        print(f"\n✅ 《{args.book}》处理完成！")
    else:
        print(f"\n✅ 所有文件处理完成！")


if __name__ == "__main__":
    main()
