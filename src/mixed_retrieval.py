#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
混合检索引擎（向量 + 关键词）

支持：
1. 纯向量检索
2. 纯关键词检索
3. 混合检索（向量 + 关键词加权）
4. 测试模式（批量测试问题并显示结果）
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# === 全局模型缓存（真正的单例模式） ===
_model_cache = {
    "embedding": None,
    "cross_encoder": None
}

def get_embedding_model(model_name: str = "BAAI/bge-m3"):
    """
    获取或创建嵌入模型（单例模式）
    
    Args:
        model_name: 模型名称
        
    Returns:
        SentenceTransformer 实例
    """
    if _model_cache["embedding"] is None:
        logger.info(f"🤖 [首次加载] Embedding 模型：{model_name}")
        start = time.time()
        from sentence_transformers import SentenceTransformer
        _model_cache["embedding"] = SentenceTransformer(model_name)
        elapsed = time.time() - start
        logger.info(f"⏱️  [性能] Embedding 模型加载耗时：{elapsed:.2f}秒")
    else:
        logger.info(f"✅ [缓存命中] 使用已加载的 Embedding 模型：{model_name}")
    return _model_cache["embedding"]

class VectorRetriever:
    """ChromaDB 向量检索器"""
    
    def __init__(self, db_path: str = None):
        # 从环境变量读取或使用默认路径
        if db_path is None:
            db_path_env = os.getenv("VECTOR_DB_PATH")
            if db_path_env:
                # 如果环境变量是绝对路径，直接使用；否则基于项目根目录构建
                if os.path.isabs(db_path_env):
                    db_path = db_path_env
                else:
                    project_root = Path(__file__).resolve().parent.parent
                    db_path = str(project_root / db_path_env)
            else:
                project_root = Path(__file__).resolve().parent.parent
                db_path = str(project_root / "src" / "data" / "vector_database")
        
        logger.info(f"[VectorRetriever] 开始初始化，db_path={db_path}")
        try:
            import chromadb
            from chromadb.config import Settings
            
            logger.info(f"[VectorRetriever] 创建 ChromaDB 客户端...")
            self.client = chromadb.PersistentClient(
                path=db_path,
                settings=chromadb.Settings(anonymized_telemetry=False)
            )
            
            # 列出所有集合并检查
            try:
                collections = self.client.list_collections()
                logger.info(f"[VectorRetriever] 数据库中共有 {len(collections)} 个集合：{[c.name for c in collections]}")
            except Exception as list_err:
                logger.warning(f"[VectorRetriever] 无法列出集合：{list_err}")
            
            logger.info(f"[VectorRetriever] 获取集合 document_chunks...")
            self.collection = self.client.get_collection("document_chunks")
            
            logger.info(f"[VectorRetriever] ✅ 向量数据库加载完成：{db_path}")
        except Exception as e:
            logger.error(f"[VectorRetriever] ❌ 初始化失败：{e}", exc_info=True)
            raise
    
    def search(self, query_embedding: List[float], top_k: int = 5, include_embeddings: bool = False) -> List[Dict[str, Any]]:
        """
        向量相似度检索
        
        Args:
            query_embedding: 查询向量化后的向量
            top_k: 返回结果数量
            include_embeddings: 是否包含 embedding 向量（用于 MMR）
            
        Returns:
            匹配的文档块列表
        """
        try:
            # 🔥 根据 include_embeddings 参数决定 include 内容
            include_list = ["documents", "metadatas", "distances"]
            if include_embeddings:
                include_list.append("embeddings")
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=include_list
            )
            
            # 整理结果
            matches = []
            if results['ids'] and len(results['ids'][0]) > 0:
                for i in range(len(results['ids'][0])):
                    match = {
                        'id': results['ids'][0][i],
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': float(results['distances'][0][i]) if results['distances'] is not None and len(results['distances'][0]) > i else None
                    }
                    # 如果需要 embeddings
                    if include_embeddings and results.get('embeddings') and len(results['embeddings']) > 0 and len(results['embeddings'][0]) > i:
                        match['embedding'] = results['embeddings'][0][i]
                    matches.append(match)
            
            return matches
            
        except Exception as e:
            logger.error(f"❌ 向量检索失败：{e}")
            return []


class KeywordRetriever:
    """关键词检索器（基于元数据匹配）"""
    
    def __init__(self, db_path: str = None):
        # 从环境变量读取或使用默认路径
        if db_path is None:
            db_path_env = os.getenv("VECTOR_DB_PATH")
            if db_path_env:
                if os.path.isabs(db_path_env):
                    db_path = db_path_env
                else:
                    project_root = Path(__file__).resolve().parent.parent
                    db_path = str(project_root / db_path_env)
            else:
                project_root = Path(__file__).resolve().parent.parent
                db_path = str(project_root / "src" / "data" / "vector_database")
        
        logger.info(f"[KeywordRetriever] 开始初始化，db_path={db_path}")
        try:
            import chromadb
            from chromadb.config import Settings
            
            logger.info(f"[KeywordRetriever] 创建 ChromaDB 客户端...")
            self.client = chromadb.PersistentClient(
                path=db_path,
                settings=chromadb.Settings(anonymized_telemetry=False)
            )
            
            logger.info(f"[KeywordRetriever] 获取集合 document_chunks...")
            self.collection = self.client.get_collection("document_chunks")
            
            logger.info(f"[KeywordRetriever] ✅ 关键词检索器初始化完成")
        except Exception as e:
            logger.error(f"[KeywordRetriever] ❌ 初始化失败：{e}", exc_info=True)
            raise
    
    def search(self, keywords: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        关键词匹配检索
        
        Args:
            keywords: 关键词列表
            top_k: 返回结果数量
            
        Returns:
            匹配的文档块列表
        """
        try:
            # 构建 where 条件（匹配 chapter 或 source 字段）
            # ChromaDB 的 where 只支持简单等值匹配，复杂匹配需要 where_document
            
            # 使用全文搜索（在 document 字段中搜索）
            keyword_query = " ".join(keywords)
            
            results = self.collection.get(
                where_document={"$contains": keyword_query},
                limit=top_k * 2  # 先多取一些，后面再排序
            )
            
            # 按关键词匹配度排序
            matches = []
            for i in range(len(results['ids'])):
                doc_text = results['documents'][i].lower()
                
                # 计算关键词匹配数
                match_count = sum(1 for kw in keywords if kw.lower() in doc_text)
                
                if match_count > 0:
                    match = {
                        'id': results['ids'][i],
                        'document': results['documents'][i],
                        'metadata': results['metadatas'][i],
                        'match_score': match_count  # 匹配得分
                    }
                    matches.append(match)
            
            # 按匹配得分降序排序
            matches.sort(key=lambda x: x['match_score'], reverse=True)
            
            return matches[:top_k]
            
        except Exception as e:
            logger.error(f"❌ 关键词检索失败：{e}")
            return []


class HybridRetriever:
    """混合检索器（向量 + 关键词）"""
    
    def __init__(self, db_path: str = None):
        # 从环境变量读取或使用默认路径
        if db_path is None:
            db_path_env = os.getenv("VECTOR_DB_PATH")
            if db_path_env:
                if os.path.isabs(db_path_env):
                    db_path = db_path_env
                else:
                    project_root = Path(__file__).resolve().parent.parent
                    db_path = str(project_root / db_path_env)
            else:
                project_root = Path(__file__).resolve().parent.parent
                db_path = str(project_root / "src" / "data" / "vector_database")
        
        logger.info(f"[HybridRetriever] 开始初始化，db_path={db_path}")
        try:
            self.vector_retriever = VectorRetriever(db_path)
            logger.info("[HybridRetriever] VectorRetriever 初始化成功")
        except Exception as e:
            logger.error(f"[HybridRetriever] VectorRetriever 初始化失败：{e}", exc_info=True)
            raise
        
        try:
            self.keyword_retriever = KeywordRetriever(db_path)
            logger.info("[HybridRetriever] KeywordRetriever 初始化成功")
        except Exception as e:
            logger.error(f"[HybridRetriever] KeywordRetriever 初始化失败：{e}", exc_info=True)
            raise
        
        # 嵌入模型（用于将查询文本向量化）
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL", 
                                               "intfloat/multilingual-e5-large")
        self.model = None
        logger.info(f"[HybridRetriever] 初始化完成，embedding_model={self.embedding_model_name}")
    
    def _get_embedding_model(self):
        """延迟加载嵌入模型（使用全局缓存）"""
        if self.model is None:
            # 使用全局缓存的模型，避免重复加载
            self.model = get_embedding_model(self.embedding_model_name)
        return self.model
    
    def embed_query(self, query: str) -> List[float]:
        """将查询文本转换为向量"""
        model = self._get_embedding_model()
        embedding = model.encode(query, normalize_embeddings=True)
        return embedding.tolist()
    
    def tokenize_query(self, query: str) -> List[str]:
        """
        使用嵌入模型的分词器对查询进行分词
        
        Args:
            query: 查询文本
            
        Returns:
            分词后的列表
        """
        model = self._get_embedding_model()
        tokenizer = model.tokenizer
        
        # 使用分词器进行分词
        tokens = tokenizer.tokenize(query)
        
        logger.info(f"🔑 原始查询：{query}")
        logger.info(f"🔑 分词结果：{tokens}")
        
        return tokens
    
    def search(self, 
               query: str, 
               keywords: Optional[List[str]] = None,
               top_k: int = 5,
               vector_weight: float = 0.7,
               keyword_weight: float = 0.3,
               use_mmr: bool = True) -> List[Dict[str, Any]]:
        """
        混合检索（支持 MMR 多样性优化）
        
        Args:
            query: 查询文本
            keywords: 关键词列表（可选）
            top_k: 返回结果数量
            vector_weight: 向量检索权重（默认 0.7）
            keyword_weight: 关键词检索权重（默认 0.3）
            use_mmr: 是否使用 MMR 增加多样性（默认 True）
            
        Returns:
            排序后的混合检索结果
        """
        # 1. 🔥 扩大候选集（为 MMR 做准备）
        # 🔥 从 6 倍改回 3 倍，配合 Query Expansion 使用
        candidate_multiplier = 3 if use_mmr else 1
        expanded_top_k = top_k * candidate_multiplier
        
        # 2. 向量检索（包含 embeddings 用于 MMR）
        query_embedding = self.embed_query(query)
        vector_results = self.vector_retriever.search(
            query_embedding, 
            top_k=expanded_top_k,
            include_embeddings=use_mmr
        )
        
        # 3. 关键词检索（如果有关键词）
        keyword_results = []
        if keywords:
            keyword_results = self.keyword_retriever.search(keywords, top_k=expanded_top_k)
        
        # 4. 融合结果（简单的加权融合）
        if not keyword_results:
            fused_results = vector_results
        else:
            # 合并两个结果集
            all_results = {}
            
            # 添加向量检索结果
            for result in vector_results:
                result_id = result['id']
                result['vector_score'] = 1.0 - (result.get('distance', 0.5) or 0.5)  # 距离转分数
                result['keyword_score'] = 0.0
                result['final_score'] = result['vector_score'] * vector_weight
                all_results[result_id] = result
            
            # 添加关键词检索结果并融合
            for result in keyword_results:
                result_id = result['id']
                result['keyword_score'] = min(result.get('match_score', 0) / len(keywords), 1.0)
                result['vector_score'] = 0.0
                
                if result_id in all_results:
                    # 已存在，更新分数
                    existing = all_results[result_id]
                    existing['keyword_score'] = max(existing['keyword_score'], 
                                                    result['keyword_score'])
                    existing['final_score'] = (
                        existing['vector_score'] * vector_weight + 
                        existing['keyword_score'] * keyword_weight
                    )
                else:
                    # 新结果，计算最终分数
                    result['final_score'] = result['keyword_score'] * keyword_weight
                    all_results[result_id] = result
            
            # 按最终分数排序
            fused_results = sorted(
                all_results.values(),
                key=lambda x: x['final_score'],
                reverse=True
            )
        
        # 5. 🔥 MMR 多样性选择（如果启用）
        if use_mmr and len(fused_results) > top_k:
            logger.info(f"[HybridRetriever] 使用 MMR 从 {len(fused_results)} 个候选中选择 {top_k} 个多样化文档")
            
            mmr = MMRRetriever(lambda_param=0.7)
            
            # 提取 embeddings
            embeddings = [doc.get('embedding', []) for doc in fused_results]
            
            # 检查是否有 embeddings
            if embeddings and len(embeddings[0]) > 0:
                diverse_results = mmr.mmr_select(
                    query_embedding=query_embedding,
                    candidates=fused_results,
                    candidate_embeddings=embeddings,
                    k=top_k
                )
                return diverse_results
            else:
                logger.warning("[HybridRetriever] 未找到 embeddings，退化为直接截取 Top-K")
        
        # 6. 直接返回 Top-K
        return fused_results[:top_k]


def test_retrieval(retriever: HybridRetriever, test_cases: List[Dict[str, Any]]):
    """
    测试检索效果
    
    Args:
        retriever: 混合检索器实例
        test_cases: 测试用例列表
    """
    print("\n" + "=" * 80)
    print("🧪 向量数据库检索效果测试（自动分词）")
    print("=" * 80)
    
    for i, test_case in enumerate(test_cases, 1):
        query = test_case['query']
        expected = test_case.get('expected', '')
        
        # 🔥 使用模型分词器自动生成关键词
        keywords = retriever.tokenize_query(query)
        
        print(f"\n【测试 {i}/{len(test_cases)}】")
        print(f"查询：{query}")
        print(f"自动分词：{', '.join(keywords)}")
        
        # 执行检索
        results = retriever.search(query, keywords=keywords, top_k=3)
        
        print(f"\n📊 检索结果（共 {len(results)} 条）:")
        print("-" * 80)
        
        for j, result in enumerate(results, 1):
            print(f"\n{j}. ID: {result['id']}")
            print(f"   章节：{result['metadata'].get('chapter', 'N/A')}")
            print(f"   页码：{result['metadata'].get('start_page', 'N/A')}-"
                  f"{result['metadata'].get('end_page', 'N/A')}")
            print(f"   综合得分：{result.get('final_score', 0):.4f}")
            if 'vector_score' in result:
                print(f"   向量分数：{result['vector_score']:.4f}")
            if 'keyword_score' in result:
                print(f"   关键词分数：{result['keyword_score']:.4f}")
            print(f"   内容：{result['document'][:150]}...")
        
        if expected:
            print(f"\n💡 预期：{expected}")
        
        print("-" * 80)
    
    print("\n✅ 测试完成！")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="混合检索引擎")
    parser.add_argument("--query", type=str, help="查询文本")
    parser.add_argument("--keywords", type=str, nargs="+", help="关键词列表")
    parser.add_argument("--top-k", type=int, default=5, help="返回结果数量")
    parser.add_argument("--test", action="store_true", help="运行测试模式")
    parser.add_argument("--db-path", type=str, default="src/data/vector_database",
                        help="向量数据库路径")
    
    args = parser.parse_args()
    
    # 初始化检索器
    retriever = HybridRetriever(db_path=args.db_path)
    
    if args.test:
        # 测试模式：使用预设的测试用例（keywords 由分词器自动生成）
        test_cases = [
            {
                "query": "朱元璋为什么屠杀功臣？",
                "expected": "找到关于朱元璋屠杀功臣原因的解释"
            },
            {
                "query": "朱元璋的画像有什么特点？",
                "expected": "找到关于朱元璋圆脸俊像和长脸丑像的描述"
            },
            {
                "query": "明朝的官方语言是什么？",
                "expected": "找到关于淮西话成为明朝官方语言的记载"
            },
            {
                "query": "问鼎天下这一章讲了什么？",
                "expected": "找到第 84-99 页的内容摘要"
            },
            {
                "query": "法律制度的规定",
                "expected": "找到关于明朝法律制度的描述"
            }
        ]
        
        test_retrieval(retriever, test_cases)
        
    elif args.query:
        # 单次查询模式
        keywords = args.keywords if args.keywords else []
        
        print(f"\n🔍 查询：{args.query}")
        if keywords:
            print(f"🔑 关键词：{', '.join(keywords)}")
        
        results = retriever.search(args.query, keywords=keywords, top_k=args.top_k)
        
        print(f"\n📊 检索结果（共 {len(results)} 条）:\n")
        
        for i, result in enumerate(results, 1):
            print(f"{i}. [{result['metadata'].get('chapter', 'N/A')}]"
                  f" 第{result['metadata'].get('start_page', '?')}-"
                  f"{result['metadata'].get('end_page', '?')}页")
            print(f"   得分：{result.get('final_score', 0):.4f}")
            print(f"   {result['document']}\n")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
