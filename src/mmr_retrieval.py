#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MMR（最大边界相关性）检索模块

功能：
1. 在保持相关性的同时增加结果多样性
2. 避免检索结果高度相似
3. 提升关键信息的覆盖率
"""

import numpy as np
from typing import List, Dict, Any, Optional
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MMRRetriever:
    """MMR 检索器 - 平衡相关性和多样性"""
    
    def __init__(self, lambda_param: float = 0.7):
        """
        初始化 MMR 检索器
        
        Args:
            lambda_param: 相关性权重 (0-1)
                - 越高越重视相关性 (1.0 = 完全按相似度排序)
                - 越低越重视多样性 (0.0 = 完全按多样性排序)
                - 推荐值：0.5-0.8（默认 0.7）
        """
        self.lambda_param = lambda_param
        logger.info(f"[MMRRetriever] 初始化完成，lambda={lambda_param}")
    
    def mmr_select(
        self,
        query_embedding: List[float],
        candidates: List[Dict[str, Any]],
        candidate_embeddings: Optional[List[List[float]]] = None,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        MMR 选择算法 - 从候选集中选择最具多样性的 k 个文档
        
        Args:
            query_embedding: 查询向量
            candidates: 候选文档列表（每个元素包含 document, metadata 等字段）
            candidate_embeddings: 候选文档的向量表示（可选）
                - 如果提供，直接使用这些向量计算相似度
                - 如果不提供，需要从 candidates 中提取 embedding 字段
            k: 需要选择的文档数量
            
        Returns:
            经过 MMR 选择的文档列表（保持选择顺序）
        """
        if not candidates or k <= 0:
            return []
        
        # 限制 k 不超过候选集大小
        k = min(k, len(candidates))
        
        # 转换为 numpy 数组
        query_vec = np.array(query_embedding).reshape(1, -1)
        
        # 处理候选向量
        if candidate_embeddings is None:
            # 尝试从 candidates 中提取 embedding 字段
            try:
                candidate_embeddings = [
                    c.get('embedding', c.get('vector', [])) 
                    for c in candidates
                ]
                # 检查是否成功提取
                if not candidate_embeddings or len(candidate_embeddings[0]) == 0:
                    logger.warning("[MMRRetriever] 未找到候选向量，退化为按相似度排序")
                    # 退化为简单的 Top-K 选择
                    return candidates[:k]
            except (KeyError, IndexError):
                logger.warning("[MMRRetriever] 无法提取候选向量，退化为按相似度排序")
                return candidates[:k]
        
        candidate_matrix = np.array(candidate_embeddings)
        
        # 🔥 使用 sklearn 的 cosine_similarity 计算相似度（标准做法）
        from sklearn.metrics.pairwise import cosine_similarity
        
        # 计算所有候选文档与查询的相似度
        query_similarities = cosine_similarity(query_vec, candidate_matrix)[0]
        
        # 计算文档间的相似度矩阵（对称矩阵）
        doc_similarity_matrix = cosine_similarity(candidate_matrix)
        
        # 🔥 获取候选文档数量
        n_candidates = len(candidates)
        
        # MMR 贪心选择过程
        selected_indices = []
        remaining_indices = list(range(n_candidates))
        
        while len(selected_indices) < k and remaining_indices:
            max_mmr_score = -float('inf')
            best_idx = None
            
            for idx in remaining_indices:
                # 与查询的相似度
                similarity_to_query = query_similarities[idx]
                
                # 与已选文档的最大相似度（用于惩罚冗余）
                if selected_indices:
                    max_similarity_to_selected = max([
                        doc_similarity_matrix[idx, sel_idx] 
                        for sel_idx in selected_indices
                    ])
                else:
                    # 第一个文档只考虑与查询的相似度
                    max_similarity_to_selected = 0.0
                
                # MMR 分数公式：
                # MMR = λ * Sim(query, doc) - (1-λ) * max(Sim(doc, selected_docs))
                mmr_score = (
                    self.lambda_param * similarity_to_query -
                    (1 - self.lambda_param) * max_similarity_to_selected
                )
                
                if mmr_score > max_mmr_score:
                    max_mmr_score = mmr_score
                    best_idx = idx
            
            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
            else:
                break
        
        # 返回选中的文档（按选择顺序排列）
        selected_docs = [candidates[i] for i in selected_indices]
        
        logger.info(f"[MMRRetriever] 从 {len(candidates)} 个候选中选择了 {len(selected_docs)} 个多样化文档")
        
        return selected_docs
    
    def rerank_with_mmr(
        self,
        query_embedding: List[float],
        documents: List[Dict[str, Any]],
        document_embeddings: Optional[List[List[float]]] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        使用 MMR 对现有检索结果进行重排序
        
        Args:
            query_embedding: 查询向量
            documents: 待重排序的文档列表
            document_embeddings: 文档的向量表示（可选）
            top_k: 返回的重排序后文档数量
            
        Returns:
            经过 MMR 重排序的文档列表
        """
        # 先扩大候选集（取前 2*top_k 个作为候选）
        expanded_k = min(top_k * 3, len(documents))
        candidates = documents[:expanded_k]
        
        # 使用 MMR 选择
        return self.mmr_select(
            query_embedding=query_embedding,
            candidates=candidates,
            candidate_embeddings=document_embeddings,
            k=top_k
        )


def test_mmr():
    """测试 MMR 功能"""
    print("\n" + "=" * 80)
    print("🧪 MMR 检索器测试")
    print("=" * 80)
    
    # 模拟数据
    query = "朱元璋的家庭背景"
    
    # 创建 10 个模拟文档（有些内容相似，有些不同）
    candidates = [
        {"id": 1, "document": "朱元璋家境贫寒，父亲是佃农", "category": "economic_status"},
        {"id": 2, "document": "朱元璋家里很穷，没有田地", "category": "economic_status"},
        {"id": 3, "document": "朱家世代赤贫，靠租种地主土地为生", "category": "economic_status"},
        {"id": 4, "document": "朱元璋出生时父亲 47 岁，母亲 42 岁", "category": "parents_age"},
        {"id": 5, "document": "陈家二娘 42 岁生下朱元璋", "category": "parents_age"},
        {"id": 6, "document": "朱元璋小时候住茅草屋", "category": "housing"},
        {"id": 7, "document": "朱家住在三间破旧的茅草房里", "category": "housing"},
        {"id": 8, "document": "朱元璋童年经常吃不饱饭", "category": "food"},
        {"id": 9, "document": "朱家粮食不足，常靠野菜度日", "category": "food"},
        {"id": 10, "document": "朱元璋是家里的第八个孩子", "category": "siblings"},
    ]
    
    # 模拟向量（简化版）
    # 同一类别的文档向量相似，不同类别的向量差异较大
    base_vectors = {
        "economic_status": np.array([0.9, 0.8, 0.7]),
        "parents_age": np.array([0.5, 0.6, 0.4]),
        "housing": np.array([0.6, 0.5, 0.8]),
        "food": np.array([0.7, 0.9, 0.6]),
        "siblings": np.array([0.4, 0.5, 0.5]),
    }
    
    # 为每个文档添加向量（加上一些随机扰动）
    np.random.seed(42)
    for candidate in candidates:
        base = base_vectors[candidate["category"]]
        noise = np.random.randn(3) * 0.1
        candidate["embedding"] = (base + noise).tolist()
    
    # 查询向量
    query_embedding = np.array([0.85, 0.75, 0.65]).tolist()
    
    # 测试 MMR
    mmr = MMRRetriever(lambda_param=0.7)
    
    print(f"\n查询：{query}")
    print(f"候选文档数：{len(candidates)}")
    print(f"需要选择：5 个")
    
    # MMR 选择
    selected = mmr.mmr_select(
        query_embedding=query_embedding,
        candidates=candidates,
        k=5
    )
    
    print("\n📊 MMR 选择结果:")
    print("-" * 80)
    for i, doc in enumerate(selected, 1):
        print(f"{i}. [ID:{doc['id']}] 类别：{doc['category']}")
        print(f"   内容：{doc['document']}")
    
    # 统计多样性
    categories = [doc['category'] for doc in selected]
    unique_categories = set(categories)
    
    print("\n📈 多样性分析:")
    print(f"- 选中 {len(selected)} 个文档")
    print(f"- 涵盖 {len(unique_categories)} 个不同类别：{list(unique_categories)}")
    
    # 对比：不使用 MMR（直接取前 5 个）
    print("\n📊 对比：不使用 MMR（直接取前 5 个）:")
    print("-" * 80)
    for i, doc in enumerate(candidates[:5], 1):
        print(f"{i}. [ID:{doc['id']}] 类别：{doc['category']}")
        print(f"   内容：{doc['document']}")
    
    baseline_categories = [doc['category'] for doc in candidates[:5]]
    baseline_unique = set(baseline_categories)
    print(f"\n涵盖 {len(baseline_unique)} 个类别：{list(baseline_unique)}")
    
    print("\n✅ MMR 测试完成！")
    print("=" * 80)


if __name__ == "__main__":
    test_mmr()
