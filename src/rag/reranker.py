#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
重排序器

对多路召回的结果进行融合和重排序
"""

from typing import List, Dict, Any
from functools import lru_cache
import math
import time
import logging

logger = logging.getLogger(__name__)

# 🔥 使用 lru_cache 优化 CrossEncoder 模型加载（单例模式）
@lru_cache(maxsize=1)
def get_cross_encoder_model(model_name: str = "BAAI/bge-reranker-v2-m3"):
    """
    获取或创建 CrossEncoder 模型（单例模式，使用 lru_cache 自动缓存）
    
    Args:
        model_name: 模型名称
        
    Returns:
        CrossEncoder 实例
    """
    logger.info(f"🤖 [首次加载] CrossEncoder 模型：{model_name}")
    start = time.time()
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(model_name)
    elapsed = time.time() - start
    logger.info(f"⏱️  [性能] CrossEncoder 模型加载耗时：{elapsed:.2f}秒")
    return model


class ResultFusion:
    """结果融合器"""
    
    @staticmethod
    def rrf_fusion(vector_results: List[Dict],
                   keyword_results: List[Dict],
                   chapter_results: List[Dict],
                   k: int = 60) -> List[Dict]:
        """
        RRF (Reciprocal Rank Fusion) 倒数排名融合
        
        Args:
            vector_results: 向量检索结果
            keyword_results: 关键词检索结果
            chapter_results: 章节检索结果
            k: 平滑常数
            
        Returns:
            融合后的结果列表
        """
        score_map = {}
        
        # 计算每个结果的 RRF 分数
        for results, weight in [
            (vector_results, 0.6),
            (keyword_results, 0.3),
            (chapter_results, 0.1)
        ]:
            for rank, doc in enumerate(results, 1):
                doc_id = doc.get('id')
                if doc_id not in score_map:
                    score_map[doc_id] = {
                        **doc,
                        'rrf_score': 0
                    }
                
                # RRF 公式：1 / (k + rank)
                rrf_score = weight * (1.0 / (k + rank))
                score_map[doc_id]['rrf_score'] += rrf_score
        
        # 转换为列表并排序
        fused_results = list(score_map.values())
        fused_results.sort(key=lambda x: x['rrf_score'], reverse=True)
        
        return fused_results
    
    @staticmethod
    def weighted_fusion(vector_results: List[Dict],
                       keyword_results: List[Dict],
                       chapter_results: List[Dict],
                       vector_weight: float = 0.5,
                       keyword_weight: float = 0.3,
                       chapter_weight: float = 0.2) -> List[Dict]:
        """
        加权分数融合
        
        Args:
            vector_results: 向量检索结果
            keyword_results: 关键词检索结果
            chapter_results: 章节检索结果
            vector_weight: 向量权重
            keyword_weight: 关键词权重
            chapter_weight: 章节权重
            
        Returns:
            融合后的结果列表
        """
        score_map = {}
        
        # 归一化各路的分数并加权
        for results, weight, score_key in [
            (vector_results, vector_weight, 'distance'),
            (keyword_results, keyword_weight, 'match_score'),
            (chapter_results, chapter_weight, 'match_score')
        ]:
            if not results:
                continue
            
            # 找到最大分数用于归一化
            if score_key == 'distance':
                # 距离越小越好，取反
                max_score = max((r.get(score_key, 0) for r in results), default=1)
                for rank, doc in enumerate(results, 1):
                    normalized = 1.0 - (doc.get(score_key, 1) / max_score if max_score else 0)
                    score_map.setdefault(doc['id'], {**doc, 'fused_score': 0})['fused_score'] += weight * normalized
            else:
                # 分数越大越好
                max_score = max((r.get(score_key, 0) for r in results), default=1)
                for doc in results:
                    normalized = doc.get(score_key, 0) / max_score if max_score else 0
                    score_map.setdefault(doc['id'], {**doc, 'fused_score': 0})['fused_score'] += weight * normalized
        
        # 转换为列表并排序
        fused_results = list(score_map.values())
        fused_results.sort(key=lambda x: x['fused_score'], reverse=True)
        
        return fused_results


class CrossEncoderReranker:
    """Cross-Encoder 重排序器"""
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        """
        初始化重排序器
        
        Args:
            model_name: Cross-Encoder 模型名称
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
    
    def _load_model(self):
        """延迟加载模型（使用全局缓存）"""
        if self.model is None:
            # 使用全局缓存的模型，避免重复加载
            self.model = get_cross_encoder_model(self.model_name)
    
    def rerank(self, 
               query: str, 
               documents: List[Dict[str, Any]], 
               top_k: int = 5) -> List[Dict[str, Any]]:
        """
        使用 Cross-Encoder 对文档进行重排序
        
        Args:
            query: 查询文本
            documents: 待排序的文档列表
            top_k: 返回的重排序结果数量
            
        Returns:
            重排序后的文档列表
        """
        if not documents:
            return []
        
        self._load_model()
        
        # 准备输入数据
        pairs = [[query, doc.get('document', '')] for doc in documents]
        
        # 预测相关性分数
        scores = self.model.predict(pairs)
        
        # 将分数添加到文档中
        for doc, score in zip(documents, scores):
            doc['rerank_score'] = float(score)
        
        # 按分数降序排序
        reranked = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)
        
        return reranked[:top_k]
