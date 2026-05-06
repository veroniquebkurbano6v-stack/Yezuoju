#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检索策略选择器

根据问题分类结果（type + keywords）选择合适的检索策略和权重配置
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class QueryType(str, Enum):
    """问题类型枚举"""
    STORY_SUMMARY = "故事梗概"
    FACT_EVIDENCE = "事实依据"
    CHARACTER_RELATION = "特定人物或人物之间的关系"
    PLOT_SEGMENT = "段落情节"


@dataclass
class RetrievalStrategy:
    """检索策略配置"""
    # 各检索方式的权重 (0-1)
    vector_weight: float = 0.5
    keyword_weight: float = 0.3
    chapter_weight: float = 0.2
    
    # 是否启用重排序
    use_reranker: bool = True
    
    # 返回结果数量
    top_k: int = 10
    
    # 重排序后返回数量
    rerank_top_k: int = 5
    
    # 描述
    description: str = ""


class StrategySelector:
    """检索策略选择器"""
    
    # 预定义的策略配置
    STRATEGIES = {
        QueryType.STORY_SUMMARY: RetrievalStrategy(
            vector_weight=0.6,
            keyword_weight=0.2,
            chapter_weight=0.2,
            use_reranker=True,
            top_k=15,
            rerank_top_k=5,
            description="故事梗概类问题：侧重向量语义相似度"
        ),
        
        QueryType.FACT_EVIDENCE: RetrievalStrategy(
            vector_weight=0.2,
            keyword_weight=0.7,
            chapter_weight=0.1,
            use_reranker=True,
            top_k=20,
            rerank_top_k=8,
            description="事实依据类问题：侧重关键词精确匹配（已优化：关键词权重从 0.6 提升至 0.7，向量权重从 0.3 降至 0.2）"
        ),
        
        QueryType.CHARACTER_RELATION: RetrievalStrategy(
            vector_weight=0.4,
            keyword_weight=0.5,
            chapter_weight=0.1,
            use_reranker=True,
            top_k=15,
            rerank_top_k=6,
            description="人物关系类问题：关键词 + 向量平衡"
        ),
        
        QueryType.PLOT_SEGMENT: RetrievalStrategy(
            vector_weight=0.7,
            keyword_weight=0.2,
            chapter_weight=0.1,
            use_reranker=True,  # 🔥 默认启用重排序
            top_k=15,
            rerank_top_k=6,
            description="段落情节类问题：侧重向量语义，启用重排序提高精度"
        ),
    }
    
    @classmethod
    def select_strategy(cls, query_type: str, keywords: List[str]) -> RetrievalStrategy:
        """
        根据问题类型和关键词选择检索策略
        
        Args:
            query_type: 问题类型（来自 classify_query_type）
            keywords: 关键词列表
            
        Returns:
            RetrievalStrategy: 检索策略配置
        """
        # 尝试匹配预定义策略
        if query_type in cls.STRATEGIES:
            strategy = cls.STRATEGIES[query_type]
            
            # 🔥 动态调整：如果有关键词包含"章节"、"第几回"等，增加章节检索权重
            chapter_keywords = {"章", "节", "回", "篇", "卷"}
            if any(kw in chapter_keywords for kw in keywords):
                strategy.chapter_weight = min(strategy.chapter_weight + 0.2, 0.5)
                strategy.vector_weight = max(strategy.vector_weight - 0.1, 0.2)
            
            return strategy
        
        # 默认策略：均衡配置
        return RetrievalStrategy(
            vector_weight=0.4,
            keyword_weight=0.4,
            chapter_weight=0.2,
            use_reranker=True,
            top_k=15,
            rerank_top_k=5,
            description="默认均衡策略"
        )
    
    @classmethod
    def get_strategy_info(cls, query_type: str) -> str:
        """获取策略说明信息"""
        if query_type in cls.STRATEGIES:
            return cls.STRATEGIES[query_type].description
        return "未知类型，使用默认策略"
