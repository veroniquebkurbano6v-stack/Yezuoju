#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LangChain 检索工具封装

将检索系统封装为 LangChain Tools 供 Agent 调用
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Type, Dict, Any, List
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from dotenv import load_dotenv

load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入检索组件
import sys
from pathlib import Path

# 添加 src 目录到 Python 路径
src_path = Path(__file__).resolve().parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from mixed_retrieval import HybridRetriever
from chapter_retrieval import ChapterRetriever
from retrieval_strategies import StrategySelector, QueryType
from reranker import ResultFusion, CrossEncoderReranker


# === 全局检索器缓存（单例模式） ===
_global_retrievers = {
    "hybrid": None,
    "chapter": None,
    "reranker": None
}

# === 全局检索结果存储（用于 Agent 调用链中传递数据） ===
_last_retrieval_result = {"docs": []}

def get_last_retrieval_result():
    """获取最后一次检索结果"""
    return _last_retrieval_result["docs"]

def clear_last_retrieval_result():
    """清空最后一次检索结果"""
    _last_retrieval_result["docs"] = []

def get_or_create_retrievers(db_path: str):
    """
    获取或创建检索器实例（单例模式，避免重复初始化）
    
    Args:
        db_path: 向量数据库路径
        
    Returns:
        tuple: (hybrid_retriever, chapter_retriever, reranker)
    """
    global _global_retrievers
    
    if _global_retrievers["hybrid"] is None:
        logger.info(f"[GlobalRetrievers] 首次初始化 HybridRetriever, db_path={db_path}")
        _global_retrievers["hybrid"] = HybridRetriever(db_path=db_path)
        logger.info("[GlobalRetrievers] ✅ HybridRetriever 初始化完成")
    
    if _global_retrievers["chapter"] is None:
        logger.info("[GlobalRetrievers] 首次初始化 ChapterRetriever")
        _global_retrievers["chapter"] = ChapterRetriever()
        logger.info("[GlobalRetrievers] ✅ ChapterRetriever 初始化完成")
    
    if _global_retrievers["reranker"] is None:
        logger.info("[GlobalRetrievers] 首次初始化 CrossEncoderReranker")
        _global_retrievers["reranker"] = CrossEncoderReranker()
        logger.info("[GlobalRetrievers] ✅ CrossEncoderReranker 初始化完成")
    
    return (
        _global_retrievers["hybrid"],
        _global_retrievers["chapter"],
        _global_retrievers["reranker"]
    )


class RetrievalInput(BaseModel):
    """检索工具输入参数"""
    query: str = Field(description="用户查询文本")
    query_type: str = Field(default="", description="问题类型（可选）")
    keywords: List[str] = Field(default_factory=list, description="关键词列表（可选）")
    top_k: int = Field(default=5, description="返回结果数量")


class SmartRetrievalTool(BaseTool):
    """智能检索工具（自动选择策略）"""
    
    name: str = "smart_retrieval"
    description: str = "智能检索工具，根据问题类型自动选择最优检索策略"
    args_schema: Type[BaseModel] = RetrievalInput
    
    # 声明私有属性
    hybrid_retriever: HybridRetriever = None
    chapter_retriever: ChapterRetriever = None
    reranker: CrossEncoderReranker = None
    use_cross_encoder: bool = True
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 从环境变量读取向量数据库路径
        db_path_env = os.getenv("VECTOR_DB_PATH")
        if db_path_env:
            # 如果环境变量是绝对路径，直接使用；否则基于项目根目录构建
            if os.path.isabs(db_path_env):
                db_path = db_path_env
            else:
                # 相对路径：基于项目根目录构建
                project_root = Path(__file__).resolve().parent.parent
                db_path = str(project_root / db_path_env)
        else:
            # 没有环境变量，使用默认值
            project_root = Path(__file__).resolve().parent.parent
            db_path = str(project_root / "src" / "data" / "vector_database")
        
        logger.info(f"[LangChainTools] 初始化，db_path={db_path}")
        
        try:
            # 使用全局检索器（单例模式，避免重复初始化）
            self.hybrid_retriever, self.chapter_retriever, self.reranker = get_or_create_retrievers(db_path)
            logger.info("[LangChainTools] ✅ 使用全局共享检索器实例")
        except Exception as e:
            logger.error(f"[LangChainTools] 初始化失败：{e}", exc_info=True)
            raise
    
    def _run(self, 
             query: str, 
             query_type: str = "", 
             keywords: List[str] = None,
             top_k: int = 5) -> str:
        """
        执行检索
        
        Args:
            query: 查询文本
            query_type: 问题类型（可选）
            keywords: 关键词列表（可选）
            top_k: 返回结果数量
            
        Returns:
            JSON 格式的检索结果
        """
        logger.info(f"[SmartRetrievalTool] 🔍 开始执行智能检索")
        logger.info(f"[SmartRetrievalTool]   传入参数：query='{query[:50]}...', query_type='{query_type}', keywords={keywords if keywords else 'None'}, top_k={top_k}")
        
        # 检查 query 是否是 JSON 格式（包含完整的查询、类型和关键词）
        import json as json_lib
        try:
            if query.startswith('{') and query.endswith('}'):
                # 这是一个 JSON 对象，解析它
                query_data = json_lib.loads(query)
                if isinstance(query_data, dict):
                    original_query = query_data.get('query', query)
                    query_type_from_json = query_data.get('query_type', query_type)
                    keywords_from_json = query_data.get('keywords', keywords)
                    
                    # 使用解析后的值
                    query = original_query
                    if query_type_from_json:
                        query_type = query_type_from_json
                    if keywords_from_json:
                        keywords = keywords_from_json
                    
                    logger.info(f"[SmartRetrievalTool] ✓ 从 JSON 中解析出：query_type='{query_type}', keywords={keywords}")
        except (json_lib.JSONDecodeError, TypeError, AttributeError) as e:
            # 不是 JSON 格式，使用原始参数
            logger.debug(f"[SmartRetrievalTool] query 不是 JSON 格式，使用原始参数")
            pass
        
        # 如果没有提供关键词，使用查询文本作为关键词
        if not keywords:
            keywords = [query]
            logger.info(f"[SmartRetrievalTool] 未提供关键词，使用查询文本作为关键词")
        else:
            logger.info(f"[SmartRetrievalTool] 使用传入的关键词：{keywords}")
        
        # 选择检索策略
        strategy = StrategySelector.select_strategy(query_type, keywords)
        
        logger.info(f"[SmartRetrievalTool] 📊 使用策略：{strategy.description}")
        logger.info(f"[SmartRetrievalTool]   向量权重：{strategy.vector_weight}, 关键词权重：{strategy.keyword_weight}, 章节权重：{strategy.chapter_weight}")
        print(f"   关键词：{keywords}")
        print(f"   关键词权重：{strategy.keyword_weight}")
        print(f"   章节权重：{strategy.chapter_weight}")
        
        # 1. 向量检索
        query_embedding = self.hybrid_retriever.embed_query(query)
        vector_results = self.hybrid_retriever.vector_retriever.search(
            query_embedding, 
            top_k=int(strategy.top_k * strategy.vector_weight * 2)
        )
        
        # 2. 关键词检索
        keyword_results = self.hybrid_retriever.keyword_retriever.search(
            keywords,
            top_k=int(strategy.top_k * strategy.keyword_weight * 2)
        )
        
        # 3. 章节检索（如果有明确的章节关键词）
        chapter_results = []
        if strategy.chapter_weight > 0.1:
            chapter_results = self.chapter_retriever.search(
                keywords,
                top_k=int(strategy.top_k * strategy.chapter_weight * 2)
            )
        
        # 4. 结果融合
        fused_results = ResultFusion.weighted_fusion(
            vector_results,
            keyword_results,
            chapter_results,
            vector_weight=strategy.vector_weight,
            keyword_weight=strategy.keyword_weight,
            chapter_weight=strategy.chapter_weight
        )
        
        # 5. 🔥 重排序（如果启用）
        logger.info(f"[SmartRetrievalTool] 重排序检查：use_reranker={strategy.use_reranker}, use_cross_encoder={self.use_cross_encoder}, fused_results 数量={len(fused_results)}")
        
        if strategy.use_reranker and self.use_cross_encoder and len(fused_results) > 0:
            logger.info(f"[SmartRetrievalTool] 🔥 开始执行 CrossEncoder 重排序...")
            reranked_results = self.reranker.rerank(
                query,
                fused_results,
                top_k=strategy.rerank_top_k
            )
            logger.info(f"[SmartRetrievalTool] ✅ 重排序完成，返回 {len(reranked_results)} 篇文档")
        else:
            logger.warning(f"[SmartRetrievalTool] ⚠️ 未启用重排序：use_reranker={strategy.use_reranker}, use_cross_encoder={self.use_cross_encoder}")
            reranked_results = fused_results[:strategy.rerank_top_k]
        
        # 6. 格式化输出
        result = {
            "query": query,
            "query_type": query_type,
            "strategy": strategy.description,
            "results_count": len(reranked_results),
            "results": []
        }
        
        for i, doc in enumerate(reranked_results[:top_k], 1):
            metadata = doc.get('metadata', {})
            # 映射元数据字段：将数据库字段映射到标准字段
            mapped_metadata = {
                'section_title': metadata.get('chapter', '未知章节'),
                'pdf_filename': metadata.get('source', '未知文件'),
                'page_number': metadata.get('start_page', 0),
                'end_page': metadata.get('end_page', 0),
                'chunk_id': metadata.get('chunk_id', ''),
                'preview': metadata.get('preview', '')
            }
            
            result["results"].append({
                "rank": i,
                "document": doc.get('document', ''),
                "metadata": mapped_metadata,
                "score": doc.get('fused_score', doc.get('rerank_score', 0))
            })
        
        # 将检索结果保存到全局变量，供 Agent 返回使用
        # 注意：需要保存包含映射元数据的完整文档格式
        clear_last_retrieval_result()
        
        # 构建包含映射元数据的文档列表
        docs_to_save = []
        for i, doc in enumerate(reranked_results[:top_k], 1):
            metadata = doc.get('metadata', {})
            mapped_metadata = {
                'section_title': metadata.get('chapter', '未知章节'),
                'pdf_filename': metadata.get('source', '未知文件'),
                'page_number': metadata.get('start_page', 0),
                'end_page': metadata.get('end_page', 0),
                'chunk_id': metadata.get('chunk_id', ''),
                'preview': metadata.get('preview', '')
            }
            
            docs_to_save.append({
                "rank": i,
                "document": doc.get('document', ''),
                "metadata": mapped_metadata,
                "score": doc.get('fused_score', doc.get('rerank_score', 0))
            })
        
        _last_retrieval_result["docs"] = docs_to_save
        logger.info(f"[SmartRetrievalTool] 已保存 {len(_last_retrieval_result['docs'])} 篇文档到全局缓存（含映射元数据）")
        
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    async def _arun(self, 
                    query: str, 
                    query_type: str = "", 
                    keywords: List[str] = None,
                    top_k: int = 5) -> str:
        """异步执行（暂不实现）"""
        raise NotImplementedError("异步检索暂不支持")





def create_retrieval_tools() -> List[BaseTool]:
    """
    创建检索工具列表
    
    Returns:
        LangChain Tool 列表
    """
    return [
        SmartRetrievalTool(),
    ]
