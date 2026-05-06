"""
LangChain 检索工具封装

职责：
1. 将检索引擎封装为 LangChain 工具
2. 供 DeepSeek Agent 调用
3. 支持智能检索策略选择
"""
import logging
from typing import Optional, Dict, Any, List
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

# 全局变量：存储最后一次检索结果（用于 Agent 提取）
_last_retrieval_result = []


def get_last_retrieval_result() -> List[Dict[str, Any]]:
    """获取最后一次检索结果"""
    return _last_retrieval_result


class SmartRetrievalTool(BaseTool):
    """
    智能检索工具
    
    根据问题类型自动选择最优检索策略
    """
    
    name: str = "smart_retrieval"
    description: str = """智能检索工具，根据问题类型自动选择最优检索策略。
    
    参数：
    - query (str, 必需): 用户查询文本
    - keywords (List[str], 可选): 关键词列表，如果不提供则自动从 query 中提取
    - top_k (int, 可选): 返回结果数量，默认 5
    - query_type (str, 可选): 查询类型，由 Agent 分析后传递（如：故事梗概、事实依据、人物关系、段落情节）
    - metadata_filter (Dict[str, str], 可选): 元数据过滤条件，用于精确筛选特定文档
      * 常见用法：{"pdf_filename": "狂人日记.pdf"} 或 {"chapter": "第一章"}
      * 当用户明确提到某本书、某个章节时使用
    
    使用示例：
    # 示例 1: 普通检索
    smart_retrieval(query="朱元璋的家庭背景", keywords=["朱元璋", "家庭", "背景", "家境", "职业"], query_type="事实依据")
    
    # 示例 2: 带元数据过滤的检索（用户明确指定了书籍）
    smart_retrieval(query="狂人日记讲了什么", keywords=["狂人日记", "内容", "故事", "情节"], query_type="故事梗概", metadata_filter={"pdf_filename": "鲁迅短篇小说集：呐喊.pdf"})
    """
    
    def __init__(self):
        super().__init__()
        self._retriever = None
        logger.info("[SmartRetrievalTool] 初始化完成")
    
    def _get_retriever(self):
        """延迟加载检索器"""
        if self._retriever is None:
            from src.config import settings
            from src.rag.mixed_retrieval import HybridRetriever
            
            logger.info(f"[SmartRetrievalTool] 加载 HybridRetriever，db_path={settings.VECTOR_DB_PATH}")
            self._retriever = HybridRetriever(db_path=settings.VECTOR_DB_PATH)
        
        return self._retriever
    
    def _run(
        self,
        query: str,
        keywords: Optional[List[str]] = None,
        top_k: int = 5,
        query_type: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        执行智能检索
        
        Args:
            query: 用户查询
            keywords: 关键词列表（可选）
            top_k: 返回结果数量
            query_type: 查询类型（由 Agent 传递，用于标识问题类型）
            metadata_filter: 元数据过滤条件（可选），如 {"pdf_filename": "狂人日记.pdf"}
            
        Returns:
            格式化的检索结果字符串
        """
        global _last_retrieval_result
        
        try:
            logger.info(f"[SmartRetrievalTool] 执行检索：query='{query}', keywords={keywords}, top_k={top_k}, query_type={query_type}, metadata_filter={metadata_filter}")
            
            # 获取检索器
            retriever = self._get_retriever()
            
            # 如果没有提供关键词，让检索器自动分词
            if not keywords:
                logger.info("[SmartRetrievalTool] 未提供关键词，使用自动分词")
                keywords = retriever.tokenize_query(query)
            
            # 🔥 调用混合检索引擎（策略选择和重排序在引擎内部完成）
            results = retriever.search(
                query=query,
                keywords=keywords,
                top_k=top_k,
                use_mmr=True,  # 启用 MMR 多样性优化
                metadata_filter=metadata_filter,  # 🔥 传递元数据过滤
                query_type=query_type,  # 🔥 传递查询类型，引擎自动选择策略
                use_reranker=True  # 🔥 默认启用 Cross-Encoder 重排序
            )
            
            # 保存到最后一次检索结果（全局变量）
            _last_retrieval_result = results
            
            logger.info(f"[SmartRetrievalTool] 检索完成，找到 {len(results)} 条结果")
            
            # 格式化输出
            if not results:
                return "未找到相关文档。"
            
            formatted_results = []
            for i, doc in enumerate(results, 1):
                metadata = doc.get('metadata', {})
                formatted_doc = f"""
文档 {i}:
- 章节: {metadata.get('chapter', '未知')}
- 页码: {metadata.get('start_page', '?')}-{metadata.get('end_page', '?')}
- 文件名: {metadata.get('source', '未知')}
- 相关性分数: {doc.get('final_score', 0):.4f}
- 内容: {doc.get('document', '')[:500]}...
"""
                formatted_results.append(formatted_doc)
            
            return "\n".join(formatted_results)
            
        except Exception as e:
            error_msg = f"检索失败: {str(e)}"
            logger.error(f"[SmartRetrievalTool] {error_msg}", exc_info=True)
            return error_msg
    
    async def _arun(
        self,
        query: str,
        keywords: Optional[List[str]] = None,
        top_k: int = 5,
        query_type: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> str:
        """异步版本（目前同步实现）"""
        return self._run(query, keywords, top_k, query_type, metadata_filter)


class MetadataFilterDirectRetrievalTool(BaseTool):
    """
    元数据过滤直接召回工具
    
    通过元数据（source、chapter、start_page、end_page）直接从向量数据库中召回文本块，
    不进行相似度计算、重排序等复杂操作，适合精确定位特定文档内容。
    """
    
    name: str = "metadata_filter_retrieval"
    description: str = """元数据过滤直接召回工具，通过元数据精确筛选文本块。
    
    参数：
    - source (str, 可选): 源文件名，如 "鲁迅短篇小说集：呐喊.pdf"
    - chapter (str, 可选): 章节名称，如 "自序"
    - start_page (int, 可选): 起始页码
    - end_page (int, 可选): 结束页码
    - limit (int, 可选): 返回结果数量限制，默认 100
    
    使用示例：
    # 示例 1: 根据文件名和章节召回
    metadata_filter_retrieval(source="鲁迅短篇小说集：呐喊.pdf", chapter="自序")
    
    # 示例 2: 根据文件名和页码范围召回
    metadata_filter_retrieval(source="鲁迅短篇小说集：呐喊.pdf", start_page=5, end_page=10)
    
    # 示例 3: 组合条件
    metadata_filter_retrieval(source="鲁迅短篇小说集：呐喊.pdf", chapter="自序", start_page=5, end_page=10)
    """
    
    def __init__(self):
        super().__init__()
        self._retriever = None
        logger.info("[MetadataFilterDirectRetrievalTool] 初始化完成")
    
    def _get_retriever(self):
        """延迟加载检索器"""
        if self._retriever is None:
            from src.config import settings
            from src.rag.mixed_retrieval import HybridRetriever
            
            logger.info(f"[MetadataFilterDirectRetrievalTool] 加载 HybridRetriever，db_path={settings.VECTOR_DB_PATH}")
            self._retriever = HybridRetriever(db_path=settings.VECTOR_DB_PATH)
        
        return self._retriever
    
    def _run(
        self,
        source: Optional[str] = None,
        chapter: Optional[str] = None,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
        limit: int = 100
    ) -> str:
        """
        执行元数据过滤直接召回
        
        Args:
            source: 源文件名
            chapter: 章节名称
            start_page: 起始页码
            end_page: 结束页码
            limit: 返回结果数量限制
        
        Returns:
            格式化的检索结果字符串
        """
        global _last_retrieval_result
        
        try:
            # 构建元数据过滤条件
            metadata_filter = {}
            if source:
                metadata_filter['source'] = source
            if chapter:
                metadata_filter['chapter'] = chapter
            if start_page:
                metadata_filter['start_page'] = start_page
            if end_page:
                metadata_filter['end_page'] = end_page
            
            logger.info(f"[MetadataFilterDirectRetrievalTool] 执行元数据过滤：{metadata_filter}, limit={limit}")
            
            # 获取检索器
            retriever = self._get_retriever()
            
            # 🔥 调用纯元数据过滤检索（直接召回，不进行相似度计算、重排序）
            results = retriever.get_by_metadata(metadata_filter, limit)
            
            # 保存到最后一次检索结果（全局变量）
            _last_retrieval_result = results
            
            logger.info(f"[MetadataFilterDirectRetrievalTool] 检索完成，找到 {len(results)} 条结果")
            
            # 格式化输出
            if not results:
                return "未找到符合条件的文档。"
            
            formatted_results = []
            for i, doc in enumerate(results, 1):
                metadata = doc.get('metadata', {})
                formatted_doc = f"""
文档 {i}:
- 章节: {metadata.get('chapter', '未知')}
- 页码: {metadata.get('start_page', '?')}-{metadata.get('end_page', '?')}
- 文件名: {metadata.get('source', '未知')}
- 内容: {doc.get('document', '')[:500]}...
"""
                formatted_results.append(formatted_doc)
            
            return "\n".join(formatted_results)
            
        except Exception as e:
            error_msg = f"检索失败: {str(e)}"
            logger.error(f"[MetadataFilterDirectRetrievalTool] {error_msg}", exc_info=True)
            return error_msg
    
    async def _arun(
        self,
        source: Optional[str] = None,
        chapter: Optional[str] = None,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
        limit: int = 100
    ) -> str:
        """异步版本（目前同步实现）"""
        return self._run(source, chapter, start_page, end_page, limit)


# 导出工具实例（单例）
smart_retrieval_tool = SmartRetrievalTool()
metadata_filter_retrieval_tool = MetadataFilterDirectRetrievalTool()


if __name__ == "__main__":
    """测试工具"""
    import sys
    from pathlib import Path
    
    # 添加项目根目录到路径
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # 测试检索
    tool = SmartRetrievalTool()
    
    test_query = "朱元璋的家庭背景"
    print(f"\n测试查询: {test_query}\n")
    
    result = tool._run(query=test_query, top_k=3)
    print(result)
