"""混合检索：向量 + 关键词 + 精排，支持中/日/英分词匹配"""

import numpy as np
import logging
import torch
import os
import glob
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from embedding_vector import DocumentChunk, SearchResult, VectorDatabase, EmbeddingModel

logger = logging.getLogger(__name__)

# 轻量分词：自动识别中/日/英并返回 token 列表（有 jieba/fugashi 时优先使用）
try:
    import jieba
except Exception:
    jieba = None
try:
    from fugashi import Tagger as MeCabTagger
    _mecab = MeCabTagger()
except Exception:
    _mecab = None
import re

def tokenize_text(text: str, lang_hint: Optional[str] = None) -> List[str]:
    if not text:
        return []
    txt = text.strip().lower()
    # 简单语言探测
    if (lang_hint or "").lower().startswith("ch") or re.search(r'[\u4e00-\u9fff]', txt):
        if jieba:
            return [t for t in jieba.cut(txt) if t.strip()]
        return [ch for ch in txt if ch.strip()]
    if (lang_hint or "").lower().startswith("ja") or re.search(r'[\u3040-\u30ff]', txt):
        if _mecab:
            return [word.surface for word in _mecab(txt)]
        return [ch for ch in txt if ch.strip()]
    return re.findall(r"[a-z0-9']{1,}", txt)

class RerankerModel:
    """Reranker精排模型"""
    
    def __init__(self, model_name: str = None):
        """
        初始化Reranker模型
        
        Args:
            model_name: 模型名称，支持以下选项：
                - "jina-reranker-v3-base": Jina的Reranker v3基础版，支持多语言和长上下文
                - "bce-reranker-base_v1": 支持中英日韩的BCEReranker
                - "BAAI/bge-reranker-v2-m3": BGE开源Reranker v2
        """
        self.model_name = model_name or os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
        self.model = None
        self.device = self._setup_device()
        self._load_model()
    
    def _setup_device(self) -> str:
        """设置计算设备"""
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"Reranker使用GPU加速: {torch.cuda.get_device_name()}")
        else:
            device = "cpu"
            logger.info("Reranker使用CPU计算")
        return device
    
    def _load_model(self):
        """加载Reranker模型"""
        try:
            from sentence_transformers import CrossEncoder
            
            logger.info(f"正在加载Reranker模型: {self.model_name}")
            self.model = CrossEncoder(self.model_name, device=self.device)
            logger.info(f"Reranker模型加载成功: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Reranker模型加载失败: {e}")
            self.model = None
    
    def rerank(self, query: str, candidates: List[SearchResult], top_k: int = 10) -> List[SearchResult]:
        """
        使用Reranker对候选结果进行精排
        
        Args:
            query: 用户查询
            candidates: 候选搜索结果列表
            top_k: 返回top_k个结果
            
        Returns:
            重排序后的结果列表
        """
        if not self.model or not candidates:
            return candidates[:top_k]
        
        try:
            # 准备查询-文档对
            pairs = [(query, candidate.text) for candidate in candidates]
            
            # 使用CrossEncoder计算相关性分数
            scores = self.model.predict(pairs)
            
            # 将分数添加到结果中并重新排序
            for candidate, score in zip(candidates, scores):
                candidate.rerank_score = float(score)
                # 综合原始分数和Reranker分数
                candidate.score = 0.3 * candidate.score + 0.7 * score
            
            # 按最终分数排序
            reranked = sorted(candidates, key=lambda x: x.score, reverse=True)
            
            logger.info(f"Reranker精排完成，从 {len(candidates)} 个候选结果中选出 {min(top_k, len(reranked))} 个")
            return reranked[:top_k]
            
        except Exception as e:
            logger.error(f"Reranker精排失败: {e}")
            return candidates[:top_k]

def scan_source_files(source_dir: str = "src/data/source") -> List[Dict[str, Any]]:
    """
    扫描指定目录下的所有PDF文件，构建文件名列表
    
    Args:
        source_dir: 源文件目录路径
        
    Returns:
        包含文件名、路径和章节标题信息的字典列表
    """
    if not os.path.exists(source_dir):
        logger.warning(f"源文件目录不存在: {source_dir}")
        return []
    
    # 扫描目录下的所有PDF文件
    pdf_files = glob.glob(os.path.join(source_dir, "**", "*.pdf"), recursive=True)
    
    file_list = []
    for pdf_path in pdf_files:
        # 获取相对路径（相对于项目根目录）
        relative_path = os.path.relpath(pdf_path, os.getcwd())
        
        # 获取文件名
        filename = os.path.basename(pdf_path)
        
        file_list.append({
            "filename": filename,
            "full_path": pdf_path,
            "relative_path": relative_path,
            "section_titles": []  # 章节标题列表，后续会填充
        })
    
    logger.info(f"扫描完成，找到 {len(file_list)} 个PDF文件")
    return file_list


def build_file_section_index(json_dir: str = "src/data/pages_title") -> Dict[str, Dict[str, Any]]:
    """
    遍历 JSON 目录，提取每个文件的 filename 和不重复的 section_title -> set(page_numbers) 映射。
    返回结构：
      {
         "filename.pdf": {
             "path": "/abs/path/to/json",
             "sections": { "section_title1": {1,2}, "section_title2": {5,6} }
         },
         ...
      }
    """
    index = {}
    json_paths = glob.glob(os.path.join(json_dir, "**", "*.json"), recursive=True)
    for p in json_paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        parent = data.get("parent_document", {})
        filename = parent.get("filename") or Path(p).stem
        chunks = data.get("document_chunks", [])
        sections = {}
        sections_tokens = {}
        for c in chunks:
            title = (c.get("section_title") or "").strip()
            page = c.get("page_number")
            if not title:
                continue
            if title not in sections:
                sections[title] = set()
            try:
                if page:
                    sections[title].add(int(page))
            except Exception:
                pass
            # 计算 token set 用于后续匹配（按父文档语言提示）
            lang = (parent.get("language") or "").strip()
            toks = tokenize_text(title, lang_hint=lang)
            sections_tokens[title] = set(toks)

        index[filename] = {
            "path": p,
            "sections": sections,
            "sections_tokens": sections_tokens
        }
    logger.info(f"构建文件-章节索引，找到 {len(index)} 个文件")
    return index


class EnhancedMetadataFilter:
    """基于 pages_title JSON 构建的元数据索引与查询工具"""
    def __init__(self, json_dir: str = "src/data/pages_title", index_path: str = "src/data/pages_title/sections_index.json", force_rebuild: bool = False):
        self.json_dir = json_dir
        self.index_path = index_path
        # 若存在持久化索引且不强制重建，则尝试加载
        if not force_rebuild and Path(self.index_path).exists():
            try:
                with open(self.index_path, "r", encoding="utf-8") as f:
                    raw = json.load(f) # 加载原始索引数据，元组 (fname, info)
                # 恢复 sets
                idx = {}
                for fname, info in raw.items():
                    sections = {k: set(v) for k, v in info.get("sections", {}).items()} #确保即使JSON数据不完整或格式异常，程序也能优雅处理，避免崩溃
                    sections_tokens = {k: set(v) for k, v in info.get("sections_tokens", {}).items()}
                    idx[fname] = {
                        "path": info.get("path"),
                        "sections": sections,
                        "sections_tokens": sections_tokens
                    }
                # 验证加载的索引数据完整性
                if not self._validate_loaded_index(idx):
                    logger.warning("加载的索引数据不完整，将重新构建")
                    raise ValueError("索引数据验证失败，触发重建")
                
                # 验证通过，使用加载的索引
                self.index = idx
                logger.info(f"已从持久化索引加载 metadata index: {self.index_path}")
                return
            except Exception as e:
                logger.warning(f"加载持久化索引失败，改为重建: {e}")

        # 否则重建并持久化
        self.index = build_file_section_index(json_dir)
        try:
            to_save = {}
            for fname, info in self.index.items():
                to_save[fname] = {
                    "path": info.get("path"),
                    "sections": {k: sorted(list(v)) for k, v in info.get("sections", {}).items()},
                    "sections_tokens": {k: sorted(list(v)) for k, v in info.get("sections_tokens", {}).items()}
                }
            Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.index_path, "w", encoding="utf-8") as f:
                json.dump(to_save, f, ensure_ascii=False, indent=2)
            logger.info(f"已持久化 metadata index 到: {self.index_path}")
        except Exception as e:
            logger.warning(f"持久化 metadata index 失败: {e}")

    def _validate_loaded_index(self, idx: Dict) -> bool:
        """验证加载的索引数据是否完整有效"""
        if not isinstance(idx, dict):
            return False
        
        for fname, info in idx.items():
            if not isinstance(info, dict):
                return False
            if "path" not in info or "sections" not in info or "sections_tokens" not in info:
                return False
            if not isinstance(info["sections"], dict) or not isinstance(info["sections_tokens"], dict):
                return False
        return True

    def get_index(self) -> Dict[str, Dict[str, Any]]:
        return self.index

    def match_query_tokens(self, q_tokens: set, threshold: float = 0.5) -> Tuple[List[str], List[str]]:
        """
        根据 query token 集合匹配文件名与 section title，返回 (matched_filenames, matched_section_titles)
        """
        matched_files = []
        matched_sections = []
        for fname, info in self.index.items():
            name_no_ext = fname.rsplit(".", 1)[0]
            name_tokens = set(tokenize_text(name_no_ext))
            if q_tokens and name_tokens:
                name_overlap = len(q_tokens & name_tokens) / max(1, len(q_tokens))
                if name_overlap >= threshold:
                    matched_files.append(fname)
                    matched_sections.extend(list(info.get("sections", {}).keys()))
                    continue
            for title, stokens in info.get("sections_tokens", {}).items():
                if not stokens:
                    continue
                overlap = len(q_tokens & stokens) / max(1, len(q_tokens))
                if overlap >= threshold:
                    matched_sections.append(title)
                    matched_files.append(fname)
        # dedupe
        return list(dict.fromkeys(matched_files)), list(dict.fromkeys(matched_sections))



class VectorRetriever:
    """向量检索器 - 支持三阶段RAG检索流程"""
    def __init__(self, vector_db: VectorDatabase, embedding_model: EmbeddingModel, 
                 use_reranker: bool = True, reranker_model: str = "BAAI/bge-reranker-v2-m3",
                 source_dir: str = "src/data/source", metadata_json_dir: str = "src/data/pages_title"):
        self.vector_db = vector_db #向量数据库实例，用于存储和检索文档块向量
        self.embedding_model = embedding_model #嵌入模型实例，用于将文本转换为向量
        self.use_reranker = use_reranker #是否使用Reranker模型
        self.reranker = None
        self.source_dir = source_dir
        # 初始化元数据过滤器
        try:
            self.metadata_filter = EnhancedMetadataFilter(metadata_json_dir)
        except Exception as e:
            logger.warning(f"元数据过滤器初始化失败: {e}")
            self.metadata_filter = None
        self.chunk_count = 0
        
        # 初始化Reranker模型
        if use_reranker:
            try:
                self.reranker = RerankerModel(reranker_model)
                logger.info(f"Reranker已启用: {reranker_model}")
            except Exception as e:
                logger.warning(f"Reranker初始化失败: {e}，将使用传统检索")
                self.use_reranker = False
        
        # 先加载文档块
        self._load_chunks()
    
    def _load_chunks(self):
        """加载文档块信息，不再将所有文档块加载到内存"""
        # 不再将所有文档块加载到内存，而是在需要时直接调用ChromaDB
        # 只获取文档块数量用于统计
        chunks = self.vector_db.get_all_chunks()
        self.chunk_count = len(chunks)
        logger.info(f"总共 {self.chunk_count} 个文档块")
    
    
    def get_file_list(self):
        """获取当前文件名列表"""
        try:
            logger.warning("file_list_manager 已移除，get_file_list 不可用，请使用 build_file_section_index 或外部方式获取文件列表。")
            return []
        except ValueError:
            logger.warning("文件名列表管理器未初始化")
            return []
    
    def retrieve_relevant_filename(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        根据用户查询检索最相关的文件名
        
        Args:
            query: 用户查询
            top_k: 返回最相关的前k个文件名
            
        Returns:
            包含文件名、路径和相似度的字典列表
        """
        try:
            logger.warning("file_list_manager 已移除，retrieve_relevant_filename 使用 fallback 策略")
            # fallback: 使用简单的索引扫描（基于 build_file_section_index）
            index = build_file_section_index()
            # 简单评分：文件名 token overlap
            q_tokens = set(tokenize_text(query))
            scores = []
            for fname in index.keys():
                name_tokens = set(tokenize_text(fname.rsplit('.',1)[0]))
                score = len(q_tokens & name_tokens) / max(1, len(q_tokens))
                scores.append((fname, score))
            scores.sort(key=lambda x: x[1], reverse=True)
            return [{"filename": s[0], "score": s[1]} for s in scores[:top_k]]
        except ValueError:
            logger.warning("文件名列表管理器未初始化")
            return []
    
    def hybrid_search(self, query: str, top_k: int = 60, alpha: float = 0.7, 
                     candidate_top_k: int = 120, use_filename_filter: bool = True) -> List[SearchResult]:
        """
        三阶段混合检索：
        阶段0：文件名相关性过滤（可选）
        阶段1：混合召回（向量+关键词）
        阶段2：Reranker精排
        阶段3：返回最终结果
        
        Args:
            query: 用户查询
            top_k: 最终返回结果数量
            alpha: 向量检索权重 (0-1)，关键词权重 = 1-alpha
            candidate_top_k: 候选结果数量（用于Reranker）
            use_filename_filter: 是否使用文件名相关性过滤
        """
        # 阶段0：尝试使用元数据过滤（基于 JSON 索引）
        filter_conditions = None
        if use_filename_filter and self.metadata_filter:
            try:
                q_tokens = set(tokenize_text(query))
                matched_files, matched_sections = self.metadata_filter.match_query_tokens(q_tokens, threshold=0.5)
                # 支持解析页码范围，如 "第10-20页"
                page_range = None
                import re
                page_match = re.search(r'[第]?\s*(\d+)\s*[-至到]\s*(\d+)\s*[页]?', query)
                if page_match:
                    page_range = (int(page_match.group(1)), int(page_match.group(2)))

                if matched_files or matched_sections or page_range:
                    ors = []
                    for t in matched_sections:
                        ors.append({"section_title": t})
                    for fn in matched_files:
                        # ChromaDB where operators do not support $contains; use exact match on pdf_filename
                        ors.append({"pdf_filename": {"$eq": fn}})
                    if ors:
                        filter_conditions = {"$or": ors}
                    if page_range:
                        start, end = page_range
                        page_filter = {"$and": [{"page_number": {"$gte": start}}, {"page_number": {"$lte": end}}]}
                        if filter_conditions:
                            filter_conditions = {"$and": [filter_conditions, page_filter]}
                        else:
                            filter_conditions = page_filter
                    logger.info(f"使用元数据过滤: files={matched_files}, sections={matched_sections}, page_range={page_range}")
            except Exception as e:
                logger.warning(f"元数据过滤失败，回退到传统文件名过滤: {e}")
                filter_conditions = None
        
        # 阶段1：混合召回 - 获取更多候选结果
        candidate_k = max(candidate_top_k, top_k * 3)  # 确保有足够候选
        
        # 生成查询向量
        query_vector = self.embedding_model.encode([query])[0]
        
        # 直接使用ChromaDB进行向量搜索
        candidates = self._vector_search(query_vector, candidate_k, filter_conditions)
        
        logger.info(f"混合检索候选结果：{len(candidates)} 个")
        
        # 阶段2：Reranker精排（如果启用）
        if self.use_reranker and self.reranker and len(candidates) > top_k:
            candidates = self.reranker.rerank(query, candidates, candidate_top_k)
            logger.info(f"Reranker精排后候选结果：{len(candidates)} 个")
        
        # 阶段3：返回最终Top K结果
        final_results = candidates[:top_k]
        logger.info(f"最终返回结果：{len(final_results)} 个")
        
        return final_results
    
    def _vector_search(self, query_vector: np.ndarray, top_k: int, filter_conditions: Optional[Dict] = None) -> List[SearchResult]:
        """向量相似度搜索 - 直接调用ChromaDB的搜索功能"""
        # 直接使用ChromaDB进行向量搜索
        search_results = self.vector_db.search_similar(
            query_vector=query_vector.tolist(),  # 转换为列表格式
            top_k=top_k,
            filter_conditions=filter_conditions
        )
        
        return search_results

    def smart_search(self, query: str, top_k: int = 100, page_range: Optional[Tuple[int, int]] = None, json_dir: str = "src/data/pages_title") -> List[SearchResult]:
        """
        智能检索入口：优先判断 query 是否直接与文件名或章节标题相关（使用 JSON 索引）。
        - 若匹配到文件名或章节标题：基于这些 section_title 构建过滤条件并直接在向量库中检索（可选页码范围）。
        - 否则：回退到 hybrid_search（向量+关键词+Reranker）。
        返回最终 top_k 个结果（默认 100）。
        """
        try:
            # 构建文件-章节索引
            index = build_file_section_index(json_dir)
            q_lower = query.lower()

            matched_filenames = []
            matched_section_titles = []

            # 使用全文分词匹配（token overlap）判断是否与 filename/section_title 直接相关
            q_tokens = set(tokenize_text(query))
            match_threshold = 0.5  # query token 覆盖率阈值

            for fname, info in index.items():
                name_no_ext = fname.rsplit(".", 1)[0]
                name_tokens = set(tokenize_text(name_no_ext))

                # 文件名匹配：query tokens 覆盖率
                if q_tokens and name_tokens:
                    name_overlap = len(q_tokens & name_tokens) / max(1, len(q_tokens))
                    if name_overlap >= match_threshold:
                        matched_filenames.append(fname)
                        matched_section_titles.extend(list(info["sections"].keys()))
                        continue

                # 部分标题匹配：使用预计算的 sections_tokens
                for title, stokens in info.get("sections_tokens", {}).items():
                    if not stokens:
                        continue
                    overlap = len(q_tokens & stokens) / max(1, len(q_tokens))
                    if overlap >= match_threshold:
                        matched_section_titles.append(title)
                        matched_filenames.append(fname)

            # 去重
            matched_section_titles = list(dict.fromkeys(matched_section_titles))
            matched_filenames = list(dict.fromkeys(matched_filenames))

            if not matched_section_titles and not matched_filenames:
                # fallback to hybrid vector+reranker search (固定返回 top_k=60，以保证足够上下文)
                logger.info("未检测到与文件名/章节直接相关的匹配，使用 hybrid_search 回退 (top_k=60)")
                return self.hybrid_search(query, top_k=60)

            # 构建 ChromaDB 的过滤条件
            where_clause = None
            ors = []
            if matched_section_titles:
                for t in matched_section_titles:
                    ors.append({"section_title": t})
            if matched_filenames:
                for fn in matched_filenames:
                    # Use exact match on stored pdf_filename metadata (more reliable than substring)
                    ors.append({"pdf_filename": {"$eq": fn}})

            if ors:
                where_clause = {"$or": ors}

            # 加入页码范围过滤（如果指定）
            if page_range and isinstance(page_range, tuple) and len(page_range) == 2:
                start, end = page_range
                page_filter = {"$and": [{"page_number": {"$gte": start}}, {"page_number": {"$lte": end}}]}
                if where_clause:
                    where_clause = {"$and": [where_clause, page_filter]}
                else:
                    where_clause = page_filter

            # 向量检索
            query_vector = self.embedding_model.encode([query])[0]
            candidates = self._vector_search(query_vector, top_k, where_clause)

            # rerank if enabled
            if self.use_reranker and self.reranker and candidates:
                candidates = self.reranker.rerank(query, candidates, top_k)

            return candidates[:top_k]

        except Exception as e:
            logger.error(f"smart_search 执行失败: {e}")
            return self.hybrid_search(query, top_k=top_k)

    def vector_only_search(self, query: str, top_k: int = 10, 
                          candidate_top_k: int = 30) -> List[SearchResult]:
        """纯向量检索（支持Reranker精排）"""
        # 第一阶段：向量检索获取候选结果
        query_vector = self.embedding_model.encode([query])[0]
        candidates = self._vector_search(query_vector, candidate_top_k)
        
        if not candidates:
            return []
        
        # 第二阶段：Reranker精排
        if self.use_reranker and self.reranker and len(candidates) > top_k:
            candidates = self.reranker.rerank(query, candidates, candidate_top_k)
        
        # 返回最终结果
        return candidates[:top_k]
    
    def keyword_only_search(self, query: str, top_k: int = 10, 
                           candidate_top_k: int = 30) -> List[SearchResult]:
        """纯关键词检索（支持Reranker精排）"""
        # 使用向量搜索获取候选结果，然后进行关键词过滤
        query_vector = self.embedding_model.encode([query])[0]
        candidates = self._vector_search(query_vector, candidate_top_k * 3)
        
        if not candidates:
            return []
        
        # 关键词过滤：基于文本内容的关键词匹配
        query_words = set(query.lower().split())
        filtered_candidates = []
        
        for candidate in candidates:
            doc_words = set(candidate.text.lower().split())
            overlap = len(query_words & doc_words)
            
            if overlap > 0:
                # 计算关键词匹配分数
                keyword_score = overlap / max(len(query_words), 1)
                # 更新分数
                candidate.score = keyword_score
                filtered_candidates.append(candidate)
        
        # 如果没有过滤出结果，使用原始候选结果
        if not filtered_candidates:
            filtered_candidates = candidates
        
        # 按关键词分数排序
        filtered_candidates.sort(key=lambda x: x.score, reverse=True)
        filtered_candidates = filtered_candidates[:candidate_top_k]
        
        # 第二阶段：Reranker精排
        if self.use_reranker and self.reranker and len(filtered_candidates) > top_k:
            filtered_candidates = self.reranker.rerank(query, filtered_candidates, candidate_top_k)
        
        # 返回最终结果
        return filtered_candidates[:top_k]



    def get_stats(self) -> Dict[str, Any]:
        """获取检索器统计信息"""
        return {
            "total_chunks": self.chunk_count
        }
