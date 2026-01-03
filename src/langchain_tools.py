"""
LangChain工具封装 - 将RAG系统集成到LangChain框架中
实现三个核心工具供DeepSeek模型智能调用
"""

import json
import logging
import math
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain.tools import tool
import numpy as _np

from mixed_retrieval import VectorRetriever
from embedding_vector import VectorDatabase, EmbeddingModel, RerankerModel

logger = logging.getLogger(__name__)

class QueryRequest:
    """查询请求封装类 - 避免Agent误解参数"""
    def __init__(self, query: str, filter_pdf: str = None):
        self.query = query
        self.filter_pdf = filter_pdf

class DocumentRetrievalTools:
    """文档检索工具集合 - LangChain工具封装"""
    
    def _load_sections_index(self, index_path: str = "src/data/pages_title/sections_index.json") -> Dict[str, Dict[str, Any]]:
        """加载已持久化的 sections_index（若存在），否则返回空字典"""
        p = Path(index_path)
        if not p.exists():
            logger.info(f"sections_index 未找到: {index_path}")
            return {}
        try:
            with open(p, "r", encoding="utf-8") as f:
                raw = json.load(f)
            # convert lists back to sets for sections values
            idx = {}
            for fname, info in raw.items():
                sections = {k: set(v) for k, v in info.get("sections", {}).items()}
                sections_tokens = {k: set(v) for k, v in info.get("sections_tokens", {}).items()}
                idx[fname] = {
                    "path": info.get("path"),
                    "sections": sections,
                    "sections_tokens": sections_tokens
                }
            return idx
        except Exception as e:
            logger.warning(f"加载 sections_index 失败: {e}")
            return {}

    def _is_valid_section_title(self, title: str) -> bool:
        """判断章节标题是否为有效文本标题（过滤掉纯数字/年份/页码样式）"""
        import re
        if not title or not title.strip():
            return False
        t = title.strip()
        if re.search(r'[\u4e00-\u9fffA-Za-z]', t):
            stripped = re.sub(r'[\s\-\._，。,、：:\d一二三四五六七八九十百千零壹贰叁肆伍陆柒捌玖]', '', t)
            return len(stripped) > 0
        return False
    
    def __init__(self, vector_db_path: str, embedding_model_name: str = "intfloat/multilingual-e5-large",
                 vector_db: VectorDatabase = None, embedding_model: EmbeddingModel = None,
                 reranker_model: RerankerModel = None):
        """
        初始化检索工具

        Args:
            vector_db_path: 向量数据库路径
            embedding_model_name: 嵌入模型名称
            vector_db: 已初始化的VectorDatabase实例（可选，用于复用）
            embedding_model: 已初始化的EmbeddingModel实例（可选，用于复用）
            reranker_model: 已初始化的RerankerModel实例（可选，用于复用）
        """
        self.vector_db_path = vector_db_path
        self.vector_db = vector_db if vector_db is not None else VectorDatabase(vector_db_path)
        self.embedding_model = embedding_model if embedding_model is not None else EmbeddingModel(embedding_model_name)
        self.retriever = VectorRetriever(self.vector_db, self.embedding_model)
        self.reranker_model = reranker_model if reranker_model is not None else RerankerModel()
        # 加载 sections index 并预计算/持久化文件名/章节向量以供智能匹配使用
        self.sections_index = self._load_sections_index()
        self.all_filenames = list(self.sections_index.keys())
        # 提取所有非纯数字章节标题
        self.all_section_titles = []
        for info in self.sections_index.values():
            for title in info.get("sections", {}).keys():
                if title and self._is_valid_section_title(title):
                    self.all_section_titles.append(title)

        # 匹配阈值（可通过环境变量覆盖）
        try:
            self.file_match_threshold = float(os.getenv("FILE_MATCH_THRESHOLD", "0.7"))
        except Exception:
            self.file_match_threshold = 0.7
        try:
            self.section_match_threshold = float(os.getenv("SECTION_MATCH_THRESHOLD", "0.65"))
        except Exception:
            self.section_match_threshold = 0.65
        try:
            self.section_rerank_alpha = float(os.getenv("SECTION_RERANK_ALPHA", "0.5"))
        except Exception:
            self.section_rerank_alpha = 0.5

        # 尝试从磁盘加载预计算的 embeddings（提高启动速度）
        emb_cache_dir = Path(os.getenv("EMB_CACHE_DIR", "src/data/pages_title"))
        cache_dir = emb_cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        meta_path = cache_dir / "sections_embeddings_meta.json"
        npz_path = cache_dir / "sections_embeddings.npz"

        self.filename_embeddings = None
        self.section_title_embeddings = None
        # 新增：按文件组织的章节数据
        self.file_sections = {}  # {filename: {"section_titles": [], "section_embeddings": np.array, "section_indices": []}}

        try:
            if meta_path.exists() and npz_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)

                npz = _np.load(str(npz_path))

                # 尝试加载新格式的向量数据
                if "file_sections" in meta:
                    # 新格式：按文件组织的章节数据
                    for filename, section_info in meta["file_sections"].items():
                        key = f"{filename}_section_embeddings"
                        if key in npz:
                            self.file_sections[filename] = {
                                "section_titles": section_info["section_titles"],
                                "section_embeddings": npz[key],
                                "section_indices": section_info.get("section_indices", [])
                            }

                    # 加载文件名向量
                    if "filename_embeddings" in npz:
                        self.filename_embeddings = npz["filename_embeddings"]

                    # 验证加载结果
                    if not self.file_sections:
                        logger.error("文件章节数据为空，meta中未找到有效的 section_titles")
                        raise RuntimeError("加载章节标题向量失败：文件章节数据为空")

                else:
                    # 旧格式：不支持，跳过兼容逻辑直接报错
                    logger.error("未检测到新格式的 file_sections 数据结构")
                    raise RuntimeError("加载章节标题向量失败：meta中缺少 file_sections 字段")

        except Exception as e:
            logger.info(f"加载 embeddings 缓存失败，稍后尝试重新计算: {e}")

        # 如未加载到新格式embeddings，则计算并持久化
        if not self.file_sections:
            logger.info("开始计算章节标题向量...")
            try:
                self._compute_and_save_embeddings_new_format()
            except Exception as e:
                logger.warning(f"计算新格式embeddings失败: {e}")
                try:
                    self._convert_old_format_to_new()
                except Exception as e2:
                    logger.error(f"转换旧格式也失败: {e2}")

        self._tools = self._create_tools()
        logger.info(f"文档检索工具初始化完成，数据库路径: {self.vector_db_path}")

    def _compute_and_save_embeddings_new_format(self):
        """计算并保存新格式的embeddings"""
        emb_cache_dir = Path(os.getenv("EMB_CACHE_DIR", "src/data/pages_title"))
        cache_dir = emb_cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        meta_path = cache_dir / "sections_embeddings_meta.json"
        npz_path = cache_dir / "sections_embeddings.npz"

        # 计算文件名embeddings
        if self.all_filenames:
            try:
                self.filename_embeddings = self.embedding_model.encode(self.all_filenames)
            except Exception as e:
                logger.warning(f"计算 filename embeddings 失败: {e}")
                self.filename_embeddings = None

        # 计算按文件组织的章节embeddings
        save_meta = {"filenames": self.all_filenames, "file_sections": {}}
        save_kwargs = {}

        if self.filename_embeddings is not None:
            save_kwargs["filename_embeddings"] = _np.array(self.filename_embeddings)

        for filename, info in self.sections_index.items():
            section_titles = []
            for title in info.get("sections", {}).keys():
                if title and self._is_valid_section_title(title):
                    section_titles.append(title)

            if section_titles:
                try:
                    section_embeddings = self.embedding_model.encode(section_titles)
                    self.file_sections[filename] = {
                        "section_titles": section_titles,
                        "section_embeddings": section_embeddings,
                        "section_indices": list(range(len(section_titles)))
                    }

                    # 为每个文件保存章节embeddings
                    save_kwargs[f"{filename}_section_embeddings"] = _np.array(section_embeddings)

                    save_meta["file_sections"][filename] = {
                        "section_titles": section_titles,
                        "section_indices": list(range(len(section_titles)))
                    }

                except Exception as e:
                    logger.warning(f"计算 {filename} 的章节embeddings失败: {e}")

        # 保存到磁盘
        try:
            if save_kwargs:
                _np.savez_compressed(str(npz_path), **save_kwargs)
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(save_meta, f, ensure_ascii=False)
                logger.info("新格式embeddings已保存到磁盘")
        except Exception as e:
            logger.warning(f"保存新格式embeddings失败: {e}")

    def _convert_old_format_to_new(self):
        """将旧格式的全局章节向量转换为新格式的按文件组织"""
        if not self.section_title_embeddings or not self.all_section_titles:
            return

        logger.info("开始转换旧格式embeddings到新格式...")

        # 重建按文件的章节映射
        section_to_file = {}
        for filename, info in self.sections_index.items():
            for title in info.get("sections", {}).keys():
                if title and self._is_valid_section_title(title):
                    section_to_file[title] = filename

        # 按文件重新组织embeddings
        file_section_data = {}
        for i, title in enumerate(self.all_section_titles):
            filename = section_to_file.get(title)
            if filename:
                if filename not in file_section_data:
                    file_section_data[filename] = {"titles": [], "embeddings": []}
                file_section_data[filename]["titles"].append(title)
                file_section_data[filename]["embeddings"].append(self.section_title_embeddings[i])

        # 更新数据结构
        for filename, data in file_section_data.items():
            self.file_sections[filename] = {
                "section_titles": data["titles"],
                "section_embeddings": _np.array(data["embeddings"]),
                "section_indices": list(range(len(data["titles"])))
            }

        logger.info(f"成功转换 {len(self.file_sections)} 个文件的章节数据")

        # 保存新格式
        try:
            self._compute_and_save_embeddings_new_format()
        except Exception as e:
            logger.warning(f"转换旧格式到新格式失败: {e}")
    
    def _create_tools(self):
        """创建LangChain工具实例"""
        tools = []
        
        def _compute_top_match(query_vec, candidate_embeddings, candidate_texts, top_k: int = 1):
            """基于余弦相似度，返回 top_k 候选 (text, score) 列表"""
            import numpy as _np
            if candidate_embeddings is None or len(candidate_texts) == 0:
                return []
            emb = _np.array(candidate_embeddings)
            q = _np.array(query_vec, dtype=float)
            try:
                emb_norm = emb / (_np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
                q_norm = q / (_np.linalg.norm(q) + 1e-12)
            except Exception:
                emb_norm = emb
                q_norm = q
            sims = (_np.dot(emb_norm, q_norm)).tolist()
            ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[:top_k]
            results = []
            for idx, score in ranked:
                results.append((candidate_texts[idx], float(score)))
            return results

        def _normalize_score(score: float, min_score: float, max_score: float) -> float:
            """将分数归一化到 0-1 范围"""
            score_range = max_score - min_score
            if score_range <= 0:
                return 0.5
            return (score - min_score) / score_range

        def _compute_top_match_with_rerank(query: str, query_vec, candidate_embeddings, candidate_texts, top_k: int = 5):
            """
            先进行向量检索获取 top_k 结果，再使用 reranker 重排
            综合分数 = 0.4 * 向量相似度分数 + 0.6 * 重排分数
            返回分数 >= 0.75 的最佳匹配
            """
            if candidate_embeddings is None or len(candidate_texts) == 0:
                return None

            # Step 1: 向量检索获取 top-5
            vector_results = _compute_top_match(query_vec, candidate_embeddings, candidate_texts, top_k=top_k)
            print(f"[vector_retrieval] 原始向量检索结果: {vector_results}")
            if not vector_results:
                return None

            # 提取向量检索结果
            vector_docs = [item[0] for item in vector_results]
            vector_scores = [item[1] for item in vector_results]

            # Step 2: Reranker 重排
            try:
                rerank_results = self.reranker_model.rerank(query, vector_docs, top_k=top_k)
                print(f"[rerank] 重排结果: {rerank_results}")
                # rerank_results 格式: [(index, rerank_score), ...]
            except Exception as e:
                logger.warning(f"Reranker 调用失败，回退到纯向量检索: {e}")
                return (vector_results[0][0], vector_results[0][1]) if vector_results else None

            # Step 3: 构建 rerank 分数映射
            rerank_score_map = {idx: score for idx, score in rerank_results}

            # Step 4: 分数融合
            min_vec_score = min(vector_scores)
            max_vec_score = max(vector_scores)

            best_result = None
            best_combined_score = 0.0

            for idx, (doc, vec_score) in enumerate(zip(vector_docs, vector_scores)):
                rerank_score = rerank_score_map.get(idx, 0.0)

                # 归一化向量分数到 0-1 范围
                normalized_vec_score = _normalize_score(vec_score, min_vec_score, max_vec_score)

                # 归一化 rerank 分数（BGE reranker 输出 logits，可能为负数）
                # 使用 sigmoid 函数将 logits 转换为 0-1 概率范围
                # sigmoid(x) = 1 / (1 + exp(-x))
                # 这样负数分数会映射到 0-0.5，正数分数映射到 0.5-1
                normalized_rerank_score = 1.0 / (1.0 + math.exp(-rerank_score))

                # 融合分数：(1 - alpha) * 向量分数 + alpha * 重排分数
                combined_score = (1 - self.section_rerank_alpha) * normalized_vec_score + self.section_rerank_alpha * normalized_rerank_score

                logger.debug(f"[rerank] {doc[:30]}... vec={vec_score:.3f}->{normalized_vec_score:.3f}, rerank={rerank_score:.3f}->{normalized_rerank_score:.3f}, combined={combined_score:.3f}")
                print(f"[rerank] {doc[:30]}... vec={vec_score:.3f}->{normalized_vec_score:.3f}, rerank={rerank_score:.3f}->{normalized_rerank_score:.3f}, combined={combined_score:.3f}")
                if combined_score > best_combined_score:
                    best_combined_score = combined_score
                    best_result = (doc, combined_score, vec_score, rerank_score)
            print(f"[rerank] 最佳匹配: {best_result[0][:50]}... combined={best_combined_score:.3f} (vec={best_result[2]:.3f}, rerank={best_result[3]:.3f})")
            if best_result and best_combined_score >= self.section_match_threshold:
                logger.info(f"[rerank] 最终匹配: {best_result[0][:50]}... combined={best_combined_score:.3f} (vec={best_result[2]:.3f}, rerank={best_result[3]:.3f})")
                return (best_result[0], best_combined_score)

            logger.info(f"[rerank] 无匹配结果 (best={best_combined_score:.3f} < {self.section_match_threshold})")
            return None

        def _build_query_with_filter(query: str, filter_pdf: str = None) -> str:
            """将 filter_pdf 拼接到 query 中"""
            if not filter_pdf:
                return query
            return f"在{filter_pdf}中，{query}"

        def smart_retrieval_impl(query: str, top_k: int = 60, filter_conditions: Dict = None, filter_pdf: str = None):
            """
            智能检索实现 - 并行匹配逻辑

            Args:
                query: 查询文本
                top_k: 返回结果数量
                filter_conditions: 过滤条件字典，如 {"pdf_filename": "xxx.pdf"}
                filter_pdf: 指定的 PDF 文件名，拼接到 query 中用于智能匹配
            """
            logger.info(f"[smart_retrieval_impl] query={query}, filter_conditions={filter_conditions}, filter_pdf={filter_pdf}")

            enhanced_query = _build_query_with_filter(query, filter_pdf)
            logger.info(f"[smart_retrieval_impl] 增强后的查询: {enhanced_query}")

            qvec = self.embedding_model.encode([enhanced_query])[0]
            
            # 智能匹配逻辑（文件名匹配 -> 章节匹配 -> 向量检索）

            # 1. 文件名匹配（有filter_pdf时跳过）
            file_match_result = None
            if not filter_pdf:
                # 只有在没有filter_pdf时才进行文件名匹配
                file_match_result = _compute_top_match(qvec, self.filename_embeddings, self.all_filenames, top_k=1)
                fname, fscore = file_match_result[0] if file_match_result else (None, 0.0)
                logger.info(f"[smart_retrieval] 文件名匹配: {fname} score={fscore:.3f}")
            else:
                # 有filter_pdf时，直接使用指定的文件名
                fname = filter_pdf
                fscore = 1.0  # 强制匹配
                logger.info(f"[smart_retrieval] 使用指定的filter_pdf: {fname}")

            # 2. 章节标题匹配（范围根据文件名匹配结果动态决定，使用向量+Reranker双阶段检索）
            chapter_match_result = None
            target_section = None
            
            if fname and fscore >= self.file_match_threshold:
                # 情况A：文件名匹配成功，只在该文件下的章节中匹配
                section_data = self.file_sections.get(fname)
                if section_data:
                    chapter_match_result = _compute_top_match_with_rerank(
                        enhanced_query,  
                        qvec,
                        section_data["section_embeddings"],
                        section_data["section_titles"],
                        top_k=5
                    )
            elif fname:
                # 情况B：有指定文件名但可能没有文件匹配，尝试在该文件下匹配章节
                section_data = self.file_sections.get(fname)
                if section_data:
                    chapter_match_result = _compute_top_match_with_rerank(
                        enhanced_query,  
                        qvec,
                        section_data["section_embeddings"],
                        section_data["section_titles"],
                        top_k=5
                    )
            else:
                # 情况C：文件名匹配失败，在所有章节中匹配
                # 需要合并所有文件的章节向量
                all_sections_data = self._get_all_sections_data()
                if all_sections_data["embeddings"] is not None:
                    chapter_match_result = _compute_top_match_with_rerank(
                        enhanced_query,
                        qvec,
                        all_sections_data["embeddings"],
                        all_sections_data["titles_with_file"],
                        top_k=5
                    )

            # 3. 结果决策
            target_filename = None
            target_section = None
            mode = "hybrid"

            # 优先使用章节匹配结果（_compute_top_match_with_rerank 已内部判断 combined_score >= 0.75）
            if chapter_match_result:
                chapter_title, cscore = chapter_match_result
                target_section = chapter_title
                
                if fname:
                    # 情况A：直接使用已匹配的文件名（适用于智能匹配或用户指定filter_pdf）
                    target_filename = fname
                    logger.info(f"[smart_retrieval] section match: {target_section} score={cscore:.3f} -> file={target_filename}")
                else:
                    # 情况B：从章节标题中解析文件名
                    if "|" in chapter_title:
                        target_filename, target_section = chapter_title.split("|", 1)
                        logger.info(f"[smart_retrieval] section match: {target_section} score={cscore:.3f} -> file={target_filename}")
                    else:
                        target_filename = self._find_file_by_section(chapter_title)
                        if target_filename:
                            logger.info(f"[smart_retrieval] section match: {target_section} score={cscore:.3f} -> file={target_filename}")

                if target_filename:
                    mode = "precise_by_section"

            # 次选文件名匹配结果
            if not target_filename and fname and fscore >= self.file_match_threshold:
                target_filename = fname
                mode = "precise_by_file"
                logger.info(f"[smart_retrieval] file match: {fname} score={fscore:.3f}")

            # 4. 执行检索
            if mode.startswith("precise") and target_filename:
                # ChromaDB expects simple dict for single-condition filters
                if target_section:
                    where = {"$and": [{"pdf_filename": {"$eq": target_filename}}, {"section_title": {"$eq": target_section}}]}
                else:
                    where = {"pdf_filename": {"$eq": target_filename}}
                logger.info(f"[smart_retrieval] using precise filter where={where}")
                results = self.retriever._vector_search(self.embedding_model.encode([enhanced_query])[0], top_k, where)
            else:
                logger.info("[smart_retrieval] 无文件名/章节匹配，不进行回退检索")
                results = []

            # attach mode info in attribute if needed
            for r in results:
                try:
                    setattr(r, "smart_mode", mode)
                except Exception:
                    pass

            # 标记是否有匹配
            for r in results:
                try:
                    setattr(r, "matched", True)
                except Exception:
                    pass

            return results

        @tool
        def smart_retrieval_tool(query_request_json: str, top_k: int = 60) -> str:
            """
            智能检索工具：根据JSON字符串检索相关文档内容。

            Args:
                query_request_json: JSON字符串格式（包含query和filter_pdf属性）
                top_k: 返回结果数量，默认 60

            Returns:
                格式化的检索结果字符串，直接包含检索到的文档内容
            """
            try:
                # 解析JSON字符串获取参数
                import json
                data = json.loads(query_request_json)
                query = data.get("query", "")
                filter_pdf = data.get("filter_pdf")

                logger.info(f"[smart_retrieval_tool] 原始query: {query}, filter_pdf: {filter_pdf}")

                # 只传递原始query，查询增强在smart_retrieval_impl中统一处理
                results = smart_retrieval_impl(query, top_k=top_k, filter_conditions=None, filter_pdf=filter_pdf)

                if not results:
                    return f"未检索到与「{query}」相关的内容。"

                output_parts = []
                output_parts.append(f"【检索结果】共找到 {len(results)} 条相关内容：\n")

                for i, r in enumerate(results, 1):
                    text = r.text.strip() if r.text else ""
                    section = r.section_title or "未知章节"
                    page = int(r.page_number) if r.page_number is not None else "?"
                    score = f"{r.score:.2f}" if r.score is not None else "?"
                    pdf_filename = r.pdf_filename or "未知文件"

                    output_parts.append(f"--- 相关文档 {i} (相似度:{score}) ---")
                    output_parts.append(f"来源：{pdf_filename} / {section}，第 {page} 页")
                    output_parts.append(f"内容：{text}")
                    output_parts.append("")

                output_parts.append("【回答要求】")
                output_parts.append("请基于以上检索结果回答用户的问题。")
                output_parts.append("如果检索结果不足以回答问题，请明确说明。")

                return "\n".join(output_parts)

            except Exception as e:
                logger.error(f"[smart_retrieval_tool] failed: {e}")
                return f"检索失败：{str(e)}"

        # ==================== 工具优先级 ====================
        PRIORITY_TOOLS = [
            smart_retrieval_tool
        ]

        # 暴露格式化工具和原始检索函数供外部直接调用（agent 内部使用）
        self.smart_retrieval = smart_retrieval_tool
        self.smart_retrieval_impl = smart_retrieval_impl

        return PRIORITY_TOOLS
    
    @property
    def tools(self):
        """
        返回所有可用的LangChain工具
        
        Returns:
            List: 包含所有工具方法的列表
        """
        return self._tools

    def _get_all_sections_data(self):
        """
        获取所有章节数据（用于全局匹配）
        返回: {"titles_with_file": [], "embeddings": []}
        注意: titles_with_file 格式为 ["文件名|章节标题", ...]
        """
        titles_with_file = []
        embeddings = []

        for filename, data in self.file_sections.items():
            for title, emb in zip(data["section_titles"], data["section_embeddings"]):
                titles_with_file.append(f"{filename}|{title}")
                embeddings.append(emb)

        return {
            "titles_with_file": titles_with_file,
            "embeddings": _np.array(embeddings) if embeddings else None
        }

    def _find_file_by_section(self, section_title: str) -> Optional[str]:
        """
        通过章节标题查找所属文件名
        """
        for filename, data in self.file_sections.items():
            if section_title in data["section_titles"]:
                return filename
        return None

def create_retrieval_tools(vector_db_path: str, embedding_model_name: str = "intfloat/multilingual-e5-large",
                           vector_db: VectorDatabase = None, embedding_model: EmbeddingModel = None,
                           reranker_model: RerankerModel = None) -> DocumentRetrievalTools:
    """
    创建文档检索工具实例

    Args:
        vector_db_path: 向量数据库路径
        embedding_model_name: 嵌入模型名称
        vector_db: 已初始化的VectorDatabase实例（可选，用于复用）
        embedding_model: 已初始化的EmbeddingModel实例（可选，用于复用）
        reranker_model: 已初始化的RerankerModel实例（可选，用于复用）

    Returns:
        DocumentRetrievalTools实例
    """
    return DocumentRetrievalTools(vector_db_path, embedding_model_name, vector_db, embedding_model, reranker_model)

def get_langchain_tools(vector_db_path: str, embedding_model_name: str = "intfloat/multilingual-e5-large",
                        vector_db: VectorDatabase = None, embedding_model: EmbeddingModel = None,
                        reranker_model: RerankerModel = None) -> DocumentRetrievalTools:
    """
    获取LangChain工具实例

    Args:
        vector_db_path: 向量数据库路径
        embedding_model_name: 嵌入模型名称
        vector_db: 已初始化的VectorDatabase实例（可选，用于复用）
        embedding_model: 已初始化的EmbeddingModel实例（可选，用于复用）
        reranker_model: 已初始化的RerankerModel实例（可选，用于复用）

    Returns:
        DocumentRetrievalTools实例
    """
    return create_retrieval_tools(vector_db_path, embedding_model_name, vector_db, embedding_model, reranker_model)