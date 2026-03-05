"""
向量检索系统核心模块 - 数据结构和数据库操作
支持多语言、GPU加速、向量数据库持久化
"""

import os
import torch
import numpy as np
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
import logging
from pathlib import Path
import chromadb
from chromadb.config import Settings
import time
from collections import OrderedDict
try:
    import diskcache
except Exception:
    diskcache = None
import numpy as _np

# 延迟导入以避免循环导入
def get_vector_retriever():
    """延迟导入VectorRetriever"""
    from mixed_retrieval import VectorRetriever
    return VectorRetriever

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def normalize_vector(vector: List[float]) -> List[float]:
    """归一化向量为单位长度"""
    try:
        arr = np.array(vector, dtype=float)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr.tolist()
    except Exception as e:
        logger.warning(f"向量归一化失败，返回原向量: {e}")
        return vector if isinstance(vector, list) else vector.tolist()

@dataclass
class DocumentChunk:
    """文档块数据结构"""
    id: str
    text: str
    vector: List[float]
    page_number: int
    section_title: str
    pdf_path: str
    language: str
    coordinates: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class SearchResult:
    """搜索结果数据结构"""
    text: str
    section_title: str
    page_number: int
    score: float
    pdf_path: str
    language: str
    coordinates: Optional[Dict[str, float]] = None
    rerank_score: Optional[float] = None  # Reranker模型计算的相关性分数
    # 新增字段，便于精确引用
    chunk_id: Optional[str] = None
    pdf_filename: Optional[str] = None
    chunk_index: Optional[int] = None
    text_hash: Optional[str] = None

class VectorDatabase:
    """向量数据库存储层 - 使用ChromaDB实现"""
    
    def __init__(self, db_path: str = "./chroma_db"):
        # 客户端配置
        self.client = chromadb.PersistentClient( #创建本地持久化数据库客户端
            path=db_path,
            settings=Settings(allow_reset=True, anonymized_telemetry=False) #允许重置数据库，禁用匿名遥测
        )
        # 获取或创建集合（类似表）。这是支持向量搜索的关键。
        self.collection = self.client.get_or_create_collection(
            name="document_chunks", #集合名称，用于存储文档块
            metadata={"hnsw:space": "cosine"}  # 使用HNSW索引和余弦相似度,用于快速查找相似向量
        )
        logger.info(f"ChromaDB数据库初始化完成: {db_path}")
    
    def store_chunks(self, chunks: List[DocumentChunk]):
        """批量存储文档块到ChromaDB"""
        ids, texts, embeddings, metadatas = [], [], [], []
        for i, chunk in enumerate(chunks):
            try:
                if not chunk.vector or len(chunk.vector) == 0:
                    raise ValueError(f"Chunk {chunk.id} 的向量为空")
                ids.append(chunk.id)
                texts.append(chunk.text)
                embeddings.append(chunk.vector)
                meta = {
                    "page_number": chunk.page_number,
                    "section_title": chunk.section_title,
                    "pdf_path": chunk.pdf_path,
                    "pdf_filename": Path(chunk.pdf_path).name,
                    "language": chunk.language,
                    "coordinates": json.dumps(chunk.coordinates) if chunk.coordinates else "",
                    "chunk_id": chunk.id
                }
                if chunk.metadata:
                    for k, v in chunk.metadata.items():
                        if k in meta:
                            continue
                        if isinstance(v, (str, int, float, bool)) or v is None:
                            meta[k] = v
                        else:
                            try:
                                meta[k] = json.dumps(v, ensure_ascii=False)
                            except Exception:
                                meta[k] = str(v)
                metadatas.append(meta)
            except Exception as e:
                logger.error(f"处理第 {i} 个chunk时出错: {e}")
                continue

        if not ids:
            logger.warning("没有有效的chunk可存储")
            return

        # 检查向量维度一致性
        first_dim = len(embeddings[0])
        for v in embeddings:
            if len(v) != first_dim:
                logger.error(f"向量维度不一致: 期望 {first_dim}, 实际 {len(v)}，中止存储")
                raise RuntimeError("Embedding dimension mismatch")
        if not hasattr(self, "embedding_dim") or self.embedding_dim is None:
            self.embedding_dim = first_dim

        # 分批存储，避免单次操作过大
        batch_size = 1000
        for i in range(0, len(ids), batch_size):
            try:
                self.collection.add(
                    ids=ids[i:i+batch_size],
                    documents=texts[i:i+batch_size],
                    embeddings=embeddings[i:i+batch_size],
                    metadatas=metadatas[i:i+batch_size]
                )
            except Exception as e:
                logger.error(f"分批存储失败（块 {i}-{i+batch_size}）: {e}")
                raise
        logger.info(f"成功存储 {len(ids)} 个块到向量数据库")
    
    def search_similar(self, query_vector: List[float], top_k: int = 10, filter_conditions: Optional[Dict] = None):
        """执行向量相似度搜索，这才是真正的向量检索"""
        # ChromaDB 会自动使用 HNSW 索引进行快速近似搜索
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=filter_conditions,  # 强大的元数据过滤，如 `{"page_number": 5}`
            include=["documents", "metadatas", "distances"]
        )
        
        # 格式化返回结果为SearchResult列表
        formatted_results = []
        for i in range(len(results["ids"][0])):
            metadata = results["metadatas"][0][i]
            # 解析额外的元数据
            chunk_id = metadata.get("chunk_id") or results["ids"][0][i]
            pdf_path = metadata.get("pdf_path", "")
            pdf_filename = metadata.get("pdf_filename") or Path(pdf_path).name
            chunk_index = metadata.get("chunk_index")
            text_hash = metadata.get("text_hash")

            formatted_results.append(SearchResult(
                text=results["documents"][0][i],
                section_title=metadata.get("section_title", ""),
                page_number=int(metadata.get("page_number", 0)) if metadata.get("page_number") is not None else 0,
                score=1 - results["distances"][0][i],  # 转换为相似度分数
                pdf_path=pdf_path,
                language=metadata.get("language", ""),
                coordinates=json.loads(metadata.get("coordinates", "{}")) if metadata.get("coordinates") else None,
                chunk_id=chunk_id,
                pdf_filename=pdf_filename,
                chunk_index=int(chunk_index) if chunk_index is not None else None,
                text_hash=text_hash
            ))
        
        return formatted_results
    
    def get_all_chunks(self) -> List[DocumentChunk]:
        """获取所有文档块"""
        chunks = []
        results = self.collection.get(
            include=["documents", "embeddings", "metadatas"]
        )
        
        for i in range(len(results["ids"])):
            metadata = results["metadatas"][i]
            chunks.append(DocumentChunk(
                id=results["ids"][i],
                text=results["documents"][i],
                vector=results["embeddings"][i],
                page_number=metadata.get("page_number", 0),
                section_title=metadata.get("section_title", ""),
                pdf_path=metadata.get("pdf_path", ""),
                language=metadata.get("language", ""),
                coordinates=json.loads(metadata.get("coordinates", "{}")) if metadata.get("coordinates") else None,
                metadata=json.loads(metadata.get("metadata", "{}")) if metadata.get("metadata") else None
            ))
        
        return chunks
    
    def clear_database(self):
        """清空数据库"""
        # 获取所有文档ID
        results = self.collection.get()
        if results["ids"]:
            self.collection.delete(ids=results["ids"])
        logger.info("数据库已清空")

    def get_all_text_hashes(self) -> set:
        """返回数据库中已有的 text_hash 集合，用于去重"""
        try:
            results = self.collection.get(include=["metadatas"])
            hashes = set()
            for md in results.get("metadatas", []):
                # metadata 中可能包含顶层 text_hash 或嵌套在 metadata 字段里（JSON 字符串）
                if isinstance(md, dict):
                    th = md.get("text_hash")
                    if th:
                        hashes.add(th)
                    else:
                        # 兼容早期将 metadata 序列化为 JSON 的字段
                        nested = md.get("metadata")
                        if nested:
                            try:
                                nm = json.loads(nested)
                                nth = nm.get("text_hash")
                                if nth:
                                    hashes.add(nth)
                            except Exception:
                                pass
            return hashes
        except Exception as e:
            logger.warning(f"获取已有 text_hash 失败: {e}")
            return set()

class EmbeddingModel:
    """嵌入模型封装"""
    
    def __init__(self, model_name: str = None, cache_size: int = 10000, cache_backend: str = "memory", disk_cache_dir: str = "src/data/emb_cache"):
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.cache = OrderedDict()
        self.cache_size = cache_size
        self.cache_backend = cache_backend
        self.disk_cache = None
        if cache_backend == "disk":
            if diskcache is None:
                logger.warning("diskcache 未安装，回退到内存缓存")
                self.cache_backend = "memory"
            else:
                try:
                    self.disk_cache = diskcache.Cache(disk_cache_dir)
                    logger.info(f"启用磁盘缓存: {disk_cache_dir}")
                except Exception as e:
                    logger.warning(f"无法初始化 diskcache，使用内存缓存: {e}")
                    self.cache_backend = "memory"
        self.stats = {
            "total_texts_encoded": 0,
            "total_time_spent": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        self._load_model()
    
    def _load_model(self):
        """加载嵌入模型"""
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"加载嵌入模型: {self.model_name} (设备: {self.device})")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            
            # 预热模型
            self.model.encode(["预热测试"], convert_to_tensor=False)
            logger.info("嵌入模型加载完成")
            
        except ImportError:
            logger.error("未安装sentence-transformers，请运行: pip install sentence-transformers")
            raise
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """生成文本嵌入向量（带缓存、归一化和自适应批次）"""
        if not self.model:
            raise RuntimeError("模型未加载")

        start_time = time.time()

        # 检查缓存
        vectors = [None] * len(texts)
        uncached_texts = []
        uncached_indices = []
        for i, t in enumerate(texts):
            hit = False
            # 优先检查磁盘缓存
            if self.cache_backend == "disk" and self.disk_cache is not None:
                try:
                    cached = self.disk_cache.get(t)
                    if cached is not None:
                        vectors[i] = np.array(cached, dtype=float)
                        self.stats["cache_hits"] += 1
                        hit = True
                except Exception as e:
                    logger.warning(f"读取 diskcache 失败: {e}")
            if hit:
                continue
            # 再检查内存缓存
            if t in self.cache:
                vectors[i] = np.array(self.cache[t], dtype=float)
                self.stats["cache_hits"] += 1
            else:
                uncached_texts.append(t)
                uncached_indices.append(i)
                self.stats["cache_misses"] += 1

        # 估算自适应批次大小
        if self.device == "cuda" and torch.cuda.is_available():
            try:
                total_mem = torch.cuda.get_device_properties(0).total_memory
                alloc = torch.cuda.memory_allocated(0)
                free = max(0, total_mem - alloc)
                estimated_per_text = 1024 * 4
                batch_size = min(256, max(1, int(free * 0.5 / estimated_per_text)))
            except Exception:
                batch_size = 32
        else:
            batch_size = 32

        # 批量编码未缓存文本
        for i in range(0, len(uncached_texts), batch_size):
            batch_texts = uncached_texts[i:i+batch_size]
            try:
                batch_vectors = self.model.encode(batch_texts, convert_to_numpy=True)
            except Exception as e:
                logger.error(f"批量编码失败，尝试逐条编码: {e}")
                batch_vectors = []
                for t in batch_texts:
                    try:
                        v = self.model.encode([t], convert_to_numpy=True)[0]
                        batch_vectors.append(v)
                    except Exception as ee:
                        logger.error(f"单条编码失败: {ee}")
                        batch_vectors.append(None)

            # 写回 vectors 和缓存
            for j, vec in enumerate(batch_vectors):
                idx = uncached_indices[i + j]
                if vec is None:
                    continue
                # 归一化
                try:
                    arr = np.array(vec, dtype=float)
                    norm = np.linalg.norm(arr)
                    if norm > 0:
                        arr = arr / norm
                    vec_list = arr.tolist()
                except Exception:
                    vec_list = vec.tolist() if hasattr(vec, "tolist") else list(vec)

                vectors[idx] = np.array(vec_list, dtype=float)
                # 更新内存缓存 (LRU)
                text_key = uncached_texts[i + j]
                if text_key in self.cache:
                    self.cache.pop(text_key)
                self.cache[text_key] = vec_list
                if len(self.cache) > self.cache_size:
                    self.cache.popitem(last=False)
                # 更新磁盘缓存（如启用）
                if self.cache_backend == "disk" and self.disk_cache is not None:
                    try:
                        self.disk_cache.set(text_key, vec_list)
                    except Exception as e:
                        logger.warning(f"写入 diskcache 失败: {e}")

        # 汇总结果并统计
        result = np.vstack([v for v in vectors if v is not None]) if any(v is not None for v in vectors) else np.array([])
        elapsed = time.time() - start_time
        self.stats["total_texts_encoded"] += len(texts)
        self.stats["total_time_spent"] += elapsed
        logger.debug(f"编码 {len(texts)} 个文本，耗时 {elapsed:.2f}s")
        return result

class RerankerModel:
    """Reranker模型封装"""
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """加载Reranker模型"""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            from torch.nn.functional import softmax
            
            logger.info(f"加载Reranker模型: {self.model_name} (设备: {self.device})")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, 
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            logger.info("Reranker模型加载完成")
            
        except ImportError:
            logger.warning("未安装transformers，使用简化版Reranker")
            self.model = None
        except Exception as e:
            logger.error(f"Reranker模型加载失败: {e}")
            self.model = None
    
    def rerank(self, query: str, documents: List[str], top_k: int = 10) -> List[Tuple[int, float]]:
        """对文档进行重新排序"""
        if not self.model:
            # 简化版Reranker：基于关键词匹配和长度惩罚
            scores = []
            query_words = set(query.lower().split())
            
            for doc in documents:
                doc_words = set(doc.lower().split())
                overlap = len(query_words & doc_words)
                
                # 基础分数：词汇重叠度
                base_score = overlap / max(len(query_words), 1)
                
                # 长度惩罚：过长文档得分降低
                length_penalty = 1.0 / (1.0 + len(doc.split()) / 100)
                
                # 综合分数
                final_score = base_score * length_penalty
                scores.append(final_score)
            
            # 获取top_k索引
            scores_with_idx = [(i, score) for i, score in enumerate(scores)]
            scores_with_idx.sort(key=lambda x: x[1], reverse=True)
            return scores_with_idx[:top_k]
        
        try:
            # 使用真实Reranker模型
            from torch.nn.functional import softmax
            
            pairs = [(query, doc) for doc in documents]
            inputs = self.tokenizer(
                pairs, 
                padding=True, 
                truncation=True, 
                return_tensors="pt",
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                scores = self.model(**inputs).logits
            
            scores = scores.squeeze().cpu().numpy()  # 直接使用回归分数
            
            # 获取top_k索引
            scores_with_idx = [(i, float(score)) for i, score in enumerate(scores)]
            scores_with_idx.sort(key=lambda x: x[1], reverse=True)
            return scores_with_idx[:top_k]
            
        except Exception as e:
            logger.error(f"Reranking失败: {e}")
            # 降级到简化版
            scores = []
            query_words = set(query.lower().split())
            
            for doc in documents:
                doc_words = set(doc.lower().split())
                overlap = len(query_words & doc_words)
                
                # 基础分数：词汇重叠度
                base_score = overlap / max(len(query_words), 1)
                
                # 长度惩罚：过长文档得分降低
                length_penalty = 1.0 / (1.0 + len(doc.split()) / 100)
                
                # 综合分数
                final_score = base_score * length_penalty
                scores.append(final_score)
            
            # 获取top_k索引
            scores_with_idx = [(i, score) for i, score in enumerate(scores)]
            scores_with_idx.sort(key=lambda x: x[1], reverse=True)
            return scores_with_idx[:top_k]



class DocumentProcessor:
    """文档处理和索引构建器"""
    
    def __init__(self, vector_db: VectorDatabase, embedding_model: EmbeddingModel):
        self.vector_db = vector_db
        self.embedding_model = embedding_model
    
    def process_pdf(self, pdf_path: str, chunk_size: int = 512, overlap: int = 50) -> List[DocumentChunk]:
        """处理PDF文件并生成文档块"""
        from data.pdf_loader import PDFLoader
        from identify_title import TitleIdentifier
        
        logger.info(f"处理PDF文件: {pdf_path}")
        
        # 解析PDF
        loader = PDFLoader()
        pages = loader.load_pdf(pdf_path)
        
        # 识别标题
        title_identifier = TitleIdentifier()
        titles = title_identifier.identify_title(pages)
        
        # 构建章节映射
        section_mapping = self._build_section_mapping(titles)
        
        # 生成文档块
        chunks = []
        chunk_id = 0
        
        for page in pages:
            # 确定当前页的章节标题
            section_title = self._get_section_title(page.page_number, section_mapping)
            
            # 合并页面文本
            full_text = page.get_full_text()
            if not full_text.strip():
                continue
            
            # 文本分块
            text_chunks = self._chunk_text(full_text, chunk_size, overlap)
            
            for chunk_text in text_chunks:
                if len(chunk_text.strip()) < 10:  # 过滤太短的文本
                    continue
                
                # 生成向量
                vector = self.embedding_model.encode([chunk_text])[0]
                
                # 创建文档块
                chunk = DocumentChunk(
                    id=f"{Path(pdf_path).stem}_{chunk_id:06d}",
                    text=chunk_text,
                    vector=vector.tolist(),
                    page_number=page.page_number,
                    section_title=section_title,
                    pdf_path=pdf_path,
                    language=page.language,
                    coordinates=self._get_coordinates(page, chunk_text)
                )
                
                chunks.append(chunk)
                chunk_id += 1
        
        logger.info(f"生成了 {len(chunks)} 个文档块")
        return chunks
    
    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """文本分块"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # 在句号处截断
            if end < len(text):
                last_period = text.rfind('。', start, end)
                last_punctuation = text.rfind('.', start, end)
                last_break = max(last_period, last_punctuation)
                
                if last_break > start + chunk_size // 2:
                    end = last_break + 1
            
            chunk = text[start:end].strip()
            chunks.append(chunk)
            
            start = end - overlap if end < len(text) else end
        
        return chunks
    
    def _build_section_mapping(self, titles: List[Dict[str, Any]]) -> Dict[int, str]:
        """构建章节页码映射"""
        mapping = {}
        for title in titles:
            start_page = title["start_page"]
            end_page = title.get("end_page", start_page)
            section_title = title["title"]
            
            for page in range(start_page, end_page + 1):
                mapping[page] = section_title
        
        return mapping
    
    def _get_section_title(self, page_number: int, section_mapping: Dict[int, str]) -> str:
        """获取页面对应的章节标题"""
        # 向后查找最近的章节标题
        for page in range(page_number, 0, -1):
            if page in section_mapping:
                return section_mapping[page]
        
        return "未知章节"
    
    def _get_coordinates(self, page, chunk_text: str) -> Optional[Dict[str, float]]:
        """获取文本块的坐标信息"""
        # 简化实现：返回第一个文本块的坐标
        if page.text_blocks:
            block = page.text_blocks[0]
            return {
                "x0": block.x0,
                "y0": block.y0,
                "x1": block.x1,
                "y1": block.y1
            }
        return None


def ingest_json_directory_to_chroma(json_dir: str, db_path: str = None, batch_size: int = 64, model_name: str = None):
    """
    从包含 precomputed JSON 的目录中读取 document_chunks，生成 embeddings 并存入 ChromaDB。
    JSON 格式应当是：
      {
        "parent_document": {..., "file_path": "...", "language": "...", "processing_date": "..."},
        "document_chunks": [ { "id", "text", "page_number", "section_title", "chunk_index", "total_chunks_in_page", "coordinates", "text_hash" }, ... ]
      }
    """
    db_path = db_path or os.getenv("VECTOR_DB_PATH", "src/data/vector_database")
    vector_db = VectorDatabase(db_path) # 初始化 ChromaDB
    embedding_model = EmbeddingModel(model_name or "intfloat/multilingual-e5-large") # 初始化嵌入模型

    json_files = list(Path(json_dir).glob("**/*.json")) # 递归查找所有 JSON 文件
    logger.info(f"发现 {len(json_files)} 个 JSON 文件用于导入：{json_dir}")

    for jf in json_files:   
        ingest_json_file_to_chroma(jf, vector_db, embedding_model, batch_size)

    logger.info("全部 JSON 导入完成")


def ingest_json_file_to_chroma(jf: Path, vector_db: VectorDatabase, embedding_model: EmbeddingModel, batch_size: int = 64):
    """处理单个 JSON 文件并将其 document_chunks 导入到 ChromaDB"""
    try:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning(f"无法读取 JSON 文件 {jf}: {e}")
        return

    parent = data.get("parent_document", {})
    chunks_data = data.get("document_chunks", [])
    if not chunks_data:
        logger.info(f"{jf} 中未找到 document_chunks，跳过")
        return

    logger.info(f"处理 {jf}，包含 {len(chunks_data)} 个块")
    # 获取已有 text_hash 集合以做去重
    existing_hashes = vector_db.get_all_text_hashes()
    logger.info(f"数据库中已有 {len(existing_hashes)} 个 text_hash，开始去重导入")

    for i in range(0, len(chunks_data), batch_size):
        batch = chunks_data[i:i+batch_size]
        # 先过滤出需要处理的 chunk（基于 text_hash 去重）
        chunks_to_process = [c for c in batch if not c.get("text_hash") or c.get("text_hash") not in existing_hashes]
        if not chunks_to_process:
            continue

        texts = [c.get("text", "") for c in chunks_to_process]
        # 尝试批量生成向量；若失败，回退到单个生成以保证鲁棒性
        try:
            vectors = embedding_model.encode(texts)
        except Exception as e:
            logger.warning(f"批量生成向量失败（文件: {jf}），尝试逐条生成: {e}")
            vectors = []
            for t in texts:
                try:
                    v = embedding_model.encode([t])[0]
                    vectors.append(v)
                except Exception as ee:
                    logger.error(f"单条生成向量失败: {ee}")
                    vectors.append(None)

        doc_chunks = []
        for idx, c in enumerate(chunks_to_process):
            vec = vectors[idx]
            if vec is None:
                continue
            # 向量归一化
            try:
                arr = _np.array(vec, dtype=float)
                norm = _np.linalg.norm(arr)
                if norm > 0:
                    arr = arr / norm
                vec_list = arr.tolist()
            except Exception:
                vec_list = vec.tolist() if hasattr(vec, "tolist") else list(vec)

            doc = DocumentChunk(
                id=c.get("id") or f"{Path(jf).stem}_{i+idx:06d}",
                text=c.get("text", ""),
                vector=vec_list,
                page_number=int(c.get("page_number", 0)) if c.get("page_number") is not None else 0,
                section_title=c.get("section_title", ""),
                pdf_path=parent.get("file_path", ""),
                language=parent.get("language", ""),
                coordinates=c.get("coordinates"),
                metadata={
                    "chunk_index": c.get("chunk_index"),
                    "total_chunks_in_page": c.get("total_chunks_in_page"),
                    "text_hash": c.get("text_hash", ""),
                    "source_json": Path(jf).name,
                    "processing_date": parent.get("processing_date")
                }
            )
            doc_chunks.append(doc)

        if not doc_chunks:
            continue

        try:
            vector_db.store_chunks(doc_chunks)
        except Exception as e:
            logger.error(f"存储到 ChromaDB 出错: {e}")
            continue

    logger.info(f"完成导入 {jf}")


def show_chroma_contents(db_path: str = None, limit: int = 20):
    """展示 ChromaDB 中的文本块（用于调试/验证）"""
    db_path = db_path or os.getenv("VECTOR_DB_PATH", "src/data/vector_database")
    vector_db = VectorDatabase(db_path)
    chunks = vector_db.get_all_chunks()
    logger.info(f"ChromaDB 包含 {len(chunks)} 个文档块")
    for i, c in enumerate(chunks[:limit], 1):
        meta = c.metadata or {}
        print(f"{i}. id={c.id} pdf={Path(c.pdf_path).name} page={c.page_number} chunk_index={meta.get('chunk_index')} text_hash={meta.get('text_hash')}")
        print(f"   section: {c.section_title}")
        print(f"   text: {c.text[:200].replace(chr(10),' ')}")
        print()

class VectorSearchSystem:
    """向量检索系统主类"""
    
    def __init__(self, db_path: str = "vector_db.sqlite"):
        self.vector_db = VectorDatabase(db_path)
        self.embedding_model = EmbeddingModel()
        self.reranker_model = RerankerModel()
        self.retriever = None
        self.document_processor = DocumentProcessor(self.vector_db, self.embedding_model)
        
        logger.info("向量检索系统初始化完成")
    
    def build_index(self, pdf_directory: str, force_rebuild: bool = False):
        """构建向量索引"""
        if force_rebuild:
            self.vector_db.clear_database()
        
        # 获取所有PDF文件
        pdf_files = list(Path(pdf_directory).rglob("*.pdf"))
        logger.info(f"找到 {len(pdf_files)} 个PDF文件")
        
        all_chunks = []
        for pdf_path in pdf_files:
            try:
                chunks = self.document_processor.process_pdf(str(pdf_path))
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"处理PDF失败 {pdf_path}: {e}")
                continue
        
        # 批量存储到数据库
        if all_chunks:
            self.vector_db.store_chunks(all_chunks)
            logger.info(f"索引构建完成，共处理 {len(all_chunks)} 个文档块")
    
    def initialize_retriever(self):
        """初始化检索器"""
        VectorRetriever = get_vector_retriever()
        self.retriever = VectorRetriever(self.vector_db, self.embedding_model)
    
    def search(self, query: str, top_k: int = 10, use_reranker: bool = True) -> List[SearchResult]:
        """执行搜索"""
        if not self.retriever:
            self.initialize_retriever()
        
        # 混合检索
        initial_results = self.retriever.hybrid_search(query, top_k=top_k * 2)
        
        if not initial_results:
            return []
        
        # Rerank精排
        if use_reranker and initial_results:
            documents = [result.text for result in initial_results]
            rerank_results = self.reranker_model.rerank(query, documents, top_k)
            
            # 重新组织结果
            final_results = []
            for doc_idx, score in rerank_results:
                result = initial_results[doc_idx]
                result.score = score
                final_results.append(result)
            
            return final_results
        
        return initial_results[:top_k]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        chunks = self.vector_db.get_all_chunks()
        
        stats = {
            "total_chunks": len(chunks),
            "total_pdfs": len(set(chunk.pdf_path for chunk in chunks)),
            "languages": list(set(chunk.language for chunk in chunks)),
            "gpu_available": torch.cuda.is_available(),
            "device": self.embedding_model.device
        }
        
        return stats

# 使用示例
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Embedding / ChromaDB 工具")
    parser.add_argument("--ingest-json-dir", type=str, help="从 JSON 目录导入 document_chunks 并生成向量")
    parser.add_argument("--db-path", type=str, default=None, help="ChromaDB 存储路径")
    parser.add_argument("--batch-size", type=int, default=64, help="批处理大小（导入时生成向量）")
    parser.add_argument("--show-db", action="store_true", help="展示 ChromaDB 中前若干文本块")
    parser.add_argument("--show-limit", type=int, default=20, help="展示时的条目上限")
    args = parser.parse_args()

    if args.ingest_json_dir:
        ingest_json_directory_to_chroma(args.ingest_json_dir, db_path=args.db_path, batch_size=args.batch_size)
    elif args.show_db:
        show_chroma_contents(db_path=args.db_path, limit=args.show_limit)
    else:
        # 默认行为：从 src/data/pages_title 中导入已经生成的 JSON 到 ChromaDB
        default_json_dir = "src/data/pages_title"
        default_db_path = args.db_path or "src/data/vector_database"
        ingest_json_directory_to_chroma(default_json_dir, db_path=default_db_path, batch_size=args.batch_size)