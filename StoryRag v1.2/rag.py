"""Lightweight RAG indexer for tri-lingual stories (Chinese/English/Japan) with vector search and reranker."""

from __future__ import annotations

import os
import re
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder

# 使用 FAISS 作为向量数据库
import faiss
import pickle

from llm import DeepSeekClient

load_dotenv()
#_file__:表示当前文件的路径  Path(__file__).parent:获取当前文件所在的目录    /"Ragdata":在该目录下创建一个名为"Ragdata"的子目录路径
DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).parent / "Ragdata"))
SUPPORTED_LANGS = [  # 变量 = [表达式 for 循环变量 in 可迭代对象 if 条件]
    lang.strip()  # 去除每个语言名称前后的空白字符串
    for lang in os.getenv("LANGS", "Chinese,English,Japan").split(",")
    if lang.strip()
]
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "Chinese")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
TOP_K = int(os.getenv("TOP_K", "4"))
# 向量检索获取的候选数量（用于 reranker 重排序）
VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "30"))
# 嵌入模型名称（支持多语言）
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
# Reranker 模型名称
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
# reranker 与向量相似度的混合比例（最终得分 = alpha * rerank_norm + (1-alpha) * vector_norm）
RERANK_MIX_ALPHA = float(os.getenv("RERANK_MIX_ALPHA", "0.7"))


@dataclass  # 使用dataclass装饰器，自动为类生成__init__、__repr__等方法
class Chunk:  # 定义Chunk类，用于表示文本块
    text: str  # 文本内容，类型为字符串
    lang: str  # 语言标识，类型为字符串
    source: str  # 来源标识，类型为字符串
    chunk_id: int  # 文本块ID，类型为整数

    @property  # 使用property装饰器，将方法转换为只读属性
    def citation(self) -> str:  # 定义citation属性，返回引用字符串，类型为字符串
        return f"{self.lang}/{self.source}#chunk{self.chunk_id}"  # 返回格式为"语言/来源#chunkID"的引用字符串


def _read_plain(path: Path) -> str:

    """
    读取指定路径的文本文件内容

    参数:
        path (Path): 文件路径对象，表示要读取的文件路径

    返回:
        str: 文件的内容，如果读取失败则返回空字符串

    注意:
        使用UTF-8编码读取文件，忽略解码错误
    """
    return path.read_text(encoding="utf-8", errors="ignore")  # 调用Path对象的read_text方法读取文件内容


def load_story_text(path: Path) -> str:
    if path.suffix.lower() != ".txt":
        raise ValueError("Only .txt files are supported for corpus loading.")
    text = _read_plain(path)
    return _clean_text(text)


def _clean_text(text: str) -> str:
    """
    清洗文本内容，去除不必要的空白字符和特殊字符

    参数:
        text (str): 待清洗的文本

    返回:
        str: 清洗后的文本
    """
    # 去除多余的空白字符，将多个连续的空格、制表符、换行符替换为单个空格
    text = re.sub(r'\s+', ' ', text)
    # 去除首尾空白
    text = text.strip()
    return text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    chunks: List[str] = []
    start = 0
    end = max(chunk_size, 1)
    while start < len(text):
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        end = start + chunk_size
    return chunks


def load_corpus(
        language: str | None = None,
        mode: str = "library",
        book_name: str | None = None,
        load_all_languages: bool = False,
) -> List[Chunk]:
    """
    加载语料库。

    Args:
        language: 指定语言，如果为 None 则使用默认语言
        mode: "library" 或 "single"
        book_name: 当 mode="single" 时指定的文件名
        load_all_languages: 如果为 True，则加载所有支持语言的数据
    """
    chunks: List[Chunk] = []

    # 如果 load_all_languages=True，加载所有语言的数据
    if load_all_languages:
        languages_to_load = SUPPORTED_LANGS
    else:
        lang = language or DEFAULT_LANGUAGE
        if lang not in SUPPORTED_LANGS:
            raise ValueError(f"Unsupported language {lang}, choose from {SUPPORTED_LANGS}")
        languages_to_load = [lang]

    # 遍历所有需要加载的语言
    for lang in languages_to_load:
        lang_dir = DATA_DIR / lang #生成对应语言的图书资源路径
        if not lang_dir.exists():
            # 如果目录不存在，跳过该语言（而不是抛出错误）
            continue

        # 类型注解:表明files是一个可迭代的Path对象集合
        files: Iterable[Path]
        if mode == "single":#在单书模式下
            if not book_name:#如果book_name为空
                raise ValueError("single mode requires book_name")#抛出错误
            files = [lang_dir / book_name]#不为空时生成用户指定图书的资源目录
        else:
            files = sorted(lang_dir.glob("*"))#在全库模式下files为所有图书资源目录
            #此时的files是是一个可迭代的文件对象

        for file_path in files:#遍历文件对象,取出每一个图书的资源路径
            #检查路径是否指向一个实际存在的文件或者文件扩展名是否为".txt"
            if not file_path.is_file() or file_path.suffix.lower() != ".txt":
                continue
            text = load_story_text(file_path)#加载对应资源路径的图书文本并进行清洗
            parts = chunk_text(text)#对图书文本进行分块处理
            for idx, part in enumerate(parts):#遍历分块后的文本
                if not part.strip():#如果文本块为空，则跳过
                    continue
                #这里的chunks是类型为Chunk的空列表
                chunks.append(
                    Chunk(
                        text=part.strip(),# 去除文本两端的空白字符
                        lang=lang,# 设置文本的语言类型
                        source=file_path.name,# 记录源文件名
                        chunk_id=idx,# 设置文本块的唯一标识符
                    )
                )

    if not chunks:
        raise ValueError(
            f"No chunks built. "
            "Only .txt files are supported; ensure txt files exist under the language folders."
        )
    return chunks


class RagIndexer:
    """Vector-based retriever with reranker for tri-lingual stories using Chroma vector database."""

    def __init__(self, chunks: List[Chunk], cache_dir: str = "index_cache", collection_name: str = "story_chunks"):
        self.chunks = chunks
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)  # 确保缓存目录存在
        
        # 初始化嵌入模型（用于向量检索）
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        # 初始化 reranker 模型
        self.reranker = CrossEncoder(RERANKER_MODEL)
        
        # 使用 FAISS 作为向量数据库
        self.index_path = self.cache_dir / "faiss_index.bin" #索引文件路径
        self.metadata_path = self.cache_dir / "faiss_metadata.pkl" #元数据文件路径
        
        # 初始化或加载 FAISS 索引
        self._init_faiss_index()
        
        # 更新索引，添加缺失的文档
        self._update_index()

    def _init_faiss_index(self):
        """初始化或加载 FAISS 索引"""
        if self.index_path.exists() and self.metadata_path.exists(): #检查索引文件路径和元数据文件路径是否存在
            # 加载已存在的索引
            self.faiss_index = faiss.read_index(str(self.index_path))
            with open(self.metadata_path, "rb") as f:
                self.metadata = pickle.load(f)
            print(f"已加载 FAISS 索引，包含 {len(self.metadata['ids'])} 个文档")
        else:
            # 创建新索引
            embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            self.faiss_index = faiss.IndexFlatIP(embedding_dim)  # 使用内积作为相似度度量
            self.metadata = {"ids": [], "texts": [], "sources": [], "langs": [], "chunk_ids": [], "mtimes": []}
            print("已创建新的 FAISS 索引")
    
    def _get_chunk_ids(self):
        """获取所有 chunk 的唯一标识"""
        return [f"{c.lang}_{c.source}_{c.chunk_id}" for c in self.chunks]
    
    def _get_file_modification_time(self, file_path: Path) -> float:
        """获取文件修改时间"""
        try:
            return os.path.getmtime(file_path)
        except OSError:
            return 0.0
    
    def _update_collection(self):
        """更新集合，添加缺失的文档或更新已修改的文档"""
        try:
            existing_data = self.collection.get()
            existing_ids = set(existing_data["ids"])
            existing_metadata = {item["id"]: item for item in zip(existing_data["ids"], existing_data["metadatas"])}
        except:
            existing_ids = set()
            existing_metadata = {}
        
        current_ids = set(self._get_chunk_ids())
        
        # 找出需要添加的新文档
        new_ids = current_ids - existing_ids
        # 找出需要更新的文档（检查文件修改时间）
        updated_ids = set()
        
        # 检查每个现有文档的源文件是否已更新
        for chunk_id in current_ids & existing_ids:
            lang, source, _ = chunk_id.split('_', 2)
            file_path = DATA_DIR / lang / source
            if not file_path.exists():
                continue
                
            current_mtime = self._get_file_modification_time(file_path)
            stored_mtime = existing_metadata.get(chunk_id, {}).get("mtime", 0)
            if current_mtime > stored_mtime:
                updated_ids.add(chunk_id)
        
        # 删除已更新或不再需要的文档
        ids_to_delete = updated_ids | (existing_ids - current_ids)
        if ids_to_delete:
            self.collection.delete(ids=list(ids_to_delete))
            print(f"已删除 {len(ids_to_delete)} 个过期文档")
        
        # 合并需要添加的文档（新文档和已更新的文档）
        ids_to_add = new_ids | updated_ids
        if not ids_to_add:
            print("所有文档都是最新的，无需更新")
            return
        
        # 准备新文档数据
        chunks_to_add = [c for c in self.chunks if f"{c.lang}_{c.source}_{c.chunk_id}" in ids_to_add]
        texts = [c.text for c in chunks_to_add]
        metadatas = []
        
        for c in chunks_to_add:
            file_path = DATA_DIR / c.lang / c.source
            mtime = self._get_file_modification_time(file_path)
            metadatas.append({
                "lang": c.lang,
                "source": c.source,
                "chunk_id": c.chunk_id,
                "mtime": mtime
            })
        
        ids = [f"{c.lang}_{c.source}_{c.chunk_id}" for c in chunks_to_add]
        
        # 生成嵌入向量
        print(f"为 {len(chunks_to_add)} 个文档生成嵌入向量...")
        start_time = time.time()
        embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        encoding_time = time.time() - start_time
        print(f"向量生成完成，耗时 {encoding_time:.2f} 秒")
        
        # 添加到集合
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        print(f"已添加 {len(chunks_to_add)} 个文档到向量数据库")

    def search(self, query: str, top_k: int = TOP_K, lang_filter: str = None) -> List[Tuple[Chunk, float]]:
        """
        使用 FAISS 向量检索 + reranker 混排进行搜索。

        流程：
        1. 使用 FAISS 检索获取 VECTOR_TOP_K 个候选
        2. 使用 reranker 对候选进行重新排序
        3. 返回 top_k 个结果

        Args:
            query: 查询字符串
            top_k: 返回的结果数量
            lang_filter: 可选的语言过滤器，只返回指定语言的结果
        """
        # 使用 FAISS 实现的搜索方法
        return self.search_faiss(query, top_k, lang_filter)
    
    def refresh_index(self):
        """强制刷新索引，重新加载所有文档并更新向量数据库"""
        print("开始刷新向量数据库索引...")
        self._update_index()
        print("索引刷新完成")
        
    def search_faiss(self, query: str, top_k: int = TOP_K, lang_filter: str = None) -> List[Tuple[Chunk, float]]:
        """
        使用 FAISS 向量检索 + reranker 混排进行搜索。

        流程：
        1. 使用 FAISS 检索获取 VECTOR_TOP_K 个候选
        2. 使用 reranker 对候选进行重新排序
        3. 返回 top_k 个结果

        Args:
            query: 查询字符串
            top_k: 返回的结果数量
            lang_filter: 可选的语言过滤器，只返回指定语言的结果
        """
        # 第一步：FAISS 检索获取候选
        # 将查询转换为嵌入向量
        query_embedding = self.embedding_model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )# 返回的形状为(向量模型的嵌入维度,)
        # 这里转二维数组是因为embedding_model.encode()中query只有一条查询语句,其返回值为一维数组，为了配合faiss_index.search()的输入格式，需要将其转换为二维数组
        query_embedding = query_embedding.reshape(1, -1)  # 转换为二维数组
        
        # 从 FAISS 索引获取候选
        distances, indices = self.faiss_index.search(query_embedding, VECTOR_TOP_K)
        
        # 转换结果格式
        vector_candidates = []
        for i, idx in enumerate(indices[0]):#indices[0]:访问第一个查询的结果
            if idx == -1:  # FAISS 返回 -1 表示没有足够的候选
                break
                
            # 如果有语言过滤器，检查是否匹配
            if lang_filter and self.metadata["langs"][idx] != lang_filter:
                continue
                
            chunk = Chunk(
                text=self.metadata["texts"][idx],
                lang=self.metadata["langs"][idx],
                source=self.metadata["sources"][idx],
                chunk_id=self.metadata["chunk_ids"][idx]
            )
            # FAISS 返回的是内积，已经归一化，可以直接作为相似度分数
            similarity = float(distances[0][i])
            vector_candidates.append((chunk, similarity))
            
        # 如果候选数量少于 top_k，直接返回
        if len(vector_candidates) <= top_k:
            return vector_candidates
            
        # 第二步：使用 reranker 重新排序
        # 准备 reranker 的输入：query 和每个候选文本的配对
        pairs = [[query, chunk.text] for chunk, _ in vector_candidates]
        
        # reranker 返回相关性分数
        rerank_scores = self.reranker.predict(pairs)
        
        # 归一化向量相似度到 0-1 区间（候选范围内）
        vec_scores = np.array([v for _, v in vector_candidates], dtype=float)
        vec_min, vec_max = float(vec_scores.min()), float(vec_scores.max())
        if vec_max - vec_min > 1e-8:
            vec_norm = (vec_scores - vec_min) / (vec_max - vec_min)
        else:
            vec_norm = np.zeros_like(vec_scores)
            
        # 归一化 reranker 分数（softmax 到 0-1）
        rerank_arr = np.array(rerank_scores, dtype=float) #将reranker返回的分数列表转换为Numpy浮点数组
        rerank_exp = np.exp(rerank_arr - rerank_arr.max()) #对reranker分数进行指数运算，并减去最大值以防止溢出
        rerank_norm = rerank_exp / rerank_exp.sum() #将指数运算后的分数归一化到0-1区间
        
        # 按比例混合得到最终得分
        alpha = max(0.0, min(1.0, RERANK_MIX_ALPHA)) #获取reranker和向量检索之前的比例
        final_scores = alpha * rerank_norm + (1 - alpha) * vec_norm #按比例混合分数
        
        reranked = [
            (chunk, float(score))
            for (chunk, _), score in zip(vector_candidates, final_scores)
        ]
        # 按混合后的综合分数排序
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        # 返回 top_k 个结果
        return reranked[:top_k]
        
    def _update_index(self):
        """更新索引，添加缺失的文档或更新已修改的文档"""
        existing_ids = set(self.metadata["ids"])
        current_ids = set(self._get_chunk_ids())
        
        # 找出需要添加的新文档
        new_ids = current_ids - existing_ids
        # 找出需要更新的文档（检查文件修改时间）
        updated_ids = set()
        
        # 检查每个现有文档的源文件是否已更新
        for i, chunk_id in enumerate(existing_ids):
            if chunk_id not in current_ids:
                continue
                
            lang = self.metadata["langs"][i]
            source = self.metadata["sources"][i]
            file_path = DATA_DIR / lang / source
            if not file_path.exists():
                continue
                
            current_mtime = self._get_file_modification_time(file_path)
            stored_mtime = self.metadata["mtimes"][i]
            if current_mtime > stored_mtime:
                updated_ids.add(chunk_id)
        
        # 合并需要添加的文档（新文档和已更新的文档）
        ids_to_add = new_ids | updated_ids
        if not ids_to_add:
            print("所有文档都是最新的，无需更新")
            return
            
        # 准备新文档数据
        chunks_to_add = [c for c in self.chunks if f"{c.lang}_{c.source}_{c.chunk_id}" in ids_to_add]
        texts = [c.text for c in chunks_to_add]
        
        # 生成嵌入向量
        print(f"为 {len(chunks_to_add)} 个文档生成嵌入向量...")
        start_time = time.time()
        embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        encoding_time = time.time() - start_time
        print(f"向量生成完成，耗时 {encoding_time:.2f} 秒")
        
        # 添加到索引
        self.faiss_index.add(embeddings)
        
        # 更新元数据
        for c in chunks_to_add:
            file_path = DATA_DIR / c.lang / c.source
            mtime = self._get_file_modification_time(file_path)
            
            self.metadata["ids"].append(f"{c.lang}_{c.source}_{c.chunk_id}")
            self.metadata["texts"].append(c.text)
            self.metadata["sources"].append(c.source)
            self.metadata["langs"].append(c.lang)
            self.metadata["chunk_ids"].append(c.chunk_id)
            self.metadata["mtimes"].append(mtime)
        
        # 保存索引和元数据
        faiss.write_index(self.faiss_index, str(self.index_path))
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)
            
        print(f"已添加 {len(chunks_to_add)} 个文档到 FAISS 索引")


def build_prompt(query: str, hits: List[Tuple[Chunk, float]]) -> str:
    citation_blocks = []
    for chunk, score in hits:
        citation_blocks.append(f"[{chunk.citation}] (sim={score:.2f})\n{chunk.text}") #保存引用文本块的来源分数及内容
    context = "\n\n".join(citation_blocks)
    return (
        "You are a helpful assistant answering questions about children's stories. "
        "Use the context chunks to answer. Always cite sources in the form [language/file#chunk].\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer with citations."
    )


def generate_answer(
        query: str,#查询字符串
        index: RagIndexer,#向量检索模型
        client: DeepSeekClient,#DeepSeekClient客户端
        answer_language: str | None = None,
        lang_filter: str | None = None,
) -> Tuple[str, List[Chunk]]: #返回一个元组，包含生成的回答和对应的文本块列表
    hits = index.search_faiss(query, top_k=TOP_K, lang_filter=lang_filter) #返回检索分数最高的前TOP_K个文本块
    prompt = build_prompt(query, hits) #构建提示字符串（包含检索的文本块及问题）
    lang_instr = (
        ""
        if not answer_language
        else f" Please respond in {answer_language}."
    ) #指定回答语言
    messages = [
        {
            "role": "system",
            "content": "You are a concise bilingual assistant. Default to responding in Chinese unless otherwise specified."+ lang_instr,
        },
        {"role": "user", "content": prompt},
    ] #创建一个消息列表，遵循OpenAI API的消息格式
    reply = client.chat_completion(messages) #发送消息给语言模型
    ordered_chunks = [chunk for chunk, _ in hits]
    return reply, ordered_chunks
