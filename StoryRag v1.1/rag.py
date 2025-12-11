"""Lightweight RAG indexer for tri-lingual stories (Chinese/English/Japanse) with vector search and reranker."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder

from llm import DeepSeekClient

load_dotenv()
#_file__:表示当前文件的路径  Path(__file__).parent:获取当前文件所在的目录    /"Ragdate":在该目录下创建一个名为"Ragdate"的子目录路径
DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).parent / "Ragdate"))
SUPPORTED_LANGS = [  # 变量 = [表达式 for 循环变量 in 可迭代对象 if 条件]
    lang.strip()  # 去除每个语言名称前后的空白字符串
    for lang in os.getenv("LANGS", "Chinese,English,Japanse").split(",")
    if lang.strip()
]
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "Chinese")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
TOP_K = int(os.getenv("TOP_K", "4"))
# 向量检索获取的候选数量（用于 reranker 重排序）
VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "20"))
# 嵌入模型名称（支持多语言）
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
# Reranker 模型名称
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
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
    return _read_plain(path)


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
            text = load_story_text(file_path)#加载对应资源路径的图书文本
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
    """Vector-based retriever with reranker for tri-lingual stories."""

    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        # 初始化嵌入模型（用于向量检索）
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        # 初始化 reranker 模型
        self.reranker = CrossEncoder(RERANKER_MODEL)
        # 预计算所有 chunk 的嵌入向量
        texts = [c.text for c in chunks]#提取Chunk对象中的文本块
        self.embeddings = self.embedding_model.encode(
            texts, 
            convert_to_numpy=True, # 将嵌入向量转换为numpy数组
            show_progress_bar=False, # 不显示进度条
            normalize_embeddings=True  # 归一化以便使用余弦相似度
        )

    def search(self, query: str, top_k: int = TOP_K) -> List[Tuple[Chunk, float]]:
        """
        使用向量检索 + reranker 混排进行搜索。
        
        流程：
        1. 使用向量检索获取 VECTOR_TOP_K 个候选
        2. 使用 reranker 对候选进行重新排序
        3. 返回 top_k 个结果
        """
        # 第一步：向量检索获取候选
        # 将查询转换为嵌入向量
        query_embedding = self.embedding_model.encode(
            query, 
            convert_to_numpy=True, 
            normalize_embeddings=True
        )
        # 计算余弦相似度
        similarities = np.dot(self.embeddings, query_embedding)
        # 获取向量检索的 top candidates
        vector_candidates = sorted(
            ((self.chunks[i], float(similarities[i])) for i in range(len(self.chunks))),
            key=lambda x: x[1],#指定按元组的第二个元素进行排序
            reverse=True,#降序排列
        )[:VECTOR_TOP_K]
        
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
        rerank_arr = np.array(rerank_scores, dtype=float)
        rerank_exp = np.exp(rerank_arr - rerank_arr.max())
        rerank_norm = rerank_exp / rerank_exp.sum()

        # 按比例混合得到最终得分
        alpha = max(0.0, min(1.0, RERANK_MIX_ALPHA))
        final_scores = alpha * rerank_norm + (1 - alpha) * vec_norm

        reranked = [
            (chunk, float(score))
            for (chunk, _), score in zip(vector_candidates, final_scores)
        ]
        # 按混合后的综合分数排序
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        # 返回 top_k 个结果
        return reranked[:top_k]


def build_prompt(query: str, hits: List[Tuple[Chunk, float]]) -> str:
    citation_blocks = []
    for chunk, score in hits:
        citation_blocks.append(f"[{chunk.citation}] (sim={score:.2f})\n{chunk.text}")
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
) -> Tuple[str, List[Chunk]]: #返回一个元组，包含生成的回答和对应的文本块列表
    hits = index.search(query, top_k=TOP_K)
    prompt = build_prompt(query, hits)
    lang_instr = (
        ""
        if not answer_language
        else f" Please respond in {answer_language}."
    )
    messages = [
        {
            "role": "system",
            "content": "You are a concise bilingual assistant." + lang_instr,
        },
        {"role": "user", "content": prompt},
    ]
    reply = client.chat_completion(messages)
    ordered_chunks = [chunk for chunk, _ in hits]
    return reply, ordered_chunks
