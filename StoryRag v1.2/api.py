"""FastAPI server exposing RAG query API and serving a simple frontend."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from langdetect import DetectorFactory, detect

from llm import DeepSeekClient, DeepSeekError
from rag import RagIndexer, SUPPORTED_LANGS, generate_answer, load_corpus, DATA_DIR

load_dotenv()
DetectorFactory.seed = 42
class QueryRequest(BaseModel):
    query: str = Field(..., description="User question")
    language: Optional[str] = Field(None, description="Language folder; if empty, auto-detect")
    mode: str = Field("library", description="library or single")
    book: Optional[str] = Field(None, description="File name when mode=single")
    refresh_index: bool = Field(False, description="Whether to refresh the vector index")
    search_lang: Optional[str] = Field(None, description="Language to search in (independent of query language)")


class Citation(BaseModel):
    citation: str
    text: str


class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    detected_language: Optional[str] = None


# 缓存索引器实例，但允许刷新
_index_cache = {}

def _load_index(language: Optional[str], mode: str, book: Optional[str], load_all_languages: bool = True, refresh: bool = False) -> RagIndexer:
    """
    加载或创建RagIndexer实例
    
    Args:
        language: 语言过滤
        mode: 模式 ("library" 或 "single")
        book: 书名
        load_all_languages: 是否加载所有语言
        refresh: 是否强制刷新索引
    
    Returns:
        RagIndexer实例
    """
    # 生成缓存键
    cache_key = f"{language}_{mode}_{book}_{load_all_languages}"
    
    # 如果需要刷新或缓存中没有，则创建新实例
    if refresh or cache_key not in _index_cache:
        chunks = load_corpus(language=language, mode=mode, book_name=book, load_all_languages=load_all_languages)
        # 使用缓存目录创建RagIndexer实例
        collection_name = f"story_chunks_{cache_key}"
        indexer = RagIndexer(chunks, cache_dir="index_cache", collection_name=collection_name)
        
        # 如果需要刷新，强制更新索引
        if refresh:
            indexer.refresh_index()
        
        # 更新缓存
        _index_cache[cache_key] = indexer
    
    return _index_cache[cache_key]


def detect_language(text: str) -> Optional[str]:
    if not text:
        return None
    try:
        code = detect(text)
    except Exception:  # noqa: BLE001
        return None
    mapping = {
        "zh-cn": "Chinese",
        "zh": "Chinese",
        "en": "English",
        "ja": "Japan",
    }
    return mapping.get(code)


def find_book_language(book: str) -> Optional[str]:
    """If language not provided, try to locate which language folder contains the book."""
    for lang in SUPPORTED_LANGS:
        candidate = os.path.join(DATA_DIR, lang, book)
        if os.path.isfile(candidate):
            return lang
    return None


def list_books() -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    for lang in SUPPORTED_LANGS:
        lang_dir = os.path.join(DATA_DIR, lang)
        if not os.path.isdir(lang_dir):
            continue
        files = [
            name
            for name in os.listdir(lang_dir)
            if os.path.isfile(os.path.join(lang_dir, name)) and name.lower().endswith(".txt")
        ]
        result[lang] = sorted(files)
    return result


def create_app() -> FastAPI:
    """
    创建并配置 FastAPI 应用实例
    Returns:
        FastAPI: 配置完成的 FastAPI 应用实例
    """
    # 初始化 FastAPI 应用，设置标题和版本号
    app = FastAPI(title="StoryRag API", version="0.1.2")

    # 添加 CORS 中间件，允许跨域请求
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 允许所有来源的请求
        allow_credentials=True,  # 允许发送凭据信息
        allow_methods=["*"],  # 允许所有 HTTP 方法
        allow_headers=["*"],  # 允许所有请求头
    )

    # 健康检查接口，用于验证服务是否正常运行
    @app.get("/api/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    # 获取书籍列表接口
    @app.get("/api/books")
    def books() -> Dict[str, List[str]]:
        return list_books()
        
    # 更新索引接口
    @app.post("/api/refresh_index")
    def refresh_index(payload: QueryRequest) -> Dict[str, str]:
        """
        刷新知识库索引接口
        
        Args:
            payload: 包含语言、模式和书名的请求体
            
        Returns:
            包含状态信息的字典
        """
        try:
            # 检测语言，如果没有指定则自动检测
            lang = payload.language
            # 如果是单本书查询模式且未指定语言，尝试从书中获取语言信息
            if payload.mode == "single" and payload.book and not lang:
                lang = find_book_language(payload.book) or lang
                
            # 加载索引并强制刷新
            load_all = payload.language is None  # 如果未指定语言，则加载所有语言
            index = _load_index(lang, payload.mode, payload.book, load_all_languages=load_all, refresh=True)
            
            return {"status": "success", "message": f"已成功刷新索引: {lang or '所有语言'}"}
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"刷新索引失败: {exc}") from exc

    # 查询接口，接收查询请求并返回结果
    @app.post("/api/query", response_model=QueryResponse)
    def query_rag(payload: QueryRequest) -> QueryResponse:
        # 检测查询语言，如果没有指定则自动检测
        doc_lang = payload.language or detect_language(payload.query)
        # 如果是单本书查询模式且未指定语言，尝试从书中获取语言信息
        if payload.mode == "single" and payload.book and not doc_lang:
            doc_lang = find_book_language(payload.book) or doc_lang

        try:
            # 默认加载所有语言的数据，使用向量数据库检索 + reranker 混排
            load_all = payload.language is None  # 如果未指定语言，则加载所有语言
            index = _load_index(doc_lang, payload.mode, payload.book, load_all_languages=load_all, refresh=payload.refresh_index)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"加载语料失败: {exc}") from exc

        client = DeepSeekClient()
        try:
            # 使用文档语言进行搜索，如果没有指定文档语言，则使用自动检测的语言
            doc_lang = payload.language or detect_language(payload.query)
            # 用户语言由search_lang参数指定，如果没有指定，则使用文档语言
            user_lang = payload.search_lang or doc_lang
            answer, contexts = generate_answer(payload.query, index, client, answer_language=user_lang, lang_filter=doc_lang)
        except DeepSeekError as exc:
            raise HTTPException(status_code=502, detail=f"调用 DeepSeek 失败: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        citations = [
            Citation(citation=c.citation, text=c.text[:500])  # limit length for payload
            for c in contexts
        ]
        # 检测回答语言，如果没有指定则自动检测
        detected_user_lang = payload.search_lang or detect_language(payload.query)
        return QueryResponse(answer=answer, citations=citations, detected_language=detected_user_lang)

    static_dir = os.path.join(os.path.dirname(__file__), "frontend")
    if os.path.isdir(static_dir):
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

