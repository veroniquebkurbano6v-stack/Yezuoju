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


class Citation(BaseModel):
    citation: str
    text: str


class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    detected_language: Optional[str] = None


@lru_cache(maxsize=16)
def _load_index(language: Optional[str], mode: str, book: Optional[str]) -> RagIndexer:
    chunks = load_corpus(language=language, mode=mode, book_name=book)
    return RagIndexer(chunks)


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
        "ja": "Japanse",
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
    app = FastAPI(title="StoryRag API", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/books")
    def books() -> Dict[str, List[str]]:
        return list_books()

    @app.post("/api/query", response_model=QueryResponse)
    def query_rag(payload: QueryRequest) -> QueryResponse:
        lang = payload.language or detect_language(payload.query)
        if payload.mode == "single" and payload.book and not lang:
            lang = find_book_language(payload.book) or lang

        try:
            index = _load_index(lang, payload.mode, payload.book)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"加载语料失败: {exc}") from exc

        client = DeepSeekClient()
        try:
            answer, contexts = generate_answer(payload.query, index, client, answer_language=lang)
        except DeepSeekError as exc:
            raise HTTPException(status_code=502, detail=f"调用 DeepSeek 失败: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        citations = [
            Citation(citation=c.citation, text=c.text[:500])  # limit length for payload
            for c in contexts
        ]
        return QueryResponse(answer=answer, citations=citations, detected_language=lang)

    static_dir = os.path.join(os.path.dirname(__file__), "frontend")
    if os.path.isdir(static_dir):
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

