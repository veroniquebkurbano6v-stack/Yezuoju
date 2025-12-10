"""Lightweight RAG indexer for tri-lingual stories (Chinese/English/Japanse)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

from dotenv import load_dotenv
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from llm import DeepSeekClient

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).parent / "Ragdate"))
SUPPORTED_LANGS = [
    lang.strip()
    for lang in os.getenv("LANGS", "Chinese,English,Japanse").split(",")
    if lang.strip()
]
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "Chinese")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
TOP_K = int(os.getenv("TOP_K", "4"))


@dataclass
class Chunk:
    text: str
    lang: str
    source: str
    chunk_id: int

    @property
    def citation(self) -> str:
        return f"{self.lang}/{self.source}#chunk{self.chunk_id}"


def _read_plain(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


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
) -> List[Chunk]:
    lang = language or DEFAULT_LANGUAGE
    if lang not in SUPPORTED_LANGS:
        raise ValueError(f"Unsupported language {lang}, choose from {SUPPORTED_LANGS}")

    lang_dir = DATA_DIR / lang
    if not lang_dir.exists():
        raise FileNotFoundError(f"Data folder not found: {lang_dir}")

    files: Iterable[Path]
    if mode == "single":
        if not book_name:
            raise ValueError("single mode requires book_name")
        files = [lang_dir / book_name]
    else:
        files = sorted(lang_dir.glob("*"))

    chunks: List[Chunk] = []
    for file_path in files:
        if not file_path.is_file() or file_path.suffix.lower() != ".txt":
            continue
        text = load_story_text(file_path)
        parts = chunk_text(text)
        for idx, part in enumerate(parts):
            if not part.strip():
                continue
            chunks.append(
                Chunk(
                    text=part.strip(),
                    lang=lang,
                    source=file_path.name,
                    chunk_id=idx,
                )
            )
    if not chunks:
        raise ValueError(
            f"No chunks built for language={lang}, mode={mode}. "
            "Only .txt files are supported; ensure txt files exist under the language folder."
        )
    return chunks


class RagIndexer:
    """Minimal TF-IDF based retriever with citation tracking."""

    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer()
        texts = [c.text for c in chunks]
        self.matrix = self.vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = TOP_K) -> List[Tuple[Chunk, float]]:
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.matrix).flatten()
        scored = sorted(
            ((self.chunks[i], score) for i, score in enumerate(sims)),
            key=lambda x: x[1],
            reverse=True,
        )
        return scored[:top_k]


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
    query: str,
    index: RagIndexer,
    client: DeepSeekClient,
    answer_language: str | None = None,
) -> Tuple[str, List[Chunk]]:
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

