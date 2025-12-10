"""CLI entry for the tri-lingual Story RAG demo."""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

from langdetect import detect

from llm import DeepSeekClient, DeepSeekError
from rag import RagIndexer, generate_answer, load_corpus

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Story RAG (Chinese/English/Japanse)")
    parser.add_argument("query", help="Question to ask about the stories")
    parser.add_argument(
        "--language",
        default=None,
        help="Language folder name (default from .env DEFAULT_LANGUAGE)",
    )
    parser.add_argument(
        "--mode",
        choices=["single", "library"],
        default="library",
        help="single=only one file, library=all files in the language folder",
    )
    parser.add_argument(
        "--book",
        help="File name inside the language folder when using --mode single",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    language = args.language
    if not language:
        try:
            code = detect(args.query)
            if code.startswith("zh"):
                language = "Chinese"
            elif code.startswith("en"):
                language = "English"
            elif code.startswith("ja"):
                language = "Japanse"
        except Exception:  # noqa: BLE001
            language = None
    try:
        chunks = load_corpus(language=language, mode=args.mode, book_name=args.book)
    except Exception as exc:  # noqa: BLE001
        print(f"加载语料失败: {exc}")
        return 1

    index = RagIndexer(chunks)
    client = DeepSeekClient()
    try:
        answer, contexts = generate_answer(args.query, index, client, answer_language=language)
    except DeepSeekError as exc:
        print(f"调用 DeepSeek 失败: {exc}")
        return 1

    print("\n--- 模型回复 ---")
    print(answer)
    print("\n--- 引用片段 ---")
    for c in contexts:
        print(f"[{c.citation}] {c.text[:180]}{'...' if len(c.text) > 180 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

