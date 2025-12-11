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
    """
    主函数，负责整个流程的控制和协调
    包括参数解析、语言检测、语料加载、索引构建、答案生成和结果展示
    返回0表示成功，1表示失败
    """
    # 解析命令行参数
    args = parse_args()
    # 获取语言参数
    language = args.language
    # 如果未指定语言，尝试自动检测
    if not language:
        try:
            # 检测输入文本的语言代码
            code = detect(args.query)
            # 根据语言代码设置对应语言
            if code.startswith("zh"):
                language = "Chinese"
            elif code.startswith("en"):
                language = "English"
            elif code.startswith("ja"):
                language = "Japanse"
        except Exception:  # noqa: BLE001
            # 如果检测失败，设置语言为None
            language = None
    try:
        # 加载语料库：如果未指定语言，则加载所有语言的数据（使用向量检索 + reranker 混排）
        load_all = language is None
        chunks = load_corpus(language=language, mode=args.mode, book_name=args.book, load_all_languages=load_all)
    except Exception as exc:  # noqa: BLE001
        # 加载失败时打印错误信息并返回错误码1
        print(f"加载语料失败: {exc}")
        return 1

    # 创建检索增强生成(RAG)索引
    index = RagIndexer(chunks)
    # 初始化 DeepSeek 客户端
    client = DeepSeekClient()
    try:
        # 生成答案并获取引用的上下文
        answer, contexts = generate_answer(args.query, index, client, answer_language=language)
    except DeepSeekError as exc:
        # 调用 DeepSeek API 失败时打印错误信息并返回错误码1
        print(f"调用 DeepSeek 失败: {exc}")
        return 1

    # 打印模型回复
    print("\n--- 模型回复 ---")
    print(answer)
    # 打印引用的上下文片段
    print("\n--- 引用片段 ---")
    for c in contexts:
        # 打印每个引用的来源和内容（限制显示长度为180字符）
        print(f"[{c.citation}] {c.text[:180]}{'...' if len(c.text) > 180 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
