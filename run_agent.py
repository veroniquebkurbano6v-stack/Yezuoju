#!/usr/bin/env python3
"""
One-step runner: initializes retrieval client (and optional DeepSeek agent) and answers user queries.
- If DEEPSEEK_API_KEY is set, will initialize DeepSeekRetrievalAgent and use its LLM to produce an answer.
- Otherwise will call the LangChain-wrapped smart_retrieval implementation directly and return retrieval results.

Outputs structured JSON including referenced text blocks, section titles and page numbers.
"""
import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# make sure src is importable
sys.path.insert(0, str(Path(__file__).parent.joinpath("src")))

from langchain_tools import get_langchain_tools
from deepseek_agent import create_deepseek_agent

def format_search_results(results):
    out = []
    for r in results:
        try:
            text = getattr(r, "text", "") or (r.get("text") if isinstance(r, dict) else "")
        except Exception:
            text = ""
        preview = (text[:800].replace("\n", " ")) if text else ""
        section = getattr(r, "section_title", None) or (r.get("section_title") if isinstance(r, dict) else None)
        page = getattr(r, "page_number", None) or (r.get("page_number") if isinstance(r, dict) else None)
        score = getattr(r, "score", None) or (r.get("score") if isinstance(r, dict) else None)
        pdf = getattr(r, "pdf_filename", None) or getattr(r, "pdf_path", None) or (r.get("pdf_filename") if isinstance(r, dict) else None)
        out.append({
            "text_preview": preview,
            "section_title": section,
            "page_number": int(page) if page is not None else None,
            "score": float(score) if score is not None else None,
            "pdf_filename": Path(pdf).name if pdf else None
        })
    return out

def run_query(query: str, top_k: int = 60):
    vector_db_path = os.getenv("VECTOR_DB_PATH", "src/data/vector_database")
    embedding_model = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")

    # initialize retrieval tools
    tools = get_langchain_tools(vector_db_path, embedding_model)

    # Always perform retrieval first (smart_retrieval impl or hybrid fallback)
    retrieved_objs = []
    if hasattr(tools, "smart_retrieval"):
        try:
            retrieved_objs = tools.smart_retrieval(query, top_k=top_k) or []
        except Exception:
            retrieved_objs = []

    # If smart_retrieval returned nothing, try hybrid_search as fallback
    if not retrieved_objs:
        try:
            retrieved_objs = tools.retriever.hybrid_search(query, top_k=top_k) or []
        except Exception:
            retrieved_objs = []

    # If still empty, try the LangChain tool wrapper which returns JSON
    if not retrieved_objs:
        try:
            out_json = tools.tools[0](query, top_k=top_k)
            parsed = json.loads(out_json)
            retrieved_list = parsed.get("results", [])
            # return retrieval-only if LLM not available
            if retrieved_list:
                return {"mode": "direct_tool_wrapper", "retrieved_chunks": retrieved_list}
        except Exception:
            pass

    # format retrieved chunks for prompt
    retrieved_struct = format_search_results(retrieved_objs)

    # Build prompt for LLM using retrieved chunks
    prompt_lines = [f"用户问题: {query}", "", "检索到的文档片段（用于回答）："]
    for i, r in enumerate(retrieved_struct[:10], 1):
        prompt_lines.append(f"{i}. 文件: {r.get('pdf_filename')}, 章节: {r.get('section_title')}, 页码: {r.get('page_number')}")
        prompt_lines.append(r.get("text_preview") or "")
        prompt_lines.append("")
    prompt_lines.append("请基于以上检索到的文档片段回答用户问题，并在回答中引用具体来源（文件名、章节、页码）。")
    prompt = "\n".join(prompt_lines)

    # If DeepSeek LLM available, use it to generate answer constrained by retrieved chunks
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if api_key:
        try:
            agent = create_deepseek_agent(vector_db_path, api_key=api_key, base_url=os.getenv("DEEPSEEK_BASE_URL"))
            llm = agent.llm
            llm_response = llm.invoke(prompt)
            answer = getattr(llm_response, "content", str(llm_response))
            # Append structured citations to the answer
            citations = []
            for i, r in enumerate(retrieved_struct, 1):
                pdf = r.get("pdf_filename") or "unknown"
                sec = r.get("section_title") or "未知章节"
                page = r.get("page_number") or "?"
                citations.append(f"[{i}] {pdf} | {sec} | 页 {page}")
            if citations:
                answer = answer.strip() + "\n\n引用：\n" + "\n".join(citations)
            return {"mode": "agent_with_llm", "answer": answer, "retrieved_chunks": retrieved_struct}
        except Exception as e:
            print(f"Warning: DeepSeek agent/LLM failed, returning retrieval only: {e}", file=sys.stderr)

    # No LLM available: return retrieval-only results
    return {"mode": "direct_retrieval", "retrieved_chunks": retrieved_struct}

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run a single query or interactive loop against the retrieval system")
    parser.add_argument("--query", "-q", type=str, help="single query to run")
    parser.add_argument("--top-k", type=int, default=60, help="number of candidate chunks to retrieve")
    args = parser.parse_args()

    if args.query:
        out = run_query(args.query, top_k=args.top_k)
        s = json.dumps(out, ensure_ascii=False, indent=2)
        try:
            print(s)
        except UnicodeEncodeError:
            # On some Windows consoles the default encoding cannot print all Unicode chars
            import sys
            sys.stdout.buffer.write(s.encode("utf-8"))
            sys.stdout.buffer.write(b"\n")
        return 0

    print("欢迎使用PDF智能检索系统！\n请输入您的问题，按Enter键查询。输入空行或按Ctrl+C退出系统。")
    try:
        while True:
            q = input("\n> ").strip()
            if not q:
                break
            resp = run_query(q, top_k=args.top_k)
            s = json.dumps(resp, ensure_ascii=False, indent=2)
            try:
                print(s)
            except UnicodeEncodeError:
                import sys
                sys.stdout.buffer.write(s.encode("utf-8"))
                sys.stdout.buffer.write(b"\n")
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())


