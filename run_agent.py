#!/usr/bin/env python3
"""
调试工具：直接调用 DeepSeekRetrievalAgent 的完整功能
- 如果设置了 DEEPSEEK_API_KEY，使用完整的 DeepSeek 代理（包括智能检索和 LLM 回答）
- 否则仅使用基础检索功能

用于测试和调试 deepseek_agent.py 的各项功能
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

def _direct_retrieval_mode(query: str, top_k: int, vector_db_path: str, embedding_model: str):
    """直接检索模式（无 LLM，仅用于降级或无 API Key 时）"""
    print(f"[DEBUG] 进入直接检索模式...", file=sys.stderr)
    
    tools = get_langchain_tools(vector_db_path, embedding_model)
    
    retrieved_objs = []
    if hasattr(tools, "smart_retrieval_impl"):
        try:
            # 使用 smart_retrieval_impl 进行检索
            retrieved_objs = tools.smart_retrieval_impl(query, top_k=top_k) or []
            print(f"[DEBUG] smart_retrieval_impl 检索到 {len(retrieved_objs)} 条结果", file=sys.stderr)
        except Exception as e:
            print(f"[DEBUG] smart_retrieval_impl 失败：{e}", file=sys.stderr)
            retrieved_objs = []
    
    if not retrieved_objs:
        try:
            retrieved_objs = tools.retriever.hybrid_search(query, top_k=top_k) or []
            print(f"[DEBUG] hybrid_search 检索到 {len(retrieved_objs)} 条结果", file=sys.stderr)
        except Exception:
            retrieved_objs = []
    
    # 格式化结果
    retrieved_struct = format_search_results(retrieved_objs)
    
    return {
        "mode": "direct_retrieval",
        "retrieved_chunks": retrieved_struct,
        "answer": "[直接检索模式] " + (f"检索到 {len(retrieved_struct)} 条相关文档片段，但无 LLM 可用，无法生成结构化回答。" if retrieved_struct else "未检索到相关内容。"),
        "success": True,
        "debug_info": {
            "fallback_mode": True,
            "chunks_found": len(retrieved_struct)
        }
    }

def check_and_build_index(vector_db_path: str, json_dir: str, source_dir: str) -> bool:
    """
    检查向量数据库是否需要构建，如果为空则自动调用 process_pipeline.py
    
    Args:
        vector_db_path: 向量数据库路径
        json_dir: JSON 文件目录
        source_dir: PDF 源目录
    
    Returns:
        bool: 是否成功构建索引
    """
    import subprocess
    from pathlib import Path
    
    # 检查数据库目录是否存在
    db_path = Path(vector_db_path)
    if not db_path.exists():
        print(f"[DEBUG] 向量数据库不存在：{vector_db_path}", file=sys.stderr)
        need_build = True
    else:
        # 检查数据库中是否有数据
        try:
            import chromadb
            from chromadb.config import Settings
            client = chromadb.PersistentClient(
                path=str(db_path),
                settings=Settings(allow_reset=True, anonymized_telemetry=False)
            )
            collection = client.get_or_create_collection(name="document_chunks")
            count = collection.count()
            
            if count == 0:
                print(f"[DEBUG] 向量数据库为空（0 个文档块）：{vector_db_path}", file=sys.stderr)
                need_build = True
            else:
                print(f"[DEBUG] 向量数据库已有 {count} 个文档块", file=sys.stderr)
                need_build = False
        except Exception as e:
            print(f"[DEBUG] 检查向量数据库失败：{e}，将尝试重建", file=sys.stderr)
            need_build = True
    
    if not need_build:
        return True
    
    # 检查 JSON 文件是否存在
    json_count = len(list(Path(json_dir).glob("**/*.json")))
    pdf_count = len(list(Path(source_dir).rglob("*.pdf")))
    
    if json_count == 0 and pdf_count == 0:
        print(f"[ERROR] 在 {source_dir} 中未找到 PDF 文件，无法构建索引", file=sys.stderr)
        print(f"[ERROR] 请先将 PDF 文件放入 {source_dir} 目录", file=sys.stderr)
        return False
    
    # 自动调用 process_pipeline.py 构建索引
    print(f"\n{'='*80}", file=sys.stderr)
    print(f"⚠️  检测到向量数据库需要构建...", file=sys.stderr)
    print(f"📊 当前状态:", file=sys.stderr)
    print(f"   - PDF 文件数：{pdf_count}", file=sys.stderr)
    print(f"   - JSON 文件数：{json_count}", file=sys.stderr)
    print(f"{'='*80}", file=sys.stderr)
    
    if json_count == 0:
        print(f"[INFO] 开始生成 JSON 文件...", file=sys.stderr)
        try:
            # 调用 process_pipeline.py --mode json-only
            script_path = Path(__file__).parent / "src" / "process_pipeline.py"
            result = subprocess.run(
                [sys.executable, str(script_path), "--mode", "json-only"],
                check=True,
                capture_output=False
            )
            print(f"[INFO] JSON 文件生成完成", file=sys.stderr)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] 生成 JSON 文件失败：{e}", file=sys.stderr)
            return False
    
    print(f"[INFO] 开始向量化处理...", file=sys.stderr)
    try:
        # 调用 process_pipeline.py --mode vector-only
        script_path = Path(__file__).parent / "src" / "process_pipeline.py"
        result = subprocess.run(
            [sys.executable, str(script_path), "--mode", "vector-only"],
            check=True,
            capture_output=False
        )
        print(f"[INFO] 向量化处理完成", file=sys.stderr)
        print(f"✅ 索引构建成功！\n", file=sys.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] 向量化处理失败：{e}", file=sys.stderr)
        return False

def run_query(query: str, top_k: int = 30, chat_history: list = None):
    """
    执行单次查询（直接调用 DeepSeekRetrievalAgent.chat 方法）
    
    Args:
        query: 用户问题
        top_k: 检索结果数量（此参数在完整模式下由 agent 内部处理）
        chat_history: 对话历史记录
    
    Returns:
        包含回答和检索结果的字典
    """
    vector_db_path = os.getenv("VECTOR_DB_PATH", "src/data/vector_database")
    embedding_model = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
    json_dir = os.getenv("JSON_DIR", "src/data/pages_title")
    source_dir = os.getenv("SOURCE_DIR", "src/data/source")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    
    # 自动检查并构建索引
    if not check_and_build_index(vector_db_path, json_dir, source_dir):
        return {
            "mode": "error",
            "answer": "❌ 向量数据库索引构建失败，请检查 PDF 文件是否存在或手动运行：python src/process_pipeline.py --mode full",
            "success": False,
            "debug_info": {
                "error": "index_build_failed"
            }
        }
    
    # 如果有 DeepSeek API Key，使用完整的 DeepSeek 代理
    if api_key:
        try:
            print(f"[DEBUG] 初始化 DeepSeek 代理...", file=sys.stderr)
            
            # 初始化代理（复用工具实例，避免重复初始化）
            tools = get_langchain_tools(vector_db_path, embedding_model)
            agent = create_deepseek_agent(
                vector_db_path, 
                api_key=api_key, 
                base_url=os.getenv("DEEPSEEK_BASE_URL"),
                tools_instance=tools
            )
            
            print(f"[DEBUG] 调用 agent.chat() 方法...", file=sys.stderr)
            
            # 直接调用完整的 chat 方法
            result = agent.chat(user_input=query, chat_history=chat_history)
            
            # 添加调试信息
            result["mode"] = "deepseek_agent_full"
            result["debug_info"] = {
                "agent_initialized": True,
                "tools_reused": True,
                "chat_method_used": True
            }
            
            print(f"[DEBUG] 查询完成，answer_source={result.get('answer_source')}", file=sys.stderr)
            return result
            
        except Exception as e:
            print(f"[DEBUG] DeepSeek agent 失败，降级到直接检索模式：{e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            # 降级到直接检索模式
            return _direct_retrieval_mode(query, top_k, vector_db_path, embedding_model)
    
    # 没有 API Key，使用直接检索模式
    print(f"[DEBUG] 未设置 DEEPSEEK_API_KEY，使用直接检索模式", file=sys.stderr)
    return _direct_retrieval_mode(query, top_k, vector_db_path, embedding_model)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run a single query or interactive loop against the retrieval system")
    parser.add_argument("--query", "-q", type=str, help="single query to run")
    parser.add_argument("--top-k", type=int, default=45, help="number of candidate chunks to retrieve")
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

    print("欢迎使用 PDF 智能检索系统（调试模式）！")
    print("本工具直接调用 deepseek_agent.py 的完整功能")
    print("请输入您的问题，按 Enter 键查询。输入空行或按 Ctrl+C 退出系统。")
        
    # 交互式模式：维护对话历史
    chat_history = []
    try:
        while True:
            q = input("\n> ").strip()
            if not q:
                break
                
            resp = run_query(q, top_k=args.top_k, chat_history=chat_history)
                
            # 更新对话历史
            chat_history.append({"role": "user", "content": q})
            if resp.get("success") and resp.get("answer"):
                chat_history.append({"role": "assistant", "content": resp["answer"]})
                
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


