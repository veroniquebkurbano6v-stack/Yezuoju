#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
智能文本块拼接脚本

基于章节标题边界，将连续页面的文本块合并成逻辑段落，并进行长度控制。
"""

import sys
import os
import json
import re
from typing import List, Dict, Any
from dotenv import load_dotenv

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.pdf_loader import PDFLoader

# 加载环境变量
load_dotenv()

# === 配置参数 ===
# 从 .env 读取配置（可在 .env 文件中调整）
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))  # 最大中文字符数
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "51"))  # 重叠字符数（约 10%）
SUMMARY_MAX_LENGTH = int(os.getenv("SUMMARY_MAX_LENGTH", "180"))  # 摘要最大长度
SUMMARY_MIN_LENGTH = int(os.getenv("SUMMARY_MIN_LENGTH", "120"))  # 摘要最小长度
NEXT_CHUNK_PREVIEW_LENGTH = int(os.getenv("NEXT_CHUNK_PREVIEW_LENGTH", "50"))  # 下文预览长度

# 🔥 打印配置信息，确认加载正确
print(f"\n配置参数:")
print(f"   CHUNK_SIZE: {CHUNK_SIZE}")
print(f"   CHUNK_OVERLAP: {CHUNK_OVERLAP}")
print(f"   SUMMARY_MAX_LENGTH: {SUMMARY_MAX_LENGTH}")
print(f"   SUMMARY_MIN_LENGTH: {SUMMARY_MIN_LENGTH}")
print(f"   NEXT_CHUNK_PREVIEW_LENGTH: {NEXT_CHUNK_PREVIEW_LENGTH}\n")

# 从 .env 读取配置
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "src/data/vector_database")
PDF_SOURCE_PATH = os.getenv("PDF_SOURCE_PATH", "src/data/source")

# 摘要生成模型配置（从 .env 加载）
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "qwen3:8b")
SUMMARY_MODEL_TEMPERATURE = float(os.getenv("SUMMARY_MODEL_TEMPERATURE", "0.5"))
SUMMARY_MODEL_TOP_P = float(os.getenv("SUMMARY_MODEL_TOP_P", "0.95"))
SUMMARY_MODEL_TOP_K = int(os.getenv("SUMMARY_MODEL_TOP_K", "50"))
SUMMARY_MODEL_MAX_TOKENS = int(os.getenv("SUMMARY_MODEL_MAX_TOKENS", "800"))
SUMMARY_MODEL_REPEAT_PENALTY = float(os.getenv("SUMMARY_MODEL_REPEAT_PENALTY", "1.05"))

# 多语言标点符号列表（从 .env 加载）
all_punctuation_str = os.getenv("all_punctuation_list", "。！？!?；;：:")
# 解析逗号分隔的标点符号列表
ALL_PUNCTUATION_LIST = [p.strip().strip("'").strip('"') for p in all_punctuation_str.split(",") if p.strip()]


def has_punctuation_at_position(text: str, position: int) -> bool:
    """
    检查文本在指定位置是否包含标点符号（高效版本）
    
    Args:
        text: 待检测的文本字符串
        position: 要检查的位置索引
        
    Returns:
        bool: 该位置是否是标点符号
    """
    if not text or not isinstance(text, str) or position < 0 or position >= len(text):
        return False
    
    # 🔥 核心优化：使用 set 的 O(1) 查找
    return text[position] in ALL_PUNCTUATION_LIST


class TextChunkMerger:
    """文本块合并器"""
    
    def __init__(self, pdf_path: str, titles_json_path: str):
        """
        初始化文本块合并器
        
        Args:
            pdf_path: PDF 文件路径
            titles_json_path: 标题 JSON 文件路径
        """
        self.pdf_path = pdf_path
        self.titles_json_path = titles_json_path
        self.pdf_loader = PDFLoader()
        self.pages = []
        self.titles = []
        
    def load_titles(self):
        """加载标题 JSON 文件"""
        print(f"📖 加载标题文件：{self.titles_json_path}")
        with open(self.titles_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.titles = data.get('titles', [])
        print(f"  ✓ 成功加载 {len(self.titles)} 个章节标题")
        return self.titles
    
    def load_pdf(self):
        """加载 PDF 文件"""
        print(f"📄 加载 PDF 文件：{self.pdf_path}")
        self.pages = self.pdf_loader.load_pdf(self.pdf_path)
        print(f"  ✓ 成功加载 {len(self.pages)} 页")
        return self.pages
    
    def merge_pages_by_chapters(self) -> List[Dict[str, Any]]:
        """
        根据章节标题合并页面文本
        
        Returns:
            合并后的章节列表，每个章节包含：
            - chapter_title: 章节标题
            - start_page: 起始页码
            - end_page: 结束页码
            - text: 完整文本
        """
        if not self.titles or not self.pages:
            raise ValueError("请先加载标题和 PDF 文件")
        
        chapters = []
        num_titles = len(self.titles)
        
        print(f"\n🔗 开始根据章节标题合并文本...")
        
        for i, title_info in enumerate(self.titles):
            chapter_title = title_info['title']
            start_page = title_info['page']
            
            # 确定结束页码（下一个章节的上一页）
            if i < num_titles - 1:
                end_page = self.titles[i + 1]['page'] - 1
            else:
                # 最后一个章节，使用 PDF 的总页数
                end_page = len(self.pages)
            
            # 确保页码在有效范围内
            start_page = max(1, min(start_page, len(self.pages)))
            end_page = max(start_page, min(end_page, len(self.pages)))
                
            # 收集该章节的所有文本块
            chapter_texts = []
            for page_idx in range(start_page - 1, end_page):
                page = self.pages[page_idx]
                # 直接拼接该页所有文本块
                page_text = page.get_full_text()
                if page_text.strip():
                    chapter_texts.append(page_text.strip())
            
            # 合并所有章节文本
            full_chapter_text = "\n".join(chapter_texts)
            
            chapters.append({
                'chapter_title': chapter_title,
                'start_page': start_page,
                'end_page': end_page,
                'text': full_chapter_text,
                'char_count': len(full_chapter_text)
            })
        
        print(f"\n✅ 章节合并完成，共 {len(chapters)} 个章节")
        return chapters
    
    def split_into_chunks(self, text: str, chapter_title: str, pdf_filename: str, start_page: int, end_page: int) -> List[Dict[str, Any]]:
        """
        将长文本分割成合适大小的块（支持多语言标点符号）
        
        Args:
            text: 要分割的文本
            chapter_title: 章节标题
            pdf_filename: PDF 文件名
            start_page: 起始页码
            end_page: 结束页码
            
        Returns:
            文本块列表
        """
        chunks = []
        
        # 如果文本长度 <= CHUNK_SIZE，直接返回
        if len(text) <= CHUNK_SIZE:
            chunks.append({
                'text': text,
                'metadata': {
                    'source': pdf_filename,
                    'chapter': chapter_title,
                    'start_page': start_page,
                    'end_page': end_page,
                    'summary': ''
                }
            })
            return chunks
        
        start = 0
        chunk_idx = 0
        
        while start < len(text):
            # 计算当前块的结束位置
            end = start + CHUNK_SIZE
            
            # 如果已经是最后一段，直接取剩余文本
            if end >= len(text):
                chunk_text = text[start:]
                chunks.append({
                    'text': chunk_text,
                    'metadata': {
                        'source': pdf_filename,
                        'chapter': chapter_title,
                        'start_page': start_page,
                        'end_page': end_page,
                        'summary': ''
                    }
                })
                break
            
            # 🔥 修复 Bug：将 best_end 初始化为 -1，确保后续比较逻辑能正确生效
            best_end = -1
            
            # 🌍 修改搜索范围：固定位置区间 [510, 530]
            search_start = start + 510
            search_end = start + 530
            
            # 确保搜索范围不超出文本边界
            if search_start >= len(text):
                search_start = end
            if search_end > len(text):
                search_end = len(text)
            
            # 🌍 多语言支持：遍历文本内容，一旦发现任意字符属于标点集合，立即在该位置进行切分
            for pos in range(search_start, search_end):
                if has_punctuation_at_position(text, pos):
                    actual_pos = pos + 1  # 切分在标点符号之后
                    
                    # ✅ 如果是第一个找到的断点，或者更接近目标位置 (end)
                    if best_end == -1 or abs(actual_pos - end) < abs(best_end - end):
                        best_end = actual_pos
            
            # 如果没有找到合适的标点符号断点，使用原始 end 位置
            if best_end == -1:
                best_end = end
            
            # 提取文本块
            chunk_text = text[start:best_end].strip()
            
            if chunk_text:
                chunks.append({
                    'text': chunk_text,
                    'metadata': {
                        'source': pdf_filename,
                        'chapter': chapter_title,
                        'start_page': start_page,
                        'end_page': end_page,
                        'summary': ''
                    }
                })
                chunk_idx += 1
            
            # 移动起始位置（考虑重叠）
            start = best_end - CHUNK_OVERLAP
            if start <= 0:
                start = best_end
        
        print(f"      → 分割为 {len(chunks)} 个块")
        return chunks
    
    def process(self) -> List[Dict[str, Any]]:
        """
        执行完整的处理流程
        
        Returns:
            所有处理后的文本块列表
        """
        # 1. 加载标题
        self.load_titles()
        
        # 2. 加载 PDF
        self.load_pdf()
        
        # 3. 合并章节
        chapters = self.merge_pages_by_chapters()
        
        # 4. 分割成长度合适的块
        all_chunks = []
        pdf_filename = os.path.basename(self.pdf_path)
        for chapter in chapters:
            chapter_chunks = self.split_into_chunks(
                text=chapter['text'],
                chapter_title=chapter['chapter_title'],
                pdf_filename=pdf_filename,
                start_page=chapter['start_page'],
                end_page=chapter['end_page']
            )
            all_chunks.extend(chapter_chunks)
        
        print(f"\n✅ 处理完成！共生成 {len(all_chunks)} 个文本块")
        return all_chunks
    
    def _build_prompt(self, current_text: str, next_preview: str = "") -> str:
        """
        构建给模型的摘要生成 prompt（精简版）
        
        Args:
            current_text: 当前文本块内容
            next_preview: 下一文本块的预览（可选）
            
        Returns:
            格式化后的 prompt 字符串
        """
        # 清理文本中的异常换行和特殊符号
        clean_text = current_text.replace('\n', ' ').replace('\r', ' ').strip()
        clean_preview = next_preview.replace('\n', ' ').replace('\r', ' ').strip() if next_preview else ""
        
        # 构建带上下文的 prompt
        context_line = f"\n（下文：{clean_preview}）" if clean_preview else ""
        
        return f"""你是专业的摘要助手。请概括以下文学片段：

要求：
1. 字数{SUMMARY_MIN_LENGTH}-{SUMMARY_MAX_LENGTH}字
2. 只概括原文内容，不添加原文没有的信息
3. 不编造故事名称或人物名字
4. 保持客观叙述，不评价分析
5. 确保句子完整，不要在半句话处结束
6. 直接开始概括，不要加标题或额外说明

正文：{clean_text}{context_line}

摘要（直接开始概括）："""
    
    def _post_process_summary(self, summary: str) -> str:
        """
        后处理生成的摘要（仅做最基本的清理）
        
        Args:
            summary: 模型生成的原始摘要
            
        Returns:
            处理后的摘要
        """
        # 直接返回模型原始输出，不做任何修改
        return summary.strip()
    
    def generate_summary(self, chunks: List[Dict[str, Any]], debug_mode: bool = False) -> List[Dict[str, Any]]:
        """
        使用本地模型为每个文本块生成摘要（带重试机制）
        
        Args:
            chunks: 已生成的文本块列表
            debug_mode: 是否为调试模式（仅处理前 3 个块并打印详细日志）
            
        Returns:
            填充了 summary 字段的文本块列表
        """
        import requests
        from time import sleep
        
        # 调试模式只处理前 3 个
        if debug_mode:
            print(f"\n🔍 [调试模式] 仅处理前 3 个文本块...")
            chunks_to_process = chunks[:3]
        else:
            chunks_to_process = chunks
        
        print(f"\n🤖 开始调用本地模型生成摘要...")
        print(f"   共需处理 {len(chunks_to_process)} 个文本块")
        
        ollama_url = "http://localhost:11434/api/generate"
        processed_count = 0
        max_retries = 3  # 🔥 最大重试次数
        
        for i, chunk in enumerate(chunks_to_process):
            success = False
            last_error = None
            
            # 🔥 重试循环
            for attempt in range(1, max_retries + 1):
                try:
                    # 获取当前文本块内容和章节信息
                    current_text = chunk['text']
                    current_chapter = chunk['metadata']['chapter']
                    
                    # 构建上下文：仅拼接下一文本块的前 50 字（仅限同一章节）
                    next_text_preview = ""
                    if i < len(chunks_to_process) - 1:
                        next_chapter = chunks_to_process[i + 1]['metadata']['chapter']
                        # 仅当属于同一章节时才拼接预览文本
                        if current_chapter == next_chapter:
                            next_text = chunks_to_process[i + 1]['text']
                            next_text_preview = next_text[:NEXT_CHUNK_PREVIEW_LENGTH]
                    
                    # 🔍 打印详细的输入信息
                    if attempt == 1 or debug_mode:
                        print(f"\n{'='*60}")
                        print(f"【块 {i+1}】章节：{current_chapter}")
                        print(f"文本长度：{len(current_text)} 字符")
                        if next_text_preview:
                            print(f"下文预览（{NEXT_CHUNK_PREVIEW_LENGTH} 字）：{next_text_preview}")
                    
                    # 🎯 构建优化的 prompt
                    prompt_text = self._build_prompt(current_text, next_text_preview)
                    
                    if attempt > 1 and not debug_mode:
                        print(f"   [重试 {attempt}/{max_retries}] 重新生成摘要...")
                    
                    # 🔍 记录模型调用开始时间
                    from time import time
                    start_time = time()
                    
                    # 调用 Ollama API（优化参数配置）
                    payload = {
                        "model": SUMMARY_MODEL,
                        "prompt": prompt_text,
                        "stream": False,
                        "options": {
                            "temperature": SUMMARY_MODEL_TEMPERATURE,
                            "top_p": SUMMARY_MODEL_TOP_P,
                            "top_k": SUMMARY_MODEL_TOP_K,
                            "num_predict": SUMMARY_MODEL_MAX_TOKENS,
                            "repeat_penalty": SUMMARY_MODEL_REPEAT_PENALTY,
                        }
                    }
                    
                    response = requests.post(ollama_url, json=payload, timeout=30)
                    response.raise_for_status()
                    
                    result_json = response.json()
                    model_output = result_json.get("response", "")
                    
                    # 🔍 记录模型调用耗时
                    end_time = time()
                    elapsed_time = end_time - start_time
                    
                    # 🔍 打印模型原始输出
                    if debug_mode or (attempt > 1 and not debug_mode):
                        print(f"\n📝 模型原始输出:")
                        print(f"{model_output}")
                        print(f"输出长度：{len(model_output)} 字符")
                        print(f"⏱️  模型调用耗时：{elapsed_time:.2f} 秒")
                    
                    # 后处理摘要
                    summary = self._post_process_summary(model_output)
                    
                    # 🔥 检查摘要是否有效
                    if not summary or len(summary.strip()) == 0:
                        raise ValueError("模型返回空摘要")
                    
                    # 写入 metadata
                    chunk['metadata']['summary'] = summary
                    processed_count += 1
                    
                    if attempt == 1:
                        print(f"✅ 最终摘要（{len(summary)} 字）：{summary[:100]}...")
                    else:
                        print(f"✅ [重试成功] 最终摘要（{len(summary)} 字）：{summary[:100]}...")
                    
                    success = True
                    break  # 🔥 成功后跳出重试循环
                    
                except Exception as e:
                    last_error = e
                    error_msg = str(e)
                    
                    if attempt == 1 or debug_mode:
                        print(f"   ⚠️ 生成第 {i + 1} 个块的摘要失败：{error_msg}")
                    
                    # 🔥 如果不是最后一次尝试，等待一段时间后重试
                    if attempt < max_retries:
                        wait_time = 1.0 * attempt  # 递增等待时间
                        print(f"   ⏳ 等待 {wait_time:.1f}秒后重试...")
                        sleep(wait_time)
                    else:
                        # 🔥 所有重试都失败后，设置默认值
                        print(f"   ❌ 已达到最大重试次数 ({max_retries})，使用默认摘要")
                        chunk['metadata']['summary'] = "摘要生成失败"
            
            # 非调试模式显示进度
            if not debug_mode:
                if (i + 1) % 20 == 0 or (i + 1) == len(chunks_to_process):
                    print(f"   进度：{i + 1}/{len(chunks_to_process)} ({(i + 1) / len(chunks_to_process) * 100:.1f}%)")
                sleep(0.2)
        
        if not debug_mode:
            print(f"\n✅ 摘要生成完成！共处理 {processed_count}/{len(chunks_to_process)} 个文本块")
        
        return chunks_to_process if debug_mode else chunks


def main():
    """主函数：支持调试模式和全量处理"""
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='智能文本块拼接脚本')
    parser.add_argument('--debug', action='store_true', help='调试模式：仅测试每个 PDF 的前 3 个文本块')
    parser.add_argument('--book', type=str, help='指定要处理的书籍名称（不含.pdf 扩展名）')
    parser.add_argument('--force', action='store_true', help='强制覆盖已存在的输出文件（危险操作）')
    args = parser.parse_args()
    
    debug_mode = args.debug
    force_mode = args.force
    
    print("=" * 60)
    if debug_mode:
        print("🔍 [调试模式] 每个 PDF 仅测试前 3 个文本块")
    else:
        print("📚 [全量模式] 处理所有 PDF 文件")
    print("=" * 60)
    
    # 设置基础路径
    source_dir = os.path.join("src", "data", "source")
    titles_dir = os.path.join("src", "data", "pages_title")
    output_dir = "src/data/chunks"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 收集所有 PDF 文件
    pdf_files = []
    
    # 如果指定了书名，只处理该书籍
    if args.book:
        book_name = args.book
        pdf_path = os.path.join(source_dir, "Chinese", f"{book_name}.pdf")
        if os.path.exists(pdf_path):
            pdf_files.append((book_name, pdf_path))
        else:
            print(f"❌ 指定的 PDF 文件不存在：{pdf_path}")
            return
    else:
        # 遍历所有子目录（Chinese, English, Japanese）
        for lang_folder in os.listdir(source_dir):
            lang_path = os.path.join(source_dir, lang_folder)
            if not os.path.isdir(lang_path):
                continue
            
            # 查找该语言目录下的所有 PDF 文件
            for filename in os.listdir(lang_path):
                if filename.endswith('.pdf'):
                    book_name = filename[:-4]  # 移除 .pdf 扩展名
                    pdf_path = os.path.join(lang_path, filename)
                    pdf_files.append((book_name, pdf_path))
    
    if not pdf_files:
        print("❌ 未找到任何 PDF 文件")
        return
    
    print(f"\n📚 找到 {len(pdf_files)} 个 PDF 文件:")
    for i, (book_name, pdf_path) in enumerate(pdf_files, 1):
        print(f"  {i}. {book_name}")
    
    # 处理每个 PDF 文件
    all_processed_chunks = []
    skipped_files = []
    
    for book_name, pdf_path in pdf_files:
        print("\n" + "=" * 60)
        print(f"📖 开始处理：《{book_name}》")
        print("=" * 60)
        
        # 查找对应的标题文件
        titles_json_path = None
        # 尝试不同的可能路径
        possible_titles_paths = [
            os.path.join(titles_dir, f"{book_name}_titles.json"),
            os.path.join(titles_dir, f"{book_name.replace(':', '_').replace('/', '_')}_titles.json"),
        ]
        
        for path in possible_titles_paths:
            if os.path.exists(path):
                titles_json_path = path
                break
        
        if not titles_json_path:
            print(f"⚠️  警告：未找到标题文件，跳过《{book_name}》")
            continue
        
        print(f"✓ 标题文件：{os.path.basename(titles_json_path)}")
        
        # 验证文件存在
        if not os.path.exists(pdf_path):
            print(f"❌ PDF 文件不存在：{pdf_path}")
            continue
        
        # 🔍 检查是否已经生成过该文件的 chunks
        output_filename = f"{book_name}_chunks.json"
        # 处理文件名中的特殊字符
        output_filename = output_filename.replace(':', '_').replace('/', '_')
        output_path = os.path.join(output_dir, output_filename)
        
        if os.path.exists(output_path) and not force_mode:
            print(f"⚠️  检测到已存在的文件：{output_path}")
            print(f"   跳过《{book_name}》的处理")
            skipped_files.append(book_name)
            continue
        elif force_mode and os.path.exists(output_path):
            print(f"⚡ [强制模式] 将覆盖已有文件：{output_path}")
        
        # 创建合并器并处理
        merger = TextChunkMerger(pdf_path, titles_json_path)
        chunks = merger.process()
        
        # 生成摘要
        if debug_mode:
            print(f"\n🔍 [调试模式] 测试前 3 个文本块")
            debug_chunks = merger.generate_summary(chunks, debug_mode=True)
            
            # 显示调试结果
            print("\n" + "="*60)
            print(f"【{book_name}】调试结果分析:")
            print("="*60)
            for i, chunk in enumerate(debug_chunks, 1):
                print(f"\n【块 {i}】")
                print(f"  章节：{chunk['metadata']['chapter']}")
                print(f"  页码：{chunk['metadata']['start_page']}-{chunk['metadata']['end_page']}")
                print(f"  文本长度：{len(chunk['text'])} 字符")
                print(f"  摘要长度：{len(chunk['metadata']['summary'])} 字符")
                print(f"  摘要内容：{chunk['metadata']['summary']}")
            
            all_processed_chunks.extend(debug_chunks)
        else:
            # 全量处理
            chunks = merger.generate_summary(chunks, debug_mode=False)
            all_processed_chunks.extend(chunks)
        
        # 保存当前 PDF 的结果（转换为最终格式）
        final_output = []
        for i, chunk in enumerate(chunks):
            # 生成语义化唯一 ID
            pdf_file = chunk["metadata"]["source"].replace(".pdf", "")
            chapter = chunk["metadata"]["chapter"]
            start_page = chunk["metadata"]["start_page"]
            end_page = chunk["metadata"]["end_page"]
            
            # 清理特殊字符
            safe_pdf = "".join(c if c.isalnum() or c in "_-" else "_" for c in pdf_file)
            safe_chapter = "".join(c if c.isalnum() or c in "_-" else "_" for c in chapter)
            
            chunk_id = f"{safe_pdf}_{safe_chapter}_{start_page}_{end_page}_{i}"
            
            # 构建最终格式
            final_chunk = {
                "id": chunk_id,
                "embedding": [],  # 稍后由嵌入模型生成
                "document": chunk["metadata"]["summary"],  # 使用摘要作为嵌入文本
                "metadata": {
                    "source": chunk["metadata"]["source"],
                    "chapter": chunk["metadata"]["chapter"],
                    "start_page": start_page,
                    "end_page": end_page,
                    "full_text": chunk["text"]  # 完整原文
                }
            }
            final_output.append(final_chunk)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 结果已保存到：{output_path}")
        print(f"   总计：{len(final_output)} 个文本块")
        print(f"   数据格式：{{id, embedding, document, metadata}}")
    
    # 显示跳过文件的统计
    if skipped_files:
        print("\n" + "=" * 60)
        print(f"⏭️  跳过的文件 ({len(skipped_files)} 个):")
        print("=" * 60)
        for i, book_name in enumerate(skipped_files, 1):
            output_filename = f"{book_name}_chunks.json".replace(':', '_').replace('/', '_')
            print(f"  {i}. 《{book_name}》 -> src/data/chunks/{output_filename}")
        print(f"\n   如需重新处理这些文件，请先删除对应的 JSON 文件")
    
    # 显示总体统计
    print("\n" + "=" * 60)
    print("📊 处理完成统计:")
    print("=" * 60)
    total_pdf_count = len(pdf_files)
    processed_count = total_pdf_count - len(skipped_files)
    print(f"   总 PDF 数量：{total_pdf_count}")
    print(f"   本次处理：{processed_count} 个")
    print(f"   跳过：{len(skipped_files)} 个")
    print(f"   生成文本块总数：{len(all_processed_chunks)}")
    print(f"   输出目录：{output_dir}")
    
    if debug_mode:
        print("\n⚠️  注意：当前为调试模式，每个 PDF 仅处理了前 3 个文本块")
        print("   如需全量处理，请运行：python src/concatenate_text_blocks.py")
    
    if force_mode:
        print("\n⚡ 提示：当前为强制覆盖模式")
    
    # 如果有跳过的文件，给出提示
    if skipped_files and not debug_mode and not force_mode:
        print("\n" + "=" * 60)
        print("💡 提示:")
        print("=" * 60)
        print(f"   有 {len(skipped_files)} 个文件已存在结果，已自动跳过")
        print("   如需重新处理，可以:")
        print("   1. 手动删除对应的 JSON 文件")
        print(f"   2. 或运行命令强制覆盖（危险操作）:")
        print(f"      python src/concatenate_text_blocks.py --force")
        print("\n   或者指定单本书籍处理:")
        print(f"      python src/concatenate_text_blocks.py --book \"书名\"")


if __name__ == "__main__":
    main()