from typing import List, Dict, Any
from dotenv import load_dotenv
import os
import ast
import json
from pathlib import Path
import time
import hashlib
import traceback

# 加载环境变量
load_dotenv()

class TitleIdentifier:
    def __init__(self):
        # 加载标点符号列表
        try:
            env_punctuation = os.getenv("all_punctuation_list", "[]")
            print(f"🔍 调试：环境变量all_punctuation_list原始值长度: {len(env_punctuation)}")
            self.all_punctuation_list = ast.literal_eval(env_punctuation)
            print(f"✅ 调试：标点符号列表成功加载，数量: {len(self.all_punctuation_list)}")
        except (SyntaxError, ValueError) as e:
            print(f"❌ 调试：标点符号列表加载失败，使用默认列表: {e}")
            self.all_punctuation_list = ['.', '。', '!', '！', '?', '？', '…', '……', '——', '――', ',', '，', '、', ';', '；', ':', '：']
        
        # 预加载关键词列表，避免每次调用时重复解析环境变量
        try:
            self.chinese_keywords = ast.literal_eval(os.getenv("chinese_title_keywords", "[]"))
            self.japanese_keywords = ast.literal_eval(os.getenv("japanese_title_keywords", "[]"))
            self.english_keywords = ast.literal_eval(os.getenv("english_title_keywords", "[]"))
        except (SyntaxError, ValueError) as e:
            print(f"❌ 关键词列表加载失败: {e}")
            self.chinese_keywords = []
            self.japanese_keywords = []
            self.english_keywords = []

        # 加载长度限制配置
        self.chinese_max_chars = int(os.getenv("chinese_title_max_chars", 7))
        self.japanese_max_chars = int(os.getenv("japanese_title_max_chars", 8))
        self.english_max_words = int(os.getenv("english_title_max_words", 7))
        
    def _get_keywords(self, language: str) -> List[str]:
        """根据语言获取关键词列表"""
        if language == "Chinese":
            return self.chinese_keywords
        elif language == "Japanese":
            return self.japanese_keywords
        elif language == "English":
            return self.english_keywords
        else:
            return []
    def check_title_length(self, text: str, language: str) -> bool:
        """检查标题长度是否符合语言限制"""
        has_chinese = any('\u4e00' <= c <= '\u9fff' for c in text)
        has_japanese = any('\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff' for c in text)
        
        if has_chinese:
            return len(text) <= self.chinese_max_chars
        elif has_japanese:
            return len(text) <= self.japanese_max_chars
        elif language == "English":
            words = text.split()
            return len(words) <= self.english_max_words
        else:
            return len(text) <= self.chinese_max_chars
    
    def check_text_centered(self, block, page_width: float, tolerance: float = 0.1) -> bool:
        """
        检查文本块是否水平居中
        
        Args:
            block: 文本块对象，假设有x0, x1属性表示水平边界
            page_width: 页面总宽度
            tolerance: 居中容忍度（默认为0.1，即页面宽度的10%）
            
        Returns:
            bool: 是否水平居中
        """
        # 如果文本块没有位置信息，返回False
        if not hasattr(block, 'x0') or not hasattr(block, 'x1'):
            return False
        
        # 计算文本块的中心位置
        block_center = (block.x0 + block.x1) / 2
        
        # 计算页面的居中区域（页面宽度的40%-60%为居中区域）
        center_start = page_width * (0.5 - tolerance)
        center_end = page_width * (0.5 + tolerance)
        
        return center_start <= block_center <= center_end
    
    def identify_title(self, pages: List[Any]) -> List[Dict[str, Any]]:
        """
        识别PDF中的标题
        
        Args:
            pages: PDFLoader.load_pdf() 返回的页面列表
            
        Returns:
            包含标题名称和起始页的列表
        """
        if not pages:
            return []
        
        # 获取语言类型
        language = pages[0].language
        print(f"识别语言: {language}")
        
        # 根据语言获取关键词（已在构造器预加载）
        keywords = self._get_keywords(language)
        print(f"关键词列表: {keywords}")
        
        titles = []
        
        # 第一阶段：遍历前10页，查找包含关键词且不含标点符号的文本块
        first_10_pages = pages[:10]
        print(f"第一阶段：遍历前 {len(first_10_pages)} 页")
        
        for page in first_10_pages:
            page_num = page.page_number
            text_blocks = page.text_blocks  # 所有文本块
            
            # 估算页面宽度（如果无法获取，使用默认值）
            page_width = 612.0  # 默认页面宽度
            try:
                if hasattr(page, 'width') and page.width > 0:
                    page_width = page.width
                elif hasattr(page, 'mediabox') and hasattr(page.mediabox, 'width'):
                    page_width = page.mediabox.width
                elif text_blocks:
                    max_x = max(block.x1 for block in text_blocks if hasattr(block, 'x1'))
                    if max_x > 0:
                        page_width = max_x
            except:
                pass  # 使用默认值
            
            for block in text_blocks:
                text = block.text.strip()
                
                if not text:
                    continue
                
                # 【首要且不可更改】检查是否不含标点符号 - 第一个强制性条件
                has_punctuation = any(punc in text for punc in self.all_punctuation_list)
                if has_punctuation:
                    continue  # 跳过含有标点符号的文本块
                
                # 检查水平居中
                is_centered = self.check_text_centered(block, page_width)
                
                # 检查是否包含关键词
                contains_keyword = any(keyword in text for keyword in keywords)
                
                # 检查长度限制
                length_ok = self.check_title_length(text, language)
                
                # 标题候选应同时满足：无标点符号 AND (包含关键词 OR 水平居中) AND 长度合适
                if not has_punctuation and (contains_keyword or is_centered) and length_ok:
                    # 符合条件，记录标题
                    titles.append({
                        "title": text,
                        "start_page": page_num
                    })
                    print(f"找到标题：'{text}'，页码：{page_num}")
                    break  # 每一页只能有一个标题
        
        print(f"第一阶段完成，共找到 {len(titles)} 个标题")
        
        # 如果第一阶段找到了标题，继续遍历剩余页面查找更多标题
        if titles:
            print("第一阶段找到标题，继续遍历剩余页面...")
            
            # 继续遍历第11页开始的剩余页面
            remaining_pages = pages[10:]
            for page in remaining_pages:
                page_num = page.page_number
                text_blocks = page.text_blocks  # 所有文本块
                
                # 估算页面宽度
                page_width = 612.0  # 默认页面宽度
                try:
                    if hasattr(page, 'width') and page.width > 0:
                        page_width = page.width
                    elif hasattr(page, 'mediabox') and hasattr(page.mediabox, 'width'):
                        page_width = page.mediabox.width
                    elif text_blocks:
                        max_x = max(block.x1 for block in text_blocks if hasattr(block, 'x1'))
                        if max_x > 0:
                            page_width = max_x
                except:
                    pass  # 使用默认值
                
                for block in text_blocks:
                    text = block.text.strip()
                    
                    if not text:
                        continue
                    
                    # 【首要且不可更改】检查是否不含标点符号
                    has_punctuation = any(punc in text for punc in self.all_punctuation_list)
                    if has_punctuation:
                        continue  # 跳过含有标点符号的文本块
                    
                    # 检查水平居中
                    is_centered = self.check_text_centered(block, page_width)
                    
                    # 检查是否包含关键词
                    contains_keyword = any(keyword in text for keyword in keywords)
                    
                    # 检查长度限制
                    length_ok = self.check_title_length(text, language)
                    
                    if not has_punctuation and (contains_keyword or is_centered) and length_ok:
                        # 符合条件，记录标题
                        titles.append({
                            "title": text,
                            "start_page": page_num
                        })
                        print(f"继续遍历找到标题：'{text}'，页码：{page_num}")
                        break  # 每一页只能有一个标题
            
            print(f"完整遍历完成，共找到 {len(titles)} 个标题")
            return self._calculate_end_pages(titles, pages)
        
        # 第二阶段：如果前10页没有找到标题，切换检索模式
        print("第二阶段：使用预备标题模式")
        
        # 收集所有文本块中不含标点符号的文本
        candidate_titles = []
        for page in pages:
            page_num = page.page_number
            text_blocks = page.text_blocks  # 所有文本块
            
            # 估算页面宽度
            page_width = 612.0  # 默认页面宽度
            try:
                if hasattr(page, 'width') and page.width > 0:
                    page_width = page.width
                elif hasattr(page, 'mediabox') and hasattr(page.mediabox, 'width'):
                    page_width = page.mediabox.width
                elif text_blocks:
                    max_x = max(block.x1 for block in text_blocks if hasattr(block, 'x1'))
                    if max_x > 0:
                        page_width = max_x
            except:
                pass  # 使用默认值
            
            # 加载关键词
            keywords = self._get_keywords(language)
            
            for block in text_blocks:
                text = block.text.strip()
                
                if not text:
                    continue
                
                # 【首要且不可更改】检查是否不含标点符号
                has_punctuation = any(punc in text for punc in self.all_punctuation_list)
                if has_punctuation:
                    print(f"🔍 调试：第二阶段跳过含标点文本 '{text}' (页码: {page_num})")
                    continue  # 跳过含有标点符号的文本块
                
                # 检查水平居中
                is_centered = self.check_text_centered(block, page_width)
                
                # 检查长度限制
                length_ok = self.check_title_length(text, language)
                
                # 检查是否包含关键词
                contains_keyword = any(keyword in text for keyword in keywords)
                
                # 【关键修复】标题候选应同时满足：无标点符号 AND (包含关键词 OR 水平居中) AND 长度合适
                if not has_punctuation and (contains_keyword or is_centered) and length_ok:
                    # 添加到预备标题列表
                    candidate_titles.append({
                        "title": text,
                        "start_page": page_num
                    })
                    print(f"🔍 调试：第二阶段通过候选文本 '{text}' (页码: {page_num}, 关键词: {contains_keyword}, 居中: {is_centered})")
                else:
                    # 调试：记录失败原因
                    reasons = []
                    if has_punctuation:
                        reasons.append("含标点")
                    if not (contains_keyword or is_centered):
                        reasons.append("无关键词且不居中")
                    if not length_ok:
                        reasons.append("长度不合规")
                    print(f"🔍 调试：第二阶段拒绝文本 '{text}' (页码: {page_num}, 原因: {', '.join(reasons)})")
        
        if not candidate_titles:
            print("未找到任何标题")
            return []
        
        # 统计预备标题的出现次数
        title_counts = {}
        for title_info in candidate_titles:
            title = title_info["title"]
            if title in title_counts:
                title_counts[title] += 1
            else:
                title_counts[title] = 1
        
        # 找出重复的标题（出现次数大于1）
        duplicate_titles = [title for title, count in title_counts.items() if count > 1]
        
        if duplicate_titles:
            # 对于每个重复的标题，只保留第一个出现的
            final_titles = []
            seen_titles = set()
            for title_info in candidate_titles:
                if title_info["title"] in duplicate_titles and title_info["title"] not in seen_titles:
                    final_titles.append(title_info)
                    seen_titles.add(title_info["title"])
            
            result_titles = self._calculate_end_pages(final_titles, pages)
        else:
            # 如果没有重复的标题，使用所有预备标题
            result_titles = self._calculate_end_pages(candidate_titles, pages)
        
        # 【最终验证】强制进行标点符号复查
        print(f"\n🛡️ 最终验证：对 {len(result_titles)} 个标题进行标点符号检查...")
        validated_titles = []
        
        for title_info in result_titles:
            title_text = title_info["title"]
            has_punctuation = any(punc in title_text for punc in self.all_punctuation_list)
            
            if has_punctuation:
                matched_puncs = [punc for punc in self.all_punctuation_list if punc in title_text]
                print(f"⚠️  移除含标点标题: '{title_text}' (页码: {title_info['start_page']}-{title_info['end_page']}, 匹配标点: {matched_puncs})")
                continue  # 跳过含标点的标题
            
            validated_titles.append(title_info)
            print(f"✅ 验证通过: '{title_text}' (页码: {title_info['start_page']}-{title_info['end_page']})")
        
        print(f"🛡️ 最终验证完成：{len(result_titles)} -> {len(validated_titles)} 个标题")
        return validated_titles
    
    def _calculate_end_pages(self, titles: List[Dict[str, Any]], pages: List[Any]) -> List[Dict[str, Any]]:
        """
        计算标题的结束页码
        
        Args:
            titles: 包含标题和起始页的列表
            pages: PDF页面列表
            
        Returns:
            包含标题、起始页和结束页的列表
        """
        if not titles:
            return titles
            
        # 按起始页排序
        sorted_titles = sorted(titles, key=lambda x: x["start_page"])
        
        for i, title_info in enumerate(sorted_titles):
            start_page = title_info["start_page"]
            
            if i < len(sorted_titles) - 1:
                # 如果不是最后一个标题，结束页是下一个标题起始页的前一页
                next_start_page = sorted_titles[i + 1]["start_page"]
                end_page = next_start_page - 1
            else:
                # 如果是最后一个标题，结束页是文档的最后一页
                end_page = max(page.page_number for page in pages)
            
            title_info["end_page"] = end_page
        
        return sorted_titles
    
    def _build_page_title_map(self, title_blocks: List[Dict[str, Any]], total_pages: int) -> List[str]:
        """
        优化版本：构建页面标题映射表，O(N+M)复杂度
        
        Args:
            title_blocks: 标题区块列表，每个包含start_page和end_page
            total_pages: 总页数
            
        Returns:
            页面标题映射列表，索引为页码-1
        """
        # 初始化页面标题映射表，索引0对应第1页
        page_title_map = [""] * total_pages
        
        # 为每个标题区块分配页面
        for block in title_blocks:
            start_page = block["start_page"]
            end_page = block["end_page"]
            title = block["title"]
            
            # 确保页码在有效范围内
            for page_num in range(start_page, end_page + 1):
                if 1 <= page_num <= total_pages:
                    page_title_map[page_num - 1] = title
        
        return page_title_map


def get_file_hash(file_path: str) -> str:
    """计算文件的MD5哈希值"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()
def process_pdf_to_page_titles(pdf_path: str, title_identifier: TitleIdentifier, pdf_loader) -> dict:
    """处理单个PDF，为每一页确定章节标题"""
    try:
        # 加载PDF页面
        pages = pdf_loader.load_pdf(pdf_path)
        if not pages:
            return None
        
        # 识别标题区块
        title_blocks = title_identifier.identify_title(pages)
        print(title_blocks)
        
        # 构建页面标题映射（若无标题则全部为空）
        total_pages = max(page.page_number for page in pages)
        page_title_map = title_identifier._build_page_title_map(title_blocks, total_pages) if title_blocks else [""] * total_pages
        
        # 【核心修改】生成文本块列表（document_chunks）
        document_chunks = []
        chunk_counter = 0

        # 本地分块函数，保持简单稳定
        def local_chunk_text(full_text: str, chunk_size: int = 512, overlap: int = 50):
            if not full_text:
                return []
            chunks = []
            start = 0
            text_len = len(full_text)
            while start < text_len:
                end = start + chunk_size
                if end < text_len:
                    # 尝试在句号或换行处优先断句
                    last_period = full_text.rfind('。', start, end)
                    last_period_en = full_text.rfind('.', start, end)
                    last_break = max(last_period, last_period_en)
                    if last_break > start + chunk_size // 2:
                        end = last_break + 1
                chunk = full_text[start:end].strip()
                if chunk:
                    chunks.append(chunk)
                start = end - overlap if end < text_len else end
            return chunks

        for page in pages:
            page_num = page.page_number
            section_title = page_title_map[page_num - 1] if page_num <= len(page_title_map) else ""
            
            # 获取页面完整文本，然后分块
            full_text = page.get_full_text()
            if not full_text or not full_text.strip():
                continue

            text_chunks = local_chunk_text(full_text, chunk_size=512, overlap=50)

            for i, chunk in enumerate(text_chunks):
                chunk_id = f"{Path(pdf_path).stem}_p{page_num:03d}_c{i:03d}"
                chunk_counter += 1

                # 获取文本块坐标（如果可能）
                coordinates = None
                try:
                    if hasattr(page, 'text_blocks') and page.text_blocks:
                        block = page.text_blocks[0]
                        coordinates = {
                            "x0": getattr(block, "x0", None),
                        "y0": getattr(block, "y0", None),
                        "x1": getattr(block, "x1", None),
                        "y1": getattr(block, "y1", None)
                    }
                except Exception:
                    coordinates = None

                document_chunks.append({
                    "id": chunk_id,
                    "text": chunk,
                    "page_number": page_num,
                    "section_title": section_title,
                    "chunk_index": i,
                    "total_chunks_in_page": len(text_chunks),
                    "coordinates": coordinates,
                    "text_hash": hashlib.md5(chunk.encode()).hexdigest()[:16]
                })
        
        # 提取语言信息
        language = pages[0].language if pages else "Unknown"
        
        # 构建结果结构，符合用户指定格式（不包含 _cache）
        result = {
            "parent_document": {
                "filename": Path(pdf_path).name,
                "file_path": pdf_path,
                "total_pages": len(pages),
                "language": language,
                "total_chunks": chunk_counter,
                "processing_date": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "document_chunks": document_chunks
        }
        
        return result
        
    except Exception as e:
        # 打印完整 traceback 以便定位问题来源（例如 "'str' object is not callable"）
        tb = traceback.format_exc()
        print(f"处理PDF失败 {pdf_path}: {tb}")
        return None

def main():
    """主函数"""
    print("📚 PDF章节标题识别工具")
    print("=" * 60)
    
    # 定义路径
    source_dir = Path("src/data/source")
    output_dir = Path("src/data/pages_title")
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化组件
    title_identifier = TitleIdentifier()
    from src.data.pdf_loader import PDFLoader
    pdf_loader = PDFLoader()
    
    # 查找所有PDF文件
    pdf_files = list(source_dir.rglob("*.pdf"))
    
    if not pdf_files:
        print(f"❌ 在 {source_dir} 中未找到PDF文件")
        return
    
    print(f"📁 找到 {len(pdf_files)} 个PDF文件")
    print(f"📂 输出目录: {output_dir}")
    print()
    
    processed_count = 0
    skipped_count = 0
    failed_count = 0
    
    for pdf_path in pdf_files:
        try:
            print(f"🔄 正在处理: {pdf_path.relative_to(source_dir)}")
            
            # 生成对应的JSON缓存文件路径
            json_filename = f"{pdf_path.stem}_titles.json"
            json_path = output_dir / json_filename
            
            # 简化缓存逻辑：如果 JSON 已存在则跳过（避免复杂的哈希比较）
            if json_path.exists():
                print(f"  ✅ 缓存存在，跳过处理")
                skipped_count += 1
                continue
            
            # 处理PDF
            result = process_pdf_to_page_titles(str(pdf_path), title_identifier, pdf_loader)
            
            if result is None:
                print(f"  ❌ 处理失败")
                failed_count += 1
                continue
            
            # 保存结果
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            # 统计信息：兼容旧/新两种 JSON 结构
            if "document_chunks" in result and isinstance(result.get("document_chunks"), list):
                unique_titles = len(set(chunk.get("section_title", "") for chunk in result["document_chunks"] if chunk.get("section_title")))
            elif "page_sections" in result and isinstance(result.get("page_sections"), list):
                unique_titles = len(set(section.get("section_title", "") for section in result["page_sections"]))
                unique_titles = len(set(chunk.get("section_title", "") for chunk in result["document_chunks"] if chunk.get("section_title")))
            else:
                unique_titles = 0

            total_pages = result["parent_document"].get("total_pages", 0)
            
            print(f"  ✅ 处理完成")
            print(f"     📄 总页数: {total_pages}")
            print(f"     📑 章节数: {unique_titles}")
            print(f"     💾 缓存文件: {json_filename}")
            
            processed_count += 1
            
        except Exception as e:
            print(f"  ❌ 处理失败: {e}")
            failed_count += 1
        
        print()  # 空行分隔
    
    # 打印汇总信息
    print("📊 处理完成汇总:")
    print(f"  ✅ 成功处理: {processed_count}")
    print(f"  ⏭️  跳过(缓存): {skipped_count}")
    print(f"  ❌ 处理失败: {failed_count}")
    print(f"  📁 总计: {len(pdf_files)}")
    print()
    
    if processed_count > 0:
        print("💡 生成的JSON文件可直接用于创建DocumentChunk，结构如下:")
        print("  - parent_document: PDF基本信息 (包含 processing_date 和 total_chunks)")
        print("  - document_chunks: 每个文本块的详细信息 (id, text, page_number, section_title, text_hash 等)")

if __name__ == "__main__":
    import multiprocessing
    import os

    # Windows 支持：在 spawn 模式下安全启动子进程
    multiprocessing.freeze_support()
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        # 如果已经设置过启动方法，跳过
        pass

    # 降低并行库线程数，减少与多进程的冲突
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    main()