from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import os
import fitz  # PyMuPDF
import multiprocessing
import json
import hashlib
import base64
import requests
from PyPDF2 import PdfReader


# 加载环境变量
load_dotenv()

class TextBlock:
    """文本块类，包装文本内容和坐标信息"""
    def __init__(self, text: str, x0: float, y0: float, x1: float, y1: float, page_number: int):
        self.text = text  # 文本内容
        self.x0 = x0      # 左上角x坐标
        self.y0 = y0      # 左上角y坐标
        self.x1 = x1      # 右下角x坐标
        self.y1 = y1      # 右下角y坐标
        self.page_number = page_number  # 页码
    
    def to_dict(self) -> Dict[str, Any]:
        """将文本块转换为字典格式"""
        return {
            "text": self.text,
            "x0": self.x0,
            "y0": self.y0,
            "x1": self.x1,
            "y1": self.y1,
            "page_number": self.page_number
        }
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"TextBlock(page={self.page_number}, text='{self.text[:50]}...', bbox=({self.x0:.2f}, {self.y0:.2f}, {self.x1:.2f}, {self.y1:.2f}))"

class Page:
    """页面类，包装一页的所有文本块"""
    def __init__(self, page_number: int, text_blocks: List[TextBlock], language: str = "", chapters: List[Dict[str, Any]] = None):
        self.page_number = page_number  # 页码
        self.text_blocks = text_blocks  # 文本块列表
        self.language = language  # 页面语言
        self.chapters = chapters if chapters is not None else []  # 章节信息（用于日语PDF）
    
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将页面转换为字典格式
        """
        return {
            "page_number": self.page_number,
            "text_blocks": [block.to_dict() for block in self.text_blocks],
            "language": self.language,
            "chapters": self.chapters  # 包含章节信息
        }
    
    def get_full_text(self) -> str:
        """获取页面的完整文本，智能合并文本块"""
        # 如果有章节信息，优先使用章节内容
        if self.chapters:
            chapter_texts = []
            for chapter in self.chapters:
                if "content" in chapter:
                    chapter_texts.append(chapter["content"])
                elif "title" in chapter:
                    chapter_texts.append(chapter["title"])
            if chapter_texts:
                return "\n".join(chapter_texts)
        
        if not self.text_blocks:
            return ""
        
        # 按垂直位置排序文本块
        sorted_blocks = sorted(self.text_blocks, key=lambda b: b.y0)
        
        # 合并文本块，增加智能段落识别
        merged_lines = []
        current_line = ""
        
        for block in sorted_blocks:
            text = block.text.strip()
            if not text:
                continue
            
            # 如果当前行不为空，添加适当的空格或换行
            if current_line:
                # 检查是否需要换行（基于垂直间距）
                vertical_space = block.y0 - (sorted_blocks[sorted_blocks.index(block)-1].y1 if sorted_blocks.index(block) > 0 else 0)
                
                if vertical_space > 10:  # 如果垂直间距较大，认为是新的段落
                    merged_lines.append(current_line)
                    current_line = text
                else:
                    # 如果是同一段落，添加空格连接
                    current_line += " " + text
            else:
                current_line = text
        
        # 添加最后一行
        if current_line:
            merged_lines.append(current_line)
        
        return "\n".join(merged_lines)
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"Page(page={self.page_number}, blocks={len(self.text_blocks)})"


class _PDFMinerBackend:
    """后备解析器：使用 pdfminer.six 从单页中提取带坐标的文本块"""
    @staticmethod
    def extract_blocks_from_page(pdf_path: str, page_num: int, language_hint: str = "") -> List['TextBlock']:
        try:
            from pdfminer.high_level import extract_pages
            from pdfminer.layout import LTTextContainer, LAParams
        except Exception as e:
            # pdfminer 未安装或导入失败
            print(f"PDFMiner 导入失败: {e}")
            return []

        blocks: List[TextBlock] = []
        try:
            laparams = LAParams()
            pages = list(extract_pages(pdf_path, page_numbers=[page_num], laparams=laparams))
            if not pages:
                return blocks
            page_layout = pages[0]
            # page_layout.bbox -> (x0, y0, x1, y1)，以左下角为原点
            page_height = page_layout.bbox[3]

            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    x0, y0_orig, x1, y1_orig = element.bbox
                    # 坐标系转换：pdfminer 左下角 -> PyMuPDF 左上角
                    y0 = page_height - y1_orig
                    y1 = page_height - y0_orig
                    text = element.get_text().replace('\x00', '').strip()
                    if text:
                        blocks.append(TextBlock(
                            text=text,
                            x0=float(x0),
                            y0=float(y0),
                            x1=float(x1),
                            y1=float(y1),
                            page_number=page_num + 1
                        ))
        except Exception as e:
            print(f"PDFMiner解析页面{page_num+1}失败: {e}")
        # 验证提取结果中是否包含 CJK/日文假名字符，若没有则认为 PDFMiner 解析可能产生乱码（如编码缺失），返回空以便上层继续其他后备策略
        try:
            import re
            cjk_pattern = re.compile(r'[\u4e00-\u9fff\u3040-\u30ff\u31f0-\u31ff]')
            has_cjk = any(cjk_pattern.search(b.text) for b in blocks)
            if not has_cjk:
                # 可能为编码导致的乱码，丢弃结果
                return []
        except Exception:
            pass
        return blocks

class PDFLoader:
    """PDF加载器类，用于解析PDF文件并返回文本块"""
    
    def load_pdf(self, pdf_path: str, max_pages: Optional[int] = None) -> List[Page]:
        """
        加载PDF文件并解析每一页的文本块
        
        Args:
            pdf_path: PDF文件的路径
            max_pages: 最大处理页数，默认处理所有页
            
        Returns:
            页面列表，每个页面包含多个文本块
        """
        # 从路径中提取语言信息
        # 路径格式如: src\data\source\Chinese\安徒生童话.pdf
        # 标准化路径分隔符
        normalized_path = pdf_path.replace('/', os.sep).replace('\\', os.sep)
        path_parts = normalized_path.split(os.sep)
        language = ""
        # 查找"source"目录后面的目录名作为语言
        if "source" in path_parts:
            source_index = path_parts.index("source")
            if source_index + 1 < len(path_parts):
                language = path_parts[source_index + 1]
        
        # 检查是否是日语PDF
        if language == "Japanese":
            return []
        else:
            # 执行原有中英文PDF处理逻辑
            pages = []
            
            # 使用PyMuPDF打开PDF
            doc = fitz.open(pdf_path)
            
            # 遍历PDF的每一页
            total_pages = len(doc)
            end_page = max_pages if max_pages and max_pages < total_pages else total_pages
            
            # print(f"总共 {total_pages} 页，将处理前 {end_page} 页")
            
            for page_num in range(end_page):
                text_blocks = []
                page = doc[page_num]
                
                # 获取页面尺寸
                page_rect = page.rect
                page_width = page_rect.width
                page_height = page_rect.height
                
                # 尝试使用PyMuPDF以多种方式获取文本
                all_text = page.get_text()  # 首先获取页面完整文本
                
                # 1. 基础提取：使用更精确的dict模式
                text_dict = page.get_text("dict")
                all_blocks = text_dict.get("blocks", [])
                
                # 2. 自定义算法：基于spans编写自己的合并逻辑
                custom_text_blocks = []
                
                # 遍历每个块
                for block in all_blocks:
                    # 跳过不是文本块的元素
                    if block.get("type") != 0:
                        continue
                    
                    lines = block.get("lines", [])
                    
                    # 按行处理
                    for line in lines:
                        spans = line.get("spans", [])
                        
                        if not spans:
                            continue
                        
                        # 合并同一行的spans
                        merged_text = ""
                        min_x0 = spans[0]["bbox"][0]
                        min_y0 = spans[0]["bbox"][1]
                        max_x1 = spans[0]["bbox"][2]
                        max_y1 = spans[0]["bbox"][3]
                        
                        # 按x坐标排序spans
                        spans_sorted = sorted(spans, key=lambda s: s["bbox"][0])
                        
                        for span in spans_sorted:
                            merged_text += span["text"]
                            # 更新边界框
                            min_x0 = min(min_x0, span["bbox"][0])
                            min_y0 = min(min_y0, span["bbox"][1])
                            max_x1 = max(max_x1, span["bbox"][2])
                            max_y1 = max(max_y1, span["bbox"][3])
                        
                        merged_text = merged_text.strip()
                        
                        if merged_text:
                            # 3. 坐标转换：确保使用正确的坐标系统
                            # 注意：PyMuPDF默认使用左上角为原点，通常不需要转换
                            text_block = TextBlock(
                                text=merged_text,
                                x0=min_x0,
                                y0=min_y0,
                                x1=max_x1,
                                y1=max_y1,
                                page_number=page_num + 1
                            )
                            custom_text_blocks.append(text_block)
                
                # 4. 进一步优化：基于段落合并行
                if len(custom_text_blocks) > 1:
                    # 按y坐标排序文本块
                    sorted_blocks = sorted(custom_text_blocks, key=lambda b: b.y0)
                    
                    merged_blocks = [sorted_blocks[0]]
                    
                    for i in range(1, len(sorted_blocks)):
                        current_block = sorted_blocks[i]
                        last_block = merged_blocks[-1]
                        
                        # 计算垂直间距
                        vertical_space = current_block.y0 - last_block.y1
                        
                        # 如果垂直间距较小且字体相同，合并为一个段落
                        # 这里简化处理，只基于垂直间距判断
                        if vertical_space < 5:  # 可调整的阈值
                            # 合并文本和边界框
                            merged_text = last_block.text + " " + current_block.text
                            merged_blocks[-1] = TextBlock(
                                text=merged_text,
                                x0=min(last_block.x0, current_block.x0),
                                y0=last_block.y0,
                                x1=max(last_block.x1, current_block.x1),
                                y1=current_block.y1,
                                page_number=page_num + 1
                            )
                        else:
                            merged_blocks.append(current_block)
                    
                    text_blocks = merged_blocks
                else:
                    text_blocks = custom_text_blocks
                
                # print(f"第 {page_num + 1} 页，优化后文本块数量: {len(text_blocks)}")
                
                # 如果自定义提取失败，尝试其他提取方式作为后备
                if not text_blocks:
                    # 尝试使用words模式作为后备
                    words = page.get_text("words")
                    # print(f"第 {page_num + 1} 页，尝试使用单词级提取: {len(words)} 个单词")
                    
                    if len(words) > 5:
                        # 按行分组
                        if words:
                            words_sorted = sorted(words, key=lambda w: (w[1], w[0]))
                            
                            lines = []
                            current_line = [words_sorted[0]]
                            
                            for word in words_sorted[1:]:
                                y_diff = abs(word[1] - current_line[-1][1])
                                if y_diff < 5:
                                    current_line.append(word)
                                else:
                                    lines.append(current_line)
                                    current_line = [word]
                            
                            if current_line:
                                lines.append(current_line)
                            
                            # print(f"第 {page_num + 1} 页，分组后的文本行数: {len(lines)}")
                            
                            # 将每行转换为文本块
                            for line in lines:
                                if line:
                                    x0 = min(word[0] for word in line)
                                    y0 = min(word[1] for word in line)
                                    x1 = max(word[2] for word in line)
                                    y1 = max(word[3] for word in line)
                                    
                                    text = " ".join(word[4] for word in line)
                                    text = text.strip()
                                    
                                    if text:
                                        text_block = TextBlock(
                                            text=text,
                                            x0=x0,
                                            y0=y0,
                                            x1=x1,
                                            y1=y1,
                                            page_number=page_num + 1
                                        )
                                        text_blocks.append(text_block)
                    
                    # 如果仍失败，使用完整文本作为最后后备
                    if not text_blocks:
                        all_text = page.get_text()
                        if all_text.strip():
                            # print(f"第 {page_num + 1} 页，使用完整文本作为后备")
                            text_block = TextBlock(
                                text=all_text.strip(),
                                x0=0,
                                y0=0,
                                x1=page_width,
                                y1=page_height,
                                page_number=page_num + 1
                            )
                            text_blocks.append(text_block)
                # 如果所有文本提取方式都失败，使用 PDFMiner 作为后备解析器（替代耗时的 OCR 流程）
                if not text_blocks:
                    try:
                        miner_blocks = _PDFMinerBackend.extract_blocks_from_page(pdf_path, page_num, language)
                        if miner_blocks:
                            # PDFMiner 返回的坐标已转换为与 PyMuPDF 一致的坐标系
                            text_blocks.extend(miner_blocks)
                    except Exception as e:
                        print(f"PDFMiner 后备解析失败（页 {page_num + 1}）: {e}")
                
                # 创建页面对象并添加到列表
                page_obj = Page(page_number=page_num + 1, text_blocks=text_blocks, language=language)
                pages.append(page_obj)
            
            # 关闭文档
            doc.close()
            
            return pages

