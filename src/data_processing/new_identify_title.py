from dotenv import load_dotenv
import os
import sys
from pathlib import Path

# 🔥 添加项目根目录到 Python 路径，支持从 src 目录直接运行
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.pdf_loader import PDFLoader
import ast
import logging
import json
import time
import hashlib
from typing import List, Dict, Any
load_dotenv()

class TitleIdentifier:
    def __init__(self):
        # 初始化 logger
        self.logger = logging.getLogger(__name__)
        
        try:
            self.pdf_loader = PDFLoader()
            env_punctuation = os.getenv("all_punctuation_list")
            self.all_punctuation_list = ast.literal_eval(env_punctuation)
            print(f"成功加载 {len(self.all_punctuation_list)} 个标点符号")
        except Exception as e:
            print(f"初始化失败：{e}")
            sys.exit(1)
        
        # 加载副标题关键词列表
        try:
            self.chinese_subtitle_keywords = ast.literal_eval(os.getenv("chinese_subtitle_keywords", "[]"))
            self.japanese_subtitle_keywords = ast.literal_eval(os.getenv("japanese_subtitle_keywords", "[]"))
            self.english_subtitle_keywords = ast.literal_eval(os.getenv("english_subtitle_keywords", "[]"))
            # 加载中文量词列表
            self.chinese_classifiers = ast.literal_eval(os.getenv("chinese_classifiers", "[]"))
            print(f"成功加载副标题关键词列表")
            print(f"成功加载中文量词列表：{len(self.chinese_classifiers)}个")
        except Exception as e:
            print(f"副标题关键词列表加载失败：{e}")
            sys.exit(1)
        
        # 加载标题长度限制配置
        try:
            self.chinese_title_max_chars = int(os.getenv("chinese_title_max_chars", "7"))
            self.japanese_title_max_chars = int(os.getenv("japanese_title_max_chars", "8"))
            self.english_title_max_words = int(os.getenv("english_title_max_words", "7"))
            print(f"成功加载标题长度限制：中文={self.chinese_title_max_chars}字符，日文={self.japanese_title_max_chars}字符，英文={self.english_title_max_words}单词")
        except Exception as e:
            print(f"标题长度限制加载失败：{e}")
            # 使用默认值
            self.chinese_title_max_chars = 7
            self.japanese_title_max_chars = 8
            self.english_title_max_words = 7
    
    def _get_subtitle_keywords(self, language: str) -> List[str]:
        """
        根据语言获取对应的副标题关键词列表
        
        Args:
            language: 语言类型（"Chinese", "Japanese", "English"）
            
        Returns:
            List[str]: 副标题关键词列表
        """
        if language == "Chinese":
            return self.chinese_subtitle_keywords
        elif language == "Japanese":
            return self.japanese_subtitle_keywords
        elif language == "English":
            return self.english_subtitle_keywords
        else:
            # 未知语言返回空列表
            print(f"⚠️  未知语言：{language}，返回空副标题关键词列表")
            return []
    
    def is_chinese_classifier_subtitle(self, title: str, has_keyword: bool) -> bool:
        """
        判断包含中文量词的标题是否应该被识别为副标题
        
        规则（按优先级排序）：
        1. 量词 + 空格组合：如果标题的第一个字符是中文量词，且标题中包含空格 → 是副标题
        2. 纯量词标题：如果标题的所有字符都是中文量词 → 是副标题
        3. 量词但无空格：如果标题第一个字符是中文量词，但不包含空格 → 不是副标题
        4. 量词 + 关键词缺失：如果标题包含中文量词，但不包含任何副标题关键词（除量词外） → 不是副标题
        5. 仅含关键词：如果标题不包含中文量词，但包含副标题关键词 → 是副标题
        6. 其他情况：以上规则均不满足 → 不是副标题
        
        Args:
            title: 待判断的标题
            has_keyword: 是否包含副标题关键词
            
        Returns:
            bool: 是否应该被识别为副标题
        """
        if not title or not isinstance(title, str):
            return False
        
        # 检查是否以中文量词开头
        first_char = title[0]
        is_classifier_start = first_char in self.chinese_classifiers
        
        # 检查是否包含空格
        has_space = ' ' in title
        
        # 检查是否所有字符都是量词
        all_classifiers = all(char in self.chinese_classifiers for char in title)
        
        # 🔥 检查是否包含非量词的关键词
        # 从关键词列表中排除量词，避免误判
        non_classifier_keywords = [kw for kw in self.chinese_subtitle_keywords if kw not in self.chinese_classifiers]
        has_non_classifier_keyword = any(kw in title for kw in non_classifier_keywords)
        
        # 规则 1：量词 + 空格组合
        if is_classifier_start and has_space:
            return True
        
        # 规则 2：纯量词标题
        if all_classifiers:
            return True
        
        # 规则 3：量词但无空格（第一个字符是量词，但没有空格）
        if is_classifier_start and not has_space:
            return False
        
        # 规则 4：量词 + 关键词缺失（包含量词但没有非量词关键词）
        if is_classifier_start and not has_non_classifier_keyword:
            return False
        
        # 规则 5：仅含关键词（不包含量词，但有非量词关键词）
        if not is_classifier_start and has_non_classifier_keyword:
            return True
        
        # 规则 6：其他情况
        return False
    
    def merge_subtitles(self, smart_titles: List[List[Dict]], language: str) -> List[List[Dict]]:
        """
        智能正副标题拼接：将同一页码的正标题和副标题拼接为完整标题
        
        Args:
            smart_titles: 智能处理后的标题列表（二维数组，包含 title 和 page）
            language: 语言类型（"Chinese", "Japanese", "English"）
            
        Returns:
            List[List[Dict]]: 拼接后的标题列表（保持二维数组结构）
        """
        if not smart_titles or len(smart_titles) == 0:
            return smart_titles
        
        # 获取对应语言的副标题关键词
        subtitle_keywords = self._get_subtitle_keywords(language)
        print(f"\n[INFO] 使用副标题关键词列表（{language}）：共{len(subtitle_keywords)}个")
        
        result = []
        i = 0
        main_title_buffer = None  # 🔥 跨组缓存最近的正标题
        
        while i < len(smart_titles):
            current_page_group = smart_titles[i]
            
            # 如果当前组为空，跳过
            if not current_page_group:
                result.append([])
                i += 1
                continue
            
            # 处理当前组内的标题
            merged_page = []
            j = 0
            
            while j < len(current_page_group):
                current_title = current_page_group[j]
                # 🔥 已拼接的标题（包含空格）不再作为副标题处理
                # 使用新的中文量词判断逻辑
                has_keyword_current = any(kw in current_title.get('title', '') for kw in subtitle_keywords)
                is_current_subtitle = (
                    ' ' not in current_title.get('title', '') and 
                    self.is_chinese_classifier_subtitle(current_title.get('title', ''), has_keyword_current)
                )
                
                # 检查是否有下一个标题且页码相同
                if j + 1 < len(current_page_group):
                    next_title = current_page_group[j + 1]
                    has_keyword_next = any(kw in next_title.get('title', '') for kw in subtitle_keywords)
                    is_next_subtitle = (
                        ' ' not in next_title.get('title', '') and 
                        self.is_chinese_classifier_subtitle(next_title.get('title', ''), has_keyword_next)
                    )
                    
                    # 如果页码不同，说明这一组已经结束
                    if current_title.get('page') != next_title.get('page'):
                        # 只添加当前标题，继续下一轮
                        merged_page.append(current_title)
                        j += 1
                        continue
                    
                    # 页码相同，进入副标题识别逻辑
                    # 规则：只能有一个元素被标记为"不包含副标题"（即正标题）
                    # 如果前一个元素包含副标题关键词，则后一个是正标题
                    if is_current_subtitle and not is_next_subtitle:
                        # 交换：next 是正标题，current 是副标题
                        main_title = next_title
                        sub_title = current_title
                        # 拼接
                        merged_title = f"{main_title['title']} {sub_title['title']}"
                        merged_page.append({
                            'title': merged_title,
                            'page': main_title['page']
                        })
                        # 更新正标题缓存
                        main_title_buffer = main_title
                        # 跳过这两个元素
                        j += 2
                    elif not is_current_subtitle and is_next_subtitle:
                        # current 是正标题，next 是副标题
                        main_title = current_title
                        sub_title = next_title
                        # 拼接
                        merged_title = f"{main_title['title']} {sub_title['title']}"
                        merged_page.append({
                            'title': merged_title,
                            'page': main_title['page']
                        })
                        # 🔥 更新正标题缓存：如果拼接后的标题包含空格，只缓存空格前的部分
                        main_title_buffer = {
                            'title': main_title['title'].split(' ')[0],  # 只保留第一部分
                            'page': main_title['page']
                        }
                        # 跳过这两个元素
                        j += 2
                    elif not is_current_subtitle and not is_next_subtitle:
                        # 两个都不是副标题，都添加
                        merged_page.append(current_title)
                        # 更新正标题缓存
                        main_title_buffer = current_title
                        j += 1
                    else:
                        # 两个都是副标题（异常情况）
                        # 优先使用跨组缓存的正标题
                        main_title = main_title_buffer
                        
                        if main_title:
                            # 找到正标题，将两个副标题分别与正标题拼接
                            # 第一个副标题与正标题拼接
                            merged_title_1 = f"{main_title['title']} {current_title['title']}"
                            merged_page.append({
                                'title': merged_title_1,
                                'page': current_title['page']
                            })
                            # 第二个副标题与正标题拼接
                            merged_title_2 = f"{main_title['title']} {next_title['title']}"
                            merged_page.append({
                                'title': merged_title_2,
                                'page': next_title['page']
                            })
                            # 🔥 不清空缓存，让后续副标题继续使用
                        else:
                            # 没找到正标题，直接添加两个副标题
                            merged_page.append(current_title)
                            merged_page.append(next_title)
                        
                        # 跳过这两个元素
                        j += 2
                else:
                    # 没有下一个标题了
                    # 检查是否是副标题且有缓存的正标题（跨组）
                    if is_current_subtitle and main_title_buffer:
                        # 与正标题拼接
                        merged_title = f"{main_title_buffer['title']} {current_title['title']}"
                        merged_page.append({
                            'title': merged_title,
                            'page': current_title['page']
                        })
                        # 🔥 不清空缓存，让后续副标题继续使用
                    else:
                        # 直接添加，如果是正标题则更新缓存
                        merged_page.append(current_title)
                        if not is_current_subtitle:
                            # 如果标题包含空格（已拼接），只缓存第一部分
                            base_title = current_title['title'].split(' ')[0]
                            main_title_buffer = {
                                'title': base_title,
                                'page': current_title['page']
                            }
                    j += 1
            
            result.append(merged_page)
            i += 1
        
        print(f"副标题拼接完成：原始{sum(len(p) for p in smart_titles)}个标题 → 拼接后{sum(len(p) for p in result)}个标题")
        return result
    
    def not_has_punctuation(self, text: str) -> bool:
        """
        判断文本是否包含标点符号（高效版本）
        
        Args:
            text: 待检测的文本字符串
            
        Returns:
            bool: 是否不包含标点符号
            
        """
        if not text or not isinstance(text, str):
            return False
        
        # 🔥 核心优化：使用 set 的 O(1) 查找 + any() 短路求值
        return not any(char in self.all_punctuation_list for char in text)
    
    def check_text_centered(self, block, page_width: float, tolerance: float = 0.1) -> bool:
        # 🔍 步骤 1：安全检查 - 确保文本块有位置信息
        if not hasattr(block, 'x0') or not hasattr(block, 'x1'):
            return False
        
        # 📏 步骤 2：获取文本块的左右边界
        left = block.x0   # 左边界坐标
        right = block.x1  # 右边界坐标
        
        # 🎯 步骤 3：计算文本块的中心点
        block_center = (left + right) / 2
        # print(f"  w{block}文本块中心点：{block_center}")
        # # 换行
        # print()
        # 📐 步骤 4：计算页面的居中区域
        # 页面中心点
        page_center = page_width / 2
        
        # 居中区域的左右边界（修改为：页面中心的 ±10%）
        center_start = page_width * (0.5 - tolerance)  # 例如：612 × 0.4 = 244.8
        center_end = page_width * (0.5 + tolerance)    # 例如：612 × 0.6 = 367.2
        
        # ✅ 步骤 5：判断文本块中心是否在居中区域内
        is_centered = center_start <= block_center <= center_end
        
        return is_centered

    # 判断文本是否符合标题的条件，最终返回数据为列表，每个列表代表每一页的标题，也就是通过筛选的文本块
    def is_valid_title(self, pdf_path: str) -> tuple[List[List[str]], str]:
        """
        识别 PDF 中每页的标题
            
        Args:
            pdf_path: PDF 文件路径
                
        Returns:
            tuple: (标题二维列表，语言类型)
            - List[List[str]]: 二维列表，外层索引是页码（从 0 开始），内层是该页的所有标题
            - str: 语言类型（如 "Chinese", "Japanese", "English"）
            - List[List[tuple[float, float]]]: 二维列表，外层索引是页码（从 0 开始），内层是该页所有标题的上下边界坐标
        """
        # 标题列表，用于记录每页的标题
        all_titles = []
        # 用于记录文本块的上下边界坐标
        block_bounds = []
        pages = self.pdf_loader.load_pdf(str(pdf_path))
        # 获取语言类型
        language = pages[0].language if pages else "Unknown"
        print(f"  识别语言：{language}")
        # 判断标题二阶段，文本块不包含标点符号且居中
        print("  判断标题...")
        for page_idx, page in enumerate(pages):
            page_titles = []
            page_bounds = []
            for block in page.text_blocks:
                # 如果文本块不含标点符号且居中，进入判断二，是否超过规定长度
                if self.not_has_punctuation(block.text) and self.check_text_centered(block, page_width=page.page_width):
                    if self.check_title_length(block.text, language):
                        page_titles.append(block.text)
                        page_bounds.append((block.y0, block.y1))
                    
            if page_titles:
                all_titles.append(page_titles)
                block_bounds.append(page_bounds)
            else:
                block_bounds.append([])
                all_titles.append([])
        
        # 拼接上下坐标相同的文本块（如果存在）并去重
        all_titles = self.remove_duplicates(self.process_block_bounds(all_titles, block_bounds, language))
        print(f"  成功识别 {len(all_titles)} 页的标题")
        return all_titles, language

    # 对传入的二维数据进行去重
    def remove_duplicates(self, data: List[List[str]]) -> List[List[str]]:
        """
        全局去重：
        1. 如果某个标题在所有页面中出现次数 >= 3 次，则认为是无关标题（如页眉页脚），从所有页面中删除
        
        Args:
            data: 二维列表，如 [["标题 1", "标题 2"], ["标题 1"], ["标题 3"]]
            language: 语言类型（如 "Chinese", "Japanese", "English"）
        
        Returns:
            去除全局重复标题后的二维列表
        
        Example:
            >>> data = [["格林童话", "第一章"], ["格林童话"], ["第二章"], ["格林童话"]]
            >>> # "格林童话"出现 3 次，被删除
            >>> result = [["第一章"], [""], ["第二章"], [""]]
        """
        from collections import Counter
        
        # 步骤 1：统计每个标题在所有页面中出现的总次数
        all_titles_counter = Counter()
        for page_titles in data:
            # 使用 set 去重，避免同一页面内重复计算
            unique_titles_in_page = set(title for title in page_titles if title)
            all_titles_counter.update(unique_titles_in_page)
        
        # 步骤 2：找出出现次数 >= 3 次的标题（认为是无关标题）
        irrelevant_titles = {title for title, count in all_titles_counter.items() if count > 3}
        
        if irrelevant_titles:
            print(f"  检测到无关标题（出现>=3 次）: {irrelevant_titles}")
        
        # 步骤 3：从所有页面中删除无关标题
        result = []
        for page_titles in data:
            # 过滤掉无关标题
            filtered_titles = [title for title in page_titles if title and title not in irrelevant_titles]
            
            # 如果过滤后为空，保留空字符串占位
            if not filtered_titles:
                filtered_titles = [""]
            
            result.append(filtered_titles)
        
        return result
    
    def check_title_length(self, title: str, language: str) -> bool:
        """
        检查标题长度是否符合语言类型的预设限制
        
        Args:
            title: 单个标题字符串
            language: 语言类型（"Chinese", "Japanese", "English"）
            
        Returns:
            bool: 标题长度是否合规（True 表示符合，False 表示超出限制）
        """
        if not title or not isinstance(title, str):
            return False
                
        # 获取对应的最大长度限制
        if language == "Chinese":
            max_length = self.chinese_title_max_chars
            # 中文字符数（一个汉字算 1 个字符）
            actual_length = len(title)
        elif language == "Japanese":
            max_length = self.japanese_title_max_chars
            # 日文字符数（包括平假名、片假名、汉字等）
            actual_length = len(title)
        elif language == "English":
            max_length = self.english_title_max_words
            # 英文单词数（按空格分割）
            actual_length = len(title.split())
        else:
            # 未知语言，报错
            raise ValueError(f"未知语言：{language}")
        
        # 检查是否符合长度限制
        return actual_length <= max_length
    # 智能标题识别逻辑，对传入的二维数据进行处理
    def process_block_bounds(self, all_titles: List[List[str]], block_bounds: List[List[tuple[float, float]]], language: str = None) -> List[List[str]]:
        """
        处理 block_bounds，将同一页面中上下边界坐标相同的标题进行拼接
        
        Args:
            all_titles: 二维列表，外层索引是页码，内层是该页的所有标题
            block_bounds: 二维列表，外层索引是页码，内层是该页所有标题的上下边界坐标 (y0, y1)
            
        Returns:
            处理后的 all_titles，相同坐标的标题已被拼接
        """
        if not all_titles or not block_bounds:
            return all_titles
        
        processed_titles = []
        
        # 遍历每一页
        for page_idx, page_titles in enumerate(all_titles):
            if page_idx >= len(block_bounds):
                processed_titles.append(page_titles)
                continue
                
            page_bounds = block_bounds[page_idx]
            
            # 如果该页没有标题或只有一个标题，直接添加
            if len(page_titles) <= 1:
                processed_titles.append(page_titles)
                continue
            
            # 处理同一页面中的标题
            merged_page_titles = []
            used_indices = set()  # 记录已处理的标题索引
            
            for i, title in enumerate(page_titles):
                if i in used_indices:
                    continue
                    
                current_bound = page_bounds[i] if i < len(page_bounds) else None
                
                # 查找后续标题中是否有相同坐标的
                same_bound_titles = [title]
                
                for j in range(i + 1, len(page_titles)):
                    if j in used_indices:
                        continue
                        
                    other_bound = page_bounds[j] if j < len(page_bounds) else None
                    
                    # 如果上下边界坐标相同，则合并
                    if current_bound and other_bound and current_bound == other_bound:
                        same_bound_titles.append(page_titles[j])
                        used_indices.add(j)
                
                # 将相同坐标的标题拼接
                merged_title = ''.join(same_bound_titles)
                
                # 🔥 立即检查长度，符合条件的才添加
                if self.check_title_length(merged_title, language):
                    merged_page_titles.append(merged_title)
                
                used_indices.add(i)
            
            # 🔥 重要：将处理后的页面标题添加到结果列表
            processed_titles.append(merged_page_titles if merged_page_titles else [""])
        
        return processed_titles

    def smart_title_identification(self, data: List[List[Dict]], batch_size: int = 20) -> List[List[Dict]]:
        """
        使用本地 Ollama 模型进行智能标题识别和合并
        
        Args:
            data: 二维列表，外层索引是页码，内层是该页的所有标题对象 {"title": "xxx", "page": xxx}
            batch_size: 每批次处理的元素数量，默认 15 个
            
        Returns:
            List[List[Dict]]: 处理后的标题数组，包含 title 和 page
        """
        import requests
        import json as json_lib
        
        if not data or not isinstance(data, list):
            raise ValueError(f"标题识别输入数据无效：{data}")
        
        all_results = []
        
        # 📊 处理所有批次（移除批次数量限制）
        for batch_idx in range(0, len(data), batch_size):
            batch_end = min(batch_idx + batch_size, len(data))
            batch_data = data[batch_idx:batch_end]
            current_batch_num = batch_idx // batch_size + 1
            total_batches = (len(data) + batch_size - 1) // batch_size
            
            print(f"\n{'='*60}")
            print(f"  处理第 {current_batch_num} 个批次")
            print(f"{'='*60}")
            print(f"\n  批次 {current_batch_num}: 第{batch_idx + 1}-{batch_end}条记录")
            print(f"  本批次数据量：{len(batch_data)} 条")
            print(f"\n📋 输入数据:")
            for i, item in enumerate(batch_data, 1):
                print(f"  {i}. {item}")
            
            try:
                # 🤖 调用 Ollama 模型进行推理
                result, _ = self._call_ollama_model(batch_data)
                
                if result:
                    # ✅ 解析成功
                    all_results.extend(result)
                    print(f"\n  ✓ 批次 {current_batch_num} 处理完成")
                    print(f"\n📊 当前累计结果:")
                    for i, page_titles in enumerate(all_results, 1):
                        print(f"  {i}. {page_titles}")
                else:
                    print(f"  ⚠️ 批次 {current_batch_num} 无有效结果")
                    
            except requests.exceptions.ConnectionError as e:
                print(f"\n  ❌ 错误：无法连接到 Ollama 服务")
                print(f"  请确保 Ollama 已在本地启动 (http://localhost:11434)")
                print(f"  异常详情：{e}")
                # 降级处理：返回原始数据
                return data
                
            except requests.exceptions.Timeout as e:
                print(f"\n  ❌ 错误：Ollama 请求超时 (60 秒)")
                print(f"  模型可能正在处理复杂请求或服务未响应")
                print(f"  异常详情：{e}")
                return data
                
            except Exception as e:
                print(f"\n  ❌ 错误：智能标题识别失败")
                print(f"  异常详情：{e}")
                import traceback
                traceback.print_exc()
                # 降级处理
                return data
        
        print(f"\n{'='*60}")
        print(f"  ✅ 所有 {total_batches} 个批次处理完成")
        print(f"{'='*60}")
        print(f"\n📊 最终结果:")
        for i, page_titles in enumerate(all_results, 1):
            print(f"  {i}. {page_titles}")
        
        return all_results
    
    def _call_ollama_model(self, batch_data: List[List[Dict]], context_title: str = "", retry_count: int = 3) -> tuple[List[List[Dict]], str]:
        """
        调用本地 Ollama API 进行模型推理（带重试机制）
        
        Args:
            batch_data: 当前批次的数据
            context_title: 上一批次的标题（已废弃，保留参数为了向后兼容）
            retry_count: 最大重试次数，默认 3 次
            
        Returns:
            tuple: (处理后的标题数组，空字符串)
        """
        import requests
        import json as json_lib
        from time import sleep
        
        # 🎯 构建 Prompt
        prompt = self._build_prompt(batch_data, context_title)
        
        # 📡 API 配置
        ollama_url = "http://localhost:11434/api/generate"
        payload = {
            "model": "qwen3:8b",  # 通义千问 8B，中文理解优秀
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # 低温度，保证输出稳定
                "top_p": 0.9,
            }
        }
        
        print(f"  📡 正在调用 Ollama API...")
        
        # 🔄 重试机制
        last_exception = None
        for attempt in range(1, retry_count + 1):
            try:
                # 🚀 发送 POST 请求（超时 60 秒）
                response = requests.post(
                    ollama_url,
                    json=payload,
                    timeout=60  # 60 秒超时
                )
                
                # ✅ 检查响应状态
                response.raise_for_status()
                
                # 📦 解析 JSON 响应
                try:
                    result_json = response.json()
                    model_output = result_json.get("response", "")
                    
                    if not model_output:
                        print(f"  ⚠️ 模型返回空响应")
                        raise RuntimeError("模型返回空响应，无法解析标题")
                    
                    print(f"\n📝 模型原始输出:\n{model_output}")
                    print()
                    
                    # 🔍 提取 JSON 部分并解析
                    titles = self._parse_model_output(model_output)
                    print(f"  ✓ 成功解析{len(titles)}个标题")
                    return titles, ""
                        
                except ValueError as e:
                    print(f"  ❌ {e}")
                    raise RuntimeError(f"标题识别失败：{e}") from e
                except json_lib.JSONDecodeError as e:
                    print(f"  ❌ JSON 解析失败：{e}")
                    raise RuntimeError(f"JSON 解析失败：{e}") from e
                except Exception as e:
                    print(f"  ❌ 响应解析失败：{e}")
                    raise RuntimeError(f"响应解析失败：{e}") from e
                    
            except requests.exceptions.ConnectionError as e:
                last_exception = e
                print(f"  ⚠️ 连接失败 (尝试 {attempt}/{retry_count}): {e}")
                if attempt < retry_count:
                    sleep(2)  # 等待 2 秒后重试
                continue
                
            except requests.exceptions.Timeout as e:
                last_exception = e
                print(f"  ⚠️ 请求超时 (尝试 {attempt}/{retry_count}, 60 秒): {e}")
                if attempt < retry_count:
                    sleep(3)  # 等待 3 秒后重试
                continue
                
            except Exception as e:
                last_exception = e
                print(f"  ⚠️ 未知错误 (尝试 {attempt}/{retry_count}): {e}")
                if attempt < retry_count:
                    sleep(2)
                continue
        
        # ❌ 所有重试都失败
        print(f"  ❌ 所有{retry_count}次重试均失败")
        if last_exception:
            print(f"  最后异常：{last_exception}")
        
        # 🚨 直接抛出异常终止程序
        raise RuntimeError(
            f"调用 Ollama API 失败，已重试{retry_count}次。"
            f"请检查 Ollama 服务是否启动 (地址：{ollama_url})，模型 '{model_name}' 是否可用。"
        ) from last_exception
    
    def _build_prompt(self, batch_data: List[List[Dict]], context_title: str = "") -> str:
        """
        构建给模型的 Prompt（精简版）
        
        Args:
            batch_data: 当前批次的数据
            context_title: 上一批次的标题（已废弃，保留参数为了向后兼容）
            
        Returns:
            str: 构建好的 Prompt
        """
        # 📝 格式化输入数据
        formatted_input = "\n".join([f"  {i+1}. {item}" for i, item in enumerate(batch_data)])
        
        # 🎯 构建完整的 Prompt（精简版）
        prompt = f"""你是 PDF 标题筛选助手。

## 任务
从传入的标题列表中筛选出有效的章节标题，删除无效内容并清洗文本。

## 输入格式
二维数组，每个元素是列表，包含该页的标题对象：{{"title": "标题", "page": 页码}}

## 处理规则

### 1. 判断有效性
- ✅ **有效标题**：真正的章节标题、序言、后记等
  - 示例："打火匣", "第一夜", "序言", "第一章", "读后记"
- ❌ **无效内容**：对话片段、乱码、无意义文本、页眉页脚
  - 示例："我想进去亲一下母鸡呀", "老太婆说", "他回答说", "N0098JH", "けん T"

### 2. 文本清洗
- **删除 Unicode 控制字符**：如 \u3000（全角空格）应替换为普通空格或删除
  - 错误：`"第二章\u3000优胜记略"`
  - 正确：`"第二章 优胜记略"`
- **删除特殊符号**：删除反斜杠 `\` 及其他转义字符
  - 错误：`"四 \\"化民成俗\\""`
  - 正确：`"四 化民成俗"`
- **修复乱码**：识别并修复明显的乱码或格式错误

### 3. 输出格式
- **重要**：只返回包含有效标题的页面，没有有效标题的页面直接跳过，不要返回空数组
- 输出必须是与输入相同的二维数组结构
- 每个元素包含 title 和 page 字段

## 输出要求
- 返回 JSON 格式的二维数组
- **只包含有有效标题的页面**
- 只保留清洗后的有效标题对象（含 title 和 page）

## 示例
输入：[
  [{{"title": "第一章", "page": 1}}], 
  [{{"title": "对话片段", "page": 2}}],
  [{{"title": "第二章\\u3000优胜记略", "page": 3}}]
]
输出：[
  [{{"title": "第一章", "page": 1}}],
  [{{"title": "第二章 优胜記略", "page": 3}}]
]

## 当前输入
{formatted_input}

请返回 JSON 二维数组："""
        
        return prompt
    
    def _parse_model_output(self, output: str) -> List[List[Dict]]:
        """
        从模型输出中提取 JSON 二维数组（包含 title 和 page）
        
        Args:
            output: 模型的原始输出文本
            
        Returns:
            List[List[Dict]]: 解析后的二维标题列表，每个元素是 {"title": "xxx", "page": xxx}
        """
        import json as json_lib
        import re
        
        # 🔍 先提取代码块中的 JSON 部分
        code_block_match = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', output)
        json_str = code_block_match.group(1).strip() if code_block_match else output.strip()
        
        print(f"  📦 从代码块中提取 JSON: {json_str[:100]}...")
        
        try:
            result = json_lib.loads(json_str)
            
            # 确保是二维数组
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], list):
                    # 验证每个元素的格式
                    validated_result = []
                    for page_group in result:
                        if isinstance(page_group, list):
                            validated_page = [item for item in page_group if isinstance(item, dict) and 'title' in item and 'page' in item]
                            validated_result.append(validated_page)
                    return validated_result
                else:
                    raise ValueError(f"输出格式错误：期望二维数组，收到一维数组")
            
            raise ValueError(f"输出不是有效的二维数组格式：{type(result)}")
        except json_lib.JSONDecodeError as e:
            raise ValueError(f"JSON 解析失败：{e}")
    
    def _fallback_flatten(self, data: List[List[Dict]]) -> List[Dict]:
        """
        降级处理：当模型调用失败时，返回扁平化的原始数据
        
        Args:
            data: 二维列表
            
        Returns:
            List[str]: 扁平化后的一维列表
        """
        result = []
        for page_titles in data:
            if page_titles:
                merged = ' '.join(page_titles)
                if merged.strip():
                    result.append(merged)
        return result
    
    def _convert_to_page_titles(self, smart_titles: List[List[Dict]]) -> Dict[str, Any]:
        """
        将处理后的标题列表转换为扁平化的 JSON 结构
        
        Args:
            smart_titles: 智能处理后的标题列表（二维数组，包含 title 和 page）
            
        Returns:
            Dict: {"titles": List[Dict]}
                  其中每个 title 对象包含 title 和 page 字段
        """
        # 扁平化所有标题
        all_titles = [title for page_group in smart_titles for title in page_group if title]
        
        return {"titles": all_titles}
    def process_all_titles(self, all_titles: List[List[str]]) -> List[List[Dict[str, Any]]]:
        """
        对传入的 all_titles 进行处理，返回包含标题和对应页码的二维数组
            
        Args:
            all_titles: 二维列表，每个元素是包含单个字符串的列表
                
        Returns:
            List[List[Dict[str, Any]]]: 过滤后的二维数组，每个元素是字典列表，
                                        每个字典包含'title'和'page'键
                                        例如：[[{'title': '第一章', 'page': 1}], [{'title': '第二章', 'page': 2}]]
        """
        result = []
        for page_index, page in enumerate(all_titles):
            # 过滤掉空列表和只包含空字符串的列表
            if page and any(title.strip() for title in page):
                # 为每个有效标题添加页码信息（页码 = 下标 + 1）
                page_with_numbers = [
                    {'title': title.strip(), 'page': page_index + 1}
                    for title in page if title.strip()
                ]
                if page_with_numbers:
                    result.append(page_with_numbers)
        
        return result

def main():
    """主函数 - 遍历并处理所有 PDF 文件"""
    print("=" * 60)
    print("开始执行...")
    # 🔥 定义路径 - new_identify_title.py 位于 src/data_processing/
    # parent 是 src/, parent.parent 是项目根目录
    project_root = Path(__file__).resolve().parent.parent.parent
    source_dir = project_root / "src" / "data" / "source"
    output_dir = project_root / "src" / "data" / "pages_title"
    
    print(f"项目根目录：{project_root}")
    print(f"PDF 源目录：{source_dir}")
    print(f"输出目录：{output_dir}")

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nPDF 章节标题识别工具（新版）")
    print("=" * 60)
    
    # 初始化组件
    print("正在初始化 TitleIdentifier...")
    title_identifier = TitleIdentifier()
    print("初始化完成！")
    
    # 页码及预备标题列表，用于存储每页的页码和预备标题
    page_info_list = []
    # 查找所有 PDF 文件
    pdf_files = list(source_dir.rglob("*.pdf"))
    if not pdf_files:
        print(f"在 {source_dir} 中未找到 PDF 文件")
        return
    
    print(f"找到 {len(pdf_files)} 个 PDF 文件")
    print(f"输出目录：{output_dir}")
    print()
    for pdf_path in pdf_files:
        try:
            print(f"正在处理：{pdf_path.relative_to(source_dir)}")
                
            # 生成对应的 JSON 缓存文件路径
            json_filename = f"{pdf_path.stem}_titles.json"
            json_path = output_dir / json_filename
            # 简化缓存逻辑：如果 JSON 已存在则跳过（避免复杂的哈希比较）
            if json_path.exists():
                print(f"  缓存存在，跳过处理")
                continue
            # 获取所有页面的标题
            all_titles, language = title_identifier.is_valid_title(pdf_path)
            
            print(f"\n  识别语言：{language}")
            print(f"  成功识别 {len(all_titles)} 页的标题")
            print(f"  原始标题列表:\n  {all_titles}")
            # 🚀 使用 Ollama 模型进行智能标题识别（批次大小=15）
            print("\n" + "="*60)
            print("开始智能标题识别...")
            print("="*60)
            all_titles = title_identifier.process_all_titles(all_titles)
            print(all_titles)
            batch_size = 15  # 固定批次大小为 15
            smart_titles = title_identifier.smart_title_identification(all_titles, batch_size=batch_size)
            # 智能正副标题拼接
            print("\n" + "="*60)
            print("开始正副标题拼接...")
            print("="*60)
            smart_titles = title_identifier.merge_subtitles(smart_titles, language)
            
            # 📄 将结果转换为每页标题的 JSON 格式
            page_titles_json = title_identifier._convert_to_page_titles(smart_titles)
            
            # 💾 保存为 JSON 文件
            json_filename = f"{pdf_path.stem}_titles.json"
            json_path = output_dir / json_filename
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(page_titles_json, f, ensure_ascii=False, indent=2)
            
            print(f"\n✅ 处理完成！")
            print(f"  原始页数：{len(all_titles)}")
            print(f"  有效标题数：{len(smart_titles)}")
            print(f"  结果已保存到：{json_path}")
        except Exception as e:
            print(f"处理 {pdf_path} 时出错：{e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()

