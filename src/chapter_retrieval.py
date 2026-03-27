#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
章节检索器

基于章节标题进行精确匹配检索
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()


class ChapterRetriever:
    """章节检索器"""
    
    def __init__(self, pages_title_dir: str = "src/data/pages_title"):
        """
        初始化章节检索器
        
        Args:
            pages_title_dir: 标题 JSON 文件目录
        """
        self.pages_title_dir = Path(pages_title_dir)
        self.chapters_cache: Dict[str, List[Dict]] = {}
        
        self._load_all_chapter_titles()
    
    def _load_all_chapter_titles(self):
        """加载所有书籍的章节标题到缓存"""
        if not self.pages_title_dir.exists():
            return
        
        for json_file in self.pages_title_dir.glob("*_titles.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 从文件名提取书名
                book_name = json_file.stem.replace('_titles', '')
                self.chapters_cache[book_name] = data.get('titles', [])
                
            except Exception as e:
                print(f"加载章节标题失败 {json_file.name}: {e}")
    
    def search(self, 
               keywords: List[str], 
               book_name: Optional[str] = None,
               top_k: int = 5) -> List[Dict[str, Any]]:
        """
        章节检索
        
        Args:
            keywords: 关键词列表（如人物名、地点名）
            book_name: 指定书籍名称（可选）
            top_k: 返回结果数量
            
        Returns:
            匹配的章节列表
        """
        matches = []
        
        # 确定搜索范围
        if book_name and book_name in self.chapters_cache:
            search_books = [book_name]
        else:
            search_books = list(self.chapters_cache.keys())
        
        # 遍历所有章节标题
        for book in search_books:
            chapters = self.chapters_cache[book]
            
            for chapter in chapters:
                title = chapter.get('title', '')
                page = chapter.get('page', 0)
                
                # 计算关键词匹配得分
                match_score = sum(1 for kw in keywords if kw in title)
                
                if match_score > 0:
                    matches.append({
                        'id': f"{book}_chapter_{page}",
                        'book': book,
                        'chapter_title': title,
                        'page': page,
                        'match_score': match_score,
                        'source_type': 'chapter_index'
                    })
        
        # 按匹配得分排序
        matches.sort(key=lambda x: x['match_score'], reverse=True)
        
        return matches[:top_k]
    
    def get_chapter_by_page(self, 
                            book_name: str, 
                            page: int) -> Optional[Dict[str, Any]]:
        """
        根据页码获取章节信息
        
        Args:
            book_name: 书籍名称
            page: 页码
            
        Returns:
            章节信息字典
        """
        if book_name not in self.chapters_cache:
            return None
        
        chapters = self.chapters_cache[book_name]
        for chapter in chapters:
            if chapter.get('page') == page:
                return chapter
        
        return None
