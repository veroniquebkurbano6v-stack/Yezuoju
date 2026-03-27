#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DeepSeek API 回复缓存模块
避免重复调用 API，提高评测效率
"""

import os
import json
import hashlib
import pickle
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class DeepSeekCache:
    """DeepSeek API 回复缓存（基于问题 MD5 哈希）"""
    
    def __init__(self, cache_dir: str = "cache/deepseek"):
        """
        初始化缓存
        
        Args:
            cache_dir: 缓存目录
        """
        self.cache_dir = Path(cache_dir)
        self.cache_file = self.cache_dir / "cache.json"
        self.metadata_file = self.cache_dir / "metadata.json"
        
        # 创建缓存目录
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载缓存
        self.cache = self._load_cache()
        self.metadata = self._load_metadata()
        
        print(f"[DeepSeekCache] 缓存已加载：{len(self.cache)} 条记录")
    
    def _load_cache(self) -> Dict:
        """加载缓存数据"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[DeepSeekCache] 加载缓存失败：{e}")
                return {}
        return {}
    
    def _load_metadata(self) -> Dict:
        """加载元数据"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[DeepSeekCache] 加载元数据失败：{e}")
                return {}
        return {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "last_updated": None
        }
    
    def _save_cache(self):
        """保存缓存数据"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[DeepSeekCache] 保存缓存失败：{e}")
    
    def _save_metadata(self):
        """保存元数据"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[DeepSeekCache] 保存元数据失败：{e}")
    
    def _generate_key(self, query: str) -> str:
        """生成查询的 MD5 哈希键"""
        return hashlib.md5(query.encode('utf-8')).hexdigest()
    
    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """
        获取缓存的回复
        
        Args:
            query: 用户查询
            
        Returns:
            缓存的回复，如果不存在则返回 None
        """
        key = self._generate_key(query)
        
        if key in self.cache:
            cached_data = self.cache[key]
            print(f"[DeepSeekCache] ✓ 缓存命中：{query[:50]}...")
            
            # 更新统计
            self.metadata["cache_hits"] += 1
            self._save_metadata()
            
            return cached_data
        
        print(f"[DeepSeekCache] ✗ 缓存未命中：{query[:50]}...")
        self.metadata["cache_misses"] += 1
        self._save_metadata()
        return None
    
    def set(self, query: str, response: Dict[str, Any]):
        """
        缓存回复
        
        Args:
            query: 用户查询
            response: DeepSeek 的回复
        """
        key = self._generate_key(query)
        
        # 添加时间戳
        response["_cached_at"] = datetime.now().isoformat()
        
        self.cache[key] = response
        self._save_cache()
        
        # 更新统计
        self.metadata["total_requests"] += 1
        self.metadata["last_updated"] = datetime.now().isoformat()
        self._save_metadata()
        
        print(f"[DeepSeekCache] 💾 已缓存：{query[:50]}...")
    
    def clear(self):
        """清空缓存"""
        self.cache = {}
        self.metadata = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "last_updated": None
        }
        self._save_cache()
        self._save_metadata()
        print("[DeepSeekCache] 🗑️ 缓存已清空")
    
    def stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total = self.metadata["cache_hits"] + self.metadata["cache_misses"]
        hit_rate = (self.metadata["cache_hits"] / total * 100) if total > 0 else 0
        
        return {
            "total_requests": self.metadata["total_requests"],
            "cache_size": len(self.cache),
            "cache_hits": self.metadata["cache_hits"],
            "cache_misses": self.metadata["cache_misses"],
            "hit_rate": f"{hit_rate:.2f}%",
            "last_updated": self.metadata.get("last_updated", "N/A")
        }
    
    def remove(self, query: str) -> bool:
        """
        移除特定查询的缓存
        
        Args:
            query: 用户查询
            
        Returns:
            是否成功移除
        """
        key = self._generate_key(query)
        
        if key in self.cache:
            del self.cache[key]
            self._save_cache()
            print(f"[DeepSeekCache] 🗑️ 已移除缓存：{query[:50]}...")
            return True
        
        return False


# 全局缓存实例（单例模式）
_global_cache = None


def get_cache(cache_dir: str = "cache/deepseek") -> DeepSeekCache:
    """获取全局缓存实例"""
    global _global_cache
    if _global_cache is None:
        _global_cache = DeepSeekCache(cache_dir)
    return _global_cache


if __name__ == "__main__":
    # 测试缓存功能
    cache = get_cache()
    
    # 测试缓存
    test_query = "朱元璋出生时的家庭状况如何？"
    test_response = {
        "answer": "出生在赤贫家庭...",
        "retrieved_docs": [],
        "success": True
    }
    
    # 首次查询（未命中）
    result = cache.get(test_query)
    if result is None:
        print("缓存未命中，设置缓存...")
        cache.set(test_query, test_response)
    
    # 再次查询（应命中）
    result = cache.get(test_query)
    if result is not None:
        print(f"✓ 缓存命中：{result}")
    
    # 显示统计
    print("\n缓存统计:")
    for key, value in cache.stats().items():
        print(f"  {key}: {value}")
