"""
Token 估算器 - StoryRag v2.0
使用 tiktoken 进行准确的 Token 计数
"""

import tiktoken
from typing import Dict


class TokenEstimator:
    """Token 估算器（单例模式）"""
    
    _encoders: Dict[str, tiktoken.Encoding] = {}
    
    @classmethod
    def get_encoder(cls, model_name: str = "gpt-3.5-turbo") -> tiktoken.Encoding:
        """
        获取指定模型的编码器
        
        Args:
            model_name: 模型名称
            
        Returns:
            tiktoken.Encoding 实例
        """
        if model_name not in cls._encoders:
            try:
                cls._encoders[model_name] = tiktoken.encoding_for_model(model_name)
            except KeyError:
                # 如果模型名称不在已知列表中，使用 gpt-3.5-turbo 的编码器
                cls._encoders[model_name] = tiktoken.get_encoding("cl100k_base")
        
        return cls._encoders[model_name]
    
    @classmethod
    def estimate_tokens(cls, text: str, model_name: str = "gpt-3.5-turbo") -> int:
        """
        估算文本的 Token 数
        
        Args:
            text: 要估算的文本
            model_name: 目标模型名称
            
        Returns:
            Token 数量
        """
        if not text:
            return 0
            
        encoder = cls.get_encoder(model_name)
        return len(encoder.encode(text))
    
    @classmethod
    def truncate_to_tokens(cls, text: str, max_tokens: int, model_name: str = "gpt-3.5-turbo") -> str:
        """
        将文本截断到指定的 Token 数
        
        Args:
            text: 原始文本
            max_tokens: 最大 Token 数
            model_name: 目标模型名称
            
        Returns:
            截断后的文本
        """
        encoder = cls.get_encoder(model_name)
        tokens = encoder.encode(text)
        
        if len(tokens) <= max_tokens:
            return text
        
        truncated_tokens = tokens[:max_tokens - 3]  # 预留 3 个 token 用于省略号
        return encoder.decode(truncated_tokens) + "..."
