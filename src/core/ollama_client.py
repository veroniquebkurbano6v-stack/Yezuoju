"""
Ollama 本地模型客户端 - 角色强化处理

职责：
1. 与本地 Ollama 服务通信
2. 将 DeepSeek Agent 生成的回复通过 Ollama 模型进行角色强化
3. 保留原始含义，增强角色设定（尤其是管家角色）
4. 处理超时、不可用等降级场景
"""

from __future__ import annotations

import os
import logging
from typing import Optional, Dict
from openai import OpenAI

logger = logging.getLogger(__name__)


_ROLE_ENHANCE_PROMPTS: Dict[str, str] = {
    "humorous_butler": """你是一个语言润色专家。请对以下管家回复进行角色强化润色，要求：

1. **严格保留原文所有的信息、事实、数据、文档引用**——不得增删改任何实质性内容
2. 增强管家语气：优雅、幽默、略带英式绅士风范
3. 自然融入管家式敬语和谦辞，但不做作
4. 可适当加入管家标志性用语（如"请容我……""老爷/女士""为您效劳"等）
5. 保持原文长度基本一致，不额外扩展

原文：
{original_response}

请输出润色后的管家回复：""",

    "scholarly_assistant": """你是一个语言润色专家。请对以下学术助手回复进行角色强化润色，要求：

1. **严格保留原文所有的信息、事实、数据、文档引用**——不得增删改任何实质性内容
2. 增强学术严谨语气
3. 使用更规范的学术表达
4. 保持原文长度基本一致

原文：
{original_response}

请输出润色后的回复：""",

    "storyteller": """你是一个语言润色专家。请对以下说书人回复进行角色强化润色，要求：

1. **严格保留原文所有的信息、事实、数据、文档引用**——不得增删改任何实质性内容
2. 增强说书人的叙事感染力
3. 添加适当的说书人语气词
4. 保持原文长度基本一致

原文：
{original_response}

请输出润色后的回复：""",
}


class OllamaRoleEnhancer:
    """Ollama 模型角色强化处理器"""

    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        self.model = os.getenv("OLLAMA_ROLE_ENHANCE_MODEL", "qwen3:8b")
        self.temperature = float(os.getenv("OLLAMA_ROLE_ENHANCE_TEMPERATURE", "0.3"))
        self.max_tokens = int(os.getenv("OLLAMA_ROLE_ENHANCE_MAX_TOKENS", "400"))
        self.timeout = float(os.getenv("OLLAMA_ROLE_ENHANCE_TIMEOUT", "30"))

        self._client: Optional[OpenAI] = None
        self._available: Optional[bool] = None

        logger.info(f"[OllamaRoleEnhancer] 初始化：model={self.model}, base_url={self.base_url}")

    @property
    def client(self) -> Optional[OpenAI]:
        """延迟初始化 OpenAI 兼容客户端"""
        if self._client is None:
            try:
                self._client = OpenAI(
                    base_url=self.base_url,
                    api_key="ollama",
                    timeout=self.timeout,
                )
                self._available = True
                logger.info("[OllamaRoleEnhancer] 客户端连接成功")
            except Exception as e:
                logger.warning(f"[OllamaRoleEnhancer] 客户端初始化失败：{e}")
                self._available = False
        return self._client

    def is_available(self) -> bool:
        """检查 Ollama 服务是否可用"""
        if self._available is not None:
            return self._available
        try:
            if self.client is None:
                return False
            self.client.models.list()
            self._available = True
            return True
        except Exception:
            self._available = False
            return False

    def enhance_response(self, original_response: str, role_id: str) -> str:
        """
        通过 Ollama 模型增强回复的角色设定

        Args:
            original_response: DeepSeek Agent 的原始回复
            role_id: 角色标识符

        Returns:
            角色强化后的回复（如果 Ollama 不可用则返回原文）
        """
        if not self.is_available():
            logger.info("[OllamaRoleEnhancer] Ollama 不可用，返回原始回复")
            return original_response

        if len(original_response) < 20:
            return original_response

        prompt_template = _ROLE_ENHANCE_PROMPTS.get(
            role_id,
            _ROLE_ENHANCE_PROMPTS["humorous_butler"]
        )
        prompt = prompt_template.format(original_response=original_response)

        try:
            logger.info(f"[OllamaRoleEnhancer] 开始角色强化，role_id={role_id}")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            enhanced = response.choices[0].message.content.strip()
            enhanced = self._clean_output(enhanced)

            if len(enhanced) < len(original_response) * 0.3:
                logger.warning("[OllamaRoleEnhancer] 强化后文本过短，返回原文")
                return original_response

            logger.info(f"[OllamaRoleEnhancer] 角色强化完成，原文{len(original_response)}字 → 强化后{len(enhanced)}字")
            return enhanced

        except Exception as e:
            logger.warning(f"[OllamaRoleEnhancer] 角色强化失败：{e}")
            return original_response

    def _clean_output(self, text: str) -> str:
        """清理 Ollama 输出中的多余标记"""
        for prefix in ["润色后的管家回复：", "润色后的回复：", "输出："]:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        return text.strip()


_enhancer_instance: Optional[OllamaRoleEnhancer] = None


def get_ollama_enhancer() -> OllamaRoleEnhancer:
    """获取 Ollama 角色强化器单例"""
    global _enhancer_instance
    if _enhancer_instance is None:
        _enhancer_instance = OllamaRoleEnhancer()
    return _enhancer_instance
