"""
角色管理器 - StoryRag v2.0

职责：
1. 加载和管理角色配置
2. 将角色设定注入 Agent 系统消息（多层注入机制）
3. 在长对话中维护角色一致性（role anchoring）
4. 监控角色漂移并触发修正

角色控制逻辑不依赖单条 prompt：
- 第一层：结构化 RoleProfile 数据
- 第二层：build_system_prompt_segment() 生成系统消息片段
- 第三层：每 N 轮注入 role_reinforcement 防止漂移
- 第四层：开场和结束使用角色标志性用语
"""

from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

from .role_profile import RoleProfile, get_role, BUILT_IN_ROLES

logger = logging.getLogger(__name__)


class RoleManager:
    """
    角色管理器

    控制角色的加载、注入和一致性维护。
    实例化后可绑定到 Agent 的对话上下文中。
    """

    def __init__(self, role_id: str = "humorous_butler"):
        """
        初始化角色管理器

        Args:
            role_id: 角色标识符
        """
        self.role_id = role_id
        self.profile: RoleProfile = get_role(role_id)
        self._turn_counter: int = 0
        self._last_reinforcement_at: int = 0
        self._reinforcement_interval: int = 5

        logger.info(f"[RoleManager] 加载角色：{self.profile.display_name} (id={role_id})")

    def switch_role(self, role_id: str):
        """
        切换角色

        Args:
            role_id: 新角色的标识符
        """
        self.role_id = role_id
        self.profile = get_role(role_id)
        self._turn_counter = 0
        self._last_reinforcement_at = 0
        logger.info(f"[RoleManager] 切换角色：{self.profile.display_name}")

    def get_role_prompt_segment(self) -> str:
        """
        获取角色系统提示片段（第二层注入）

        此片段将被拼接进 Agent 的 system_message 中。
        由 RoleProfile.build_system_prompt_segment() 结构化生成。
        """
        return self.profile.build_system_prompt_segment()

    def get_opening_message(self, user_name: str = "主人") -> str:
        """获取角色开场消息"""
        return self.profile.build_opening_message(user_name)

    def on_turn_start(self) -> Optional[str]:
        """
        每轮对话开始时调用

        返回：
        - None: 无需额外操作
        - str: 需要注入的角色强化片段（第三层注入）
        """
        self._turn_counter += 1

        if self._turn_counter - self._last_reinforcement_at >= self._reinforcement_interval:
            self._last_reinforcement_at = self._turn_counter
            reinforcement = self.profile.build_role_reinforcement()
            logger.debug(f"[RoleManager] 第 {self._turn_counter} 轮注入角色强化：{reinforcement[:80]}...")
            return reinforcement

        return None

    def get_tone_message(self, scenario: str) -> str:
        """
        获取特定场景的语气变体（第四层注入）

        Args:
            scenario: 场景标识，如 'greeting', 'error', 'success', 'farewell'

        Returns:
            对应场景的语气消息，如果未配置则返回空字符串
        """
        return self.profile.tone_variants.get(scenario, "")

    def get_system_message_prefix(self) -> str:
        """
        生成完整的角色系统消息前缀

        此方法提供一种更高级的注入方式：
        将角色设定与任务指令分离，分别管理。
        """
        return f"[角色：{self.profile.display_name}]\n{self.get_role_prompt_segment()}"

    def wrap_answer(self, answer: str) -> str:
        """
        包装 Agent 的回答（可选的后处理）

        在回答末尾可以添加角色标志性用语。
        仅当回答长度适中且有合适位置时使用。

        Args:
            answer: Agent 原始回答

        Returns:
            包装后的回答
        """
        if not self.profile.signature_phrases or len(answer) < 50:
            return answer

        import random
        if random.random() < 0.3:
            phrase = random.choice(self.profile.signature_phrases)
            return f"{answer}\n\n*{phrase}*"

        return answer

    def detect_role_drift(self, answer: str) -> bool:
        """
        检测回答是否出现角色漂移（简单的关键字检测）

        检查是否出现了角色禁止的表述模式。

        Args:
            answer: Agent 的回答文本

        Returns:
            True 表示检测到角色漂移
        """
        drift_indicators = [
            "作为AI",
            "作为人工智能",
            "根据我的训练数据",
            "作为一个语言模型",
        ]
        for indicator in drift_indicators:
            if indicator in answer:
                logger.warning(f"[RoleManager] 检测到角色漂移：回答中包含 '{indicator}'")
                return True
        return False

    def to_config_dict(self) -> Dict[str, Any]:
        """导出当前角色配置字典"""
        return {
            "role_id": self.role_id,
            "profile": self.profile.to_dict(),
            "turn_counter": self._turn_counter,
            "reinforcement_interval": self._reinforcement_interval,
        }


# ============================================================
# 全局单例（默认角色）
# ============================================================

_default_role_manager: Optional[RoleManager] = None


def get_role_manager(role_id: str = None) -> RoleManager:
    """
    获取角色管理器实例

    Args:
        role_id: 角色标识符。如果不提供则返回默认实例
    """
    global _default_role_manager
    if role_id is None:
        if _default_role_manager is None:
            _default_role_manager = RoleManager("humorous_butler")
        return _default_role_manager
    return RoleManager(role_id)
