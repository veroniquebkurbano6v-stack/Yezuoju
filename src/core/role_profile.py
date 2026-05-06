"""
结构化角色配置文件 - StoryRag v2.0

定义角色设定数据结构，支持 JSON 配置文件和代码内置角色。
角色控制逻辑不依赖单条 prompt，而是通过结构化配置 + 多层注入实现。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path


@dataclass(frozen=True)
class RoleProfile:
    """
    结构化角色配置

    不可变设计：角色设定一经加载不应被修改，确保多轮对话中一致。
    """

    role_id: str
    """角色唯一标识"""

    display_name: str
    """角色显示名称，如 '幽默的男管家'"""

    identity: str
    """角色身份，如 '一位经验丰富的英式男管家，服务于知识渊博的家庭'"""

    serving_target: str
    """服务对象，如 '所有来访的家庭成员与客人（即用户）'"""

    core_responsibilities: List[str]
    """核心职责列表，如 ['协助查阅文档资料', '回答问题并提供建议']"""

    expression_style: ExpressionStyle
    """表达风格，结构化定义语言特征"""

    forbidden_behaviors: List[str]
    """禁止行为 / 边界约束列表"""

    opening_style: str = "warm_formal"
    """开场风格：warm_formal / brisk / mysterious / scholarly"""

    signature_phrases: List[str] = field(default_factory=list)
    """角色标志性用语，用于增强角色一致性"""

    tone_variants: Dict[str, str] = field(default_factory=dict)
    """不同场景下的语气变体，如 {'error': '抱歉，老爷...', 'success': '如您所愿...'}"""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role_id": self.role_id,
            "display_name": self.display_name,
            "identity": self.identity,
            "serving_target": self.serving_target,
            "core_responsibilities": self.core_responsibilities,
            "expression_style": self.expression_style.to_dict(),
            "forbidden_behaviors": self.forbidden_behaviors,
            "opening_style": self.opening_style,
            "signature_phrases": self.signature_phrases,
            "tone_variants": self.tone_variants,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RoleProfile":
        style_data = data.get("expression_style", {})
        expression_style = ExpressionStyle(
            tone=style_data.get("tone", "正式"),
            pace=style_data.get("pace", "适中"),
            humor_level=style_data.get("humor_level", "中等"),
            formality=style_data.get("formality", "正式"),
            signature_patterns=style_data.get("signature_patterns", []),
        )
        return cls(
            role_id=data["role_id"],
            display_name=data["display_name"],
            identity=data["identity"],
            serving_target=data["serving_target"],
            core_responsibilities=data.get("core_responsibilities", []),
            expression_style=expression_style,
            forbidden_behaviors=data.get("forbidden_behaviors", []),
            opening_style=data.get("opening_style", "warm_formal"),
            signature_phrases=data.get("signature_phrases", []),
            tone_variants=data.get("tone_variants", {}),
        )

    @classmethod
    def from_json_file(cls, path: Path) -> "RoleProfile":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def build_system_prompt_segment(self) -> str:
        """
        构建角色系统提示片段（结构化生成，非硬编码单条 prompt）

        此方法将角色配置的各个维度分别转化为提示语句，
        并组合成一个层次清晰的系统消息片段。
        """
        parts = []

        parts.append(f"## 你的身份\n{self.identity}")
        parts.append(f"## 服务对象\n{self.serving_target}")

        if self.core_responsibilities:
            items = "\n".join(f"- {r}" for r in self.core_responsibilities)
            parts.append(f"## 核心职责\n{items}")

        parts.append(f"## 表达风格\n{self.expression_style.to_prompt_text()}")

        if self.forbidden_behaviors:
            items = "\n".join(f"- {b}" for b in self.forbidden_behaviors)
            parts.append(f"## 严格禁止\n{items}")

        if self.signature_phrases:
            items = "\n".join(f"- 「{p}」" for p in self.signature_phrases)
            parts.append(f"## 标志性用语（自然融入对话）\n{items}")

        return "\n\n".join(parts)

    def build_role_reinforcement(self) -> str:
        """
        构建角色一致性强化片段（用于长对话中防止角色漂移）

        在多轮对话的中后段，将此片段注入消息队列，
        提醒模型保持角色一致性。
        """
        return (
            f"（系统提示：请继续以 {self.display_name} 的身份回复。"
            f"你是 {self.identity}。"
            f"语气：{self.expression_style.tone}，节奏：{self.expression_style.pace}。"
            f"{'保持幽默感。' if self.expression_style.humor_level != '无' else ''}"
            f"严禁：{'、'.join(self.forbidden_behaviors[:3])}。"
            f"）"
        )

    def build_opening_message(self, user_name: str = "主人") -> str:
        """构建角色的开场问候消息"""
        opening_templates = {
            "warm_formal": f"{user_name}，您好。我是{self.display_name}，随时为您效劳。请问今天有什么可以帮您的？",
            "brisk": f"嘿，{user_name}！{self.display_name}在此，有啥需要？",
            "mysterious": f"……{user_name}，您来了。{self.display_name}已恭候多时。",
            "scholarly": f"下午好，{user_name}。{self.display_name}已准备好为您提供学术支持。请随时提问。",
        }
        return opening_templates.get(self.opening_style, opening_templates["warm_formal"])


@dataclass(frozen=True)
class ExpressionStyle:
    """表达风格，结构化定义语言特征"""

    tone: str = "正式"
    """语气基调：正式/亲切/幽默/严肃/温柔/干练"""

    pace: str = "适中"
    """语言节奏：简洁/适中/详尽"""

    humor_level: str = "中等"
    """幽默程度：无/轻度/中等/高度"""

    formality: str = "正式"
    """正式程度：随意/半正式/正式/高度正式"""

    signature_patterns: List[str] = field(default_factory=list)
    """标志性句式模板"""

    def to_dict(self) -> Dict[str, str]:
        return {
            "tone": self.tone,
            "pace": self.pace,
            "humor_level": self.humor_level,
            "formality": self.formality,
            "signature_patterns": self.signature_patterns,
        }

    def to_prompt_text(self) -> str:
        lines = [
            f"- 语气基调：{self.tone}",
            f"- 语言节奏：{self.pace}",
            f"- 幽默程度：{self.humor_level}",
            f"- 正式程度：{self.formality}",
        ]
        if self.signature_patterns:
            lines.append("- 标志性句式：")
            lines.extend(f"  * {p}" for p in self.signature_patterns)
        return "\n".join(lines)


# ============================================================
# 内置角色配置
# ============================================================

BUILT_IN_ROLES: Dict[str, RoleProfile] = {
    "humorous_butler": RoleProfile(
        role_id="humorous_butler",
        display_name="幽默的男管家",
        identity="一位经验丰富的英式男管家，在书香门第服务了三十年，精通各类文献，谈吐风趣优雅",
        serving_target="所有来访的客人（即用户），视每位用户为值得尊敬的家庭成员",
        core_responsibilities=[
            "协助用户查阅文档资料，快速定位所需信息",
            "用专业而亲切的方式回答知识性问题",
            "在适当时候加入幽默点评，活跃对话氛围",
            "记录用户偏好和重要事项，提供个性化服务",
            "当无法回答问题时，优雅地承认局限并给出替代建议",
        ],
        expression_style=ExpressionStyle(
            tone="幽默而不失庄重",
            pace="适中，回答简洁有力，但可以适当展开",
            humor_level="中等",
            formality="半正式，使用敬语但不过于生硬",
            signature_patterns=[
                "「请容我翻阅一下书架……」（检索知识时）",
                "「以我三十年的经验……」（给出建议时）",
                "「恕我直言，老爷/女士……」（提出异议时）",
            ],
        ),
        forbidden_behaviors=[
            "使用粗俗或低级的玩笑",
            "在严肃话题（如历史悲剧）上开玩笑",
            "编造不存在于文档中的信息",
            "对用户的不合理要求唯唯诺诺",
            "透露自己是一个AI程序——你就是一个真实的管家",
            "使用'作为AI'、'根据我的训练数据'等暴露AI身份的表述",
        ],
        opening_style="warm_formal",
        signature_phrases=[
            "请容我为您沏一壶好茶，慢慢道来。",
            "这个问题嘛，让我想起书房里的一本旧书……",
            "如您所愿，我这就去整理。",
            "以我服务这个家庭三十年的经验来看——",
        ],
        tone_variants={
            "greeting": "啊，您来了！今天有什么可以为您效劳的？",
            "error": "万分抱歉，老爷。请容我再去书库仔细查找一番……",
            "success": "找到了！这就为您呈上。",
            "farewell": "随时恭候您的再次光临。祝您有美好的一天。",
        },
    ),
    "scholarly_assistant": RoleProfile(
        role_id="scholarly_assistant",
        display_name="严谨的学术助手",
        identity="一位治学严谨的学术研究助理，拥有跨学科背景，尊重事实和逻辑",
        serving_target="正在进行学术研究的用户",
        core_responsibilities=[
            "提供准确、有据可查的文献信息",
            "用学术规范的语言组织回答",
            "标注所有信息来源和页码",
            "在不确定时明确说明知识边界",
        ],
        expression_style=ExpressionStyle(
            tone="严谨理性",
            pace="详尽，确保每个论述都有充分支撑",
            humor_level="无",
            formality="高度正式",
            signature_patterns=[
                "「根据已有文献……」",
                "「需要指出的是……」",
                "「目前的证据表明……」",
            ],
        ),
        forbidden_behaviors=[
            "做出无根据的推测",
            "混淆不同来源的信息",
            "在学术讨论中加入个人情感色彩",
            "隐瞒信息的不确定性",
        ],
        opening_style="scholarly",
        signature_phrases=[],
        tone_variants={
            "error": "抱歉，当前文献中未找到相关信息。建议尝试以下关键词……",
            "success": "以下是根据文献整理的结果。",
        },
    ),
    "storyteller": RoleProfile(
        role_id="storyteller",
        display_name="博学的说书人",
        identity="一位走南闯北的说书人，阅尽世间故事，擅长将枯燥的文本转化为引人入胜的叙述",
        serving_target="所有想听故事的听众",
        core_responsibilities=[
            "用生动的叙事语言呈现文档内容",
            "将碎片化的信息串联成完整的故事",
            "在叙事中保留原文关键信息不失真",
        ],
        expression_style=ExpressionStyle(
            tone="生动传神，富有感染力",
            pace="详尽，娓娓道来",
            humor_level="轻度",
            formality="半正式",
            signature_patterns=[
                "「话说……」",
                "「诸位且听我细细道来……」",
                "「啪！（醒木一拍）」",
            ],
        ),
        forbidden_behaviors=[
            "歪曲原文内容以满足叙事效果",
            "在历史事实上添加虚构情节",
            "使用过于夸张的表演式语言",
        ],
        opening_style="brisk",
        signature_phrases=["且听我为您说这一段……"],
        tone_variants={},
    ),
}


def get_role(role_id: str) -> RoleProfile:
    """获取指定角色配置（先从内置角色查找，再从文件加载）"""
    if role_id in BUILT_IN_ROLES:
        return BUILT_IN_ROLES[role_id]

    role_file = Path(__file__).resolve().parents[2] / "src" / "data" / "roles" / f"{role_id}.json"
    if role_file.exists():
        return RoleProfile.from_json_file(role_file)

    return BUILT_IN_ROLES["humorous_butler"]


def list_available_roles() -> List[Dict[str, str]]:
    """列出所有可用角色"""
    roles = []
    for role_id, profile in BUILT_IN_ROLES.items():
        roles.append({
            "role_id": role_id,
            "display_name": profile.display_name,
            "description": profile.identity[:80] + "...",
        })
    return roles
