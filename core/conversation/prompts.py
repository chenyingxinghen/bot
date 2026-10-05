"""system prompt 片段构造。

把「角色人设 / 长期记忆 / 时间感知 / QQ 控制标记 / 生图指令」等零散片段拼成
一段完整的 system prompt，供 ``ConversationService`` 使用。所有函数纯函数、无副作用。
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

from core.character.client import CharacterCard, build_persona_prompt
from core.modes.manager import Mode


def build_time_prompt(now_str: str, gap_str: str) -> str:
    """当前时间与距上次对话间隔的中文描述。"""
    base = f"当前时间：{now_str}。" if now_str else ""
    return (base + gap_str).strip()


def build_memory_section(memory_text: str) -> str:
    if not memory_text or not memory_text.strip():
        return ""
    return "【长期记忆】\n" + memory_text.strip()


def build_persona_section(
    mode: Mode, card: CharacterCard | None, user_name: str, context_text: str | None = None
) -> str:
    """按“模式规则 → 当前角色资料”组装身份指令。

    模式 prompt 定义稳定的行为边界；角色卡只补充当前身份、性格、场景和命中的
    世界书，避免有角色卡时反而丢掉模式级规则。
    """
    parts = [mode.prompt.strip()]
    if mode.key == "tavern" and card is not None:
        parts.append(build_persona_prompt(card, user_name, context_text))
    return "\n\n".join(part for part in parts if part)


def build_qq_control_prompt(
    sticker_enabled: bool,
    *,
    qq_enabled: bool = True,
    allow_quote: bool = False,
    image_enabled: bool = False,
) -> str:
    """只描述本轮实际可用的输出控制协议，避免重复或无效指令。"""
    lines = ["【可选输出控制】以下是可用的输出控制协议，当你需要使用某项功能时，请使用对应标记；控制标记放在对应内容行开头："]
    if qq_enabled and allow_quote:
        lines.append(
            "- [回复: mN]：引用历史消息中对应的 [mN]；每条回复至多引用一个编号。"
        )
        lines.append(
            "- 带 [mN] 前缀的历史消息可以被引用，但裸 [mN] 只是只读定位标签，"
            "禁止复制、复述或直接输出；需要引用时只能使用上一条所示的完整控制标记。"
        )
    if qq_enabled:
        lines.append("- [QQ表情: 名称]：插入 QQ 原生表情，如 [QQ表情: 微笑]。")
    if qq_enabled and sticker_enabled:
        lines.append("- [表情: 表情包名]：发送本地表情包。")
    if image_enabled:
        lines.append(
            "- [生图: 生图提示词]：生成并发送图片。当用户提到生成、绘制、拍摄图片、发送照片等时，"
            "必须输出一行此标记：[生图: 生图提示词]，并写出完整、具体的画面描述；"
            "一律不得出现任何角色名称或人名（含“当前角色”之类的字样），直接描述外貌、衣着、动作、场景与光线，不叙述情节。"
        )
    return "\n".join(lines)


def assemble_system_prompt(
    *,
    persona: str,
    memory: str = "",
    time_prompt: str = "",
    qq_controls: str | None = None,
    image_tool: bool | None = None,
) -> str:
    """按稳定优先级拼装 system prompt；空片段自动跳过。

    ``image_tool`` 仅为向后兼容保留。生图协议由 ``qq_controls`` 统一说明，避免同一
    指令在模式 prompt、QQ 控制段和工具段重复出现。
    """
    parts: list[str] = [persona]
    memory_block = build_memory_section(memory)
    if memory_block:
        parts.append(memory_block)
    if time_prompt:
        parts.append("【时间】\n" + time_prompt)
    if qq_controls:
        parts.append(qq_controls)
    return "\n\n".join(p.strip() for p in parts if p and p.strip())
