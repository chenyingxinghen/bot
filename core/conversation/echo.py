"""历史回显检测与重试策略。

模型偶尔会把上下文里的历史对话原样复述回来当作回复，这种情况要在发送前过滤掉，
必要时触发一次重试。逻辑核心在 ``core.llm.reply._filter_history_echo``，这里只做
薄封装与重试决策。
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

from core.llm.reply import _filter_history_echo


def filter_history_echo(raw_reply: str, ctx: list[dict]) -> tuple[str, int]:
    """过滤掉复述历史的文本，返回 (清洗后文本, 命中历史条数)。"""
    return _filter_history_echo(raw_reply, ctx)


def should_retry(matched_count: int, echo_retry_context: int) -> bool:
    """命中条数较多且配置允许重试时返回 True。"""
    return matched_count >= 2 and echo_retry_context > 0
