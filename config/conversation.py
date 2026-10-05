"""对话业务相关的核心配置参数。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config.base import resolve_bot_path


@dataclass(frozen=True)
class ConversationSettings:
    max_context: int
    reply_delay_min: float
    reply_delay_max: float
    max_reply_chars: int
    max_reply_actions: int
    stream_pause: float       # 多条消息之间的流式间隔（秒）；<=0 表示不间隔（瞬间发完）
    max_bubble_chars: int     # 合并连续短行的最大气泡长度（字符）；<=0 表示不合并（每行一条气泡）
    context_timeout: int
    bot_output_prefix: str
    llm_echo_retry_context: int
    max_inference_workers: int
    max_pending_tasks: int
    works_state_path: Path   # writer「当前作品」选择状态文件（按作品隔离记忆用）

    @classmethod
    def from_config(cls, config: object, bot_root: Path) -> "ConversationSettings":
        return cls(
            max_context=int(getattr(config, "max_context", 20)),
            reply_delay_min=float(getattr(config, "reply_delay_min", 1)),
            reply_delay_max=float(getattr(config, "reply_delay_max", 4)),
            max_reply_chars=int(getattr(config, "max_reply_chars", 0)),
            max_reply_actions=int(getattr(config, "max_reply_actions", 3)),
            stream_pause=float(getattr(config, "stream_pause", 0.4)),
            max_bubble_chars=int(getattr(config, "max_bubble_chars", 200)),
            context_timeout=int(getattr(config, "context_timeout", 3600 * 2)),
            bot_output_prefix=str(getattr(config, "bot_output_prefix", "[BOT_OUTPUT] ")),
            llm_echo_retry_context=int(getattr(config, "llm_echo_retry_context", 6)),
            max_inference_workers=int(getattr(config, "max_inference_workers", 2)),
            max_pending_tasks=int(getattr(config, "max_pending_tasks", 20)),
            works_state_path=resolve_bot_path(
                bot_root, str(getattr(config, "works_state_path", "data/works_state.json"))
            ),
        )
