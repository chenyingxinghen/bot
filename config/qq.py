"""QQ 引用、表情包和戳一戳功能参数。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config.base import as_bool, resolve_bot_path


@dataclass(frozen=True)
class QQSettings:
    quote_history_size: int
    sticker_enabled: bool
    sticker_dir: Path
    sticker_prompt_limit: int
    poke_back: bool
    poke_text_reply: bool
    poke_cooldown: float

    @classmethod
    def from_config(cls, config: object, bot_root: Path) -> "QQSettings":
        return cls(
            quote_history_size=int(getattr(config, "quote_history_size", 12)),
            sticker_enabled=as_bool(getattr(config, "sticker_enabled", True), True),
            sticker_dir=resolve_bot_path(
                bot_root,
                str(getattr(config, "sticker_dir", "data/stickers")),
            ),
            sticker_prompt_limit=int(getattr(config, "sticker_prompt_limit", 30)),
            poke_back=as_bool(getattr(config, "poke_back", True), True),
            poke_text_reply=as_bool(getattr(config, "poke_text_reply", True), True),
            poke_cooldown=float(getattr(config, "poke_cooldown", 15)),
        )
