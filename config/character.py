"""酒馆模式和角色卡相关参数。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config.base import resolve_bot_path


@dataclass(frozen=True)
class CharacterSettings:
    sillytavern_data_root: Path
    sillytavern_user: str
    sillytavern_default_character: str
    sillytavern_user_name: str
    sillytavern_url: str
    sillytavern_source: str
    sillytavern_reverse_proxy: str
    sillytavern_model: str
    sillytavern_api_key: str
    tavern_character_state_path: Path

    @classmethod
    def from_config(cls, config: object, bot_root: Path) -> "CharacterSettings":
        return cls(
            sillytavern_data_root=resolve_bot_path(
                bot_root,
                str(getattr(
                    config, "sillytavern_data_root",
                    "G:/git_proj/SillyTavern/data",
                )),
            ),
            sillytavern_user=str(getattr(config, "sillytavern_user", "default-user")),
            sillytavern_default_character=str(getattr(
                config, "sillytavern_default_character", "auto")),
            sillytavern_user_name=str(getattr(config, "sillytavern_user_name", "你")),
            sillytavern_url=str(getattr(config, "sillytavern_url", "")).rstrip("/"),
            sillytavern_source=str(getattr(config, "sillytavern_source", "openai")),
            sillytavern_reverse_proxy=str(getattr(
                config, "sillytavern_reverse_proxy",
                "http://127.0.0.1:11434/v1",
            )),
            sillytavern_model=str(getattr(config, "sillytavern_model", "")),
            sillytavern_api_key=str(getattr(config, "sillytavern_api_key", "")),
            tavern_character_state_path=resolve_bot_path(
                bot_root,
                str(getattr(
                    config, "tavern_character_state_path",
                    "data/tavern_characters.json",
                )),
            ),
        )
