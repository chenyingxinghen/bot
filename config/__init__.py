"""单一配置归宿，负责组装所有领域配置。"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from config.character import CharacterSettings
from config.conversation import ConversationSettings
from config.image import ImageSettings
from config.llm import LLMSettings
from config.memory import MemorySettings
from config.modes import ModesSettings
from config.platforms import PlatformSettings
from config.qq import QQSettings


@dataclass(frozen=True)
class AppConfig:
    llm: LLMSettings
    memory: MemorySettings
    qq: QQSettings
    image: ImageSettings
    character: CharacterSettings
    modes: ModesSettings
    conversation: ConversationSettings
    platforms: PlatformSettings


class _ConfigOverlay:
    """把系统环境变量叠加到 NoneBot 配置上。

    NoneBot 的基础 Config 模型不会保留未声明的自定义字段，而项目业务参数都来自
    `.env/.env.prod`。这里统一按“环境变量优先、driver.config 兜底”读取，避免各领域
    配置各自调用 os.getenv，也保证 `.env.prod` 的 override 语义真实生效。
    """

    def __init__(self, base: object) -> None:
        self._base = base

    def __getattr__(self, name: str):
        env_name = name.upper()
        if env_name in os.environ:
            return os.environ[env_name]
        return getattr(self._base, name)


def build_app_config(driver_config: object, bot_root: Path) -> AppConfig:
    """组合各个子配置，构建统一的 AppConfig；环境变量覆盖 NoneBot 默认配置。"""
    config = _ConfigOverlay(driver_config)
    return AppConfig(
        llm=LLMSettings.from_config(config),
        memory=MemorySettings.from_config(config, bot_root),
        qq=QQSettings.from_config(config, bot_root),
        image=ImageSettings.from_config(config, bot_root),
        character=CharacterSettings.from_config(config, bot_root),
        modes=ModesSettings.from_config(config, bot_root),
        conversation=ConversationSettings.from_config(config, bot_root),
        platforms=PlatformSettings.from_config(config, bot_root),
    )
