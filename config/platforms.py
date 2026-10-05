"""Web 与飞书入口配置。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config.base import as_bool, resolve_bot_path


@dataclass(frozen=True)
class WebSettings:
    enabled: bool
    path: str
    token: str  # 会话令牌签名密钥（同时作为启用前的必填项）
    accounts_file: Path  # 账号存储文件
    history_db: Path  # Web 实际投递历史（文本、命令回复和图片）
    max_accounts: int  # 全局最大账号数；个人 BOT 默认 1（仅允许注册一个账号）
    static_dir: Path
    stream: bool

    @classmethod
    def from_config(cls, config: object, bot_root: Path) -> "WebSettings":
        path = "/" + str(getattr(config, "web_path", "chat")).strip("/")
        return cls(
            enabled=as_bool(getattr(config, "web_enabled", False), False),
            path=path,
            token=str(getattr(config, "web_token", "")).strip(),
            accounts_file=resolve_bot_path(
                bot_root,
                str(getattr(config, "web_accounts_file", "data/web_accounts.json")),
            ),
            history_db=resolve_bot_path(
                bot_root,
                str(getattr(config, "web_history_db", "data/web_history.db")),
            ),
            max_accounts=int(getattr(config, "web_max_accounts", 1) or 1),
            static_dir=resolve_bot_path(
                bot_root, str(getattr(config, "web_static_dir", "core/web/static"))
            ),
            stream=as_bool(getattr(config, "web_stream", True), True),
        )


@dataclass(frozen=True)
class FeishuSettings:
    enabled: bool
    app_id: str
    app_secret: str
    bot_id: str

    @classmethod
    def from_config(cls, config: object) -> "FeishuSettings":
        return cls(
            enabled=as_bool(getattr(config, "feishu_enabled", False), False),
            app_id=str(getattr(config, "feishu_app_id", "")).strip(),
            app_secret=str(getattr(config, "feishu_app_secret", "")).strip(),
            bot_id=str(getattr(config, "feishu_bot_id", "main")).strip() or "main",
        )


@dataclass(frozen=True)
class PlatformSettings:
    web: WebSettings
    feishu: FeishuSettings

    @classmethod
    def from_config(cls, config: object, bot_root: Path) -> "PlatformSettings":
        return cls(
            web=WebSettings.from_config(config, bot_root),
            feishu=FeishuSettings.from_config(config),
        )
