"""各模块共享的轻量配置解析工具。"""

from __future__ import annotations

from pathlib import Path


def resolve_bot_path(bot_root: Path, value: str | Path) -> Path:
    """把配置中的相对路径统一解析到 bot 根目录。"""
    path = Path(value).expanduser()
    return path if path.is_absolute() else bot_root / path


def as_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
