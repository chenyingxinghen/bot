"""「当前作品」选择：为 (self_id, peer_id) 记住 writer 模式下正在写哪部作品。

用于记忆的命名空间隔离——不同作品的剧情 / 设定 / 写作互动互不串味。
状态持久化到 JSON，与 ``CharacterSelection`` 同构。默认空串表示「尚未选定作品」
（此时 writer 记忆不分区，行为退化为旧逻辑）。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from core.state import atomic_write_json

logger = logging.getLogger(__name__)


class WorkSelection:
    """为 (self_id, peer_id) 记忆「当前作品标题」，持久化到 JSON。"""

    def __init__(self, state_path: str | Path, default: str = "") -> None:
        self._state_path = Path(state_path)
        self._default = default
        self._state: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if self._state_path.exists():
            try:
                self._state = json.loads(
                    self._state_path.read_text(encoding="utf-8")
                )
            except Exception:
                self._state = {}

    def _save(self) -> None:
        try:
            atomic_write_json(self._state_path, self._state)
        except Exception as exc:  # noqa: BLE001
            logger.warning("保存当前作品选择失败：%s", exc)

    def get(self, key_pair: tuple[str, str]) -> str:
        return self._state.get(":".join(key_pair), self._default)

    def set(self, key_pair: tuple[str, str], work: str) -> None:
        self._state[":".join(key_pair)] = work
        self._save()
