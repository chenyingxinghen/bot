"""Web 实际投递历史：先持久化，再尝试通过 WebSocket 推送。

该存储与领域层 ``messages`` 表分离。领域表保存模型对话语义；这里保存浏览器
实际应展示的事件（用户文本/图片、Bot 文本、命令回复、通知和生成图片），用于
WebSocket 断线、页面刷新或服务重启后的最近两轮恢复。
"""

from __future__ import annotations

import base64
import hashlib
import mimetypes
import sqlite3
import time
from pathlib import Path
from typing import Iterable


class WebHistoryStore:
    """按 Web 用户和轮次持久化可展示事件。"""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.media_dir = self.db_path.parent / "web_history_media"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS web_history_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    event_key TEXT NOT NULL,
                    role TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    text TEXT NOT NULL DEFAULT '',
                    image_path TEXT NOT NULL DEFAULT '',
                    reply_to INTEGER,
                    sent_at INTEGER NOT NULL,
                    UNIQUE(username, event_key)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_web_history_user_turn
                ON web_history_events(username, turn_id, id)
                """
            )

    def append_event(
        self,
        *,
        username: str,
        turn_id: str,
        event_key: str,
        role: str,
        kind: str,
        text: str = "",
        image_path: str | Path = "",
        reply_to: int | None = None,
    ) -> int:
        """幂等写入一条事件并返回持久化行 ID。"""
        now = int(time.time() * 1000)
        path_value = str(Path(image_path).resolve()) if image_path else ""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO web_history_events
                    (username, turn_id, event_key, role, kind, text,
                     image_path, reply_to, sent_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    username,
                    turn_id,
                    event_key,
                    role,
                    kind,
                    text,
                    path_value,
                    reply_to,
                    now,
                ),
            )
            row = conn.execute(
                """
                SELECT id FROM web_history_events
                WHERE username = ? AND event_key = ?
                """,
                (username, event_key),
            ).fetchone()
        if row is None:  # pragma: no cover - SQLite 写入异常时的防御分支
            raise RuntimeError("Web history event could not be stored")
        return int(row["id"])

    def append_user_message(
        self,
        username: str,
        turn_id: str,
        text: str,
        images: Iterable[str] = (),
    ) -> list[int]:
        """接收入站消息后立即保存用户文本和上传图片。"""
        ids: list[int] = []
        if text:
            ids.append(
                self.append_event(
                    username=username,
                    turn_id=turn_id,
                    event_key=f"user:{turn_id}:text",
                    role="user",
                    kind="text",
                    text=text,
                )
            )
        for index, data_url in enumerate(images):
            path = self._save_data_url(data_url)
            if path is None:
                continue
            ids.append(
                self.append_event(
                    username=username,
                    turn_id=turn_id,
                    event_key=f"user:{turn_id}:image:{index}",
                    role="user",
                    kind="image",
                    image_path=path,
                )
            )
        return ids

    def recent_rounds(self, username: str, rounds: int = 2) -> list[dict]:
        """返回该用户最近 N 个用户轮次中的全部可展示事件。"""
        rounds = max(1, int(rounds))
        with self._connect() as conn:
            turn_rows = conn.execute(
                """
                SELECT turn_id, MAX(id) AS last_id
                FROM web_history_events
                WHERE username = ? AND role = 'user'
                GROUP BY turn_id
                ORDER BY last_id DESC
                LIMIT ?
                """,
                (username, rounds),
            ).fetchall()
            turn_ids = [str(row["turn_id"]) for row in reversed(turn_rows)]
            if not turn_ids:
                return []
            placeholders = ",".join("?" for _ in turn_ids)
            rows = conn.execute(
                f"""
                SELECT id, turn_id, role, kind, text, image_path, reply_to, sent_at
                FROM web_history_events
                WHERE username = ? AND turn_id IN ({placeholders})
                ORDER BY id ASC
                """,
                (username, *turn_ids),
            ).fetchall()
        return [event for row in rows if (event := self._serialize(row)) is not None]

    def _serialize(self, row: sqlite3.Row) -> dict | None:
        event = {
            "id": int(row["id"]),
            "turn_id": str(row["turn_id"]),
            "type": str(row["kind"]),
            "role": str(row["role"]),
            "sent_at": int(row["sent_at"]),
        }
        if row["reply_to"] is not None:
            event["reply_to"] = int(row["reply_to"])
        if row["kind"] == "image":
            path = Path(str(row["image_path"]))
            if not path.is_file():
                return None
            event["data_url"] = self.path_to_data_url(path)
        else:
            event["text"] = str(row["text"])
        return event

    def _save_data_url(self, data_url: str) -> Path | None:
        if not data_url.startswith("data:image/") or "," not in data_url:
            return None
        header, encoded = data_url.split(",", 1)
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError):
            return None
        if not raw:
            return None
        mime = header[5:].split(";", 1)[0].lower()
        extension = mimetypes.guess_extension(mime) or ".img"
        digest = hashlib.sha256(raw).hexdigest()
        self.media_dir.mkdir(parents=True, exist_ok=True)
        path = self.media_dir / f"{digest}{extension}"
        if not path.exists():
            path.write_bytes(raw)
        return path

    @staticmethod
    def path_to_data_url(path: Path) -> str:
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"
