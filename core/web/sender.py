"""WebSocket 平台发送器：所有可恢复事件先落库，再尝试即时投递。"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Iterable

from starlette.websockets import WebSocket

from core.image.service import ImageService
from core.platform.models import IncomingMessage
from core.qq.interactions import ReplyAction, StickerCatalog, parse_qq_face_segments
from core.web.history import WebHistoryStore

logger = logging.getLogger(__name__)


class WebSocketSender:
    def __init__(
        self,
        websocket: WebSocket,
        image_service: ImageService,
        stickers: StickerCatalog,
        history: WebHistoryStore,
        username: str,
        turn_id: str,
        stream_pause: float = 0.0,
    ) -> None:
        self.websocket = websocket
        self.image_service = image_service
        self.stickers = stickers
        self.history = history
        self.username = username
        self.turn_id = turn_id
        self.stream_pause = stream_pause
        self._next_id = 0
        self._event_index = 0
        self._connected = True

    async def send(
        self, message: IncomingMessage, actions: Iterable[ReplyAction]
    ) -> list[int | None]:
        """生成结构化动作并持久化；连接关闭只影响即时推送。"""
        sent: list[int | None] = []
        for index, action in enumerate(actions):
            if index and self.stream_pause > 0:
                await asyncio.sleep(self.stream_pause)
            sent.append(await self._send_one(action))
        return sent

    async def send_notice(self, message: IncomingMessage, text: str) -> None:
        self._event_index += 1
        event_id = self.history.append_event(
            username=self.username,
            turn_id=self.turn_id,
            event_key=f"bot:{self.turn_id}:notice:{self._event_index}",
            role="notice",
            kind="text",
            text=text,
        )
        await self._send_json(
            {"type": "notice", "id": event_id, "text": text, "history_id": event_id}
        )

    async def stream_id(self) -> int:
        """为一条流式回复分配当前连接内消息 id（供实时气泡定位）。"""
        self._next_id += 1
        return self._next_id

    async def send_delta(self, message_id: int, text: str) -> None:
        """推送一段流式文本增量；失败时通知调度器停止后续即时写入。"""
        if not self._connected:
            raise RuntimeError("websocket closed")
        try:
            await self.websocket.send_json(
                {"type": "delta", "id": message_id, "text": text}
            )
        except Exception as exc:  # noqa: BLE001
            self._connected = False
            raise RuntimeError("websocket closed") from exc

    async def persist_stream(
        self, message: IncomingMessage, message_id: int, text: str
    ) -> int | None:
        """无论连接状态如何，都把完整流式正文写入历史。"""
        if not text:
            return None
        return self.history.append_event(
            username=self.username,
            turn_id=self.turn_id,
            event_key=f"bot:{self.turn_id}:stream",
            role="assistant",
            kind="text",
            text=text,
        )

    async def end_stream(self, message_id: int, history_id: int | None = None) -> None:
        """标记实时回复结束；历史行 ID 供前端把本地气泡与持久记录关联。"""
        if not self._connected:
            raise RuntimeError("websocket closed")
        try:
            payload = {"type": "delta_end", "id": message_id}
            if history_id is not None:
                payload["history_id"] = history_id
            await self.websocket.send_json(payload)
        except Exception as exc:  # noqa: BLE001
            self._connected = False
            raise RuntimeError("websocket closed") from exc

    async def _send_one(self, action: ReplyAction) -> int | None:
        paths: list[Path] = []
        if action.image_prompt and not action.image_path:
            paths = list(await self.image_service.generate(action.image_prompt))
        elif action.image_path:
            paths = [Path(action.image_path)]
        elif action.sticker_path:
            paths = [Path(action.sticker_path)]
        elif action.sticker_name:
            sticker = self.stickers.resolve(action.sticker_name)
            if sticker is not None:
                paths = [sticker]

        text = "".join(
            str(value)
            for kind, value in parse_qq_face_segments(action.text or "")
            if kind == "text"
        )
        if not text and action.text:
            text = action.text

        event_ids: list[int] = []
        if text or not paths:
            self._event_index += 1
            event_id = self.history.append_event(
                username=self.username,
                turn_id=self.turn_id,
                event_key=f"bot:{self.turn_id}:text:{self._event_index}",
                role="assistant",
                kind="text",
                text=text,
                reply_to=action.reply_message_id,
            )
            event_ids.append(event_id)
            await self._send_json(
                {
                    "type": "message",
                    "id": event_id,
                    "history_id": event_id,
                    "text": text,
                    "reply_to": action.reply_message_id,
                }
            )

        for path in paths:
            if not path.is_file():
                logger.warning("Web 待发送图片不存在：%s", path)
                continue
            self._event_index += 1
            event_id = self.history.append_event(
                username=self.username,
                turn_id=self.turn_id,
                event_key=f"bot:{self.turn_id}:image:{self._event_index}",
                role="assistant",
                kind="image",
                image_path=path,
                reply_to=action.reply_message_id,
            )
            event_ids.append(event_id)
            await self._send_json(
                {
                    "type": "image",
                    "id": event_id,
                    "history_id": event_id,
                    "data_url": self.history.path_to_data_url(path),
                    "reply_to": action.reply_message_id,
                }
            )
        return event_ids[0] if event_ids else None

    async def _send_json(self, payload: dict) -> bool:
        """尽力即时投递；预期的断线不向上抛，也不打印错误栈。"""
        if not self._connected:
            return False
        try:
            await self.websocket.send_json(payload)
            return True
        except Exception as exc:  # noqa: BLE001
            self._connected = False
            logger.info("Web 客户端已离线，事件已保存等待历史同步：%s", exc)
            return False
