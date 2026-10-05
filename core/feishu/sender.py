"""飞书平台发送器。SDK 仅在启用飞书入口时才按需导入。"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Iterable

from core.feishu.adapter import FeishuContext
from core.image.service import ImageService
from core.platform.identity import stable_message_id
from core.platform.models import IncomingMessage
from core.qq.interactions import ReplyAction, StickerCatalog, parse_qq_face_segments

logger = logging.getLogger(__name__)


class FeishuSender:
    def __init__(
        self,
        client: object,
        image_service: ImageService,
        stickers: StickerCatalog,
        *,
        stream_pause: float = 0.0,
    ) -> None:
        self.client = client
        self.image_service = image_service
        self.stickers = stickers
        self.stream_pause = stream_pause
        self._reply_map: dict[int, str] = {}

    async def send(
        self, message: IncomingMessage, actions: Iterable[ReplyAction]
    ) -> list[int | None]:
        sent: list[int | None] = []
        for index, action in enumerate(actions):
            if index and self.stream_pause > 0:
                await asyncio.sleep(self.stream_pause)
            sent.append(await self._send_one(message, action))
        return sent

    async def send_notice(self, message: IncomingMessage, text: str) -> None:
        await self._reply_text(message, text, None)

    async def _send_one(
        self, message: IncomingMessage, action: ReplyAction
    ) -> int | None:
        text = "".join(
            str(value) for kind, value in parse_qq_face_segments(action.text or "")
            if kind == "text"
        ) or (action.text or "")
        target_native = self._reply_map.get(action.reply_message_id or 0)
        if text:
            return await self._reply_text(message, text, target_native)

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
        last: int | None = None
        for path in paths:
            last = await self._reply_image(message, path, target_native)
        return last

    async def _reply_text(
        self, message: IncomingMessage, text: str, target_native: str | None
    ) -> int | None:
        lark = _load_lark()
        native_id = target_native or _incoming_native_id(message)
        request = (
            lark.im.v1.ReplyMessageRequest.builder()
            .message_id(native_id)
            .request_body(
                lark.im.v1.ReplyMessageRequestBody.builder()
                .content(json.dumps({"text": text}, ensure_ascii=False))
                .msg_type("text")
                .build()
            )
            .build()
        )
        response = await asyncio.to_thread(self.client.im.v1.message.reply, request)
        return self._remember_response(response)

    async def _reply_image(
        self, message: IncomingMessage, path: Path, target_native: str | None
    ) -> int | None:
        lark = _load_lark()
        upload = (
            lark.im.v1.CreateImageRequest.builder()
            .request_body(
                lark.im.v1.CreateImageRequestBody.builder()
                .image_type("message")
                .image(path.open("rb"))
                .build()
            )
            .build()
        )
        try:
            uploaded = await asyncio.to_thread(self.client.im.v1.image.create, upload)
        finally:
            image = getattr(getattr(upload, "request_body", None), "image", None)
            if image is not None:
                image.close()
        if not uploaded.success():
            raise RuntimeError(f"飞书图片上传失败：{uploaded.code} {uploaded.msg}")
        image_key = uploaded.data.image_key
        native_id = target_native or _incoming_native_id(message)
        request = (
            lark.im.v1.ReplyMessageRequest.builder()
            .message_id(native_id)
            .request_body(
                lark.im.v1.ReplyMessageRequestBody.builder()
                .content(json.dumps({"image_key": image_key}))
                .msg_type("image")
                .build()
            )
            .build()
        )
        response = await asyncio.to_thread(self.client.im.v1.message.reply, request)
        return self._remember_response(response)

    def _remember_response(self, response: object) -> int | None:
        if not response.success():
            raise RuntimeError(f"飞书回复失败：{response.code} {response.msg}")
        native_id = str(getattr(response.data, "message_id", "") or "")
        if not native_id:
            return None
        internal_id = stable_message_id("feishu", native_id)
        self._reply_map[internal_id] = native_id
        if len(self._reply_map) > 500:
            self._reply_map.pop(next(iter(self._reply_map)))
        return internal_id


def _incoming_native_id(message: IncomingMessage) -> str:
    event = message.native_event if isinstance(message.native_event, dict) else {}
    context = event.get("feishu")
    if isinstance(context, FeishuContext):
        return context.native_message_id
    if message.message_id:
        return message.message_id
    raise ValueError("飞书消息缺少 message_id")


def _load_lark():
    try:
        import lark_oapi as lark
    except ImportError as exc:
        raise RuntimeError("启用飞书入口需要安装 lark-oapi") from exc
    return lark
