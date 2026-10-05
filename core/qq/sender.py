"""QQSender：把 ``ReplyAction`` 翻译成 OneBot 消息并发送。

这是整个 core 里唯一允许出现 ``bot.send`` 的地方（发送胶水层），其它 core 模块
只负责产出 ``ReplyAction`` 数据。支持文本 / 引用回复 / 表情包 / 生图四种动作，
单个动作发送失败不影响其余动作，并打降级日志。
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)

from pathlib import Path
from typing import Iterable

from nonebot.adapters.onebot.v11 import Bot, Event, Message, MessageSegment

from config.qq import QQSettings
from core.image.service import ImageService
from core.qq.interactions import (
    ReplyAction,
    StickerCatalog,
    extract_message_id,
    parse_qq_face_segments,
)


class QQSender:
    def __init__(
        self,
        settings: QQSettings,
        sticker_catalog: StickerCatalog,
        image_service: ImageService | None = None,
        stream_pause: float = 0.0,
    ) -> None:
        self.settings = settings
        self.stickers = sticker_catalog
        self.image_service = image_service
        self.stream_pause = stream_pause

    async def send(
        self, bot: Bot, event: Event, actions: Iterable[ReplyAction]
    ) -> list[int | None]:
        """发送一组动作，按动作位置返回消息 id；失败或空动作位置为 ``None``。

        保留动作位置可让上层准确判断哪条消息实际发送成功。当
        ``stream_pause > 0`` 时，在连续两条消息之间按内容长度插入停顿，
        形成「换行分隔的流式发送」效果，而不是瞬间把所有消息一起发出。
        """
        sent: list[int | None] = []
        for index, action in enumerate(actions):
            # 流式间隔：首条不延迟；后续按本条文本长度停顿，模拟逐条打字
            if index > 0 and self.stream_pause > 0:
                text_len = len(action.text or "")
                pause = self.stream_pause + min(text_len / 500, 0.6)
                await asyncio.sleep(pause)
            try:
                sent.append(await self._send_one(bot, event, action))
            except Exception as exc:  # noqa: BLE001
                logger.error("发送动作失败：%s", exc)
                sent.append(None)
        return sent

    async def _send_one(self, bot: Bot, event: Event, action: ReplyAction) -> int | None:
        # 生图动作：调 ImageService 生成后再逐张发送
        if action.image_prompt and not action.image_path:
            if self.image_service is None:
                logger.warning("收到生图请求但 ImageService 未配置。")
                result = await bot.send(event, Message("生图服务暂未配置，当前无法生成图片。"))
                return extract_message_id(result)
            paths = await self.image_service.generate(action.image_prompt)
            last: int | None = None
            for index, path in enumerate(paths):
                reply_id = action.reply_message_id if index == 0 else None
                last = await self._send_image(bot, event, path, reply_id)
            return last

        if action.image_path:
            return await self._send_image(bot, event, action.image_path, action.reply_message_id)

        # 文本 / 引用 / 表情
        message = Message()
        if action.reply_message_id:
            message += MessageSegment.reply(action.reply_message_id)
        if action.sticker_path:
            message += MessageSegment.image(str(action.sticker_path))
        elif action.sticker_name:
            resolved = self.stickers.resolve(action.sticker_name)
            if resolved is not None:
                message += MessageSegment.image(str(resolved))
        if action.text:
            for segment_type, value in parse_qq_face_segments(action.text):
                if segment_type == "face":
                    message += MessageSegment.face(int(value))
                elif value:
                    message += MessageSegment.text(str(value))
        if not message:
            return None
        result = await bot.send(event, message)
        return extract_message_id(result)

    async def _send_image(
        self, bot: Bot, event: Event, path: Path, reply_id: int | None = None
    ) -> int | None:
        message = Message()
        if reply_id:
            message += MessageSegment.reply(reply_id)
        message += MessageSegment.image(str(path))
        result = await bot.send(event, message)
        return extract_message_id(result)
