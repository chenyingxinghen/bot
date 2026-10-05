"""OneBot/NapCat 与平台抽象层之间的适配器。"""

from __future__ import annotations

from typing import Iterable

from nonebot.adapters.onebot.v11 import Bot, Event, Message

from core.image.extract import _extract_images
from core.platform.models import IncomingMessage
from core.qq.interactions import ReplyAction
from core.qq.sender import QQSender


async def incoming_from_private_event(event: Event) -> IncomingMessage:
    """将 OneBot 私聊事件转换为平台无关消息，保留历史身份键。"""
    self_id = str(getattr(event, "self_id"))
    user_id = str(getattr(event, "user_id"))
    message_id = getattr(event, "message_id", None)
    return IncomingMessage(
        platform="qq",
        self_id=self_id,
        user_id=user_id,
        text=str(event.get_plaintext()).strip(),
        images=tuple(await _extract_images(event)),
        message_id=str(message_id) if message_id is not None else None,
        conversation_id=user_id,
        native_event=event,
    )


class QQPlatformSender:
    """绑定本次 OneBot Bot 实例，把通用发送协议委托给 QQSender。"""

    def __init__(self, bot: Bot, qq_sender: QQSender) -> None:
        self.bot = bot
        self.qq_sender = qq_sender

    async def send(
        self, message: IncomingMessage, actions: Iterable[ReplyAction]
    ) -> list[int | None]:
        if message.native_event is None:
            raise ValueError("QQ 消息缺少原始 OneBot 事件")
        return await self.qq_sender.send(self.bot, message.native_event, actions)

    async def send_notice(self, message: IncomingMessage, text: str) -> None:
        if message.native_event is None:
            raise ValueError("QQ 消息缺少原始 OneBot 事件")
        await self.bot.send(message.native_event, Message(text))
