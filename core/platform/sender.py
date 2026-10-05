"""平台发送器协议。"""

from __future__ import annotations

from typing import Iterable, Protocol

from core.platform.models import IncomingMessage
from core.qq.interactions import ReplyAction


class PlatformSender(Protocol):
    """把领域动作翻译为具体平台消息。"""

    async def send(
        self, message: IncomingMessage, actions: Iterable[ReplyAction]
    ) -> list[int | None]: ...

    async def send_notice(self, message: IncomingMessage, text: str) -> None: ...
