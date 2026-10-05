"""飞书事件到统一入站消息的转换。"""

from __future__ import annotations

import json
from dataclasses import dataclass

from core.platform.identity import stable_message_id
from core.platform.models import IncomingMessage


@dataclass(frozen=True)
class FeishuContext:
    native_message_id: str
    chat_id: str


def incoming_from_event(data: object, *, bot_id: str) -> IncomingMessage | None:
    event = getattr(data, "event", None)
    message = getattr(event, "message", None)
    sender = getattr(event, "sender", None)
    sender_id = getattr(sender, "sender_id", None)
    open_id = str(getattr(sender_id, "open_id", "") or "").strip()
    native_id = str(getattr(message, "message_id", "") or "").strip()
    chat_id = str(getattr(message, "chat_id", "") or "").strip()
    msg_type = str(getattr(message, "message_type", "") or "").strip()
    if not open_id or not native_id:
        return None

    content_raw = str(getattr(message, "content", "") or "")
    try:
        content = json.loads(content_raw) if content_raw else {}
    except json.JSONDecodeError:
        content = {}
    text = str(content.get("text", "")).strip() if msg_type == "text" else ""
    if msg_type not in {"text"}:
        text = f"[暂不支持的飞书消息类型:{msg_type or 'unknown'}]"

    internal_id = stable_message_id("feishu", native_id)
    return IncomingMessage(
        platform="feishu",
        self_id=f"feishu:{bot_id}",
        user_id=f"feishu:{open_id}",
        text=text,
        message_id=native_id,
        conversation_id=chat_id or open_id,
        native_event={"message_id": internal_id, "feishu": FeishuContext(native_id, chat_id)},
    )
