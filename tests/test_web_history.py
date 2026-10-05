"""Web 实际投递历史与断线恢复测试。"""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path

from core.platform.models import IncomingMessage
from core.qq.interactions import ReplyAction, StickerCatalog
from core.web.history import WebHistoryStore
from core.web.sender import WebSocketSender


class _ClosedSocket:
    async def send_json(self, payload):
        raise RuntimeError("connection is closing")


class _ImageService:
    def __init__(self, path: Path):
        self.path = path
        self.calls = 0

    async def generate(self, prompt):
        self.calls += 1
        return [self.path]


def _png_data_url(raw: bytes = b"fake-png") -> str:
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


def test_recent_two_rounds_include_text_and_images(tmp_path):
    store = WebHistoryStore(tmp_path / "history.db")
    store.append_user_message("alice", "turn-1", "第一问")
    store.append_event(
        username="alice", turn_id="turn-1", event_key="a1",
        role="assistant", kind="text", text="第一答"
    )
    store.append_user_message("alice", "turn-2", "第二问", [_png_data_url(b"u2")])
    bot_image = tmp_path / "bot.png"
    bot_image.write_bytes(b"bot-image")
    store.append_event(
        username="alice", turn_id="turn-2", event_key="a2",
        role="assistant", kind="image", image_path=bot_image
    )
    store.append_user_message("alice", "turn-3", "第三问")
    store.append_event(
        username="alice", turn_id="turn-3", event_key="a3",
        role="assistant", kind="text", text="第三答"
    )

    recent = store.recent_rounds("alice", rounds=2)
    assert {item["turn_id"] for item in recent} == {"turn-2", "turn-3"}
    assert [item["type"] for item in recent] == ["text", "image", "image", "text", "text"]
    assert all(item.get("data_url", "").startswith("data:image/") for item in recent if item["type"] == "image")
    assert store.recent_rounds("bob", rounds=2) == []


def test_event_key_is_idempotent_per_user(tmp_path):
    store = WebHistoryStore(tmp_path / "history.db")
    first = store.append_event(
        username="alice", turn_id="1", event_key="same",
        role="assistant", kind="text", text="原文"
    )
    second = store.append_event(
        username="alice", turn_id="1", event_key="same",
        role="assistant", kind="text", text="不应覆盖"
    )
    assert first == second
    # 没有用户事件时不构成可恢复轮次。
    assert store.recent_rounds("alice") == []


def test_sender_persists_command_and_generated_image_after_disconnect(tmp_path):
    history = WebHistoryStore(tmp_path / "history.db")
    history.append_user_message("alice", "turn-1", "画一张猫")
    generated = tmp_path / "generated.png"
    generated.write_bytes(b"generated-image")
    image_service = _ImageService(generated)
    sender = WebSocketSender(
        _ClosedSocket(), image_service, StickerCatalog(tmp_path), history,
        "alice", "turn-1"
    )
    message = IncomingMessage(
        platform="web", self_id="web:main", user_id="web:alice", text="画一张猫"
    )

    async def run():
        command_ids = await sender.send(message, [ReplyAction(text="已切换模式。")])
        image_ids = await sender.send(message, [ReplyAction(image_prompt="一只猫")])
        return command_ids, image_ids

    command_ids, image_ids = asyncio.run(run())
    assert command_ids[0] is not None and image_ids[0] is not None
    assert image_service.calls == 1
    recent = history.recent_rounds("alice")
    assert [item["type"] for item in recent] == ["text", "text", "image"]
    assert recent[1]["text"] == "已切换模式。"
    assert recent[2]["data_url"].startswith("data:image/png;base64,")
