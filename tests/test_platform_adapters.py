from __future__ import annotations

import json
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from config.platforms import WebSettings
from core.feishu.adapter import FeishuContext, incoming_from_event
from core.image.service import ImageService
from core.platform.identity import stable_message_id
from core.qq.interactions import StickerCatalog
from core.web.app import install_web_routes


class FakeConversationService:
    def recent_message_window(self, self_id, user_id, rounds=2):
        return [
            {"id": 10, "role": "user", "text": "历史问题", "sent_at": 1, "mode": "clone"},
            {"id": 11, "role": "assistant", "text": "历史回复", "sent_at": 2, "mode": "clone"},
        ]


class FakeDispatcher:
    def __init__(self):
        self.messages = []
        self.conversation_service = FakeConversationService()

    async def dispatch(self, message, sender):
        self.messages.append(message)
        await sender.send_notice(message, "已收到")
        return True

    async def dispatch_stream(self, message, sender):
        self.messages.append(message)
        await sender.send_notice(message, "已收到")
        return True


class FakeImageService:
    async def generate(self, prompt):
        return []


def test_stable_message_id_is_repeatable_and_positive():
    first = stable_message_id("feishu", "om_123")
    assert first == stable_message_id("feishu", "om_123")
    assert first > 0
    assert first != stable_message_id("feishu", "om_456")


def test_feishu_text_event_to_incoming_message():
    data = SimpleNamespace(
        event=SimpleNamespace(
            sender=SimpleNamespace(sender_id=SimpleNamespace(open_id="ou_user")),
            message=SimpleNamespace(
                message_id="om_msg",
                chat_id="oc_chat",
                message_type="text",
                content=json.dumps({"text": "你好"}, ensure_ascii=False),
            ),
        )
    )
    message = incoming_from_event(data, bot_id="main")
    assert message is not None
    assert message.self_id == "feishu:main"
    assert message.user_id == "feishu:ou_user"
    assert message.text == "你好"
    assert message.native_event["message_id"] == stable_message_id("feishu", "om_msg")
    assert isinstance(message.native_event["feishu"], FeishuContext)


def test_web_routes_require_login_and_accept_message(tmp_path):
    source = __import__("pathlib").Path(__file__).resolve().parents[1] / "core" / "web" / "static"
    settings = WebSettings(
        enabled=True,
        path="/chat",
        token="secret-token",
        accounts_file=tmp_path / "web_accounts.json",
        history_db=tmp_path / "web_history.db",
        max_accounts=1,
        static_dir=source,
        stream=True,
    )
    app = FastAPI()
    dispatcher = FakeDispatcher()
    install_web_routes(
        app,
        settings,
        dispatcher,
        FakeImageService(),
        StickerCatalog(tmp_path),
    )

    client = TestClient(app)
    assert client.get("/chat").status_code == 200
    assert client.get("/chat/").status_code == 200
    assert client.get("/chat/icon.svg").headers["content-type"].startswith("image/svg+xml")
    assert client.get("/chat/health").json()["ok"] is True

    # 注册 -> 登录 -> 拿到会话令牌
    reg = client.post(
        "/chat/register",
        json={"username": "owner", "password": "secret123", "device_id": "dev-A"},
    )
    assert reg.status_code == 200 and reg.json()["ok"] is True
    token = reg.json()["token"]

    with client.websocket_connect("/chat/ws") as socket:
        socket.send_json({"type": "auth", "token": token})
        ready = socket.receive_json()
        assert ready["type"] == "ready"
        assert ready["username"] == "owner"
        history = socket.receive_json()
        assert history["type"] == "history"
        assert [item["text"] for item in history["messages"]] == ["历史问题", "历史回复"]
        socket.send_json({"type": "history_sync"})
        assert socket.receive_json()["type"] == "history"
        socket.send_json({"type": "message", "id": "12", "text": "测试", "images": []})
        notice = socket.receive_json()
        assert notice["type"] == "notice" and notice["text"] == "已收到"
        socket.send_json({"type": "history_sync"})
        synced = socket.receive_json()
        assert [item["text"] for item in synced["messages"]] == ["测试", "已收到"]

    assert dispatcher.messages[0].self_id == "web:main"
    assert dispatcher.messages[0].user_id == "web:owner"
    assert dispatcher.messages[0].conversation_id == "owner"
    assert dispatcher.messages[0].native_event == {"message_id": "12"}
