"""Web 登录 / 注册相关测试。"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from config.platforms import WebSettings
from core.image.service import ImageService
from core.qq.interactions import StickerCatalog
from core.web.app import install_web_routes
from core.web.auth import AccountStore


def _make_app(tmp_path: Path) -> TestClient:
    static = Path(__file__).resolve().parents[1] / "core" / "web" / "static"
    settings = WebSettings(
        enabled=True,
        path="/chat",
        token="secret-token",
        accounts_file=tmp_path / "web_accounts.json",
        history_db=tmp_path / "web_history.db",
        max_accounts=2,
        static_dir=static,
        stream=False,
    )
    app = FastAPI()
    install_web_routes(
        app, settings, _DummyDispatcher(), FakeImage(), StickerCatalog(tmp_path)
    )
    return TestClient(app)


class _DummyDispatcher:
    async def dispatch(self, message, sender):
        return True

    async def dispatch_stream(self, message, sender):
        return True


class FakeImage:
    async def generate(self, prompt):
        return []


def test_register_then_login_then_me(tmp_path):
    client = _make_app(tmp_path)
    reg = client.post(
        "/chat/register",
        json={"username": "alice", "password": "secret1", "device_id": "dev-1"},
    )
    assert reg.status_code == 200 and reg.json()["ok"] is True
    token = reg.json()["token"]

    me = client.get("/chat/me", headers={"Authorization": f"Bearer {token}"})
    assert me.json() == {"ok": True, "username": "alice"}

    login = client.post(
        "/chat/login",
        json={"username": "alice", "password": "secret1", "device_id": "dev-1"},
    )
    assert login.status_code == 200 and login.json()["ok"] is True
    assert isinstance(login.json()["token"], str) and login.json()["token"]


def test_register_rejects_weak_credentials(tmp_path):
    client = _make_app(tmp_path)
    bad_user = client.post(
        "/chat/register",
        json={"username": "a", "password": "secret1", "device_id": "dev-1"},
    )
    assert bad_user.status_code == 409
    bad_pw = client.post(
        "/chat/register",
        json={"username": "bob", "password": "123", "device_id": "dev-2"},
    )
    assert bad_pw.status_code == 409


def test_same_device_cannot_register_twice(tmp_path):
    client = _make_app(tmp_path)
    first = client.post(
        "/chat/register",
        json={"username": "alice", "password": "secret1", "device_id": "dev-1"},
    )
    assert first.status_code == 200
    second = client.post(
        "/chat/register",
        json={"username": "bob", "password": "secret2", "device_id": "dev-1"},
    )
    assert second.status_code == 409
    assert "设备" in second.json()["error"]


def test_taken_username_rejected(tmp_path):
    client = _make_app(tmp_path)
    client.post(
        "/chat/register",
        json={"username": "alice", "password": "secret1", "device_id": "dev-1"},
    )
    dup = client.post(
        "/chat/register",
        json={"username": "alice", "password": "secret2", "device_id": "dev-2"},
    )
    assert dup.status_code == 409
    assert "用户名" in dup.json()["error"]


def test_max_accounts_cap_enforced(tmp_path):
    client = _make_app(tmp_path)
    assert client.post(
        "/chat/register",
        json={"username": "acc1", "password": "secret1", "device_id": "d1"},
    ).status_code == 200
    assert client.post(
        "/chat/register",
        json={"username": "acc2", "password": "secret2", "device_id": "d2"},
    ).status_code == 200
    third = client.post(
        "/chat/register",
        json={"username": "acc3", "password": "secret3", "device_id": "d3"},
    )
    assert third.status_code == 409


def test_login_wrong_password_rejected(tmp_path):
    client = _make_app(tmp_path)
    client.post(
        "/chat/register",
        json={"username": "alice", "password": "secret1", "device_id": "dev-1"},
    )
    bad = client.post(
        "/chat/login",
        json={"username": "alice", "password": "wrong", "device_id": "dev-1"},
    )
    assert bad.status_code == 401


def test_websocket_rejects_bad_token(tmp_path):
    client = _make_app(tmp_path)
    with client.websocket_connect("/chat/ws") as socket:
        socket.send_json({"type": "auth", "token": "not-a-valid-token"})
        resp = socket.receive_json()
        assert resp["type"] == "auth_error"


def test_account_store_persists_and_captures_meta(tmp_path):
    store = AccountStore(
        tmp_path / "accounts.json", secret="s", max_accounts=1
    )
    ok, _, account = store.register("alice", "secret1", "dev-9", "203.0.113.7")
    assert ok and account.register_ip == "203.0.113.7"

    # 重新加载（模拟进程重启）后设备限注册一次仍生效
    reloaded = AccountStore(
        tmp_path / "accounts.json", secret="s", max_accounts=1
    )
    assert reloaded.count() == 1
    ok2, error2, _ = reloaded.register("bob", "secret2", "dev-9", "198.51.100.5")
    assert not ok2 and "设备" in error2

    # 错误密码登录失败
    assert reloaded.login("alice", "nope", "198.51.100.5")[0] is False
    ok3, _, _ = reloaded.login("alice", "secret1", "198.51.100.5")
    assert ok3
    # 登录 IP/时间必须持久化，而不是只更新当前进程内的 Account 对象。
    after_login = AccountStore(
        tmp_path / "accounts.json", secret="s", max_accounts=1
    )
    persisted = after_login._accounts["alice"]
    assert persisted.last_login_ip == "198.51.100.5"
    assert persisted.last_login_at > 0


def test_login_allows_changed_device_and_ip_but_records_latest(tmp_path):
    """当前设备/IP用于注册限流与审计，不是强制登录绑定。"""
    path = tmp_path / "accounts.json"
    store = AccountStore(path, secret="s", max_accounts=1)
    assert store.register("alice", "secret1", "device-a", "203.0.113.7")[0]

    ok, _, account = store.login(
        "alice", "secret1", "198.51.100.9", device_id="device-b"
    )
    assert ok
    assert account.device_id == "device-a"
    assert account.last_login_ip == "198.51.100.9"


def test_session_token_expiry_tamper_and_account_membership(tmp_path):
    store = AccountStore(tmp_path / "tokens.json", secret="s", max_accounts=2)
    assert store.register("alice", "secret1", "device-a", "127.0.0.1")[0]
    token = store.create_token("alice")
    assert store.verify_token(token) == "alice"
    # 签名正确但账号不存在也不能通过鉴权
    assert store.verify_token(store.create_token("ghost")) is None
    # 篡改签名
    assert store.verify_token(token + "x") is None
    # 过期令牌
    expired = store.create_token("alice", ttl=-10)
    assert store.verify_token(expired) is None
