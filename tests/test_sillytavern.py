"""SillyTavern 角色卡接入层测试。

覆盖：占位符替换、角色卡解析（v3 JSON / PNG）、persona/lorebook 构造、
角色发现与解析、以及 Ollama 直连 / SillyTavern 服务端代理（含失败回退）。
（用 asyncio.run 包裹异步调用，避免依赖 pytest-asyncio。）
"""

from __future__ import annotations

import asyncio
import base64
import json
import struct
import zlib
from pathlib import Path

import pytest

from core.character.client import (
    CharacterCard,
    CharacterSelection,
    SillyTavernClient,
    _replace_placeholders,
    build_lorebook_note,
    build_persona_prompt,
    parse_character_card,
)


# ── 工具：构造合成角色卡 ──────────────────────────────

def _make_v3_json_card(tmp_path: Path, name: str = "TestChar") -> Path:
    obj = {
        "spec": "chara_card_v3",
        "spec_version": "3.0",
        "data": {
            "name": name,
            "description": "姓名：{{char}}\n设定：喜欢{{user}}",
            "personality": "温柔",
            "scenario": "{{user}} 与 {{char}} 在咖啡馆",
            "first_mes": "你好呀",
            "mes_example": "{{user}}: 在吗？\n{{char}}: 在的",
            "character_book": {
                "entries": [
                    {
                        "keys": ["魔法", "咒语"],
                        "content": "世界观：这是一个有魔法的世界",
                        "insertion_order": 0,
                    }
                ]
            },
        },
    }
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    return p


def _make_png_card(tmp_path: Path, name: str = "PngChar") -> Path:
    """构造一个最小合法 PNG，内嵌 ccv3 tEXt 块。"""
    card_obj = {
        "spec": "chara_card_v3",
        "spec_version": "3.0",
        "data": {
            "name": name,
            "description": "PNG 角色 {{char}}，对手是 {{user}}",
            "scenario": "test scenario",
            "first_mes": "hi",
            "mes_example": "",
            "character_book": None,
        },
    }
    raw = base64.b64encode(
        json.dumps(card_obj, ensure_ascii=False).encode("utf-8")
    ).decode("ascii")

    def chunk(ctype: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + ctype
            + data
            + struct.pack(">I", zlib.crc32(ctype + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)
    text_data = b"ccv3\x00" + raw.encode("utf-8")
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"tEXt", text_data)
        + chunk(b"IEND", b"")
    )
    p = tmp_path / f"{name}.png"
    p.write_bytes(png)
    return p


# ── 占位符 / prompt 构造 ───────────────────────────────

def test_replace_placeholders():
    out = _replace_placeholders("我是{{char}}，对面是{{user}}", "爱丽丝", "用户")
    assert "爱丽丝" in out and "用户" in out
    assert "{{char}}" not in out and "{{user}}" not in out


def test_parse_v3_json_card(tmp_path):
    p = _make_v3_json_card(tmp_path)
    card = parse_character_card(p)
    assert card.name == "TestChar"
    assert "{{char}}" in card.description
    assert card.has_lorebook
    assert card.character_book["entries"][0]["keys"] == ["魔法", "咒语"]


def test_parse_v3_png_card(tmp_path):
    p = _make_png_card(tmp_path)
    card = parse_character_card(p)
    assert card.name == "PngChar"
    assert "{{char}}" in card.description


def test_build_persona_prompt_replaces_placeholders(tmp_path):
    card = parse_character_card(_make_v3_json_card(tmp_path))
    persona = build_persona_prompt(card, "用户")
    assert "TestChar" in persona
    assert "用户" in persona
    assert "{{char}}" not in persona and "{{user}}" not in persona
    assert "【角色设定" in persona


def test_build_lorebook_note_keyword_match(tmp_path):
    card = parse_character_card(_make_v3_json_card(tmp_path))
    note = build_lorebook_note(card, "今天我们聊了魔法")
    assert note is not None
    assert "有魔法的世界" in note
    assert build_lorebook_note(card, "今天天气真好") is None


# ── 角色发现 / 解析 ───────────────────────────────────

def test_discover_and_resolve(tmp_path):
    chars_dir = tmp_path / "default-user" / "characters"
    chars_dir.mkdir(parents=True)
    _make_v3_json_card(chars_dir, "Alice")
    _make_png_card(chars_dir, "Bob")
    client = SillyTavernClient(data_root=tmp_path, user="default-user")
    names = client.list_character_names()
    assert "Alice" in names and "Bob" in names

    auto = client.resolve_character("auto")
    assert auto is not None

    by_name = client.resolve_character("bob")
    assert by_name is not None and by_name.name == "Bob"

    missing = client.resolve_character("nobody")
    assert missing is None


def test_resolve_character_no_cards(tmp_path):
    client = SillyTavernClient(data_root=tmp_path)
    assert client.resolve_character("auto") is None


# ── 生成：Ollama 直连 / 代理 / 回退 ────────────────────

class _FakeResponse:
    def __init__(self, payload, cookies=None):
        self._payload = payload
        self.cookies = cookies or {}

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, get_payload=None, post_payload=None, cookies=None):
        self._get = get_payload
        self._post = post_payload
        self._cookies = cookies or {}
        self.post_calls = []
        self.get_calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url, **kw):
        self.get_calls.append(url)
        return _FakeResponse(self._get, self._cookies)

    async def post(self, url, **kw):
        self.post_calls.append((url, kw.get("json")))
        return _FakeResponse(self._post, self._cookies)


def test_generate_via_ollama(monkeypatch, tmp_path):
    parse_character_card(_make_v3_json_card(tmp_path))
    client = SillyTavernClient(data_root=tmp_path, ollama_api_base="http://x/v1")
    fake = _FakeClient(post_payload={"message": {"content": "你好，我是TestChar"}})
    monkeypatch.setattr("core.character.client.httpx.AsyncClient", lambda *a, **k: fake)
    out = asyncio.run(client.generate_via_ollama(
        [{"role": "system", "content": "x"}, {"role": "user", "content": "hi"}],
        model="m", temperature=0.9, num_predict=256,
    ))
    assert out == "你好，我是TestChar"
    assert fake.post_calls[0][0].endswith("/api/chat")
    assert "/v1/api/chat" not in fake.post_calls[0][0]


def test_generate_via_sillytavern_proxy(monkeypatch, tmp_path):
    parse_character_card(_make_v3_json_card(tmp_path))
    client = SillyTavernClient(
        data_root=tmp_path, st_url="http://127.0.0.1:8000",
        st_source="openai", st_reverse_proxy="http://127.0.0.1:11434/v1",
    )
    fake = _FakeClient(
        get_payload={"token": "CSRF123"},
        post_payload={"choices": [{"message": {"content": "代理回复内容"}}]},
        cookies={"connect.sid": "abc"},
    )
    monkeypatch.setattr("core.character.client.httpx.AsyncClient", lambda *a, **k: fake)
    out = asyncio.run(client.generate_via_sillytavern(
        [{"role": "system", "content": "x"}, {"role": "user", "content": "hi"}],
        model="m", temperature=0.9, num_predict=256,
    ))
    assert out == "代理回复内容"
    assert any("csrf-token" in u for u in fake.get_calls)
    posted = fake.post_calls[0][1]
    assert posted["reverse_proxy"] == "http://127.0.0.1:11434/v1"
    assert posted["chat_completion_source"] == "openai"


def test_chat_falls_back_to_ollama_on_proxy_error(monkeypatch, tmp_path):
    parse_character_card(_make_v3_json_card(tmp_path))
    client = SillyTavernClient(
        data_root=tmp_path, st_url="http://127.0.0.1:8000",
        ollama_api_base="http://127.0.0.1:11434",
    )

    class _ProxyDownClient(_FakeClient):
        async def get(self, url, **kw):
            # 模拟 SillyTavern 服务端代理不可达
            raise RuntimeError("sillytavern down")

        async def post(self, url, **kw):
            # 兜底走 Ollama 直连时，返回 Ollama 格式响应
            return _FakeResponse({"message": {"content": "Ollama 兜底回复"}})

    monkeypatch.setattr("core.character.client.httpx.AsyncClient",
                        lambda *a, **k: _ProxyDownClient())
    out = asyncio.run(client.chat(
        [{"role": "user", "content": "hi"}], model="m",
        temperature=0.9, num_predict=256,
    ))
    assert out == "Ollama 兜底回复"


def test_chat_direct_ollama_when_no_url(monkeypatch, tmp_path):
    parse_character_card(_make_v3_json_card(tmp_path))
    client = SillyTavernClient(data_root=tmp_path, ollama_api_base="http://127.0.0.1:11434")
    fake = _FakeClient(post_payload={"message": {"content": "直连结果"}})
    monkeypatch.setattr("core.character.client.httpx.AsyncClient", lambda *a, **k: fake)
    out = asyncio.run(client.chat(
        [{"role": "user", "content": "hi"}], model="m",
        temperature=0.9, num_predict=256,
    ))
    assert out == "直连结果"
    assert not fake.get_calls


# ── 角色选择持久化 ──────────────────────────────────────

def test_character_selection_persist(tmp_path):
    state = tmp_path / "tavern_characters.json"
    sel = CharacterSelection(state, default="auto")
    assert sel.get(("self1", "peer1")) == "auto"
    sel.set(("self1", "peer1"), "Alice")
    assert sel.get(("self1", "peer1")) == "Alice"
    sel2 = CharacterSelection(state, default="auto")
    assert sel2.get(("self1", "peer1")) == "Alice"
