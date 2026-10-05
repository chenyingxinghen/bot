"""SillyTavern 角色卡接入层。

本模块让 QQ bot 的「酒馆」模式直接复用用户在 SillyTavern
（G:\\git_proj\\SillyTavern）里配置好的角色卡与世界书，而不是自己重新写一套角色扮演提示词
——即「避免重复造轮子」。

设计要点：
- 角色卡是 PNG（chara_card_v2 / v3），把定义存在 PNG 的 tEXt 文本块里。
  v2 用关键字 ``chara``、v3 用 ``ccv3``，内容均为 base64 编码的 JSON。
- 本模块只做「读取 + 构造酒馆风格 prompt + 调用生成」，不依赖 SillyTavern 进程是否运行。
  - 默认走 Ollama 直连（与 bot 其它模式同一套推理链路，最稳）。
  - 若配置了 ``SILLYTAVERN_URL``，则优先把拼好的消息发给正在运行的 SillyTavern
    服务端（/api/backends/chat-completions/generate），失败自动回退 Ollama。
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

import base64
import json
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx

from core.state import atomic_write_json


# SillyTavern 里常见的占位符
_CHAR_PH = "{{char}}"
_USER_PH = "{{user}}"


@dataclass
class CharacterCard:
    """从 SillyTavern 角色卡解析出的结构化信息。"""

    name: str
    description: str = ""
    personality: str = ""
    scenario: str = ""
    first_mes: str = ""
    mes_example: str = ""
    creator_notes: str = ""
    character_book: Optional[dict] = None
    world: str = ""
    source_path: Optional[Path] = None

    @property
    def has_lorebook(self) -> bool:
        if not self.character_book:
            return False
        entries = self.character_book.get("entries")
        return bool(entries)


@dataclass
class LoreEntry:
    keys: list[str]
    content: str
    insertion_order: int = 0


def _replace_placeholders(text: str, char_name: str, user_name: str) -> str:
    """将 {{char}} / {{user}} 等占位符替换成实际名称。"""
    if not text:
        return text
    return (
        text.replace(_CHAR_PH, char_name)
        .replace(_USER_PH, user_name)
        .replace("{{char}}", char_name)
        .replace("{{user}}", user_name)
    )


def render_first_message(card: CharacterCard, user_name: str) -> str:
    """渲染角色卡开场白，并替换 SillyTavern 常用占位符。"""
    return _replace_placeholders(
        card.first_mes.strip(), card.name or "角色", user_name
    ).strip()


# ─── PNG / 角色卡解析 ───────────────────────────────────

def _read_png_text_chunks(path: Path) -> dict[str, str]:
    """读取 PNG 的全部 tEXt 文本块，返回 {关键字: 文本}。"""
    data = Path(path).read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"不是合法的 PNG 文件：{path}")
    pos = 8
    chunks: dict[str, str] = {}
    n = len(data)
    while pos < n:
        if pos + 8 > n:
            break
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        ctype = data[pos + 4:pos + 8].decode("latin1")
        cdata = data[pos + 8:pos + 8 + length]
        if ctype == "tEXt":
            sep = cdata.find(b"\x00")
            if sep != -1:
                key = cdata[:sep].decode("latin1")
                val = cdata[sep + 1:].decode("utf-8", "replace")
                chunks[key] = val
        pos += 12 + length
    return chunks


def _decode_card_json(raw: str) -> dict:
    """角色卡文本块可能是 base64 或明文 JSON，两种都尝试。"""
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    try:
        decoded = base64.b64decode(raw, validate=False).decode("utf-8", "replace")
        return json.loads(decoded)
    except Exception:
        raise ValueError("无法解析角色卡 JSON（既不是 JSON 也不是 base64 JSON）")


def parse_character_card(path: str | Path) -> CharacterCard:
    """解析一个 SillyTavern 角色卡（.png 或 .json）。"""
    path = Path(path)
    if path.suffix.lower() == ".png":
        chunks = _read_png_text_chunks(path)
        raw = None
        for key in ("ccv3", "chara"):
            if key in chunks:
                raw = chunks[key]
                break
        if raw is None:
            raise ValueError(f"PNG 中未找到 ccv3/chara 角色卡块：{path}")
        obj = _decode_card_json(raw)
    elif path.suffix.lower() == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"不支持的角色卡格式：{path.suffix}")

    # v3 把字段放在 data 下；v2 可能直接平铺，也可能在 data 下
    data = obj.get("data", obj) if isinstance(obj, dict) else {}

    def pick(*names: str, default: str = "") -> str:
        for nm in names:
            val = data.get(nm, obj.get(nm)) if isinstance(obj, dict) else None
            if isinstance(val, str) and val.strip():
                return val
        return default

    name = pick("name") or path.stem
    description = pick("description")
    personality = pick("personality")
    scenario = pick("scenario")
    first_mes = pick("first_mes")
    mes_example = pick("mes_example")
    creator_notes = pick("creator_notes", "creatorcomment")
    world = ""
    character_book = None

    extensions = data.get("extensions")
    if isinstance(extensions, dict):
        world = str(extensions.get("world", "") or "")

    cb = data.get("character_book")
    if isinstance(cb, dict) and cb.get("entries"):
        character_book = cb
    elif isinstance(cb, str) and cb.strip().lower() not in ("", "none"):
        try:
            character_book = json.loads(cb)
        except Exception:
            character_book = None

    return CharacterCard(
        name=name,
        description=description,
        personality=personality,
        scenario=scenario,
        first_mes=first_mes,
        mes_example=mes_example,
        creator_notes=creator_notes,
        character_book=character_book,
        world=world,
        source_path=path,
    )


# ─── 角色卡 → 酒馆风格 prompt ──────────────────────────

def build_persona_prompt(card: CharacterCard, user_name: str, context_text: str | None = None) -> str:
    """根据角色卡生成 system prompt 的角色设定部分。

    采用社区通用的 TavernAI 角色扮演格式：角色设定 + 场景 + 扮演约束。
    ``context_text`` 用于世界书（lorebook）命中——仅当提供时才尝试注入，
    以对齐 SillyTavern 服务端的提示词工程（直连 Ollama 同样生效）。
    """
    char_name = card.name or "角色"
    parts: list[str] = []

    # 模式级行为边界由 data/modes/tavern.txt 统一定义；角色卡只提供当前身份资料，
    # 避免“保持角色 / 不声明 AI”等规则在 system prompt 中重复。
    parts.append(f"【当前角色】\n你是「{char_name}」。")

    if card.description:
        desc = _replace_placeholders(card.description, char_name, user_name)
        parts.append("【角色设定】\n" + desc)
    if card.personality:
        parts.append("【性格】\n" + _replace_placeholders(card.personality, char_name, user_name))
    if card.scenario:
        sc = _replace_placeholders(card.scenario, char_name, user_name)
        parts.append("【场景】\n" + sc)

    lore = build_lorebook_note(card, context_text) if context_text else None
    if lore:
        parts.append(lore)

    return "\n\n".join(parts)


def example_dialogue_to_messages(
    card: CharacterCard, user_name: str, max_turns: int = 6
) -> list[dict]:
    """返回有限的最近示例轮次，避免大型角色卡长期挤占真实对话上下文。"""
    turns = _split_example_dialogue(card.mes_example, card.name or "角色", user_name)
    if max_turns <= 0:
        return []
    selected = turns[-max_turns:]
    # few-shot 从 user 开始更符合常见 chat template；孤立 assistant 示例直接丢弃。
    while selected and selected[0]["role"] == "assistant":
        selected.pop(0)
    return selected


def _split_example_dialogue(text: str, char_name: str, user_name: str) -> list[dict]:
    """从 mes_example 解析出 user/assistant 轮次（尽力而为）。"""
    text = _replace_placeholders(text, char_name, user_name).strip()
    if not text:
        return []
    # 常见分隔符：<START> 、{{user}}: / {{char}}: 、用户: / char_name + ：
    blocks = [b.strip() for b in text.replace("<START>", "").split("\n\n") if b.strip()]
    turns: list[dict] = []
    for block in blocks:
        low = block.lower()
        if low.startswith(("{{user}}", "user:", "用户:", "你:")):
            speaker = "user"
            content = block.split(":", 1)[1].strip() if ":" in block else block
        elif low.startswith(("{{char}}", char_name.lower() + ":", "bot:")):
            speaker = "assistant"
            content = block.split(":", 1)[1].strip() if ":" in block else block
        else:
            # 没有明确说话人，跳过或当成角色台词
            continue
        if content:
            turns.append({"role": speaker, "content": content})
    return turns


def build_lorebook_note(card: CharacterCard, context_text: str) -> Optional[str]:
    """根据近期上下文命中世界书条目，返回需要注入的「世界信息」。"""
    if not card.has_lorebook:
        return None
    ctx = (context_text or "").lower()
    matched: list[tuple[int, str]] = []
    for entry in card.character_book.get("entries", []):
        keys = entry.get("keys") or []
        if isinstance(keys, str):
            keys = [keys]
        content = entry.get("content", "")
        if not content or not keys:
            continue
        if any(k and k.lower() in ctx for k in keys):
            order = int(entry.get("insertion_order", 0) or 0)
            matched.append((order, content))
    if not matched:
        return None
    matched.sort(key=lambda x: x[0])
    body = "\n\n".join(content for _, content in matched)
    return "[世界书 / 记忆]\n" + body


# ─── 生成客户端 ──────────────────────────────────────────

class SillyTavernClient:
    """负责「读取 + 生成」。生成默认走 Ollama，可选走 SillyTavern 服务端代理。"""

    def __init__(
        self,
        data_root: str | Path,
        user: str = "default-user",
        ollama_api_base: str = "http://127.0.0.1:11434",
        ollama_api_key: str = "",
        ollama_keep_alive: str = "30m",
        ollama_think: bool = False,
        st_url: str = "",
        st_source: str = "openai",
        st_reverse_proxy: str = "http://127.0.0.1:11434/v1",
        st_model: str = "",
        st_api_key: str = "",
        user_name: str = "你",
        default_character: str = "auto",
        request_timeout: float = 120.0,
    ) -> None:
        self.data_root = Path(data_root)
        self.user = user
        self.ollama_api_base = ollama_api_base.rstrip("/")
        self.ollama_api_key = ollama_api_key
        self.ollama_keep_alive = ollama_keep_alive
        self.ollama_think = ollama_think
        self.st_url = st_url.rstrip("/") if st_url else ""
        self.st_source = st_source
        self.st_reverse_proxy = st_reverse_proxy.rstrip("/") if st_reverse_proxy else ""
        self.st_model = st_model
        self.st_api_key = st_api_key
        self.user_name = user_name
        self.default_character = default_character
        self.request_timeout = request_timeout

    # ── 角色发现 / 加载 ──

    def characters_dir(self) -> Path:
        return self.data_root / self.user / "characters"

    def discover_characters(self) -> dict[str, Path]:
        """返回 {匹配的小写文件名或角色名: 卡片路径}。"""
        out: dict[str, Path] = {}
        d = self.characters_dir()
        if not d.exists():
            return out
        for p in sorted(d.iterdir()):
            if p.suffix.lower() not in (".png", ".json", ".charx"):
                continue
            try:
                card = parse_character_card(p)
            except Exception:
                # 解析失败的卡片不阻塞其它卡片
                out.setdefault(p.stem.lower(), p)
                continue
            out[p.stem.lower()] = p
            if card.name:
                out[card.name.lower()] = p
        return out

    def list_character_names(self) -> list[str]:
        names: list[str] = []
        for p in self.discover_characters().values():
            try:
                names.append(parse_character_card(p).name)
            except Exception:
                names.append(p.stem)
        # 去重保序
        seen = set()
        uniq = []
        for n in names:
            if n not in seen:
                seen.add(n)
                uniq.append(n)
        return uniq

    def resolve_character(self, name: Optional[str]) -> Optional[CharacterCard]:
        cards = self.discover_characters()
        if not cards:
            return None
        if not name or name.strip().lower() in ("auto", "默认", "default"):
            # 选第一个可用卡片
            first_path = next(iter(cards.values()))
            try:
                return parse_character_card(first_path)
            except Exception:
                return None
        key = name.strip().lower()
        target = cards.get(key)
        if target is None:
            # 宽松包含匹配
            for k, p in cards.items():
                if key in k:
                    target = p
                    break
        if target is None:
            return None
        try:
            return parse_character_card(target)
        except Exception:
            return None

    # ── 生成 ──

    async def generate_via_ollama(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        num_predict: int,
    ) -> str:
        base = self.ollama_api_base
        if base.endswith("/v1"):
            base = base[: -len("/v1")]
        url = f"{base}/api/chat"
        headers = {"Content-Type": "application/json"}
        if self.ollama_api_key and self.ollama_api_key != "你的魔搭token":
            headers["Authorization"] = f"Bearer {self.ollama_api_key}"
        options: dict = {
            "temperature": temperature,
            "repeat_penalty": 1.05,
        }
        if num_predict > 0:
            # num_predict<=0 表示不限制输出长度，省略该字段让模型按上下文填满
            options["num_predict"] = int(num_predict)
        payload = {
            "model": model,
            "keep_alive": self.ollama_keep_alive,
            "messages": messages,
            "think": self.ollama_think,
            "stream": False,
            "options": options,
        }
        async with httpx.AsyncClient(timeout=self.request_timeout) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        content = (data.get("message", {}) or {}).get("content") or ""
        if "</think>" in content:
            content = content.split("</think>", 1)[1]
        return content.strip()

    async def generate_via_sillytavern(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        num_predict: int,
    ) -> str:
        """调用正在运行的 SillyTavern 服务端（真实代理）。失败抛异常交给回退。"""
        if not self.st_url:
            raise RuntimeError("未配置 SILLYTAVERN_URL")
        async with httpx.AsyncClient(
            timeout=self.request_timeout, follow_redirects=True
        ) as client:
            tok_resp = await client.get(f"{self.st_url}/csrf-token")
            tok_resp.raise_for_status()
            token = tok_resp.json().get("token", "")
            cookies = tok_resp.cookies

            body = {
                "chat_completion_source": self.st_source,
                "reverse_proxy": self.st_reverse_proxy,
                "model": self.st_model or model,
                "messages": messages,
                "stream": False,
            "temperature": temperature,
            "max_tokens": num_predict if num_predict > 0 else -1,
            }
            headers = {"Content-Type": "application/json", "x-csrf-token": token}
            if self.st_api_key:
                headers["Authorization"] = f"Bearer {self.st_api_key}"

            resp = await client.post(
                f"{self.st_url}/api/backends/chat-completions/generate",
                json=body,
                headers=headers,
                cookies=cookies,
            )
            resp.raise_for_status()
            data = resp.json()

        # SillyTavern 透传后端响应，兼容 OpenAI / 原生格式
        content = ""
        if isinstance(data, dict):
            choices = data.get("choices")
            if isinstance(choices, list) and choices:
                content = (choices[0].get("message", {}) or {}).get("content") or ""
            if not content:
                content = data.get("message", {}).get("content") or data.get("content") or ""
        if not content:
            raise RuntimeError("SillyTavern 代理未返回正确内容")
        return content.strip()

    async def chat(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        num_predict: int,
    ) -> str:
        """生成主入口：优先 SillyTavern 服务端代理，失败回退 Ollama 直连。"""
        if self.st_url:
            try:
                return await self.generate_via_sillytavern(
                    messages, model, temperature, num_predict
                )
            except Exception as exc:
                logger.warning(
                    f"SillyTavern 代理失败，回退 Ollama 直连：{type(exc).__name__}: {exc}"
                )
        return await self.generate_via_ollama(
            messages, model, temperature, num_predict
        )


class CharacterSelection:
    """为 (self_id, peer_id) 记忆「当前酒馆角色」，持久化到 JSON。"""

    def __init__(self, state_path: str | Path, default: str = "auto") -> None:
        self._state_path = Path(state_path)
        self._default = default
        self._state: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if self._state_path.exists():
            try:
                self._state = json.loads(
                    self._state_path.read_text(encoding="utf-8")
                )
            except Exception:
                self._state = {}

    def _save(self) -> None:
        try:
            atomic_write_json(self._state_path, self._state)
        except Exception as exc:  # noqa: BLE001
            logger.warning("保存酒馆角色选择失败：%s", exc)

    def get(self, key_pair: tuple[str, str]) -> str:
        return self._state.get(":".join(key_pair), self._default)

    def set(self, key_pair: tuple[str, str], character: str) -> None:
        self._state[":".join(key_pair)] = character
        self._save()
