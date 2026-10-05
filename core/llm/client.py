"""Ollama 直连 LLM 客户端（与 nonebot 解耦，便于单测）。

只负责把消息序列发给 Ollama 的 ``/api/chat`` 并取回内容，不感知任何对话业务。
支持多模态：消息 content 可以是字符串，也可以是
``[{"type": "text", "text": ...}, {"type": "image_url", "image_url": {"url": "data:..."}}]``
这样的片段列表；图片会被收敛到 Ollama 的 ``images`` 字段（base64 / data URL）。
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

import httpx

from config.llm import LLMSettings


_DEFAULT_TIMEOUT = 120.0


def _normalize_data_url(value: str) -> str:
    """把图片输入规范成 data URL：已带 data: 前缀的保持不变，否则按 jpeg base64 处理。"""
    value = (value or "").strip()
    if value.startswith("data:"):
        return value
    return "data:image/jpeg;base64," + value


class OllamaClient:
    """对 Ollama ``/api/chat`` 的最小封装。"""

    def __init__(self, settings: LLMSettings) -> None:
        self.s = settings
        base = settings.api_base.rstrip("/")
        if base.endswith("/v1"):
            base = base[: -len("/v1")]
        self._base = base

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        key = self.s.api_key
        if key and key != "你的魔搭token":
            headers["Authorization"] = f"Bearer {key}"
        return headers

    @staticmethod
    def _messages_to_ollama(messages: list[dict]) -> list[dict]:
        """把内部多模态消息格式转换为 Ollama 接受的格式。"""
        out: list[dict] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content")
            images: list[str] = []
            if isinstance(content, list):
                parts: list[str] = []
                for seg in content:
                    seg_type = seg.get("type")
                    if seg_type == "text":
                        parts.append(seg.get("text", ""))
                    elif seg_type == "image_url":
                        url = seg.get("image_url", {}).get("url", "")
                        if url:
                            images.append(_normalize_data_url(url))
                text = "".join(parts)
            else:
                text = content or ""
            item: dict = {"role": role, "content": text}
            if images:
                item["images"] = images
            out.append(item)
        return out

    async def request(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        num_predict: int,
    ) -> dict:
        """发送一次聊天请求，返回 Ollama 原始 JSON。调用方负责解析内容。"""
        url = f"{self._base}/api/chat"
        options: dict = {
            "temperature": temperature,
            "repeat_penalty": 1.05,
        }
        if num_predict > 0:
            # num_predict<=0 表示不限制输出长度，省略该字段让模型按上下文填满
            options["num_predict"] = int(num_predict)
        payload = {
            "model": model,
            "keep_alive": self.s.keep_alive,
            "messages": self._messages_to_ollama(messages),
            "think": self.s.think,
            "stream": False,
            "options": options,
        }
        async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
            resp = await client.post(url, headers=self._headers(), json=payload)
            resp.raise_for_status()
            return resp.json()

    async def chat(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        num_predict: int,
    ) -> str:
        """发送请求并清洗出纯文本回复（去掉 <think> 与 markdown 噪音）。"""
        from core.llm.reply import _clean_llm_reply

        data = await self.request(messages, model, temperature, num_predict)
        return _clean_llm_reply(data)

    async def stream(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        num_predict: int,
    ):
        """流式聊天：逐块产出已清洗的纯文本增量。

        与 ``chat`` 等价的内容处理（去掉 ``<think>`` 推理块、markdown 噪音），
        但改为边收边产出，供 Web 入口做打字机效果。调用方负责把这些增量拼回
        完整文本用于后续动作解析。
        """
        import json

        url = f"{self._base}/api/chat"
        options: dict = {
            "temperature": temperature,
            "repeat_penalty": 1.05,
        }
        if num_predict > 0:
            options["num_predict"] = int(num_predict)
        payload = {
            "model": model,
            "keep_alive": self.s.keep_alive,
            "messages": self._messages_to_ollama(messages),
            "think": self.s.think,
            "stream": True,
            "options": options,
        }
        in_think = False
        async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
            async with client.stream(
                "POST", url, headers=self._headers(), json=payload
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data:"):
                        line = line[5:]
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if data.get("done"):
                        # 末尾可能附带完整 usage；直接结束
                        return
                    content = (data.get("message", {}) or {}).get("content", "")
                    if not content:
                        continue
                    content = self._strip_think(content, in_think)
                    in_think = content.in_think
                    content = content.text.replace("**", "").replace("*", "").replace("`", "")
                    if content:
                        yield content

    @staticmethod
    def _strip_think(content: str, in_think: bool) -> "_ThinkResult":
        from typing import NamedTuple

        class _ThinkResult(NamedTuple):
            text: str
            in_think: bool

        if in_think:
            end = content.find("</think>")
            if end == -1:
                return _ThinkResult("", True)
            return _ThinkResult(content[end + len("</think>"):], False)
        start = content.find("<think>")
        if start == -1:
            # 没有开标签：若仍有残留的 </think>（极少见，如上一块已消费 <think>），
            # 也把它和之前的内容一起丢弃，避免推理标记泄漏到显示。
            end = content.find("</think>")
            if end != -1:
                return _ThinkResult(content[end + len("</think>"):], False)
            return _ThinkResult(content, False)
        before = content[:start]
        rest = content[start + len("<think>"):]
        end = rest.find("</think>")
        if end == -1:
            return _ThinkResult(before, True)
        return _ThinkResult(before + rest[end + len("</think>"):], False)
