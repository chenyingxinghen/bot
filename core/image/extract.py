"""从 OneBot 事件中提取图片为 base64 data URL（原 chat_style 内联逻辑迁出）。

核心只依赖事件上的 ``message``（OneBot 消息段列表），对 nonebot 类型不强制，
方便在不引入 nonebot 的环境下做单元测试：传入任意带 ``message`` 属性的对象即可。
"""

from __future__ import annotations

import base64
import logging

logger = logging.getLogger(__name__)

from pathlib import Path

import httpx


def _as_data_url(candidate: str, default_ext: str = "jpeg") -> str:
    """把各种图片引用归一为 data URL。"""
    if candidate.startswith("base64://"):
        return "data:image/" + default_ext + ";base64," + candidate[len("base64://") :]
    if candidate.startswith("data:"):
        return candidate
    if candidate.startswith("http://") or candidate.startswith("https://"):
        return candidate  # URL 交给调用方下载
    path = Path(candidate)
    if path.exists():
        raw = path.read_bytes()
        ext = path.suffix.lower().lstrip(".") or default_ext
        return f"data:image/{ext};base64," + base64.b64encode(raw).decode("ascii")
    # NapCat 的 file 字段有时直接是 base64，兜底处理
    return "data:image/" + default_ext + ";base64," + candidate


async def _resolve_image(candidate: str) -> str:
    if candidate.startswith("http://") or candidate.startswith("https://"):
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            resp = await client.get(candidate)
            resp.raise_for_status()
            ctype = resp.headers.get("content-type", "image/jpeg").split(";")[0] or "image/jpeg"
            return f"data:{ctype};base64," + base64.b64encode(resp.content).decode("ascii")
    return _as_data_url(candidate)


async def _extract_images(event: object) -> list[str]:
    """从事件里提取所有图片，返回 base64 data URL 列表。"""
    message = getattr(event, "message", None)
    if not message:
        return []
    results: list[str] = []
    for seg in message:
        if getattr(seg, "type", None) != "image":
            continue
        data = getattr(seg, "data", None) or {}
        candidate = data.get("base64") or data.get("url") or data.get("file")
        if not candidate:
            continue
        try:
            results.append(await _resolve_image(candidate))
        except Exception as exc:  # noqa: BLE001
            logger.warning("提取图片失败：%s", exc)
    return results
