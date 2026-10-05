"""平台原生标识与现有整数消息 ID 之间的稳定映射工具。"""

from __future__ import annotations

import hashlib


def stable_message_id(platform: str, native_id: str) -> int:
    """生成稳定的 63 位正整数，供现有引用与持久化逻辑使用。"""
    raw = f"{platform}:{native_id}".encode("utf-8")
    value = int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "big")
    return value & 0x7FFF_FFFF_FFFF_FFFF or 1
