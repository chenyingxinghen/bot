"""记忆模块的路径、向量服务和在线抽取参数。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from config.base import as_bool, resolve_bot_path


@dataclass(frozen=True)
class EmbeddingSettings:
    api_base: str = "http://127.0.0.1:11434"
    api_key: str = ""
    model: str = "qwen3-embedding:0.6b"
    timeout: float = 30.0
    num_gpu: int = 0
    keep_alive: str = "30m"

    @classmethod
    def from_config(cls, config: object) -> "EmbeddingSettings":
        return cls(
            api_base=str(getattr(config, "embedding_api_base", cls.api_base)),
            api_key=str(getattr(config, "embedding_api_key", cls.api_key)),
            model=str(getattr(config, "embedding_model", cls.model)),
            timeout=float(getattr(config, "embedding_timeout", cls.timeout)),
            num_gpu=int(getattr(config, "embedding_num_gpu", cls.num_gpu)),
            keep_alive=str(getattr(config, "embedding_keep_alive", cls.keep_alive)),
        )

    @classmethod
    def from_environment(cls) -> "EmbeddingSettings":
        return cls(
            api_base=os.getenv("EMBEDDING_API_BASE", cls.api_base),
            api_key=os.getenv("EMBEDDING_API_KEY", os.getenv("LLM_API_KEY", "")),
            model=os.getenv("EMBEDDING_MODEL", cls.model),
            timeout=float(os.getenv("EMBEDDING_TIMEOUT", "120")),
            num_gpu=int(os.getenv("EMBEDDING_NUM_GPU", "0")),
            keep_alive=os.getenv("EMBEDDING_KEEP_ALIVE", cls.keep_alive),
        )


@dataclass(frozen=True)
class MemoryLLMSettings:
    api_base: str = "http://127.0.0.1:11434"
    api_key: str = ""
    model: str = "qwen3:8b"
    keep_alive: str = "30m"

    @classmethod
    def from_environment(cls) -> "MemoryLLMSettings":
        return cls(
            api_base=os.getenv("LLM_API_BASE", cls.api_base),
            api_key=os.getenv("LLM_API_KEY", cls.api_key),
            model=os.getenv("LLM_MODEL", cls.model),
            keep_alive=os.getenv("LLM_KEEP_ALIVE", cls.keep_alive),
        )


@dataclass(frozen=True)
class OnlineExtractionSettings:
    enabled: bool = True
    clone_enabled: bool = True
    tavern_enabled: bool = True
    writer_enabled: bool = True
    min_messages: int = 16
    min_partner_messages: int = 6
    min_semantic_chars: int = 160
    min_confidence: float = 0.75
    semantic_duplicate_threshold: float = 0.84
    max_messages: int = 48
    max_items: int = 3
    timeout: float = 120.0
    tavern_min_messages: int = 8
    tavern_min_partner_messages: int = 4
    writer_min_messages: int = 8
    writer_min_partner_messages: int = 4

    @classmethod
    def from_config(cls, config: object) -> "OnlineExtractionSettings":
        return cls(
            enabled=as_bool(getattr(config, "memory_auto_extract", True), True),
            clone_enabled=as_bool(getattr(
                config, "memory_clone_auto_extract", True
            ), True),
            tavern_enabled=as_bool(getattr(
                config, "memory_tavern_auto_extract", True
            ), True),
            writer_enabled=as_bool(getattr(
                config, "memory_writer_auto_extract", True
            ), True),
            min_messages=int(getattr(config, "memory_extract_min_messages", cls.min_messages)),
            min_partner_messages=int(getattr(
                config, "memory_extract_min_partner_messages", cls.min_partner_messages
            )),
            min_semantic_chars=int(getattr(
                config, "memory_extract_min_semantic_chars", cls.min_semantic_chars
            )),
            min_confidence=float(getattr(
                config, "memory_extract_min_confidence", cls.min_confidence
            )),
            semantic_duplicate_threshold=float(getattr(
                config,
                "memory_extract_duplicate_threshold",
                cls.semantic_duplicate_threshold,
            )),
            max_messages=int(getattr(
                config, "memory_extract_max_messages", cls.max_messages
            )),
            max_items=int(getattr(config, "memory_extract_max_items", cls.max_items)),
            timeout=float(getattr(config, "memory_extract_timeout", cls.timeout)),
            tavern_min_messages=int(getattr(
                config, "memory_tavern_min_messages", cls.tavern_min_messages
            )),
            tavern_min_partner_messages=int(getattr(
                config,
                "memory_tavern_min_partner_messages",
                cls.tavern_min_partner_messages,
            )),
            writer_min_messages=int(getattr(
                config, "memory_writer_min_messages", cls.writer_min_messages
            )),
            writer_min_partner_messages=int(getattr(
                config,
                "memory_writer_min_partner_messages",
                cls.writer_min_partner_messages,
            )),
        )

    def enabled_for(self, mode_key: str) -> bool:
        if not self.enabled:
            return False
        if mode_key == "clone":
            return self.clone_enabled
        if mode_key == "tavern":
            return self.tavern_enabled
        if mode_key == "writer":
            return self.writer_enabled
        return False

    def thresholds_for(self, mode_key: str) -> tuple[int, int]:
        if mode_key == "tavern":
            return self.tavern_min_messages, self.tavern_min_partner_messages
        if mode_key == "writer":
            return self.writer_min_messages, self.writer_min_partner_messages
        return self.min_messages, self.min_partner_messages


@dataclass(frozen=True)
class MemorySettings:
    database: Path
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    online_extraction: OnlineExtractionSettings = field(
        default_factory=OnlineExtractionSettings
    )

    @classmethod
    def from_config(cls, config: object, bot_root: Path) -> "MemorySettings":
        raw = str(getattr(config, "memory_db", "data/memory_{mode}.db"))
        if "{mode}" not in raw:
            # 向后兼容：旧配置未写 {mode} 占位符，自动按模式分库，避免所有模式共用一库
            p = Path(raw)
            raw = str(p.parent / (p.stem + "_{mode}" + p.suffix))
        database = resolve_bot_path(bot_root, raw)
        return cls(
            database=database,
            embedding=EmbeddingSettings.from_config(config),
            online_extraction=OnlineExtractionSettings.from_config(config),
        )
