"""LLM 模型和 API 配置。"""

from __future__ import annotations

from dataclasses import dataclass

from config.base import as_bool


@dataclass(frozen=True)
class LLMSettings:
    api_base: str
    api_key: str
    model: str
    vision_model: str
    temperature: float
    num_predict: int
    keep_alive: str
    think: bool
    log_full_io: bool

    @classmethod
    def from_config(cls, config: object) -> "LLMSettings":
        llm_model = str(getattr(
            config,
            "llm_model",
            "Qwen/Qwen3-235B-A22B-Instruct",
        ))
        llm_vision_model = str(getattr(config, "llm_vision_model", llm_model))
        return cls(
            api_base=str(getattr(
                config,
                "llm_api_base",
                "https://api-inference.modelscope.cn/v1/",
            )),
            api_key=str(getattr(config, "llm_api_key", "")),
            model=llm_model,
            vision_model=llm_vision_model,
            temperature=float(getattr(config, "llm_temperature", 0.6)),
            num_predict=int(getattr(config, "llm_num_predict", 192)),
            keep_alive=str(getattr(config, "llm_keep_alive", "30m")),
            think=as_bool(getattr(config, "llm_think", False), False),
            log_full_io=as_bool(getattr(config, "llm_log_full_io", False), False),
        )
