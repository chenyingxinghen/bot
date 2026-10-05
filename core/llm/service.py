"""LLM 服务层：封装重试与降级，对对话业务无感知。

供 ``core/conversation`` 调用，便于以后无缝替换后端（如换成别的 chat completion
服务），对话层不需要知道传输细节。
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

from config.llm import LLMSettings
from core.llm.client import OllamaClient


class LLMService:
    """把一次「对话请求」封装成可重试的调用。"""

    def __init__(self, client: OllamaClient, settings: LLMSettings) -> None:
        self.client = client
        self.s = settings

    async def call(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        vision: bool = False,
        temperature: float | None = None,
        num_predict: int | None = None,
    ) -> str:
        """生成回复文本。

        ``model`` 缺省时按 ``vision`` 选择视觉模型或默认模型；``temperature`` /
        ``num_predict`` 缺省时回落到全局配置。失败时重试一次，仍失败则抛出。
        """
        model = model or (self.s.vision_model if vision else self.s.model)
        temperature = self.s.temperature if temperature is None else temperature
        num_predict = self.s.num_predict if num_predict is None else num_predict

        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                return await self.client.chat(messages, model, temperature, num_predict)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning(
                    "LLM 调用失败（第 %d 次）：%s: %s",
                    attempt + 1,
                    type(exc).__name__,
                    exc,
                )
        raise RuntimeError(f"LLM 调用失败：{last_exc}") from last_exc

    async def stream(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        vision: bool = False,
        temperature: float | None = None,
        num_predict: int | None = None,
    ):
        """流式生成：逐块产出文本增量。

        与 ``call`` 共享模型/温度/长度选择逻辑，但省略失败重试（流式连接中途
        重试意义不大，由上层捕获异常并提示）。
        """
        model = model or (self.s.vision_model if vision else self.s.model)
        temperature = self.s.temperature if temperature is None else temperature
        num_predict = self.s.num_predict if num_predict is None else num_predict
        async for delta in self.client.stream(
            messages, model, temperature, num_predict
        ):
            yield delta
