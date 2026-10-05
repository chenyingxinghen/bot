"""ImageService：包装 ComfyUIClient，对对话业务暴露 ``generate(prompt) -> list[Path]``。"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

from pathlib import Path

from config.image import ImageSettings
from core.image.comfyui import ComfyUIClient


class ImageService:
    def __init__(self, client: ComfyUIClient, settings: ImageSettings | None = None) -> None:
        self.client = client
        self.settings = settings

    async def generate(self, prompt: str) -> list[Path]:
        """根据文本提示词生成图片，返回本地文件路径列表。"""
        return await self.client.generate(prompt)
