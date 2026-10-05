"""生图参数和 ComfyUI 配置。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImageSettings:
    comfyui: "ComfyUISettings"
    max_concurrent_images: int
    max_images: int
    max_image_bytes: int

    @classmethod
    def from_config(cls, config: object, bot_root: Path) -> "ImageSettings":
        # 局部导入，避免 config -> core -> config 的循环导入（config 尚未完整初始化时）
        from core.image.comfyui import ComfyUISettings

        return cls(
            comfyui=ComfyUISettings.from_config(config, bot_root),
            max_concurrent_images=int(
                getattr(config, "comfyui_max_concurrent_images", 1)
            ),
            max_images=int(getattr(config, "max_images", 4)),
            max_image_bytes=int(getattr(config, "max_image_bytes", 10 * 1024 * 1024)),
        )
