"""记忆相关设置的统一导出入口。

重构后将配置的唯一归宿收归 ``config`` 包（见 ``config.memory``），
此处仅做兼容再导出，供 core 内部模块（如 ``core.memory.online``）与
``tools`` 下的脚本沿用 ``core.memory.settings`` 这一导入路径。
"""
from config.memory import (
    EmbeddingSettings,
    MemoryLLMSettings,
    OnlineExtractionSettings,
    MemorySettings,
)

__all__ = [
    "EmbeddingSettings",
    "MemoryLLMSettings",
    "OnlineExtractionSettings",
    "MemorySettings",
]
