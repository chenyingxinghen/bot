"""进程日志输出编码初始化。"""

from __future__ import annotations

import io
import os
import sys
from typing import TextIO


def _configure_stream_utf8(stream: TextIO | None) -> None:
    if stream is None:
        return
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8", errors="backslashreplace")


def configure_process_logging_encoding() -> None:
    """固定 stdout/stderr 为 UTF-8，避免 Windows 服务重定向写出 GBK 日志。"""
    os.environ["PYTHONIOENCODING"] = "utf-8"
    _configure_stream_utf8(sys.stdout)
    _configure_stream_utf8(sys.stderr)


def configure_app_logging(level: str | None = None) -> None:
    """把 bot 自身（stdlib logging）的应用日志桥接到 NoneBot 的 loguru（stdout）。

    NoneBot 只把 ``LoguruHandler`` 挂在 ``websockets.client`` 这一个 logger 上；
    stdlib 的 root logger 默认级别为 WARNING 且没有 handler，导致 ``core.*`` /
    ``config.*`` 等的 INFO 日志（含 LLM 输入输出）被丢弃，只有 WARNING 以上能经
    Python 的 lastResort handler 落到 stderr。这里给 root 挂上 ``LoguruHandler`` 并把
    root 级别降到 ``INFO``（或 ``LOG_LEVEL`` 指定的级别），让应用日志随 nonebot 日志
    一起进 stdout；同时关掉 uvicorn / websockets 等自带 handler 的第三方 logger 的上行
    传播，避免同一行日志重复输出。
    """
    import logging

    try:
        from nonebot.log import LoguruHandler
    except Exception:  # pragma: no cover - nonebot 未就绪时跳过
        return

    effective = (level or os.environ.get("LOG_LEVEL") or "INFO").upper()
    lvl = getattr(logging, effective, logging.INFO)

    # 这些第三方库自己已配 handler，关掉向 root 的传播，避免与下面的 root handler 重复
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "websockets.client"):
        logging.getLogger(name).propagate = False

    root = logging.getLogger()
    if not any(isinstance(h, LoguruHandler) for h in root.handlers):
        root.addHandler(LoguruHandler())
    root.setLevel(lvl)
