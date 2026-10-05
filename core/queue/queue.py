"""全局推理任务队列。

性能受限的机器上，同时跑太多 LLM 推理或 ComfyUI 生图会直接卡死。这里用
固定数量的工作协程串行消费任务，从全局层面把「消息回复」「生成图片」这两类重
任务的并发数限制住；队列满时拒绝新任务，由调用方提示用户稍后再试。
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

import asyncio
import time
from dataclasses import dataclass
from typing import Awaitable, Callable


# 一个不接收参数、返回协程的工厂；队列 worker 调用它来真正执行任务
TaskFactory = Callable[[], Awaitable[None]]


@dataclass
class QueueStats:
    submitted: int = 0
    completed: int = 0
    failed: int = 0
    rejected: int = 0
    active: int = 0
    pending: int = 0

    def as_dict(self) -> dict:
        return {
            "submitted": self.submitted,
            "completed": self.completed,
            "failed": self.failed,
            "rejected": self.rejected,
            "active": self.active,
            "pending": self.pending,
        }


class InferenceQueue:
    """固定 worker 数的异步任务队列，控制重推理任务的并发度。"""

    def __init__(
        self,
        max_workers: int = 2,
        max_pending: int = 20,
        auto_start: bool = True,
    ) -> None:
        self._max_workers = max(1, int(max_workers))
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max(1, int(max_pending)))
        self._workers: list[asyncio.Task] = []
        self._running = False
        self._auto_start = bool(auto_start)
        self._stats = QueueStats()

    @property
    def stats(self) -> QueueStats:
        # 实时反映队列中等待的任务数
        self._stats.pending = self._queue.qsize()
        return self._stats

    def start(self) -> None:
        """同步启动钩子（NoneBot on_startup 会以 run_sync 在线程中调用）。

        该线程没有运行中的事件循环，因此**不能**在此 create_task。
        这里只打日志；真正的 worker 协程会在首次 submit() 时，于运行中
        的事件循环里由 _ensure_started() 惰性创建。这样即使启动钩子被同步调用，
        队列也能在第一次提交任务时自愈，不会因无事件循环而崩溃。
        """
        logger.info(
            f"推理任务队列就绪：{self._max_workers} 个工作协程，"
            f"待处理上限 {self._queue.maxsize}"
            f"（worker 将在首次提交任务时启动）"
        )

    def _ensure_started(self) -> None:
        """在运行中的事件循环里惰性创建工作协程（首次 submit 时调用）。

        必须在已 running loop 的协程内调用（submit 是 async，天然满足）。
        """
        if self._workers:
            return
        self._running = True
        self._workers = [
            asyncio.create_task(self._worker(i), name=f"inference-worker-{i}")
            for i in range(self._max_workers)
        ]
        logger.info(
            f"推理任务队列 worker 已启动：{self._max_workers} 个工作协程"
        )

    async def submit(self, factory: TaskFactory, *, kind: str = "reply") -> bool:
        """提交一个任务工厂。

        返回 False 表示队列已满、任务被拒绝（调用方应提示用户繁忙）。
        """
        if self._auto_start:
            self._ensure_started()
        self._stats.submitted += 1
        try:
            self._queue.put_nowait((kind, factory))
            return True
        except asyncio.QueueFull:
            self._stats.rejected += 1
            logger.warning(f"推理任务队列已满，已拒绝一个 {kind} 任务")
            return False

    async def shutdown(self, timeout: float = 30.0) -> None:
        """停止接收新任务，并等待已入队任务处理完（优雅退出）。"""
        if not self._running:
            return
        self._running = False
        # 等待在途任务：队列空且没有活跃 worker 时退出
        deadline = time.monotonic() + timeout
        while (not self._queue.empty() or self._stats.active > 0) and time.monotonic() < deadline:
            await asyncio.sleep(0.2)
        for worker in self._workers:
            worker.cancel()
        self._workers.clear()
        logger.info("推理任务队列已关闭")

    async def _worker(self, worker_id: int) -> None:
        while self._running or not self._queue.empty():
            try:
                kind, factory = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                if not self._running:
                    break
                continue
            self._stats.active += 1
            try:
                if factory is not None:
                    await factory()
                self._stats.completed += 1
            except asyncio.CancelledError:
                self._stats.failed += 1
                raise
            except Exception as exc:  # noqa: BLE001
                self._stats.failed += 1
                logger.error(
                    f"推理任务执行失败 [worker-{worker_id}/{kind}]: "
                    f"{type(exc).__name__}: {exc}"
                )
            finally:
                self._stats.active = max(0, self._stats.active - 1)
                self._queue.task_done()
        logger.info(f"推理队列工作协程 worker-{worker_id} 已退出")
