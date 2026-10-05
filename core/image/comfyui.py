"""ComfyUI 文生图客户端。

把文本提示词封装成 ComfyUI 的工作流，提交到 `/prompt`，轮询 `/history` 拿到输出
节点里的图片，再经 `/view` 下载到本地返回路径。

workflow 由用户从自己的 ComfyUI 里导出（菜单 Export (API Format)），节点位置通过
配置指定。当前生产工作流由 COMFYUI_WORKFLOW 指向 data/t2i.json；更换工作流时需要
同步更新正向、负向、采样、尺寸、输出和模型加载节点映射。
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

import asyncio
import copy
import json
import random
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode

import httpx



@dataclass(frozen=True)
class ComfyUISettings:
    url: str
    workflow_path: Path
    positive_node: str
    positive_key: str
    negative_node: str
    negative_key: str
    seed_node: str
    seed_key: str
    size_node: str
    width_key: str
    height_key: str
    output_node: str
    ckpt_node: str | None
    ckpt_key: str
    ckpt_name: str | None
    negative_prompt: str
    width: int
    height: int
    steps: int
    cfg: float
    sampler: str
    scheduler: str
    denoise: float
    seed: int | None
    timeout: float
    poll_interval: float
    max_images: int
    max_image_bytes: int
    output_dir: Path
    max_concurrent_images: int
    sampler_node: str | None = None

    @classmethod
    def from_config(cls, config: object, bot_root: Path) -> "ComfyUISettings":
        def env(name: str, default):
            return getattr(config, name, default)

        seed_raw = env("comfyui_seed", "")
        seed = (
            int(seed_raw)
            if str(seed_raw).strip() not in ("", "-1", "None", "null")
            else None
        )
        return cls(
            url=str(env("comfyui_url", "http://127.0.0.1:8188")).rstrip("/"),
            workflow_path=(
                bot_root / str(env("comfyui_workflow", "data/t2i.json"))
            ).resolve(),
            positive_node=str(env("comfyui_positive_node", "2")),
            positive_key=str(env("comfyui_positive_key", "text")),
            negative_node=str(env("comfyui_negative_node", "3")),
            negative_key=str(env("comfyui_negative_key", "text")),
            seed_node=str(env("comfyui_seed_node", "5")),
            seed_key=str(env("comfyui_seed_key", "seed")),
            size_node=str(env("comfyui_size_node", "4")),
            width_key=str(env("comfyui_width_key", "width")),
            height_key=str(env("comfyui_height_key", "height")),
            output_node=str(env("comfyui_output_node", "7")),
            ckpt_node=(str(env("comfyui_ckpt_node", "")) or None),
            ckpt_key=str(env("comfyui_ckpt_key", "ckpt_name")),
            ckpt_name=(str(env("comfyui_ckpt_name", "")) or None),
            negative_prompt=str(env("comfyui_negative_prompt", "")),
            width=int(env("comfyui_width", 768)),
            height=int(env("comfyui_height", 768)),
            steps=int(env("comfyui_steps", 28)),
            cfg=float(env("comfyui_cfg", 7.0)),
            sampler=str(env("comfyui_sampler", "euler")),
            scheduler=str(env("comfyui_scheduler", "normal")),
            denoise=float(env("comfyui_denoise", 1.0)),
            seed=seed,
            timeout=float(env("comfyui_timeout", 300)),
            poll_interval=float(env("comfyui_poll_interval", 1.5)),
            max_images=min(
                int(env("comfyui_max_images", 4)), int(env("max_images", 4))
            ),
            max_image_bytes=int(env("max_image_bytes", 10 * 1024 * 1024)),
            output_dir=(
                bot_root / str(env("comfyui_output_dir", "data/generated"))
            ).resolve(),
            max_concurrent_images=int(env("comfyui_max_concurrent_images", 1)),
            sampler_node=(str(env("comfyui_sampler_node", env("comfyui_seed_node", "5"))) or None),
        )


class ComfyUIClient:
    def __init__(self, settings: ComfyUISettings) -> None:
        self.s = settings
        self._workflow_cache: dict | None = None
        # 限制同时进行的生图请求数（对应 .env 的 COMFYUI_MAX_CONCURRENT_IMAGES）
        self._sem = asyncio.Semaphore(max(1, int(settings.max_concurrent_images)))

    @staticmethod
    def _raise_for_status(response: httpx.Response, operation: str) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            body = response.text.strip()
            request_url = str(getattr(response.request, "url", ""))
            logger.error(
                "ComfyUI %s请求失败：status=%s url=%s body=%s",
                operation,
                response.status_code,
                request_url,
                body or "<empty>",
            )
            detail = body or str(exc)
            raise RuntimeError(
                f"ComfyUI {operation}失败（HTTP {response.status_code}）：{detail}"
            ) from exc

    def _load_workflow(self) -> dict:
        if self._workflow_cache is None:
            if not self.s.workflow_path.exists():
                raise FileNotFoundError(
                    f"未找到 ComfyUI workflow 文件：{self.s.workflow_path}\n"
                    "请在 ComfyUI 中导出 API 格式的工作流，或保留默认模板。"
                )
            self._workflow_cache = json.loads(
                self.s.workflow_path.read_text(encoding="utf-8")
            )
        return copy.deepcopy(self._workflow_cache)

    def _patch(
        self,
        workflow: dict,
        prompt: str,
        negative: str,
        seed: int | None,
        width: int,
        height: int,
    ) -> dict:
        wf = workflow
        required = {
            "positive": self.s.positive_node,
            "negative": self.s.negative_node,
            "seed": self.s.seed_node,
            "size": self.s.size_node,
            "output": self.s.output_node,
        }
        missing = [f"{label}={node}" for label, node in required.items() if node not in wf]
        if missing:
            raise ValueError("ComfyUI workflow 缺少配置节点：" + "、".join(missing))

        positive = wf[self.s.positive_node]
        positive.setdefault("inputs", {})[self.s.positive_key] = prompt
        negative_node = wf[self.s.negative_node]
        negative_node.setdefault("inputs", {})[self.s.negative_key] = negative

        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        seed_node = wf[self.s.seed_node]
        seed_node.setdefault("inputs", {})[self.s.seed_key] = seed

        size_node = wf[self.s.size_node]
        size_node.setdefault("inputs", {})[self.s.width_key] = width
        size_node.setdefault("inputs", {})[self.s.height_key] = height

        if self.s.sampler_node:
            sampler_node = wf.get(self.s.sampler_node)
            if sampler_node is None:
                raise ValueError(f"ComfyUI workflow 缺少采样节点：{self.s.sampler_node}")
            inputs = sampler_node.setdefault("inputs", {})
            inputs["steps"] = self.s.steps
            inputs["cfg"] = self.s.cfg
            inputs["sampler_name"] = self.s.sampler
            inputs["scheduler"] = self.s.scheduler
            inputs["denoise"] = self.s.denoise

        if self.s.ckpt_node and self.s.ckpt_name:
            ckpt = wf.get(self.s.ckpt_node)
            if ckpt is not None:
                ckpt.setdefault("inputs", {})[self.s.ckpt_key] = self.s.ckpt_name
        return wf

    async def generate(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        seed: int | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> list[Path]:
        """生成一张（或多张）图片，返回本地文件路径列表。"""
        width = width or self.s.width
        height = height or self.s.height
        negative = negative_prompt if negative_prompt is not None else self.s.negative_prompt
        workflow = self._patch(self._load_workflow(), prompt, negative, seed, width, height)
        # 调试可见性：记录真正提交给 ComfyUI 的提示词（截断避免刷屏）。
        # 正向提示词可能来自模型 [生图:] 标记或关键词兜底，二者都在此可见。
        try:
            actual_seed = workflow[self.s.seed_node]["inputs"].get(self.s.seed_key)
        except Exception:  # noqa: BLE001
            actual_seed = None
        logger.info(
            "ComfyUI 提交生图（%dx%d, seed=%s）：正向提示词=%r 负向提示词=%r",
            width, height, actual_seed, prompt[:600], negative[:300],
        )
        client_id = uuid.uuid4().hex
        payload = {"prompt": workflow, "client_id": client_id}
        base = self.s.url
        self.s.output_dir.mkdir(parents=True, exist_ok=True)

        # 受信号量约束：同时最多 max_concurrent_images 个生图请求打到 ComfyUI
        async with self._sem:
            async with httpx.AsyncClient(timeout=self.s.timeout, follow_redirects=True) as client:
                r = await client.post(f"{base}/prompt", json=payload)
                self._raise_for_status(r, "提交 workflow")
                data = r.json()
                prompt_id = data.get("prompt_id")
                if not prompt_id:
                    raise RuntimeError(f"ComfyUI 未返回 prompt_id：{data}")

                history_url = f"{base}/history/{prompt_id}"
                deadline = time.monotonic() + self.s.timeout
                last_status = None
                while time.monotonic() < deadline:
                    hr = await client.get(history_url)
                    self._raise_for_status(hr, "查询执行历史")
                    hdata = hr.json()
                    if prompt_id in hdata:
                        entry = hdata[prompt_id]
                        status = entry.get("status", {})
                        if status.get("status_str") == "error":
                            msgs = status.get("messages", [])
                            raise RuntimeError(f"ComfyUI 执行失败：{msgs}")
                        outputs = entry.get("outputs", {})
                        out = outputs.get(self.s.output_node, {})
                        images = out.get("images", [])
                        if images:
                            paths: list[Path] = []
                            for img in images[: self.s.max_images]:
                                fname = img.get("filename")
                                if not fname:
                                    continue
                                sub = img.get("subfolder", "")
                                itype = img.get("type", "output")
                                query = urlencode({
                                    "filename": fname,
                                    "subfolder": sub,
                                    "type": itype,
                                })
                                vr = await client.get(f"{base}/view?{query}")
                                self._raise_for_status(vr, "下载生成图片")
                                content_type = str(vr.headers.get("content-type", ""))
                                if content_type and not content_type.lower().startswith("image/"):
                                    raise RuntimeError(
                                        f"ComfyUI 返回了非图片内容：{content_type}"
                                    )
                                if len(vr.content) > self.s.max_image_bytes:
                                    raise RuntimeError(
                                        f"ComfyUI 图片超过大小限制：{len(vr.content)} > "
                                        f"{self.s.max_image_bytes}"
                                    )
                                ext = Path(fname).suffix.lower() or ".png"
                                if ext not in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
                                    ext = ".png"
                                out_name = (
                                    f"gen_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}{ext}"
                                )
                                out_path = self.s.output_dir / out_name
                                out_path.write_bytes(vr.content)
                                paths.append(out_path)
                            if paths:
                                logger.info(
                                    f"ComfyUI 生图完成（prompt_id={prompt_id}）：{paths}"
                                )
                                return paths
                        last_status = status
                    await asyncio.sleep(self.s.poll_interval)

            raise TimeoutError(
                f"ComfyUI 在 {self.s.timeout}s 内未完成生成（prompt_id={prompt_id}"
                f"），最后状态 {last_status}。请检查 ComfyUI 队列是否堆积或 workflow 是否正确。"
            )
