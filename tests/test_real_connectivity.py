"""真实连通性测试：用 .env 里的真实配置装配 services，实际探测各后端可达性。

运行（在 bot/ 目录，需装好依赖的 Python）：
    python tests/test_real_connectivity.py

会真实发起请求：
- 本地 Ollama LLM（chat completion，走 app.services.llm_service）
- 本地 Ollama Embedding（/api/embed，走 app.services.embedding_client）
- ComfyUI（HTTP 探测，不真正生图）
- SillyTavern 角色目录（本地文件读取，非网络）

后端未启动时会如实报告 FAIL 及具体错误（这就是连通性测试想看到的结果）。
"""

from __future__ import annotations

import asyncio
import os
import sys
import traceback
from pathlib import Path

BOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BOT_DIR))
os.chdir(BOT_DIR)  # nonebot 从 cwd 加载 .env

results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))


async def main() -> None:
    # 用真实配置构建 services（同时验证 build_services + app 入口装配）
    try:
        import app  # noqa: F401
    except Exception:  # pragma: no cover
        traceback.print_exc()
        record("import.app", False, "无法导入 app（依赖/配置问题）")
        sys.exit(1)
    record("import.app", True, "services 已用真实 .env 配置装配")

    services = app.services

    # 0) 构造完整性：Services 必须携带 online_extractor（曾漏装导致运行时 AttributeError）
    record(
        "services.online_extractor",
        getattr(services, "online_extractor", None) is not None,
        "online_extractor 已装配",
    )

    # 1) Ollama LLM 真实调用
    try:
        model = services.llm_service.s.model
        reply = await services.llm_service.call(
            [{"role": "user", "content": "只回复一个字：好"}],
            model=model, vision=False, temperature=0.0, num_predict=8,
        )
        record("ollama.llm", bool(reply and reply.strip()),
               f"model={model} reply={reply!r}")
    except Exception as exc:  # pragma: no cover
        record("ollama.llm", False, f"{type(exc).__name__}: {exc}")

    # 2) Ollama Embedding 真实调用
    try:
        vecs = await services.embedding_client.embed_async(["你好世界"])
        dim = len(vecs[0]) if vecs and vecs[0] else 0
        record("ollama.embedding", dim > 0,
               f"model={services.embedding_client.model} dim={dim}")
    except Exception as exc:  # pragma: no cover
        record("ollama.embedding", False, f"{type(exc).__name__}: {exc}")

    # 3) ComfyUI 可达性探测（轻量 GET，不真正生图）
    try:
        import httpx
        url = str(app.app_config.image.comfyui.url).rstrip("/") + "/"
        resp = httpx.get(url, timeout=5.0)
        record("comfyui.reachable", resp.status_code < 500,
               f"{url} -> HTTP {resp.status_code}")
    except Exception as exc:  # pragma: no cover
        record("comfyui.reachable", False, f"{type(exc).__name__}: {exc}")

    # 4) SillyTavern 角色目录（本地文件，非网络）
    try:
        names = services.sillytavern_client.list_character_names()
        record("sillytavern.chars", isinstance(names, list),
               f"找到 {len(names)} 个角色卡")
    except Exception as exc:  # pragma: no cover
        record("sillytavern.chars", False, f"{type(exc).__name__}: {exc}")

    # 收尾：关闭推理队列，避免进程挂起
    try:
        await services.inference_queue.shutdown()
    except Exception:
        pass

    failed = [n for n, ok, _ in results if not ok]
    print()
    if failed:
        print(f"{len(failed)} 项未通过：{', '.join(failed)}")
        sys.exit(1)
    print("所有真实连通性检查通过。")


if __name__ == "__main__":
    asyncio.run(main())
