from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

from core.image.comfyui import ComfyUIClient, ComfyUISettings
from core.queue.queue import InferenceQueue
from core.modes.manager import Mode, ModeManager, build_modes
from core.qq.interactions import ReplyAction, StickerCatalog, parse_reply_actions


BOT_ROOT = Path(__file__).resolve().parents[1]


# ─── 生图标记解析 ──────────────────────────────────────

def test_parse_image_marker():
    catalog = StickerCatalog(BOT_ROOT / "data" / "stickers")
    # 单句带生图标记（须写在句子最前面，与 [[reply:]]/[[sticker:]] 一致）；
    # image_prompt 应出现在最后一个 chunk，且控制标记被剥离
    acts = parse_reply_actions(
        "[[image: 一只橘猫]] 看这里", {}, catalog, max_chars=80, max_actions=6)
    assert any(a.image_prompt for a in acts)
    last = acts[-1]
    assert last.image_prompt == "一只橘猫"
    assert "[[image:" not in last.text
    assert "看这里" in last.text

    # 生图别名 [[生图: ...]]
    acts = parse_reply_actions(
        "[[生图: 星空下的城市]] 送给你的", {}, catalog, max_chars=80, max_actions=6)
    assert acts[-1].image_prompt == "星空下的城市"
    assert "星空下的城市" not in acts[-1].text
    assert "送给你的" in acts[-1].text

    # 只有生图标记、没有文字：生成独立动作
    acts = parse_reply_actions(
        "[[image: 纯风景]]", {}, catalog, max_chars=80, max_actions=6)
    assert len(acts) == 1
    assert acts[0].image_prompt == "纯风景"
    assert acts[0].text == ""


def test_writer_image_marker_before_body_survives_bubble_merge():
    """模型把独立生图行放在长正文前时，后续纯文本不得覆盖生图副作用。"""
    catalog = StickerCatalog(BOT_ROOT / "data" / "stickers")
    raw = (
        "[生图: warm amber booth, intimate cinematic portrait]\n"
        "【时间】：2026年8月5日 星期三 00:42\n"
        "第一段正文。\n"
        "第二段正文。"
    )
    acts = parse_reply_actions(
        raw, {}, catalog, max_chars=15000, max_actions=60, max_bubble_chars=200
    )
    image_actions = [action for action in acts if action.image_prompt]
    assert len(image_actions) == 1
    assert image_actions[0].image_prompt == (
        "warm amber booth, intimate cinematic portrait"
    )
    assert any("第一段正文" in action.text for action in acts)


def test_parse_no_split_for_writer():
    catalog = StickerCatalog(BOT_ROOT / "data" / "stickers")
    # 单行超长文本，max_chars<=0 表示不按字符切分（小说作家模式）
    long_text = "第二段比较长" * 20
    acts = parse_reply_actions(long_text, {}, catalog, max_chars=0, max_actions=0)
    assert len(acts) == 1
    assert acts[0].text == long_text


# ─── 模式管理器持久化 ──────────────────────────────────

def _make_modes():
    base = dict(vision_model="vm", vision=False, prompt="p", temperature=0.8,
                num_predict=200, use_memory=False, qq_controls=True, image_tool=True)
    return {
        "tavern": Mode(key="tavern", label="酒馆", model="m1", **base),
        "clone": Mode(key="clone", label="模仿", model="m2", **base),
        "writer": Mode(key="writer", label="作家", model="m3", **base),
    }


def test_mode_manager_persistence():
    with tempfile.TemporaryDirectory() as tmp:
        state = Path(tmp) / "modes_state.json"
        mgr = ModeManager(_make_modes(), state, default="clone")
        assert mgr.get(("s", "u")).key == "clone"
        assert mgr.set(("s", "u"), "tavern").key == "tavern"
        # 重新加载应保留状态
        mgr2 = ModeManager(_make_modes(), state, default="clone")
        assert mgr2.get(("s", "u")).key == "tavern"
        assert mgr2.set(("s", "u"), "nope") is None  # 未知模式不生效


# ─── 推理队列 ──────────────────────────────────────────

async def _noop(i, results):
    results.append(i)


def test_inference_queue_processes_and_counts():
    async def _run():
        q = InferenceQueue(max_workers=2, max_pending=10)
        q.start()
        results = []
        for i in range(5):
            await q.submit(lambda i=i: _noop(i, results))
        for _ in range(100):
            if q.stats.completed >= 5:
                break
            await asyncio.sleep(0.02)
        assert sorted(results) == [0, 1, 2, 3, 4]
        assert q.stats.completed == 5
        await q.shutdown()
        # 关闭后 workers 已清空
        assert q._workers == []
    asyncio.run(_run())


def test_start_is_safe_without_event_loop():
    # 复现启动钩子在同步线程（无运行中的事件循环）中被调用的场景：
    # 旧实现会在 start() 里直接 create_task，抛 RuntimeError:
    # "no running event loop"。修复后 start() 必须安全无副作用。
    q = InferenceQueue(max_workers=2, max_pending=5)
    q.start()  # 同步调用，当前线程无 loop，不得抛异常
    assert q._workers == []  # 未惰性建协程（要等首次 submit）
    # 显式在另一条无 loop 的线程里再调一次，进一步确认安全
    import threading

    err = {}

    def _call():
        try:
            q.start()
        except Exception as exc:  # noqa: BLE001
            err["e"] = exc

    t = threading.Thread(target=_call)
    t.start()
    t.join()
    assert "e" not in err, f"start() 在无 loop 线程抛异常 {err.get('e')}"


def test_inference_queue_rejects_when_full():
    async def _run():
        # auto_start=False：不创建 worker，任务纯堆积，用于精确验证队列容量上限
        q = InferenceQueue(max_workers=1, max_pending=2, auto_start=False)
        ok1 = await q.submit(lambda: _noop(0, []))
        ok2 = await q.submit(lambda: _noop(1, []))
        ok3 = await q.submit(lambda: _noop(2, []))
        assert ok1 and ok2 and not ok3
        assert q.stats.rejected == 1
        assert q.stats.submitted == 3
        await q.shutdown()
    asyncio.run(_run())


# ─── ComfyUI 客户端（mock httpx）──────────────────────

class _FakeResponse:
    def __init__(self, json_data=None, content=b"", headers=None):
        self._json = json_data or {}
        self.content = content
        self.headers = headers or {}

    def raise_for_status(self):
        return None

    def json(self):
        return self._json


class _FakeAsyncClient:
    def __init__(self, *args, **kwargs):
        self._prompt_id = "pid-abc"

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None):
        return _FakeResponse({"prompt_id": self._prompt_id})

    async def get(self, url):
        if url.endswith(f"/history/{self._prompt_id}"):
            return _FakeResponse({
                self._prompt_id: {
                    "outputs": {"7": {"images": [
                        {"filename": "out.png", "subfolder": "", "type": "output"}
                    ]}}
                }
            })
        if "/view" in url:
            return _FakeResponse(content=b"PNGDATA", headers={"content-type": "image/png"})
        return _FakeResponse({})


class _RejectingAsyncClient:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None):
        request = httpx.Request("POST", url)
        return httpx.Response(
            400,
            request=request,
            json={
                "error": {"type": "prompt_outputs_failed_validation"},
                "node_errors": {
                    "66": {"errors": [{"details": "unet_name not in list"}]}
                },
            },
        )


def test_comfyui_patch_and_generate(tmp_path):
    workflow = Path(tmp_path) / "workflow.json"
    workflow.write_text(json.dumps({
        "2": {"inputs": {"text": ""}},
        "3": {"inputs": {"text": ""}},
        "4": {"inputs": {"width": 512, "height": 512}},
        "5": {"inputs": {"seed": 1, "steps": 1, "cfg": 1,
                           "sampler_name": "euler", "scheduler": "normal", "denoise": 1}},
        "7": {"inputs": {}},
    }), encoding="utf-8")
    settings = ComfyUISettings(
        url="http://fake", workflow_path=workflow,
        positive_node="2", positive_key="text",
        negative_node="3", negative_key="text",
        seed_node="5", seed_key="seed",
        size_node="4", width_key="width", height_key="height",
        output_node="7", ckpt_node=None, ckpt_key="ckpt_name", ckpt_name=None,
        negative_prompt="", width=768, height=768, steps=28, cfg=7.0,
        sampler="euler", scheduler="normal", denoise=1.0, seed=None,
        timeout=10, poll_interval=0.01, max_images=4, max_image_bytes=1024,
        output_dir=Path(tmp_path), max_concurrent_images=1, sampler_node="5",
    )
    client = ComfyUIClient(settings)
    # 纯逻辑：_patch 应正确替换正/负提示词、种子、尺寸
    wf = client._load_workflow()
    patched = client._patch(wf, "a cat", "bad", 42, 1024, 768)
    assert patched["2"]["inputs"]["text"] == "a cat"
    assert patched["3"]["inputs"]["text"] == "bad"
    assert patched["5"]["inputs"]["seed"] == 42
    assert patched["4"]["inputs"]["width"] == 1024
    assert patched["4"]["inputs"]["height"] == 768
    assert patched["5"]["inputs"]["steps"] == 28
    assert patched["5"]["inputs"]["cfg"] == 7.0

    with mock.patch("httpx.AsyncClient", _FakeAsyncClient):
        paths = asyncio.run(client.generate("a cat"))
    assert len(paths) == 1
    assert paths[0].read_bytes() == b"PNGDATA"


def test_comfyui_submit_error_includes_response_body(tmp_path):
    workflow = Path(tmp_path) / "workflow.json"
    workflow.write_text(json.dumps({
        "2": {"inputs": {"text": ""}},
        "3": {"inputs": {"text": ""}},
        "4": {"inputs": {"width": 512, "height": 512}},
        "5": {"inputs": {"seed": 1, "steps": 1, "cfg": 1,
                           "sampler_name": "euler", "scheduler": "normal", "denoise": 1}},
        "7": {"inputs": {}},
    }), encoding="utf-8")
    settings = ComfyUISettings(
        url="http://fake", workflow_path=workflow,
        positive_node="2", positive_key="text",
        negative_node="3", negative_key="text",
        seed_node="5", seed_key="seed",
        size_node="4", width_key="width", height_key="height",
        output_node="7", ckpt_node=None, ckpt_key="ckpt_name", ckpt_name=None,
        negative_prompt="", width=768, height=768, steps=28, cfg=7.0,
        sampler="euler", scheduler="normal", denoise=1.0, seed=None,
        timeout=10, poll_interval=0.01, max_images=4, max_image_bytes=1024,
        output_dir=Path(tmp_path), max_concurrent_images=1, sampler_node="5",
    )
    client = ComfyUIClient(settings)
    with mock.patch("httpx.AsyncClient", _RejectingAsyncClient):
        try:
            asyncio.run(client.generate("a cat"))
        except RuntimeError as exc:
            message = str(exc)
        else:
            raise AssertionError("ComfyUI 400 应抛出 RuntimeError")
    assert "HTTP 400" in message
    assert "prompt_outputs_failed_validation" in message
    assert "node_errors" in message
    assert "unet_name not in list" in message


if __name__ == "__main__":
    test_inference_queue_processes_and_counts()
    test_inference_queue_rejects_when_full()
    test_comfyui_patch_and_generate(tempfile.mkdtemp())
    test_comfyui_submit_error_includes_response_body(tempfile.mkdtemp())
    test_parse_image_marker()
    test_parse_no_split_for_writer()
    test_mode_manager_persistence()
    print("modes_queue tests passed")
