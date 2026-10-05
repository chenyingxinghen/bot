"""真实全链路验证：用真实 .env 配置 + 本机真实后端（Ollama / ComfyUI / SillyTavern）。

覆盖：
1. 三种模式切换（酒馆 gemma / 模仿 myclone-vl / 作家）并真实调 LLM 出文字；
2. SillyTavern 角色卡发现；
3. ComfyUI 真实文生图出文件；
4. 拆分逻辑：文字回复先发、带生图动作异步补发（kind="image" 单独进队列），
   且异步补发的图片确实由 ComfyUI 生成落盘。

不依赖真实 QQ（用 FakeBot 接管 bot.send，仅验证 bot 编排逻辑 + 真实后端交互）。
单项后端失败只记 FAIL，不中断其它项。
"""

import sys
import time
import asyncio
from pathlib import Path

# 把 bot 根加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 防止 nonebot.init 误解析测试脚本自身的 argv
sys.argv = ["bot"]

import app  # 触发 nonebot.init + build_services（与生产同一条装配路径）

from core.qq.interactions import ReplyAction


checks: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, extra: str = "") -> None:
    checks.append((name, ok, extra))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {extra}" if extra else ""))


def _gen_dir():
    svc = app.services.image_service
    if svc is not None and getattr(svc, "client", None) is not None:
        return svc.client.s.output_dir
    return Path("data/generated")


class FakeBot:
    """接管 bot.send：仅记录并返回伪 message_id，不真正下发到 QQ。"""

    def __init__(self) -> None:
        self._n = 0

    async def send(self, event, message, **kwargs):
        self._n += 1
        return {"message_id": self._n}


class SendRecorder:
    """包裹真实 qq_sender.send，记录每次发送的类型与顺序。"""

    def __init__(self, real_send) -> None:
        self.real = real_send
        self.calls: list[tuple[str, float]] = []  # (label, 相对时刻)
        self._t0 = time.monotonic()

    async def send(self, bot, event, actions):
        is_image = any(bool(a.image_prompt) for a in actions)
        label = "image" if is_image else "text"
        self.calls.append((label, time.monotonic() - self._t0))
        return await self.real(bot, event, actions)


async def main() -> None:
    services = app.services
    q = services.inference_queue

    # ── 阶段 1：三种模式 + 真实 LLM 文字 ──
    try:
        modes = services.mode_manager
        # 切换并验证模型名
        for key in ("tavern", "clone", "writer"):
            m = modes.set(("e2e_bot", "e2e_user"), key)
            check(f"模式切换->{key}", m is not None and m.key == key,
                  m.model if m else "None")
        # 各模式真实调 LLM（酒馆走 SillyTavern/Ollama；其余直连 Ollama）
        for key in ("clone", "tavern", "writer"):
            modes.set(("e2e_bot", "e2e_user"), key)
            try:
                acts = await services.conversation_service.generate_reply(
                    "e2e_bot", "e2e_user", "用一句话介绍你自己。", []
                )
                text = " ".join(a.text for a in acts if a.text).strip()
                check(f"真实LLM出文字[{key}]", bool(text),
                      text[:40] + ("…" if len(text) > 40 else ""))
            except Exception as exc:
                check(f"真实LLM出文字[{key}]", False, f"{type(exc).__name__}: {exc}")
    except Exception as exc:
        check("阶段1-模式/LLM", False, f"{type(exc).__name__}: {exc}")

    # ── 阶段 2：SillyTavern 角色卡发现 ──
    try:
        names = services.sillytavern_client.list_character_names()
        check("SillyTavern角色卡发现", len(names) > 0, f"找到 {len(names)} 个：{names[:3]}")
    except Exception as exc:
        check("SillyTavern角色卡发现", False, f"{type(exc).__name__}: {exc}")

    # ── 阶段 3：ComfyUI 真实文生图 ──
    real_img_path = None
    try:
        before = set(_gen_dir().glob("*")) if _gen_dir().exists() else set()
        paths = await services.image_service.generate("一只在星空下睡觉的橘猫，油画风格")
        after = set(_gen_dir().glob("*")) if _gen_dir().exists() else set()
        new = after - before
        ok = bool(paths) and all(p.exists() for p in paths) and bool(new)
        real_img_path = paths[0] if paths else None
        check("ComfyUI真实生图出文件", ok,
              f"返回 {len(paths)} 张，落盘 {len(new)} 张")
    except Exception as exc:
        check("ComfyUI真实生图出文件", False, f"{type(exc).__name__}: {exc}")

    # ── 阶段 4：拆分逻辑（文字先回 / 图片异步补发）─
    try:
        # 真实 qq_sender.send 包一层记录器
        recorder = SendRecorder(services.qq_sender.send)
        services.qq_sender.send = recorder.send

        # 让 generate_reply 直接返回「文字 + 生图」两条动作，验证拆分与异步补发
        async def fake_reply(self_id, user_id, text, images, *, event):
            return [
                ReplyAction(text="我先回你一句文字～"),
                ReplyAction(image_prompt="赛博朋克风格的未来城市夜景"),
            ]

        services.conversation_service.generate_reply = fake_reply

        from nonebot.adapters.onebot.v11 import PrivateMessageEvent

        gen_dir = _gen_dir()
        before_phase4 = set(gen_dir.glob("*")) if gen_dir.exists() else set()

        EVENT = PrivateMessageEvent.model_validate(
            {
                "time": 1700000000, "self_id": 999, "post_type": "message",
                "message_type": "private", "sub_type": "friend",
                "message_id": 7, "user_id": 111, "message": "来张图",
                "raw_message": "来张图", "font": 0,
                "sender": {"user_id": 111, "nickname": "tester", "sex": "unknown", "age": 0},
            }
        )
        bot = FakeBot()
        await app.handle_private_msg(bot, EVENT)

        # 等到：text 与 image 两次发送都发生，且异步生图【真正落盘】（新文件出现），
        # 而非仅发起（否则 shutdown 可能在 ComfyUI 出图途中取消图片任务）。
        deadline = time.monotonic() + 300
        new_files: set = set()
        while time.monotonic() < deadline:
            await asyncio.sleep(0.2)
            labels = [c[0] for c in recorder.calls]
            if "text" in labels and "image" in labels and gen_dir.exists():
                new_files = set(gen_dir.glob("*")) - before_phase4
                if new_files:
                    break

        labels = [c[0] for c in recorder.calls]
        check("拆分-文字已发送", "text" in labels)
        check("拆分-图片异步补发已发送", "image" in labels)
        if "text" in labels and "image" in labels:
            check("拆分-文字先于图片",
                  labels.index("text") < labels.index("image"),
                  f"顺序={labels}")
        check("拆分-异步生图已落盘(新文件)", bool(new_files),
              f"本次新增 {len(new_files)} 个文件")
        check("队列-两任务均完成无异常",
              q.stats.completed >= 2,
              f"completed={q.stats.completed} failed={q.stats.failed} rejected={q.stats.rejected}")
    except Exception as exc:
        check("阶段4-拆分逻辑", False, f"{type(exc).__name__}: {exc}")
    finally:
        try:
            await q.shutdown()
        except Exception:
            pass

    ok = all(ok_ for _, ok_, _ in checks)
    print("ALL E2E PASS" if ok else "SOME E2E CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
