"""handler 级冒烟测试：真正调用 app.handle_private_msg，验证运行时接线。

此测试直接 import app（与生产启动同一条装配路径：nonebot.init -> build_services），
然后构造一个 OneBot 私有消息事件、mock 掉重推理（LLM/发信），调用真实 handler。
目的：catch 那种「只装配 services 不跑 handler」测不出来的运行时属性缺失
（例如 services.online_extractor 之前漏装进 Services 数据类，导致 AttributeError）。

不依赖真实 Ollama / ComfyUI / QQ；队列任务在内存里跑完即验证。
"""

import sys
from pathlib import Path

# 把 bot 根目录加入 sys.path，使 `import app / config / core` 可用
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 防止 nonebot.init 误解析测试脚本自身的 argv
sys.argv = ["bot"]

import asyncio

from nonebot.adapters.onebot.v11 import PrivateMessageEvent

import app  # 触发 nonebot.init + build_services（与生产启动一致）
from core.qq.interactions import ReplyAction


checks: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, extra: str = "") -> None:
    checks.append((name, ok, extra))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {extra}" if extra else ""))


# --- mock 重推理，避免真实调 LLM / 发信 ---
# 注意：挂到实例属性上是普通函数调用（不会自动绑定 self），签名需与调用处一致
_gen = {"n": 0}


async def fake_generate_reply(self_id, user_id, text, images, *, event):
    _gen["n"] += 1
    return [ReplyAction(text="ok")]


_sent: list = []


async def fake_send(bot, event, actions):
    _sent.append(actions)
    return [123]


app.services.conversation_service.generate_reply = fake_generate_reply
app.services.qq_sender.send = fake_send


EVENT = PrivateMessageEvent.model_validate(
    {
        "time": 1700000000,
        "self_id": 999,
        "post_type": "message",
        "message_type": "private",
        "sub_type": "friend",
        "message_id": 1,
        "user_id": 111,
        "message": "hello",
        "raw_message": "hello",
        "font": 0,
        "sender": {"user_id": 111, "nickname": "tester", "sex": "unknown", "age": 0},
    }
)


async def main() -> None:
    # 成功路径由 QQPlatformSender 委托给已 mock 的 qq_sender，不调用 bot.send。
    bot = object()
    await app.handle_private_msg(bot, EVENT)

    # 让推理队列把 _run 跑完（worker 在运行中的事件循环里惰性启动）
    q = app.services.inference_queue
    for _ in range(150):
        await asyncio.sleep(0.02)
        if q.stats.completed >= 1 or q.stats.failed >= 1:
            break
    await q.shutdown()

    check("services.online_extractor 存在", app.services.online_extractor is not None)
    check(
        "online_extractor 不再依赖 handler 覆盖共享 self_id",
        app.services.online_extractor.self_id == "",
        str(app.services.online_extractor.self_id),
    )
    check("generate_reply 被调用", _gen["n"] >= 1)
    check("qq_sender.send 被调用", len(_sent) >= 1)
    check("队列任务完成(无异常)", q.stats.completed >= 1, f"completed={q.stats.completed} failed={q.stats.failed}")

    ok = all(ok_ for _, ok_, _ in checks)
    print("ALL HANDLER SMOKE PASS" if ok else "SOME HANDLER CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
