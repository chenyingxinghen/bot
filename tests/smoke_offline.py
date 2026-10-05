"""离线冒烟测试：验证 上下文→prompt→解析→落库 管线（不依赖真实 Ollama/ComfyUI）。

运行：在 bot/ 目录下用任意装好依赖的 Python 执行
    python tests/smoke_offline.py

全部用 mock LLM + 内存 SQLite，断网可跑。覆盖：
- 组合根 import（验证 app->core->config 依赖图完整）
- 正常对话：记忆注入 system prompt、ReplyAction 解析、落库、上下文追加
- 表情包 / 生图 控制前缀解析
- /mode /status /image 命令短路
- QQSender 可构造
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

BOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BOT_DIR))

# 组合根与全部子服务能否正常导入（验证依赖图 app->core->config 完整）
from core import build_services  # noqa: F401  (仅验证导入)
from core.conversation.service import ConversationService
from core.modes.manager import Mode, ModeManager
from core.memory.store import MemoryStore
from core.qq.interactions import ReplyAction, StickerCatalog
from core.qq.sender import QQSender


class MockLLMService:
    """替代真实 LLMService.call：记录收到的 messages，返回固定回复。"""

    def __init__(self, reply: str = "你好呀，今天过得怎么样？") -> None:
        self.reply = reply
        self.calls: list[list[dict]] = []

    async def call(self, messages, *, model=None, vision=False, temperature=None, num_predict=None) -> str:
        self.calls.append([dict(m) for m in messages])
        return self.reply


class FakeEmbedding:
    def __init__(self) -> None:
        self.model = "test-embed"

    async def embed_async(self, texts):
        return [[0.1] * 64 for _ in texts]


def make_service(reply: str, *, use_memory: bool = True, sticker_file: str | None = None):
    tmp = Path(tempfile.mkdtemp(prefix="bot_smoke_"))
    db = tmp / "mem.sqlite"
    sticker_dir = tmp / "stickers"
    sticker_dir.mkdir()
    if sticker_file:
        (sticker_dir / sticker_file).write_bytes(b"GIF89a")

    store = MemoryStore(db)
    mock_llm = MockLLMService(reply)
    fake_embed = FakeEmbedding()

    conv_settings = SimpleNamespace(
        context_timeout=1800, max_context=20, max_reply_chars=80, max_reply_actions=6,
        max_bubble_chars=200, llm_echo_retry_context=6, bot_output_prefix="[BOT_OUTPUT] ",
    )
    qq_settings = SimpleNamespace(
        sticker_dir=sticker_dir, sticker_enabled=True, quote_history_size=8
    )

    modes = {
        "clone": Mode(key="clone", label="人类模仿", model="m", vision_model="m",
                      vision=True, prompt="你是模仿者。", temperature=0.6, num_predict=192,
                      use_memory=use_memory, qq_controls=True, image_tool=True),
        "tavern": Mode(key="tavern", label="酒馆角色扮演", model="t", vision_model="t",
                       vision=False, prompt="你在角色扮演。", temperature=0.9, num_predict=256,
                       use_memory=False, qq_controls=True, image_tool=True),
        "writer": Mode(key="writer", label="小说作家", model="w", vision_model="w",
                       vision=False, prompt="你在写小说。", temperature=0.8, num_predict=512,
                       use_memory=False, qq_controls=False, image_tool=True,
                       max_reply_chars=300, max_reply_actions=20),
    }
    mm = ModeManager(modes, state_path=tmp / "modes_state.json", default="clone")

    svc = ConversationService(
        memory_stores={"clone": store, "tavern": store, "writer": store},
        llm_service=mock_llm,
        mode_manager=mm,
        conversation_settings=conv_settings,
        qq_settings=qq_settings,
        sillytavern_client=None,
        character_selection=None,
        user_name="你",
        embedding_client=fake_embed,
        online_extractor=None,
        image_service=None,
        inference_queue=None,
    )
    return svc, store, mock_llm, mm


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        raise AssertionError(f"{name} FAILED: {detail}")


async def main() -> None:
    # 1) 记忆注入 + 正常对话 + 落库
    svc, store, mock_llm, mm = make_service("我也很高兴认识你！", use_memory=True)
    store.save_memories([{
        "owner_id": "bot1", "kind": "person", "subject_id": "user123",
        "summary": "用户喜欢吃苹果", "keywords": ["苹果", "水果"], "confidence": 0.9,
    }], evidence_ids=["seed"])
    actions = await svc.generate_reply("bot1", "user123", "我喜欢吃苹果")
    check("normal.chat.produces_reply",
          len(actions) == 1 and actions[0].text == "我也很高兴认识你！", repr(actions))
    sys_msg = next((m for m in mock_llm.calls[-1] if m["role"] == "system"), None)
    check("memory.injected_into_prompt",
          sys_msg is not None and "用户喜欢吃苹果" in sys_msg["content"],
          (sys_msg["content"][:120] if sys_msg else "NO SYSTEM MSG"))
    # get_online_extraction_batch 只回传部分列，这里直接查库验证 send_type
    with store.connect() as db:
        rows = db.execute(
            "SELECT sender_id, send_type, text FROM messages "
            "WHERE peer_id='user123' AND source_file='runtime' ORDER BY id"
        ).fetchall()
    check("persist.runtime_messages", len(rows) == 2, f"got {len(rows)} rows")
    if rows:
        check("persist.user_msg",
              rows[0]["text"] == "我喜欢吃苹果" and rows[0]["send_type"] == 0
              and rows[0]["sender_id"] == "user123")
        check("persist.bot_msg",
              rows[1]["text"] == "我也很高兴认识你！" and rows[1]["send_type"] == 1
              and rows[1]["sender_id"] == "bot1")
    st = store.get_interaction_state("bot1", "user123", mode_key="clone")
    check("persist.interaction_state",
          st is not None and st["last_user_at"] and st["last_bot_at"],
          str(dict(st)) if st else "None")
    from core.conversation.context import _get_conversation
    ctx = _get_conversation("bot1", "user123", 1800, mode_key="clone")
    check("context.assistant_added", any(m["role"] == "assistant" for m in ctx),
          f"ctx len={len(ctx)}")

    # 2) 表情包解析
    svc2, store2, mock2, _ = make_service("[表情: doge] 哈哈", sticker_file="doge.gif")
    acts2 = await svc2.generate_reply("bot1", "u2", "发个表情")
    check("sticker.resolved",
          len(acts2) == 1 and acts2[0].sticker_name == "doge" and acts2[0].text == "哈哈",
          repr(acts2))
    check("sticker.path_set", bool(acts2) and acts2[0].sticker_path is not None,
          str(acts2[0].sticker_path) if acts2 else "")

    # 2b) 引用历史消息（真实 message_id 解析）
    svc_q, _, mock_q, _ = make_service("[回复: m1] 收到", sticker_file=None)
    fake_event = SimpleNamespace(message_id=123)
    acts_q = await svc_q.generate_reply("bot1", "uq", "在吗", event=fake_event)
    check("quote.resolves_real_id",
          len(acts_q) == 1 and acts_q[0].reply_message_id == 123 and acts_q[0].text == "收到",
          repr(acts_q))
    sys_q = next((m for m in mock_q.calls[-1] if m["role"] == "system"), {})
    user_q = next((m for m in mock_q.calls[-1] if m["role"] == "user"), {})
    check("quote.labels_injected_without_history_duplication",
          "带 [mN] 前缀" in sys_q.get("content", "")
          and user_q.get("content", "").startswith("[m1] ")
          and "m1（对方）" not in sys_q.get("content", ""),
          str(sys_q.get("content", ""))[:120])

    # 3) LLM 发起生图
    svc3, _, mock3, _ = make_service("[生图: 一只橘猫坐在窗台]", sticker_file=None)
    acts3 = await svc3.generate_reply("bot1", "u3", "给我画只猫")
    check("image.prompt_parsed",
          len(acts3) == 1 and acts3[0].image_prompt == "一只橘猫坐在窗台", repr(acts3))

    # 4) 命令：/mode
    svc4, _, _, mm4 = make_service("x")
    res = await svc4.generate_reply("bot1", "u4", "/mode 酒馆")
    check("cmd.mode_switch", len(res) == 1 and "酒馆角色扮演" in res[0].text, repr(res))
    check("cmd.mode_persisted", mm4.get(("bot1", "u4")).key == "tavern",
          mm4.get(("bot1", "u4")).key)

    # 5) 命令：/status
    res5 = await svc4.generate_reply("bot1", "u4", "/status")
    check("cmd.status", len(res5) == 1 and "当前模式" in res5[0].text, repr(res5))

    # 6) 命令：/image
    res6 = await svc4.generate_reply("bot1", "u4", "/image 一只柴犬")
    check("cmd.image", len(res6) == 1 and res6[0].image_prompt == "一只柴犬", repr(res6))

    # 7) QQSender 可构造（不发送）
    svc7, store7, _, _ = make_service("x")
    sender = QQSender(settings=svc7.qq_settings,
                      sticker_catalog=StickerCatalog(svc7.qq_settings.sticker_dir))
    check("qqsender.instantiable", sender is not None)

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
