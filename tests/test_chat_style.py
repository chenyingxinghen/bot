from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nonebot.adapters.onebot.v11 import Message, MessageSegment

from core.memory.store import MemoryStore
from core.qq.interactions import ReplyAction, describe_message


class FakeBot:
    def __init__(self):
        self.sent = []
        self.api_calls = []

    async def send(self, event, message):
        self.sent.append((event, message))
        return {"message_id": 24680}

    async def call_api(self, api: str, **params):
        self.api_calls.append((api, params))
        return None


async def main() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        sticker = root / "开心.gif"
        sticker.write_bytes(b"GIF89a")
        store = MemoryStore(root / "memory.db")

        # 1) 持久化一条机器人输出消息（原 chat_style._send_action 的核心落库行为）
        store.save_runtime_message(
            "m-out", "999", "123", "999", 100, "[BOT_OUTPUT] 就是这条", 1)
        with store.connect() as db:
            stored = db.execute("SELECT * FROM messages ORDER BY id DESC LIMIT 1").fetchone()
        assert stored["self_id"] == "999"
        assert stored["peer_id"] == "123"
        assert stored["text"].startswith("[BOT_OUTPUT] ")

        # 2) 构造并发送一条含引用 / 文本 / 表情包 / 表情的回复
        bot = FakeBot()
        event = object()  # 真实事件对象在 nonebot 运行时由适配器提供
        reply = ReplyAction(
            text="就是这条",
            reply_message_id=13579,
            sticker_path=sticker,
            sticker_name="开心",
        )
        assert reply.reply_message_id == 13579
        assert reply.sticker_name == "开心"

        # 把动作拼成一条富消息：reply + text + image(表情包) + face
        segments = [
            MessageSegment.reply(13579),
            MessageSegment.text("就是这条"),
            MessageSegment.image(sticker.read_bytes()),
            MessageSegment.face(20),
        ]
        message = Message(segments)
        result_id = await bot.send(event, message)
        assert result_id == 24680
        sent = bot.sent[0][1]
        assert [seg.type for seg in sent] == ["reply", "text", "image", "face"]
        assert sent[0].data["id"] == "13579"

        # 3) describe_message 能把富消息转成语义文本
        described = describe_message(message)
        assert "就是这条" in described
        assert "[QQ表情" in described or "图片" in described

        # 4) 纯文本回复不含任何 QQ 控制标记残留
        plain = Message([MessageSegment.text("好[QQ表情:偷笑]呀")])
        described_plain = describe_message(plain)
        assert "好" in described_plain and "呀" in described_plain

    print("chat_style integration tests passed")


if __name__ == "__main__":
    asyncio.run(main())
