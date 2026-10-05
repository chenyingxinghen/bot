from __future__ import annotations

import asyncio
import tempfile
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nonebot.adapters.onebot.v11 import Message, MessageSegment

from config.qq import QQSettings
from core.qq.interactions import (
    HistoryMessage,
    ReplyAction,
    StickerCatalog,
    build_quote_options,
    describe_message,
    extract_message_id,
    parse_qq_face_segments,
    parse_reply_actions,
    strip_bare_quote_labels,
)
from core.qq.sender import QQSender


def test_message_description() -> None:
    message = Message([
        MessageSegment.text("你看"),
        MessageSegment.face(182),
        MessageSegment("image", {
            "file": "emoji.gif",
            "summary": "[动画表情]开心猫",
            "emoji_id": "abc",
        }),
        MessageSegment("record", {"file": "voice.amr"}),
    ])
    result = describe_message(message)
    assert "你看" in result
    assert "[QQ表情:笑哭]" in result
    assert "[QQ表情:[动画表情]开心猫]" in result
    assert "[语音消息]" in result

    unknown = describe_message(Message([MessageSegment.face(9999)]))
    assert unknown == "[QQ表情:id=9999]"

    rendered = parse_qq_face_segments("好[QQ表情:偷笑]呀[QQ表情:id=318]")
    assert rendered == [
        ("text", "好"),
        ("face", 20),
        ("text", "呀"),
        ("face", 318),
    ]
    assert parse_qq_face_segments("[QQ表情:不存在]") == [("text", "[QQ表情:不存在]")]
    assert parse_qq_face_segments("[QQ表情: 🥺]") == [("text", "🥺")]
    assert parse_qq_face_segments("唉[QQ表情:🥺]好吧") == [
        ("text", "唉"), ("text", "🥺"), ("text", "好吧")
    ]


def test_quote_and_sticker_actions() -> None:
    history = [
        HistoryMessage(101, "对方", "第一条"),
        HistoryMessage(202, "机器人", "第二条"),
    ]
    prompt, targets = build_quote_options(history)
    assert "m1（对方）：第一条" in prompt
    assert targets == {"m1": 101, "m2": 202}

    with tempfile.TemporaryDirectory() as temp_dir:
        sticker = Path(temp_dir) / "开心.gif"
        sticker.write_bytes(b"GIF89a")
        catalog = StickerCatalog(Path(temp_dir))
        assert catalog.names() == ["开心"]

        actions = parse_reply_actions(
            "[[reply:m1]][[sticker:开心]]就是这条\n普通回复",
            targets,
            catalog,
            max_chars=80,
        )
        assert len(actions) == 2
        assert actions[0].reply_message_id == 101
        assert actions[0].sticker_path == sticker.resolve()
        assert actions[0].text == "就是这条"
        assert actions[1].reply_message_id is None
        assert actions[1].text == "普通回复"

        chinese_control = parse_reply_actions(
            "[引用消息：m2][QQ表情:偷笑]",
            targets,
            catalog,
        )
        assert len(chinese_control) == 1
        assert chinese_control[0].reply_message_id == 202
        assert chinese_control[0].text == "[QQ表情:偷笑]"


def test_bare_quote_labels_are_removed_without_breaking_real_quotes() -> None:
    """模型误抄的裸 [mN] 不应发出；合法引用控制仍应生效。"""
    with tempfile.TemporaryDirectory() as temp_dir:
        catalog = StickerCatalog(Path(temp_dir))
        targets = {"m8": 108, "m9": 109}

        leaked = "[m9] [m9] [m8] *直面恐惧……*"
        assert strip_bare_quote_labels(leaked) == "*直面恐惧……*"
        actions = parse_reply_actions(
            leaked, targets, catalog, max_chars=0, max_bubble_chars=200
        )
        assert len(actions) == 1
        assert actions[0].text == "*直面恐惧……*"
        assert actions[0].reply_message_id is None

        quoted = parse_reply_actions(
            "[回复: m9] [m8] 正式引用",
            targets,
            catalog,
            max_chars=0,
        )
        assert len(quoted) == 1
        assert quoted[0].reply_message_id == 109
        assert quoted[0].text == "正式引用"

        with_face = strip_bare_quote_labels("[m9] [QQ表情: 微笑] [m8] 回应")
        assert with_face == "[QQ表情: 微笑] 回应"
        assert strip_bare_quote_labels("正文提到 [m8] 时保留") == "正文提到 [m8] 时保留"


def test_merge_short_actions() -> None:
    """连续短行应合并成一条气泡；超长则断开；控制动作作为边界。"""
    with tempfile.TemporaryDirectory() as temp_dir:
        catalog = StickerCatalog(Path(temp_dir))

        # 1) 多条短行合并为一条
        reply = "今天天气真好\n我们去散步吧\n顺便买点吃的"
        actions = parse_reply_actions(reply, {}, catalog, max_chars=0, max_bubble_chars=200)
        assert len(actions) == 1
        assert actions[0].text == "今天天气真好\n我们去散步吧\n顺便买点吃的"

        # 2) 累计超过上限则断开成多条（前两句合并，第三句另起）
        reply2 = "一二三四五六七八九十\n短句\n另一句较长文本"
        actions2 = parse_reply_actions(reply2, {}, catalog, max_chars=0, max_bubble_chars=15)
        assert len(actions2) == 2
        assert actions2[0].text == "一二三四五六七八九十\n短句"
        assert actions2[1].text == "另一句较长文本"

        # 3) 带引用的动作作为边界，不参与合并
        reply3 = "普通文本\n[引用消息：m1]这是引用回复"
        targets = {"m1": 101}
        actions3 = parse_reply_actions(reply3, targets, catalog, max_chars=0, max_bubble_chars=200)
        assert len(actions3) == 2
        assert actions3[0].text == "普通文本"
        assert actions3[1].reply_message_id == 101

        # 4) max_bubble_chars<=0 时不合并（每行一条）
        reply4 = "第一行\n第二行"
        actions4 = parse_reply_actions(reply4, {}, catalog, max_chars=0, max_bubble_chars=0)
        assert len(actions4) == 2


def test_send_result_id() -> None:
    assert extract_message_id({"message_id": 123}) == 123
    assert extract_message_id({"message_id": "456"}) == 456
    assert extract_message_id({}) is None
    assert extract_message_id("789") == 789


def _fake_bot() -> object:
    """返回一个记录已发消息的假 Bot（仅用于测试发送层）。"""
    class _Bot:
        def __init__(self) -> None:
            self.sent: list = []
            self._n = 0

        async def send(self, event, message) -> dict:
            self._n += 1
            self.sent.append(message)
            return {"message_id": self._n}

    return _Bot()


def _qq_settings(temp_dir: str) -> QQSettings:
    return QQSettings(
        quote_history_size=12,
        sticker_enabled=False,
        sticker_dir=Path(temp_dir),
        sticker_prompt_limit=30,
        poke_back=False,
        poke_text_reply=False,
        poke_cooldown=15,
    )


def test_streaming_send_preserves_order_and_count() -> None:
    with tempfile.TemporaryDirectory() as td:
        sender = QQSender(
            settings=_qq_settings(td),
            sticker_catalog=StickerCatalog(Path(td)),
            image_service=None,
            stream_pause=0.0,
        )
        actions = [ReplyAction(text=f"第{i}行内容") for i in range(5)]
        bot = _fake_bot()
        ids = asyncio.run(sender.send(bot, object(), actions))
        assert ids == [1, 2, 3, 4, 5]
        assert len(bot.sent) == 5


def test_streaming_send_keeps_failure_positions() -> None:
    with tempfile.TemporaryDirectory() as td:
        sender = QQSender(
            settings=_qq_settings(td),
            sticker_catalog=StickerCatalog(Path(td)),
            image_service=None,
            stream_pause=0.0,
        )

        class _PartialFailureBot:
            def __init__(self) -> None:
                self.calls = 0

            async def send(self, event, message):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("first failed")
                return {"message_id": 9}

        ids = asyncio.run(sender.send(
            _PartialFailureBot(), object(),
            [ReplyAction(text="第一条"), ReplyAction(text="第二条")],
        ))
        assert ids == [None, 9]


def test_streaming_send_applies_pause_between_messages() -> None:
    import time

    with tempfile.TemporaryDirectory() as td:
        # stream_pause=0.2，4 条消息 → 首条不延迟，后续 3 条各插入 >=0.2s 停顿
        sender = QQSender(
            settings=_qq_settings(td),
            sticker_catalog=StickerCatalog(Path(td)),
            image_service=None,
            stream_pause=0.2,
        )
        actions = [ReplyAction(text="x" * 100) for _ in range(4)]
        bot = _fake_bot()
        start = time.monotonic()
        ids = asyncio.run(sender.send(bot, object(), actions))
        elapsed = time.monotonic() - start
        assert ids == [1, 2, 3, 4]
        assert elapsed >= 0.25  # 弱断言：确认确实插入了间隔（留足余量防抖动）


def main() -> None:
    test_message_description()
    test_quote_and_sticker_actions()
    test_send_result_id()
    test_streaming_send_preserves_order_and_count()
    test_streaming_send_keeps_failure_positions()
    test_streaming_send_applies_pause_between_messages()
    print("QQ interaction tests passed")


if __name__ == "__main__":
    main()
