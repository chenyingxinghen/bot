"""流式输出相关测试：清洗器、LLM 流式、调度器流式编排。"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from core.conversation.service import ConversationService, _StreamCleaner
from core.llm.client import OllamaClient
from core.llm.service import LLMService
from core.platform.dispatcher import MessageDispatcher
from core.platform.models import IncomingMessage


def test_stream_cleaner_strips_inline_control_protocol():
    c = _StreamCleaner()
    # 控制协议不跨边界时被整体滤除，正文正常显示
    out = c.feed("先说点话[生图: 一只猫]然后继续")
    assert out == "先说点话然后继续"
    assert c.finish() == ""


def test_stream_cleaner_handles_split_protocol():
    c = _StreamCleaner()
    first = c.feed("你好[生图: 画")
    # 不完整前缀必须暂存，不能把半截协议闪给用户
    assert first == "你好"
    second = c.feed("一只猫]，然后结束")
    assert second == "，然后结束"
    assert c.finish() == ""


def test_stream_cleaner_strips_quote_and_meta_labels():
    c = _StreamCleaner()
    out = c.feed("[回复: m5]这是引用内容[m3]普通文本")
    assert "[回复:" not in out
    assert "[m3]" not in out
    assert "这是引用内容" in out
    assert "普通文本" in out


def test_strip_think_static():
    assert OllamaClient._strip_think("想一下</think>答案", False) == ("答案", False)
    res = OllamaClient._strip_think("<think>还没想完", True)
    assert res.text == "" and res.in_think is True
    res2 = OllamaClient._strip_think("前文<think>思考</think>后记", False)
    assert res2.text == "前文后记" and res2.in_think is False


class _FakeStreamClient:
    def __init__(self, chunks):
        self._chunks = chunks

    async def stream(self, messages, model, temperature, num_predict):
        for chunk in self._chunks:
            yield chunk


def test_tavern_without_proxy_uses_native_llm_stream():
    service = object.__new__(ConversationService)
    service.sillytavern_client = SimpleNamespace(st_url="", chat=AsyncMock())

    class _StreamService:
        async def stream(self, messages, **kwargs):
            yield "酒馆"
            yield "流式"

    service.llm_service = _StreamService()
    mode = SimpleNamespace(
        key="tavern", model="m", vision=False, temperature=0.7, num_predict=128
    )

    async def _run():
        return [part async for part in service._generate_stream(mode, [])]

    assert asyncio.run(_run()) == ["酒馆", "流式"]
    service.sillytavern_client.chat.assert_not_awaited()


def test_tavern_with_proxy_keeps_single_frame_fallback():
    service = object.__new__(ConversationService)
    service.sillytavern_client = SimpleNamespace(
        st_url="http://127.0.0.1:8000",
        chat=AsyncMock(return_value="代理完整回复"),
    )
    service.llm_service = AsyncMock()
    mode = SimpleNamespace(
        key="tavern", model="m", vision=False, temperature=0.7, num_predict=128
    )

    async def _run():
        return [part async for part in service._generate_stream(mode, [])]

    assert asyncio.run(_run()) == ["代理完整回复"]
    service.sillytavern_client.chat.assert_awaited_once()


def test_llm_service_stream_delegates():
    class _FakeSettings:
        model = "m"
        vision_model = "vm"
        temperature = 0.7
        num_predict = 256

    service = LLMService(_FakeStreamClient(["你", "好", "世界"]), _FakeSettings())

    async def _run():
        return [d async for d in service.stream([{"role": "user", "content": "x"}])]

    assert asyncio.run(_run()) == ["你", "好", "世界"]


class _FakeQueue:
    async def submit(self, factory, *, kind="reply"):
        await factory()
        return True


class _FakeSender:
    def __init__(self):
        self.deltas = []
        self.ends = []
        self.sent = []
        self.notices = []
        self.persisted = []

    async def stream_id(self):
        return 1

    async def send_delta(self, message_id, text):
        self.deltas.append((message_id, text))

    async def persist_stream(self, message, message_id, text):
        self.persisted.append((message_id, text))
        return 99

    async def end_stream(self, message_id, history_id=None):
        self.ends.append((message_id, history_id))

    async def send(self, message, actions):
        self.sent.append(list(actions))
        return list(range(len(actions)))

    async def send_notice(self, message, text):
        self.notices.append(text)


class _FakeConv:
    def __init__(self, events):
        self._events = events
        self.registered = []

    async def generate_reply_stream(self, *args, **kwargs):
        for ev in self._events:
            yield ev

    def register_sent_ids(self, self_id, user_id, ids):
        self.registered.append((self_id, user_id, list(ids)))


def _web_message():
    return IncomingMessage(
        platform="web",
        self_id="web:main",
        user_id="web:owner",
        text="你好",
        images=(),
    )


def test_dispatch_stream_streams_text_and_registers_id():
    sender = _FakeSender()
    conv = _FakeConv([
        {"delta": "你"},
        {"delta": "好"},
        {"actions": []},
    ])
    dispatcher = MessageDispatcher(conv, _FakeQueue())

    async def _run():
        return await dispatcher.dispatch_stream(_web_message(), sender)

    accepted = asyncio.run(_run())
    assert accepted is True
    assert sender.deltas == [(1, "你"), (1, "好")]
    assert sender.ends == [(1, 99)]
    assert sender.persisted == [(1, "你好")]
    assert conv.registered == [("web:main", "web:owner", [1])]


def test_dispatch_stream_handles_image_actions_after_text():
    from core.qq.interactions import ReplyAction

    sender = _FakeSender()
    conv = _FakeConv([
        {"delta": "好的，给你画一张"},
        {"actions": [ReplyAction(image_prompt="一只猫")]},
    ])
    dispatcher = MessageDispatcher(conv, _FakeQueue())

    async def _run():
        await dispatcher.dispatch_stream(_web_message(), sender)

    asyncio.run(_run())
    # 文本已流式显示，生图作为独立动作补发（不重发文本）
    assert sender.deltas == [(1, "好的，给你画一张")]
    assert sender.ends == [(1, 99)]
    # 生图动作经队列提交后由 sender.send 真正发出
    assert sender.sent == [[ReplyAction(image_prompt="一只猫")]]


def test_dispatch_stream_continues_generation_after_client_disconnect():
    class _DisconnectingSender(_FakeSender):
        def __init__(self):
            super().__init__()
            self.calls = 0

        async def send_delta(self, message_id, text):
            self.calls += 1
            raise RuntimeError("websocket closed")

    sender = _DisconnectingSender()
    conv = _FakeConv([
        {"delta": "离线"},
        {"delta": "后仍生成"},
        {"actions": []},
    ])
    dispatcher = MessageDispatcher(conv, _FakeQueue())

    async def _run():
        return await dispatcher.dispatch_stream(_web_message(), sender)

    assert asyncio.run(_run()) is True
    # 发送首次失败后不再继续写 socket，但生成器必须消费到 actions 结束。
    assert sender.calls == 1
    assert sender.ends == []
    assert sender.persisted == [(1, "离线后仍生成")]
    assert conv.registered == [("web:main", "web:owner", [1])]


def test_dispatch_stream_still_runs_image_after_delta_disconnect():
    from core.qq.interactions import ReplyAction

    class _DisconnectingSender(_FakeSender):
        async def send_delta(self, message_id, text):
            raise RuntimeError("websocket closed")

    sender = _DisconnectingSender()
    conv = _FakeConv([
        {"delta": "图片稍后完成"},
        {"actions": [ReplyAction(image_prompt="一只猫")]},
    ])
    dispatcher = MessageDispatcher(conv, _FakeQueue())

    async def _run():
        await dispatcher.dispatch_stream(_web_message(), sender)

    asyncio.run(_run())
    assert sender.persisted == [(1, "图片稍后完成")]
    assert sender.sent == [[ReplyAction(image_prompt="一只猫")]]


def test_dispatch_stream_command_path_sends_whole_actions():
    from core.qq.interactions import ReplyAction

    sender = _FakeSender()
    conv = _FakeConv([{"actions": [ReplyAction(text="已切换到酒馆模式。")]}])
    dispatcher = MessageDispatcher(conv, _FakeQueue())

    async def _run():
        await dispatcher.dispatch_stream(_web_message(), sender)

    asyncio.run(_run())
    # 命令不走流式：没有 delta，整体发送且回填 id
    assert sender.deltas == []
    assert sender.sent == [[ReplyAction(text="已切换到酒馆模式。")]]
    assert conv.registered == [("web:main", "web:owner", [0])]


def test_dispatch_stream_command_send_failure_has_no_second_notice():
    from core.qq.interactions import ReplyAction

    class _FailingSender(_FakeSender):
        async def send(self, message, actions):
            raise RuntimeError("history write failed")

    sender = _FailingSender()
    conv = _FakeConv([{"actions": [ReplyAction(text="命令回复")]}])
    dispatcher = MessageDispatcher(conv, _FakeQueue())

    async def _run():
        await dispatcher.dispatch_stream(_web_message(), sender)

    asyncio.run(_run())
    assert sender.notices == []
    assert conv.registered == []
