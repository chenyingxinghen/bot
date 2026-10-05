from __future__ import annotations

import asyncio

from core.platform.dispatcher import MessageDispatcher
from core.platform.models import IncomingMessage
from core.qq.interactions import ReplyAction
from core.queue.queue import InferenceQueue


class FakeConversation:
    def __init__(self, actions):
        self.actions = actions
        self.calls = []
        self.registered = []

    async def generate_reply(self, self_id, user_id, text, images, *, event):
        self.calls.append((self_id, user_id, text, tuple(images), event))
        return self.actions

    def register_sent_ids(self, self_id, user_id, ids):
        self.registered.append((self_id, user_id, list(ids)))


class FakeSender:
    def __init__(self):
        self.calls = []
        self.notices = []

    async def send(self, message, actions):
        actions = list(actions)
        self.calls.append(actions)
        return [100 + len(self.calls)] * len(actions)

    async def send_notice(self, message, text):
        self.notices.append(text)


async def _wait(queue: InferenceQueue, completed: int) -> None:
    for _ in range(300):
        await asyncio.sleep(0.01)
        if queue.stats.completed >= completed or queue.stats.failed:
            return


def test_dispatcher_splits_text_and_image_actions():
    async def run():
        queue = InferenceQueue(max_workers=1, max_pending=10)
        conversation = FakeConversation(
            [ReplyAction(text="先回文字"), ReplyAction(image_prompt="画一只猫")]
        )
        sender = FakeSender()
        dispatcher = MessageDispatcher(conversation, queue)
        message = IncomingMessage(
            platform="web",
            self_id="web:main",
            user_id="web:owner",
            text="你好",
            native_event={"message_id": 7},
        )

        assert await dispatcher.dispatch(message, sender) is True
        await _wait(queue, 2)
        await queue.shutdown()

        assert conversation.calls[0][:3] == ("web:main", "web:owner", "你好")
        assert [bool(call[0].image_prompt) for call in sender.calls] == [False, True]
        assert conversation.registered == [("web:main", "web:owner", [101])]
        assert not sender.notices

    asyncio.run(run())


def test_dispatcher_reports_full_queue():
    async def run():
        queue = InferenceQueue(max_workers=1, max_pending=1, auto_start=False)
        conversation = FakeConversation([ReplyAction(text="ok")])
        sender = FakeSender()
        dispatcher = MessageDispatcher(conversation, queue)
        message = IncomingMessage(platform="web", self_id="web:main", user_id="web:u")

        assert await dispatcher.dispatch(message, sender) is True
        assert await dispatcher.dispatch(message, sender) is False
        assert sender.notices == ["我有点忙，请稍后再试～"]

    asyncio.run(run())
