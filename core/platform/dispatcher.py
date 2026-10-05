"""统一消息调度：平台入口与对话领域之间的编排层。"""

from __future__ import annotations

import logging

from core.conversation.service import ConversationService
from core.platform.models import IncomingMessage
from core.platform.sender import PlatformSender
from core.queue.queue import InferenceQueue

logger = logging.getLogger(__name__)


class MessageDispatcher:
    """处理所有入口共用的队列、生成、发送和消息 ID 回填逻辑。"""

    def __init__(
        self,
        conversation_service: ConversationService,
        inference_queue: InferenceQueue,
    ) -> None:
        self.conversation_service = conversation_service
        self.inference_queue = inference_queue

    async def dispatch(
        self, message: IncomingMessage, sender: PlatformSender
    ) -> bool:
        """提交一条入站消息；成功入队返回 True，队列满返回 False。"""

        async def _run() -> None:
            try:
                actions = await self.conversation_service.generate_reply(
                    message.self_id,
                    message.user_id,
                    message.text,
                    message.images,
                    event=message.native_event,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("生成回复失败 [%s]：%s", message.platform, exc)
                await self._safe_notice(
                    sender, message, "刚才处理消息时出了点问题，请稍后再试。"
                )
                return

            if not actions:
                return

            text_actions = [action for action in actions if not action.image_prompt]
            image_actions = [action for action in actions if action.image_prompt]

            if text_actions:
                try:
                    sent_ids = await sender.send(message, text_actions)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("发送回复失败 [%s]：%s", message.platform, exc)
                    await self._safe_notice(sender, message, "回复发送失败，请稍后再试。")
                else:
                    self.conversation_service.register_sent_ids(
                        message.self_id, message.user_id, sent_ids
                    )

            for image_action in image_actions:
                async def _generate_image(action=image_action) -> None:
                    try:
                        await sender.send(message, [action])
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("异步生图发送失败 [%s]：%s", message.platform, exc)

                accepted = await self.inference_queue.submit(
                    _generate_image, kind=f"image:{message.platform}"
                )
                if accepted:
                    logger.info(
                        "生图任务已入队 [%s]：prompt=%r",
                        message.platform,
                        image_action.image_prompt[:300],
                    )
                if not accepted:
                    await self._safe_notice(
                        sender, message, "生图任务有点多，稍后再帮你画～"
                    )

        accepted = await self.inference_queue.submit(
            _run, kind=f"reply:{message.platform}"
        )
        if not accepted:
            await self._safe_notice(sender, message, "我有点忙，请稍后再试～")
        return accepted

    @staticmethod
    async def _safe_notice(
        sender: PlatformSender, message: IncomingMessage, text: str
    ) -> None:
        try:
            await sender.send_notice(message, text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("发送系统提示失败 [%s]：%s", message.platform, exc)

    async def dispatch_stream(
        self, message: IncomingMessage, sender: "WebSocketSender"
    ) -> bool:
        """流式提交一条入站消息；增量文本通过 ``sender`` 直接推回客户端。

        仅在 Web 入口使用：``sender`` 必须实现 ``stream_id`` / ``send_delta`` /
        ``end_stream``。队列满时返回 False（调用方提示繁忙）。命令与空消息走单帧
        ``actions``，与普通 ``dispatch`` 行为一致。
        """

        async def _run() -> None:
            mid: int | None = None
            streamed = False
            delivery_failed = False
            streamed_text: list[str] = []
            final_actions: list = []
            try:
                async for evt in self.conversation_service.generate_reply_stream(
                    message.self_id,
                    message.user_id,
                    message.text,
                    message.images,
                    event=message.native_event,
                ):
                    if "delta" in evt and evt["delta"]:
                        streamed_text.append(evt["delta"])
                        if mid is None:
                            mid = await sender.stream_id()
                        if not delivery_failed:
                            try:
                                await sender.send_delta(mid, evt["delta"])
                            except Exception as exc:  # noqa: BLE001
                                # 客户端离线不应中断生成器；必须继续消费到 _finalize，
                                # 让完整回复落库，重连后由最近会话窗口补发。
                                delivery_failed = True
                                logger.info(
                                    "Web 流式连接已断开，继续后台生成并落库：%s", exc
                                )
                        streamed = True
                    elif "actions" in evt:
                        final_actions = evt["actions"]
            except Exception as exc:  # noqa: BLE001
                logger.exception("流式生成失败 [%s]：%s", message.platform, exc)
                if mid is not None:
                    try:
                        await sender.end_stream(mid)
                    except Exception:  # noqa: BLE001
                        pass
                await self._safe_notice(
                    sender, message, "刚才处理消息时出了点问题，请稍后再试。"
                )
                return

            if streamed:
                history_id = await sender.persist_stream(
                    message, mid, "".join(streamed_text)
                )
                if not delivery_failed:
                    try:
                        await sender.end_stream(mid, history_id)
                    except Exception as exc:  # noqa: BLE001
                        delivery_failed = True
                        logger.info("Web 流式结束标记发送失败：%s", exc)
                # 文本已实时显示；只补发生图等结构化动作，避免文本重复。
                # Web sender 会先保存生成结果，因此连接离线时仍须继续执行生图任务。
                image_actions = [a for a in final_actions if a.image_prompt]
                for image_action in image_actions:
                    async def _generate_image(action=image_action) -> None:
                        try:
                            await sender.send(message, [action])
                        except Exception as exc:  # noqa: BLE001
                            logger.warning(
                                "异步生图发送失败 [%s]：%s", message.platform, exc
                            )

                    accepted = await self.inference_queue.submit(
                        _generate_image, kind=f"image:{message.platform}"
                    )
                    if not accepted:
                        await self._safe_notice(
                            sender, message, "生图任务有点多，稍后再帮你画～"
                        )
                # 回填流式子弹的 id，供后续引用定位
                if mid is not None:
                    self.conversation_service.register_sent_ids(
                        message.self_id, message.user_id, [mid]
                    )
            elif final_actions:
                # 命令/非流式单帧：整体发送（与普通 dispatch 一致）
                try:
                    sent_ids = await sender.send(message, final_actions)
                except Exception as exc:  # noqa: BLE001
                    # Web sender 内部会吞掉预期断线；抵达这里意味着持久化或生成失败，
                    # 此时不要再向同一条可能已关闭的连接发送二次 notice。
                    logger.error("Web 回复处理失败：%s", exc)
                    return
                self.conversation_service.register_sent_ids(
                    message.self_id, message.user_id, sent_ids
                )

        accepted = await self.inference_queue.submit(
            _run, kind=f"reply:{message.platform}"
        )
        if not accepted:
            await self._safe_notice(sender, message, "我有点忙，请稍后再试～")
        return accepted
