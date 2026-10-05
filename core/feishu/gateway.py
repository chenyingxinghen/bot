"""飞书官方 WebSocket 长连接网关。"""

from __future__ import annotations

import asyncio
import logging
import threading

from config.platforms import FeishuSettings
from core.feishu.adapter import incoming_from_event
from core.feishu.sender import FeishuSender
from core.image.service import ImageService
from core.platform.dispatcher import MessageDispatcher
from core.qq.interactions import StickerCatalog

logger = logging.getLogger(__name__)


class FeishuGateway:
    def __init__(
        self,
        settings: FeishuSettings,
        dispatcher: MessageDispatcher,
        image_service: ImageService,
        stickers: StickerCatalog,
        *,
        stream_pause: float = 0.0,
    ) -> None:
        self.settings = settings
        self.dispatcher = dispatcher
        self.image_service = image_service
        self.stickers = stickers
        self.stream_pause = stream_pause
        self._thread: threading.Thread | None = None
        self._ws_client: object | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._client_loop: asyncio.AbstractEventLoop | None = None

    def start(self, loop: asyncio.AbstractEventLoop) -> bool:
        if not self.settings.enabled:
            logger.info("飞书入口：关闭")
            return False
        if not self.settings.app_id or not self.settings.app_secret:
            logger.error("飞书入口已启用但 FEISHU_APP_ID/FEISHU_APP_SECRET 不完整")
            return False
        if self._thread and self._thread.is_alive():
            return True

        try:
            import lark_oapi as lark
        except ImportError:
            logger.error("飞书入口已启用但未安装 lark-oapi")
            return False

        self._loop = loop
        client = lark.Client.builder().app_id(self.settings.app_id).app_secret(
            self.settings.app_secret
        ).build()
        sender = FeishuSender(
            client,
            self.image_service,
            self.stickers,
            stream_pause=self.stream_pause,
        )

        def on_message(data) -> None:
            message = incoming_from_event(data, bot_id=self.settings.bot_id)
            if message is None or self._loop is None:
                return
            future = asyncio.run_coroutine_threadsafe(
                self.dispatcher.dispatch(message, sender), self._loop
            )
            future.add_done_callback(self._log_dispatch_error)

        handler = (
            lark.EventDispatcherHandler.builder("", "")
            .register_p2_im_message_receive_v1(on_message)
            .build()
        )
        self._ws_client = lark.ws.Client(
            self.settings.app_id,
            self.settings.app_secret,
            event_handler=handler,
            log_level=lark.LogLevel.INFO,
        )
        self._thread = threading.Thread(
            target=self._run_client,
            name="feishu-websocket",
            daemon=True,
        )
        self._thread.start()
        logger.info("飞书长连接线程已启动")
        return True

    def stop(self) -> None:
        client = self._ws_client
        self._ws_client = None
        stop = getattr(client, "stop", None)
        if callable(stop):
            try:
                stop()
            except Exception as exc:  # noqa: BLE001
                logger.warning("停止飞书长连接失败：%s", exc)
        client_loop = self._client_loop
        self._client_loop = None
        if client_loop is not None and not client_loop.is_closed():
            # SDK 的 start() 阻塞在自己的事件循环里，只能从外部请求停止
            try:
                client_loop.call_soon_threadsafe(client_loop.stop)
            except Exception as exc:  # noqa: BLE001
                logger.warning("停止飞书事件循环失败：%s", exc)

    def _run_client(self) -> None:
        # lark-oapi 的 ws.Client.start() 使用模块级全局 loop，
        # 该 loop 在 import 时绑定的是主线程（NoneBot）正在运行的事件循环，
        # 直接 run_until_complete 会抛 "This event loop is already running"。
        # 因此在本线程内新建独立事件循环，并覆盖 SDK 模块级 loop。
        try:
            import lark_oapi.ws.client as lark_ws_client
        except ImportError as exc:  # noqa: BLE001
            logger.error("飞书长连接启动失败，缺少 lark-oapi：%s", exc)
            return

        client_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(client_loop)
        lark_ws_client.loop = client_loop
        self._client_loop = client_loop
        try:
            self._ws_client.start()
        except Exception as exc:  # noqa: BLE001
            text = str(exc)
            if "app_id or app_secret is invalid" in text:
                logger.error(
                    "飞书长连接鉴权失败：FEISHU_APP_ID/FEISHU_APP_SECRET 无效，"
                    "请到开放平台「凭证与基础信息」核对后更新 .env.prod（%s）",
                    text,
                )
            else:
                logger.exception("飞书长连接退出：%s", exc)
        finally:
            try:
                if not client_loop.is_closed():
                    client_loop.close()
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    def _log_dispatch_error(future) -> None:
        try:
            future.result()
        except Exception as exc:  # noqa: BLE001
            logger.exception("飞书消息提交失败：%s", exc)
