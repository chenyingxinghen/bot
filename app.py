"""NoneBot 启动入口 + 消息 handler（薄）。

只做三件事：
1. 启动并构建配置 ``AppConfig``；
2. 用 ``build_services`` 装配所有领域服务；
3. 注册一个私有消息 handler，把事件翻译成对 ``ConversationService`` 的调用，
   把返回的 ``ReplyAction`` 交给 ``QQSender`` 发送。所有业务编排都在 core 里。
"""

from pathlib import Path

from logging_setup import (
    configure_process_logging_encoding,
    configure_app_logging,
)

# Windows 服务常把 stdout/stderr 直接重定向到文件；必须在日志框架初始化前
# 固定为 UTF-8，否则会按系统本地代码页写出 GBK/GB18030 日志。
configure_process_logging_encoding()

import logging
import nonebot

logger = logging.getLogger(__name__)
from nonebot.adapters.onebot.v11 import Adapter as OneBotV11Adapter
from nonebot.adapters.onebot.v11 import Bot, PrivateMessageEvent
from nonebot.rule import Rule

from config import build_app_config
from core import build_services
from core.feishu import FeishuGateway
from core.qq.adapter import QQPlatformSender, incoming_from_private_event
from core.qq.interactions import StickerCatalog
from core.web import install_web_routes

_bot_dir = Path(__file__).resolve().parent

# 显式从 bot 目录加载 .env / .env.prod，避免依赖启动目录(CWD)。
# 否则以服务方式运行(AppDirectory 非 bot 目录)时 nonebot 找不到 .env.prod，
# 会回退默认端口，导致"端口错误 / NapCat 连不上"。
from dotenv import load_dotenv

load_dotenv(_bot_dir / ".env")
load_dotenv(_bot_dir / ".env.prod", override=True)

nonebot.init()

# 让 bot 自身（stdlib logging）的应用日志经 NoneBot 的 loguru 桥接到 stdout，
# 否则 core.* 的 INFO 日志（含 LLM 输入输出）会被 stdlib root 默认 WARNING 吞掉。
configure_app_logging()

driver = nonebot.get_driver()
driver.register_adapter(OneBotV11Adapter)

# 1. 启动时构建配置
app_config = build_app_config(driver.config, _bot_dir)

# 2. 装配服务（组合根）
services = build_services(app_config, _bot_dir)
inference_queue = services.inference_queue
inference_queue.start()

# Web/PWA 与 QQ 共用 NoneBot 的 FastAPI 服务和统一消息调度器。
_stickers = StickerCatalog(app_config.qq.sticker_dir)
install_web_routes(
    nonebot.get_app(),
    app_config.platforms.web,
    services.message_dispatcher,
    services.image_service,
    _stickers,
    stream_pause=app_config.conversation.stream_pause,
)
feishu_gateway = FeishuGateway(
    app_config.platforms.feishu,
    services.message_dispatcher,
    services.image_service,
    _stickers,
    stream_pause=app_config.conversation.stream_pause,
)


@driver.on_startup
async def _startup_platforms():
    import asyncio

    feishu_gateway.start(asyncio.get_running_loop())


async def _is_private(event: PrivateMessageEvent) -> bool:
    return True


async def _not_self(event: PrivateMessageEvent) -> bool:
    return str(event.user_id) != str(event.self_id)


# 3. 注册薄 handler
private_chat = nonebot.on_message(
    rule=Rule(_is_private, _not_self),
    priority=10,
    block=True,
)


@private_chat.handle()
async def handle_private_msg(bot: Bot, event: PrivateMessageEvent):
    message = await incoming_from_private_event(event)
    sender = QQPlatformSender(bot, services.qq_sender)
    await services.message_dispatcher.dispatch(message, sender)


@driver.on_shutdown
async def _shutdown():
    feishu_gateway.stop()
    await inference_queue.shutdown()


nonebot.load_from_toml(str(_bot_dir / "pyproject.toml"))

if __name__ == "__main__":
    nonebot.run()
