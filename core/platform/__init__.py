"""平台无关消息接入层。"""

from core.platform.dispatcher import MessageDispatcher
from core.platform.models import IncomingMessage
from core.platform.sender import PlatformSender

__all__ = ["IncomingMessage", "MessageDispatcher", "PlatformSender"]
