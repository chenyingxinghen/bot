"""平台无关的入站消息模型。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class IncomingMessage:
    """统一描述来自 QQ、Web、飞书等入口的一条用户消息。

    ``self_id`` 与 ``user_id`` 是传给 ConversationService 的持久化身份键。
    QQ 适配器保留历史纯数字 ID；新增平台使用 ``平台:原始ID`` 防止碰撞。
    ``native_event`` 仅用于兼容真实消息 ID 等平台元数据，领域层不依赖其类型。
    """

    platform: str
    self_id: str
    user_id: str
    text: str = ""
    images: tuple[str, ...] = ()
    message_id: str | None = None
    conversation_id: str | None = None
    native_event: object | None = field(default=None, compare=False, repr=False)
