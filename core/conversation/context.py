import time
from collections import defaultdict
ConversationKey = tuple[str, str, str, str]

conversations: dict[ConversationKey, list[dict]] = defaultdict(list)
last_active: dict[ConversationKey, float] = {}


def _conversation_key(
    self_id: str | int,
    peer_id: str | int,
    mode_key: str = "",
    namespace: str | None = None,
) -> ConversationKey:
    return str(self_id), str(peer_id), str(mode_key or ""), str(namespace or "")


def _get_conversation(
    self_id: str,
    user_id: str,
    context_timeout: int,
    mode_key: str = "",
    namespace: str | None = None,
) -> list[dict]:
    """获取某模式/命名空间的短期上下文，超时自动清空。"""
    key = _conversation_key(self_id, user_id, mode_key, namespace)
    now = time.time()
    if key in last_active and (now - last_active[key]) > context_timeout:
        conversations[key].clear()
    last_active[key] = now
    return conversations[key]


def _clear_conversation(
    self_id: str | int,
    peer_id: str | int,
    mode_key: str = "",
    namespace: str | None = None,
) -> int:
    """清空一个模式/命名空间的短期上下文，返回删除的消息数。"""
    key = _conversation_key(self_id, peer_id, mode_key, namespace)
    removed = len(conversations.get(key, ()))
    conversations.pop(key, None)
    last_active.pop(key, None)
    return removed


def _clear_mode_conversations(
    self_id: str | int,
    peer_id: str | int,
    mode_key: str,
) -> int:
    """清空某用户在一个模式下的全部命名空间短期上下文。"""
    prefix = (str(self_id), str(peer_id), str(mode_key or ""))
    keys = [key for key in conversations if key[:3] == prefix]
    removed = sum(len(conversations.get(key, ())) for key in keys)
    for key in keys:
        conversations.pop(key, None)
        last_active.pop(key, None)
    return removed


def _add_message(
    self_id: str,
    user_id: str,
    role: str,
    content: str,
    max_context: int,
    context_timeout: int,
    msg_id: int | None = None,
    mode_key: str = "",
    namespace: str | None = None,
    *,
    preserve_leading_assistant: bool = False,
    origin: str | None = None,
    sent_id_index: int | None = None,
):
    """添加一条消息到上下文，保持长度 <= MAX_CONTEXT。

    ``msg_id`` 是这条消息在 QQ 端的真实消息 id（来自事件或发送返回值），
    供后续「引用历史消息」功能定位被引用的那条消息。
    角色卡开场白是合法的首条 assistant 消息，可用
    ``preserve_leading_assistant`` 标记，避免被普通的上下文头部清理误删。
    """
    ctx = _get_conversation(
        self_id, user_id, context_timeout, mode_key=mode_key, namespace=namespace
    )
    entry = {"role": role, "content": content, "msg_id": msg_id}
    if preserve_leading_assistant:
        entry["preserve_leading_assistant"] = True
    if origin:
        entry["origin"] = origin
    if sent_id_index is not None:
        entry["sent_id_index"] = int(sent_id_index)
    ctx.append(entry)
    while len(ctx) > max_context:
        ctx.pop(0)
    while (
        ctx
        and ctx[0]["role"] == "assistant"
        and not ctx[0].get("preserve_leading_assistant")
    ):
        ctx.pop(0)
