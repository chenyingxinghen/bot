import time
from datetime import datetime
from core.memory.store import MemoryStore

_WEEKDAYS = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]


def _now_str() -> str:
    """当前时间的中文描述，用于注入 system prompt"""
    now = datetime.now()
    return f"{now:%Y-%m-%d} {_WEEKDAYS[now.weekday()]} {now:%H:%M}"


def _gap_str(
    self_id: str,
    user_id: str,
    memory_store: MemoryStore,
    mode_key: str = "",
    namespace: str | None = None,
) -> str:
    """当前模式/命名空间距上次互动的间隔；无历史或很短则返回空串。"""
    state = memory_store.get_interaction_state(
        self_id, user_id, mode_key=mode_key, namespace=namespace
    )
    if not state or not state["last_user_at"]:
        return ""
    gap = time.time() - int(state["last_user_at"])
    if gap < 120:
        return ""
    if gap < 3600:
        return f"[距上次对话 {int(gap // 60)} 分钟后] "
    if gap < 3600 * 24:
        return f"[距上次对话 {int(gap // 3600)} 小时后] "
    return f"[距上次对话 {int(gap // 86400)} 天后] "
