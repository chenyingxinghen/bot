"""对话大脑：把一次用户消息变成一组 ``ReplyAction``。

职责（原 ``plugins/chat_style`` 的「大脑」整体迁入此处）：
1. 解析命令（/mode、/status、/character、/image），命中则短路返回；
2. 取会话上下文、当前模式、角色卡；
3. 固定注入基础记忆胶囊，并按当前话题补充召回长期记忆；
4. 组装 system prompt，调用 LLM 生成；
5. 过滤历史回显，解析成 ``ReplyAction``（引用 / 表情 / 生图）；
6. 落库用户消息与 bot 回复，触发在线记忆抽取。

全程不依赖 nonebot：``event`` 只是可选透传，真正发送在 ``core/qq/sender.py``。
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass
from typing import Iterable, Mapping

logger = logging.getLogger(__name__)


def _slug(value: str) -> str:
    """把作品/角色名规整成安全的命名空间片段（小写、非字母数字→下划线）。"""
    slug = "".join(ch if ch.isalnum() else "_" for ch in (value or "").strip().casefold())
    return slug.strip("_") or "default"

from core.commands.parser import parse_command, resolve_mode_arg
from core.commands.registry import (
    command_owner,
    resolve_command_arg,
    root_prompt_text,
    help_root_text,
    help_mode_text,
    help_command_text,
)
from core.conversation.context import (
    _add_message,
    _clear_conversation,
    _clear_mode_conversations,
    _get_conversation,
)
from core.conversation.echo import filter_history_echo, should_retry
from core.conversation.time_aware import _gap_str, _now_str
from core.conversation.work_selection import WorkSelection
from core.conversation.prompts import (
    assemble_system_prompt,
    build_persona_section,
    build_qq_control_prompt,
    build_time_prompt,
)
from core.qq.interactions import (
    HistoryMessage,
    ReplyAction,
    StickerCatalog,
    build_quote_options,
    extract_image_prompts,
    parse_reply_actions,
    strip_bare_quote_labels,
)

from config.conversation import ConversationSettings
from config.qq import QQSettings
from core.character.client import (
    CharacterCard,
    SillyTavernClient,
    example_dialogue_to_messages,
    render_first_message,
)
from core.image.service import ImageService
from core.llm.service import LLMService
from core.memory.embedding import OllamaEmbeddingClient
from core.memory.online import OnlineMemoryExtractor
from core.memory.store import MemoryStore, format_memory_time
from core.modes.manager import Mode, ModeManager


def _event_message_id(event: object) -> int | None:
    """从 OneBot 事件里取真实消息 id（兼容属性访问与字典）。"""
    if event is None:
        return None
    value = getattr(event, "message_id", None)
    if value is None and isinstance(event, dict):
        value = event.get("message_id")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass
class _Prepared:
    """一次回复生成所需的全部上下文，由 ``_prepare`` 装配、``_finalize`` 消费。"""

    self_id: str
    user_id: str
    text: str
    image_list: list[str]
    mode: "Mode"
    namespace: str | None
    ctx: list[dict]
    user_msg_id: int | None
    messages: list[dict]
    quote_targets: dict
    card: "CharacterCard | None" = None


# 流式显示需要即时滤掉的控制协议（避免 [生图:] / [回复: mN] / [QQ表情:] 闪过）。
# 形态与 core/qq/interactions.py 的解析保持一致（单/双层括号、中英文控制词）。
_STREAM_CTRL_PREFIX = re.compile(
    r"\[{1,2}\s*"
    r"(?:reply|sticker|image|draw|生图|画图|图片|引用消息|引用|回复|表情|QQ表情)"
    r"\s*[:：]"
)
_STREAM_CTRL_FULL = re.compile(
    r"\[{1,2}\s*"
    r"(?:reply|sticker|image|draw|生图|画图|图片|引用消息|引用|回复|表情|QQ表情)"
    r"\s*[:：][^\]\n]*\]{1,2}"
)
_STREAM_META = re.compile(r"\[m\d+\]")


class _StreamCleaner:
    """把流式原始增量清洗成可直接显示的文本。

    控制协议可能跨 token 边界被切开，因此保留不完整的前缀到下次再处理；
    结束时 ``finish`` 丢弃任何仍未闭合的控制前缀。
    """

    def __init__(self) -> None:
        self._buf = ""

    def feed(self, chunk: str) -> str:
        self._buf += chunk
        text = _STREAM_CTRL_FULL.sub("", self._buf)
        text = _STREAM_META.sub("", text)
        matches = list(_STREAM_CTRL_PREFIX.finditer(text))
        if matches:
            last = matches[-1]
            tail = text[last.start():]
            if "]" not in tail:
                self._buf = tail
                return text[: last.start()]
        self._buf = ""
        return text

    def finish(self) -> str:
        text = _STREAM_CTRL_FULL.sub("", self._buf)
        text = _STREAM_META.sub("", text)
        text = _STREAM_CTRL_PREFIX.sub("", text)
        self._buf = ""
        return text


class ConversationService:
    def __init__(
        self,
        *,
        memory_stores: Mapping[str, MemoryStore],
        llm_service: LLMService,
        mode_manager: ModeManager,
        conversation_settings: ConversationSettings,
        qq_settings: QQSettings,
        sillytavern_client: SillyTavernClient | None = None,
        character_selection: object | None = None,
        work_selection: object | None = None,
        user_name: str = "你",
        embedding_client: OllamaEmbeddingClient | None = None,
        online_extractor: OnlineMemoryExtractor | None = None,
        image_service: ImageService | None = None,
        inference_queue: object | None = None,
    ) -> None:
        self.memory_stores = memory_stores
        self.llm_service = llm_service
        self.mode_manager = mode_manager
        self.cs = conversation_settings
        self.qq_settings = qq_settings
        self.sillytavern_client = sillytavern_client
        self.character_selection = character_selection
        self.work_selection = work_selection
        self.user_name = user_name
        self.embedding_client = embedding_client
        self.online_extractor = online_extractor
        self.image_service = image_service
        self.inference_queue = inference_queue
        self._sticker_catalog = StickerCatalog(qq_settings.sticker_dir)
        self._extracting_scopes: set[tuple[str, str, str, str]] = set()
        # 每个进程、每个作用域只尝试一次持久化短期上下文恢复。上下文在本进程内
        # 超时后不应再次从数据库复活；新进程会重新建立此集合并按需恢复。
        self._restored_context_scopes: set[tuple[str, str, str, str]] = set()

    def _restore_context_if_needed(
        self,
        self_id: str,
        user_id: str,
        mode_key: str,
        namespace: str | None,
    ) -> list[dict]:
        """进程首次访问作用域时，从 runtime 消息惰性恢复未超时的短期上下文。"""
        scope_key = (self_id, user_id, mode_key, namespace or "")
        ctx = _get_conversation(
            self_id,
            user_id,
            self.cs.context_timeout,
            mode_key=mode_key,
            namespace=namespace,
        )
        if scope_key in self._restored_context_scopes or ctx:
            self._restored_context_scopes.add(scope_key)
            return ctx

        self._restored_context_scopes.add(scope_key)
        since = int(time.time()) - max(0, int(self.cs.context_timeout))
        try:
            rows = self.memory_stores[mode_key].load_recent_context(
                self_id,
                user_id,
                mode_key=mode_key,
                namespace=namespace,
                limit=self.cs.max_context,
                since=since,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "短期上下文恢复失败：mode=%s namespace=%s error=%s",
                mode_key,
                namespace or "",
                exc,
            )
            return ctx

        for row in rows:
            content = strip_bare_quote_labels(str(row["text"] or ""))
            if not content:
                continue
            send_type = int(row["send_type"] or 0)
            role = "assistant" if send_type == 1 else "user"
            msg_id: int | None = None
            raw_msg_id = str(row["msg_id"] or "")
            if raw_msg_id.isdigit():
                msg_id = int(raw_msg_id)
            entry = {"role": role, "content": content, "msg_id": msg_id}
            # 只有明确持久化的角色开场白允许作为首条 assistant；普通窗口若从
            # assistant 截断开始，仍按既有规则丢弃孤立回复。
            if role == "assistant" and raw_msg_id.startswith("rt-greeting-"):
                entry["preserve_leading_assistant"] = True
            ctx.append(entry)

        while len(ctx) > self.cs.max_context:
            ctx.pop(0)
        while (
            ctx
            and ctx[0]["role"] == "assistant"
            and not ctx[0].get("preserve_leading_assistant")
        ):
            ctx.pop(0)
        if ctx:
            logger.info(
                "已恢复短期上下文：mode=%s namespace=%s messages=%d",
                mode_key,
                namespace or "",
                len(ctx),
            )
        return ctx

    # ── 记忆命名空间（模式内第三维隔离：角色 / 作品）──

    def _current_namespace(self, mode: Mode, self_id: str, user_id: str) -> str | None:
        """根据当前模式与生失效的「实体」算出记忆命名空间。

        - tavern：当前酒馆角色（character_selection）→ ``char:{slug}``
        - writer：当前作品（work_selection）→ ``work:{slug}``
        - clone：无第三维 → None
        返回 None 时记忆键退化为旧格式（按 QQ 号分），向后兼容。
        """
        if mode.key == "tavern" and self.character_selection is not None:
            char = self.character_selection.get((self_id, user_id))
            return f"char:{_slug(char)}" if char else None
        if mode.key == "writer" and self.work_selection is not None:
            work = self.work_selection.get((self_id, user_id))
            return f"work:{_slug(work)}" if work else None
        return None

    def _namespace_label(self, mode: Mode, self_id: str, user_id: str) -> str:
        """给记忆清理回执用的可读标注（当前角色 / 当前作品）。"""
        if mode.key == "tavern" and self.character_selection is not None:
            name = self.character_selection.get((self_id, user_id))
            return f"（角色《{name}》）" if name else ""
        if mode.key == "writer" and self.work_selection is not None:
            name = self.work_selection.get((self_id, user_id))
            return f"（作品《{name}》）" if name else ""
        return ""

    _KIND_LABELS = {
        "self": "角色/自我设定",
        "person": "关于你",
        "relationship": "互动关系",
        "episode": "事件/剧情",
    }

    def _memory_review_text(self, mode: Mode, rows: list[dict], layer: str,
                            self_id: str | None = None, user_id: str | None = None) -> str:
        """把记忆列表渲染成可读回顾（只读；分层）。"""
        if not rows:
            if layer == "mode":
                scope_hint = "（模式层）"
            else:
                nl = self._namespace_label(mode, self_id, user_id) if self_id else ""
                scope_hint = nl or "（当前会话 / 未选定角色或作品）"
            return (f"「{mode.label}」模式{scope_hint}目前还没有沉淀任何记忆。\n"
                    f"多聊几句，我会自动记住；用 /记忆 模式 看本模式全部，/清记忆 可清理。")
        grouped: dict[str, list[str]] = {}
        for r in rows:
            kind = r.get("kind", "episode")
            when = format_memory_time(kind, r.get("valid_from"), r.get("valid_to"))
            conf = float(r.get("confidence") or 0.0)
            line = f"· {r.get('summary', '')}（{when}；置信 {conf:.0%}）"
            grouped.setdefault(self._KIND_LABELS.get(kind, kind), []).append(line)
        if layer == "mode":
            scope_hint = "【模式层】"
        else:
            nl = self._namespace_label(mode, self_id, user_id) if self_id else ""
            scope_hint = f"【命名空间层{nl}】" if nl else "【命名空间层·当前会话】"
        lines = [f"「{mode.label}」模式记忆回顾 {scope_hint}（共 {len(rows)} 条）", ""]
        for label in ("角色/自我设定", "关于你", "互动关系", "事件/剧情"):
            items = grouped.get(label)
            if not items:
                continue
            lines.append(f"▸ {label}")
            lines.extend(items)
            lines.append("")
        lines.append("用 /记忆 模式 看本模式全部；/清记忆 可清理当前作用域数据。")
        return "\n".join(lines)

    # ── 命令处理 ──

    async def _handle_command(
        self, self_id: str, user_id: str, cmd: tuple, raw_text: str
    ) -> list[ReplyAction] | None:
        kind = cmd[0]
        if kind == "mode":
            mode_key = cmd[1] if len(cmd) > 1 else None
            if not mode_key:
                return [ReplyAction(text="无法识别该模式，可用：酒馆 / 模仿 / 作家。")]
            mode = self.mode_manager.set((self_id, user_id), mode_key)
            if mode is None:
                return [ReplyAction(text=f"未知模式：{mode_key}")]
            return [ReplyAction(text=f"已切换到「{mode.label}」模式。")]
        if kind == "mode_unknown":
            return [ReplyAction(text=f"未知模式：{cmd[1]}")]
        if kind == "status":
            return [ReplyAction(text=self._status_text(self_id, user_id))]
        if kind == "menu":
            return [ReplyAction(text=self._menu_text())]
        if kind == "clear_memory":
            mode = self.mode_manager.get((self_id, user_id))
            scope = cmd[1] if len(cmd) > 1 else None
            store = self.memory_stores[mode.key]
            if scope in ("all", "mode"):
                # 模式层：只清当前 bot 与当前用户在本模式的全部命名空间
                deleted = store.clear_user(self_id, user_id, mode.key)
                cleared_context = _clear_mode_conversations(self_id, user_id, mode.key)
                return [ReplyAction(
                    text=f"已清空你在「{mode.label}」模式的全部记忆（{deleted} 条长期记忆、"
                         f"{cleared_context} 条短期上下文；模式层清理，状态已重置，其他 QQ 用户不受影响，"
                         f"相应数据库原始对话已删除）。用 /清记忆 可只清当前角色/作品。")]
            # 默认：命名空间层（当前角色/作品）；未选定则按当前会话清理
            namespace = self._current_namespace(mode, self_id, user_id)
            deleted = store.clear_session(
                self_id, user_id, namespace, mode_key=mode.key
            )
            cleared_context = _clear_conversation(
                self_id, user_id, mode_key=mode.key, namespace=namespace
            )
            label = self._namespace_label(mode, self_id, user_id)
            if namespace:
                return [ReplyAction(
                    text=f"已清除「{mode.label}」模式记忆{label}（{deleted} 条长期记忆、"
                         f"{cleared_context} 条短期上下文；当前命名空间状态已重置）；"
                         f"关于你的全局记忆与其他角色/作品保留；当前作用域数据库原始对话已删除。")]
            return [ReplyAction(
                text=f"已清除「{mode.label}」模式与本会话的记忆（{deleted} 条长期记忆、"
                     f"{cleared_context} 条短期上下文；会话状态已重置；未选定角色/作品，"
                     f"按当前会话清理，数据库原始对话已删除）。用 /角色 或 /作品 可让记忆按实体隔离后再清。")]
        if kind == "memory":
            mode = self.mode_manager.get((self_id, user_id))
            scope = cmd[1] if len(cmd) > 1 else None
            store = self.memory_stores[mode.key]
            if scope in ("all", "mode"):
                rows = store.list_user(self_id, user_id)
                return [ReplyAction(text=self._memory_review_text(mode, rows, layer="mode"))]
            namespace = self._current_namespace(mode, self_id, user_id)
            rows = store.list_memories(self_id, user_id, namespace)
            return [ReplyAction(text=self._memory_review_text(
                mode, rows, layer="namespace", self_id=self_id, user_id=user_id))]
        if kind == "work":
            name = cmd[1] if len(cmd) > 1 else ""
            mode = self.mode_manager.get((self_id, user_id))
            if not name:
                cur = (self.work_selection.get((self_id, user_id))
                       if self.work_selection is not None else "")
                tip = f"当前作品：《{cur}》。" if cur else "尚未选定作品（记忆不分区）。"
                return [ReplyAction(text=f"{tip}用 /作品 <标题> 开始或切换一部作品，记忆将按作品隔离。")]
            if self.work_selection is not None:
                self.work_selection.set((self_id, user_id), name)
            confirm = (f"已切换到作品：《{name}》（writer 模式记忆将按此作品隔离；"
                       f"切回旧作请用 /作品 <旧标题>）。")
            hint = self._cross_mode_hint("work", mode)
            return [ReplyAction(text=(hint + "\n\n" + confirm) if hint else confirm)]
        if kind == "character":
            name = cmd[1] if len(cmd) > 1 else ""
            mode = self.mode_manager.get((self_id, user_id))
            if not name:
                names = (
                    self.sillytavern_client.list_character_names()
                    if self.sillytavern_client
                    else []
                )
                listing = "、".join(names) if names else "（未找到任何角色卡）"
                return [
                    ReplyAction(text=f"当前可用角色：{listing}。用 /角色 <名称> 切换。")
                ]
            if self.sillytavern_client is None:
                return [ReplyAction(text="角色卡服务未配置，无法切换角色。")]
            try:
                card = self.sillytavern_client.resolve_character(name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("切换角色时解析角色卡失败：%s", exc)
                card = None
            if card is None:
                return [ReplyAction(
                    text=f"未找到角色卡：{name}。发送 /角色 查看可用角色。"
                )]

            canonical_name = card.name.strip() or name
            if self.character_selection is not None:
                self.character_selection.set((self_id, user_id), canonical_name)
            confirm = f"已切换到角色：{canonical_name}（酒馆模式生效）。"
            hint = self._cross_mode_hint("character", mode)
            actions = [ReplyAction(text=(hint + "\n" + confirm) if hint else confirm)]

            namespace = f"char:{_slug(canonical_name)}"
            greeting = render_first_message(card, self.user_name)
            if greeting and self._is_first_character_visit(self_id, user_id, namespace):
                self._remember_character_greeting(
                    self_id, user_id, namespace, greeting, sent_id_index=len(actions)
                )
                actions.append(ReplyAction(text=greeting))
            return actions
        if kind == "help":
            arg = cmd[1] if len(cmd) > 1 else ""
            return [ReplyAction(text=self._help_text(self_id, user_id, arg))]
        if kind == "image":
            prompt = cmd[1] if len(cmd) > 1 else ""
            if not prompt:
                return [ReplyAction(text="请在 /image 后给出生图描述。")]
            return [ReplyAction(image_prompt=prompt)]
        return None

    def _is_first_character_visit(
        self, self_id: str, user_id: str, namespace: str
    ) -> bool:
        """仅当当前角色的短期、长期与持久交互状态均为空时视为首次见面。"""
        ctx = _get_conversation(
            self_id,
            user_id,
            self.cs.context_timeout,
            mode_key="tavern",
            namespace=namespace,
        )
        if ctx:
            return False
        store = self.memory_stores["tavern"]
        if store.has_namespace_memories(self_id, user_id, namespace):
            return False
        return store.get_interaction_state(
            self_id, user_id, mode_key="tavern", namespace=namespace
        ) is None

    def _remember_character_greeting(
        self,
        self_id: str,
        user_id: str,
        namespace: str,
        greeting: str,
        *,
        sent_id_index: int,
    ) -> None:
        """先将首次开场白放入短期上下文；发送成功后再持久化。"""
        _add_message(
            self_id,
            user_id,
            "assistant",
            greeting,
            self.cs.max_context,
            self.cs.context_timeout,
            mode_key="tavern",
            namespace=namespace,
            preserve_leading_assistant=True,
            origin="character_greeting",
            sent_id_index=sent_id_index,
        )

    def _persist_character_greeting(
        self,
        self_id: str,
        user_id: str,
        namespace: str | None,
        greeting: str,
        message_id: int,
    ) -> None:
        """开场白确认发送成功后落库，失败发送不会消耗“一次”机会。"""
        now = int(time.time())
        store = self.memory_stores["tavern"]
        try:
            store.save_runtime_message(
                f"rt-greeting-{message_id}",
                self_id,
                self_id,
                user_id,
                now,
                greeting,
                1,
                mode_key="tavern",
                namespace=namespace,
            )
            store.touch_interaction(
                self_id,
                user_id,
                bot_at=now,
                mode_key="tavern",
                namespace=namespace,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("角色开场白落库失败：%s", exc)

    def _status_text(self, self_id: str, user_id: str) -> str:
        mode = self.mode_manager.get((self_id, user_id))
        lines = [f"当前模式：{mode.label}"]
        if self.inference_queue is not None:
            st = self.inference_queue.stats
            lines.append(
                f"推理队列：在途 {st.active} / 待处理 {st.pending} / "
                f"累计提交 {st.submitted} / 拒绝 {st.rejected}"
            )
        return "\n".join(lines)

    def _menu_text(self) -> str:
        """根处的启动提示（/菜单 显示，也可在 bot 启动时发送）。"""
        return root_prompt_text(self.mode_manager.available())

    def _cross_mode_hint(self, kind: str, mode: "Mode") -> str:
        """模式专属命令在「非所属模式」下使用时的提示。

        命令仍会执行（work/character 是按 peer 存的偏好，跨模式设置无害、
        且可预先设定），但前置一句提示说明它属于哪个模式、切过去后才会生效。
        返回空串表示无需提示（命令属于当前模式或本身是全局命令）。
        """
        owner = command_owner(kind)
        if not owner or owner == mode.key:
            return ""
        owner_mode = self.mode_manager._modes.get(owner)
        owner_label = owner_mode.label if owner_mode else owner
        label = "作品" if kind == "work" else "角色"
        effect = ("writer 模式记忆将按此作品隔离" if kind == "work"
                  else "酒馆模式对话将使用此角色")
        return (f"ℹ️ /{label} 是【{owner_label}】模式的命令，你当前在【{mode.label}】模式。"
                f"已为你记录，切换到 {owner_label}（/{owner}）后{effect}。")

    def _help_text(self, self_id: str, user_id: str, arg: str) -> str:
        """分层帮助：根 / 模式 / 命令。"""
        modes = self.mode_manager.available()
        if not arg:
            return help_root_text(modes)
        mk = resolve_mode_arg(arg)
        if mk:
            return help_mode_text(modes, mk) or help_root_text(modes)
        ck = resolve_command_arg(arg)
        if ck:
            return help_command_text(modes, ck) or help_root_text(modes)
        return (f"未找到关于「{arg}」的帮助。发送 /帮助 查看全局指令，"
                f"/帮助 <模式> 看某模式命令，/help <命令> 看命令用法。")

    # ── 记忆召回策略 ──

    _LOW_INFORMATION_TEXTS = {
        "嗯", "哦", "好", "好的", "行", "可以", "知道了", "收到", "哈哈", "哈哈哈",
        "谢谢", "谢了", "晚安", "早安", "在吗", "？", "?", "。", "...", "……",
    }
    _CONTEXT_DEPENDENT_PREFIXES = (
        "继续", "接着", "然后呢", "后来呢", "那个", "这个", "他呢", "她呢", "它呢",
        "为什么", "怎么了", "真的吗", "还有呢", "再说说", "上次", "刚才",
    )

    @classmethod
    def _memory_query(cls, text: str, ctx: list[dict]) -> str:
        """为记忆召回构造紧凑查询；低信息闲聊不触发随机向量命中。"""
        current = " ".join((text or "").split()).strip()
        compact = current.casefold().strip("，。！？!?~～ ")
        if not compact or compact in cls._LOW_INFORMATION_TEXTS:
            return ""
        needs_context = len(compact) <= 8 or compact.startswith(cls._CONTEXT_DEPENDENT_PREFIXES)
        if not needs_context:
            return current[:500]
        previous = [
            " ".join(str(item.get("content") or "").split()).strip()
            for item in ctx[:-1]
            if item.get("role") in {"user", "assistant"} and item.get("content")
        ][-2:]
        return "\n".join([*previous, current])[-800:]

    @staticmethod
    def _memory_policy(mode: Mode) -> dict:
        """按业务模式控制基础胶囊、话题召回和最终注入预算。"""
        if mode.key == "writer":
            return {
                "capsule_chars": 700,
                "capsule_kind_limits": {"person": 2, "self": 4, "relationship": 4},
                "related_chars": 1100,
                "total_chars": 1800,
                "allowed_kinds": {"self", "person", "relationship", "episode"},
                "limit": 5,
                "stable_limit": 0,
                "vector_min_similarity": .52,
            }
        if mode.key == "tavern":
            return {
                "capsule_chars": 600,
                "capsule_kind_limits": {"person": 3, "self": 3, "relationship": 3},
                "related_chars": 800,
                "total_chars": 1400,
                "allowed_kinds": {"self", "person", "relationship", "episode"},
                "limit": 3,
                "stable_limit": 0,
                "vector_min_similarity": .54,
            }
        return {
            "capsule_chars": 500,
            "capsule_kind_limits": {"person": 5, "self": 2, "relationship": 3},
            "related_chars": 600,
            "total_chars": 1100,
            "allowed_kinds": {"self", "person", "relationship", "episode"},
            "limit": 2,
            "stable_limit": 0,
            "vector_min_similarity": .58,
        }

    @staticmethod
    def _render_memory_capsule(rows: list[dict], max_chars: int) -> str:
        """把稳定基础记忆渲染成紧凑胶囊，不携带面向用户的置信度等展示信息。"""
        if not rows or max_chars <= 0:
            return ""
        labels = {"person": "关于对方", "self": "当前自我", "relationship": "双方关系"}
        grouped: dict[str, list[str]] = {}
        for row in rows:
            kind = str(row.get("kind") or "")
            summary = " ".join(str(row.get("summary") or "").split()).strip()
            if kind in labels and summary:
                grouped.setdefault(kind, []).append(summary)
        lines: list[str] = []
        length = 0
        for kind in ("person", "self", "relationship"):
            items = grouped.get(kind, [])
            if not items:
                continue
            heading = f"【{labels[kind]}】"
            added_heading = False
            for summary in items:
                line = f"- {summary}"
                additions = [heading, line] if not added_heading else [line]
                needed = sum(len(item) + 1 for item in additions)
                if length + needed > max_chars:
                    return "\n".join(lines)
                lines.extend(additions)
                length += needed
                added_heading = True
        return "\n".join(lines)

    @staticmethod
    def _memory_summary_key(line: str) -> str:
        """从两种记忆渲染格式中提取可用于去重的摘要正文。"""
        text = line.strip()
        if not text.startswith("-"):
            return ""
        text = text[1:].strip()
        if text.startswith("[") and "]" in text:
            text = text.split("]", 1)[1].strip()
        text = re.sub(r"（可信度\s+[0-9.]+）$", "", text).strip()
        return re.sub(r"\s+", "", text).casefold()

    @classmethod
    def _merge_memory_context(
        cls, capsule_text: str, related_text: str, max_chars: int
    ) -> str:
        """基础胶囊优先，与话题记忆合并、逐条去重并执行统一字符预算。"""
        if max_chars <= 0:
            return ""
        lines: list[str] = []
        used: set[str] = set()
        length = 0
        for block in (capsule_text, related_text):
            if not block:
                continue
            pending_heading = ""
            for raw_line in block.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith("【") and line.endswith("】"):
                    pending_heading = line
                    continue
                key = cls._memory_summary_key(line)
                if key and key in used:
                    continue
                additions = [pending_heading, line] if pending_heading else [line]
                pending_heading = ""
                needed = sum(len(addition) + 1 for addition in additions)
                if length + needed > max_chars:
                    if not lines:
                        return line[:max_chars]
                    return "\n".join(lines)
                lines.extend(additions)
                length += needed
                if key:
                    used.add(key)
        return "\n".join(lines)

    # ── 生图意图与确定性兜底 ──

    _IMAGE_REQUEST_WORDS = (
        "生成图片", "生成一张", "生成张", "帮我生成", "生图", "画图", "画一张", "画张",
        "帮我画", "做一张图", "来一张图", "发张图", "发图片", "拍张照片", "拍一张照片",
        "发照片", "重发图片", "再发一次图片", "没收到图片",
    )

    @classmethod
    def _has_explicit_image_intent(cls, text: str) -> bool:
        compact = "".join((text or "").casefold().split())
        return any(word in compact for word in cls._IMAGE_REQUEST_WORDS)

    @staticmethod
    def _fallback_image_prompt(text: str, ctx: list[dict], card: CharacterCard | None) -> str:
        """LLM 漏标记时，把明确生图请求与最近场景压成可执行的图片描述。"""
        current = " ".join((text or "").split()).strip()
        recent = [
            " ".join(str(item.get("content") or "").split()).strip()
            for item in ctx[:-1]
            if item.get("content")
        ][-3:]
        scene = "\n".join(recent)[-1200:]
        prompt = (
            f"根据最近对话生成当前场景图片。"
            f"用户最新要求：{current or '生成图片'}。"
            f"最近场景：{scene or '沿用当前角色、服装、环境与动作设定。'}"
        )
        return prompt[:1800]

    # ── 主流程 ──

    async def generate_reply(
        self,
        self_id: str | int,
        user_id: str | int,
        text: str,
        images: Iterable[str] | None = None,
        *,
        event: object | None = None,
    ) -> list[ReplyAction]:
        self_id = str(self_id)
        user_id = str(user_id)
        text = (text or "").strip()
        image_list = list(images or [])

        # 1. 命令短路
        cmd = parse_command(text)
        if cmd is not None:
            handled = await self._handle_command(self_id, user_id, cmd, text)
            if handled is not None:
                return handled

        if not text and not image_list:
            return []

        prepared = await self._prepare(self_id, user_id, text, image_list, event)
        try:
            raw = await self._generate(prepared.mode, prepared.messages)
        except Exception:
            self._rollback_user_message(prepared)
            raise

        # 8. 历史回显过滤；大量回显时带精简上下文重试一次
        raw, matched = filter_history_echo(raw, prepared.ctx)
        if should_retry(matched, self.cs.llm_echo_retry_context):
            retry_messages = [prepared.messages[0]] + prepared.messages[-self.cs.llm_echo_retry_context:]
            retry_messages.append({
                "role": "system",
                "content": "只回答用户最新一条消息，不要复述任何历史对话。",
            })
            retry_raw = await self._generate(prepared.mode, retry_messages)
            filtered_retry, retry_matched = filter_history_echo(retry_raw, prepared.ctx)
            if filtered_retry.strip() and retry_matched < matched:
                raw = filtered_retry
        return await self._finalize(prepared, raw)

    async def _prepare(
        self,
        self_id: str,
        user_id: str,
        text: str,
        image_list: list[str],
        event: object | None,
    ) -> _Prepared:
        """装配一次回复所需的全部上下文（命令短路与空消息已在调用方处理）。"""
        # 2. 模式与上下文
        mode = self.mode_manager.get((self_id, user_id))
        namespace = self._current_namespace(mode, self_id, user_id)
        ctx = self._restore_context_if_needed(
            self_id, user_id, mode.key, namespace
        )

        # 3. 用户消息入上下文（图片不进上下文，仅用于本次 LLM 调用）
        #    msg_id 取事件里的真实 QQ 消息 id，供后续「引用历史消息」定位
        user_msg_id = _event_message_id(event)
        _add_message(
            self_id,
            user_id,
            "user",
            text,
            self.cs.max_context,
            self.cs.context_timeout,
            msg_id=user_msg_id,
            mode_key=mode.key,
            namespace=namespace,
        )

        # 4. 角色卡（酒馆模式）
        card: CharacterCard | None = None
        if mode.key == "tavern" and self.sillytavern_client is not None:
            name = (
                self.character_selection.get((self_id, user_id))
                if self.character_selection is not None
                else None
            )
            try:
                card = self.sillytavern_client.resolve_character(name)
            except Exception as exc:
                logger.warning("解析角色卡失败：%s", exc)
                card = None

        # 5. 长期记忆分两层：稳定基础胶囊每轮常驻，当前话题再做按需召回。
        #    embedding 只增强话题召回；失败时仍保留基础胶囊与词面/时间回退。
        #    tavern/writer 还会召回“尚未达到在线抽取阈值”的持久化对话尾巴，避免
        #    重启后短期上下文超时、长期抽取又未触发时出现记忆断层。
        memory_text = ""
        if mode.use_memory:
            policy = self._memory_policy(mode)
            store = self.memory_stores[mode.key]
            capsule_text = ""
            related_text = ""
            try:
                capsule_rows = store.baseline_memories(
                    self_id,
                    user_id,
                    namespace=namespace,
                    min_confidence=.85,
                    kind_limits=policy["capsule_kind_limits"],
                )
                capsule_text = self._render_memory_capsule(
                    capsule_rows, policy["capsule_chars"]
                )
            except Exception as exc:
                logger.warning("基础记忆胶囊读取失败：%s", exc)

            recall_query = self._memory_query(text, ctx) if text else ""
            if recall_query:
                query_vec = None
                embedding_model = None
                if self.embedding_client is not None:
                    try:
                        query_vec = (await self.embedding_client.embed_async([recall_query]))[0]
                        embedding_model = self.embedding_client.model
                    except Exception as exc:
                        logger.warning("记忆向量查询失败，回退词面召回：%s", exc)
                try:
                    related_text = store.search_context(
                        recall_query,
                        self_id,
                        user_id,
                        query_embedding=query_vec,
                        embedding_model=embedding_model,
                        namespace=namespace,
                        allowed_kinds=policy["allowed_kinds"],
                        limit=policy["limit"],
                        stable_limit=policy["stable_limit"],
                        max_chars=policy["related_chars"],
                        vector_min_similarity=policy["vector_min_similarity"],
                    )
                    if mode.key in {"tavern", "writer"}:
                        pending_text = store.search_pending_runtime_context(
                            recall_query,
                            self_id,
                            user_id,
                            mode_key=mode.key,
                            namespace=namespace,
                        )
                        if pending_text:
                            related_text = "\n\n".join(
                                part for part in (related_text, pending_text) if part
                            )
                except Exception as exc:
                    logger.warning("记忆检索失败：%s", exc)
            memory_text = self._merge_memory_context(
                capsule_text, related_text, policy["total_chars"]
            )

        # 6. 组装 messages
        # 先算可引用历史（带真实消息 id），把标签注入 prompt 让 LLM 能用 [回复: mN]
        quote_history = [
            HistoryMessage(
                message_id=int(m["msg_id"]) if m.get("msg_id") else 0,
                sender="对方" if m["role"] == "user" else "我",
                text=m["content"],
            )
            for m in ctx
            if m["role"] in ("user", "assistant")
        ]
        quote_text, quote_targets = build_quote_options(
            quote_history, limit=self.qq_settings.quote_history_size
        )

        # 当前用户消息已是 ctx[-1]，世界书匹配只拼一次；仅保留最近上下文并限制字符预算。
        lore_context = "\n".join(
            " ".join(str(m.get("content") or "").split())
            for m in ctx[-6:] if m.get("content")
        )[-2000:]
        persona = build_persona_section(mode, card, self.user_name, lore_context)
        controls = None
        if mode.qq_controls or mode.image_tool:
            controls = build_qq_control_prompt(
                self.qq_settings.sticker_enabled if mode.qq_controls else False,
                qq_enabled=mode.qq_controls,
                allow_quote=bool(mode.qq_controls and quote_targets),
                image_enabled=mode.image_tool,
            )
            if mode.qq_controls and quote_text:
                # 不再复制历史正文；下方直接在对应 chat message 前标 [mN]。
                controls += (
                    "\n带 [mN] 前缀的历史消息可被引用；这些裸标签属于输入元数据，"
                    "不要复制到正文。确需引用时只输出一个上述完整引用控制标记。"
                )
        system = assemble_system_prompt(
            persona=persona,
            memory=memory_text,
            time_prompt=build_time_prompt(
                _now_str(),
                _gap_str(
                    self_id,
                    user_id,
                    self.memory_stores[mode.key],
                    mode_key=mode.key,
                    namespace=namespace,
                ),
            ),
            qq_controls=controls,
        )
        messages = [{"role": "system", "content": system}]
        if mode.key == "tavern" and card is not None and card.mes_example.strip():
            messages.extend(example_dialogue_to_messages(card, self.user_name, max_turns=6))
        quote_labels_by_id = {
            message_id: label for label, message_id in quote_targets.items()
        }
        for i, m in enumerate(ctx):
            # 兼容修复前已经进入短期上下文的协议泄漏，避免下一轮再次模仿并叠加。
            message_text = strip_bare_quote_labels(str(m.get("content") or ""))
            label = quote_labels_by_id.get(int(m["msg_id"])) if m.get("msg_id") else None
            if label:
                message_text = f"[{label}] {message_text}"
            if mode.vision and image_list and i == len(ctx) - 1 and m.get("role") == "user":
                content = [{"type": "text", "text": message_text}]
                content += [
                    {"type": "image_url", "image_url": {"url": u}} for u in image_list
                ]
                messages.append({"role": "user", "content": content})
            else:
                messages.append({"role": m["role"], "content": message_text})

        return _Prepared(
            self_id=self_id,
            user_id=user_id,
            text=text,
            image_list=image_list,
            mode=mode,
            namespace=namespace,
            ctx=ctx,
            user_msg_id=user_msg_id,
            messages=messages,
            quote_targets=quote_targets,
            card=card,
        )

    def _rollback_user_message(self, prepared: _Prepared) -> None:
        """LLM 失败或空回复时，回滚本轮在内存上下文里追加的用户消息。"""
        ctx = prepared.ctx
        if (
            ctx
            and ctx[-1].get("role") == "user"
            and ctx[-1].get("msg_id") == prepared.user_msg_id
        ):
            ctx.pop()

    async def _finalize(
        self, prepared: _Prepared, raw: str
    ) -> list[ReplyAction]:
        """清洗、解析动作、落库并触发抽取（步骤 9-10）。"""
        # `[mN]` 是注入历史消息的只读定位元数据。模型若误抄到输出行首，必须在
        # 空回复判定、动作解析和持久化之前统一清理，避免协议泄漏继续进入上下文。
        raw = strip_bare_quote_labels(raw)
        if not raw.strip():
            self._rollback_user_message(prepared)
            return [ReplyAction(text="刚才的回复和历史内容重复了，请换个说法再试一次。")]

        mode = prepared.mode
        # 9. 解析动作（quote_targets 已在 _prepare 算好；raw 已完成标签清理）
        max_chars = (
            mode.max_reply_chars
            if mode.max_reply_chars is not None
            else self.cs.max_reply_chars
        )
        max_actions = (
            mode.max_reply_actions
            if mode.max_reply_actions is not None
            else self.cs.max_reply_actions
        )
        max_bubble_chars = (
            mode.max_bubble_chars
            if mode.max_bubble_chars is not None
            else self.cs.max_bubble_chars
        )
        actions = parse_reply_actions(
            raw,
            prepared.quote_targets,
            self._sticker_catalog,
            max_chars=max_chars,
            max_actions=max_actions,
            max_bubble_chars=max_bubble_chars,
        )
        image_action_count = sum(bool(action.image_prompt) for action in actions)
        if mode.image_tool:
            logger.info(
                "回复动作解析：mode=%s actions=%d image_actions=%d image_markers=%d",
                mode.key,
                len(actions),
                image_action_count,
                len(extract_image_prompts(raw)),
            )
        # 用户明确要求生图时，不能把“模型是否恰好遵守控制格式”当成任务触发条件。
        # 若模型已输出标记则尊重其 prompt；若漏标记，使用最近场景生成一个独立生图动作。
        if (
            mode.image_tool
            and self._has_explicit_image_intent(prepared.text)
            and not any(action.image_prompt for action in actions)
            and not extract_image_prompts(raw)
        ):
            actions.append(ReplyAction(
                image_prompt=self._fallback_image_prompt(
                    prepared.text, prepared.ctx, prepared.card
                )
            ))
            logger.info("LLM 未输出生图标记，已根据明确用户意图创建生图任务")

        # 10. 落库与触发抽取（按当前模式落到各自的记忆库，实现模式间隔离）
        self._persist(mode, prepared.self_id, prepared.user_id, prepared.text, raw, prepared.namespace)
        self._maybe_extract(mode.key, prepared.self_id, prepared.user_id, prepared.namespace)

        return actions

    async def generate_reply_stream(
        self,
        self_id: str | int,
        user_id: str | int,
        text: str,
        images: Iterable[str] | None = None,
        *,
        event: object | None = None,
    ):
        """流式生成：逐块产出文本增量，结束再产出最终动作。

        产生两类事件字典：``{"delta": 文本增量}`` 与 ``{"actions": [ReplyAction]}``。
        命令短路与空消息走 ``actions`` 单帧（不流式）。流式时控制协议已实时滤除，
        因此最终 ``actions`` 仅用于派生生图等结构化动作，文本不需要重发。
        """
        self_id = str(self_id)
        user_id = str(user_id)
        text = (text or "").strip()
        image_list = list(images or [])

        # 1. 命令短路
        cmd = parse_command(text)
        if cmd is not None:
            handled = await self._handle_command(self_id, user_id, cmd, text)
            if handled is not None:
                yield {"actions": handled}
                return

        if not text and not image_list:
            return

        prepared = await self._prepare(self_id, user_id, text, image_list, event)
        cleaner = _StreamCleaner()
        raw_acc = ""
        try:
            async for chunk in self._generate_stream(prepared.mode, prepared.messages):
                raw_acc += chunk
                display = cleaner.feed(chunk)
                if display:
                    yield {"delta": display}
        except Exception:
            self._rollback_user_message(prepared)
            raise

        raw = strip_bare_quote_labels(raw_acc)
        raw, _ = filter_history_echo(raw, prepared.ctx)
        finish_text = cleaner.finish()
        if finish_text:
            yield {"delta": finish_text}
        if not raw.strip():
            self._rollback_user_message(prepared)
            yield {"actions": [ReplyAction(text="刚才的回复和历史内容重复了，请换个说法再试一次。")]}
            return
        actions = await self._finalize(prepared, raw)
        yield {"actions": actions}

    async def _generate_stream(self, mode: Mode, messages: list[dict]):
        """流式生成文本增量，并按配置记录完整对话 LLM 输入输出。

        酒馆客户端在未配置 ``SILLYTAVERN_URL`` 时本来也是直连同一 Ollama；若仍
        调用其非流式 ``chat``，Web 只能在生成完成后收到一个整段 delta。仅当确实
        配置了 SillyTavern 服务端代理时才暂时回退整段输出，否则统一走 LLMService
        的原生 Ollama 流式接口，角色卡 prompt 已在 ``_prepare`` 阶段完整组装。
        """
        llm_settings = getattr(self.llm_service, "s", None)
        log_full_io = bool(getattr(llm_settings, "log_full_io", False))
        if log_full_io:
            logger.info(
                "LLM 完整输入：mode=%s model=%s messages=%s",
                mode.key,
                mode.model,
                json.dumps(messages, ensure_ascii=False, default=str),
            )

        use_sillytavern_proxy = bool(
            mode.key == "tavern"
            and self.sillytavern_client is not None
            and getattr(self.sillytavern_client, "st_url", "")
        )
        if use_sillytavern_proxy:
            output = await self.sillytavern_client.chat(
                messages, mode.model, mode.temperature, mode.num_predict
            )
            if log_full_io:
                logger.info(
                    "LLM 完整输出：mode=%s model=%s output=%s",
                    mode.key,
                    mode.model,
                    output,
                )
            yield output
            return

        output_parts: list[str] = []
        async for delta in self.llm_service.stream(
            messages,
            model=mode.model,
            vision=mode.vision,
            temperature=mode.temperature,
            num_predict=mode.num_predict,
        ):
            output_parts.append(delta)
            yield delta
        if log_full_io:
            logger.info(
                "LLM 完整输出：mode=%s model=%s output=%s",
                mode.key,
                mode.model,
                "".join(output_parts),
            )

    async def _generate(self, mode: Mode, messages: list[dict]) -> str:
        llm_settings = getattr(self.llm_service, "s", None)
        log_full_io = bool(getattr(llm_settings, "log_full_io", False))
        if log_full_io:
            logger.info(
                "LLM 完整输入：mode=%s model=%s messages=%s",
                mode.key,
                mode.model,
                json.dumps(messages, ensure_ascii=False, default=str),
            )
        if mode.key == "tavern" and self.sillytavern_client is not None:
            output = await self.sillytavern_client.chat(
                messages, mode.model, mode.temperature, mode.num_predict
            )
        else:
            output = await self.llm_service.call(
                messages,
                model=mode.model,
                vision=mode.vision,
                temperature=mode.temperature,
                num_predict=mode.num_predict,
            )
        if log_full_io:
            logger.info(
                "LLM 完整输出：mode=%s model=%s output=%s",
                mode.key,
                mode.model,
                output,
            )
        return output

    def _persist(
        self,
        mode: Mode,
        self_id: str,
        user_id: str,
        user_text: str,
        raw_reply: str,
        namespace: str | None,
    ) -> None:
        now = int(time.time())
        store = self.memory_stores[mode.key]
        try:
            store.save_runtime_message(
                f"rt-{uuid.uuid4().hex}",
                self_id,
                user_id,
                user_id,
                now,
                user_text,
                0,
                mode_key=mode.key,
                namespace=namespace,
            )
            store.touch_interaction(
                self_id,
                user_id,
                user_at=now,
                mode_key=mode.key,
                namespace=namespace,
            )
            if raw_reply:
                store.save_runtime_message(
                    f"rt-{uuid.uuid4().hex}",
                    self_id,
                    self_id,
                    user_id,
                    now,
                    raw_reply,
                    1,
                    mode_key=mode.key,
                    namespace=namespace,
                )
                store.touch_interaction(
                    self_id,
                    user_id,
                    bot_at=now,
                    mode_key=mode.key,
                    namespace=namespace,
                )
                _add_message(
                    self_id,
                    user_id,
                    "assistant",
                    raw_reply,
                    self.cs.max_context,
                    self.cs.context_timeout,
                    mode_key=mode.key,
                    namespace=namespace,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("对话落库失败：%s", exc)

    def recent_message_window(
        self, self_id: str | int, user_id: str | int, rounds: int = 2
    ) -> list[dict]:
        """返回当前模式/命名空间最近若干轮、可直接给 Web 展示的消息。"""
        self_id, user_id = str(self_id), str(user_id)
        mode = self.mode_manager.get((self_id, user_id))
        namespace = self._current_namespace(mode, self_id, user_id)
        rows = self.memory_stores[mode.key].load_recent_rounds(
            self_id,
            user_id,
            mode_key=mode.key,
            namespace=namespace,
            rounds=rounds,
        )
        messages: list[dict] = []
        for row in rows:
            raw = strip_bare_quote_labels(str(row["text"] or ""))
            send_type = int(row["send_type"] or 0)
            role = "assistant" if send_type == 1 else "user"
            if role == "assistant":
                actions = parse_reply_actions(
                    raw,
                    {},
                    self._sticker_catalog,
                    max_chars=0,
                    max_actions=0,
                    max_bubble_chars=0,
                )
                text = "\n".join(action.text for action in actions if action.text).strip()
            else:
                text = raw.strip()
            if not text:
                continue
            messages.append(
                {
                    "id": int(row["id"]),
                    "role": role,
                    "text": text,
                    "sent_at": int(row["sent_at"] or 0),
                    "mode": mode.key,
                }
            )
        return messages

    def register_sent_ids(
        self, self_id: str, user_id: str, ids: Iterable[int | None]
    ) -> None:
        """按动作位置回填真实消息 id，供引用定位与开场白成功确认。"""
        ids = [int(i) if i is not None else None for i in ids]
        if not ids:
            return
        mode = self.mode_manager.get((self_id, user_id))
        scopes = [(mode.key, self._current_namespace(mode, self_id, user_id))]
        # /角色 可在其他模式预先执行；其开场白仍属于 tavern 角色上下文。
        if self.character_selection is not None:
            character = self.character_selection.get((self_id, user_id))
            tavern_scope = ("tavern", f"char:{_slug(character)}" if character else None)
            if tavern_scope not in scopes:
                scopes.append(tavern_scope)

        contexts = [
            _get_conversation(
                self_id,
                user_id,
                self.cs.context_timeout,
                mode_key=mode_key,
                namespace=namespace,
            )
            for mode_key, namespace in scopes
        ]
        # 命令产生的特殊 assistant 消息带动作索引，跨模式时也必须优先精确回填。
        for (_, scope_namespace), ctx in zip(scopes, contexts):
            for entry in reversed(ctx):
                index = entry.get("sent_id_index")
                if (
                    entry.get("role") == "assistant"
                    and not entry.get("msg_id")
                    and isinstance(index, int)
                ):
                    if 0 <= index < len(ids) and ids[index] is not None:
                        message_id = int(ids[index])
                        entry["msg_id"] = message_id
                        if entry.get("origin") == "character_greeting":
                            self._persist_character_greeting(
                                self_id,
                                user_id,
                                scope_namespace,
                                str(entry.get("content") or ""),
                                message_id,
                            )
                    elif entry.get("origin") == "character_greeting":
                        # 开场白未实际发出：撤销待发送上下文，使下次切换可以重试。
                        ctx.remove(entry)
                    return
        # 普通对话只回填当前模式中最后一条尚未绑定的 assistant 消息。
        first_sent_id = next((message_id for message_id in ids if message_id is not None), None)
        if first_sent_id is None:
            return
        for entry in reversed(contexts[0]):
            if entry.get("role") == "assistant" and not entry.get("msg_id"):
                entry["msg_id"] = int(first_sent_id)
                return

    def _maybe_extract(self, mode_key: str, self_id: str, user_id: str,
                       namespace: str | None = None) -> None:
        if self.online_extractor is None or not self.online_extractor.enabled_for(mode_key):
            return
        peer_id = user_id

        import asyncio

        scope_key = (self_id, peer_id, mode_key, namespace or "")
        if scope_key in self._extracting_scopes:
            return
        self._extracting_scopes.add(scope_key)

        async def _run() -> None:
            try:
                await self.online_extractor.maybe_extract(
                    peer_id, mode_key, namespace, self_id=self_id
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("在线记忆抽取失败：%s", exc)
            finally:
                self._extracting_scopes.discard(scope_key)

        async def _submit() -> None:
            accepted = await self.inference_queue.submit(_run, kind="memory")
            if not accepted:
                self._extracting_scopes.discard(scope_key)
                logger.warning("在线记忆抽取队列已满，已跳过当前触发")

        if self.inference_queue is not None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                self._extracting_scopes.discard(scope_key)
                return
            loop.create_task(_submit())
        else:
            self._extracting_scopes.discard(scope_key)
        # 无队列时不做抽取，避免阻塞（抽取不是回复的关键路径）
