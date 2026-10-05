"""命令路由与处理全覆盖测试。

不依赖 nonebot：直接驱动 ``parse_command``（解析）与 ``ConversationService._handle_command``
（处理），并验证指令在 ``generate_reply`` 中确实短路、不会落到 LLM。
"""

import asyncio
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, AsyncMock

sys.path.insert(0, "F:/NapCat.Shell/bot")

from core.commands.parser import parse_command
from core.modes.manager import Mode, ModeManager
from core.conversation.service import ConversationService
from core.conversation.context import _add_message, _get_conversation
from core.qq.interactions import ReplyAction
from core.memory.store import MemoryStore
from core.character.client import (
    CharacterCard, build_persona_prompt, example_dialogue_to_messages,
    render_first_message,
)


# ── 构造最小可用的 ConversationService ──

def _make_mode(key, label):
    return Mode(
        key=key, label=label, model="m", vision_model="m", vision=False,
        prompt="p", temperature=0.9, num_predict=256, use_memory=False,
        qq_controls=True, image_tool=True,
    )


def _build_service(
    tmp_path: Path, *, memory_stores: dict[str, object] | None = None
) -> ConversationService:
    modes = {
        "tavern": _make_mode("tavern", "酒馆角色扮演"),
        "clone": _make_mode("clone", "人类模仿"),
        "writer": _make_mode("writer", "小说作家"),
    }
    mode_manager = ModeManager(modes, tmp_path / "modes_state.json", default="clone")

    qq_settings = SimpleNamespace(
        sticker_dir=tmp_path / "stickers",
        quote_history_size=5,
        sticker_enabled=True,
    )

    sillytavern_client = MagicMock()
    sillytavern_client.list_character_names.return_value = ["Bocchi", "Melissa"]
    sillytavern_client.resolve_character.return_value = CharacterCard(name="Bocchi")

    character_selection = MagicMock()
    character_selection.get.return_value = None
    character_selection.set.return_value = None

    inference_queue = MagicMock()
    inference_queue.stats = SimpleNamespace(active=0, pending=0, submitted=3, rejected=0)

    if memory_stores is None:
        memory_stores = {
            "clone": MagicMock(), "tavern": MagicMock(), "writer": MagicMock()
        }
    for store in memory_stores.values():
        if isinstance(store, MagicMock):
            store.has_namespace_memories.return_value = False
            store.has_runtime_messages.return_value = False
            store.get_interaction_state.return_value = None
            store.load_recent_context.return_value = []

    conversation_settings = SimpleNamespace(
        max_context=20,
        context_timeout=7200,
        max_reply_chars=0,
        max_reply_actions=3,
        max_bubble_chars=200,
        llm_echo_retry_context=6,
    )

    return ConversationService(
        memory_stores=memory_stores,
        llm_service=AsyncMock(),
        mode_manager=mode_manager,
        conversation_settings=conversation_settings,
        qq_settings=qq_settings,
        sillytavern_client=sillytavern_client,
        character_selection=character_selection,
        work_selection=MagicMock(),
        user_name="你",
        embedding_client=None,
        online_extractor=None,
        image_service=MagicMock(),
        inference_queue=inference_queue,
    )


# ── A. 纯解析：每种指令应返回正确的 (kind, *args) ──

def test_parse_all_commands():
    cases = {
        # 模式（带参）
        "/mode 酒馆": ("mode", "tavern"),
        "/模式 模仿": ("mode", "clone"),
        "/setmode writer": ("mode", "writer"),
        "作家": ("mode", "writer"),
        # 模式（带斜杠的快捷形式，菜单里 /tavern /clone /writer 就是这么印的）
        "/tavern": ("mode", "tavern"),
        "/clone": ("mode", "clone"),
        "/writer": ("mode", "writer"),
        "/酒馆": ("mode", "tavern"),
        "/模仿": ("mode", "clone"),
        "/作家": ("mode", "writer"),
        "/mode 不存在的": ("mode_unknown", "不存在的"),
        # 模式（裸，无参）
        "/mode": ("mode", None),
        "模式": ("mode", None),
        # 状态
        "/状态": ("status",),
        "队列": ("status",),
        # 菜单（中英文都要能识别）
        "/menu": ("menu",),
        "/菜单": ("menu",),
        "/帮助": ("help",),
        "帮助": ("help",),
        # 帮助带参数（分层）
        "/help 作家": ("help", "作家"),
        "/help 作品": ("help", "作品"),
        "帮助 tavern": ("help", "tavern"),
        # 角色（带参）
        "/角色 Bocchi": ("character", "Bocchi"),
        "角色 小明": ("character", "小明"),
        # 角色（裸）
        "/角色": ("character", ""),
        "人物": ("character", ""),
        # 生图（带参）
        "/生图 一只猫": ("image", "一只猫"),
        "/image a cat": ("image", "a cat"),
        # 生图（裸）
        "/生图": ("image", ""),
        "/image": ("image", ""),
    }
    for text, expected in cases.items():
        got = parse_command(text)
        assert got == expected, f"parse_command({text!r}) = {got}, 期望 {expected}"


def test_parse_non_command_returns_none():
    # 普通对话不应被识别为指令
    for text in ("你好啊", "今天天气不错", "我想去酒馆玩", "帮我画只猫好吗"):
        assert parse_command(text) is None, f"{text!r} 不应被识别为指令"


# ── B. 处理：每种 kind 应返回正确的 ReplyAction 与副作用 ──

def test_handle_mode_switch():
    async def _run():
        svc = _build_service(Path("/tmp/ns1"))
        actions = await svc._handle_command("s", "u", ("mode", "tavern"), "/mode 酒馆")
        assert len(actions) == 1 and isinstance(actions[0], ReplyAction)
        assert "已切换到" in actions[0].text and "酒馆" in actions[0].text
        assert svc.mode_manager.current_key(("s", "u")) == "tavern"
    asyncio.run(_run())


def test_handle_mode_none_shows_available():
    async def _run():
        svc = _build_service(Path("/tmp/ns2"))
        actions = await svc._handle_command("s", "u", ("mode", None), "/mode")
        assert "可用" in actions[0].text
        assert "酒馆" in actions[0].text and "模仿" in actions[0].text and "作家" in actions[0].text
    asyncio.run(_run())


def test_handle_mode_unknown():
    async def _run():
        svc = _build_service(Path("/tmp/ns3"))
        actions = await svc._handle_command("s", "u", ("mode_unknown", "xyz"), "/mode xyz")
        assert "未知模式" in actions[0].text
    asyncio.run(_run())


def test_handle_status():
    async def _run():
        svc = _build_service(Path("/tmp/ns4"))
        actions = await svc._handle_command("s", "u", ("status",), "/状态")
        txt = actions[0].text
        assert "当前模式" in txt and "推理队列" in txt
    asyncio.run(_run())


def test_handle_character_with_name():
    async def _run():
        svc = _build_service(Path("/tmp/ns5"))
        actions = await svc._handle_command("s", "u", ("character", "Bocchi"), "/角色 Bocchi")
        svc.character_selection.set.assert_called_once_with(("s", "u"), "Bocchi")
        assert "已切换到角色" in actions[0].text and "Bocchi" in actions[0].text
    asyncio.run(_run())


def test_handle_character_list():
    async def _run():
        svc = _build_service(Path("/tmp/ns6"))
        actions = await svc._handle_command("s", "u", ("character", ""), "/角色")
        txt = actions[0].text
        assert "当前可用角色" in txt and "Bocchi" in txt and "Melissa" in txt
    asyncio.run(_run())


def test_character_first_message_sent_and_persisted_when_scope_is_empty():
    async def _run():
        svc = _build_service(Path("/tmp/greeting-empty"))
        svc.sillytavern_client.resolve_character.return_value = CharacterCard(
            name="后藤一里", first_mes="{{char}}小声说：你好，{{user}}……"
        )
        svc.mode_manager.set(("greeting-empty-s", "greeting-empty-u"), "tavern")
        actions = await svc._handle_command(
            "greeting-empty-s", "greeting-empty-u", ("character", "波奇"), "/角色 波奇"
        )

        assert [action.text for action in actions] == [
            "已切换到角色：后藤一里（酒馆模式生效）。",
            "后藤一里小声说：你好，你……",
        ]
        svc.character_selection.set.assert_called_once_with(
            ("greeting-empty-s", "greeting-empty-u"), "后藤一里"
        )
        store = svc.memory_stores["tavern"]
        store.has_namespace_memories.assert_called_once_with(
            "greeting-empty-s", "greeting-empty-u", "char:后藤一里"
        )
        store.get_interaction_state.assert_called_once_with(
            "greeting-empty-s",
            "greeting-empty-u",
            mode_key="tavern",
            namespace="char:后藤一里",
        )
        store.save_runtime_message.assert_not_called()

        ctx = _get_conversation(
            "greeting-empty-s",
            "greeting-empty-u",
            svc.cs.context_timeout,
            mode_key="tavern",
            namespace="char:后藤一里",
        )
        assert len(ctx) == 1
        assert ctx[0]["role"] == "assistant"
        assert ctx[0]["content"] == "后藤一里小声说：你好，你……"
        assert ctx[0]["origin"] == "character_greeting"

        svc.character_selection.get.return_value = "后藤一里"
        svc.register_sent_ids("greeting-empty-s", "greeting-empty-u", [100, 101])
        assert ctx[0]["msg_id"] == 101
        save_args = store.save_runtime_message.call_args.args
        assert save_args[0] == "rt-greeting-101"
        assert save_args[1:8] == (
            "greeting-empty-s",
            "greeting-empty-s",
            "greeting-empty-u",
            save_args[4],
            "后藤一里小声说：你好，你……",
            1,
        )
        assert store.save_runtime_message.call_args.kwargs == {
            "mode_key": "tavern",
            "namespace": "char:后藤一里",
        }
    asyncio.run(_run())


def test_character_greeting_send_failure_does_not_persist_first_visit():
    async def _run():
        svc = _build_service(Path("/tmp/greeting-send-failure"))
        svc.sillytavern_client.resolve_character.return_value = CharacterCard(
            name="Bocchi", first_mes="你好呀"
        )
        await svc._handle_command(
            "greeting-failure-s", "greeting-failure-u", ("character", "Bocchi"), "/角色 Bocchi"
        )
        svc.character_selection.get.return_value = "Bocchi"
        # 仅切换回执发送成功，开场白动作位置明确为失败。
        svc.register_sent_ids("greeting-failure-s", "greeting-failure-u", [300, None])
        svc.memory_stores["tavern"].save_runtime_message.assert_not_called()
        ctx = _get_conversation(
            "greeting-failure-s", "greeting-failure-u", svc.cs.context_timeout,
            mode_key="tavern", namespace="char:bocchi",
        )
        assert ctx == []
        retry = await svc._handle_command(
            "greeting-failure-s", "greeting-failure-u", ("character", "Bocchi"), "/角色 Bocchi"
        )
        assert len(retry) == 2 and retry[1].text == "你好呀"
    asyncio.run(_run())


def test_character_greeting_persists_when_only_greeting_send_succeeds():
    async def _run():
        svc = _build_service(Path("/tmp/greeting-only-success"))
        svc.sillytavern_client.resolve_character.return_value = CharacterCard(
            name="Bocchi", first_mes="你好呀"
        )
        await svc._handle_command(
            "greeting-only-s", "greeting-only-u", ("character", "Bocchi"), "/角色 Bocchi"
        )
        svc.character_selection.get.return_value = "Bocchi"
        # 切换回执失败，但开场白成功，仍应精确持久化开场白。
        svc.register_sent_ids("greeting-only-s", "greeting-only-u", [None, 301])
        save_args = svc.memory_stores["tavern"].save_runtime_message.call_args.args
        assert save_args[0] == "rt-greeting-301"
        assert save_args[5] == "你好呀"
    asyncio.run(_run())


def test_character_greeting_sent_id_is_bound_across_modes():
    async def _run():
        svc = _build_service(Path("/tmp/greeting-cross-mode-id"))
        svc.sillytavern_client.resolve_character.return_value = CharacterCard(
            name="Bocchi", first_mes="你好呀"
        )
        _add_message(
            "greeting-cross-s", "greeting-cross-u", "user", "旧问题",
            svc.cs.max_context, svc.cs.context_timeout, mode_key="clone",
        )
        _add_message(
            "greeting-cross-s", "greeting-cross-u", "assistant", "clone 中未绑定的旧回复",
            svc.cs.max_context, svc.cs.context_timeout, mode_key="clone",
        )
        await svc._handle_command(
            "greeting-cross-s", "greeting-cross-u", ("character", "Bocchi"), "/角色 Bocchi"
        )
        svc.character_selection.get.return_value = "Bocchi"
        svc.register_sent_ids("greeting-cross-s", "greeting-cross-u", [200, 201])

        tavern_ctx = _get_conversation(
            "greeting-cross-s",
            "greeting-cross-u",
            svc.cs.context_timeout,
            mode_key="tavern",
            namespace="char:bocchi",
        )
        clone_ctx = _get_conversation(
            "greeting-cross-s", "greeting-cross-u", svc.cs.context_timeout,
            mode_key="clone",
        )
        assert tavern_ctx[0]["msg_id"] == 201
        assert clone_ctx[-1]["content"] == "clone 中未绑定的旧回复"
        assert clone_ctx[-1]["msg_id"] is None
    asyncio.run(_run())


def test_character_greeting_is_available_to_next_model_turn():
    async def _run():
        svc = _build_service(Path("/tmp/greeting-next-turn"))
        card = CharacterCard(name="Bocchi", first_mes="终于见到你了。")
        svc.sillytavern_client.resolve_character.return_value = card
        svc.sillytavern_client.chat = AsyncMock(return_value="嗯，我们继续聊吧。")
        svc.mode_manager.set(("greeting-next-s", "greeting-next-u"), "tavern")

        actions = await svc._handle_command(
            "greeting-next-s", "greeting-next-u", ("character", "Bocchi"), "/角色 Bocchi"
        )
        assert len(actions) == 2
        svc.character_selection.get.return_value = "Bocchi"
        reply = await svc.generate_reply("greeting-next-s", "greeting-next-u", "你好")
        assert reply[0].text == "嗯，我们继续聊吧。"

        model_messages = svc.sillytavern_client.chat.await_args.args[0]
        greeting_index = next(
            i for i, message in enumerate(model_messages)
            if message.get("role") == "assistant" and message.get("content") == "终于见到你了。"
        )
        user_index = next(
            i for i, message in enumerate(model_messages)
            if message.get("role") == "user" and "你好" in str(message.get("content"))
        )
        assert greeting_index < user_index
    asyncio.run(_run())


def test_character_first_message_not_sent_twice_in_same_process():
    async def _run():
        svc = _build_service(Path("/tmp/greeting-repeat"))
        svc.sillytavern_client.resolve_character.return_value = CharacterCard(
            name="Bocchi", first_mes="你好呀"
        )
        first = await svc._handle_command(
            "greeting-repeat-s", "greeting-repeat-u", ("character", "Bocchi"), "/角色 Bocchi"
        )
        second = await svc._handle_command(
            "greeting-repeat-s", "greeting-repeat-u", ("character", "Bocchi"), "/角色 Bocchi"
        )
        assert len(first) == 2
        assert len(second) == 1
        # 尚未经过发送层回填 ID，因此不应提前持久化。
        assert svc.memory_stores["tavern"].save_runtime_message.call_count == 0
    asyncio.run(_run())


def test_character_first_message_not_sent_when_interaction_state_exists():
    async def _run():
        svc = _build_service(Path("/tmp/greeting-state"))
        svc.sillytavern_client.resolve_character.return_value = CharacterCard(
            name="Bocchi", first_mes="你好呀"
        )
        svc.memory_stores["tavern"].get_interaction_state.return_value = {
            "last_bot_at": 1
        }
        actions = await svc._handle_command(
            "greeting-state-s", "greeting-state-u", ("character", "Bocchi"), "/角色 Bocchi"
        )
        assert len(actions) == 1
        svc.memory_stores["tavern"].save_runtime_message.assert_not_called()
    asyncio.run(_run())


def test_character_first_message_not_sent_when_short_context_exists():
    async def _run():
        svc = _build_service(Path("/tmp/greeting-short"))
        svc.sillytavern_client.resolve_character.return_value = CharacterCard(
            name="Bocchi", first_mes="你好呀"
        )
        _add_message(
            "greeting-short-s",
            "greeting-short-u",
            "user",
            "我们见过",
            svc.cs.max_context,
            svc.cs.context_timeout,
            mode_key="tavern",
            namespace="char:bocchi",
        )
        actions = await svc._handle_command(
            "greeting-short-s", "greeting-short-u", ("character", "Bocchi"), "/角色 Bocchi"
        )
        assert len(actions) == 1
        svc.memory_stores["tavern"].has_namespace_memories.assert_not_called()
    asyncio.run(_run())


def test_character_first_message_not_sent_when_namespace_memory_exists():
    async def _run():
        svc = _build_service(Path("/tmp/greeting-long"))
        svc.sillytavern_client.resolve_character.return_value = CharacterCard(
            name="Bocchi", first_mes="你好呀"
        )
        svc.memory_stores["tavern"].has_namespace_memories.return_value = True
        actions = await svc._handle_command(
            "greeting-long-s", "greeting-long-u", ("character", "Bocchi"), "/角色 Bocchi"
        )
        assert len(actions) == 1
        svc.memory_stores["tavern"].get_interaction_state.assert_not_called()
    asyncio.run(_run())


def test_clear_memory_resets_character_first_visit_state():
    async def _run():
        svc = _build_service(Path("/tmp/greeting-clear-reset"))
        svc.mode_manager.set(("clear-reset-s", "clear-reset-u"), "tavern")
        svc.character_selection.get.return_value = "Bocchi"
        svc.sillytavern_client.resolve_character.return_value = CharacterCard(
            name="Bocchi", first_mes="重新开始吧"
        )
        store = svc.memory_stores["tavern"]
        store.get_interaction_state.return_value = {"last_bot_at": 1}
        before = await svc._handle_command(
            "clear-reset-s", "clear-reset-u", ("character", "Bocchi"), "/角色 Bocchi"
        )
        assert len(before) == 1

        await svc._handle_command(
            "clear-reset-s", "clear-reset-u", ("clear_memory",), "/清记忆"
        )
        # 模拟真实 MemoryStore.clear_session 已删除 interaction_state。
        store.get_interaction_state.return_value = None
        after = await svc._handle_command(
            "clear-reset-s", "clear-reset-u", ("character", "Bocchi"), "/角色 Bocchi"
        )
        assert len(after) == 2
        assert after[1].text == "重新开始吧"
        # 原始消息可以保留，首次状态不再由历史 messages 直接决定。
        store.has_runtime_messages.assert_not_called()
    asyncio.run(_run())


def test_character_without_first_message_only_confirms_switch():
    async def _run():
        svc = _build_service(Path("/tmp/greeting-none"))
        svc.sillytavern_client.resolve_character.return_value = CharacterCard(
            name="Bocchi", first_mes="  "
        )
        actions = await svc._handle_command(
            "greeting-none-s", "greeting-none-u", ("character", "Bocchi"), "/角色 Bocchi"
        )
        assert len(actions) == 1
        svc.memory_stores["tavern"].has_namespace_memories.assert_not_called()
    asyncio.run(_run())


def test_invalid_character_does_not_change_selection():
    async def _run():
        svc = _build_service(Path("/tmp/greeting-invalid"))
        svc.sillytavern_client.resolve_character.return_value = None
        actions = await svc._handle_command(
            "greeting-invalid-s", "greeting-invalid-u", ("character", "不存在"), "/角色 不存在"
        )
        assert len(actions) == 1 and "未找到角色卡" in actions[0].text
        svc.character_selection.set.assert_not_called()
    asyncio.run(_run())


def test_render_first_message_replaces_placeholders():
    card = CharacterCard(name="Alice", first_mes="{{char}} meets {{user}}")
    assert render_first_message(card, "Bob") == "Alice meets Bob"


def test_handle_image_with_prompt():
    async def _run():
        svc = _build_service(Path("/tmp/ns7"))
        actions = await svc._handle_command("s", "u", ("image", "一只猫"), "/生图 一只猫")
        assert actions[0].image_prompt == "一只猫"
    asyncio.run(_run())


def test_handle_image_empty_prompt():
    async def _run():
        svc = _build_service(Path("/tmp/ns8"))
        actions = await svc._handle_command("s", "u", ("image", ""), "/生图")
        assert "请在" in actions[0].text
    asyncio.run(_run())


def test_parse_clear_memory_commands():
    assert parse_command("/清记忆") == ("clear_memory",)
    assert parse_command("/clearmem") == ("clear_memory",)
    assert parse_command("清记忆") == ("clear_memory",)
    assert parse_command("/清记忆 全部") == ("clear_memory", "all")
    assert parse_command("/记忆清理 所有") == ("clear_memory", "all")
    assert parse_command("/clearmem all") == ("clear_memory", "all")


def test_handle_clear_memory_session():
    async def _run():
        svc = _build_service(Path("/tmp/ns21"))
        _add_message(
            "s", "u", "user", "待清短期上下文", svc.cs.max_context,
            svc.cs.context_timeout, mode_key="clone",
        )
        actions = await svc._handle_command("s", "u", ("clear_memory",), "/清记忆")
        svc.memory_stores["clone"].clear_session.assert_called_once_with(
            "s", "u", None, mode_key="clone")
        ctx = _get_conversation("s", "u", svc.cs.context_timeout, mode_key="clone")
        assert ctx == []
        assert "状态已重置" in actions[0].text
        assert "1 条短期上下文" in actions[0].text
    asyncio.run(_run())


def test_handle_clear_memory_all():
    async def _run():
        svc = _build_service(Path("/tmp/ns22"))
        _add_message(
            "s", "u", "user", "默认命名空间", svc.cs.max_context,
            svc.cs.context_timeout, mode_key="clone",
        )
        _add_message(
            "s", "u", "user", "另一个命名空间", svc.cs.max_context,
            svc.cs.context_timeout, mode_key="clone", namespace="char:unused",
        )
        actions = await svc._handle_command("s", "u", ("clear_memory", "all"), "/清记忆 全部")
        svc.memory_stores["clone"].clear_user.assert_called_once_with("s", "u", "clone")
        assert _get_conversation("s", "u", svc.cs.context_timeout, mode_key="clone") == []
        assert _get_conversation(
            "s", "u", svc.cs.context_timeout, mode_key="clone", namespace="char:unused"
        ) == []
        assert "模式层清理，状态已重置" in actions[0].text
        assert "2 条短期上下文" in actions[0].text
    asyncio.run(_run())


def test_parse_clear_memory_mode():
    # 模式层触发词：/清记忆 模式 与 /清记忆 全部 都应落到 clear_all（模式层）
    assert parse_command("/清记忆 模式") == ("clear_memory", "mode")
    assert parse_command("清记忆 模式") == ("clear_memory", "mode")
    assert parse_command("/clearmem mode") == ("clear_memory", "mode")
    # 全部 仍为兼容写法（模式层）
    assert parse_command("/清记忆 全部") == ("clear_memory", "all")


def test_handle_clear_memory_mode():
    async def _run():
        svc = _build_service(Path("/tmp/ns25"))
        # 切到 tavern 也应只清「tavern」模式库（模式层），不影响其它模式
        svc.mode_manager.set(("s", "u"), "tavern")
        actions = await svc._handle_command("s", "u", ("clear_memory", "mode"), "/清记忆 模式")
        svc.memory_stores["tavern"].clear_user.assert_called_once_with("s", "u", "tavern")
        svc.memory_stores["clone"].clear_user.assert_not_called()
        svc.memory_stores["writer"].clear_user.assert_not_called()
        assert "模式层清理" in actions[0].text
    asyncio.run(_run())


def test_parse_memory_view():
    # 只读回顾命令，分层触发词与 /清记忆 对齐
    assert parse_command("/记忆") == ("memory",)
    assert parse_command("查看记忆") == ("memory",)
    assert parse_command("/mem") == ("memory",)
    assert parse_command("/记忆 模式") == ("memory", "mode")
    assert parse_command("/记忆 全部") == ("memory", "mode")
    # /记忆 不应被 /记忆清理（清理命令）误吞
    assert parse_command("/记忆清理") == ("clear_memory",)


def test_handle_memory_view_namespace():
    async def _run():
        import time
        svc = _build_service(Path("/tmp/ns26"))
        now = int(time.time())
        svc.memory_stores["clone"].list_memories.return_value = [
            {"kind": "person", "summary": "用户喜欢科幻", "keywords": "科幻",
             "confidence": .9, "valid_from": now, "valid_to": None},
        ]
        actions = await svc._handle_command("s", "u", ("memory",), "/记忆")
        svc.memory_stores["clone"].list_memories.assert_called_once_with("s", "u", None)
        svc.memory_stores["clone"].list_all.assert_not_called()
        assert "记忆回顾" in actions[0].text
        assert "用户喜欢科幻" in actions[0].text
    asyncio.run(_run())


def test_handle_memory_view_mode():
    async def _run():
        import time
        svc = _build_service(Path("/tmp/ns27"))
        now = int(time.time())
        svc.memory_stores["clone"].list_user.return_value = [
            {"kind": "person", "summary": "用户喜欢科幻", "keywords": "科幻",
             "confidence": .9, "valid_from": now, "valid_to": None},
            {"kind": "episode", "summary": "一起看了电影", "keywords": "电影",
             "confidence": .8, "valid_from": now, "valid_to": None},
        ]
        actions = await svc._handle_command("s", "u", ("memory", "mode"), "/记忆 模式")
        svc.memory_stores["clone"].list_user.assert_called_once_with("s", "u")
        svc.memory_stores["clone"].list_memories.assert_not_called()
        assert "模式层" in actions[0].text
        assert "用户喜欢科幻" in actions[0].text
        assert "一起看了电影" in actions[0].text
    asyncio.run(_run())


def test_persona_injects_lorebook_with_context():
    card = CharacterCard(
        name="A", description="d", personality="p", scenario="s",
        character_book={"entries": [{"keys": ["魔法"], "content": "世界存在魔法", "insertion_order": 0}]},
    )
    p = build_persona_prompt(card, "用户", context_text="他提到了魔法")
    assert "世界存在魔法" in p
    p2 = build_persona_prompt(card, "用户")
    assert "世界存在魔法" not in p2


def test_example_dialogue_to_messages():
    # ST 约定：多轮示例对话用空行（\n\n）分隔，_split_example_dialogue 按块切分。
    card = CharacterCard(name="A", mes_example="<START>\n\n{{user}}: 你好\n\n{{char}}: 你好呀")
    msgs = example_dialogue_to_messages(card, "用户")
    assert len(msgs) == 2
    assert msgs[0] == {"role": "user", "content": "你好"}
    assert msgs[1] == {"role": "assistant", "content": "你好呀"}


def test_handle_menu_shows_root_prompt():
    async def _run():
        svc = _build_service(Path("/tmp/ns9"))
        actions = await svc._handle_command("s", "u", ("menu",), "/菜单")
        txt = actions[0].text
        # /menu 现在显示根处的启动提示（欢迎语 + 三种模式概览）
        assert "欢迎使用多模式 QQ 机器人" in txt
        assert "/作家" in txt and "/酒馆" in txt and "/模仿" in txt
        assert "酒馆角色扮演" in txt and "小说作家" in txt and "人类模仿" in txt
    asyncio.run(_run())


def test_handle_help_root():
    async def _run():
        svc = _build_service(Path("/tmp/ns11"))
        # /help 与 /help （无参）都给根层指令参考
        for cmd in (("help",), ("help", "")):
            actions = await svc._handle_command("s", "u", cmd, "/help")
            txt = actions[0].text
            assert "指令帮助" in txt
            assert "全局命令" in txt
            assert "/作家" in txt and "/清记忆" in txt and "/生图" in txt
            assert "/帮助 <模式>" in txt  # 引导到分层
    asyncio.run(_run())


def test_handle_help_mode():
    async def _run():
        svc = _build_service(Path("/tmp/ns12"))
        # /help 作家 → 作家模式专属命令（含 /作品）
        actions = await svc._handle_command("s", "u", ("help", "作家"), "/help 作家")
        txt = actions[0].text
        assert "小说作家" in txt
        assert "本模式专属" in txt
        assert "/作品" in txt
        # /help tavern → 酒馆模式专属命令（含 /角色）
        actions = await svc._handle_command("s", "u", ("help", "tavern"), "/help tavern")
        txt = actions[0].text
        assert "酒馆角色扮演" in txt
        assert "/角色" in txt
    asyncio.run(_run())


def test_handle_help_command():
    async def _run():
        svc = _build_service(Path("/tmp/ns13"))
        # /help 作品 → 命令详细用法，标注 writer 模式专属
        actions = await svc._handle_command("s", "u", ("help", "作品"), "/help 作品")
        txt = actions[0].text
        assert "/作品" in txt
        assert "writer 记忆将按作品隔离" in txt
        assert "小说作家 模式专属" in txt
        # /help 角色
        actions = await svc._handle_command("s", "u", ("help", "角色"), "/help 角色")
        txt = actions[0].text
        assert "/角色" in txt
        assert "不同角色的记忆互相隔离" in txt
    asyncio.run(_run())


def test_handle_help_unknown():
    async def _run():
        svc = _build_service(Path("/tmp/ns21"))
        actions = await svc._handle_command("s", "u", ("help", "不存在xx"), "/help 不存在xx")
        assert "未找到" in actions[0].text
    asyncio.run(_run())


def test_cross_mode_hint_work_in_tavern():
    async def _run():
        svc = _build_service(Path("/tmp/ns25"))
        svc.mode_manager.set(("s", "u"), "tavern")  # 当前在酒馆，却用作家命令
        actions = await svc._handle_command("s", "u", ("work", "我的小说"), "/作品 我的小说")
        txt = actions[0].text
        # 仍执行（记录了作品），但提示它属于作家模式
        svc.work_selection.set.assert_called_once_with(("s", "u"), "我的小说")
        assert "ℹ️" in txt
        assert "酒馆角色扮演" in txt and "小说作家" in txt
        assert "已切换到作品" in txt
    asyncio.run(_run())


def test_cross_mode_hint_character_in_writer():
    async def _run():
        svc = _build_service(Path("/tmp/ns23"))
        svc.mode_manager.set(("s", "u"), "writer")  # 当前在作家，却用酒馆命令
        actions = await svc._handle_command("s", "u", ("character", "Bocchi"), "/角色 Bocchi")
        txt = actions[0].text
        svc.character_selection.set.assert_called_once_with(("s", "u"), "Bocchi")
        assert "ℹ️" in txt
        assert "小说作家" in txt and "酒馆角色扮演" in txt
        assert "已切换到角色" in txt
    asyncio.run(_run())


def test_no_cross_mode_hint_when_same_mode():
    async def _run():
        svc = _build_service(Path("/tmp/ns24"))
        svc.mode_manager.set(("s", "u"), "writer")  # 作家模式用 /作品 → 无提示
        actions = await svc._handle_command("s", "u", ("work", "x"), "/作品 x")
        txt = actions[0].text
        assert "ℹ️" not in txt
        assert "已切换到作品" in txt
    asyncio.run(_run())


def test_handle_unknown_kind_returns_none():
    async def _run():
        svc = _build_service(Path("/tmp/ns10"))
        assert await svc._handle_command("s", "u", ("foobar",), "x") is None
    asyncio.run(_run())


# ── C. 端到端路由：指令在 generate_reply 中短路，不会触达 LLM ──

def test_generate_reply_command_short_circuits_llm():
    async def _run():
        svc = _build_service(Path("/tmp/ns11"))
        actions = await svc.generate_reply("s", "u", "/状态")
        assert len(actions) == 1
        assert "当前模式" in actions[0].text
        # 命令应短路，不应调用 LLM
        svc.llm_service.call.assert_not_called()
    asyncio.run(_run())


def test_generate_reply_menu_short_circuits_llm():
    async def _run():
        svc = _build_service(Path("/tmp/ns12"))
        # 用英文 /menu 也要能触发菜单（根启动提示），且短路不触达 LLM
        actions = await svc.generate_reply("s", "u", "/menu")
        assert "欢迎使用多模式 QQ 机器人" in actions[0].text
        svc.llm_service.call.assert_not_called()
    asyncio.run(_run())


def test_generate_reply_slash_mode_short_circuits_llm():
    async def _run():
        svc = _build_service(Path("/tmp/ns13"))
        # 菜单里印的 /tavern 必须真能切换模式，且短路不触达 LLM
        actions = await svc.generate_reply("s", "u", "/tavern")
        assert "已切换到" in actions[0].text and "酒馆" in actions[0].text
        assert svc.mode_manager.current_key(("s", "u")) == "tavern"
        svc.llm_service.call.assert_not_called()
    asyncio.run(_run())


def test_low_information_message_still_injects_baseline_memory_capsule():
    async def _run():
        root = Path("/tmp") / f"memory-capsule-{time.time_ns()}"
        stores = {
            key: MemoryStore(root / f"memory_{key}.db")
            for key in ("clone", "tavern", "writer")
        }
        now = int(time.time())
        stores["clone"].save_memories([
            {"owner_id": "capsule-s", "kind": "person", "subject_id": "capsule-u",
             "summary": "用户喜欢科幻小说", "keywords": ["科幻"], "confidence": .93,
             "valid_from": now, "valid_to": None},
            {"owner_id": "capsule-s", "kind": "episode",
             "subject_id": "relationship:capsule-s:capsule-u",
             "summary": "昨天一起看了一场电影", "keywords": ["电影"], "confidence": .95,
             "valid_from": now, "valid_to": None},
        ], ["capsule-seed"])
        svc = _build_service(root, memory_stores=stores)
        svc.mode_manager._modes["clone"] = replace(
            svc.mode_manager._modes["clone"], use_memory=True
        )
        svc.llm_service.call.return_value = "我记得。"

        actions = await svc.generate_reply("capsule-s", "capsule-u", "嗯")
        assert actions[0].text == "我记得。"
        messages = svc.llm_service.call.await_args.args[0]
        system = messages[0]["content"]
        assert "【长期记忆】" in system
        assert "【关于对方】" in system
        assert "用户喜欢科幻小说" in system
        assert "昨天一起看了一场电影" not in system
    asyncio.run(_run())


def test_generate_reply_strips_bare_quote_labels_before_send_and_persist():
    async def _run():
        svc = _build_service(Path("/tmp/ns26"))
        svc.llm_service.call.return_value = "[m9] [m9] [m8] 正常正文"
        actions = await svc.generate_reply("label-s", "label-u", "继续")
        assert len(actions) == 1
        assert actions[0].text == "正常正文"

        ctx = _get_conversation(
            "label-s", "label-u", svc.cs.context_timeout,
            mode_key="clone", namespace=None,
        )
        assert ctx[-1]["role"] == "assistant"
        assert ctx[-1]["content"] == "正常正文"
        saved_texts = [call.args[5] for call in svc.memory_stores["clone"].save_runtime_message.call_args_list]
        assert saved_texts[-1] == "正常正文"
    asyncio.run(_run())


def test_restart_restores_recent_context_from_runtime_messages():
    async def _run():
        root = Path("/tmp") / f"ns27-{time.time_ns()}"
        stores = {
            key: MemoryStore(root / f"memory_{key}.db")
            for key in ("clone", "tavern", "writer")
        }
        now = int(time.time())
        stores["clone"].save_runtime_message(
            "qq-user-10", "restore-s", "restore-u", "restore-u", now - 2,
            "重启前用户消息", 0, mode_key="clone",
        )
        stores["clone"].save_runtime_message(
            "qq-bot-11", "restore-s", "restore-s", "restore-u", now - 1,
            "[m9] 重启前机器人回复", 1, mode_key="clone",
        )

        svc = _build_service(root, memory_stores=stores)
        svc.llm_service.call.return_value = "恢复成功"
        actions = await svc.generate_reply("restore-s", "restore-u", "继续")
        assert actions[0].text == "恢复成功"

        messages = svc.llm_service.call.await_args.args[0]
        contents = [item.get("content") for item in messages]
        assert "重启前用户消息" in contents
        assert "重启前机器人回复" in contents
        assert "[m9] 重启前机器人回复" not in contents
    asyncio.run(_run())


def test_tavern_restart_recalls_unextracted_runtime_tail_after_context_timeout():
    async def _run():
        root = Path("/tmp") / f"ns-tavern-tail-{time.time_ns()}"
        stores = {
            key: MemoryStore(root / f"memory_{key}.db")
            for key in ("clone", "tavern", "writer")
        }
        now = int(time.time())
        stores["tavern"].save_runtime_message(
            "tail-user", "tavern-restore-s", "tavern-restore-u",
            "tavern-restore-u", now - 9000, "以后叫我主人", 0,
            mode_key="tavern", namespace="char:bocchi",
        )
        stores["tavern"].save_runtime_message(
            "tail-bot", "tavern-restore-s", "tavern-restore-s",
            "tavern-restore-u", now - 8999, "好的，主人。", 1,
            mode_key="tavern", namespace="char:bocchi",
        )

        svc = _build_service(root, memory_stores=stores)
        svc.mode_manager._modes["tavern"] = replace(
            svc.mode_manager._modes["tavern"], use_memory=True)
        svc.mode_manager.set(("tavern-restore-s", "tavern-restore-u"), "tavern")
        svc.character_selection.get.return_value = "Bocchi"
        svc.sillytavern_client.chat = AsyncMock(return_value="当然叫你主人。")
        actions = await svc.generate_reply(
            "tavern-restore-s", "tavern-restore-u", "你应该叫我什么？")
        assert actions[0].text == "当然叫你主人。"

        messages = svc.sillytavern_client.chat.await_args.args[0]
        system = messages[0]["content"]
        assert "尚未完成长期抽取的相关对话" in system
        assert "以后叫我主人" in system
        # 超过 2 小时的 runtime 尾巴通过记忆区召回，不应伪装成连续短期 chat history。
        assert "以后叫我主人" not in [item.get("content") for item in messages[1:]]
    asyncio.run(_run())


def test_restart_does_not_restore_context_before_clear_watermark():
    async def _run():
        root = Path("/tmp") / f"ns28-{time.time_ns()}"
        stores = {
            key: MemoryStore(root / f"memory_{key}.db")
            for key in ("clone", "tavern", "writer")
        }
        now = int(time.time())
        stores["clone"].save_runtime_message(
            "old-before-clear", "clear-s", "clear-u", "clear-u", now - 2,
            "不应恢复的旧消息", 0, mode_key="clone",
        )
        stores["clone"].clear_session("clear-s", "clear-u", mode_key="clone")

        svc = _build_service(root, memory_stores=stores)
        svc.llm_service.call.return_value = "清理后回复"
        await svc.generate_reply("clear-s", "clear-u", "新消息")
        messages = svc.llm_service.call.await_args.args[0]
        contents = [item.get("content") for item in messages]
        assert "不应恢复的旧消息" not in contents
    asyncio.run(_run())


def test_current_namespace_tavern_uses_character():
    svc = _build_service(Path("/tmp/nsn1"))
    svc.character_selection.get.return_value = "Bocchi"
    ns = svc._current_namespace(_make_mode("tavern", "酒馆"), "s", "u")
    assert ns == "char:bocchi"


def test_current_namespace_writer_uses_work():
    svc = _build_service(Path("/tmp/nsn2"))
    svc.work_selection.get.return_value = "novel2"
    ns = svc._current_namespace(_make_mode("writer", "作家"), "s", "u")
    assert ns == "work:novel2"


def test_current_namespace_clone_is_none():
    svc = _build_service(Path("/tmp/nsn3"))
    assert svc._current_namespace(_make_mode("clone", "模仿"), "s", "u") is None


def test_current_namespace_none_when_no_selection():
    svc = _build_service(Path("/tmp/nsn4"))
    svc.character_selection.get.return_value = ""  # 未选定角色
    assert svc._current_namespace(_make_mode("tavern", "酒馆"), "s", "u") is None


def test_parse_work_command():
    assert parse_command("/作品 我的小说") == ("work", "我的小说")
    assert parse_command("/work novel2") == ("work", "novel2")
    assert parse_command("/novel") == ("work", "")
    assert parse_command("作品") == ("work", "")


def test_handle_work_switch():
    async def _run():
        svc = _build_service(Path("/tmp/nsn5"))
        actions = await svc._handle_command("s", "u", ("work", "第二部"), "/作品 第二部")
        svc.work_selection.set.assert_called_once_with(("s", "u"), "第二部")
        assert "已切换到作品" in actions[0].text and "第二部" in actions[0].text
    asyncio.run(_run())


def test_handle_clear_memory_scoped_by_character():
    async def _run():
        svc = _build_service(Path("/tmp/nsn6"))
        svc.mode_manager.set(("s", "u"), "tavern")
        svc.character_selection.get.return_value = "Bocchi"
        _add_message(
            "s", "u", "user", "Bocchi 上下文", svc.cs.max_context,
            svc.cs.context_timeout, mode_key="tavern", namespace="char:bocchi",
        )
        _add_message(
            "s", "u", "user", "其他角色上下文", svc.cs.max_context,
            svc.cs.context_timeout, mode_key="tavern", namespace="char:melissa",
        )
        await svc._handle_command("s", "u", ("clear_memory",), "/清记忆")
        svc.memory_stores["tavern"].clear_session.assert_called_once_with(
            "s", "u", "char:bocchi", mode_key="tavern")
        assert _get_conversation(
            "s", "u", svc.cs.context_timeout,
            mode_key="tavern", namespace="char:bocchi",
        ) == []
        other = _get_conversation(
            "s", "u", svc.cs.context_timeout,
            mode_key="tavern", namespace="char:melissa",
        )
        assert other[0]["content"] == "其他角色上下文"
    asyncio.run(_run())


if __name__ == "__main__":
    test_parse_all_commands()
    test_parse_non_command_returns_none()
    test_handle_mode_switch()
    test_handle_mode_none_shows_available()
    test_handle_mode_unknown()
    test_handle_status()
    test_handle_character_with_name()
    test_handle_character_list()
    test_handle_image_with_prompt()
    test_handle_image_empty_prompt()
    test_handle_menu_shows_root_prompt()
    test_handle_help_root()
    test_handle_help_mode()
    test_handle_help_command()
    test_handle_help_unknown()
    test_cross_mode_hint_work_in_tavern()
    test_cross_mode_hint_character_in_writer()
    test_no_cross_mode_hint_when_same_mode()
    test_handle_unknown_kind_returns_none()
    test_generate_reply_command_short_circuits_llm()
    test_generate_reply_menu_short_circuits_llm()
    test_generate_reply_slash_mode_short_circuits_llm()
    test_generate_reply_strips_bare_quote_labels_before_send_and_persist()
    test_restart_restores_recent_context_from_runtime_messages()
    test_tavern_restart_recalls_unextracted_runtime_tail_after_context_timeout()
    test_restart_does_not_restore_context_before_clear_watermark()
    test_current_namespace_tavern_uses_character()
    test_current_namespace_writer_uses_work()
    test_current_namespace_clone_is_none()
    test_current_namespace_none_when_no_selection()
    test_parse_work_command()
    test_handle_work_switch()
    test_handle_clear_memory_scoped_by_character()
    print("ALL ROUTING TESTS PASSED")
