from pathlib import Path
from types import SimpleNamespace

from core.character.client import CharacterCard, example_dialogue_to_messages
from core.conversation.prompts import (
    assemble_system_prompt,
    build_persona_section,
    build_qq_control_prompt,
)
from core.conversation.service import ConversationService
from core.modes.manager import Mode, build_modes


def _mode(key: str = "tavern") -> Mode:
    return Mode(
        key=key,
        label=key,
        model="m",
        vision_model="m",
        vision=False,
        prompt="模式行为规则",
        temperature=.8,
        num_predict=0,
        use_memory=True,
        qq_controls=True,
        image_tool=True,
    )


def test_tavern_persona_keeps_mode_rules_and_avoids_duplicate_identity_rules():
    card = CharacterCard(name="波奇", description="{{char}} 是吉他手")
    prompt = build_persona_section(_mode(), card, "用户", "吉他")
    assert "模式行为规则" in prompt
    assert "【当前角色】" in prompt
    assert "波奇 是吉他手" in prompt
    assert prompt.count("不要声明自己是 AI") == 0


def test_qq_controls_describe_each_enabled_capability_once():
    controls = build_qq_control_prompt(
        True, allow_quote=True, image_enabled=True
    )
    assert controls.count("[回复: mN]") == 1
    assert controls.count("[QQ表情: 名称]") == 1
    assert controls.count("[表情: 表情包名]") == 1
    # 说明行只允许出现一次；说明行内部的使用示例不计入重复说明
    def _declaration_count(text: str, marker: str) -> int:
        return sum(
            1 for line in text.splitlines() if line.lstrip().startswith(f"- {marker}")
        )

    assert _declaration_count(controls, "[生图: 生图提示词]") == 1
    system = assemble_system_prompt(
        persona="p", qq_controls=controls, image_tool=True
    )
    assert _declaration_count(system, "[生图: 生图提示词]") == 1


def test_writer_only_receives_image_control():
    controls = build_qq_control_prompt(
        False, qq_enabled=False, image_enabled=True
    )
    assert "[生图: 生图提示词]" in controls
    assert "[回复:" not in controls
    assert "[QQ表情:" not in controls
    assert "[表情:" not in controls


def test_example_dialogue_is_capped_and_starts_with_user():
    card = CharacterCard(
        name="A",
        mes_example=(
            "A: 孤立回复\n\n用户: 一\n\nA: 二\n\n用户: 三\n\n"
            "A: 四\n\n用户: 五\n\nA: 六\n\n用户: 七\n\nA: 八"
        ),
    )
    messages = example_dialogue_to_messages(card, "用户", max_turns=4)
    assert len(messages) <= 4
    assert messages[0]["role"] == "user"
    assert messages[-1]["content"] == "八"


def test_memory_query_skips_low_information_and_expands_ellipsis():
    ctx = [
        {"role": "user", "content": "小林去了钟楼"},
        {"role": "assistant", "content": "他在那里找到了一封信"},
        {"role": "user", "content": "然后呢"},
    ]
    assert ConversationService._memory_query("嗯", ctx) == ""
    query = ConversationService._memory_query("然后呢", ctx)
    assert "钟楼" in query
    assert "一封信" in query
    assert query.endswith("然后呢")


def test_memory_capsule_is_compact_and_excludes_display_metadata():
    text = ConversationService._render_memory_capsule([
        {"kind": "person", "summary": "用户喜欢科幻", "confidence": .95},
        {"kind": "relationship", "summary": "双方约定称呼对方为主人", "confidence": .9},
        {"kind": "episode", "summary": "昨天去了 live", "confidence": .99},
    ], max_chars=200)
    assert "【关于对方】" in text
    assert "【双方关系】" in text
    assert "用户喜欢科幻" in text
    assert "称呼对方为主人" in text
    assert "昨天去了 live" not in text
    assert "置信" not in text


def test_memory_capsule_never_leaves_an_empty_heading_when_budget_is_tight():
    text = ConversationService._render_memory_capsule([
        {"kind": "person", "summary": "短事实"},
        {"kind": "relationship", "summary": "这条关系记忆会超过剩余预算"},
    ], max_chars=20)
    assert text == "【关于对方】\n- 短事实"
    assert "【双方关系】" not in text


def test_memory_merge_prefers_capsule_and_deduplicates_related_summary():
    capsule = "【关于对方】\n- 用户喜欢科幻"
    related = (
        "【当前话题相关】\n"
        "- [person，自 2026-08-04 起] 用户喜欢科幻（可信度 0.95）\n"
        "- [episode，发生于 2026-08-03] 昨天一起看了电影（可信度 0.90）"
    )
    merged = ConversationService._merge_memory_context(capsule, related, max_chars=300)
    assert merged.count("用户喜欢科幻") == 1
    assert "昨天一起看了电影" in merged
    assert merged.startswith("【关于对方】")


def test_memory_policy_reserves_capsule_and_related_budgets_per_mode():
    for key in ("clone", "tavern", "writer"):
        policy = ConversationService._memory_policy(_mode(key))
        assert policy["capsule_chars"] > 0
        assert policy["related_chars"] > 0
        assert policy["total_chars"] >= policy["capsule_chars"]
        assert "episode" not in policy["capsule_kind_limits"]


def test_mode_prompts_use_current_control_protocol_only():
    root = Path(__file__).resolve().parents[1]
    settings = SimpleNamespace(
        llm_model="m",
        llm_vision_model="vm",
        mode_tavern_model="t",
        mode_clone_model="c",
        mode_writer_model="w",
        llm_temperature=.6,
        llm_num_predict=0,
        mode_clone_prompt_path=root / "data" / "modes" / "clone.txt",
    )
    modes = build_modes(settings, root)
    assert "[[reply:" not in modes["clone"].prompt
    assert "stiker" not in modes["clone"].prompt
    assert "[生图:" not in modes["tavern"].prompt
    assert "[生图:" not in modes["writer"].prompt
