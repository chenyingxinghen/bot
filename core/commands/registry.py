"""命令树（树形分支）与分层帮助文本。

命令按「模式分支」组织：

- 全局命令（``owner_mode=None``）：任何模式都可用
  （模式切换 / 状态 / 菜单 / 帮助 / 清记忆 / 生图）。
- writer 分支：``/作品``
- tavern 分支：``/角色``
- clone 分支：无专属命令

``command_owner(kind)`` 返回该命令所属模式 key 或 ``None``（全局）。
当某个模式专属命令在「非所属模式」下被使用时，handler 会给出跨模式提示
（见 ``service._cross_mode_hint``）。

所有帮助文本构造都是**纯函数**，模式列表 ``modes`` 由调用方（service）传入，
避免与 service 形成环依赖。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CommandInfo:
    """一条命令的元数据。"""

    kind: str
    label: str               # 中文名（如「当前作品」）
    owner_mode: str | None   # None = 全局命令
    trigger: str             # 主触发词（如 /作品）
    summary: str             # 一句话说明
    usage: str               # 详细用法（/help <命令> 展示）


# ── 全局命令 ──
GLOBAL: dict[str, CommandInfo] = {
    "mode": CommandInfo(
        "mode", "切换模式", None, "/作家",
        "在酒馆 / 模仿 / 作家三种模式间切换",
        "用法：直接发送 /作家、/酒馆、/模仿 切换；或 /模式 <名称>。\n"
        "切换后该模式的记忆、角色 / 作品上下文相互独立，不会互相串味。",
    ),
    "status": CommandInfo(
        "status", "查看状态", None, "/状态",
        "查看当前模式与推理队列",
        "用法：/状态。返回当前所在模式，以及推理队列的在途 / 待处理 / 累计提交数。",
    ),
    "clear_memory": CommandInfo(
        "clear_memory", "清除记忆", None, "/清记忆",
        "分层清除记忆：当前角色/作品，或整模式",
        "记忆清理是分层的，且只在当前模式内生效（各模式记忆库相互独立）：\n"
        "  /清记忆         【命名空间层】清除当前角色（酒馆）或当前作品（作家）的记忆；\n"
        "                 未选定角色/作品时按当前会话清理。关于你的全局记忆与其他角色/作品保留。\n"
        "  /清记忆 模式     【模式层】清空你在当前模式的全部记忆（所有角色/作品）。\n"
        "  /清记忆 全部     同上（兼容写法，不影响其他 QQ 用户）。\n"
        "清理会同步删除对应作用域的数据库原始对话；切到另一模式再 /清记忆 不会影响其它模式。",
    ),
    "memory": CommandInfo(
        "memory", "查看记忆", None, "/记忆",
        "分层回顾：当前角色/作品，或整模式（只读）",
        "只读回顾已记住的内容，不会影响记忆本身：\n"
        "  /记忆         【命名空间层】查看当前角色（酒馆）或当前作品（作家）记住的内容；\n"
        "               未选定角色/作品时查看本会话。\n"
        "  /记忆 模式     【模式层】查看你在当前模式的全部记忆（所有角色/作品）。\n"
        "  /记忆 全部     同上（兼容写法，不会展示其他 QQ 用户）。\n"
        "用 /清记忆 可清理；各模式记忆相互独立。",
    ),
    "image": CommandInfo(
        "image", "直接生图", None, "/生图",
        "根据描述直接生成图片",
        "用法：/生图 <描述> 或 /image <描述>。例如 /生图 雨夜下的赛博城市。\n"
        "三种模式下都能用；酒馆 / 作家 / 模仿模式中也可以让 AI 在对话里直接生图。",
    ),
    "menu": CommandInfo(
        "menu", "启动提示", None, "/菜单",
        "显示根处的启动提示与各模式概览",
        "用法：/菜单。重新显示机器人启动时的欢迎提示与三种模式总览。",
    ),
    "help": CommandInfo(
        "help", "帮助", None, "/帮助",
        "分层帮助：根 / 模式 / 命令",
        "用法：\n"
        "  /帮助            查看全局指令参考\n"
        "  /帮助 <模式>     查看某模式的专属命令（如 /帮助 作家、/help tavern）\n"
        "  /帮助 <命令>     查看某命令的详细用法（如 /帮助 作品、/help 角色）",
    ),
}

# ── 各模式专属命令 ──
BRANCH: dict[str, dict[str, CommandInfo]] = {
    "writer": {
        "work": CommandInfo(
            "work", "当前作品", "writer", "/作品",
            "writer 模式记忆按作品隔离",
            "用法：/作品 <标题> 开始或切换一部作品，writer 记忆将按作品隔离；\n"
            "/作品（无参数）查看当前作品。切回旧作用 /作品 <旧标题>。\n"
            "提示：开始写新小说前先 /作品 <新标题>，否则记忆会与上一篇混在一起。",
        ),
    },
    "tavern": {
        "character": CommandInfo(
            "character", "当前角色", "tavern", "/角色",
            "酒馆模式按角色隔离记忆",
            "用法：/角色 <名称> 切换酒馆角色（如 /角色 Bocchi）；\n"
            "/角色（无参数）列出可用角色。不同角色的记忆互相隔离。",
        ),
    },
}


def command_owner(kind: str) -> str | None:
    """返回命令所属模式 key；全局命令返回 None。"""
    if kind in GLOBAL:
        return None
    for mode_key, cmds in BRANCH.items():
        if kind in cmds:
            return mode_key
    return None


# 命令参数的别名 → kind（用于 /help <命令>）
_COMMAND_ALIASES: dict[str, list[str]] = {
    "mode": ["mode", "模式", "切换模式"],
    "status": ["status", "状态"],
    "clear_memory": ["clear_memory", "清记忆", "清除记忆", "清理记忆", "记忆清理", "clearmem"],
    "memory": ["memory", "记忆", "查看记忆", "记忆回顾", "mem"],
    "image": ["image", "生图", "画图", "draw"],
    "menu": ["menu", "菜单"],
    "help": ["help", "帮助"],
    "work": ["work", "作品", "novel"],
    "character": ["character", "角色", "人物", "char"],
}


def resolve_command_arg(arg: str) -> str | None:
    """把 /help <命令> 的参数归一化成命令 kind；无法识别返回 None。"""
    w = (arg or "").strip().casefold().lstrip("/")
    if not w:
        return None
    for kind, aliases in _COMMAND_ALIASES.items():
        if w == kind or w in aliases:
            return kind
    return None


def _mode_label(modes: list, key: str) -> str:
    for m in modes:
        if m.key == key:
            return m.label
    return key


def root_prompt_text(modes: list) -> str:
    """根处的启动提示（/菜单 显示，也可在 bot 启动时发送）。"""
    lines = ["欢迎使用多模式 QQ 机器人 🤖", ""]
    lines.append("我支持三种模式，用 /作家 /酒馆 /模仿 切换：")
    for m in modes:
        if m.key == "tavern":
            extra = "，用 /角色 切换角色"
        elif m.key == "writer":
            extra = "，用 /作品 隔离不同作品的记忆"
        else:
            extra = ""
        lines.append(f"  · {m.label}（/{m.key}）{extra}")
    lines.append("")
    lines.append("发送 /帮助 查看完整指令；/帮助 <模式> 看某模式的专属命令。")
    return "\n".join(lines)


def help_root_text(modes: list) -> str:
    """根层帮助：全局命令参考 + 各模式专属命令引导。"""
    lines = ["指令帮助（根）", "",
             "【全局命令】（任何模式可用）"]
    order = ["mode", "status", "clear_memory", "memory", "image", "menu", "help"]
    for kind in order:
        c = GLOBAL[kind]
        lines.append(f"  {c.trigger}  {c.summary}")
    lines.append("")
    lines.append("各模式还有专属命令：")
    for m in modes:
        branch = BRANCH.get(m.key, {})
        if branch:
            names = "、".join(c.trigger for c in branch.values())
            lines.append(f"  · {m.label}：{names}")
    lines.append("")
    lines.append("查看详情：/帮助 <模式>（如 /help 作家）或 /帮助 <命令>（如 /help 作品）。")
    return "\n".join(lines)


def help_mode_text(modes: list, mode_key: str) -> str | None:
    """模式层帮助：该模式的通用命令 + 专属命令。"""
    mode = next((m for m in modes if m.key == mode_key), None)
    if mode is None:
        return None
    lines = [f"「{mode.label}」模式命令（/{mode.key}）", "",
             "【通用命令】"]
    order = ["mode", "status", "clear_memory", "memory", "image", "menu", "help"]
    for kind in order:
        c = GLOBAL[kind]
        lines.append(f"  {c.trigger}  {c.summary}")
    branch = BRANCH.get(mode_key, {})
    if branch:
        lines.append("")
        lines.append("【本模式专属】")
        for c in branch.values():
            lines.append(f"  {c.trigger}  {c.summary}")
    lines.append("")
    lines.append(f"用 /帮助 <命令> 查看某条命令的详细用法（如 /help 作品）。")
    return "\n".join(lines)


def help_command_text(modes: list, kind: str) -> str | None:
    """命令层帮助：单条命令的详细用法。"""
    info = GLOBAL.get(kind)
    if info is None:
        for branch in BRANCH.values():
            if kind in branch:
                info = branch[kind]
                break
    if info is None:
        return None
    owner_label = ""
    if info.owner_mode:
        owner_label = f"（{_mode_label(modes, info.owner_mode)} 模式专属）"
    return f"{info.trigger} · {info.label}{owner_label}\n\n{info.usage}"
