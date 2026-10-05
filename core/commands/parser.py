"""聊天命令解析（与 nonebot 解耦，便于单元测试 / 集成测试独立调用）。"
支持的命令：
- /mode <模式>  /模式 <模式>  模式 <模式>  /setmode <模式>   （+ 裸词 "酒馆"/"模仿"/"作家"，以及带斜杠的 /tavern /clone /writer /酒馆 /模仿 /作家）
  - 带参数：-> ("mode", 模式key)；无法识别 -> ("mode_unknown", 原文)
  - 裸 /mode /模式 /setmode /模式（无参数）：-> ("mode", None)，由处理器给出可用模式提示
- /状态 /status 状态 /队列 队列                       -> ("status",)
- /menu /菜单 菜单                                    -> ("menu",)
- /帮助 /help 帮助                                     -> ("help",)
- /帮助 <参数> /help <参数> 帮助 <参数>                -> ("help", 参数)
  - <参数> 为模式名（如 作家/tavern）时给该模式专属帮助；
    为命令名（如 作品/角色）时给该命令用法；否则给根帮助。
- /清记忆（默认）                                      -> ("clear_memory",)
  - 命名空间层：清当前角色（酒馆）/ 当前作品（作家）的记忆；未选定则按当前会话清理
- /清记忆 模式 /清记忆 全部 /清记忆 所有                 -> ("clear_memory", "mode") / ("clear_memory", "all")
  - 模式层：清空当前模式的全部记忆（所有角色/作品、所有会话）；`全部` 为兼容写法
- /记忆（默认）                                        -> ("memory",)
  - 命名空间层（只读）：查看当前角色（酒馆）/ 当前作品（作家）记住的内容；未选定则看本会话
- /记忆 模式 /记忆 全部 /记忆 所有                       -> ("memory", "mode")
  - 模式层（只读）：查看当前模式的全部记忆；`全部` 为兼容写法
- /生图 <提示词> /画图 <提示词> /image <提示词> 生图 <提示词> 画图 <提示词>
  - 带参数：-> ("image", 提示词)
  - 裸 /生图 /image /画图 /生图 /画图（无参数）：-> ("image", "")，由处理器提示需给描述
- /角色 <名称> /character <名称> /char <名称> 角色 <名称> 人物 <名称>
  - 带参数：-> ("character", 名称)
  - 裸 /角色 /character /char /角色 /人物（无参数）：-> ("character", "")，列出可用角色
- /作品 <标题> /work <标题> /novel <标题> 作品 <标题>
  - 带参数：-> ("work", 标题)；writer 模式记忆按作品隔离
  - 裸 /作品 /work /novel 作品（无参数）：-> ("work", "")，提示当前作品与用法

返回值是 ``(kind, *args)`` 形式的元组；无法识别返回 ``None``。"""

from __future__ import annotations

from typing import Optional, Tuple

# 模式别名（模式 key 映射）
MODE_ALIASES: dict[str, list[str]] = {
    "tavern": ["tavern", "酒馆", "角色扮演", "rp", "tavern角色扮演"],
    "clone": ["clone", "模仿", "人类模仿", "克隆", "myclone"],
    "writer": ["writer", "作家", "小说", "小说作家", "写作", "创作"],
}
# 直接作为整条消息出现的模式词（不带斜杠）
EXACT_MODE_WORDS: dict[str, str] = {
    "tavern": "tavern", "酒馆": "tavern",
    "clone": "clone", "模仿": "clone",
    "writer": "writer", "作家": "writer", "小说": "writer",
}
# 带斜杠的模式词（与 /菜单 里显示的 /tavern /clone /writer 对齐，避免「菜单宣传的命令用不了」）
SLASH_MODE_WORDS: dict[str, str] = {
    "/tavern": "tavern", "/酒馆": "tavern",
    "/clone": "clone", "/模仿": "clone",
    "/writer": "writer", "/作家": "writer", "/小说": "writer",
}

# 兼容旧代码里引用的小写别名
_MODE_ALIASES = MODE_ALIASES
_EXACT_MODE_WORDS = EXACT_MODE_WORDS


def resolve_mode_arg(arg: str) -> Optional[str]:
    """把用户输入的模式参数归一化成模式 key；无法识别返回 None"""
    w = (arg or "").strip().casefold()
    if not w:
        return None
    for key, aliases in MODE_ALIASES.items():
        if w == key or w in aliases:
            return key
    for key, aliases in MODE_ALIASES.items():
        if any(a in w for a in aliases):
            return key
    return None


# 兼容旧引用名
_resolve_mode_arg = resolve_mode_arg


def parse_command(raw: str, key=None) -> Optional[Tuple]:
    """解析一条聊天文本是否为机器人命令。

    ``key`` 仅为兼容插件层调用签名（会话键），解析逻辑并不依赖它。
    返回 ``(kind, *args)`` 或 ``None``。
    """
    t = (raw or "").strip()
    if not t:
        return None
    low = t.casefold()

    for prefix in ("/mode ", "/模式 ", "模式 ", "/setmode "):
        if low.startswith(prefix):
            arg = t[len(prefix):].strip()
            mk = resolve_mode_arg(arg)
            return ("mode", mk) if mk else ("mode_unknown", arg)

    if low in EXACT_MODE_WORDS:
        return ("mode", EXACT_MODE_WORDS[low])

    if low in SLASH_MODE_WORDS:
        return ("mode", SLASH_MODE_WORDS[low])

    if low in ("/状态", "/status", "状态", "/队列", "队列"):
        return ("status",)

    if low in ("/menu", "/菜单", "菜单"):
        return ("menu",)
    if low in ("/帮助", "/help", "帮助"):
        return ("help",)
    for p in ("/帮助 ", "/help ", "帮助 "):
        if low.startswith(p):
            arg = t[len(p):].strip()
            return ("help", arg)

    if low in ("/清记忆", "/清除记忆", "/清理记忆", "/记忆清理", "/clearmem",
               "清记忆", "清除记忆", "清理记忆", "记忆清理"):
        return ("clear_memory",)
    if low in ("/清记忆 模式", "/记忆清理 模式", "/clearmem mode",
               "清记忆 模式", "记忆清理 模式"):
        return ("clear_memory", "mode")
    if low in ("/清记忆 全部", "/清记忆 所有", "/记忆清理 全部", "/记忆清理 所有",
               "/clearmem all", "清记忆 全部", "清记忆 所有", "记忆清理 全部", "记忆清理 所有"):
        return ("clear_memory", "all")

    # 记忆回顾（只读，分层）：默认看当前角色/作品；模式 看本模式全部
    if low in ("/记忆", "/查看记忆", "/记忆回顾", "/mem", "记忆", "查看记忆", "记忆回顾"):
        return ("memory",)
    if low in ("/记忆 模式", "/记忆 全部", "/记忆 所有", "/查看记忆 模式",
               "/mem mode", "/mem all",
               "记忆 模式", "记忆 全部", "记忆 所有", "查看记忆 模式"):
        return ("memory", "mode")

    # 作品（writer 模式记忆按作品隔离）：/作品 <标题> /work <标题> /novel <标题>
    if low in ("/作品", "/work", "/novel", "作品"):
        return ("work", "")
    for p in ("/作品 ", "/work ", "/novel ", "作品 "):
        if low.startswith(p):
            arg = t[len(p):].strip()
            return ("work", arg)

    # 无参数的「列出角色 / 列出可生图」等，也允许裸词（不带尾随空格）
    if low in ("/角色", "/character", "/char", "角色", "人物"):
        return ("character", "")

    for p in ("/生图 ", "/画图 ", "/image ", "/draw ", "生图 ", "画图 "):
        if low.startswith(p):
            prompt = t[len(p):].strip()
            if prompt:
                return ("image", prompt)

    for p in ("/角色 ", "/character ", "/char ", "角色 ", "人物 "):
        if low.startswith(p):
            arg = t[len(p):].strip()
            return ("character", arg)

    # 无参数形式：给出提示而非静默落到 LLM
    if low in ("/mode", "/模式", "/setmode", "模式"):
        return ("mode", None)
    if low in ("/生图", "/image", "/画图", "生图", "画图"):
        return ("image", "")

    return None


# 兼容旧引用名
_parse_command = parse_command
