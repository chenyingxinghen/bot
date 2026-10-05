from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Protocol


class SegmentLike(Protocol):
    type: str
    data: Mapping[str, object]


# OneBot/NapCat 使用 QQ 表情 QSid。未知或新版表情仍会保留 ID，避免静默丢失
# 表情名依据 OneBot 11 / QQ 官方表情对照表（id→名称）；缺失 id 走原始 name 兜底
QQ_FACE_NAMES: dict[str, str] = {
    "0": "惊讶", "1": "撇嘴", "2": "色", "3": "发呆", "4": "得意",
    "10": "尴尬", "11": "发怒", "12": "调皮", "13": "呲牙", "14": "微笑",
    "15": "难过", "16": "酷", "18": "抓狂", "19": "吐", "20": "偷笑",
    "21": "可爱", "22": "白眼", "23": "傲慢", "24": "饥饿", "25": "困",
    "42": "爱情", "43": "跳跳", "46": "猪头", "49": "拥抱", "53": "蛋糕",
    "174": "无奈", "175": "卖萌", "176": "小纠结", "177": "喷血",
    "178": "斜眼笑", "179": "doge", "182": "笑哭", "183": "我最美",
    "212": "托腮", "262": "脑阔疼", "263": "沧桑", "264": "捂脸",
    "269": "暗中观察", "270": "emm", "271": "吃瓜", "272": "呵呵哒",
}

MEDIA_PLACEHOLDERS = {
    "record": "[语音消息]",
    "video": "[视频消息]",
    "file": "[文件]",
    "json": "[卡片消息]",
    "xml": "[卡片消息]",
    "forward": "[合并转发消息]",
    "dice": "[骰子]",
    "rps": "[猜拳]",
    "poke": "[戳一戳表情]",
}

STICKER_EXTENSIONS = {".png", ".gif", ".webp", ".jpg", ".jpeg", ".bmp"}
_CONTROL_PREFIX = re.compile(
    r"^\s*\[{1,2}\s*(reply|sticker|image|draw|生图|画图|图片|引用消息|引用|回复|表情)\s*[:：]\s*"
    r"([^\]\r\n]+?)\s*\]{1,2}\s*",
    re.I,
)
# LLM 偶尔会把生图标记放在句中、Markdown 行内代码中，或使用中文书名括号。
# 只对明确的 image/draw/生图/画图 控制词做兼容，不匹配普通“图片”描述，避免误触发。
_INLINE_IMAGE_TOKEN = re.compile(
    r"`{0,3}\s*(?:\[{1,2}|【)\s*(image|draw|生图|画图)\s*[:：]\s*"
    r"([^\]\r\n】]+?)\s*(?:\]{1,2}|】)\s*`{0,3}",
    re.I,
)
_QQ_FACE_TOKEN = re.compile(r"\[QQ表情\s*[:：]\s*([^\]\r\n]+?)\s*\]", re.I)
# `[mN]` 只用于给 LLM 标记可引用的历史消息，不是合法输出控制。部分模型会
# 模仿输入前缀并连续输出多个标签，因此在解析和持久化前移除行首泄漏标签。
_BARE_QUOTE_LABEL = re.compile(r"^\s*\[\s*m\d+\s*\]\s*", re.I)
_CONTROL_ALIASES = {
    "引用消息": "reply",
    "引用": "reply",
    "回复": "reply",
    "表情": "sticker",
    "生图": "image",
    "画图": "image",
    "draw": "image",
    "图片": "image",
}
QQ_FACE_IDS_BY_NAME: dict[str, int] = {
    name.casefold(): int(face_id) for face_id, name in QQ_FACE_NAMES.items()
}


@dataclass(frozen=True)
class HistoryMessage:
    message_id: int
    sender: str
    text: str


@dataclass(frozen=True)
class ReplyAction:
    text: str = ""
    reply_message_id: int | None = None
    sticker_path: Path | None = None
    sticker_name: str | None = None
    # 生图请求：LLM 通过 [[image: 描述]] / [[生图: 描述]] 标记发起
    # 由发送层调用 ComfyUI 生成并发送图片
    image_prompt: str | None = None
    # 已生成图片的本地路径（通常由发送层填充，测试可直接指定）
    image_path: Path | None = None


def _clean_name(value: object) -> str:
    text = str(value or "").strip()
    return text.removeprefix("/").strip()


def describe_face(data: Mapping[str, object]) -> str:
    face_id = str(data.get("id", "?")).strip()
    raw = data.get("raw")
    raw_data = raw if isinstance(raw, Mapping) else {}
    name = next(
        (_clean_name(value) for value in (
            data.get("name"), data.get("faceText"), raw_data.get("faceText"),
            raw_data.get("faceName"), raw_data.get("QDes"),
        ) if _clean_name(value)),
        "",
    )
    name = name or QQ_FACE_NAMES.get(face_id, "")
    return f"[QQ表情:{name}]" if name else f"[QQ表情:id={face_id}]"


def resolve_face_id(value: object) -> int | None:
    """把模型输出的 QQ 表情名称或 id=数字 还原为 OneBot face ID。"""
    token = _clean_name(value)
    if token.lower().startswith("id="):
        token = token[3:].strip()
    if token.isdigit():
        return int(token)
    return QQ_FACE_IDS_BY_NAME.get(token.casefold())


def _looks_like_unicode_emoji(value: str) -> bool:
    """粗略判断控制值是否包含可直接显示的 Unicode emoji。"""
    return any(
        0x1F000 <= ord(char) <= 0x1FAFF
        or 0x2600 <= ord(char) <= 0x27BF
        for char in value
    )


def parse_qq_face_segments(text: str) -> list[tuple[str, str | int]]:
    """把文字中的 [QQ表情:名称] 拆为有序的 text/face 消息段。

    已知名称或 ID 转为 QQ 原生 face；Unicode emoji（如 🥺）去掉协议外壳后
    直接作为文本发送；其他未知名称保留原文，避免拼写错误被静默吞掉。
    """
    parts: list[tuple[str, str | int]] = []
    cursor = 0
    for match in _QQ_FACE_TOKEN.finditer(text):
        token = match.group(1).strip()
        face_id = resolve_face_id(token)
        if face_id is None and not _looks_like_unicode_emoji(token):
            continue
        if match.start() > cursor:
            parts.append(("text", text[cursor:match.start()]))
        if face_id is not None:
            parts.append(("face", face_id))
        else:
            parts.append(("text", token))
        cursor = match.end()
    if cursor < len(text):
        parts.append(("text", text[cursor:]))
    return parts or [("text", text)]


def describe_message(message: Iterable[SegmentLike]) -> str:
    """把 OneBot 富消息转换为适合交给语言模型的语义文本。"""
    parts: list[str] = []
    for segment in message:
        kind = str(segment.type)
        data = segment.data
        if kind == "text":
            parts.append(str(data.get("text", "")))
        elif kind == "face":
            parts.append(describe_face(data))
        elif kind == "mface":
            summary = _clean_name(data.get("summary")) or _clean_name(data.get("name"))
            parts.append(f"[QQ表情:{summary or '商城表情'}]")
        elif kind == "image":
            summary = _clean_name(data.get("summary"))
            is_sticker = bool(data.get("emoji_id")) or "表情" in summary
            if is_sticker:
                parts.append(f"[QQ表情:{summary or '动画表情'}]")
            else:
                parts.append(f"[图片:{summary}]" if summary and summary != "图片" else "[图片]")
        elif kind == "at":
            target = str(data.get("name") or data.get("qq") or "某人")
            parts.append(f"[@{target}]")
        elif kind == "reply":
            # NoneBot 通常会提前取出 reply 段；保留此分支兼容未取出的消息
            parts.append("[引用了一条消息]")
        elif kind in MEDIA_PLACEHOLDERS:
            parts.append(MEDIA_PLACEHOLDERS[kind])
        else:
            parts.append(f"[QQ消息:{kind}]")
    return "".join(parts).strip()


class StickerCatalog:
    def __init__(self, root: Path):
        self.root = root
        self._stickers: dict[str, Path] = {}

    def refresh(self) -> dict[str, Path]:
        self.root.mkdir(parents=True, exist_ok=True)
        stickers: dict[str, Path] = {}
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in STICKER_EXTENSIONS:
                continue
            relative = path.relative_to(self.root).with_suffix("").as_posix()
            stickers[relative] = path.resolve()
            # 根目录文件可直接用文件名；子目录文件仍优先使用完整相对名
            if "/" not in relative:
                stickers.setdefault(path.stem, path.resolve())
        self._stickers = stickers
        return dict(stickers)

    def names(self, limit: int = 30) -> list[str]:
        self.refresh()
        return list(self._stickers)[:max(0, limit)]

    def resolve(self, name: str) -> Path | None:
        if not self._stickers:
            self.refresh()
        normalized = name.strip().replace("\\", "/")
        direct = self._stickers.get(normalized)
        if direct:
            return direct
        lowered = normalized.casefold()
        return next((path for alias, path in self._stickers.items()
                     if alias.casefold() == lowered), None)


def build_quote_options(history: Iterable[HistoryMessage], limit: int = 8) -> tuple[str, dict[str, int]]:
    usable = [item for item in history if item.message_id > 0 and item.text.strip()][-limit:]
    if not usable:
        return "", {}
    targets: dict[str, int] = {}
    lines = []
    for index, item in enumerate(usable, 1):
        label = f"m{index}"
        targets[label] = item.message_id
        text = " ".join(item.text.split())[:100]
        lines.append(f"{label}（{item.sender}）：{text}")
    return "\n".join(lines), targets


def _split_text(text: str, max_chars: int) -> list[str]:
    if max_chars <= 0:
        # max_chars<=0 表示不切分，整段作为一条消息（小说作家等长文本模式）
        return [text] if text else []
    chunks: list[str] = []
    for paragraph in (part.strip() for part in text.splitlines() if part.strip()):
        rest = paragraph
        while len(rest) > max_chars:
            window = rest[:max_chars + 1]
            cuts = [window.rfind(mark) + 1 for mark in "。！？" if window.rfind(mark) >= 0]
            cut = max(cuts, default=max_chars)
            chunks.append(rest[:cut].strip())
            rest = rest[cut:].strip()
        if rest:
            chunks.append(rest)
    return chunks


def extract_image_prompts(text: str) -> list[str]:
    """提取明确的 LLM 生图控制标记，供解析与显式请求兜底复用。"""
    return [match.group(2).strip() for match in _INLINE_IMAGE_TOKEN.finditer(text or "")
            if match.group(2).strip()]


def strip_bare_quote_labels(text: str) -> str:
    """移除模型误抄到输出行首的一个或多个裸 ``[mN]`` 定位标签。

    合法输出控制标记可以位于裸标签之前，因此先暂存连续控制前缀，再清理其后的
    裸标签并原样拼回控制前缀。正文中间的 ``[mN]`` 不受影响。
    """
    cleaned_lines: list[str] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        controls: list[str] = []
        while line:
            if match := _CONTROL_PREFIX.match(line):
                controls.append(line[:match.end()].strip())
                line = line[match.end():].lstrip()
                continue
            if match := _QQ_FACE_TOKEN.match(line):
                controls.append(line[:match.end()].strip())
                line = line[match.end():].lstrip()
                continue
            if _BARE_QUOTE_LABEL.match(line):
                line = _BARE_QUOTE_LABEL.sub("", line, count=1).lstrip()
                continue
            break
        cleaned = " ".join([*controls, line]).strip()
        if cleaned:
            cleaned_lines.append(cleaned)
    return "\n".join(cleaned_lines)


def parse_reply_actions(
    raw_reply: str,
    quote_targets: Mapping[str, int],
    stickers: StickerCatalog,
    max_chars: int = 0,
    max_actions: int = 6,
    max_bubble_chars: int = 0,
) -> list[ReplyAction]:
    """解析模型的引用/表情/生图控制前缀，兼容中英文和单双层括号。

    当 ``max_bubble_chars > 0`` 时，解析完成后会调用 ``_merge_short_actions`` 把
    连续的纯文本短气泡合并成一条，减少换行带来的刷屏感。

    ``max_actions`` 只限制**纯文本气泡**的数量；生图 / 引用 / 表情包等带副作用的
    动作不受此限，否则长回复（尤其是小说作家模式的整章输出）一旦超过文本气泡上限，
    末尾的 ``[生图:]`` 标记会因提前 break 而永远无法触发生图任务。
    """
    actions: list[ReplyAction] = []
    raw_reply = strip_bare_quote_labels(raw_reply)
    text_count = 0  # 仅统计纯文本气泡，用于 max_actions 限制
    for raw_line in (line.strip() for line in raw_reply.splitlines() if line.strip()):
        line = raw_line
        reply_id: int | None = None
        sticker_path: Path | None = None
        sticker_name: str | None = None
        image_prompt: str | None = None

        # 先抽取任意位置的明确生图标记。正常行首格式也会被这里处理；移除标记后，
        # 同行剩余文字仍作为该生图动作的说明发送。
        inline_match = _INLINE_IMAGE_TOKEN.search(line)
        if inline_match:
            image_prompt = inline_match.group(2).strip()
            line = (line[:inline_match.start()] + line[inline_match.end():]).strip()

        while match := _CONTROL_PREFIX.match(line):
            control, value = match.group(1).lower(), match.group(2).strip()
            control = _CONTROL_ALIASES.get(control, control)
            line = line[match.end():]
            if control == "reply":
                reply_id = quote_targets.get(value.lower())
            elif control == "sticker":
                sticker_path = stickers.resolve(value)
                sticker_name = value if sticker_path else None
            elif control == "image":
                image_prompt = value

        chunks = _split_text(line, max_chars)
        if not chunks:
            if sticker_path or image_prompt:
                actions.append(ReplyAction(
                    reply_message_id=reply_id,
                    sticker_path=sticker_path,
                    sticker_name=sticker_name,
                    image_prompt=image_prompt,
                ))
            continue

        for index, chunk in enumerate(chunks):
            is_last = index == len(chunks) - 1
            action = ReplyAction(
                text=chunk,
                reply_message_id=reply_id if index == 0 else None,
                sticker_path=sticker_path if is_last else None,
                sticker_name=sticker_name if is_last else None,
                image_prompt=image_prompt if is_last else None,
            )
            # 纯文本气泡受 max_actions 限制；带副作用（生图/引用/表情）的动作必须保留。
            has_side_effect = bool(
                action.image_prompt or action.sticker_path or action.reply_message_id
            )
            if action.text and not has_side_effect:
                if max_actions > 0 and text_count >= max_actions:
                    continue  # 超出文本气泡上限：跳过该纯文本气泡，但绝不提前 break，
                             # 否则后续行里的 [生图:] 等控制标记会被整段丢弃。
                text_count += 1
            actions.append(action)

    if max_bubble_chars > 0:
        actions = _merge_short_actions(actions, max_bubble_chars)
    return actions


def _merge_short_actions(
    actions: list[ReplyAction], max_bubble_chars: int
) -> list[ReplyAction]:
    """把连续的纯文本短动作合并为一条气泡，减少刷屏。

    带控制副作用的动作（引用 / 表情 / 生图）作为分隔边界，不参与合并；
    合并时以 ``\\n`` 连接，且累计长度不超过 ``max_bubble_chars``。
    """
    merged: list[ReplyAction] = []
    for action in actions:
        can_merge = (
            action.text is not None
            and not action.reply_message_id
            and not action.sticker_path
            and not action.sticker_name
            and not action.image_prompt
        )
        previous_is_plain_text = bool(
            merged
            and merged[-1].text is not None
            and not merged[-1].reply_message_id
            and not merged[-1].sticker_path
            and not merged[-1].sticker_name
            and not merged[-1].image_prompt
        )
        if can_merge and previous_is_plain_text:
            candidate = merged[-1].text + "\n" + action.text
            if len(candidate) <= max_bubble_chars:
                merged[-1] = ReplyAction(text=candidate)
                continue
        merged.append(action)
    return merged


def extract_message_id(send_result: object) -> int | None:
    if isinstance(send_result, Mapping):
        value = send_result.get("message_id")
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None
    try:
        return int(send_result) if send_result is not None else None
    except (TypeError, ValueError):
        return None
