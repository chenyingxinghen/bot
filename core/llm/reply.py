import re

_OUTPUT_CONTROL_PREFIX = re.compile(
    r"^\s*\[{1,2}\s*(?:reply|sticker|引用消息|引用|回复|表情)?\s*[:：]"
    r"[^\]\r\n]+\]{1,2}\s*",
    re.I,
)

def _clean_llm_reply(data: dict) -> str:
    reply = (data.get("message", {}).get("content") or "").strip()
    if "</think>" in reply:
        reply = reply.split("</think>", 1)[1].strip()
    return reply.replace("**", "").replace("*", "").replace("`", "")

def _normalize_dialogue_line(value: str) -> str:
    line = value.strip()
    while match := _OUTPUT_CONTROL_PREFIX.match(line):
        line = line[match.end():]
    return " ".join(line.split()).casefold()

def _filter_history_echo(raw_reply: str, ctx: list[dict]) -> tuple[str, int]:
    history_lines = {
        normalized
        for message in ctx
        for line in str(message.get("content", "")).splitlines()
        if len(normalized := _normalize_dialogue_line(line)) >= 3
    }
    if not history_lines:
        return raw_reply, 0

    reply_normalized = _normalize_dialogue_line(raw_reply)
    matched_history = {
        line for line in history_lines if line in reply_normalized
    }
    if len(matched_history) < 2:
        return raw_reply, 0

    kept_lines: list[str] = []
    for line in (item.strip() for item in raw_reply.splitlines() if item.strip()):
        normalized = _normalize_dialogue_line(line)
        contained = sum(1 for history in history_lines if history in normalized)
        if normalized in history_lines or contained >= 2:
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines), len(matched_history)
