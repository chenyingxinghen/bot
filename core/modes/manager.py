"""多模式（角色）系统。

三种聊天模式共享同一套消息收发与记忆基础设施，区别只在「用哪个模型 + 哪套系统
提示词 + 哪些控制标记」：

- tavern 酒馆角色扮演：用 gemma-4-e4b-uncensored-hauhaucs-aggressive，沉浸式第一
  人称角色扮演（酒馆 Tavern 风格）。
- clone  人类模仿：用 myclone-vl（视觉），模仿对方说话风格（即原 chat_style 行为）。
- writer 小说作家：用写作模型，创作连贯、有画面感的小说段落，可插图。

模式按 (self_id, peer_id) 维度记忆，切换后长期有效；持久化到 modes_state.json。
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

import json
from dataclasses import dataclass
from pathlib import Path

from core.state import atomic_write_json


@dataclass(frozen=True)
class Mode:
    key: str
    label: str
    model: str
    vision_model: str
    vision: bool
    prompt: str
    temperature: float
    num_predict: int            # <=0 表示不限制输出长度（交给模型/上下文决定）
    use_memory: bool          # 是否注入长期记忆（仅模仿模式需要）
    qq_controls: bool         # 是否启用引用/表情包等 QQ 控制标记
    image_tool: bool          # 是否允许 LLM 调用生图工具
    max_reply_chars: int | None = None    # None 表示使用全局值；0 表示不切分
    max_reply_actions: int | None = None  # None 表示使用全局值；0 表示不限条数
    max_bubble_chars: int | None = None   # None 表示使用全局值；<=0 表示不合并（每行一条气泡）


_DEFAULT_CLONE_PROMPT = "你是一个普通的 QQ 用户，说话简短随意。"
_DEFAULT_TAVERN_PROMPT = (
    "你正在通过 QQ 与对方进行沉浸式角色扮演，使用第一人称“我”，"
    "保持角色人设，不要跳出角色，也不要声明自己是 AI。"
)
_DEFAULT_WRITER_PROMPT = (
    "你是一位小说作家，根据对方的设定与灵感创作连贯、有画面感的小说段落。"
)


def _read_prompt(path: Path, fallback: str) -> str:
    try:
        if path.exists():
            return path.read_text(encoding="utf-8").strip() or fallback
    except Exception:
        pass
    return fallback


def build_modes(settings, bot_root: Path) -> dict[str, Mode]:
    """根据配置构造三种模式。    settings 需提供 llm_model / llm_vision_model /
    mode_tavern_model / mode_clone_model / mode_writer_model / llm_temperature /
    llm_num_predict / clone_prompt_path。
    """
    llm_model = settings.llm_model
    vision_model = settings.llm_vision_model
    tavern_prompt = _read_prompt(bot_root / "data/modes/tavern.txt", _DEFAULT_TAVERN_PROMPT)
    writer_prompt = _read_prompt(bot_root / "data/modes/writer.txt", _DEFAULT_WRITER_PROMPT)
    clone_prompt = _read_prompt(settings.mode_clone_prompt_path, _DEFAULT_CLONE_PROMPT)

    return {
        "tavern": Mode(
            key="tavern",
            label="酒馆角色扮演",
            model=settings.mode_tavern_model,
            vision_model=settings.mode_tavern_model,
            vision=False,
            prompt=tavern_prompt,
            temperature=0.9,
            num_predict=0,
            use_memory=True,
            qq_controls=False,
            image_tool=True,
        ),
        "clone": Mode(
            key="clone",
            label="人类模仿",
            model=settings.mode_clone_model,
            vision_model=vision_model,
            vision=True,
            prompt=clone_prompt,
            temperature=settings.llm_temperature,
            num_predict=0,
            use_memory=True,
            qq_controls=True,
            image_tool=True,
        ),
        "writer": Mode(
            key="writer",
            label="小说作家",
            model=settings.mode_writer_model,
            vision_model=vision_model,
            vision=False,
            prompt=writer_prompt,
            temperature=0.8,
            num_predict=0,
            use_memory=True,
            qq_controls=False,
            image_tool=True,
            # 小说作家是长文模式：放宽气泡上限，避免整章输出在末尾 [生图:] 之前被截断。
            max_reply_actions=60,
        ),
    }


class ModeManager:
    """按会话维度记录当前模式，并持久化到 JSON。"""""

    def __init__(
        self,
        modes: dict[str, Mode],
        state_path: Path,
        default: str = "clone",
    ) -> None:
        self._modes = modes
        self._state_path = state_path
        self._default = default if default in modes else next(iter(modes))
        self._state: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if self._state_path.exists():
            try:
                self._state = json.loads(
                    self._state_path.read_text(encoding="utf-8")
                )
            except Exception:
                self._state = {}

    def _save(self) -> None:
        try:
            atomic_write_json(self._state_path, self._state)
        except Exception as exc:  # noqa: BLE001
            logger.warning("保存模式状态失败：%s", exc)

    def get(self, key_pair: tuple[str, str]) -> Mode:
        mode_key = self._state.get(":".join(key_pair), self._default)
        return self._modes.get(mode_key, self._modes[self._default])

    def set(self, key_pair: tuple[str, str], mode_key: str) -> Mode | None:
        if mode_key not in self._modes:
            return None
        self._state[":".join(key_pair)] = mode_key
        self._save()
        return self._modes[mode_key]

    def current_key(self, key_pair: tuple[str, str]) -> str:
        return self._state.get(":".join(key_pair), self._default)

    def available(self) -> list[Mode]:
        return list(self._modes.values())
