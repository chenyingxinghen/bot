"""三种模式使用的模型和状态参数。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config.base import resolve_bot_path


@dataclass(frozen=True)
class ModesSettings:
    mode_tavern_model: str
    mode_clone_model: str
    mode_writer_model: str
    modes_state_path: Path
    llm_model: str
    llm_vision_model: str
    llm_temperature: float
    llm_num_predict: int
    mode_clone_prompt_path: Path

    @classmethod
    def from_config(cls, config: object, bot_root: Path) -> "ModesSettings":
        llm_model = str(getattr(
            config,
            "llm_model",
            "Qwen/Qwen3-235B-A22B-Instruct",
        ))
        llm_vision_model = str(getattr(config, "llm_vision_model", llm_model))

        return cls(
            mode_tavern_model=str(getattr(
                config, "mode_tavern_model",
                "gemma-4-e4b-uncensored-hauhaucs-aggressive",
            )),
            mode_clone_model=str(getattr(config, "mode_clone_model", llm_vision_model)),
            mode_writer_model=str(getattr(config, "mode_writer_model", llm_model)),
            modes_state_path=resolve_bot_path(
                bot_root,
                str(getattr(config, "modes_state_path", "data/modes_state.json")),
            ),
            llm_model=llm_model,
            llm_vision_model=llm_vision_model,
            llm_temperature=float(getattr(config, "llm_temperature", 0.6)),
            llm_num_predict=int(getattr(config, "llm_num_predict", 192)),
            mode_clone_prompt_path=resolve_bot_path(
                bot_root,
                str(getattr(config, "clone_prompt_path", "data/modes/clone.txt")),
            ),
        )
