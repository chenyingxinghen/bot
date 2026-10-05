"""小型 JSON 状态文件的原子持久化辅助。"""
from __future__ import annotations

import json
import os
from pathlib import Path


def atomic_write_json(path: str | Path, data: object) -> None:
    """先写同目录临时文件并 fsync，再用 os.replace 原子替换目标。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, target)
