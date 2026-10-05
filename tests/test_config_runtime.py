from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from config import build_app_config


def test_environment_overrides_driver_config(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_API_BASE", "http://localhost:11434/v1")
    monkeypatch.setenv("LLM_LOG_FULL_IO", "true")
    monkeypatch.setenv("MODE_TAVERN_MODEL", "tavern-prod")
    monkeypatch.setenv("MODE_CLONE_MODEL", "clone-prod")
    monkeypatch.setenv("MODE_WRITER_MODEL", "writer-prod")
    monkeypatch.setenv("MAX_REPLY_CHARS", "15000")
    monkeypatch.setenv("MEMORY_DB", "data/memory.db")
    monkeypatch.setenv("MEMORY_AUTO_EXTRACT", "true")
    monkeypatch.setenv("MEMORY_CLONE_AUTO_EXTRACT", "false")
    monkeypatch.setenv("MEMORY_TAVERN_AUTO_EXTRACT", "true")
    monkeypatch.setenv("MEMORY_WRITER_AUTO_EXTRACT", "true")
    monkeypatch.setenv("MEMORY_TAVERN_MIN_MESSAGES", "8")
    monkeypatch.setenv("MEMORY_TAVERN_MIN_PARTNER_MESSAGES", "4")
    monkeypatch.setenv("COMFYUI_WORKFLOW", "data/t2i.json")
    monkeypatch.setenv("WEB_ENABLED", "true")
    monkeypatch.setenv("WEB_PATH", "/assistant")
    monkeypatch.setenv("WEB_TOKEN", "test-secret")
    monkeypatch.setenv("FEISHU_ENABLED", "true")
    monkeypatch.setenv("FEISHU_APP_ID", "cli_test")

    config = build_app_config(SimpleNamespace(), Path(tmp_path))

    assert config.llm.api_base == "http://localhost:11434/v1"
    assert config.llm.log_full_io is True
    assert config.modes.mode_tavern_model == "tavern-prod"
    assert config.modes.mode_clone_model == "clone-prod"
    assert config.modes.mode_writer_model == "writer-prod"
    assert config.conversation.max_reply_chars == 15000
    assert config.memory.database == Path(tmp_path) / "data" / "memory_{mode}.db"
    extraction = config.memory.online_extraction
    assert extraction.enabled is True
    assert extraction.enabled_for("clone") is False
    assert extraction.enabled_for("tavern") is True
    assert extraction.enabled_for("writer") is True
    assert extraction.thresholds_for("tavern") == (8, 4)
    assert config.image.comfyui.workflow_path == Path(tmp_path) / "data" / "t2i.json"
    assert config.platforms.web.enabled is True
    assert config.platforms.web.path == "/assistant"
    assert config.platforms.web.token == "test-secret"
    assert config.platforms.feishu.enabled is True
    assert config.platforms.feishu.app_id == "cli_test"
