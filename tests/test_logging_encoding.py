from __future__ import annotations

import io
import os
import sys

from logging_setup import configure_process_logging_encoding


def test_configure_process_logging_encoding_uses_utf8(monkeypatch):
    stdout = io.TextIOWrapper(io.BytesIO(), encoding="gb18030")
    stderr = io.TextIOWrapper(io.BytesIO(), encoding="gb18030")
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)
    monkeypatch.delenv("PYTHONIOENCODING", raising=False)

    configure_process_logging_encoding()

    assert stdout.encoding.lower().replace("-", "") == "utf8"
    assert stderr.encoding.lower().replace("-", "") == "utf8"
    assert stdout.errors == "backslashreplace"
    assert stderr.errors == "backslashreplace"
    assert os.environ["PYTHONIOENCODING"] == "utf-8"
