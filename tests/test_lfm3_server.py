"""Unit tests for the standalone LFM3 OpenAI-compatible server.

The module only imports the standard library at import time; torch and
transformers are loaded lazily in ``_import_runtime``/``main``. That keeps
these tests runnable in the app's normal test environment.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "lfm3_server", ROOT / "colab" / "lfm3_server.py")
assert SPEC and SPEC.loader
lfm3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lfm3)


def test_normalize_messages_flattens_text_parts():
    messages = lfm3._normalize_messages([
        {"role": "system", "content": "rules"},
        {"role": "user", "content": [
            {"type": "text", "text": "first"},
            {"type": "image", "image_url": "ignored"},
            {"type": "text", "text": "second"},
        ]},
    ])
    assert messages == [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "first\nsecond"},
    ]


def test_normalize_messages_rejects_empty_and_bad_roles():
    for value in (None, [], [{"role": "wizard", "content": "x"}]):
        try:
            lfm3._normalize_messages(value)
        except lfm3.BadRequest:
            pass
        else:  # pragma: no cover - assertion path
            raise AssertionError(f"expected BadRequest for {value!r}")


def test_stop_list_accepts_string_and_array():
    assert lfm3._stop_list(None) == []
    assert lfm3._stop_list("</s>") == ["</s>"]
    assert lfm3._stop_list(["</s>", "", 7]) == ["</s>", "7"]


def test_apply_stop_truncates_at_first_match():
    assert lfm3._apply_stop("hello STOP world", ["STOP"]) == ("hello", True)
    assert lfm3._apply_stop("hello world", ["STOP"]) == ("hello world", False)


def test_clamp_max_new_tokens_leaves_prompt_room():
    assert lfm3._clamp_max_new_tokens(16384, 4096) == 4096
    assert lfm3._clamp_max_new_tokens(4096, 4096) == 3072
    assert lfm3._clamp_max_new_tokens(2048, 4096) == 1024
    assert lfm3._clamp_max_new_tokens(4096, 0) == 1


def test_chat_response_shape():
    result = {
        "text": '{"ok": true}',
        "prompt_tokens": 12,
        "completion_tokens": 4,
        "finish_reason": "stop",
    }
    payload = lfm3._chat_response(result, "lfm3-1.7b")
    assert payload["object"] == "chat.completion"
    assert payload["model"] == "lfm3-1.7b"
    assert payload["choices"][0]["message"] == {
        "role": "assistant", "content": '{"ok": true}'}
    assert payload["choices"][0]["finish_reason"] == "stop"
    assert payload["usage"] == {
        "prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16}
