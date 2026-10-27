"""The OpenAI-compatible moment-picker backend (llm_backend.py).

Self-hosters asked for a pipeline that never calls Google. The transcript
passes go to any /chat/completions server when LLM_BASE_URL is set; these
tests pin the contract main.py relies on: same (parsed, cost) shape as the
Gemini stage, schema validation, and the response_format fallback ladder.
"""
import json

import httpx
import pytest

import gemini_worker
import llm_backend


@pytest.fixture
def local(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://llm.test/v1")
    monkeypatch.setenv("LLM_MODEL", "qwen2.5:14b")
    for name in (
        "LLM_PROVIDER", "LLM_TEMPERATURE", "LLM_TOP_K",
        "LLM_REPETITION_PENALTY", "LLM_MAX_TOKENS", "LLM_REASONING_EFFORT",
        "LLM_DISABLE_THINKING", "LLM_CHAT_TEMPLATE_KWARGS",
    ):
        monkeypatch.delenv(name, raising=False)


def _serve(handler, monkeypatch):
    """Route llm_backend's httpx client through an in-process handler."""
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(llm_backend, "_client",
                        lambda **kw: httpx.Client(transport=transport, **kw))


def _completion(payload, usage=None):
    return httpx.Response(200, json={
        "choices": [{"message": {"role": "assistant", "content": json.dumps(payload)}}],
        "usage": usage or {"prompt_tokens": 120, "completion_tokens": 40},
    })


# --- activation -----------------------------------------------------------

def test_inactive_without_a_base_url(monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert llm_backend.active() is False
    assert llm_backend.describe() is None


def test_base_url_alone_activates_and_describes(local):
    assert llm_backend.active() is True
    assert llm_backend.describe() == {
        "provider": "openai", "model": "qwen2.5:14b", "baseUrl": "http://llm.test/v1"}


def test_explicit_gemini_provider_wins_over_base_url(local, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    assert llm_backend.active() is False


# --- per-job engine selection ---------------------------------------------

def test_normalize_engine_defaults_and_rejects_unknown():
    assert llm_backend.normalize_engine(None) == "auto"
    assert llm_backend.normalize_engine("") == "auto"
    assert llm_backend.normalize_engine("default") == "auto"
    assert llm_backend.normalize_engine(" LOCAL ") == "local"
    with pytest.raises(ValueError):
        llm_backend.normalize_engine("openai")
    with pytest.raises(ValueError):
        llm_backend.normalize_engine(1)


def test_apply_engine_forces_local_and_gemini():
    env = {"LLM_BASE_URL": "http://llm.test/v1", "LLM_MODEL": "lfm3",
           "LLM_API_KEY": "secret", "LLM_PROVIDER": "gemini"}
    llm_backend.apply_engine(env, "local")
    assert env["LLM_PROVIDER"] == "openai"
    assert env["LLM_BASE_URL"] == "http://llm.test/v1"
    assert env["LLM_MODEL"] == "lfm3"

    llm_backend.apply_engine(env, "gemini")
    assert env["LLM_PROVIDER"] == "gemini"
    assert env["LLM_BASE_URL"] == ""
    assert env["LLM_MODEL"] == ""
    assert "LLM_API_KEY" not in env


def test_apply_engine_auto_is_a_noop():
    env = {"LLM_PROVIDER": "openai", "LLM_BASE_URL": "http://llm.test/v1"}
    assert llm_backend.apply_engine(env, "auto") == {
        "LLM_PROVIDER": "openai", "LLM_BASE_URL": "http://llm.test/v1"}


def test_local_client_ignores_proxy_env(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:8089/v1")
    with llm_backend._client() as client:
        assert client.trust_env is False
    monkeypatch.setenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    with llm_backend._client() as client:
        assert client.trust_env is True


# --- generate_json ---------------------------------------------------------

def test_returns_validated_payload_and_zero_cost(local, monkeypatch):
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers.get("authorization")
        return _completion({"windows": [{"id": "w0", "start": 0, "end": 90, "score": 88, "reason": "hook"}]})

    _serve(handler, monkeypatch)
    parsed, cost = llm_backend.generate_json("prompt", gemini_worker.ScoreResponse)

    assert seen["url"] == "http://llm.test/v1/chat/completions"
    assert seen["body"]["model"] == "qwen2.5:14b"
    assert seen["body"]["response_format"]["type"] == "json_schema"
    assert seen["auth"] == "Bearer ollama"  # documented placeholder when no key is set
    assert parsed["windows"][0]["score"] == 88
    assert cost["total_cost"] == 0.0 and cost["local"] is True
    assert cost["input_tokens"] == 120 and cost["output_tokens"] == 40


def test_falls_back_to_json_object_when_schema_mode_is_rejected(local, monkeypatch):
    formats = []

    def handler(request):
        body = json.loads(request.content)
        fmt = (body.get("response_format") or {}).get("type")
        formats.append(fmt)
        if fmt == "json_schema":
            return httpx.Response(400, json={"error": {"message": "response_format json_schema not supported"}})
        return _completion({"windows": []})

    _serve(handler, monkeypatch)
    parsed, _ = llm_backend.generate_json("prompt", gemini_worker.ScoreResponse)
    assert formats == ["json_schema", "json_object"]
    assert parsed == {"windows": []}


def test_unrelated_400_is_not_treated_as_a_format_rejection(local, monkeypatch):
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(400, json={"error": {
            "message": "invalid format for field 'temperature'"}})

    _serve(handler, monkeypatch)
    with pytest.raises(RuntimeError, match="temperature"):
        llm_backend.generate_json("prompt", gemini_worker.ScoreResponse)
    assert len(calls) == 1


def test_code_fenced_json_is_accepted(local, monkeypatch):
    """Small models wrap the object in ```json fences even when told not to."""
    def handler(request):
        text = "```json\n" + json.dumps({"windows": []}) + "\n```"
        return httpx.Response(200, json={"choices": [{"message": {"content": text}}], "usage": {}})

    _serve(handler, monkeypatch)
    parsed, _ = llm_backend.generate_json("prompt", gemini_worker.ScoreResponse)
    assert parsed == {"windows": []}


def test_complete_think_block_is_stripped_before_parsing(local, monkeypatch):
    answer = json.dumps({"windows": []})

    def handler(request):
        text = "thinking\nI should score the hook first.\n</think>\n" + answer
        return httpx.Response(200, json={"choices": [{"message": {"content": text}}], "usage": {}})

    _serve(handler, monkeypatch)
    parsed, _ = llm_backend.generate_json("prompt", gemini_worker.ScoreResponse)
    assert parsed == {"windows": []}


def test_stray_closing_think_marker_keeps_only_the_answer(local, monkeypatch):
    answer = json.dumps({"windows": []})

    def handler(request):
        text = "thoughts with {fake json}\n</think>\n" + answer
        return httpx.Response(200, json={"choices": [{"message": {"content": text}}], "usage": {}})

    _serve(handler, monkeypatch)
    parsed, _ = llm_backend.generate_json("prompt", gemini_worker.ScoreResponse)
    assert parsed == {"windows": []}


def test_reasoning_content_is_used_when_content_is_empty(local, monkeypatch):
    answer = json.dumps({"windows": []})

    def handler(request):
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "", "reasoning_content": answer}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                      "completion_tokens_details": {"reasoning_tokens": 3}},
        })

    _serve(handler, monkeypatch)
    parsed, cost = llm_backend.generate_json("prompt", gemini_worker.ScoreResponse)
    assert parsed == {"windows": []}
    assert cost["thinking_tokens"] == 3


def test_generation_options_from_env(local, monkeypatch):
    monkeypatch.setenv("LLM_TEMPERATURE", "0.1")
    monkeypatch.setenv("LLM_TOP_K", "50")
    monkeypatch.setenv("LLM_REPETITION_PENALTY", "1.1")
    monkeypatch.setenv("LLM_MAX_TOKENS", "4096")
    monkeypatch.setenv("LLM_REASONING_EFFORT", "low")
    monkeypatch.setenv("LLM_DISABLE_THINKING", "1")
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return _completion({"windows": []})

    _serve(handler, monkeypatch)
    llm_backend.generate_json("prompt", gemini_worker.ScoreResponse)
    body = seen["body"]
    assert body["temperature"] == 0.1
    assert body["top_k"] == 50
    # llama.cpp's OpenAI endpoint reads repeat_penalty, not repetition_penalty.
    assert body["repeat_penalty"] == 1.1
    assert "repetition_penalty" not in body
    assert body["max_tokens"] == 4096
    assert body["reasoning_effort"] == "low"
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_invalid_generation_options_fall_back_to_defaults(local, monkeypatch):
    monkeypatch.setenv("LLM_TEMPERATURE", "hot")
    monkeypatch.setenv("LLM_TOP_K", "-1")
    monkeypatch.setenv("LLM_MAX_TOKENS", "zero")
    monkeypatch.setenv("LLM_CHAT_TEMPLATE_KWARGS", "not json")
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return _completion({"windows": []})

    _serve(handler, monkeypatch)
    llm_backend.generate_json("prompt", gemini_worker.ScoreResponse)
    body = seen["body"]
    assert body["temperature"] == llm_backend.DEFAULT_TEMPERATURE
    assert "top_k" not in body
    assert "max_tokens" not in body
    assert "chat_template_kwargs" not in body


def test_invalid_template_kwargs_still_honor_disable_thinking(local, monkeypatch):
    monkeypatch.setenv("LLM_CHAT_TEMPLATE_KWARGS", "not json")
    monkeypatch.setenv("LLM_DISABLE_THINKING", "1")
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return _completion({"windows": []})

    _serve(handler, monkeypatch)
    llm_backend.generate_json("prompt", gemini_worker.ScoreResponse)
    assert seen["body"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_schema_violation_raises_instead_of_leaking_into_the_pipeline(local, monkeypatch):
    _serve(lambda r: _completion({"windows": [{"id": "w0"}]}), monkeypatch)
    with pytest.raises(Exception) as exc:
        llm_backend.generate_json("prompt", gemini_worker.ScoreResponse)
    assert "validation error" in str(exc.value)


def test_server_errors_surface_with_status_and_url(local, monkeypatch):
    _serve(lambda r: httpx.Response(503, text="loading model"), monkeypatch)
    with pytest.raises(RuntimeError) as exc:
        llm_backend.generate_json("prompt", gemini_worker.ScoreResponse)
    assert "503" in str(exc.value) and "loading model" in str(exc.value)


# --- main.py routing (needs the heavy deps; skipped on minimal CI) ------------

def test_stage_routes_to_local_backend_and_retries_transient(local, monkeypatch):
    main = pytest.importorskip("main")
    monkeypatch.setattr(main.time, "sleep", lambda *_: None)
    calls = {"n": 0}

    def fake_generate(prompt, schema, model=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("ConnectError: connection refused")
        return {"windows": [{"id": "w0", "start": 0, "end": 1, "score": 50, "reason": ""}]}, {"total_cost": 0.0}

    monkeypatch.setattr(llm_backend, "generate_json", fake_generate)
    parsed, cost = main._run_gemini_stage(None, "qwen2.5:14b", "prompt", gemini_worker.ScoreResponse)
    assert calls["n"] == 2
    assert parsed["windows"][0]["score"] == 50
    assert cost["total_cost"] == 0.0


def test_score_batch_shrinks_for_local_models(local, monkeypatch):
    main = pytest.importorskip("main")
    monkeypatch.delenv("LLM_SCORE_BATCH", raising=False)
    assert main.score_batch_size() == 3
    monkeypatch.setenv("LLM_SCORE_BATCH", "5")
    assert main.score_batch_size() == 5
    monkeypatch.delenv("LLM_BASE_URL")
    monkeypatch.delenv("LLM_SCORE_BATCH")
    assert main.score_batch_size() == 8
