"""Text-only LLM backend for the moment picker: any OpenAI-compatible server.

Ollama, LM Studio, vLLM, llama.cpp's server, LocalAI and OpenRouter all speak
``POST {base}/chat/completions``. When ``LLM_BASE_URL`` is set (or
``LLM_PROVIDER=openai``), the transcript scoring and detail passes in
``main.get_viral_clips`` go here instead of Gemini, so a self-hosted install
can run the whole pipeline without a Google key.

The Colab notebook uses this path to run **LiquidAI LFM3-1.7B** through
llama.cpp (``LLM_MODEL=lfm3-1.7b``). LFM3 is a reasoning model: its chat
template may wrap the answer in  thinking...</think>, so this module strips those
blocks and also accepts servers that return the answer in
``reasoning_content``. Optional ``LLM_TEMPERATURE``, ``LLM_TOP_K``,
``LLM_REPETITION_PENALTY`` and ``LLM_MAX_TOKENS`` tune the request; they are
only sent when set, so generic OpenAI-compatible servers keep working.
``LLM_DISABLE_THINKING=1`` sends ``chat_template_kwargs`` with
``enable_thinking: false`` (the LFM3 template's own switch), and
``LLM_CHAT_TEMPLATE_KWARGS`` can pass any other JSON object to the template.

What stays on Gemini, because it needs a model that can look at frames or a
video file: the layout picker (``layout_picker.py``), the on-screen content
detector (``screencast_layout.py``) and the silent-video path
(``main.get_visual_clips``). Without a Gemini key those degrade the way they
already did: the first two return "none", the third fails the job with a clear
message. Nothing in this module is imported by them.

Structured output: the prompts already spell out the exact JSON shape, so a
plain ``json_object`` mode is enough for most models. The request first asks
for ``json_schema`` (Ollama, vLLM, llama.cpp and LM Studio enforce it); a
server that rejects that field gets the same request again with
``json_object``, then with no ``response_format`` at all. Whatever comes back
is validated with the same pydantic model Gemini's ``response_schema`` uses,
so ``main.py`` sees one shape regardless of provider.
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional, Tuple, Type

import httpx
from pydantic import BaseModel

DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_TIMEOUT = 600.0  # local models on CPU are slow; a scoring batch can take minutes
DEFAULT_TEMPERATURE = 0.2

# LFM3 and other reasoning models put their chain of thought in  thinking...</think>
# blocks (or in a separate reasoning_content field). None of it belongs in the
# JSON the pipeline validates, and a 1.7B model can spend most of its output
# budget thinking unless the template is told not to.
_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.IGNORECASE | re.DOTALL)
_THINK_OPEN_RE = re.compile(r"<think\b[^>]*>", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</think\s*>", re.IGNORECASE)
_CHANNEL_THOUGHT_RE = re.compile(
    r"<\|channel>thought.*?<channel\|>", re.IGNORECASE | re.DOTALL)

# Per-job engine selection exposed by the dashboard. "auto" keeps whatever the
# server's LLM_* environment says; the other two force one provider for a job.
ENGINES = ("auto", "gemini", "local")


def normalize_engine(value: Optional[str]) -> str:
    """Validate a dashboard ``analysis_engine`` value.

    Raises ``ValueError`` with a user-facing message so the API can answer 400
    instead of silently running a different provider than the one requested.
    """
    if value is not None and not isinstance(value, str):
        raise ValueError("analysis_engine must be 'auto', 'gemini' or 'local'")
    engine = (value or "auto").strip().lower()
    if engine in ("", "default"):
        engine = "auto"
    if engine not in ENGINES:
        raise ValueError("analysis_engine must be 'auto', 'gemini' or 'local'")
    return engine


def apply_engine(env: dict, engine: Optional[str]) -> dict:
    """Apply a per-job engine override to a pipeline subprocess environment.

    ``auto`` leaves the environment untouched. ``local`` forces the
    OpenAI-compatible backend (the base URL/model already come from the server
    environment). ``gemini`` forces Gemini and blanks the local endpoint so a
    stale ``LLM_BASE_URL`` cannot win if a future code path forgets to check
    ``LLM_PROVIDER``.
    """
    engine = normalize_engine(engine)
    if engine == "local":
        env["LLM_PROVIDER"] = "openai"
    elif engine == "gemini":
        env["LLM_PROVIDER"] = "gemini"
        env["LLM_BASE_URL"] = ""
        env["LLM_MODEL"] = ""
        env.pop("LLM_API_KEY", None)
    return env


def provider() -> str:
    """``"openai"`` when a compatible endpoint is configured, else ``"gemini"``."""
    explicit = (os.environ.get("LLM_PROVIDER") or "").strip().lower()
    if explicit in ("openai", "ollama", "local", "openai-compatible"):
        return "openai"
    if explicit == "gemini":
        return "gemini"
    return "openai" if base_url() else "gemini"


def base_url() -> str:
    return (os.environ.get("LLM_BASE_URL") or "").strip().rstrip("/")


def model_name() -> str:
    return (os.environ.get("LLM_MODEL") or "").strip() or DEFAULT_MODEL


def active() -> bool:
    """True when the moment picker should call the OpenAI-compatible server."""
    return provider() == "openai" and bool(base_url())


def describe() -> Optional[dict]:
    """What ``/api/config`` tells the dashboard, or ``None`` when inactive."""
    if not active():
        return None
    return {"provider": "openai", "model": model_name(), "baseUrl": base_url()}


def _timeout() -> float:
    try:
        return float(os.environ.get("LLM_TIMEOUT") or DEFAULT_TIMEOUT)
    except ValueError:
        return DEFAULT_TIMEOUT


def _headers() -> dict:
    # Ollama ignores the key but the OpenAI client convention (and vLLM with
    # --api-key) wants the header present; "ollama" is the documented placeholder.
    key = (os.environ.get("LLM_API_KEY") or "ollama").strip()
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _client(**kwargs) -> httpx.Client:
    """Factory so tests can swap in ``httpx.MockTransport``."""
    # A proxy configured for the container must not swallow calls to a model
    # server on the same machine (the Colab notebook runs llama.cpp on
    # 127.0.0.1); httpx would otherwise honour HTTP_PROXY for it.
    if base_url().startswith(("http://127.0.0.1", "http://localhost")):
        kwargs.setdefault("trust_env", False)
    return httpx.Client(timeout=_timeout(), **kwargs)


def _response_formats(schema: Type[BaseModel]):
    name = getattr(schema, "__name__", "response").lower()
    # OpenAI-style structured output. llama.cpp, vLLM, LM Studio and Ollama
    # all read the schema from the nested "json_schema" object. Older servers
    # only know "json_object", which is the next rung; a 1.7B model needs the
    # grammar constraint more than a hosted model does.
    yield {"type": "json_schema",
           "json_schema": {"name": name, "schema": schema.model_json_schema()}}
    yield {"type": "json_object"}
    yield None


def _is_format_rejection(resp: httpx.Response) -> bool:
    if resp.status_code not in (400, 422):
        return False
    body = resp.text.lower()
    return ("response_format" in body or "json_schema" in body
            or "json_object" in body or "schema" in body)


def _env_float(name: str, default: Optional[float] = None) -> Optional[float]:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: Optional[int] = None) -> Optional[int]:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _truthy(value: Optional[str]) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


def _chat_template_kwargs() -> Optional[dict]:
    raw = (os.environ.get("LLM_CHAT_TEMPLATE_KWARGS") or "").strip()
    if raw:
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = None
        if isinstance(value, dict):
            return value
        # Invalid or non-object JSON: fall through to the simple flag instead
        # of silently leaving a reasoning model in thinking mode.
    if _truthy(os.environ.get("LLM_DISABLE_THINKING")):
        # LFM3's chat template reads this variable; llama.cpp forwards it
        # verbatim. Other servers only see it when this flag is set.
        return {"enable_thinking": False}
    return None


def _generation_options() -> dict:
    """Sampling knobs from the environment.

    Defaults match the old hard-coded temperature. The rest are opt-in because
    ``top_k`` and ``repeat_penalty`` are llama.cpp extensions that a strict
    OpenAI-compatible server may reject. ``LLM_REPETITION_PENALTY`` is sent as
    ``repeat_penalty`` because that is the field llama.cpp's OpenAI endpoint
    reads; a ``repetition_penalty`` field is ignored there.
    """
    options = {"temperature": _env_float("LLM_TEMPERATURE", DEFAULT_TEMPERATURE)}
    top_k = _env_int("LLM_TOP_K")
    if top_k is not None and top_k > 0:
        options["top_k"] = top_k
    penalty = _env_float("LLM_REPETITION_PENALTY")
    if penalty is not None and penalty > 0:
        options["repeat_penalty"] = penalty
    max_tokens = _env_int("LLM_MAX_TOKENS")
    if max_tokens is not None and max_tokens > 0:
        options["max_tokens"] = max_tokens
    effort = (os.environ.get("LLM_REASONING_EFFORT") or "").strip()
    if effort:
        options["reasoning_effort"] = effort
    template_kwargs = _chat_template_kwargs()
    if template_kwargs:
        options["chat_template_kwargs"] = template_kwargs
    return options


def _strip_reasoning(text: str) -> str:
    """Remove chain-of-thought markup so only the answer remains."""
    if not text:
        return ""
    text = _THINK_BLOCK_RE.sub("", text)
    text = _CHANNEL_THOUGHT_RE.sub("", text)
    # A server that streams only the closing marker leaves the thoughts before
    # it; the answer is everything after the last one.
    if _THINK_CLOSE_RE.search(text):
        text = _THINK_CLOSE_RE.split(text)[-1]
    # A truncated response can leave the opening tag without a close.
    text = _THINK_OPEN_RE.sub("", text)
    text = text.replace("<|channel>thought", "").replace("<channel|>", "")
    return text.strip()


def _message_text(value) -> str:
    if isinstance(value, list):
        return "".join(
            str(part.get("text") or "") if isinstance(part, dict) else str(part)
            for part in value
        )
    return str(value or "")


def _response_texts(data: dict) -> Tuple[str, str]:
    """Return ``(content, reasoning)`` from a chat-completions response."""
    choices = data.get("choices") or []
    if not choices:
        return "", ""
    choice = choices[0] or {}
    message = choice.get("message") or {}
    content = _message_text(message.get("content"))
    reasoning = _message_text(
        message.get("reasoning_content")
        or message.get("reasoning")
        or choice.get("reasoning_content")
    )
    return content, reasoning


def generate_json(prompt: str, schema: Type[BaseModel], model: Optional[str] = None,
                  ) -> Tuple[dict, Optional[dict]]:
    """One chat completion that must come back as JSON matching ``schema``.

    Returns ``(parsed_dict, cost_analysis)`` in the exact shape
    ``main._run_gemini_stage`` returns, so the caller does not branch on the
    provider. Raises on HTTP errors, empty bodies and schema violations; the
    retry policy lives in the caller, same as for Gemini.
    """
    import gemini_worker  # local import: keeps this module free of the google SDK

    url = f"{base_url()}/chat/completions"
    model = model or model_name()
    messages = [
        {"role": "system", "content": "You answer with a single JSON object and nothing else."},
        {"role": "user", "content": prompt},
    ]
    last_rejection: Optional[str] = None
    generation = _generation_options()
    with _client() as client:
        for fmt in _response_formats(schema):
            body = {"model": model, "messages": messages, "stream": False, **generation}
            if fmt is not None:
                body["response_format"] = fmt
            resp = client.post(url, json=body, headers=_headers())
            if fmt is not None and _is_format_rejection(resp):
                # The server does not know this response_format flavour; the
                # next loop iteration asks for a looser one.
                last_rejection = resp.text[:200]
                continue
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"LLM server {resp.status_code} from {url}: {resp.text[:300]}")
            data = resp.json()
            break
        else:
            raise RuntimeError(
                f"LLM server rejected every response_format variant: {last_rejection}")

    content, reasoning = _response_texts(data)
    answer = _strip_reasoning(content)
    try:
        parsed = gemini_worker._parse_json_response_text(answer)
    except ValueError:
        # Some servers (and llama.cpp with --reasoning-format deepseek) put the
        # whole answer in reasoning_content and leave content empty.
        fallback = _strip_reasoning(reasoning)
        if not fallback or fallback == answer:
            raise
        parsed = gemini_worker._parse_json_response_text(fallback)
    # Validate against the same schema Gemini enforces server-side, so a
    # local model that drops a field fails here with a readable error
    # (retried by the caller) instead of deep inside the clip pipeline.
    validated = schema.model_validate(parsed).model_dump()

    usage = data.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    cost = {
        "input_tokens": int(usage.get("prompt_tokens") or 0),
        "output_tokens": int(usage.get("completion_tokens") or 0),
        "thinking_tokens": int(details.get("reasoning_tokens")
                               or usage.get("reasoning_tokens") or 0),
        "input_cost": 0.0,
        "output_cost": 0.0,
        "total_cost": 0.0,
        "model": model,
        "price_estimated": False,
        "local": True,
    }
    return validated, cost
