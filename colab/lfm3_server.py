#!/usr/bin/env python3
"""Small OpenAI-compatible server for LiquidAI/LFM3-1.7B.

The OpenShorts moment picker only needs two endpoints:

    GET  /v1/models
    POST /v1/chat/completions

This file loads the checkpoint directly with ``transformers`` (no llama.cpp,
no GGUF, no vLLM) and serves those endpoints with the Python standard library.
It is deliberately tiny so the Colab notebook can run it inside an isolated
virtualenv with ``transformers==5.21.0`` while the main app keeps its own
(older) transformers for SDXL.

Run it manually:

    pip install "transformers==5.21.0" accelerate
    python colab/lfm3_server.py --model LiquidAI/LFM3-1.7B \
        --served-model-name lfm3-1.7b --port 8089

The notebook pre-downloads the checkpoint and passes a local directory:

    python colab/lfm3_server.py --model /content/openshorts-models/LFM3-1.7B \
        --download-only --local-dir /content/openshorts-models/LFM3-1.7B
    python colab/lfm3_server.py --model /content/openshorts-models/LFM3-1.7B

Only text is supported. LFM3's chat template is applied with
``enable_thinking=False`` by default so the model spends its output budget on
the JSON answer instead of a chain of thought.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional
from urllib.parse import urlsplit

DEFAULT_MODEL = "LiquidAI/LFM3-1.7B"
DEFAULT_SERVED_NAME = "lfm3-1.7b"
DEFAULT_MAX_CONTEXT = 16384
DEFAULT_MAX_NEW_TOKENS = 4096

# Populated by main().
MODEL = None
TOKENIZER = None
TORCH = None
MODEL_PATH = ""
SERVED_NAME = DEFAULT_SERVED_NAME
MAX_CONTEXT = DEFAULT_MAX_CONTEXT
MAX_NEW_TOKENS = DEFAULT_MAX_NEW_TOKENS
MODEL_DEVICE = "cpu"
MODEL_DTYPE = "float32"
TRITON_ACTIVE = False
API_KEY: Optional[str] = None
STARTED_AT = int(time.time())
GENERATE_LOCK = threading.Lock()


class BadRequest(Exception):
    """A request the server can explain to the client as HTTP 400."""


def log(message: str) -> None:
    print(f"[lfm3] {message}", flush=True)


def _import_runtime():
    """Import torch/transformers with a readable error outside the venv."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - depends on the runtime
        sys.exit(
            "LFM3 server needs torch and transformers. Install them with:\n"
            '  pip install "transformers==5.21.0" accelerate\n'
            f"Import error: {exc}"
        )
    return torch, transformers, AutoModelForCausalLM, AutoTokenizer


def _download_model(model_id: str, local_dir: Optional[str]) -> str:
    """Download only the files needed for inference and return the path."""
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # pragma: no cover - depends on the runtime
        sys.exit(f"huggingface_hub is required for --download-only: {exc}")

    if os.path.isdir(model_id) and not local_dir:
        log(f"model already exists at {model_id}")
        return os.path.abspath(model_id)
    target = os.path.abspath(os.path.expanduser(local_dir or model_id))
    log(f"downloading {model_id} -> {target}")
    snapshot_download(
        repo_id=model_id,
        local_dir=target,
        token=(os.environ.get("HF_TOKEN") or None),
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.py",
            "*.jinja",
            "tokenizer*",
            "special_tokens_map.json",
        ],
    )
    weights = []
    for name in os.listdir(target):
        if name.endswith(".safetensors"):
            weights.append(os.path.join(target, name))
    total = sum(os.path.getsize(path) for path in weights)
    if total < 1_000_000_000:
        raise RuntimeError(
            f"model files look incomplete in {target} "
            f"(safetensors total {total / 1e9:.2f} GB)"
        )
    log(f"model ready at {target} ({total / 1e9:.2f} GB of weights)")
    return target


def _clamp_max_new_tokens(context: int, requested: int) -> int:
    """Leave at least 1024 tokens for the prompt (system rules + transcript)."""
    return max(1, min(int(requested), int(context) - 1024))


def _resolve_device(torch, requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is false")
    if requested not in ("cpu", "cuda"):
        raise RuntimeError(f"unknown device {requested!r}")
    return requested


def _resolve_dtype(torch, requested: str, device: str):
    if requested == "auto":
        if device == "cuda":
            major, _minor = torch.cuda.get_device_capability(0)
            # T4/P100 are pre-Ampere: fp16 is the fast path there, bf16 is not
            # supported by the hardware. Ampere+ gets bf16 for stability.
            return torch.bfloat16 if major >= 8 else torch.float16
        # CPU fp32 is the most compatible fallback. bf16/fp16 CPU kernels are
        # slower and not available on every Colab host.
        return torch.float32
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if requested not in mapping:
        raise RuntimeError(f"unknown dtype {requested!r}")
    return mapping[requested]


def _load_model(torch, transformers, auto_model, auto_tokenizer, model_path: str,
                device: str, dtype):
    log(f"loading tokenizer from {model_path}")
    tokenizer = auto_tokenizer.from_pretrained(
        model_path, trust_remote_code=True,
        token=(os.environ.get("HF_TOKEN") or None),
    )

    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "dtype": dtype,
        "token": (os.environ.get("HF_TOKEN") or None),
    }
    if device == "cuda":
        kwargs["device_map"] = "auto"
        # SDPA is supported by the remote model and is the fastest safe
        # attention backend on T4/Ampere without flash-attn installed.
        kwargs["attn_implementation"] = "sdpa"
    else:
        kwargs["low_cpu_mem_usage"] = True

    log(f"loading weights ({device}, {str(dtype).replace('torch.', '')})")
    model = auto_model.from_pretrained(model_path, **kwargs)
    if device == "cpu":
        model = model.to("cpu")
    model.eval()
    log(f"transformers {transformers.__version__} | torch {torch.__version__}")
    return model, tokenizer


def _model_device(torch, model):
    try:
        return model.device
    except Exception:
        pass
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _set_triton(model, enabled: bool) -> bool:
    """Enable/disable LFM3's optional Triton conv kernels.

    The remote model imports ``can_use_lfm3_conv`` into its own module. Patching
    that module's global is enough to force the documented PyTorch fallback.
    """
    module = sys.modules.get(type(model).__module__)
    if module is None or not hasattr(module, "can_use_lfm3_conv"):
        return False
    if enabled:
        return True
    module.can_use_lfm3_conv = lambda *_args, **_kwargs: False
    return False


def _normalize_content(content: Any) -> Any:
    """Keep text parts from OpenAI-style content arrays."""
    if isinstance(content, str) or content is None:
        return content or ""
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") in (None, "text"):
                parts.append(str(part.get("text") or ""))
        return "\n".join(parts)
    return str(content)


def _normalize_messages(messages: Any) -> list[dict]:
    if not isinstance(messages, list) or not messages:
        raise BadRequest("messages must be a non-empty array")
    normalized = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise BadRequest(f"messages[{index}] must be an object")
        role = str(message.get("role") or "").strip()
        if role not in ("system", "user", "assistant", "tool"):
            raise BadRequest(f"messages[{index}].role is not supported: {role!r}")
        item = {"role": role, "content": _normalize_content(message.get("content"))}
        if message.get("name"):
            item["name"] = str(message["name"])
        if message.get("tool_calls") is not None:
            item["tool_calls"] = message["tool_calls"]
        if message.get("tool_call_id") is not None:
            item["tool_call_id"] = message["tool_call_id"]
        normalized.append(item)
    return normalized


def _apply_template(messages: list[dict], template_kwargs: dict,
                    tools: Any) -> str:
    kwargs = {"enable_thinking": False}
    kwargs.update(template_kwargs or {})
    call_kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
        **kwargs,
    }
    if tools:
        call_kwargs["tools"] = tools
    try:
        return TOKENIZER.apply_chat_template(messages, **call_kwargs)
    except TypeError as exc:
        # A typo in LLM_CHAT_TEMPLATE_KWARGS should not make every job fail.
        if kwargs != {"enable_thinking": False}:
            log(f"chat template rejected kwargs {kwargs}: {exc}; retrying defaults")
            call_kwargs.pop("tools", None)
            call_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": False,
                **({"tools": tools} if tools else {}),
            }
            return TOKENIZER.apply_chat_template(messages, **call_kwargs)
        raise


def _truncate_input_ids(torch, input_ids, max_new_tokens: int):
    """Keep the instructions at the head and the newest transcript at the tail."""
    limit = MAX_CONTEXT - max_new_tokens
    if limit < 256:
        raise BadRequest(
            f"max_tokens={max_new_tokens} leaves no room in the "
            f"{MAX_CONTEXT}-token context"
        )
    original = int(input_ids.shape[1])
    if original <= limit:
        return input_ids, 0
    head = min(512, limit // 4)
    tail = limit - head
    truncated = torch.cat(
        [input_ids[:, :head], input_ids[:, -tail:]], dim=1)
    return truncated, original - limit


def _stop_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return []


def _apply_stop(text: str, stops: list[str]) -> tuple[str, bool]:
    cut = len(text)
    for stop in stops:
        index = text.find(stop)
        if index != -1:
            cut = min(cut, index)
    if cut < len(text):
        return text[:cut].rstrip(), True
    return text.rstrip(), False


def _generate(messages: list[dict], *, max_new_tokens: int,
              temperature: float, top_p: Optional[float],
              top_k: Optional[int], repetition_penalty: Optional[float],
              template_kwargs: dict, tools: Any, stops: list[str],
              seed: Optional[int]) -> dict:
    if TOKENIZER is None or MODEL is None:
        raise RuntimeError("model is not loaded")
    if max_new_tokens < 1:
        raise BadRequest("max_tokens must be positive")
    max_new_tokens = min(max_new_tokens, MAX_NEW_TOKENS)
    if max_new_tokens >= MAX_CONTEXT:
        raise BadRequest(
            f"max_tokens={max_new_tokens} is larger than the "
            f"{MAX_CONTEXT}-token context"
        )

    prompt = _apply_template(messages, template_kwargs, tools)
    encoded = TOKENIZER(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")
    input_ids, dropped = _truncate_input_ids(TORCH, input_ids, max_new_tokens)
    if dropped:
        log(f"prompt truncated by {dropped} tokens to fit {MAX_CONTEXT}")
    if attention_mask is not None:
        if dropped:
            # The attention mask must match the truncated ids. Rebuild it from
            # the ids because the head+tail splice can contain padding ids.
            attention_mask = TORCH.ones_like(input_ids)
        else:
            attention_mask = attention_mask[:, -input_ids.shape[1]:]

    device = _model_device(TORCH, MODEL)
    inputs = {"input_ids": input_ids.to(device)}
    if attention_mask is not None:
        inputs["attention_mask"] = attention_mask.to(device)

    do_sample = temperature > 0
    generation: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "use_cache": True,
        "pad_token_id": TOKENIZER.pad_token_id or TOKENIZER.eos_token_id,
        "eos_token_id": TOKENIZER.eos_token_id,
    }
    if do_sample:
        generation["temperature"] = max(float(temperature), 1e-5)
        if top_p is not None and 0 < top_p < 1:
            generation["top_p"] = float(top_p)
        if top_k is not None and top_k > 0:
            generation["top_k"] = int(top_k)
    if repetition_penalty is not None and repetition_penalty > 0:
        generation["repetition_penalty"] = float(repetition_penalty)
    if seed is not None:
        TORCH.manual_seed(int(seed))

    prompt_tokens = int(input_ids.shape[1])
    with TORCH.inference_mode():
        output = MODEL.generate(**inputs, **generation)
    new_tokens = output[0, prompt_tokens:]
    completion_tokens = int(new_tokens.shape[0])
    text = TOKENIZER.decode(new_tokens.cpu(), skip_special_tokens=True)
    text, stopped = _apply_stop(text, stops)

    eos_ids = TOKENIZER.eos_token_id
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    hit_eos = bool(completion_tokens and int(new_tokens[-1]) in (eos_ids or []))
    finish_reason = "stop" if (completion_tokens == 0 or hit_eos or stopped) else "length"
    return {
        "text": text.strip(),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
    }


def _chat_response(result: dict, model_name: str) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result["text"]},
                "finish_reason": result["finish_reason"],
            }
        ],
        "usage": {
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
        },
    }


def _warmup(torch) -> None:
    log("warmup generation...")
    try:
        _generate(
            [{"role": "user", "content": "Reply with the single word OK."}],
            max_new_tokens=8, temperature=0.0, top_p=None, top_k=None,
            repetition_penalty=None, template_kwargs={"enable_thinking": False},
            tools=None, stops=[], seed=None,
        )
    except Exception as exc:
        # The custom conv kernels are the most likely CUDA-specific failure.
        # Retry once without them instead of making the whole server unusable.
        if TRITON_ACTIVE:
            log(
                f"Triton warmup failed ({type(exc).__name__}: {exc}); "
                "retrying with the PyTorch fallback"
            )
            _set_triton(MODEL, False)
            globals()["TRITON_ACTIVE"] = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _generate(
                [{"role": "user", "content": "Reply with the single word OK."}],
                max_new_tokens=8, temperature=0.0, top_p=None, top_k=None,
                repetition_penalty=None,
                template_kwargs={"enable_thinking": False},
                tools=None, stops=[], seed=None,
            )
        else:
            raise
    log("warmup done")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "OpenShortsLFM3/1.0"

    def log_message(self, fmt: str, *args) -> None:
        # Keep health-check noise out of the Colab log but keep errors.
        request = str(args[0]) if args else ""
        if "GET /health" in request or "GET /v1/models" in request:
            return
        print(f"[lfm3] {self.address_string()} {fmt % args}", flush=True)

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _error(self, status: int, message: str, error_type: str = "invalid_request_error") -> None:
        self._send_json(status, {
            "error": {
                "message": message,
                "type": error_type,
                "param": None,
                "code": None,
            }
        })

    def _authorized(self) -> bool:
        if not API_KEY:
            return True
        header = self.headers.get("Authorization") or ""
        return header == f"Bearer {API_KEY}"

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path.rstrip("/") or "/"
        if path == "/health":
            self._send_json(200, {
                "status": "ok",
                "model": SERVED_NAME,
                "path": MODEL_PATH,
                "device": MODEL_DEVICE,
                "dtype": MODEL_DTYPE,
                "triton": TRITON_ACTIVE,
                "max_context": MAX_CONTEXT,
                "uptime_seconds": int(time.time()) - STARTED_AT,
            })
            return
        if path in ("/", "/v1"):
            self._send_json(200, {
                "service": "OpenShorts LFM3 server",
                "endpoints": ["/health", "/v1/models", "/v1/chat/completions"],
            })
            return
        if path == "/v1/models":
            if not self._authorized():
                self._error(401, "invalid API key", "authentication_error")
                return
            self._send_json(200, {
                "object": "list",
                "data": [
                    {
                        "id": SERVED_NAME,
                        "object": "model",
                        "created": STARTED_AT,
                        "owned_by": "local",
                    }
                ],
            })
            return
        self._error(404, f"unknown path: {path}", "not_found_error")

    def do_POST(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path.rstrip("/")
        if path != "/v1/chat/completions":
            self._error(404, f"unknown path: {path}", "not_found_error")
            return
        if not self._authorized():
            self._error(401, "invalid API key", "authentication_error")
            return

        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0:
            self._error(400, "request body is empty")
            return
        if length > 64 * 1024 * 1024:
            self._error(413, "request body is larger than 64 MB", "request_too_large")
            return
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._error(400, f"invalid JSON body: {exc}")
            return
        if not isinstance(body, dict):
            self._error(400, "request body must be a JSON object")
            return
        if body.get("stream"):
            self._error(
                400,
                "stream=true is not supported; send stream=false",
            )
            return

        try:
            messages = _normalize_messages(body.get("messages"))
            max_tokens = body.get("max_completion_tokens")
            if max_tokens is None:
                max_tokens = body.get("max_tokens")
            if max_tokens is None:
                max_tokens = MAX_NEW_TOKENS
            max_tokens = int(max_tokens)
            temperature = float(body.get("temperature", 0.1))
            top_p = body.get("top_p")
            top_p = None if top_p is None else float(top_p)
            top_k = body.get("top_k")
            top_k = None if top_k is None else int(top_k)
            penalty = body.get("repetition_penalty")
            if penalty is None:
                penalty = body.get("repeat_penalty")
            penalty = None if penalty is None else float(penalty)
            template_kwargs = body.get("chat_template_kwargs") or {}
            if not isinstance(template_kwargs, dict):
                raise BadRequest("chat_template_kwargs must be an object")
            stops = _stop_list(body.get("stop"))
            seed = body.get("seed")
            seed = None if seed is None else int(seed)
        except (TypeError, ValueError) as exc:
            self._error(400, f"invalid request field: {exc}")
            return
        except BadRequest as exc:
            self._error(400, str(exc))
            return

        try:
            # One model, one GPU: serialize generation so two jobs cannot
            # corrupt each other's KV cache or exhaust VRAM.
            with GENERATE_LOCK:
                result = _generate(
                    messages,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=penalty,
                    template_kwargs=template_kwargs,
                    tools=body.get("tools"),
                    stops=stops,
                    seed=seed,
                )
        except BadRequest as exc:
            self._error(400, str(exc))
            return
        except Exception as exc:  # pragma: no cover - runtime failure
            traceback.print_exc()
            self._error(
                500,
                f"generation failed: {type(exc).__name__}: {exc}",
                "server_error",
            )
            return

        self._send_json(200, _chat_response(result, SERVED_NAME))


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenAI-compatible server for LiquidAI/LFM3-1.7B")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Hugging Face id or local checkpoint directory")
    parser.add_argument("--local-dir", default=None,
                        help="download the checkpoint here (used with --download-only)")
    parser.add_argument("--download-only", action="store_true",
                        help="download the checkpoint and exit")
    parser.add_argument("--served-model-name", default=DEFAULT_SERVED_NAME)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8089)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dtype", default="auto",
                        choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--max-context", type=int, default=DEFAULT_MAX_CONTEXT,
                        help="prompt + output token budget (default 16384)")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
                        help="hard cap for one completion (default 4096)")
    parser.add_argument("--triton", default="auto", choices=["auto", "on", "off"],
                        help="LFM3 conv kernels: auto disables them before Ampere")
    parser.add_argument("--api-key", default=os.environ.get("LFM3_API_KEY") or None,
                        help="optional bearer token (the app sends LLM_API_KEY)")
    parser.add_argument("--no-warmup", action="store_true",
                        help="skip the tiny startup generation")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    global MODEL, TOKENIZER, TORCH, MODEL_PATH, SERVED_NAME, MAX_CONTEXT, MAX_NEW_TOKENS
    global MODEL_DEVICE, MODEL_DTYPE, TRITON_ACTIVE, API_KEY, STARTED_AT

    # A local directory is normalised; a Hugging Face id such as
    # "LiquidAI/LFM3-1.7B" must stay untouched.
    _model_arg = os.path.expanduser(args.model)
    MODEL_PATH = os.path.abspath(_model_arg) if os.path.isdir(_model_arg) else _model_arg
    SERVED_NAME = args.served_model_name.strip() or DEFAULT_SERVED_NAME
    MAX_CONTEXT = max(2048, int(args.max_context))
    MAX_NEW_TOKENS = _clamp_max_new_tokens(MAX_CONTEXT, args.max_new_tokens)
    if MAX_NEW_TOKENS != int(args.max_new_tokens):
        log(
            f"max_new_tokens {args.max_new_tokens} clamped to {MAX_NEW_TOKENS} "
            f"to keep room for the prompt in a {MAX_CONTEXT}-token context"
        )
    API_KEY = args.api_key
    STARTED_AT = int(time.time())

    if args.download_only:
        _download_model(args.model, args.local_dir)
        return 0

    torch, transformers, auto_model, auto_tokenizer = _import_runtime()
    TORCH = torch

    if os.path.isdir(MODEL_PATH):
        config = os.path.join(MODEL_PATH, "config.json")
        if not os.path.exists(config):
            sys.exit(
                f"{MODEL_PATH} is not a checkpoint directory (config.json missing). "
                "Run with --download-only --local-dir first."
            )
    elif not MODEL_PATH:
        sys.exit("--model is empty")

    MODEL_DEVICE = _resolve_device(torch, args.device)
    dtype = _resolve_dtype(torch, args.dtype, MODEL_DEVICE)
    MODEL_DTYPE = str(dtype).replace("torch.", "")
    log(
        f"model={args.model} | device={MODEL_DEVICE} | dtype={MODEL_DTYPE} | "
        f"context={MAX_CONTEXT} | max_new_tokens={MAX_NEW_TOKENS}"
    )

    MODEL, TOKENIZER = _load_model(
        torch, transformers, auto_model, auto_tokenizer,
        MODEL_PATH, MODEL_DEVICE, dtype,
    )

    # Decide whether to use LFM3's Triton conv kernels. The PyTorch fallback is
    # slower but works on every Colab GPU; the custom kernels need Triton >=3.3
    # and are safest on Ampere+.
    want_triton = args.triton != "off" and MODEL_DEVICE == "cuda"
    if args.triton == "auto" and MODEL_DEVICE == "cuda":
        major, minor = torch.cuda.get_device_capability(0)
        want_triton = major >= 8
        log(f"gpu compute capability {major}.{minor}")
    if os.environ.get("LFM3_DISABLE_TRITON", "").strip().lower() in ("1", "true", "yes", "on"):
        want_triton = False
    TRITON_ACTIVE = _set_triton(MODEL, want_triton)
    log(f"triton conv kernels: {'on' if TRITON_ACTIVE else 'off (pytorch fallback)'}")

    if not args.no_warmup:
        _warmup(torch)

    server = ThreadingHTTPServer((args.host, int(args.port)), Handler)
    server.daemon_threads = True
    log(f"ready on http://{args.host}:{args.port}/v1 | model={SERVED_NAME}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("stopping")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
