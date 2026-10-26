"""Transcript export and structured clip-selection import.

The normal clip picker calls Gemini twice (score windows, then detail clips).
When that quota is exhausted -- or the user simply prefers another model -- the
dashboard exports a structured transcript and later imports the other AI's
answer. This module owns both sides of that contract so app.py, main.py and the
tests share one implementation.

Standard-library only (plus clip_selection, which is also pure), so importing
it never pulls in torch, OpenCV or the Gemini SDK.
"""

import json
import math
import re
import time

from clip_selection import (build_transcript_windows, clip_count_targets,
                            clip_duration_bounds)

EXPORT_SCHEMA = "openshorts.transcript-export.v1"
SELECTION_SCHEMA = "openshorts.clip-selection.v1"

# A single import may not smuggle in an unbounded render queue. The UI/prompt
# ask for far fewer; this is a sanity ceiling, not a target.
MAX_CLIPS = 50

# Below this a "clip" is a timestamp typo, not a short.
MIN_CLIP_SECONDS = 1.0


class SelectionError(ValueError):
    """A user-supplied selection that cannot be turned into renderable clips."""


def _rounded(value, digits=3):
    return round(float(value), digits)


def _finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _segment_list(transcript):
    """Validated, JSON-safe segments from either a Whisper result or metadata."""
    segments = []
    for segment in (transcript or {}).get("segments", []):
        try:
            start = float(segment.get("start", 0))
            end = float(segment.get("end", 0))
        except (TypeError, ValueError):
            continue
        text = str(segment.get("text") or "").strip()
        if not text or not math.isfinite(start) or not math.isfinite(end) or end <= start:
            continue
        segments.append({
            "start": _rounded(max(0.0, start)),
            "end": _rounded(end),
            "text": text,
        })
    return segments


def _effective_duration(transcript, duration):
    try:
        duration = float(duration or 0)
    except (TypeError, ValueError):
        duration = 0.0
    if math.isfinite(duration) and duration > 0:
        return duration
    segments = _segment_list(transcript)
    return segments[-1]["end"] if segments else 0.0


def _prompt(duration, language, min_secs, max_secs, min_clips, max_clips):
    example = {
        "schema": SELECTION_SCHEMA,
        "clips": [
            {
                "start": 12.5,
                "end": 42.0,
                "score": 85,
                "title": "A specific YouTube Shorts title (max 100 chars)",
                "hook": "The strongest 10-word hook",
                "reason": "Why this moment works, in one sentence (max 20 words)",
                "description_tiktok": "1-2 sentences + 3-5 hashtags",
                "description_instagram": "1-2 sentences + 3-5 hashtags",
            }
        ],
    }
    return (
        "You are a senior short-form video editor. The JSON you received is an\n"
        "OpenShorts transcript export. Pick the moments that would work as\n"
        "standalone YouTube Shorts / TikToks / Reels.\n\n"
        f"SOURCE\n"
        f"- Language: {language}\n"
        f"- Duration: {duration:.3f} seconds\n"
        f"- Target clip length: {min_secs:g}-{max_secs:g} seconds\n"
        f"- Return {min_clips}-{max_clips} clips, best first.\n\n"
        "RULES\n"
        "- Timestamps are ABSOLUTE SECONDS from the start of the video.\n"
        f"- 0 <= start < end <= {duration:.3f}.\n"
        "- Open on the hook: the first 2 seconds must stop a cold viewer.\n"
        "- A clip must stand alone. Move the start back instead of cutting the payoff.\n"
        "- Prefer cuts on segment boundaries. Never cut in the middle of a word.\n"
        "- No two clips may cover the same moment.\n"
        f"- Write title, hook, reason and descriptions in {language}.\n"
        "- title: max 100 characters. hook: max 10 words. score: 0-100.\n"
        "- reason: one sentence, max 20 words, naming what makes this moment work.\n"
        "- Descriptions: 1-2 sentences that tease the payoff + 3-5 relevant hashtags.\n"
        "- The windows array groups consecutive transcript segments. Use it for\n"
        "  navigation; clips may start and end anywhere in the transcript.\n\n"
        "OUTPUT\n"
        "Return ONLY valid JSON, with no markdown fences and no commentary:\n"
        f"{json.dumps(example, ensure_ascii=False, indent=2)}"
    )


def build_transcript_export(transcript, duration, *, job_id=None, language=None,
                            min_secs=None, max_secs=None):
    """The JSON the dashboard downloads and the user hands to another AI.

    It carries both the raw timed transcript and the same candidate windows the
    Gemini pipeline uses. The windows are navigation aid only; the import
    accepts any valid timestamp inside the source.
    """
    transcript = transcript or {}
    duration = _effective_duration(transcript, duration)
    default_min, default_max = clip_duration_bounds()
    min_secs = float(default_min if min_secs is None else min_secs)
    max_secs = float(default_max if max_secs is None else max_secs)
    if max_secs < min_secs:
        min_secs, max_secs = default_min, default_max

    segments = _segment_list(transcript)
    language = str(language or transcript.get("language") or "unknown")
    window_source = {"segments": segments,
                     "text": str(transcript.get("text") or "")}
    windows = build_transcript_windows(
        window_source, duration,
        window_seconds=max(90, int(max_secs * 1.5)))
    min_clips, max_clips = clip_count_targets(len(windows))

    payload = {
        "schema": EXPORT_SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "job_id": job_id,
        "language": language,
        "duration_seconds": _rounded(duration),
        "clip_duration_seconds": {"min": min_secs, "max": max_secs},
        "recommended_clip_count": {"min": min_clips, "max": max_clips},
        "prompt": _prompt(duration, language, min_secs, max_secs,
                          min_clips, max_clips),
        "output_contract": {
            "schema": SELECTION_SCHEMA,
            "clips": [{
                "start": "number (absolute seconds)",
                "end": "number (absolute seconds)",
                "score": "integer 0-100 (optional, default 70)",
                "title": "optional, max 100 chars",
                "hook": "optional, max 10 words",
                "reason": "optional, one sentence",
                "description_tiktok": "optional",
                "description_instagram": "optional",
            }],
        },
        "transcript": {
            "text": str(transcript.get("text") or
                        " ".join(s["text"] for s in segments)).strip(),
            "segments": segments,
        },
        "windows": windows,
    }
    return payload


def _strip_code_fences(text):
    text = (text or "").strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines:
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


_FENCED_JSON_RE = re.compile(
    r"```(?:json|javascript|js)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def _embedded_json(text):
    """JSON an AI wrapped in prose: a fenced block, then the outer braces."""
    match = _FENCED_JSON_RE.search(text)
    if match:
        yield match.group(1).strip()
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if starts:
        start = min(starts)
        end = max(text.rfind("}"), text.rfind("]"))
        if end > start:
            yield text[start:end + 1].strip()


def _loads(payload):
    if isinstance(payload, (dict, list)):
        return payload
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8", errors="replace")
    if not isinstance(payload, str):
        raise SelectionError("Clip selection must be a JSON object or a JSON string.")
    text = _strip_code_fences(payload)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        # Chat UIs often prepend "Here is the JSON:" despite the prompt. Accept
        # the JSON inside the reply instead of sending the user back to edit it.
        for candidate in _embedded_json(text):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        raise SelectionError(
            f"That is not valid JSON (line {exc.lineno}, column {exc.colno}): "
            f"{exc.msg}.") from exc


def _unwrap(payload):
    """Accept the documented shape plus the app's own metadata shape."""
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        raise SelectionError("Clip selection must be a JSON object with a clips list.")
    for key in ("clips", "shorts"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    # A few chat tools wrap the answer in {"result": ...} / {"data": ...}.
    for key in ("selection", "result", "output", "data"):
        value = payload.get(key)
        if isinstance(value, (dict, list)):
            return _unwrap(value)
    raise SelectionError(
        "No clips found. Expected a top-level \"clips\" list "
        "(or the app's \"shorts\" list).")


def _number(clip, aliases):
    for name in aliases:
        if name not in clip or clip[name] is None or clip[name] == "":
            continue
        value = clip[name]
        if isinstance(value, bool):
            raise SelectionError(f"{name} must be a number, not true/false.")
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise SelectionError(f"{name} must be a number (got {value!r}).") from None
        if not math.isfinite(number):
            raise SelectionError(f"{name} must be a finite number (got {value!r}).")
        return number
    raise SelectionError(f"Missing required field: {aliases[0]}.")


def _text(clip, aliases):
    for name in aliases:
        value = clip.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _score(clip, warnings, index):
    raw = None
    for name in ("score", "predicted_score", "viral_score"):
        if clip.get(name) not in (None, ""):
            raw = clip[name]
            break
    if raw is None:
        return 70
    try:
        if isinstance(raw, bool):
            raise ValueError
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError
        score = int(round(value))
    except (TypeError, ValueError, OverflowError):
        warnings.append(f"Clip {index + 1}: score is not a number; using 70.")
        return 70
    if score < 0 or score > 100:
        warnings.append(f"Clip {index + 1}: score {score} clipped to 0-100.")
    return max(0, min(100, score))


def _truncate(text, limit):
    """Cut on a word boundary when possible; a hard limit when not."""
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    head = cut.rsplit(" ", 1)[0].strip()
    return head or cut.strip()


def _minutes_seconds(seconds):
    seconds = max(0, int(round(float(seconds))))
    return f"{seconds // 60}:{seconds % 60:02d}"


def _transcript_text(transcript, start, end):
    parts = []
    for segment in (transcript or {}).get("segments", []):
        try:
            seg_start = float(segment.get("start", 0))
            seg_end = float(segment.get("end", 0))
        except (TypeError, ValueError):
            continue
        if seg_end > start and seg_start < end:
            text = str(segment.get("text") or "").strip()
            if text:
                parts.append(text)
    return " ".join(parts).strip()


def _fallback_title(transcript, start, end):
    text = _transcript_text(transcript, start, end)
    if text:
        sentence = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0].strip()
        title = _truncate(sentence or text, 100)
        if title:
            return title
    return f"Clip {_minutes_seconds(start)}-{_minutes_seconds(end)}"


def _short_hook(title):
    words = title.split()
    return _truncate(" ".join(words[:10]), 80)


def parse_selection_payload(payload, *, duration, min_secs=15.0, max_secs=60.0,
                            transcript=None):
    """Validate an external AI's answer and normalize it to app clip fields.

    Returns ``{"schema", "clips", "warnings"}``. Raises SelectionError with a
    human-readable message for anything that would make the renderer fail.
    """
    if not _finite(duration):
        raise SelectionError("The source duration is unknown, so timestamps cannot be validated.")
    duration = float(duration)
    if duration <= 0:
        raise SelectionError("The source duration is unknown, so timestamps cannot be validated.")
    min_secs = float(min_secs or 0)
    max_secs = float(max_secs or 0)
    clips = _unwrap(_loads(payload))
    if not clips:
        raise SelectionError("The clip list is empty.")
    if len(clips) > MAX_CLIPS:
        raise SelectionError(
            f"Too many clips ({len(clips)}). Import at most {MAX_CLIPS}.")

    normalized = []
    warnings = []
    for index, raw in enumerate(clips):
        if not isinstance(raw, dict):
            raise SelectionError(f"Clip {index + 1} is not a JSON object.")
        try:
            start = _number(raw, ("start", "start_time", "start_seconds"))
            end = _number(raw, ("end", "end_time", "end_seconds"))
        except SelectionError as exc:
            raise SelectionError(f"Clip {index + 1}: {exc}") from None
        if start < 0:
            warnings.append(f"Clip {index + 1}: start was negative; clamped to 0.")
            start = 0.0
        if end > duration + 0.01:
            warnings.append(
                f"Clip {index + 1}: end {end:g}s is past the source; clamped to {duration:g}s.")
            end = duration
        if end <= start:
            raise SelectionError(
                f"Clip {index + 1}: end ({end:g}s) must be greater than start ({start:g}s).")
        length = end - start
        if length < MIN_CLIP_SECONDS:
            raise SelectionError(
                f"Clip {index + 1}: {length:.2f}s is too short to render "
                f"(minimum {MIN_CLIP_SECONDS:g}s).")
        if min_secs and length < min_secs:
            warnings.append(
                f"Clip {index + 1}: {length:.1f}s is below the requested "
                f"minimum ({min_secs:g}s); it will be kept as close as possible.")
        if max_secs and length > max_secs:
            warnings.append(
                f"Clip {index + 1}: {length:.1f}s is above the requested "
                f"maximum ({max_secs:g}s); it will be trimmed on a word boundary.")

        title = _text(raw, ("title", "video_title_for_youtube_short",
                            "youtube_title", "name"))
        if not title:
            title = _fallback_title(transcript, start, end)
        title = _truncate(title, 100) or _fallback_title(transcript, start, end)

        hook = _text(raw, ("hook", "viral_hook_text", "hook_text", "overlay_text"))
        hook = _short_hook(hook or title)

        reason = _text(raw, ("reason", "why", "explanation"))
        tiktok = _text(raw, ("description_tiktok", "video_description_for_tiktok",
                             "tiktok_description", "tiktok"))
        instagram = _text(raw, ("description_instagram",
                                "video_description_for_instagram",
                                "instagram_description", "instagram"))

        normalized.append({
            "start": _rounded(start),
            "end": _rounded(end),
            "source_window_id": _text(raw, ("source_window_id", "window_id")),
            "predicted_score": _score(raw, warnings, index),
            "video_description_for_tiktok": tiktok or title,
            "video_description_for_instagram": instagram or title,
            "video_title_for_youtube_short": title,
            "viral_hook_text": hook,
            "why": reason,
        })

    normalized.sort(key=lambda clip: clip["start"])
    return {"schema": SELECTION_SCHEMA, "clips": normalized, "warnings": warnings}
