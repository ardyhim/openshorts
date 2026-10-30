#!/usr/bin/env python3
"""OpenShorts-compatible transcript generator.

Input: a YouTube URL/ID, an SRT/VTT/TXT subtitle, a transcript JSON, or "-"
for stdin. Output: one of the app's own transcript contracts.

The three contracts mirrored here, and where they live in the app:

  export    manual_clip_selection.build_transcript_export()
            -> openshorts.transcript-export.v1
  pipeline  transcribe_backends.transcribe_media()
            -> {"text", "language", "segments": [{"start", "end", "text", "words"}]}
  youtube   free_tools.fetch_transcript()
            -> {"video_id", "title", ..., "segments": [{"start", "dur", "text"}]}

Text/SRT output follows dashboard/tools/transcript.js.

Standard library only, except yt-dlp, which is imported only for a YouTube
input. Run with `python openshorts_transcript.py --help`.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import sys
import time

EXPORT_SCHEMA = "openshorts.transcript-export.v1"
SELECTION_SCHEMA = "openshorts.clip-selection.v1"

# Same defaults as clip_selection.clip_duration_bounds().
DEFAULT_MIN_SECONDS = 15.0
DEFAULT_MAX_SECONDS = 60.0

_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
_TIME_RE = re.compile(
    r"^(?:(\d{1,3}):)?(\d{1,2}):(\d{2})(?:[.,](\d{1,3}))?$")


class TranscriptError(Exception):
    """A user-facing failure, with the same codes the app returns."""

    def __init__(self, code, message, status=400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


def configure_stdio():
    """UTF-8 stdout/stderr, like subtitles._configure_stdio in the app.

    Windows consoles otherwise die on a curly quote or a CJK title with
    UnicodeEncodeError before the JSON is written.
    """
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Time and text helpers
# --------------------------------------------------------------------------- #

def parse_timestamp(value):
    """'HH:MM:SS,mmm', 'MM:SS.mmm' or plain seconds -> float seconds."""
    text = str(value or "").strip().replace(",", ".")
    match = _TIME_RE.match(text)
    if not match:
        try:
            number = float(text)
        except (TypeError, ValueError):
            raise ValueError(f"not a timestamp: {value!r}") from None
        if not math.isfinite(number):
            raise ValueError(f"not a finite timestamp: {value!r}")
        return number
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2))
    seconds = int(match.group(3))
    fraction = (match.group(4) or "").ljust(3, "0")[:3]
    return hours * 3600 + minutes * 60 + seconds + int(fraction) / 1000.0


def clock(seconds, srt=False):
    """The dashboard's timestamp format: m:ss, h:mm:ss, or SRT HH:MM:SS,mmm."""
    milliseconds = int(round(float(seconds or 0) * 1000))
    hours, remainder = divmod(milliseconds, 3600000)
    minutes, remainder = divmod(remainder, 60000)
    secs, millis = divmod(remainder, 1000)
    if srt:
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def clean_text(value):
    """Subtitle text without HTML, VTT cue tags or ASS override blocks."""
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]*>", "", text)
    text = re.sub(r"\{\\[^}]*\}", "", text)
    return re.sub(r"\s+", " ", text).strip()


def detect_language(text):
    """Best-effort ISO 639-1 code, the same classifier Parakeet uses.

    py3langid is optional; without it the caller keeps 'unknown' rather than
    guessing from a filename.
    """
    sample = str(text or "").strip()
    if len(sample) < 20:
        return ""
    try:
        import py3langid
        language, _score = py3langid.classify(sample[:4000])
        return language or ""
    except Exception:
        return ""


def _blocks(data):
    text = str(data or "").replace("\r\n", "\n").replace("\r", "\n")
    text = text.lstrip("\ufeff").strip()
    return re.split(r"\n[ \t]*\n+", text) if text else []


def _cue_line(lines):
    for index, line in enumerate(lines):
        if "-->" in line:
            return index
    return None


# --------------------------------------------------------------------------- #
# Subtitle parsers
# --------------------------------------------------------------------------- #

def parse_srt(data):
    """SRT -> [{start, end, text}], in chronological order."""
    cues = []
    for block in _blocks(data):
        lines = [line.rstrip() for line in block.split("\n")]
        index = _cue_line(lines)
        if index is None:
            continue
        left, _, right = lines[index].partition("-->")
        try:
            start = parse_timestamp(left.strip().split()[0])
            end = parse_timestamp(right.strip().split()[0])
        except (ValueError, IndexError):
            continue
        text = clean_text(" ".join(lines[index + 1:]))
        if text and end > start:
            cues.append({"start": round(start, 3), "end": round(end, 3),
                         "text": text})
    cues.sort(key=lambda cue: cue["start"])
    return cues


def parse_vtt(data):
    """WebVTT -> [{start, end, text}], skipping NOTE/STYLE/REGION blocks."""
    cues = []
    for block in _blocks(data):
        lines = [line.rstrip() for line in block.split("\n")]
        first = (lines[0].strip() if lines else "").upper()
        if first.startswith(("NOTE", "STYLE", "REGION")):
            continue
        index = _cue_line(lines)
        if index is None:
            continue
        left, _, right = lines[index].partition("-->")
        # The right side may carry cue settings: "00:02.000 align:start".
        right_parts = right.strip().split()
        if not right_parts:
            continue
        try:
            start = parse_timestamp(left.strip().split()[0])
            end = parse_timestamp(right_parts[0])
        except (ValueError, IndexError):
            continue
        text = clean_text(" ".join(lines[index + 1:]))
        if text and end > start:
            cues.append({"start": round(start, 3), "end": round(end, 3),
                         "text": text})
    cues.sort(key=lambda cue: cue["start"])
    return cues


def _txt_timestamp_pattern():
    return r"(?:\d{1,3}:)?\d{1,2}:\d{2}(?:[.,]\d{1,3})?"


def parse_txt(data, duration=None):
    """Timestamped TXT -> (cues, implicit_last_end) or ([], plain_text).

    Accepts the common shapes:
      [00:01:23] text
      00:01:23 - text
      00:00:01,000 --> 00:00:03,000 text

    A single timestamp is a start; the next line's start becomes its end.
    The final cue needs --duration, otherwise the caller must not build a
    JSON transcript from it (that would invent a timecode).
    """
    stamp = _txt_timestamp_pattern()
    full = re.compile(rf"^\s*\[?({stamp})\]?\s*-->\s*\[?({stamp})\]?\s*(.*)$")
    one = re.compile(rf"^\s*\[?({stamp})\]?\s*(?:[-–—|:]\s*)?(.*)$")

    cues = []
    plain_lines = []
    for raw in str(data or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw.strip()
        if not line:
            continue
        match = full.match(line)
        if match:
            try:
                start = parse_timestamp(match.group(1))
                end = parse_timestamp(match.group(2))
            except ValueError:
                plain_lines.append(line)
                continue
            text = clean_text(match.group(3))
            if text and end > start:
                cues.append({"start": round(start, 3), "end": round(end, 3),
                             "text": text})
            continue
        match = one.match(line)
        if match:
            try:
                start = parse_timestamp(match.group(1))
            except ValueError:
                plain_lines.append(line)
                continue
            text = clean_text(match.group(2))
            if text:
                cues.append({"start": round(start, 3), "end": None, "text": text})
            continue
        if cues and not plain_lines:
            cues[-1]["text"] = clean_text(f"{cues[-1]['text']} {line}")
        else:
            plain_lines.append(line)

    if not cues:
        return [], " ".join(plain_lines).strip()

    cues.sort(key=lambda cue: cue["start"])
    for index, cue in enumerate(cues):
        if cue["end"] is None:
            if index + 1 < len(cues):
                cue["end"] = cues[index + 1]["start"]
            elif duration is not None:
                cue["end"] = round(float(duration), 3)
            else:
                cue["end"] = round(cue["start"] + 2.0, 3)
                return cues, True
        if cue["end"] <= cue["start"]:
            cue["end"] = round(cue["start"] + 0.05, 3)
    return cues, False


# --------------------------------------------------------------------------- #
# YouTube captions (mirrors free_tools.py)
# --------------------------------------------------------------------------- #

def extract_video_id(raw):
    """The 11-character id of one YouTube video, or None."""
    text = str(raw or "").strip()
    if _VIDEO_ID.match(text):
        return text
    if not re.match(r"^https?://", text, re.I):
        text = "https://" + text
    try:
        from urllib.parse import parse_qs, urlparse
        parsed = urlparse(text)
    except ValueError:
        return None
    host = (parsed.hostname or "").lower()
    if host == "youtu.be":
        candidate = parsed.path.strip("/").split("/")[0]
        return candidate if _VIDEO_ID.match(candidate) else None
    if not (host == "youtube.com" or host.endswith(".youtube.com")
            or host == "youtube-nocookie.com"
            or host.endswith(".youtube-nocookie.com")):
        return None
    parts = [part for part in parsed.path.split("/") if part]
    if parts and parts[0] in ("watch", "watch_popup"):
        candidate = (parse_qs(parsed.query).get("v") or [""])[0]
        return candidate if _VIDEO_ID.match(candidate) else None
    if len(parts) >= 2 and parts[0] in ("shorts", "live", "embed", "v", "e"):
        return parts[1] if _VIDEO_ID.match(parts[1]) else None
    return None


def _base_lang(code):
    return str(code or "").split("-")[0].lower()


def _track_name(formats, code):
    for item in formats or []:
        if item.get("name"):
            return re.sub(r"\s*\(.*?\)\s*$", "", item["name"]).strip() or code
    return code


def choose_track(info, lang=None):
    """Pick the caption track: (kind, key, languages). Mirrors free_tools."""
    manual = {key: value for key, value in (info.get("subtitles") or {}).items()
              if key != "live_chat" and value}
    auto = info.get("automatic_captions") or {}
    video_lang = info.get("language") or ""
    orig = [key for key in auto if key.endswith("-orig")]
    if not orig and video_lang and video_lang in auto:
        orig = [video_lang]

    languages = [{"code": key, "kind": "manual", "request": key,
                  "name": _track_name(manual[key], key)} for key in manual]
    for key in orig:
        code = key[:-5] if key.endswith("-orig") else key
        languages.append({"code": code, "kind": "auto",
                          "request": f"{code}-orig",
                          "name": _track_name(auto[key], code) + " (auto-generated)"})

    video_base = _base_lang(video_lang)
    languages.sort(key=lambda item: (_base_lang(item["code"]) != video_base,
                                     _base_lang(item["code"]) != "en",
                                     item["kind"] != "manual",
                                     item["name"].lower()))

    def auto_key(code):
        for key in orig:
            if key == f"{code}-orig" or key == code:
                return key
        return None

    if lang and lang.endswith("-orig"):
        key = auto_key(lang[:-5])
        if key:
            return "auto", key, languages
        lang = lang[:-5]
    if lang:
        if lang in manual:
            return "manual", lang, languages
        key = auto_key(lang)
        if key:
            return "auto", key, languages
    if video_lang:
        for key in manual:
            if key == video_lang or _base_lang(key) == _base_lang(video_lang):
                return "manual", key, languages
    if orig:
        return "auto", orig[0], languages
    if manual:
        for key in manual:
            if _base_lang(key) == "en":
                return "manual", key, languages
        return "manual", next(iter(manual)), languages
    return None, None, languages


def parse_json3(data):
    """YouTube json3 captions -> [{start, dur, text}] in seconds."""
    if isinstance(data, (bytes, str)):
        data = json.loads(data)
    cues = []
    for event in data.get("events") or []:
        segments = event.get("segs")
        if not segments:
            continue
        text = "".join(segment.get("utf8", "") for segment in segments)
        text = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
        if not text:
            continue
        start = (event.get("tStartMs") or 0) / 1000.0
        duration = (event.get("dDurationMs") or 0) / 1000.0
        cues.append({"start": round(start, 3), "dur": round(duration, 3),
                     "text": text})
    for current, following in zip(cues, cues[1:]):
        if current["start"] + current["dur"] > following["start"]:
            current["dur"] = round(max(0.0, following["start"] - current["start"]), 3)
    return cues


def is_degraded(info):
    """True for a watch page served without its player response."""
    return not info.get("formats") and not info.get("duration")


_UNAVAILABLE = re.compile(
    r"private video|video unavailable|has been removed|not available|members-only|"
    r"confirm your age|age-restricted|copyright|terminated|does not exist|premieres in",
    re.I)

_BLOCKED = re.compile(
    r"not a bot|sign in to confirm|429|too many requests|403|"
    r"unable to download|timed out|proxy|connection", re.I)


class _QuietLogger:
    def debug(self, msg):
        pass

    def info(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        pass


def fetch_youtube_transcript(video_id, lang=None, proxy=None, cookies=None):
    """Blocking YouTube caption fetch. Mirrors free_tools.fetch_transcript()."""
    try:
        import yt_dlp
    except ImportError:
        raise TranscriptError(
            "missing_dependency",
            "YouTube input needs yt-dlp. Install it with: pip install yt-dlp",
            500) from None

    url = f"https://www.youtube.com/watch?v={video_id}"
    options = {
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "logger": _QuietLogger(),
        "socket_timeout": 15,
        "ignore_no_formats_error": True,
        "extractor_args": {"youtube": {"player_client": ["default"]}},
    }
    if proxy:
        options["proxy"] = proxy
    if cookies:
        if not os.path.isfile(cookies):
            raise TranscriptError("bad_cookies",
                                  f"Cookies file not found: {cookies}", 400)
        options["cookiefile"] = cookies

    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=False)
            kind, key, languages = choose_track(info, lang)
            meta = {
                "video_id": video_id,
                "title": info.get("title") or "",
                "channel": info.get("channel") or info.get("uploader") or "",
                "duration": info.get("duration") or 0,
                "languages": languages,
            }
            if not kind and is_degraded(info):
                raise TranscriptError(
                    "youtube_blocked",
                    "YouTube returned an empty player response. Try again, or pass --cookies.",
                    503)
            if not kind:
                raise TranscriptError(
                    "no_captions",
                    "This video has no captions on YouTube, not even automatic ones.",
                    404)
            pool = (info.get("subtitles") if kind == "manual"
                    else info.get("automatic_captions")) or {}
            fmt = next((item for item in pool.get(key, [])
                        if item.get("ext") == "json3"), None)
            if not fmt:
                raise TranscriptError(
                    "no_captions",
                    "YouTube lists captions for this video but does not serve them "
                    "in a readable format.", 404)
            raw = ydl.urlopen(fmt["url"]).read()
    except TranscriptError:
        raise
    except Exception as exc:  # noqa: BLE001 - classified like the app
        message = str(exc)
        if _UNAVAILABLE.search(message) and not re.search(r"not a bot", message, re.I):
            raise TranscriptError(
                "unavailable",
                "YouTube says this video is private, removed, age-restricted or not "
                "available, so its captions cannot be read.", 422) from None
        if _BLOCKED.search(message):
            raise TranscriptError(
                "youtube_blocked",
                "YouTube is not answering right now. Try again in a minute, or "
                "open the video on YouTube and use \"Show transcript\".", 503) from None
        raise TranscriptError(
            "youtube_error",
            f"Could not read the YouTube captions: {message[:300]}", 502) from None

    segments = parse_json3(raw)
    if not segments:
        raise TranscriptError("no_captions",
                              "The caption track for this video is empty.", 404)
    code = key[:-5] if key.endswith("-orig") else key
    return {**meta, "language": code, "kind": kind, "segments": segments,
            "cached": False}


# --------------------------------------------------------------------------- #
# Shared transcript helpers (mirrors transcribe_backends / manual_clip_selection)
# --------------------------------------------------------------------------- #

def _number(value, default=0.0):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _rounded(value, digits=3):
    return round(float(value), digits)


def _segment_list(transcript):
    """Validated, JSON-safe segments from either a transcript or metadata."""
    segments = []
    for segment in (transcript or {}).get("segments", []):
        start = _number(segment.get("start", 0))
        end = _number(segment.get("end", 0))
        text = str(segment.get("text") or "").strip()
        if not text or end <= start:
            continue
        segments.append({
            "start": _rounded(max(0.0, start)),
            "end": _rounded(end),
            "text": text,
        })
    return segments


def _effective_duration(transcript, duration):
    duration = _number(duration, 0.0)
    if duration > 0:
        return duration
    segments = _segment_list(transcript)
    return segments[-1]["end"] if segments else 0.0


def clip_duration_bounds():
    """CLIP_MIN_SECONDS / CLIP_MAX_SECONDS, clamped like the app."""
    def read(name, default):
        try:
            return float(os.environ.get(name, ""))
        except (TypeError, ValueError):
            return default

    low = read("CLIP_MIN_SECONDS", DEFAULT_MIN_SECONDS)
    high = read("CLIP_MAX_SECONDS", DEFAULT_MAX_SECONDS)
    low = min(max(low, 5.0), 175.0)
    high = min(max(high, 10.0), 180.0)
    if high < low + 5.0:
        high = min(180.0, low + 5.0)
    return round(low, 3), round(high, 3)


def clip_count_targets(window_count):
    """The app's clip-count band for a given number of windows."""
    count = max(1, int(window_count or 1))
    low = max(2, min(6, count // 2 + 2))
    high = min(12, max(4, count * 2))
    low = min(low, high)

    def override(name, current):
        raw = os.environ.get(name)
        if not raw:
            return current
        try:
            return max(1, int(raw))
        except ValueError:
            return current

    low = override("CLIP_TARGET_MIN", low)
    high = override("CLIP_TARGET_MAX", high)
    return low, max(low, high)


def build_transcript_windows(transcript_result, video_duration,
                             window_seconds=90, overlap_seconds=30):
    """The app's scoring windows, aligned to segment boundaries."""
    segments = []
    for segment in transcript_result.get("segments", []):
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        segments.append((_number(segment.get("start", 0)),
                         _number(segment.get("end", 0)), text))

    windows = []
    window_index = 1
    index = 0
    total = len(segments)
    while index < total:
        window_start = segments[index][0]
        end_index = index
        while (end_index + 1 < total
               and segments[end_index + 1][1] - window_start <= window_seconds * 1.25):
            end_index += 1
            if segments[end_index][1] - window_start >= window_seconds:
                break
        window_end = segments[end_index][1]
        windows.append({
            "id": f"window_{window_index:03d}",
            "start": round(window_start, 3),
            "end": round(window_end, 3),
            "text": " ".join(seg[2] for seg in segments[index:end_index + 1]),
        })
        window_index += 1

        if end_index >= total - 1:
            break
        target = window_end - overlap_seconds
        next_index = index + 1
        while next_index <= end_index and segments[next_index][0] < target:
            next_index += 1
        index = max(next_index, index + 1)

    if not windows:
        windows.append({
            "id": "window_001",
            "start": 0.0,
            "end": round(_number(video_duration), 3),
            "text": str(transcript_result.get("text", "") or ""),
        })
    return windows


def _prompt(duration, language, min_secs, max_secs, min_clips, max_clips):
    """Verbatim copy of manual_clip_selection._prompt()."""
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
    """Verbatim contract of manual_clip_selection.build_transcript_export()."""
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
                        " ".join(segment["text"] for segment in segments)).strip(),
            "segments": segments,
        },
        "windows": windows,
    }
    return payload


# --------------------------------------------------------------------------- #
# Normalization: any source -> the three contracts
# --------------------------------------------------------------------------- #

def synthesize_words(text, start, end):
    """Estimate per-word times from a subtitle line.

    Subtitles carry line timings, not word timings. The estimate distributes
    the cue duration by word length and keeps the app's leading-space
    convention. It is valid for caption layout; it is not ASR word timing.
    """
    tokens = str(text or "").split()
    if not tokens or end <= start:
        return []
    weights = [max(1, len(token)) for token in tokens]
    total = float(sum(weights))
    words = []
    cursor = float(start)
    consumed = 0
    for index, (token, weight) in enumerate(zip(tokens, weights)):
        consumed += weight
        word_end = float(start) + (float(end) - float(start)) * consumed / total
        word_end = max(word_end, cursor + 0.01)
        if index == len(tokens) - 1:
            word_end = float(end)
        words.append({"word": " " + token,
                      "start": round(cursor, 3),
                      "end": round(min(word_end, float(end)), 3)})
        cursor = word_end
    return words


def _normalize_word_list(words):
    clean = []
    for word in words or []:
        if not isinstance(word, dict):
            continue
        text = str(word.get("word", word.get("w", "")) or "")
        if not text:
            continue
        start = _number(word.get("start", word.get("s", 0)))
        end = _number(word.get("end", word.get("e", start + 0.05)), start + 0.05)
        if end <= start:
            end = start + 0.05
        clean.append({"word": text, "start": round(start, 3),
                      "end": round(end, 3)})
    clean.sort(key=lambda word: word["start"])
    if clean and not any(word["word"].startswith(" ") for word in clean):
        for word in clean:
            word["word"] = " " + word["word"]
    return clean


def normalize_segments(raw_segments, *, source_text="", language="unknown"):
    segments = []
    for raw in raw_segments or []:
        if not isinstance(raw, dict):
            continue
        start = _number(raw.get("start", raw.get("s", 0)))
        end = raw.get("end", raw.get("e"))
        if end is None:
            duration = raw.get("dur", raw.get("d"))
            end = start + _number(duration, 0.0)
        end = _number(end, start)
        text = str(raw.get("text", raw.get("caption", raw.get("content", ""))) or "").strip()
        if not text or end <= start:
            continue
        words = _normalize_word_list(raw.get("words"))
        if not words:
            words = synthesize_words(text, start, end)
        segments.append({"start": round(max(0.0, start), 3),
                         "end": round(end, 3), "text": text, "words": words})
    segments.sort(key=lambda segment: segment["start"])
    text = str(source_text or "").strip()
    if not text:
        text = " ".join(segment["text"] for segment in segments).strip()
    language = str(language or "").strip() or detect_language(text)
    return {"text": text, "language": language or "unknown",
            "segments": segments}


def load_youtube(source, args):
    video_id = extract_video_id(source)
    if not video_id:
        raise TranscriptError(
            "bad_url",
            "Paste the link of one YouTube video (youtube.com/watch?v=..., "
            "youtu.be/... or a Shorts link).", 400)
    result = fetch_youtube_transcript(
        video_id, lang=args.lang, proxy=args.proxy, cookies=args.cookies)
    if args.list_languages:
        return {"kind": "languages", "video": result}
    segments = [{"start": cue["start"],
                 "end": round(cue["start"] + cue["dur"], 3),
                 "text": cue["text"]} for cue in result["segments"]]
    transcript = normalize_segments(segments, language=result["language"])
    return {
        "kind": "youtube",
        "video": result,
        "transcript": transcript,
        "duration": _number(result.get("duration"), 0.0)
        or (transcript["segments"][-1]["end"] if transcript["segments"] else 0.0),
        "implicit_last_end": False,
    }


def load_subtitle(text, *, language="unknown", duration=None, filename=""):
    lowered = (filename or "").lower()
    implicit_last_end = False
    if lowered.endswith(".srt"):
        cues, plain = parse_srt(text), ""
    elif lowered.endswith(".vtt"):
        cues, plain = parse_vtt(text), ""
    elif re.search(r"^\s*WEBVTT\b", text or "", re.I | re.M):
        cues, plain = parse_vtt(text), ""
    elif "-->" in (text or ""):
        # A subtitle pasted or piped through stdin, or an SRT with a .txt
        # extension. Cue blocks are unambiguous; try SRT then WebVTT.
        cues = parse_srt(text)
        if not cues:
            cues = parse_vtt(text)
        plain = ""
    else:
        cues, extra = parse_txt(text, duration=duration)
        if isinstance(extra, bool):
            implicit_last_end = extra
            plain = ""
        else:
            plain = extra
    if not cues:
        plain = plain or ""
        return {
            "kind": "text",
            "video": None,
            "transcript": {"text": plain,
                           "language": str(language or "").strip()
                           or detect_language(plain) or "unknown",
                           "segments": []},
            "plain_text": plain,
            "duration": _number(duration, 0.0),
            "implicit_last_end": False,
        }
    transcript = normalize_segments(cues, language=language)
    return {
        "kind": "subtitle",
        "video": None,
        "transcript": transcript,
        "duration": _number(duration, 0.0)
        or (transcript["segments"][-1]["end"] if transcript["segments"] else 0.0),
        "implicit_last_end": implicit_last_end,
    }


def load_json(data, *, language=None, duration=None):
    if isinstance(data, list):
        transcript = normalize_segments(data, language=language)
        return {"kind": "json", "video": None, "transcript": transcript,
                "duration": _number(duration, 0.0) or _last_end(transcript),
                "implicit_last_end": False}

    if not isinstance(data, dict):
        raise TranscriptError("bad_json", "Transcript JSON must be an object or a list.", 400)

    if data.get("schema") == EXPORT_SCHEMA or isinstance(data.get("transcript"), dict):
        inner = data.get("transcript") or {}
        transcript = normalize_segments(
            inner.get("segments"), source_text=inner.get("text", ""),
            language=language or data.get("language") or inner.get("language"))
        return {
            "kind": "json",
            "video": None,
            "transcript": transcript,
            "duration": _number(duration if duration is not None
                                else data.get("duration_seconds"), 0.0)
            or _last_end(transcript),
            "implicit_last_end": False,
        }

    if data.get("segments") is not None:
        video = None
        kind = "json"
        if data.get("video_id") and any("dur" in (seg or {}) for seg in data["segments"]):
            kind = "youtube"
            video = {
                "video_id": data.get("video_id"),
                "title": data.get("title") or "",
                "channel": data.get("channel") or "",
                "duration": data.get("duration") or 0,
                "languages": data.get("languages") or [],
                "language": data.get("language") or "unknown",
                "kind": data.get("kind") or "manual",
                "segments": data.get("segments") or [],
                "cached": bool(data.get("cached")),
            }
        transcript = normalize_segments(
            data.get("segments"), source_text=data.get("text", ""),
            language=language or data.get("language"))
        if video is None:
            video_duration = 0.0
        else:
            video_duration = _number(video.get("duration"), 0.0)
        return {"kind": kind, "video": video, "transcript": transcript,
                "duration": _number(duration if duration is not None
                                    else video_duration, 0.0) or _last_end(transcript),
                "implicit_last_end": False}

    raise TranscriptError(
        "bad_json",
        "No transcript found. Expected a transcript export, a pipeline "
        "transcript, a YouTube tool response, or a list of segments.", 400)


def _last_end(transcript):
    segments = (transcript or {}).get("segments") or []
    return _number(segments[-1].get("end"), 0.0) if segments else 0.0


def _looks_like_youtube(text):
    if _VIDEO_ID.match(text or ""):
        return True
    return bool(re.match(
        r"^(https?://)?(www\.|m\.)?(youtube\.com|youtu\.be|youtube-nocookie\.com)/",
        text or "", re.I))


def load_source(source, args):
    if source == "-":
        data = sys.stdin.read()
        stripped = data.strip()
        if not stripped:
            raise TranscriptError("empty", "No data on stdin.", 400)
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                return load_json(json.loads(stripped), language=args.language,
                                 duration=args.duration)
            except json.JSONDecodeError:
                pass
        return load_subtitle(data, language=args.language,
                             duration=args.duration, filename="<stdin>")

    if os.path.isfile(source):
        if source.lower().endswith(".json"):
            try:
                with open(source, "r", encoding="utf-8-sig") as handle:
                    data = json.load(handle)
            except (OSError, json.JSONDecodeError) as exc:
                raise TranscriptError("bad_json", f"Could not read JSON: {exc}", 400) from None
            return load_json(data, language=args.language,
                             duration=args.duration)
        try:
            with open(source, "r", encoding="utf-8-sig", errors="replace") as handle:
                data = handle.read()
        except OSError as exc:
            raise TranscriptError("bad_file", f"Could not read {source}: {exc}", 400) from None
        return load_subtitle(data, language=args.language,
                             duration=args.duration, filename=source)

    if _looks_like_youtube(source):
        return load_youtube(source, args)

    raise TranscriptError(
        "bad_input",
        f"No such file: {source}. Pass a YouTube URL, an .srt/.vtt/.txt/.json "
        "file, or '-' for stdin.", 400)


# --------------------------------------------------------------------------- #
# Output contracts
# --------------------------------------------------------------------------- #

def _require_known_end(source):
    if source.get("implicit_last_end"):
        raise TranscriptError(
            "ambiguous_duration",
            "The final timestamped line has no end time. Pass --duration "
            "<seconds> so the transcript does not invent one.", 400)


def to_pipeline(source):
    _require_known_end(source)
    transcript = source["transcript"]
    if not transcript.get("segments"):
        raise TranscriptError(
            "no_timestamps",
            "This input has no timestamps, so it cannot become a transcript "
            "JSON. Provide SRT/VTT/JSON, or use --format text.", 400)
    return transcript


def to_youtube(source):
    video = source.get("video")
    if not video:
        raise TranscriptError(
            "not_youtube",
            "--format youtube needs a YouTube input (or a YouTube tool JSON).",
            400)
    segments = [{"start": segment["start"],
                 "dur": round(segment["end"] - segment["start"], 3),
                 "text": segment["text"]}
                for segment in source["transcript"]["segments"]]
    return {
        "video_id": video.get("video_id"),
        "title": video.get("title") or "",
        "channel": video.get("channel") or "",
        "duration": video.get("duration") or source.get("duration") or 0,
        "languages": video.get("languages") or [],
        "language": source["transcript"].get("language")
        or video.get("language") or "unknown",
        "kind": video.get("kind") or "manual",
        "segments": segments,
        "cached": bool(video.get("cached")),
    }


def to_export(source, args):
    _require_known_end(source)
    transcript = source["transcript"]
    if not transcript.get("segments"):
        raise TranscriptError(
            "no_timestamps",
            "This input has no timestamps, so it cannot become a transcript "
            "export. Provide SRT/VTT/JSON, or use --format text.", 400)
    return build_transcript_export(
        transcript, source.get("duration"), job_id=args.job_id,
        language=args.language or transcript.get("language"),
        min_secs=args.min_seconds, max_secs=args.max_seconds)


def to_text(source, timestamps=False):
    transcript = source["transcript"]
    segments = transcript.get("segments") or []
    if not segments:
        text = source.get("plain_text") or transcript.get("text") or ""
        return re.sub(r" {2,}", " ", text).strip()
    if timestamps:
        text = "\n".join(f"[{clock(segment['start'])}] {segment['text']}"
                         for segment in segments)
    else:
        text = " ".join(segment["text"] for segment in segments)
    return re.sub(r" {2,}", " ", text).strip()


def to_srt(source):
    segments = source["transcript"].get("segments") or []
    if not segments:
        raise TranscriptError(
            "no_timestamps",
            "This input has no timestamps, so it cannot become SRT.", 400)
    lines = []
    for index, segment in enumerate(segments, 1):
        duration = max(segment["end"] - segment["start"], 0.5)
        end = segment["start"] + duration
        lines.append(
            f"{index}\n{clock(segment['start'], srt=True)} --> "
            f"{clock(end, srt=True)}\n{segment['text']}\n")
    return "\n".join(lines)


def render(source, args):
    if args.format == "pipeline":
        payload = to_pipeline(source)
        return json.dumps(payload, ensure_ascii=False,
                          indent=None if args.compact else 2)
    if args.format == "export":
        payload = to_export(source, args)
        return json.dumps(payload, ensure_ascii=False,
                          indent=None if args.compact else 2)
    if args.format == "youtube":
        payload = to_youtube(source)
        return json.dumps(payload, ensure_ascii=False,
                          indent=None if args.compact else 2)
    if args.format == "text":
        return to_text(source, timestamps=args.timestamps)
    if args.format == "srt":
        return to_srt(source)
    raise TranscriptError("bad_format", f"Unknown format: {args.format}", 400)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser():
    parser = argparse.ArgumentParser(
        prog="openshorts-transcript",
        description="Generate an OpenShorts-compatible transcript from a "
                    "YouTube URL/ID or a subtitle/transcript file.")
    parser.add_argument(
        "source",
        help="YouTube URL/ID, .srt/.vtt/.txt/.json file, or '-' for stdin.")
    parser.add_argument(
        "--format", choices=("export", "pipeline", "youtube", "text", "srt"),
        default="export",
        help="export = openshorts.transcript-export.v1 (default); pipeline = "
             "word-level source_transcript.json; youtube = the public "
             "/api/tools/youtube-transcript response; text/srt = readable "
             "outputs.")
    parser.add_argument(
        "--lang", default=None,
        help="YouTube caption track to request (e.g. en, es-orig).")
    parser.add_argument(
        "--language", default=None,
        help="Language code written to the transcript (defaults to the "
             "caption track, the JSON's language, or 'unknown').")
    parser.add_argument(
        "--duration", type=float, default=None,
        help="Source duration in seconds. Required for a TXT whose last line "
             "has no end timestamp.")
    parser.add_argument(
        "--job-id", default=None,
        help="job_id to write into the export (default: null).")
    parser.add_argument(
        "--min-seconds", type=float, default=None,
        help="Export clip minimum (default: CLIP_MIN_SECONDS or 15).")
    parser.add_argument(
        "--max-seconds", type=float, default=None,
        help="Export clip maximum (default: CLIP_MAX_SECONDS or 60).")
    parser.add_argument(
        "--timestamps", action="store_true",
        help="With --format text, prefix each line with [m:ss].")
    parser.add_argument(
        "--compact", action="store_true",
        help="Emit compact JSON instead of indented JSON.")
    parser.add_argument(
        "--out", "-o", default=None,
        help="Write to this file instead of stdout.")
    parser.add_argument(
        "--proxy", default=None,
        help="Proxy URL for the YouTube request.")
    parser.add_argument(
        "--cookies", default=None,
        help="Netscape cookies.txt for YouTube (age-restricted/blocked videos).")
    parser.add_argument(
        "--list-languages", action="store_true",
        help="YouTube only: print the available caption tracks as JSON.")
    return parser


def main(argv=None):
    configure_stdio()
    args = build_parser().parse_args(argv)
    try:
        if args.list_languages and not _looks_like_youtube(args.source):
            raise TranscriptError(
                "not_youtube",
                "--list-languages needs a YouTube URL or ID.", 400)
        source = load_source(args.source, args)
        if source.get("kind") == "languages":
            result = source["video"]
            print(json.dumps(result.get("languages") or [], ensure_ascii=False,
                             indent=None if args.compact else 2))
            return 0
        if args.language:
            source["transcript"]["language"] = args.language
        if args.duration is not None:
            last_end = _last_end(source["transcript"])
            if args.duration < last_end:
                raise TranscriptError(
                    "bad_duration",
                    f"--duration ({args.duration:g}s) is before the last "
                    f"segment end ({last_end:g}s).", 400)
            source["duration"] = args.duration
        output = render(source, args)
        if args.out:
            directory = os.path.dirname(os.path.abspath(args.out))
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(output)
                if output and not output.endswith("\n"):
                    handle.write("\n")
        else:
            sys.stdout.write(output)
            if output and not output.endswith("\n"):
                sys.stdout.write("\n")
        return 0
    except TranscriptError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        return 2
    except BrokenPipeError:
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001 - keep the CLI honest
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
