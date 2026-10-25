"""
Free / local replacements for the paid providers used by OpenShorts.

Everything here is optional and lazily imported: a normal install (with fal.ai,
ElevenLabs, Gemini image generation and S3) never touches `diffusers`, `torch`
or the Wav2Lip checkout, and `FREE_MODE=0` leaves the original code paths
byte-for-byte intact.

Enabled with ``FREE_MODE=1`` (or automatically when the matching key is
missing):

    fal.ai Flux (actor + b-roll)   -> local SDXL (RealVisXL V4.0) via diffusers
    fal.ai Kling Avatar / lipsync  -> local Wav2Lip (+ still-image fallback)
    ElevenLabs TTS                 -> edge-tts (free Microsoft neural voices)
    ElevenLabs dubbing             -> faster-whisper + Gemini/Google translate + edge-tts
    Gemini image thumbnails        -> real video frame + burned text (PIL)
    Gemini titles/description      -> deterministic transcript-based fallbacks
    AWS S3                         -> local /output (handled by s3_uploader)

The module is written to degrade instead of raising: no GPU means generated
gradient placeholders instead of SDXL images, no Wav2Lip means a Ken Burns
still image with the right audio, no translation key means the free Google web
translator.
"""

from __future__ import annotations

import asyncio
import gc
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Dict, List, Optional

# Keep model downloads inside the project by default so a Colab run can be
# inspected (and so a second run re-uses the cache).
ROOT = os.path.dirname(os.path.abspath(__file__))
THIRD_PARTY = os.path.join(ROOT, "third_party")
os.makedirs(THIRD_PARTY, exist_ok=True)

TRUTHY = ("1", "true", "yes", "on")


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in TRUTHY


def free_mode() -> bool:
    """True when the deployment asked for the free/local pipeline."""
    return _env_bool("FREE_MODE")


def use_free_images(fal_key: Optional[str]) -> bool:
    """FREE_MODE wins; otherwise a missing fal.ai key falls back to local."""
    return free_mode() or not (fal_key or "").strip()


def use_free_tts(elevenlabs_key: Optional[str]) -> bool:
    return free_mode() or not (elevenlabs_key or "").strip()


def use_free_dubbing(elevenlabs_key: Optional[str]) -> bool:
    return free_mode() or not (elevenlabs_key or "").strip()


def use_free_thumbnails(gemini_key: Optional[str]) -> bool:
    """Free frame-composed thumbnails when Gemini is absent (or in FREE_MODE)."""
    return free_mode() or not (gemini_key or "").strip()


# ═══════════════════════════════════════════════════════════════════════
# Small helpers
# ═══════════════════════════════════════════════════════════════════════

def _run(cmd: List[str], timeout: int = 900, cwd: Optional[str] = None,
         check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout,
        encoding="utf-8", errors="replace",
    )
    if check and proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-1500:]
        raise RuntimeError(f"command failed ({cmd[0]}): {tail}")
    return proc


def _duration(path: str) -> float:
    try:
        proc = _run([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", path,
        ], timeout=60, check=False)
        return float((proc.stdout or "").strip() or 0)
    except Exception:
        return 0.0


def _has_audio(path: str) -> bool:
    try:
        proc = _run([
            "ffprobe", "-v", "error", "-select_streams", "a",
            "-show_entries", "stream=codec_type", "-of", "csv=p=0", path,
        ], timeout=60, check=False)
        return "audio" in (proc.stdout or "")
    except Exception:
        return False


def _download(url: str, dest: str, timeout: int = 900) -> str:
    """Stream a URL to disk with a .part file so a retry never sees half a file."""
    import httpx

    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    part = dest + ".part"
    with httpx.Client(timeout=httpx.Timeout(60.0, read=timeout), follow_redirects=True) as client:
        with client.stream("GET", url) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length") or 0)
            done = 0
            with open(part, "wb") as f:
                for chunk in resp.iter_bytes(1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        done += len(chunk)
            if total and done < total:
                raise RuntimeError(f"short download: {done}/{total} bytes from {url}")
    os.replace(part, dest)
    return dest


def _run_async(coro):
    """Run an asyncio coroutine from sync worker threads (SaaS pipeline) or an
    event loop that is already running (FastAPI handler)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    box = {}

    def runner():
        try:
            box["value"] = asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            box["error"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box.get("value")


# ═══════════════════════════════════════════════════════════════════════
# edge-tts: free Microsoft neural voices
# ═══════════════════════════════════════════════════════════════════════

# Curated subset. edge-tts ships many more; these cover every language the
# dubbing UI exposes plus the two AI Shorts script languages.
FREE_VOICE_CATALOG: List[Dict[str, str]] = [
    # English
    {"id": "en-US-AriaNeural", "gender": "female", "accent": "American", "language": "en", "name": "Aria (US, female)"},
    {"id": "en-US-JennyNeural", "gender": "female", "accent": "American", "language": "en", "name": "Jenny (US, female)"},
    {"id": "en-US-GuyNeural", "gender": "male", "accent": "American", "language": "en", "name": "Guy (US, male)"},
    {"id": "en-US-ChristopherNeural", "gender": "male", "accent": "American", "language": "en", "name": "Christopher (US, male)"},
    {"id": "en-GB-SoniaNeural", "gender": "female", "accent": "British", "language": "en", "name": "Sonia (UK, female)"},
    {"id": "en-GB-RyanNeural", "gender": "male", "accent": "British", "language": "en", "name": "Ryan (UK, male)"},
    # Spanish
    {"id": "es-ES-ElviraNeural", "gender": "female", "accent": "Spanish", "language": "es", "name": "Elvira (ES, female)"},
    {"id": "es-ES-AlvaroNeural", "gender": "male", "accent": "Spanish", "language": "es", "name": "Álvaro (ES, male)"},
    {"id": "es-MX-DaliaNeural", "gender": "female", "accent": "Latin American", "language": "es", "name": "Dalia (MX, female)"},
    {"id": "es-MX-JorgeNeural", "gender": "male", "accent": "Latin American", "language": "es", "name": "Jorge (MX, male)"},
    # Indonesian
    {"id": "id-ID-GadisNeural", "gender": "female", "accent": "Indonesian", "language": "id", "name": "Gadis (ID, female)"},
    {"id": "id-ID-ArdiNeural", "gender": "male", "accent": "Indonesian", "language": "id", "name": "Ardi (ID, male)"},
    # French
    {"id": "fr-FR-DeniseNeural", "gender": "female", "accent": "French", "language": "fr", "name": "Denise (FR, female)"},
    {"id": "fr-FR-HenriNeural", "gender": "male", "accent": "French", "language": "fr", "name": "Henri (FR, male)"},
    # German
    {"id": "de-DE-KatjaNeural", "gender": "female", "accent": "German", "language": "de", "name": "Katja (DE, female)"},
    {"id": "de-DE-ConradNeural", "gender": "male", "accent": "German", "language": "de", "name": "Conrad (DE, male)"},
    # Italian
    {"id": "it-IT-ElsaNeural", "gender": "female", "accent": "Italian", "language": "it", "name": "Elsa (IT, female)"},
    {"id": "it-IT-DiegoNeural", "gender": "male", "accent": "Italian", "language": "it", "name": "Diego (IT, male)"},
    # Portuguese
    {"id": "pt-BR-FranciscaNeural", "gender": "female", "accent": "Brazilian", "language": "pt", "name": "Francisca (BR, female)"},
    {"id": "pt-BR-AntonioNeural", "gender": "male", "accent": "Brazilian", "language": "pt", "name": "Antônio (BR, male)"},
    # Hindi
    {"id": "hi-IN-SwaraNeural", "gender": "female", "accent": "Hindi", "language": "hi", "name": "Swara (IN, female)"},
    {"id": "hi-IN-MadhurNeural", "gender": "male", "accent": "Hindi", "language": "hi", "name": "Madhur (IN, male)"},
    # Japanese / Korean / Chinese
    {"id": "ja-JP-NanamiNeural", "gender": "female", "accent": "Japanese", "language": "ja", "name": "Nanami (JP, female)"},
    {"id": "ja-JP-KeitaNeural", "gender": "male", "accent": "Japanese", "language": "ja", "name": "Keita (JP, male)"},
    {"id": "ko-KR-SunHiNeural", "gender": "female", "accent": "Korean", "language": "ko", "name": "SunHi (KR, female)"},
    {"id": "ko-KR-InJoonNeural", "gender": "male", "accent": "Korean", "language": "ko", "name": "InJoon (KR, male)"},
    {"id": "zh-CN-XiaoxiaoNeural", "gender": "female", "accent": "Mandarin", "language": "zh", "name": "Xiaoxiao (CN, female)"},
    {"id": "zh-CN-YunxiNeural", "gender": "male", "accent": "Mandarin", "language": "zh", "name": "Yunxi (CN, male)"},
    # Arabic
    {"id": "ar-EG-SalmaNeural", "gender": "female", "accent": "Arabic", "language": "ar", "name": "Salma (EG, female)"},
    {"id": "ar-EG-ShakirNeural", "gender": "male", "accent": "Arabic", "language": "ar", "name": "Shakir (EG, male)"},
    # Russian / Turkish / Dutch / Polish
    {"id": "ru-RU-SvetlanaNeural", "gender": "female", "accent": "Russian", "language": "ru", "name": "Svetlana (RU, female)"},
    {"id": "ru-RU-DmitryNeural", "gender": "male", "accent": "Russian", "language": "ru", "name": "Dmitry (RU, male)"},
    {"id": "tr-TR-EmelNeural", "gender": "female", "accent": "Turkish", "language": "tr", "name": "Emel (TR, female)"},
    {"id": "tr-TR-AhmetNeural", "gender": "male", "accent": "Turkish", "language": "tr", "name": "Ahmet (TR, male)"},
    {"id": "nl-NL-ColetteNeural", "gender": "female", "accent": "Dutch", "language": "nl", "name": "Colette (NL, female)"},
    {"id": "nl-NL-MaartenNeural", "gender": "male", "accent": "Dutch", "language": "nl", "name": "Maarten (NL, male)"},
    {"id": "pl-PL-ZofiaNeural", "gender": "female", "accent": "Polish", "language": "pl", "name": "Zofia (PL, female)"},
    {"id": "pl-PL-MarekNeural", "gender": "male", "accent": "Polish", "language": "pl", "name": "Marek (PL, male)"},
    # South-East Asia
    {"id": "vi-VN-HoaiMyNeural", "gender": "female", "accent": "Vietnamese", "language": "vi", "name": "HoaiMy (VN, female)"},
    {"id": "vi-VN-NamMinhNeural", "gender": "male", "accent": "Vietnamese", "language": "vi", "name": "NamMinh (VN, male)"},
    {"id": "th-TH-PremwadeeNeural", "gender": "female", "accent": "Thai", "language": "th", "name": "Premwadee (TH, female)"},
    {"id": "th-TH-NiwatNeural", "gender": "male", "accent": "Thai", "language": "th", "name": "Niwat (TH, male)"},
    {"id": "fil-PH-BlessicaNeural", "gender": "female", "accent": "Filipino", "language": "fil", "name": "Blessica (PH, female)"},
    {"id": "fil-PH-AngeloNeural", "gender": "male", "accent": "Filipino", "language": "fil", "name": "Angelo (PH, male)"},
    {"id": "ms-MY-YasminNeural", "gender": "female", "accent": "Malay", "language": "ms", "name": "Yasmin (MY, female)"},
    {"id": "ms-MY-OsmanNeural", "gender": "male", "accent": "Malay", "language": "ms", "name": "Osman (MY, male)"},
    # Europe (the rest of the dubbing language list)
    {"id": "uk-UA-PolinaNeural", "gender": "female", "accent": "Ukrainian", "language": "uk", "name": "Polina (UA, female)"},
    {"id": "uk-UA-OstapNeural", "gender": "male", "accent": "Ukrainian", "language": "uk", "name": "Ostap (UA, male)"},
    {"id": "el-GR-AthinaNeural", "gender": "female", "accent": "Greek", "language": "el", "name": "Athina (GR, female)"},
    {"id": "el-GR-NestorasNeural", "gender": "male", "accent": "Greek", "language": "el", "name": "Nestoras (GR, male)"},
    {"id": "cs-CZ-VlastaNeural", "gender": "female", "accent": "Czech", "language": "cs", "name": "Vlasta (CZ, female)"},
    {"id": "cs-CZ-AntoninNeural", "gender": "male", "accent": "Czech", "language": "cs", "name": "Antonín (CZ, male)"},
    {"id": "fi-FI-NooraNeural", "gender": "female", "accent": "Finnish", "language": "fi", "name": "Noora (FI, female)"},
    {"id": "fi-FI-HarriNeural", "gender": "male", "accent": "Finnish", "language": "fi", "name": "Harri (FI, male)"},
    {"id": "ro-RO-AlinaNeural", "gender": "female", "accent": "Romanian", "language": "ro", "name": "Alina (RO, female)"},
    {"id": "ro-RO-EmilNeural", "gender": "male", "accent": "Romanian", "language": "ro", "name": "Emil (RO, male)"},
    {"id": "da-DK-ChristelNeural", "gender": "female", "accent": "Danish", "language": "da", "name": "Christel (DK, female)"},
    {"id": "da-DK-JeppeNeural", "gender": "male", "accent": "Danish", "language": "da", "name": "Jeppe (DK, male)"},
    {"id": "bg-BG-KalinaNeural", "gender": "female", "accent": "Bulgarian", "language": "bg", "name": "Kalina (BG, female)"},
    {"id": "bg-BG-BorislavNeural", "gender": "male", "accent": "Bulgarian", "language": "bg", "name": "Borislav (BG, male)"},
    {"id": "hr-HR-GabrijelaNeural", "gender": "female", "accent": "Croatian", "language": "hr", "name": "Gabrijela (HR, female)"},
    {"id": "hr-HR-SreckoNeural", "gender": "male", "accent": "Croatian", "language": "hr", "name": "Srećko (HR, male)"},
    {"id": "sk-SK-ViktoriaNeural", "gender": "female", "accent": "Slovak", "language": "sk", "name": "Viktória (SK, female)"},
    {"id": "sk-SK-LukasNeural", "gender": "male", "accent": "Slovak", "language": "sk", "name": "Lukáš (SK, male)"},
    {"id": "sv-SE-SofieNeural", "gender": "female", "accent": "Swedish", "language": "sv", "name": "Sofie (SE, female)"},
    {"id": "sv-SE-MattiasNeural", "gender": "male", "accent": "Swedish", "language": "sv", "name": "Mattias (SE, male)"},
    {"id": "ta-IN-PallaviNeural", "gender": "female", "accent": "Tamil", "language": "ta", "name": "Pallavi (IN, female)"},
    {"id": "ta-IN-ValluvarNeural", "gender": "male", "accent": "Tamil", "language": "ta", "name": "Valluvar (IN, male)"},
]

_ELEVEN_TO_EDGE = {
    "21m00Tcm4TlvDq8ikWAM": "en-US-AriaNeural",       # Rachel
    "EXAVITQu4vr4xnSDxMaL": "en-US-JennyNeural",      # Bella
    "29vD33N1CtxCmqQRPOHJ": "en-US-GuyNeural",        # Drew
    "ErXwobaYiN019PkySvjV": "en-US-ChristopherNeural",  # Antoni
    "TxGEqnHWrfWFTfGW9XjX": "en-US-BrianNeural",      # Josh (fallback if absent)
    "yoZ06aMxZJJ28mfd3POQ": "en-US-AndrewNeural",     # Sam (fallback if absent)
}

_EDGE_IDS = {v["id"] for v in FREE_VOICE_CATALOG}
# A couple of catalog entries above use newer voices that older edge-tts
# builds may not know. They are only fallbacks through ElevenLabs ids, so a
# missing voice is caught and retried with the language default.
_FALLBACK_VOICES = {
    "en": {"female": "en-US-AriaNeural", "male": "en-US-GuyNeural"},
    "es": {"female": "es-ES-ElviraNeural", "male": "es-ES-AlvaroNeural"},
    "id": {"female": "id-ID-GadisNeural", "male": "id-ID-ArdiNeural"},
    "fr": {"female": "fr-FR-DeniseNeural", "male": "fr-FR-HenriNeural"},
    "de": {"female": "de-DE-KatjaNeural", "male": "de-DE-ConradNeural"},
    "it": {"female": "it-IT-ElsaNeural", "male": "it-IT-DiegoNeural"},
    "pt": {"female": "pt-BR-FranciscaNeural", "male": "pt-BR-AntonioNeural"},
    "hi": {"female": "hi-IN-SwaraNeural", "male": "hi-IN-MadhurNeural"},
    "ja": {"female": "ja-JP-NanamiNeural", "male": "ja-JP-KeitaNeural"},
    "ko": {"female": "ko-KR-SunHiNeural", "male": "ko-KR-InJoonNeural"},
    "zh": {"female": "zh-CN-XiaoxiaoNeural", "male": "zh-CN-YunxiNeural"},
    "ar": {"female": "ar-EG-SalmaNeural", "male": "ar-EG-ShakirNeural"},
    "ru": {"female": "ru-RU-SvetlanaNeural", "male": "ru-RU-DmitryNeural"},
    "tr": {"female": "tr-TR-EmelNeural", "male": "tr-TR-AhmetNeural"},
    "nl": {"female": "nl-NL-ColetteNeural", "male": "nl-NL-MaartenNeural"},
    "pl": {"female": "pl-PL-ZofiaNeural", "male": "pl-PL-MarekNeural"},
    "vi": {"female": "vi-VN-HoaiMyNeural", "male": "vi-VN-NamMinhNeural"},
    "th": {"female": "th-TH-PremwadeeNeural", "male": "th-TH-NiwatNeural"},
    "fil": {"female": "fil-PH-BlessicaNeural", "male": "fil-PH-AngeloNeural"},
    "ms": {"female": "ms-MY-YasminNeural", "male": "ms-MY-OsmanNeural"},
    "uk": {"female": "uk-UA-PolinaNeural", "male": "uk-UA-OstapNeural"},
    "el": {"female": "el-GR-AthinaNeural", "male": "el-GR-NestorasNeural"},
    "cs": {"female": "cs-CZ-VlastaNeural", "male": "cs-CZ-AntoninNeural"},
    "fi": {"female": "fi-FI-NooraNeural", "male": "fi-FI-HarriNeural"},
    "ro": {"female": "ro-RO-AlinaNeural", "male": "ro-RO-EmilNeural"},
    "da": {"female": "da-DK-ChristelNeural", "male": "da-DK-JeppeNeural"},
    "bg": {"female": "bg-BG-KalinaNeural", "male": "bg-BG-BorislavNeural"},
    "hr": {"female": "hr-HR-GabrijelaNeural", "male": "hr-HR-SreckoNeural"},
    "sk": {"female": "sk-SK-ViktoriaNeural", "male": "sk-SK-LukasNeural"},
    "sv": {"female": "sv-SE-SofieNeural", "male": "sv-SE-MattiasNeural"},
    "ta": {"female": "ta-IN-PallaviNeural", "male": "ta-IN-ValluvarNeural"},
}


def _language_of(code: Optional[str], default: str = "en") -> str:
    code = (code or "").strip().lower().replace("_", "-")
    if not code:
        return default
    base = code.split("-")[0]
    return base if base in _FALLBACK_VOICES else default


def free_voice_for_language(language: Optional[str], gender: str = "female") -> str:
    lang = _language_of(language)
    table = _FALLBACK_VOICES.get(lang, _FALLBACK_VOICES["en"])
    return table.get((gender or "female").lower(), table["female"])


def resolve_free_voice(voice_id: Optional[str], language: Optional[str] = None,
                       gender: str = "female") -> str:
    """Accept an edge voice id, an ElevenLabs id (mapped), or a UI default."""
    voice_id = (voice_id or "").strip()
    lang = _language_of(language)
    if voice_id in _EDGE_IDS:
        # Keep an explicitly selected English voice for English; for another
        # language prefer that language's catalog entry.
        if voice_id.startswith(lang + "-") or lang == "en":
            return voice_id
    mapped = _ELEVEN_TO_EDGE.get(voice_id)
    if mapped and lang == "en" and mapped in _EDGE_IDS:
        return mapped
    return free_voice_for_language(lang, gender)


def list_free_voices(language: Optional[str] = None) -> List[dict]:
    """Shape matches /api/saasshorts/voices (and the ElevenLabs payload enough
    for the dashboard's gender/accent filters)."""
    lang = _language_of(language) if language else None
    out = []
    for v in FREE_VOICE_CATALOG:
        if lang and v["language"] != lang:
            continue
        out.append({
            "voice_id": v["id"],
            "name": v["name"],
            "category": "free (edge-tts)",
            "labels": {"gender": v["gender"], "accent": v["accent"], "language": v["language"]},
            "preview_url": "",
        })
    return out


def edge_tts_save(text: str, output_path: str, voice: Optional[str] = None,
                  language: Optional[str] = None, gender: str = "female",
                  rate: str = "+0%", pitch: str = "+0Hz",
                  log=print) -> str:
    """Synthesise `text` to `output_path` (mp3) with edge-tts. Retries once
    with the language default if a curated voice is unknown to the installed
    edge-tts build."""
    text = (text or "").strip()
    if not text:
        raise ValueError("edge-tts: empty text")
    try:
        import edge_tts  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "edge-tts is not installed. Run: pip install edge-tts"
        ) from exc

    chosen = resolve_free_voice(voice, language, gender)
    candidates = [chosen]
    fallback = free_voice_for_language(language, gender)
    if fallback not in candidates:
        candidates.append(fallback)
    if "en-US-AriaNeural" not in candidates:
        candidates.append("en-US-AriaNeural")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    last_error = None
    for attempt_voice in candidates:
        for attempt in range(2):
            try:
                import edge_tts
                communicate = edge_tts.Communicate(
                    text, attempt_voice, rate=rate, pitch=pitch,
                )
                _run_async(communicate.save(output_path))
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    log(f"[edge-tts] {len(text)} chars -> {os.path.basename(output_path)} ({attempt_voice})")
                    return output_path
                raise RuntimeError("edge-tts produced an empty file")
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                log(f"[edge-tts] attempt {attempt + 1} with {attempt_voice} failed: {exc}")
                time.sleep(1.5)
    raise RuntimeError(f"edge-tts failed for all voices: {last_error}")


def probe_free_tts() -> bool:
    try:
        import edge_tts  # noqa: F401
        return True
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════
# Local image generation (SDXL / RealVisXL V4.0)
# ═══════════════════════════════════════════════════════════════════════

_IMAGE_LOCK = threading.RLock()
_PIPE = None
_PIPE_MODEL = None

NEGATIVE_PROMPT = (
    "cartoon, anime, illustration, painting, 3d render, cgi, plastic skin, "
    "deformed hands, extra fingers, extra limbs, distorted face, blurry, "
    "low quality, jpeg artifacts, watermark, text, logo, signature"
)

ACTOR_PROMPT_SUFFIX = (
    "raw candid selfie, photorealistic, natural skin texture, soft window light, "
    "shot on a phone front camera, slightly grainy, realistic proportions, "
    "looking at the camera, authentic social media creator"
)


def _torch():
    import torch
    return torch


def _free_image_model() -> str:
    return os.environ.get("FREE_IMAGE_MODEL", "SG161222/RealVisXL_V4.0")


def _load_image_pipeline(log=print):
    """Load SDXL on the GPU once. Returns None when torch/diffusers/GPU is
    unavailable, which makes the callers use a PIL placeholder instead."""
    global _PIPE, _PIPE_MODEL

    model_id = _free_image_model()
    with _IMAGE_LOCK:
        if _PIPE is not None and _PIPE_MODEL == model_id:
            return _PIPE
        if not model_id:
            log("[free-images] FREE_IMAGE_MODEL is empty; using placeholder images.")
            return None
        try:
            import torch
            from diffusers import StableDiffusionXLPipeline
        except Exception as exc:  # noqa: BLE001
            log(f"[free-images] diffusers/torch unavailable: {exc}")
            return None
        if not torch.cuda.is_available():
            log("[free-images] CUDA not available; using placeholder images.")
            return None

        log(f"[free-images] Loading {model_id} (fp16) on {torch.cuda.get_device_name(0)}...")
        kwargs = {
            "torch_dtype": torch.float16,
            "use_safetensors": True,
        }
        pipe = None
        # RealVisXL publishes fp16 variants; fall back to the base weights if
        # the mirror has changed.
        for variant in ("fp16", None):
            try:
                kw = dict(kwargs)
                if variant:
                    kw["variant"] = variant
                pipe = StableDiffusionXLPipeline.from_pretrained(model_id, **kw)
                break
            except Exception as exc:  # noqa: BLE001
                log(f"[free-images] load variant={variant} failed: {exc}")
        if pipe is None:
            return None

        pipe = pipe.to("cuda")
        try:
            pipe.enable_attention_slicing()
            pipe.enable_vae_slicing()
        except Exception:
            pass
        try:
            pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass
        _PIPE, _PIPE_MODEL = pipe, model_id
        return pipe


def release_image_pipeline(log=print):
    """Drop the SDXL pipeline and return its VRAM. Called before Wav2Lip and
    whisper, both of which need the same 16 GB on a T4."""
    global _PIPE, _PIPE_MODEL
    with _IMAGE_LOCK:
        if _PIPE is None:
            return
        _PIPE = None
        _PIPE_MODEL = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    log("[free-images] Released SDXL VRAM.")


def _placeholder_image(prompt: str, path: str, width: int = 864,
                       height: int = 1536, portrait: bool = False) -> str:
    """A decent gradient card for a machine without CUDA. Not a substitute for
    SDXL, but it keeps the pipeline (and therefore the UI) working."""
    from PIL import Image, ImageDraw, ImageFilter

    seed = int(hashlib.sha256((prompt or "openshorts").encode("utf-8")).hexdigest()[:8], 16)
    rnd = random.Random(seed)
    c1 = (rnd.randint(20, 90), rnd.randint(30, 110), rnd.randint(70, 160))
    c2 = (rnd.randint(120, 230), rnd.randint(80, 180), rnd.randint(40, 150))

    img = Image.new("RGB", (width, height), c1)
    draw = ImageDraw.Draw(img)
    for y in range(height):
        t = y / max(1, height - 1)
        color = tuple(int(c1[i] * (1 - t) + c2[i] * t) for i in range(3))
        draw.line([(0, y), (width, y)], fill=color)

    if portrait:
        # A soft head-and-shoulders silhouette so an actor placeholder still
        # reads as a person in the UI.
        cx, cy = width // 2, int(height * 0.38)
        head_r = int(width * 0.18)
        draw.ellipse([cx - head_r, cy - head_r, cx + head_r, cy + head_r],
                     fill=(235, 220, 210))
        body_w, body_h = int(width * 0.55), int(height * 0.42)
        draw.rounded_rectangle(
            [cx - body_w // 2, cy + head_r + int(height * 0.02),
             cx + body_w // 2, cy + head_r + body_h],
            radius=int(width * 0.12), fill=(30, 35, 48),
        )

    for _ in range(7):
        r = rnd.randint(width // 10, width // 3)
        x, y = rnd.randint(-r, width), rnd.randint(-r, height)
        layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        d.ellipse([x, y, x + r, y + r], fill=(*c2, rnd.randint(12, 40)))
        img = Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")

    img = img.filter(ImageFilter.GaussianBlur(1.2))
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    fmt = "JPEG" if os.path.splitext(path)[1].lower() in (".jpg", ".jpeg") else "PNG"
    img.save(path, fmt)
    return path


def generate_image_free(prompt: str, output_path: str, width: int = 864,
                        height: int = 1536, steps: Optional[int] = None,
                        seed: Optional[int] = None, portrait: bool = False,
                        log=print) -> str:
    """One local SDXL image. Serialised: a T4 cannot hold three SDXL calls and
    diffusers pipelines are not thread-safe."""
    steps = int(os.environ.get("FREE_IMAGE_STEPS", steps or 24))
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with _IMAGE_LOCK:
        pipe = _load_image_pipeline(log=log)
        if pipe is None:
            return _placeholder_image(prompt, output_path, width, height, portrait)

        try:
            import torch
            generator = torch.Generator(device="cuda").manual_seed(
                int(seed if seed is not None else random.randint(0, 2 ** 31 - 1))
            )
            log(f"[free-images] SDXL {width}x{height}, {steps} steps...")
            image = pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=float(os.environ.get("FREE_IMAGE_GUIDANCE", "5.0")),
                generator=generator,
            ).images[0]
            image.save(output_path, "PNG")
            return output_path
        except Exception as exc:  # noqa: BLE001 - OOM must not kill the job
            log(f"[free-images] SDXL failed ({exc}); retrying at a smaller size.")
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                generator = torch.Generator(device="cuda").manual_seed(
                    int(seed if seed is not None else random.randint(0, 2 ** 31 - 1))
                )
                image = pipe(
                    prompt=prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    width=max(512, width * 3 // 4),
                    height=max(512, height * 3 // 4),
                    num_inference_steps=max(12, steps - 6),
                    guidance_scale=5.0,
                    generator=generator,
                ).images[0]
                image.save(output_path, "PNG")
                return output_path
            except Exception as exc2:  # noqa: BLE001
                log(f"[free-images] SDXL retry failed ({exc2}); using placeholder.")
                return _placeholder_image(prompt, output_path, width, height, portrait)


def generate_actor_images_free(description: str, output_dir: str, title_slug: str,
                               num_options: int = 3,
                               product_description: Optional[str] = None,
                               log=print) -> List[str]:
    """Local replacement for fal-ai/flux-2-pro actor portraits."""
    os.makedirs(output_dir, exist_ok=True)
    clean = (description or "a friendly creator in their late twenties").strip()
    # Same intent as the paid prompt: a candid selfie, not a studio portrait.
    if product_description:
        prompt = (
            f"candid phone selfie of {clean}, casually holding a {product_description}, "
            f"showing it to the camera with a natural smile, at home, {ACTOR_PROMPT_SUFFIX}"
        )
    else:
        prompt = (
            f"candid phone selfie of {clean}, sitting at a desk at home, looking at the "
            f"camera with a relaxed smile, headphones around the neck, {ACTOR_PROMPT_SUFFIX}"
        )

    paths = []
    for i in range(max(1, num_options)):
        path = os.path.join(output_dir, f"{title_slug}_actor_option_{i}.png")
        if os.path.exists(path) and os.path.getsize(path) > 0:
            paths.append(path)
            continue
        log(f"[free-images] Actor option {i + 1}/{num_options}...")
        generate_image_free(
            prompt, path, width=864, height=1536,
            seed=random.randint(0, 999_999), portrait=True, log=log,
        )
        paths.append(path)
    return sorted(paths)


def ken_burns_video(image_path: str, output_path: str, duration: float,
                    width: int = 1080, height: int = 1920, fps: int = 30,
                    log=print) -> str:
    """Slow push-in on a still, with silent audio so concat/composite stays
    uniform. This is the same visual the paid b-roll path already used."""
    from ffmpeg_utils import video_encode_args, DELIVERY

    duration = max(1.0, float(duration))
    frames = max(1, int(duration * fps))
    vf = (
        f"scale={width * 2}:{height * 2}:force_original_aspect_ratio=increase,"
        f"crop={width * 2}:{height * 2},"
        f"zoompan=z='1+0.15*on/{frames}':"
        f"x='iw/2-(iw/zoom/2)+10*on/{frames}':"
        f"y='ih/2-(ih/zoom/2)':"
        f"d={frames}:s={width}x{height}:fps={fps},"
        f"setsar=1"
    )
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", image_path,
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
        "-vf", vf,
        "-t", f"{duration:.3f}",
        "-map", "0:v", "-map", "1:a",
        *video_encode_args(DELIVERY),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-shortest",
        output_path,
    ]
    _run(cmd, timeout=600)
    log(f"[free-images] Ken Burns clip: {os.path.basename(output_path)} ({duration:.1f}s)")
    return output_path


def generate_broll_free(prompt: str, output_path: str, duration: str = "5",
                        log=print) -> str:
    """Local replacement for fal-ai/flux-2-pro b-roll: one SDXL still + Ken Burns."""
    dur = max(1.0, float(duration or 5))
    img_path = output_path.replace(".mp4", "_img.png")
    full_prompt = (
        f"{prompt}. Cinematic, shallow depth of field, professional photography, "
        "natural colours, vertical composition, no text, no logos."
    )
    generate_image_free(
        full_prompt, img_path, width=864, height=1536,
        seed=random.randint(0, 999_999), log=log,
    )
    ken_burns_video(img_path, output_path, dur, log=log)
    try:
        os.remove(img_path)
    except OSError:
        pass
    return output_path


# ═══════════════════════════════════════════════════════════════════════
# Wav2Lip: local talking head
# ═══════════════════════════════════════════════════════════════════════

WAV2LIP_REPO = "https://github.com/Rudrabha/Wav2Lip.git"
WAV2LIP_CHECKPOINT_URL = "https://huggingface.co/Nekochu/Wav2Lip/resolve/main/wav2lip_gan.pth"
S3FD_URL = "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth"

_WAV2LIP_LOCK = threading.Lock()
_WAV2LIP_STATE = {"checked": False, "ready": False, "error": ""}


def wav2lip_dir() -> str:
    return os.environ.get("WAV2LIP_DIR") or os.path.join(THIRD_PARTY, "Wav2Lip")


def _patch_file(path: str, replacements: List[tuple]) -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    original = text
    for old, new in replacements:
        if old in text:
            text = text.replace(old, new)
    if text != original:
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)


def _patch_wav2lip(repo: str, log=print) -> None:
    """The upstream repo predates librosa 0.10, torch 2.6 and OpenCV 4.9.
    All patches are idempotent and only touch the known lines."""
    audio_py = os.path.join(repo, "audio.py")
    _patch_file(audio_py, [
        ("librosa.core.load(", "librosa.load("),
        ("librosa.filters.mel(hp.sample_rate, hp.n_fft, n_mels=hp.num_mels,",
         "librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft, n_mels=hp.num_mels,"),
        ("librosa.filters.mel(hp.sample_rate,hp.n_fft, n_mels=hp.num_mels,",
         "librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft, n_mels=hp.num_mels,"),
    ])

    inference_py = os.path.join(repo, "inference.py")
    _patch_file(inference_py, [
        ("args.face.split('.')[1]", "os.path.splitext(args.face)[1][1:]"),
        ("cv2.cv2.ROTATE_90_CLOCKWISE", "cv2.ROTATE_90_CLOCKWISE"),
        ("import torch, face_detection",
         "import torch, face_detection\n"
         "# torch >= 2.6 defaults weights_only=True, which rejects the old Wav2Lip\n"
         "# checkpoints. Scope the compatibility shim to this inference process.\n"
         "if not getattr(torch.load, '_openshorts_compat', False):\n"
         "    _openshorts_torch_load = torch.load\n"
         "    def _openshorts_load(*a, **kw):\n"
         "        kw.setdefault('weights_only', False)\n"
         "        return _openshorts_torch_load(*a, **kw)\n"
         "    _openshorts_load._openshorts_compat = True\n"
         "    torch.load = _openshorts_load"),
    ])

    sfd_py = os.path.join(repo, "face_detection", "detection", "sfd", "sfd_detector.py")
    _patch_file(sfd_py, [
        ("torch.load(path_to_detector)",
         "torch.load(path_to_detector, map_location='cpu', weights_only=False)"),
    ])

    # numpy >= 1.24 removed the np.float alias; a few old utility files use it.
    for rel in ("face_detection/detection/sfd/bbox.py",
                "face_detection/detection/sfd/detect.py"):
        _patch_file(os.path.join(repo, rel), [("np.float)", "float)")])


def ensure_wav2lip(install: bool = True, log=print) -> bool:
    """Clone the repo and fetch the two checkpoints once. Thread-safe and
    cached; safe to call from every job."""
    with _WAV2LIP_LOCK:
        if _WAV2LIP_STATE["checked"]:
            return _WAV2LIP_STATE["ready"]

        repo = wav2lip_dir()
        ckpt = os.environ.get("WAV2LIP_CHECKPOINT") or os.path.join(
            repo, "checkpoints", "wav2lip_gan.pth")
        s3fd = os.path.join(repo, "face_detection", "detection", "sfd", "s3fd.pth")
        ready = os.path.exists(os.path.join(repo, "inference.py")) and \
            os.path.exists(ckpt) and os.path.getsize(ckpt) > 10_000_000

        if not ready and install and not _env_bool("WAV2LIP_DISABLE"):
            try:
                if not os.path.exists(os.path.join(repo, "inference.py")):
                    log("[wav2lip] Cloning the Wav2Lip repository (one time)...")
                    os.makedirs(os.path.dirname(repo), exist_ok=True)
                    _run(["git", "clone", "--depth", "1", WAV2LIP_REPO, repo],
                         timeout=600)
                os.makedirs(os.path.dirname(ckpt), exist_ok=True)
                if not os.path.exists(ckpt) or os.path.getsize(ckpt) < 10_000_000:
                    log("[wav2lip] Downloading wav2lip_gan.pth (~436 MB)...")
                    _download(WAV2LIP_CHECKPOINT_URL, ckpt, timeout=1800)
                os.makedirs(os.path.dirname(s3fd), exist_ok=True)
                if not os.path.exists(s3fd) or os.path.getsize(s3fd) < 1_000_000:
                    log("[wav2lip] Downloading the S3FD face detector (~86 MB)...")
                    _download(S3FD_URL, s3fd, timeout=900)
                _patch_wav2lip(repo, log=log)
                ready = os.path.exists(os.path.join(repo, "inference.py")) and \
                    os.path.getsize(ckpt) > 10_000_000
                if ready:
                    log("[wav2lip] Ready.")
            except Exception as exc:  # noqa: BLE001
                _WAV2LIP_STATE["error"] = str(exc)
                log(f"[wav2lip] Setup failed ({exc}); will use the still-image fallback.")

        _WAV2LIP_STATE["checked"] = True
        _WAV2LIP_STATE["ready"] = bool(ready)
        return bool(ready)


def _still_image_video(image_path: str, audio_path: str, output_path: str,
                       log=print) -> str:
    """Wav2Lip fallback: the actor image with a slow push-in and the real
    voice track. Better than failing the job; subtitles still get burned."""
    from ffmpeg_utils import video_encode_args, DELIVERY

    dur = _duration(audio_path) or 10.0
    fps = 30
    frames = max(1, int(dur * fps))
    vf = (
        "scale=2160:3840:force_original_aspect_ratio=increase,"
        "crop=2160:3840,"
        f"zoompan=z='1+0.06*on/{frames}':"
        "x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
        f"d={frames}:s=1080x1920:fps={fps},setsar=1"
    )
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", image_path,
        "-i", audio_path,
        "-vf", vf,
        "-t", f"{dur:.3f}",
        "-map", "0:v", "-map", "1:a",
        *video_encode_args(DELIVERY),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        output_path,
    ]
    _run(cmd, timeout=900)
    log("[wav2lip] Still-image talking head fallback ready.")
    return output_path


def generate_talking_head_free(image_path: str, audio_path: str, output_path: str,
                               log=print) -> str:
    """Wav2Lip static-image lip-sync. Falls back to a Ken Burns still."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"actor image not found: {image_path}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"voiceover not found: {audio_path}")

    if not ensure_wav2lip(log=log):
        return _still_image_video(image_path, audio_path, output_path, log=log)

    repo = wav2lip_dir()
    ckpt = os.environ.get("WAV2LIP_CHECKPOINT") or os.path.join(
        repo, "checkpoints", "wav2lip_gan.pth")
    work = tempfile.mkdtemp(prefix="openshorts_wav2lip_")
    try:
        face = os.path.join(work, "face.png")
        wav = os.path.join(work, "voice.wav")
        out = os.path.join(work, "result.mp4")
        shutil.copy2(image_path, face)

        # Wav2Lip calls librosa.load; a clean 16 kHz mono wav avoids relying on
        # whichever mp3 backend the runtime has.
        _run([
            "ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1",
            wav,
        ], timeout=300)
        os.makedirs(os.path.join(repo, "temp"), exist_ok=True)

        cmd = [
            sys.executable, "inference.py",
            "--checkpoint_path", ckpt,
            "--face", face,
            "--audio", wav,
            "--outfile", out,
            "--static", "True",
            "--pads", "0", "20", "0", "0",
            "--resize_factor", "1",
            "--nosmooth",
        ]
        log("[wav2lip] Running lip-sync (this takes 1-4 minutes on a T4)...")
        timeout = int(os.environ.get("WAV2LIP_TIMEOUT", "1800"))
        proc = _run(cmd, timeout=timeout, cwd=repo, check=False)
        if proc.returncode != 0 or not os.path.exists(out) or os.path.getsize(out) == 0:
            tail = (proc.stderr or proc.stdout or "")[-1200:]
            log(f"[wav2lip] Inference failed: {tail}")
            return _still_image_video(image_path, audio_path, output_path, log=log)

        shutil.copy2(out, output_path)
        log(f"[wav2lip] Talking head ready: {os.path.basename(output_path)}")
        return output_path
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════
# Free dubbing: whisper -> translate -> edge-tts -> ffmpeg
# ═══════════════════════════════════════════════════════════════════════

# deep-translator / Google uses a handful of codes that differ from ISO 639-1.
_GOOGLE_LANG = {
    "zh": "zh-CN",
    "fil": "tl",
    "he": "iw",
    "pt": "pt",
    "ms": "ms",
}

_LANG_NAMES = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "it": "Italian", "pt": "Portuguese", "pl": "Polish", "hi": "Hindi",
    "ja": "Japanese", "ko": "Korean", "zh": "Chinese", "ar": "Arabic",
    "ru": "Russian", "tr": "Turkish", "nl": "Dutch", "sv": "Swedish",
    "id": "Indonesian", "fil": "Filipino", "ms": "Malay", "vi": "Vietnamese",
    "th": "Thai", "uk": "Ukrainian", "el": "Greek", "cs": "Czech",
    "fi": "Finnish", "ro": "Romanian", "da": "Danish", "bg": "Bulgarian",
    "hr": "Croatian", "sk": "Slovak", "ta": "Tamil",
}


def _translate_with_gemini(lines: List[str], target_language: str,
                           api_key: str, log=print) -> Optional[List[str]]:
    if not api_key or not lines:
        return None
    try:
        from google import genai
        from google.genai import types
    except Exception:
        return None
    model = os.environ.get("GEMINI_MODEL") or "gemini-2.5-flash-lite"
    target = _LANG_NAMES.get(_language_of(target_language), target_language)
    client = genai.Client(api_key=api_key)
    out: List[str] = []
    try:
        for start in range(0, len(lines), 40):
            batch = lines[start:start + 40]
            prompt = (
                f"Translate each subtitle line into {target}. Keep the same order and "
                "the same number of lines. Keep each translation short enough to be "
                "spoken in about the same time as the original. Do not add notes. "
                "Return a JSON array of strings only.\n\n"
                + json.dumps(batch, ensure_ascii=False)
            )
            response = client.models.generate_content(
                model=model,
                contents=[prompt],
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            text = (response.text or "").strip()
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\n?", "", text)
                text = re.sub(r"\n?```$", "", text)
            parsed = json.loads(text)
            if not isinstance(parsed, list) or len(parsed) != len(batch):
                raise ValueError("translation shape mismatch")
            out.extend(str(x) for x in parsed)
        log(f"[free-dub] Gemini translated {len(out)} lines to {target}.")
        return out
    except Exception as exc:  # noqa: BLE001
        log(f"[free-dub] Gemini translation failed ({exc}); falling back to Google.")
        return None


def _translate_with_google(lines: List[str], target_language: str,
                           log=print) -> List[str]:
    try:
        from deep_translator import GoogleTranslator
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Translation needs either a Gemini key or deep-translator. "
            "Install it with: pip install deep-translator"
        ) from exc

    code = _GOOGLE_LANG.get(_language_of(target_language), _language_of(target_language))
    translator = GoogleTranslator(source="auto", target=code)
    out: List[str] = []
    for i, line in enumerate(lines):
        try:
            translated = translator.translate(line)
        except Exception as exc:  # noqa: BLE001
            print(f"[free-dub] Google translate failed on line {i} ({exc}); keeping source.")
            translated = line
        out.append((translated or line).strip())
        time.sleep(0.15)  # be polite to the free endpoint
    log(f"[free-dub] Google translated {len(out)} lines to {code}.")
    return out


def _atempo_to_fit(src: str, dst: str, target_duration: float, log=print) -> float:
    """Speed a TTS segment up (never below 1.0) so it fits its subtitle slot.
    Slowing TTS down sounds worse than letting a short line end early."""
    source_duration = _duration(src)
    tempo = 1.0
    if target_duration > 0 and source_duration > target_duration * 1.05:
        tempo = min(1.45, source_duration / target_duration)
    args = ["ffmpeg", "-y", "-i", src]
    if tempo > 1.01:
        args += ["-filter:a", f"atempo={tempo:.3f}"]
    args += ["-ar", "44100", "-ac", "2", dst]
    _run(args, timeout=300)
    return tempo


def dub_video_free(video_path: str, output_path: str, target_language: str,
                   gemini_key: Optional[str] = None,
                   source_language: Optional[str] = None,
                   log=print) -> str:
    """Replace the clip's voice with translated edge-tts audio.

    Keeps the original timing: every translated segment is placed at the start
    of its source segment, so cuts and on-screen actions still line up.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    from transcribe_backends import transcribe_media, release_models

    log("[free-dub] Transcribing the clip with faster-whisper...")
    try:
        transcript = transcribe_media(video_path)
    finally:
        release_models()

    segments = []
    for seg in transcript.get("segments", []):
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        start = max(0.0, float(seg.get("start") or 0))
        end = max(start + 0.2, float(seg.get("end") or start + 0.2))
        segments.append({"start": start, "end": end, "text": text})
    if not segments:
        raise RuntimeError(
            "No speech was detected in this clip, so there is nothing to dub. "
            "Try a clip with spoken audio."
        )

    lines = [s["text"] for s in segments]
    log(f"[free-dub] Translating {len(lines)} segments to {target_language}...")
    translated = _translate_with_gemini(lines, target_language, gemini_key or "", log=log)
    if translated is None:
        translated = _translate_with_google(lines, target_language, log=log)

    work = tempfile.mkdtemp(prefix="openshorts_dub_")
    try:
        wav_paths: List[str] = []
        placements: List[tuple] = []
        for i, (seg, text) in enumerate(zip(segments, translated)):
            mp3 = os.path.join(work, f"seg_{i:03d}.mp3")
            wav = os.path.join(work, f"seg_{i:03d}.wav")
            try:
                edge_tts_save(
                    text, mp3, language=target_language,
                    gender=os.environ.get("FREE_DUB_VOICE_GENDER", "female"),
                    log=log,
                )
                tempo = _atempo_to_fit(mp3, wav, seg["end"] - seg["start"], log=log)
                wav_paths.append(wav)
                placements.append((int(seg["start"] * 1000), tempo))
            except Exception as exc:  # noqa: BLE001
                log(f"[free-dub] Segment {i} skipped: {exc}")

        if not wav_paths:
            raise RuntimeError("edge-tts could not synthesise any translated segment")

        # Inputs: 0 = video, 1..N = segment wavs.
        inputs = ["-i", video_path]
        for wav in wav_paths:
            inputs += ["-i", wav]

        filters = []
        mix_inputs = []
        for i, (delay_ms, _tempo) in enumerate(placements, start=1):
            filters.append(
                f"[{i}:a]aresample=44100,adelay={delay_ms}|{delay_ms},"
                f"volume=1.0[d{i}]"
            )
            mix_inputs.append(f"[d{i}]")
        filters.append(
            "".join(mix_inputs)
            + f"amix=inputs={len(mix_inputs)}:normalize=0,"
              "alimiter=limit=0.95,apad[dub]"
        )

        dur = _duration(video_path)
        cmd = [
            "ffmpeg", "-y", *inputs,
            "-filter_complex", ";".join(filters),
            "-map", "0:v:0", "-map", "[dub]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        ]
        if dur > 0:
            cmd += ["-t", f"{dur:.3f}"]
        cmd += ["-shortest", output_path]
        try:
            _run(cmd, timeout=1800)
        except RuntimeError:
            # A source with an unusual codec cannot be stream-copied into mp4.
            from ffmpeg_utils import video_encode_args, DELIVERY
            cmd = [
                "ffmpeg", "-y", *inputs,
                "-filter_complex", ";".join(filters),
                "-map", "0:v:0", "-map", "[dub]",
                *video_encode_args(DELIVERY),
                "-c:a", "aac", "-b:a", "192k",
            ]
            if dur > 0:
                cmd += ["-t", f"{dur:.3f}"]
            cmd += ["-shortest", output_path]
            _run(cmd, timeout=1800)
        log(f"[free-dub] Dubbed video ready: {os.path.basename(output_path)}")
        return output_path
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════
# Free thumbnails: real frame + PIL text
# ═══════════════════════════════════════════════════════════════════════

_STOPWORDS = {
    "the", "and", "that", "this", "with", "from", "your", "you", "for", "are",
    "was", "were", "have", "has", "had", "but", "not", "all", "can", "will",
    "just", "like", "about", "what", "when", "where", "how", "why", "who",
    "into", "than", "then", "them", "they", "our", "out", "get", "got", "its",
    "it's", "i'm", "don't", "doesn't", "didn't", "isn't", "very", "really",
    "que", "los", "las", "una", "uno", "por", "para", "con", "del", "como",
    "pero", "más", "este", "esta", "esto", "eso", "sus", "mis", "tus",
    "yang", "dan", "itu", "ini", "dengan", "untuk", "dari", "anda", "kamu",
    "saya", "aku", "tidak", "ada", "juga", "akan", "bisa", "sudah", "karena",
}


def _sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?。！？])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _clean_sentence(text: str) -> str:
    text = re.sub(r"^(so|and|but|okay|ok|well|you know|um+|uh+|like|right|"
                  r"bueno|pues|vale|entonces|jadi|oke|nah|terus)\b[, ]*",
                  "", text.strip(), flags=re.I)
    return text.strip(" -–—,;:")


def _keywords(text: str, limit: int = 8) -> List[str]:
    words = re.findall(r"[A-Za-zÀ-ÿ0-9']{3,}", text or "")
    counts: Dict[str, int] = {}
    for w in words:
        low = w.lower()
        if low in _STOPWORDS or low.isdigit():
            continue
        counts[low] = counts.get(low, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], -len(kv[0])))
    return [w for w, _ in ranked[:limit]]


def _title_from_text(text: str, max_chars: int = 65) -> str:
    text = _clean_sentence(text)
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip(" ,.;:-") + "…"


def _thumbnail_words(text: str, limit: int = 3) -> str:
    words = _keywords(text, limit=limit)
    if words:
        return " ".join(w.upper() for w in words[:limit])
    words = re.findall(r"\S+", text or "")
    return " ".join(w.upper() for w in words[:limit])[:40]


def free_title_data(transcript: dict, count: int = 10) -> dict:
    """Deterministic titles/thumbnail texts from the transcript. No API key.

    It cannot be as sharp as Gemini, but it is specific to the video, keeps the
    source language, and gives the user an editable starting point.
    """
    segments = transcript.get("segments", [])
    full_text = transcript.get("text", "") or " ".join(
        s.get("text", "") for s in segments
    )
    sentences = [_clean_sentence(s) for s in _sentences(full_text)]
    sentences = [s for s in sentences if len(s) >= 12]
    summary = " ".join(sentences[:3])[:600] or (full_text or "")[:600]

    candidates: List[str] = []
    seen = set()
    for s in sentences:
        title = _title_from_text(s)
        key = title.lower()
        if title and key not in seen and 12 <= len(title) <= 70:
            candidates.append(title)
            seen.add(key)
    # Prefer sentences with numbers and concrete nouns, but keep timeline order.
    def score(t: str) -> tuple:
        has_number = bool(re.search(r"\d", t))
        return (0 if has_number else 1, -len(t))
    candidates = sorted(candidates, key=score)

    titles: List[str] = []
    for t in candidates:
        titles.append(t)
        if len(titles) >= count:
            break
    # Not enough sentences: shorten/deduplicate fragments rather than inventing
    # an English title for a non-English video.
    frag_index = 0
    while len(titles) < count and sentences:
        base = sentences[frag_index % len(sentences)]
        words = base.split()
        take = max(3, len(words) - (frag_index // max(1, len(sentences))) - 1)
        frag = _title_from_text(" ".join(words[:take]))
        if frag and frag.lower() not in seen:
            titles.append(frag)
            seen.add(frag.lower())
        frag_index += 1
        if frag_index > count * 6:
            break
    if not titles:
        titles = ["Untitled video"]

    thumbnail_texts = [_thumbnail_words(t) for t in titles]
    recommended = [{"index": 0, "reason": "Most concrete statement in the transcript."}]
    if len(titles) > 1:
        recommended.append({"index": 1, "reason": "Second clearest hook."})

    return {
        "titles": titles[:count],
        "thumbnail_texts": thumbnail_texts[:count],
        "transcript_summary": summary,
        "language": transcript.get("language", "en"),
        "segments": segments,
        "video_duration": segments[-1]["end"] if segments else 0,
        "recommended": recommended,
    }


def free_refine_titles(context: str, user_message: str, count: int = 8) -> dict:
    """Refine without an LLM: keep the transcript context, use the user's
    message as a direction when it already looks like a title, and return
    trimmed variants. The UI treats these like any other suggestion."""
    message = (user_message or "").strip()
    sentences = [_clean_sentence(s) for s in _sentences(context or "")]
    titles: List[str] = []
    seen = set()

    def add(t: str):
        t = _title_from_text(t)
        if t and t.lower() not in seen and len(titles) < count:
            titles.append(t)
            seen.add(t.lower())

    # A short message is usually the new title itself ("make it about X" is not).
    if 8 <= len(message) <= 70 and not re.match(
            r"(?i)^(make|write|give|try|use|change|shorten|another|more|less|the)\b", message):
        add(message)
    for s in sentences:
        add(s)
        if len(titles) >= count:
            break
    # Add specific keyword-led variants so a short context still fills the grid.
    for kw in _keywords(context, limit=count):
        add(f"{kw.capitalize()}: {sentences[0] if sentences else message}")
        if len(titles) >= count:
            break

    if not titles:
        titles = [message[:65] or "Untitled video"]
    return {
        "titles": titles,
        "thumbnail_texts": [_thumbnail_words(t) for t in titles],
        "language": "",
    }


def free_youtube_description(title: str, segments: List[dict],
                             language: str = "en", video_duration: float = 0) -> dict:
    """Description with chapters, built from the transcript without Gemini."""
    texts = [(_clean_sentence(s.get("text", ""))) for s in segments]
    summary = " ".join(t for t in texts[:3] if t)[:700]
    lines = [summary or title, ""]

    chapters = []
    last_mark = -999
    for seg in segments:
        start = float(seg.get("start") or 0)
        text = _clean_sentence(seg.get("text", ""))
        if not text:
            continue
        if start == 0 or start - last_mark >= 90:
            label = _title_from_text(text, max_chars=48)
            if label:
                chapters.append((int(start), label))
                last_mark = start
    if not chapters:
        chapters = [(0, title[:48] or "Intro")]
    lines.append("Chapters:")
    for sec, label in chapters:
        lines.append(f"{sec // 60}:{sec % 60:02d} {label}")
    lines.append("")
    tags = _keywords(" ".join(texts), limit=6)
    if tags:
        lines.append(" ".join("#" + re.sub(r"[^0-9A-Za-zÀ-ÿ_]", "", t) for t in tags if t))
    return {"description": "\n".join(lines).strip()}


def _cover(img, width: int, height: int):
    from PIL import Image
    img = img.convert("RGB")
    scale = max(width / img.width, height / img.height)
    img = img.resize((int(img.width * scale + 0.5), int(img.height * scale + 0.5)),
                     Image.LANCZOS)
    left = (img.width - width) // 2
    top = (img.height - height) // 2
    return img.crop((left, top, left + width, top + height))


def _scrim(img, position: str, strength: int = 170):
    """Darken the side where the headline goes; frames are usually busy."""
    from PIL import Image, ImageDraw
    width, height = img.size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    steps = 80
    if position in ("left", "right"):
        span = int(width * 0.62)
        for i in range(steps):
            alpha = int(strength * (1 - i / steps) ** 1.6)
            x0 = int(span * i / steps)
            x1 = int(span * (i + 1) / steps)
            if position == "left":
                draw.rectangle([x0, 0, x1, height], fill=alpha)
            else:
                draw.rectangle([width - x1, 0, width - x0, height], fill=alpha)
    else:
        span = int(height * 0.55)
        for i in range(steps):
            alpha = int(strength * (1 - i / steps) ** 1.6)
            y0 = int(span * i / steps)
            y1 = int(span * (i + 1) / steps)
            if position == "top":
                draw.rectangle([0, y0, width, y1], fill=alpha)
            else:
                draw.rectangle([0, height - y1, width, height - y0], fill=alpha)
    black = Image.new("RGB", (width, height), (0, 0, 0))
    return Image.composite(black, img, mask)


def _gradient_thumbnail(text: str, width: int = 1280, height: int = 720):
    from PIL import Image, ImageDraw
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
    rnd = random.Random(seed)
    c1 = (rnd.randint(15, 55), rnd.randint(25, 70), rnd.randint(60, 120))
    c2 = (rnd.randint(120, 220), rnd.randint(45, 120), rnd.randint(30, 90))
    img = Image.new("RGB", (width, height), c1)
    draw = ImageDraw.Draw(img)
    for x in range(width):
        t = x / max(1, width - 1)
        draw.line([(x, 0), (x, height)],
                  fill=tuple(int(c1[i] * (1 - t) + c2[i] * t) for i in range(3)))
    return img


def generate_thumbnails_free(title: str, session_id: str, count: int = 3,
                             video_path: Optional[str] = None,
                             face_image_path: Optional[str] = None,
                             bg_image_path: Optional[str] = None,
                             frame_reference: Optional[dict] = None,
                             thumbnail_text_hint: str = "",
                             burn_text: bool = True,
                             log=print) -> List[dict]:
    """Compose thumbnails from real frames + burned text (no image model)."""
    from PIL import Image, ImageEnhance

    # Imported lazily: thumbnail.py imports this module for its free fallbacks.
    from thumbnail import burn_thumbnail_text, finalize_thumbnail

    out_dir = os.path.join("output", "thumbnails", session_id)
    os.makedirs(out_dir, exist_ok=True)
    batch = hashlib.md5(f"{title}-{time.time()}".encode()).hexdigest()[:6]

    sources: List[tuple] = []  # (PIL image, why)
    def add_source(path, why):
        try:
            if path and os.path.exists(path):
                sources.append((Image.open(path), why))
        except Exception as exc:  # noqa: BLE001
            log(f"[free-thumb] Could not open {path}: {exc}")

    if face_image_path:
        add_source(face_image_path, "your uploaded face")
    if bg_image_path:
        add_source(bg_image_path, "your uploaded background")
    if frame_reference and frame_reference.get("path"):
        add_source(frame_reference["path"], "the frame you picked")

    if video_path and os.path.exists(video_path):
        try:
            from layout_picker import sample_frames
            blobs = sample_frames(video_path, n=max(count * 3, 6), width=1280)
            for i, blob in enumerate(blobs):
                import io
                sources.append((Image.open(io.BytesIO(blob)),
                                f"video frame {i + 1}"))
        except Exception as exc:  # noqa: BLE001
            log(f"[free-thumb] Frame sampling failed: {exc}")

    # Spread picks across the runtime rather than taking the first N.
    if len(sources) > count:
        step = len(sources) / float(count)
        sources = [sources[min(len(sources) - 1, int(i * step))] for i in range(count)]
    while len(sources) < count:
        sources.append((None, "generated background"))

    positions = ("left", "right", "top", "bottom")
    colors = ("yellow", "white")
    hint = (thumbnail_text_hint or "").strip().upper()
    base_text = hint or _thumbnail_words(title)

    results = []
    for i in range(count):
        img, why = sources[i]
        if img is None:
            pil = _gradient_thumbnail(title)
            why = "generated background"
        else:
            pil = _cover(img, 1280, 720)
            pil = ImageEnhance.Contrast(pil).enhance(1.06)
            pil = ImageEnhance.Color(pil).enhance(1.08)
            pos = positions[i % len(positions)]
            pil = _scrim(pil, pos, strength=175)
        pos = positions[i % len(positions)]
        color = colors[i % len(colors)]
        text = base_text
        if i == 1 and base_text:
            words = base_text.split()
            if len(words) > 2:
                text = " ".join(words[:2])
        if burn_text:
            pil = burn_thumbnail_text(pil, text, pos, color)
        out_path = os.path.join(out_dir, f"thumb_{batch}_{i + 1}.jpg")
        # The base is a real frame and the text is drawn, not model-generated:
        # no AI provenance tag is warranted here.
        finalize_thumbnail(pil, out_path, ai_generated=False)
        results.append({
            "url": f"/thumbnails/{session_id}/{os.path.basename(out_path)}",
            "text": text,
            "why": f"Composed from {why} with burned-in text.",
            "fallback": False,
        })
        log(f"[free-thumb] {os.path.basename(out_path)} <- {why}")

    return results


def free_thumbnail_concepts(client, title: str, count: int,
                            thumbnail_text_hint: str = "") -> List[dict]:
    """Concept objects shaped like thumbnail.normalise_concepts output. Used
    when no Gemini key is configured; the renderer only reads text/position."""
    base = (thumbnail_text_hint or "").strip().upper()[:40] or _thumbnail_words(title)
    positions = ("left", "right", "top", "bottom")
    out = []
    for i in range(max(1, count)):
        text = base
        if i == 1 and base:
            words = base.split()
            if len(words) > 2:
                text = " ".join(words[:2])
        out.append({
            "text": text,
            "text_position": positions[i % len(positions)],
            "text_color": "yellow" if i % 2 == 0 else "white",
            "scene": "real video frame with a dark scrim and a bold headline",
            "why": "Free mode: composed from an actual frame, so the thumbnail shows the real video.",
        })
    return out


# ═══════════════════════════════════════════════════════════════════════
# Free AI Shorts fallback (analysis + scripts without a Gemini key)
# ═══════════════════════════════════════════════════════════════════════

_ACTOR_DESCRIPTIONS = {
    "female": [
        "a 26 year old European woman with light brown wavy hair, wearing a simple white top, natural minimal makeup, friendly face",
        "a 31 year old Latina woman with dark curly hair, wearing a mustard cardigan, warm smile, casual everyday look",
        "a 24 year old Asian woman with straight black hair, wearing a beige t-shirt, soft natural light, relaxed expression",
    ],
    "male": [
        "a 29 year old European man with short dark hair and light stubble, wearing a navy t-shirt, smart casual look",
        "a 33 year old Black man with a short fade and glasses, wearing an olive overshirt, friendly expression",
        "a 25 year old Asian man with medium black hair, wearing a grey hoodie, relaxed casual look",
    ],
}

_PAIN_FALLBACK = {
    "en": [
        "wasting hours every week on work that should take minutes",
        "paying for too many tools that still do not talk to each other",
        "losing track of tasks across spreadsheets, notes and chat",
    ],
    "es": [
        "perder horas cada semana en tareas que deberían llevar minutos",
        "pagar demasiadas herramientas que ni siquiera se conectan entre sí",
        "perder el hilo de todo entre hojas de cálculo, notas y chats",
    ],
    "id": [
        "buang-buang waktu berjam-jam tiap minggu untuk kerjaan yang harusnya selesai dalam hitungan menit",
        "bayar terlalu banyak tools yang ujung-ujungnya nggak saling terhubung",
        "kehilangan jejak pekerjaan karena tersebar di spreadsheet, catatan, dan chat",
    ],
}


def free_analysis(scraped: Optional[dict] = None,
                  description: Optional[str] = None,
                  url: Optional[str] = None) -> dict:
    """A useful analysis object without any API call. Pulls the product name,
    one-liner, headings and features straight from the scraped page."""
    scraped = scraped or {}
    description = (description or "").strip()
    title = (scraped.get("title") or "").strip()
    meta = (scraped.get("meta_description") or "").strip()
    headings = [h for h in scraped.get("headings", []) if h][:8]
    text = " ".join([title, meta, description] + headings)

    domain = ""
    if url:
        try:
            domain = url.replace("https://", "").replace("http://", "").split("/")[0]
        except Exception:
            domain = ""
    product_name = (title.split("|")[0].split("–")[0].strip()
                    or description.split(".")[0][:60]
                    or domain or "Your product")
    one_liner = meta or description[:220] or f"{product_name} helps you get more done with less busywork."

    pains = []
    for i, pain in enumerate(_PAIN_FALLBACK["en"]):
        pains.append({
            "pain": pain,
            "intensity": "high" if i == 0 else "medium",
            "emotional_trigger": "frustration" if i != 1 else "money-loss",
            "source": "general",
        })

    return {
        "product_name": product_name,
        "one_liner": one_liner,
        "description": description or one_liner,
        "value_proposition": one_liner,
        "target_audience": ["founders", "small teams", "creators"],
        "pain_points": pains,
        "key_features": headings[:5] or [one_liner],
        "unique_selling_points": [
            "faster than the manual way",
            "one place instead of five tabs",
            "simple enough to use on day one",
        ],
        "competitors": [],
        "pricing_model": "freemium",
        "pricing_details": "",
        "industry": "software",
        "user_sentiment_summary": "",
        "emotional_hooks": [
            "Stop wasting hours on this.",
            "I wish I knew about this sooner.",
            "Your competitors are already doing this.",
        ],
        "transformation_story": f"Before: busywork. After: {product_name} handles it.",
        "viral_angles": [
            {"angle": "The hidden time cost", "platform": "both", "style": "ugc",
             "why_viral": "Everyone recognises the wasted hours."},
        ],
        "tone": "casual and authentic",
        "_free_fallback": True,
        "_product_url": url or "",
    }


_SCRIPT_TEMPLATES = {
    "en": {
        "hook": "Okay, I just found something that completely changed how I work.",
        "problem": "My problem was {pain}, and honestly it was exhausting.",
        "solution": "{product} does the heavy lifting for me in minutes, without the usual headache.",
        "demo": "It is genuinely simple: you open it and everything is already organised.",
        "cta": "If you are done wasting time, the link is in my bio.",
        "caption": "The tool I wish I had found sooner. {product} saves me hours every week.",
    },
    "es": {
        "hook": "Vale, acabo de encontrar algo que me ha cambiado la forma de trabajar.",
        "problem": "Mi problema era {pain}, y la verdad es que era agotador.",
        "solution": "{product} hace el trabajo pesado por mí en minutos, sin complicaciones.",
        "demo": "Es muy fácil: lo abres y todo ya está organizado.",
        "cta": "Si estás harto de perder tiempo, el enlace está en mi bio.",
        "caption": "La herramienta que ojalá haber encontrado antes. {product} me ahorra horas cada semana.",
    },
    "id": {
        "hook": "Oke, aku baru nemu sesuatu yang bener-bener ngubah cara kerjaku.",
        "problem": "Dulu aku {pain}, dan jujur itu capek banget.",
        "solution": "{product} ngerjain bagian beratnya buat aku dalam hitungan menit, tanpa ribet.",
        "demo": "Simpel banget: tinggal buka, semuanya udah rapi.",
        "cta": "Kalau kamu capek buang waktu, link-nya ada di bio.",
        "caption": "Tools yang aku harap nemu dari dulu. {product} hemat waktuku tiap minggu.",
    },
}

_BROLL_PROMPTS = [
    "close-up of a frustrated person at a messy desk, laptop with cluttered spreadsheets, warm evening light, shallow depth of field",
    "over-the-shoulder shot of a clean modern dashboard on a laptop, soft office light, plants in the background, cinematic",
    "person confidently closing a laptop and smiling, tidy desk, morning light through a window, shallow depth of field",
    "minimalist phone screen showing a simple organised task list, held in one hand, blurred cafe background, cinematic",
]


def free_scripts(analysis: dict, num_scripts: int = 3, style: str = "ugc",
                 language: str = "en", actor_gender: str = "female") -> List[dict]:
    """Five-segment UGC scripts that match the shape saasshorts expects."""
    lang = language if language in _SCRIPT_TEMPLATES else "en"
    tpl = _SCRIPT_TEMPLATES[lang]
    product = (analysis.get("product_name") or "this tool").strip()
    pains = [p.get("pain", "") for p in analysis.get("pain_points", []) if p.get("pain")]
    # free_analysis ships English pain points for every language; when that is
    # the source, use the pain list for the requested script language instead.
    if not pains or analysis.get("_free_fallback"):
        pains = _PAIN_FALLBACK[lang]
    actors = _ACTOR_DESCRIPTIONS.get(
        "male" if (actor_gender or "").lower().startswith("m") else "female"
    )
    hashtags = ["#saas", "#productivity", "#techtools", "#ugc", "#shorts"]

    def _speech_seconds(text: str) -> float:
        words = len(re.findall(r"\S+", text or ""))
        return max(2.5, words / 2.6)

    scripts = []
    for i in range(max(1, num_scripts)):
        pain = pains[i % len(pains)]
        # Pain templates are phrased "my problem was <pain>" / "dulu aku <pain>";
        # trim a trailing period so the sentence flows.
        pain_text = pain.rstrip(".")
        narration = {
            "hook": tpl["hook"],
            "problem": tpl["problem"].format(pain=pain_text),
            "solution": tpl["solution"].format(product=product),
            "demo": tpl["demo"],
            "cta": tpl["cta"],
        }
        # Time the segments from the actual narration length instead of a fixed
        # 24s grid, so the b-roll inserts still land on the right sentence when
        # edge-tts speaks faster or slower than expected.
        plan = [
            ("hook", "actor_talking", None, "excited", "I found something"),
            ("problem", "broll", _BROLL_PROMPTS[0], "frustrated", "The old way"),
            ("solution", "actor_talking", None, "confident", "The fix"),
            ("demo", "broll", _BROLL_PROMPTS[3], "excited", "So simple"),
            ("cta", "actor_talking", None, "confident", "Link in bio"),
        ]
        segments = []
        cursor = 0.0
        for kind, visual, broll_prompt, emotion, subtitle_text in plan:
            dur = _speech_seconds(narration[kind])
            start = round(cursor, 2)
            end = round(cursor + dur, 2)
            segments.append({
                "type": kind, "start": start, "end": end,
                "narration": narration[kind], "visual": visual,
                "broll_prompt": broll_prompt, "emotion": emotion,
                "subtitle_text": subtitle_text,
            })
            cursor += dur
        scripts.append({
            "title": f"{product} - {style} #{i + 1}",
            "style": style,
            "duration_seconds": int(round(cursor)),
            "target_platform": "tiktok",
            "hook_text": "I found something",
            "segments": segments,
            "full_narration": " ".join(s["narration"] for s in segments),
            "actor_description": actors[i % len(actors)],
            "hashtags": hashtags,
            "caption": tpl["caption"].format(product=product),
            "language": lang,
            "_product_name": product,
        })
    return scripts
