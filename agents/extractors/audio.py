"""
agents/extractors/audio.py — Whisper-based audio & video transcription.

Execution path (each step tried in order, first success wins):
  1. faster-whisper (local, CPU int8, zero cost)
  2. Replicate API → openai/whisper large-v3 (needs REPLICATE_API_KEY)

Video pre-processing (mp4 / mkv / webm / mov / avi):
  ffmpeg extracts a 16 kHz mono WAV before transcription.
  If ffmpeg is absent, the raw bytes are passed directly to Whisper
  (works for many mp4/webm containers that faster-whisper can decode).

Async interface (preferred in async callers):
    extractor = WhisperAudioExtractor()
    text = await extractor.async_extract(audio_bytes, url="conference.mp3")

Sync interface (kept for factory.extract() compatibility):
    text = WhisperAudioExtractor().extract(audio_bytes)

Model size is controlled by SCOUT_WHISPER_MODEL env var (default "base"):
  tiny   ~39 MB  — very fast, lower accuracy
  base   ~74 MB  — good balance             ← default
  small  ~244 MB — better accuracy
  medium ~769 MB — high accuracy
  large-v3 ~1.5 GB — best accuracy, needs ≥4 GB RAM

Segment-level language detection is returned in the ExtractResult namedtuple
so callers can populate ScoutSource.language.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import NamedTuple, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Format sets
# ---------------------------------------------------------------------------

_VIDEO_EXTS = frozenset({".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v", ".wmv"})
_AUDIO_EXTS = frozenset({".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".opus", ".wma"})
_ALL_MEDIA_EXTS = _VIDEO_EXTS | _AUDIO_EXTS

# Replicate model version for Whisper large-v3 (pinned for reproducibility)
_REPLICATE_WHISPER_VERSION = (
    "3ab86df6c8f54c11309d4d1f930ac292bad43ace52d10c80d87eb258b3c9f79c"
)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


class ExtractResult(NamedTuple):
    text: str
    language: str          # BCP-47 code detected by Whisper, "" if unknown
    duration_s: float      # audio duration in seconds, 0.0 if unknown
    backend: str           # "faster_whisper" | "replicate" | "none"


# ---------------------------------------------------------------------------
# Helper: infer suffix from URL or default to .wav
# ---------------------------------------------------------------------------


def _infer_suffix(url: str) -> str:
    """Return lower-case file extension from URL, or '.wav' as fallback."""
    try:
        path = Path(url.split("?")[0])
        ext = path.suffix.lower()
        if ext in _ALL_MEDIA_EXTS:
            return ext
    except Exception:
        pass
    return ".wav"


# ---------------------------------------------------------------------------
# Helper: check ffmpeg availability (cached after first call)
# ---------------------------------------------------------------------------

_FFMPEG_OK: Optional[bool] = None


def _has_ffmpeg() -> bool:
    global _FFMPEG_OK
    if _FFMPEG_OK is None:
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                timeout=5,
            )
            _FFMPEG_OK = True
        except Exception:
            _FFMPEG_OK = False
    return _FFMPEG_OK


# ---------------------------------------------------------------------------
# Helper: extract audio from video via ffmpeg (sync, for to_thread)
# ---------------------------------------------------------------------------


def _extract_audio_ffmpeg(video_path: str, wav_path: str) -> bool:
    """
    Extract 16 kHz mono PCM WAV from a video file using ffmpeg.
    Returns True on success. Never raises.
    """
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vn",                    # no video
                "-acodec", "pcm_s16le",   # 16-bit PCM
                "-ar", "16000",           # 16 kHz sample rate (Whisper native)
                "-ac", "1",               # mono
                wav_path,
            ],
            capture_output=True,
            timeout=300,                  # 5-min max for long recordings
        )
        if result.returncode != 0:
            logger.debug(
                "[audio] ffmpeg failed (rc=%d): %s",
                result.returncode,
                result.stderr[-500:].decode("utf-8", errors="replace"),
            )
            return False
        return os.path.exists(wav_path) and os.path.getsize(wav_path) > 0
    except FileNotFoundError:
        logger.debug("[audio] ffmpeg not found on PATH")
        return False
    except subprocess.TimeoutExpired:
        logger.warning("[audio] ffmpeg timed out after 300s")
        return False
    except Exception as exc:
        logger.debug("[audio] ffmpeg error: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Backend 1: faster-whisper (sync, for to_thread)
# ---------------------------------------------------------------------------


def _transcribe_faster_whisper(audio_path: str) -> Optional[ExtractResult]:
    """
    Transcribe audio_path with faster-whisper.
    Returns ExtractResult or None if library unavailable / failed.
    """
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError:
        logger.debug("[audio] faster-whisper not installed (pip install faster-whisper)")
        return None

    try:
        from config.settings import settings
        model_size = getattr(settings, "scout_whisper_model", "base")
    except Exception:
        model_size = "base"

    try:
        model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",   # quantized — minimal RAM, fast on CPU
        )
        segments, info = model.transcribe(
            audio_path,
            beam_size=5,
            vad_filter=True,       # skip silence — reduces hallucinations
            vad_parameters={"min_silence_duration_ms": 500},
        )
        text_parts: list[str] = []
        for seg in segments:
            t = seg.text.strip()
            if t:
                text_parts.append(t)
        text = " ".join(text_parts).strip()
        lang = getattr(info, "language", "") or ""
        duration = getattr(info, "duration", 0.0) or 0.0
        logger.info(
            "[audio] faster-whisper: lang=%s duration=%.1fs chars=%d",
            lang, duration, len(text),
        )
        return ExtractResult(text=text, language=lang, duration_s=duration, backend="faster_whisper")
    except Exception as exc:
        logger.warning("[audio] faster-whisper transcription error: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Backend 2: Replicate API (async)
# ---------------------------------------------------------------------------


async def _transcribe_replicate(audio_bytes: bytes) -> Optional[ExtractResult]:
    """
    Transcribe via Replicate's hosted Whisper large-v3.
    Requires REPLICATE_API_KEY in settings. Never raises.
    """
    try:
        from config.settings import settings
        api_key = getattr(settings, "replicate_api_key", "")
        if not api_key:
            logger.debug("[audio] REPLICATE_API_KEY not set — skipping Replicate fallback")
            return None
    except Exception:
        return None

    try:
        import httpx

        b64 = base64.b64encode(audio_bytes).decode()
        data_uri = f"data:audio/wav;base64,{b64}"

        async with httpx.AsyncClient(timeout=300.0) as client:
            # Create prediction
            create_resp = await client.post(
                "https://api.replicate.com/v1/predictions",
                headers={
                    "Authorization": f"Token {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "version": _REPLICATE_WHISPER_VERSION,
                    "input": {
                        "audio": data_uri,
                        "model": "large-v3",
                        "language": "auto",
                        "transcription": "plain text",
                        "translate": False,
                    },
                },
            )
            if create_resp.status_code not in (200, 201):
                logger.warning(
                    "[audio] Replicate create prediction failed: %d",
                    create_resp.status_code,
                )
                return None

            pred = create_resp.json()
            pred_id = pred.get("id", "")
            poll_url = f"https://api.replicate.com/v1/predictions/{pred_id}"

            # Poll until done (max 10 min)
            for _ in range(120):
                await asyncio.sleep(5)
                poll_resp = await client.get(
                    poll_url,
                    headers={"Authorization": f"Token {api_key}"},
                )
                if poll_resp.status_code != 200:
                    continue
                state = poll_resp.json()
                status = state.get("status", "")
                if status == "succeeded":
                    output = state.get("output", "")
                    if isinstance(output, list):
                        output = " ".join(output)
                    text = (output or "").strip()
                    logger.info(
                        "[audio] Replicate Whisper: chars=%d", len(text),
                    )
                    return ExtractResult(
                        text=text, language="", duration_s=0.0, backend="replicate"
                    )
                if status in ("failed", "canceled"):
                    logger.warning(
                        "[audio] Replicate prediction %s: %s",
                        pred_id, status,
                    )
                    return None

        logger.warning("[audio] Replicate prediction timed out")
        return None
    except Exception as exc:
        logger.warning("[audio] Replicate API error: %s", exc)
        return None


# ---------------------------------------------------------------------------
# WhisperAudioExtractor — the class wired into ExtractorFactory
# ---------------------------------------------------------------------------


class WhisperAudioExtractor:
    """
    Transcribes audio and video files.

    Implements both the sync extract() interface (for ExtractorFactory)
    and async async_extract() (for use in async callers like _process_domain).
    """

    # ------------------------------------------------------------------
    # Sync interface (BaseExtractor compatible)
    # ------------------------------------------------------------------

    def extract(self, content: bytes | str) -> str:
        """
        Synchronous transcription — blocks the calling thread.
        Use async_extract() in async code to avoid blocking the event loop.
        """
        if isinstance(content, str):
            content = content.encode("utf-8", errors="replace")
        suffix = ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            result = _transcribe_faster_whisper(tmp_path)
            if result:
                return result.text
            logger.warning(
                "[audio] No transcription backend available. "
                "Install faster-whisper or set REPLICATE_API_KEY."
            )
            return ""
        finally:
            _safe_unlink(tmp_path)

    # ------------------------------------------------------------------
    # Async interface (preferred)
    # ------------------------------------------------------------------

    async def async_extract(
        self,
        content: bytes | str,
        *,
        url: str = "",
    ) -> ExtractResult:
        """
        Async transcription — CPU work runs in a thread pool via asyncio.to_thread.

        Args:
            content: raw audio or video bytes
            url:     source URL — used to infer file format/extension
        """
        if isinstance(content, str):
            content = content.encode("utf-8", errors="replace")

        suffix = _infer_suffix(url)
        is_video = suffix in _VIDEO_EXTS

        # Write to temp file
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            input_path = tmp.name

        audio_path = input_path
        wav_path: Optional[str] = None

        try:
            # Video: extract audio track first
            if is_video:
                if _has_ffmpeg():
                    wav_path = input_path + ".wav"
                    ok = await asyncio.to_thread(_extract_audio_ffmpeg, input_path, wav_path)
                    if ok:
                        audio_path = wav_path
                    else:
                        # Pass raw video bytes directly to Whisper (may work for mp4/webm)
                        logger.debug(
                            "[audio] ffmpeg extraction failed — passing raw %s to Whisper",
                            suffix,
                        )
                else:
                    logger.debug("[audio] ffmpeg unavailable — passing raw %s to Whisper", suffix)

            # Backend 1: faster-whisper (thread pool)
            result = await asyncio.to_thread(_transcribe_faster_whisper, audio_path)
            if result:
                return result

            # Backend 2: Replicate API (async HTTP)
            with open(audio_path, "rb") as fh:
                audio_bytes = fh.read()
            replicate_result = await _transcribe_replicate(audio_bytes)
            if replicate_result:
                return replicate_result

            logger.warning(
                "[audio] All transcription backends failed for url=%s. "
                "Install faster-whisper or set REPLICATE_API_KEY.",
                url or suffix,
            )
            return ExtractResult(text="", language="", duration_s=0.0, backend="none")

        finally:
            _safe_unlink(input_path)
            if wav_path:
                _safe_unlink(wav_path)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _safe_unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass
