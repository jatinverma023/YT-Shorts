"""
Transcribe audio -> SRT captions, auto-detecting English or Hindi.
Uses OpenAI's Whisper API (whisper-1), which natively supports Hindi and
returns segment-level timestamps we can turn into an SRT file.
"""
import logging
import subprocess

from openai import OpenAI

from config import OPENAI_API_KEY, GROQ_API_KEY

log = logging.getLogger("transcribe")

api_key = GROQ_API_KEY or OPENAI_API_KEY
if api_key.startswith("gsk_") or GROQ_API_KEY:
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )
    WHISPER_MODEL = "whisper-large-v3"
    log.info("Using Groq Whisper-large-v3 transcription")
else:
    client = OpenAI(api_key=api_key)
    WHISPER_MODEL = "whisper-1"
    log.info("Using OpenAI Whisper-1 transcription")


def extract_audio(video_path, audio_path):
    """Pull a mono 16kHz wav out of the video for transcription."""
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", "16000",
        audio_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return audio_path


def _format_timestamp(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    hh, ms = divmod(ms, 3600000)
    mm, ms = divmod(ms, 60000)
    ss, ms = divmod(ms, 1000)
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def transcribe_to_srt(audio_path, srt_path):
    """
    Calls Whisper with verbose_json to get segment timestamps.
    Language is auto-detected (works for English and Hindi automatically);
    Whisper returns the detected language too, which we log and return.
    """
    with open(audio_path, "rb") as f:
        result = client.audio.transcriptions.create(
            model=WHISPER_MODEL,
            file=f,
            response_format="verbose_json",
            # no "language" param passed -> auto-detect
        )

    detected_lang = getattr(result, "language", "unknown")
    log.info("Detected language: %s", detected_lang)

    segments = result.segments  # list of {start, end, text, ...}
    with open(srt_path, "w", encoding="utf-8") as out:
        for i, seg in enumerate(segments, start=1):
            start = _format_timestamp(seg["start"] if isinstance(seg, dict) else seg.start)
            end = _format_timestamp(seg["end"] if isinstance(seg, dict) else seg.end)
            text = (seg["text"] if isinstance(seg, dict) else seg.text).strip()
            out.write(f"{i}\n{start} --> {end}\n{text}\n\n")

    return srt_path, detected_lang
