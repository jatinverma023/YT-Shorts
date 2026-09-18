"""
Transcribe audio with word-level timestamps (Groq Whisper-large-v3 or OpenAI Whisper)
and generate animated, word-level karaoke/pop-style burned-in ASS captions.
"""
import logging
import subprocess

from openai import OpenAI

from config import (
    OPENAI_API_KEY, GROQ_API_KEY, TARGET_WIDTH, TARGET_HEIGHT,
    SUBTITLE_FONT, SUBTITLE_FONT_SIZE, CAPTION_HIGHLIGHT_COLOR,
    CAPTION_BASE_COLOR, CAPTION_OUTLINE_COLOR, WORDS_PER_PHRASE,
)

log = logging.getLogger("transcribe")


def get_transcribe_client():
    api_key = GROQ_API_KEY or OPENAI_API_KEY
    if not api_key:
        raise ValueError("Neither GROQ_API_KEY nor OPENAI_API_KEY is configured.")
    if api_key.startswith("gsk_") or GROQ_API_KEY:
        log.info("Using Groq Whisper-large-v3 transcription with word timestamps")
        return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1"), "whisper-large-v3"
    else:
        log.info("Using OpenAI Whisper-1 transcription with word timestamps")
        return OpenAI(api_key=api_key), "whisper-1"


def extract_audio(video_path, audio_path, max_seconds=None):
    """Pull a mono 16kHz wav out of the video for transcription, optionally trimmed."""
    cmd = ["ffmpeg", "-y", "-i", video_path]
    if max_seconds:
        cmd.extend(["-t", str(max_seconds)])
    cmd.extend(["-vn", "-ac", "1", "-ar", "16000", audio_path])
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        log.error("FFmpeg extract_audio failed: %s", e.stderr)
        raise
    return audio_path


def _format_srt_timestamp(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    hh, ms = divmod(ms, 3600000)
    mm, ms = divmod(ms, 60000)
    ss, ms = divmod(ms, 1000)
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def _format_ass_time(seconds: float) -> str:
    """Format seconds into ASS timestamp format: H:MM:SS.cc"""
    centis = int(round(seconds * 100))
    hh, centis = divmod(centis, 360000)
    mm, centis = divmod(centis, 6000)
    ss, centis = divmod(centis, 100)
    return f"{hh}:{mm:02d}:{ss:02d}.{centis:02d}"


def transcribe_audio(audio_path):
    """
    Calls Whisper with verbose_json and word-level timestamps.
    Returns:
      {
        "text": full_text_string,
        "language": detected_language,
        "words": [{"word": str, "start": float, "end": float}, ...],
        "segments": [{"text": str, "start": float, "end": float}, ...]
      }
    """
    client, whisper_model = get_transcribe_client()
    with open(audio_path, "rb") as f:
        # Request both word and segment timestamps
        try:
            result = client.audio.transcriptions.create(
                model=whisper_model,
                file=f,
                response_format="verbose_json",
                timestamp_granularities=["word", "segment"],
            )
        except Exception as e:
            log.warning("Word timestamp request failed (%s). Retrying with verbose_json.", e)
            f.seek(0)
            result = client.audio.transcriptions.create(
                model=whisper_model,
                file=f,
                response_format="verbose_json",
            )

    detected_lang = getattr(result, "language", "unknown")
    full_text = getattr(result, "text", "").strip()
    raw_words = getattr(result, "words", None) or []
    raw_segments = getattr(result, "segments", None) or []

    # Standardize words list
    words = []
    for w in raw_words:
        w_dict = w if isinstance(w, dict) else w.__dict__
        word_text = str(w_dict.get("word", "")).strip()
        if word_text:
            words.append({
                "word": word_text,
                "start": float(w_dict.get("start", 0.0)),
                "end": float(w_dict.get("end", 0.0)),
            })

    # Standardize segments list
    segments = []
    for s in raw_segments:
        s_dict = s if isinstance(s, dict) else s.__dict__
        segments.append({
            "text": str(s_dict.get("text", "")).strip(),
            "start": float(s_dict.get("start", 0.0)),
            "end": float(s_dict.get("end", 0.0)),
        })

    log.info(
        "Transcribed %d words across %d segments. Language: %s",
        len(words), len(segments), detected_lang,
    )
    return {
        "text": full_text,
        "language": detected_lang,
        "words": words,
        "segments": segments,
    }


def generate_ass_captions(words, ass_path, target_width=TARGET_WIDTH, target_height=TARGET_HEIGHT):
    """
    Generates an ASS subtitle file with animated, pop/karaoke word-level highlights.
    Displays words in short phrases (WORDS_PER_PHRASE), with the active word popped & highlighted.
    """
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {target_width}
PlayResY: {target_height}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{SUBTITLE_FONT},{SUBTITLE_FONT_SIZE},{CAPTION_BASE_COLOR},&H000000FF,{CAPTION_OUTLINE_COLOR},&H80000000,-1,0,0,0,100,100,0,0,1,3.5,1.5,2,40,40,260,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    if not words:
        with open(ass_path, "w", encoding="utf-8") as f:
            f.write(header)
        return ass_path

    dialogue_lines = []

    # Chunk words into small phrases (e.g. 3-4 words)
    phrase_size = max(2, WORDS_PER_PHRASE)
    phrases = [words[i:i + phrase_size] for i in range(0, len(words), phrase_size)]

    for phrase in phrases:
        if not phrase:
            continue
        n_words = len(phrase)
        for idx in range(n_words):
            current_word = phrase[idx]
            w_start = current_word["start"]
            # Connect seamlessly to next word's start to avoid visual flickering
            if idx < n_words - 1:
                w_end = phrase[idx + 1]["start"]
            else:
                w_end = current_word["end"] + 0.1

            # Build line with the current word highlighted and scaled
            rendered_words = []
            for j, w in enumerate(phrase):
                clean_w = w["word"].replace("{", "").replace("}", "")
                if j == idx:
                    # Active word: Highlight color + 15% pop-up scale
                    rendered_words.append(
                        f"{{\\c{CAPTION_HIGHLIGHT_COLOR}\\fscx115\\fscy115}}{clean_w}{{\\r}}"
                    )
                else:
                    # Inactive word: Base white
                    rendered_words.append(
                        f"{{\\c{CAPTION_BASE_COLOR}}}{clean_w}"
                    )

            text_line = " ".join(rendered_words)
            start_str = _format_ass_time(w_start)
            end_str = _format_ass_time(max(w_end, w_start + 0.05))
            dialogue_lines.append(
                f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{text_line}"
            )

    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(dialogue_lines) + "\n")

    log.info("Generated animated ASS captions at %s with %d events", ass_path, len(dialogue_lines))
    return ass_path


def transcribe_to_srt(audio_path, srt_path):
    """Fallback legacy SRT generator if needed."""
    result = transcribe_audio(audio_path)
    segments = result["segments"]
    with open(srt_path, "w", encoding="utf-8") as out:
        for i, seg in enumerate(segments, start=1):
            start = _format_srt_timestamp(seg["start"])
            end = _format_srt_timestamp(seg["end"])
            text = seg["text"].strip()
            out.write(f"{i}\n{start} --> {end}\n{text}\n\n")
    return srt_path, result["language"]
