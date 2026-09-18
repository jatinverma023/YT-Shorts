"""
Transcribe audio with word-level timestamps (Groq Whisper-large-v3 or OpenAI Whisper)
and generate animated, word-level karaoke/pop-style burned-in ASS captions.
"""
import logging
import math
import os
import subprocess
import time

from openai import OpenAI

from config import (
    OPENAI_API_KEY, GROQ_API_KEY, TARGET_WIDTH, TARGET_HEIGHT,
    SUBTITLE_FONT, SUBTITLE_FONT_SIZE, CAPTION_HIGHLIGHT_COLOR,
    CAPTION_BASE_COLOR, CAPTION_OUTLINE_COLOR, WORDS_PER_PHRASE,
    ENABLE_TOP_PUNCHLINE, PUNCHLINE_FONT, PUNCHLINE_FONT_SIZE,
    PUNCHLINE_MARGIN_TOP, PUNCHLINE_COLOR, PUNCHLINE_OUTLINE_COLOR,
    CAPTION_LANGUAGE_MODE, CAPTION_UPPERCASE, CAPTION_MAX_LINES,
    CAPTION_MAX_CHARS_PER_LINE, CAPTION_MAX_WORDS, CAPTION_MARGIN_BOTTOM,
)
from transliterate import romanize_words, has_devanagari

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
    """Pull mono 16kHz audio (WAV or 32k MP3) out of the video for transcription, optionally trimmed."""
    cmd = ["ffmpeg", "-y", "-i", video_path]
    if max_seconds:
        cmd.extend(["-t", str(max_seconds)])
    cmd.extend(["-vn", "-ac", "1", "-ar", "16000"])
    if audio_path.lower().endswith(".mp3"):
        cmd.extend(["-b:a", "32k"])
    cmd.append(audio_path)
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


def _transcribe_file_single(client, whisper_model, file_path: str):
    """Transcribes a single audio file with retry logic for network resilience."""
    last_err = None
    for attempt in range(1, 4):
        try:
            with open(file_path, "rb") as f:
                try:
                    return client.audio.transcriptions.create(
                        model=whisper_model,
                        file=f,
                        response_format="verbose_json",
                        timestamp_granularities=["word", "segment"],
                    )
                except Exception as e:
                    log.warning("Word timestamp request failed (%s). Retrying with verbose_json.", e)
                    f.seek(0)
                    return client.audio.transcriptions.create(
                        model=whisper_model,
                        file=f,
                        response_format="verbose_json",
                    )
        except Exception as err:
            last_err = err
            log.warning("Whisper transcription attempt %d/3 failed: %s. Retrying in %ds...", attempt, err, attempt * 2)
            time.sleep(attempt * 2)
    raise last_err


def _transcribe_large_audio(audio_path: str, client, whisper_model: str) -> dict:
    """
    Splits long audio (>20MB or multi-hour) into 10-minute (600s) chunks,
    transcribes each chunk independently, offsets timestamps, and merges cleanly.
    Guarantees that files never exceed Groq's 25MB upload limit.
    """
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", audio_path,
    ]
    try:
        total_duration = float(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip())
    except Exception as e:
        log.warning("Could not probe audio duration (%s). Defaulting to 3600s.", e)
        total_duration = 3600.0

    chunk_seconds = 600.0
    num_chunks = int(math.ceil(total_duration / chunk_seconds))
    base_dir = os.path.dirname(audio_path)
    base_name = os.path.splitext(os.path.basename(audio_path))[0]

    log.info(
        "Audio file %s is %.1f MB (duration: %.1fs). Splitting into %d chunks of %ds for Whisper transcription...",
        audio_path, os.path.getsize(audio_path) / (1024 * 1024), total_duration, num_chunks, int(chunk_seconds),
    )

    all_words = []
    all_segments = []
    text_parts = []
    detected_lang = "unknown"

    for i in range(num_chunks):
        c_start = i * chunk_seconds
        c_dur = min(chunk_seconds, total_duration - c_start)
        if c_dur <= 0.5:
            break
        chunk_file = os.path.join(base_dir, f"{base_name}_chunk{i}.mp3")
        cmd_extract = [
            "ffmpeg", "-y",
            "-ss", f"{c_start:.3f}",
            "-t", f"{c_dur:.3f}",
            "-i", audio_path,
            "-vn", "-ac", "1", "-ar", "16000", "-b:a", "32k",
            chunk_file,
        ]
        try:
            subprocess.run(cmd_extract, check=True, capture_output=True, text=True)
            log.info("Transcribing chunk %d/%d [%.1fs - %.1fs]...", i + 1, num_chunks, c_start, c_start + c_dur)
            chunk_res = _transcribe_file_single(client, whisper_model, chunk_file)

            lang = getattr(chunk_res, "language", "unknown")
            if detected_lang == "unknown" and lang != "unknown":
                detected_lang = lang

            txt = getattr(chunk_res, "text", "").strip()
            if txt:
                text_parts.append(txt)

            c_words = getattr(chunk_res, "words", None) or []
            for w in c_words:
                w_dict = w if isinstance(w, dict) else w.__dict__
                word_text = str(w_dict.get("word", "")).strip()
                if word_text:
                    all_words.append({
                        "word": word_text,
                        "start": round(float(w_dict.get("start", 0.0)) + c_start, 3),
                        "end": round(float(w_dict.get("end", 0.0)) + c_start, 3),
                    })

            c_segs = getattr(chunk_res, "segments", None) or []
            for s in c_segs:
                s_dict = s if isinstance(s, dict) else s.__dict__
                seg_text = str(s_dict.get("text", "")).strip()
                if seg_text:
                    all_segments.append({
                        "text": seg_text,
                        "start": round(float(s_dict.get("start", 0.0)) + c_start, 3),
                        "end": round(float(s_dict.get("end", 0.0)) + c_start, 3),
                    })
        finally:
            if os.path.exists(chunk_file):
                try:
                    os.remove(chunk_file)
                except Exception:
                    pass

    full_text = " ".join(text_parts).strip()
    log.info(
        "Finished chunked transcription: %d words across %d segments. Language: %s",
        len(all_words), len(all_segments), detected_lang,
    )
    return {
        "text": full_text,
        "language": detected_lang,
        "words": all_words,
        "segments": all_segments,
    }


def transcribe_audio(audio_path):
    """
    Calls Whisper with verbose_json and word-level timestamps.
    Automatically handles files exceeding the 20MB safe limit by chunking.
    Returns:
      {
        "text": full_text_string,
        "language": detected_language,
        "words": [{"word": str, "start": float, "end": float}, ...],
        "segments": [{"text": str, "start": float, "end": float}, ...]
      }
    """
    client, whisper_model = get_transcribe_client()
    file_size = os.path.getsize(audio_path)
    MAX_DIRECT_BYTES = 20 * 1024 * 1024  # 20MB threshold to guarantee safety against 25MB limit

    if file_size > MAX_DIRECT_BYTES:
        return _transcribe_large_audio(audio_path, client, whisper_model)

    result = _transcribe_file_single(client, whisper_model, audio_path)

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


def _sanitize_ass(text: str) -> str:
    """Escapes/removes braces and backslashes to prevent ASS tag injection."""
    if not text:
        return ""
    return str(text).replace("\\", "").replace("{", "").replace("}", "").strip()


def find_phrase_split(phrase: list, max_chars_per_line: int = CAPTION_MAX_CHARS_PER_LINE) -> int:
    """
    Finds the optimal word index k to split phrase into 2 lines.
    Returns len(phrase) if phrase comfortably fits on 1 line.
    """
    n = len(phrase)
    if n <= 2 or CAPTION_MAX_LINES < 2:
        return n

    full_text = " ".join(str(w.get("word", "")) for w in phrase)
    if len(full_text) <= max_chars_per_line and n <= 3:
        return n

    best_k = n
    best_penalty = float("inf")

    for k in range(1, n):
        l1 = " ".join(str(w.get("word", "")) for w in phrase[:k])
        l2 = " ".join(str(w.get("word", "")) for w in phrase[k:])
        len1 = len(l1)
        len2 = len(l2)

        penalty = max(len1, len2) + abs(len1 - len2) * 0.5
        if len1 > max_chars_per_line:
            penalty += 40.0
        if len2 > max_chars_per_line:
            penalty += 40.0

        if penalty < best_penalty:
            best_penalty = penalty
            best_k = k

    return best_k


def group_words_into_phrases(
    words: list,
    max_words: int = None,
    max_chars: int = None,
    max_gap: float = 0.40,
    max_duration: float = 3.2,
) -> list:
    """
    Groups words into short phrases (1-2 lines on screen) based on:
    - Natural silence/pause breaks (gap between words >= max_gap)
    - Strong punctuation (. ! ? ,)
    - Maximum word count per phrase (default CAPTION_MAX_WORDS = 8)
    - Maximum character length per phrase
    - Maximum duration
    """
    if not words:
        return []

    if max_words is None:
        max_words = CAPTION_MAX_WORDS
    if max_chars is None:
        max_chars = CAPTION_MAX_CHARS_PER_LINE * 2

    phrases = []
    current_phrase = []

    for w in words:
        if not current_phrase:
            current_phrase.append(w)
            continue

        prev_w = current_phrase[-1]
        gap = float(w.get("start", 0.0)) - float(prev_w.get("end", 0.0))
        phrase_dur = float(w.get("end", 0.0)) - float(current_phrase[0].get("start", 0.0))
        curr_text = " ".join(str(item.get("word", "")) for item in current_phrase + [w])

        prev_word_str = str(prev_w.get("word", "")).rstrip()
        has_clause_break = prev_word_str.endswith((".", "!", "?", ","))
        is_gap_break = gap >= max_gap
        is_max_words = len(current_phrase) >= max_words
        is_max_chars = len(curr_text) > max_chars
        is_max_duration = phrase_dur > max_duration

        if has_clause_break or is_gap_break or is_max_words or is_max_chars or is_max_duration:
            phrases.append(current_phrase)
            current_phrase = [w]
        else:
            current_phrase.append(w)

    if current_phrase:
        phrases.append(current_phrase)

    return phrases


def generate_ass_captions(
    words,
    ass_path,
    target_width=TARGET_WIDTH,
    target_height=TARGET_HEIGHT,
    punchline: str = None,
    total_duration: float = None,
):
    """
    Generates an ASS subtitle file with:
    - Top punchline hook header (Alignment: 8 = Top Center) matching viral Shorts design
    - Animated, pop/karaoke word-level highlights in lower third (Alignment: 2 = Bottom Center).
    - Modern Roman Hindi / Hinglish conversion preserving timestamps and English terms.
    - Balanced 1-2 line wrapping with safe bottom margin.
    """
    # Romanize words if configured (guarantees strict 1:1 timestamp preservation)
    if words and CAPTION_LANGUAGE_MODE == "romanized":
        words = romanize_words(words)

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {target_width}
PlayResY: {target_height}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{SUBTITLE_FONT},{SUBTITLE_FONT_SIZE},{CAPTION_BASE_COLOR},&H000000FF,{CAPTION_OUTLINE_COLOR},&H80000000,-1,0,0,0,100,100,0,0,1,3.5,1.5,2,40,40,{CAPTION_MARGIN_BOTTOM},1
Style: HeaderPunchline,{PUNCHLINE_FONT},{PUNCHLINE_FONT_SIZE},{PUNCHLINE_COLOR},&H000000FF,{PUNCHLINE_OUTLINE_COLOR},&H80000000,-1,0,0,0,100,100,0,0,1,3.5,1.5,8,50,50,{PUNCHLINE_MARGIN_TOP},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    dialogue_lines = []

    # Calculate overall duration for top punchline header
    if total_duration and total_duration > 0:
        p_end = float(total_duration)
    elif words:
        p_end = max((float(w.get("end", 0.0)) for w in words), default=59.0) + 1.0
    else:
        p_end = 59.0

    # Add top punchline event if enabled and provided
    if ENABLE_TOP_PUNCHLINE and punchline:
        clean_punchline = _sanitize_ass(punchline)
        if clean_punchline:
            if has_devanagari(clean_punchline) and CAPTION_LANGUAGE_MODE == "romanized":
                p_words = [{"word": w, "start": 0.0, "end": 0.0} for w in clean_punchline.split()]
                clean_punchline = " ".join(pw["word"] for pw in romanize_words(p_words))
            p_start_str = _format_ass_time(0.0)
            p_end_str = _format_ass_time(p_end)
            dialogue_lines.append(
                f"Dialogue: 1,{p_start_str},{p_end_str},HeaderPunchline,,0,0,0,,{clean_punchline}"
            )

    if not words:
        with open(ass_path, "w", encoding="utf-8") as f:
            f.write(header + "\n".join(dialogue_lines) + "\n")
        return ass_path

    # Group words into natural, balanced phrases (1-2 lines)
    phrases = group_words_into_phrases(words)

    for p_idx, phrase in enumerate(phrases):
        if not phrase:
            continue
        n_words = len(phrase)
        split_k = find_phrase_split(phrase, CAPTION_MAX_CHARS_PER_LINE)

        # Lookahead to next phrase start time to guarantee zero overlap
        next_phrase_start = None
        if p_idx + 1 < len(phrases) and phrases[p_idx + 1]:
            next_phrase_start = phrases[p_idx + 1][0].get("start")

        for idx in range(n_words):
            current_word = phrase[idx]
            w_start = current_word["start"]
            if idx < n_words - 1:
                # Next word's start within the same phrase
                w_end = phrase[idx + 1]["start"]
            else:
                # Last word in phrase: match original word end timestamp exactly
                w_end = current_word["end"]
                if next_phrase_start is not None:
                    w_end = min(w_end, next_phrase_start)

            def _render_word(w_obj, is_act):
                clean_w = _sanitize_ass(w_obj.get("word", ""))
                if is_act:
                    # Active word: Highlight color + 10% pop scale, then return to base style
                    return f"{{\\c{CAPTION_HIGHLIGHT_COLOR}\\fscx110\\fscy110}}{clean_w}{{\\c{CAPTION_BASE_COLOR}\\fscx100\\fscy100}}"
                else:
                    # Inactive word: Base white color + normal 100% scale
                    return f"{{\\c{CAPTION_BASE_COLOR}\\fscx100\\fscy100}}{clean_w}"

            line1_rendered = [_render_word(phrase[j], j == idx) for j in range(split_k)]
            line2_rendered = [_render_word(phrase[j], j == idx) for j in range(split_k, n_words)]

            if line2_rendered:
                text_line = " ".join(line1_rendered) + "\\N" + " ".join(line2_rendered)
            else:
                text_line = " ".join(line1_rendered)

            start_str = _format_ass_time(w_start)
            end_str = _format_ass_time(max(w_end, w_start + 0.05))
            dialogue_lines.append(
                f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{text_line}"
            )

    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(dialogue_lines) + "\n")

    log.info("Generated animated ASS captions at %s with %d events (punchline: %s)", ass_path, len(dialogue_lines), punchline)
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
