"""
AI Multi-Clip Detection.
Analyzes full-length video transcripts with timestamps to identify non-overlapping,
high-retention clip candidates suitable for standalone YouTube Shorts.
"""
import json
import logging
import re
from openai import OpenAI

from config import (
    GROQ_API_KEY, OPENAI_API_KEY, GROQ_CHAT_MODEL,
    MIN_CLIP_SECONDS, MAX_CLIP_SECONDS, MAX_CLIPS_PER_VIDEO,
)

log = logging.getLogger("clip_detection")


def _get_best_groq_model(client):
    preferred = [
        "openai/gpt-oss-120b",
        "llama-3.3-70b-versatile",
        "llama-3.1-70b-versatile",
        "openai/gpt-oss-20b",
        "groq/compound",
        "qwen/qwen3.8-27b",
    ]
    try:
        models = [m.id for m in client.models.list().data]
        for p in preferred:
            if p in models:
                return p
        for m in models:
            if "whisper" not in m.lower() and "guard" not in m.lower():
                return m
    except Exception:
        pass
    return "openai/gpt-oss-120b"


def _get_llm_client():
    api_key = GROQ_API_KEY or OPENAI_API_KEY
    if not api_key:
        return None, None
    if api_key.startswith("gsk_") or GROQ_API_KEY:
        client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        model = GROQ_CHAT_MODEL or _get_best_groq_model(client)
        return client, model
    return OpenAI(api_key=api_key), "gpt-4o-mini"


def detect_clips_from_transcript(
    segments: list,
    total_duration: float,
    min_clip_seconds: int = MIN_CLIP_SECONDS,
    max_clip_seconds: int = MAX_CLIP_SECONDS,
    max_clips: int = MAX_CLIPS_PER_VIDEO,
) -> list:
    """
    Asks LLM to find up to `max_clips` non-overlapping, high-engagement clips from transcript.
    Returns list of dicts:
      [
        {
          "start_time": float,
          "end_time": float,
          "duration": float,
          "hook_summary": str,
          "title_idea": str,
        },
        ...
      ]
    """
    total_duration = max(1.0, float(total_duration))

    # If video is already shorter than max_clip_seconds, return a single clip covering the whole video
    if total_duration <= max_clip_seconds:
        log.info("Video is %.1fs (<= %ds). Using single clip.", total_duration, max_clip_seconds)
        return [{
            "start_time": 0.0,
            "end_time": round(total_duration, 2),
            "duration": round(total_duration, 2),
            "hook_summary": "Full video clip",
            "title_idea": "Key Highlight #shorts",
            "punchline": "Watch Till The End 🔥",
        }]

    client, model = _get_llm_client()
    if not client or not segments:
        log.warning("No LLM client or empty segments. Falling back to default initial clip.")
        return _fallback_clips(total_duration, min_clip_seconds, max_clip_seconds, max_clips)

    # Format transcript with timestamps
    formatted_transcript_lines = []
    for s in segments:
        st = float(s.get("start", 0.0))
        et = float(s.get("end", 0.0))
        txt = str(s.get("text", "")).strip()
        if txt:
            formatted_transcript_lines.append(f"[{st:.1f}s - {et:.1f}s] {txt}")

    transcript_text = "\n".join(formatted_transcript_lines)
    # Truncate if extremely large to stay within safe prompt boundaries
    if len(transcript_text) > 40000:
        transcript_text = transcript_text[:40000] + "\n...[transcript truncated]"

    prompt = f"""You are a master YouTube Shorts viral strategist and video editor.
Analyze the timestamped transcript below from a video of total duration {total_duration:.1f} seconds.

Your task: Identify up to {max_clips} the MOST ENGAGING, high-retention, standalone segments suitable for YouTube Shorts.

Requirements:
1. Each clip MUST be between {min_clip_seconds} and {max_clip_seconds} seconds long (end_time - start_time >= {min_clip_seconds} and <= {max_clip_seconds}).
2. Each clip MUST start with a compelling hook or question and deliver a self-contained takeaway, insight, punchline, or story.
3. Clips MUST NOT overlap in time (start_time of one clip cannot fall inside another clip).
4. All start_time and end_time values MUST be within 0.0 and {total_duration:.1f} seconds.
5. SEMANTIC DIVERSITY: Each clip MUST explore a distinct topic, concept, story, or moment from the video. Do NOT select multiple clips that make the same core point or reiterate the same takeaway.
6. Rank your choices in order of predicted viewer retention and viral potential.

Transcript:
{transcript_text}

Respond ONLY with valid JSON in this exact structure:
{{
  "clips": [
    {{
      "start_time": 12.5,
      "end_time": 54.0,
      "hook_summary": "Explains why morning screen time ruins focus and what to do instead.",
      "title_idea": "The Morning Habit Destroying Your Brain #shorts",
      "punchline": "Stop Ruining Your Mornings 🛑"
    }}
  ]
}}
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            response_format={"type": "json_object"} if "llama" in model or "gpt" in model else None,
        )
        content = response.choices[0].message.content.strip()
        content = re.sub(r"^```json\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
        data = json.loads(content)
        raw_clips = data.get("clips", [])
        validated = _validate_and_filter_clips(
            raw_clips, total_duration, min_clip_seconds, max_clip_seconds, max_clips
        )
        if validated:
            log.info("Successfully detected %d high-retention clips.", len(validated))
            return validated
    except Exception as e:
        log.warning("AI clip detection failed: %s. Using fallback clips.", e)

    return _fallback_clips(total_duration, min_clip_seconds, max_clip_seconds, max_clips)


def _is_similar_summary(summary1: str, summary2: str, threshold: float = 0.3) -> bool:
    """
    Checks semantic word overlap between two hook summaries to prevent picking
    multiple clips that make the same core point.
    Uses overlap coefficient and Jaccard index on key content words.
    """
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with",
        "about", "why", "how", "what", "is", "it", "this", "that", "of", "from",
        "explains", "details", "shows", "discusses", "talks", "reveals", "key", "moment",
    }
    words1 = set(re.findall(r"\w+", summary1.lower())) - stop_words
    words2 = set(re.findall(r"\w+", summary2.lower())) - stop_words
    if not words1 or not words2:
        return False
    intersection = words1.intersection(words2)
    # Overlap coefficient: ratio of shared words to the smaller set
    overlap_coef = len(intersection) / min(len(words1), len(words2))
    jaccard = len(intersection) / len(words1.union(words2))
    return overlap_coef >= 0.4 or jaccard >= threshold


def _validate_and_filter_clips(
    raw_clips: list,
    total_duration: float,
    min_clip_seconds: int,
    max_clip_seconds: int,
    max_clips: int,
) -> list:
    """Validates bounds, resolves overlaps, deduplicates topics, and sorts clips."""
    accepted = []

    for item in raw_clips:
        try:
            start = float(item.get("start_time", 0.0))
            end = float(item.get("end_time", 0.0))
            summary = str(item.get("hook_summary", "")).strip() or "Key moment from video"
            title = str(item.get("title_idea", "")).strip() or "Must Watch Insight #shorts"
            punchline = str(item.get("punchline", "")).strip().strip('"').strip("'").replace("{", "").replace("}", "")[:60]
            if not punchline:
                clean_t = re.sub(r"#shorts", "", title, flags=re.IGNORECASE).strip()
                punchline = f"{clean_t[:45]} ✨" if clean_t else "Watch Till The End 🔥"
        except (ValueError, TypeError):
            continue

        # Bound clamps
        start = max(0.0, start)
        end = min(total_duration, end)

        if end <= start:
            continue

        duration = end - start
        if duration < min_clip_seconds:
            # If slightly short, try extending end if within total_duration
            if start + min_clip_seconds <= total_duration:
                end = start + min_clip_seconds
                duration = end - start
            else:
                continue

        if duration > max_clip_seconds:
            end = start + max_clip_seconds
            duration = max_clip_seconds

        candidate = {
            "start_time": round(start, 2),
            "end_time": round(end, 2),
            "duration": round(duration, 2),
            "hook_summary": summary,
            "title_idea": title,
            "punchline": punchline,
        }

        # Check overlap and semantic duplication with already accepted clips
        has_conflict = False
        for acc in accepted:
            # 1. Temporal overlap: max(s1, s2) < min(e1, e2)
            if max(candidate["start_time"], acc["start_time"]) < min(candidate["end_time"], acc["end_time"]):
                has_conflict = True
                break
            # 2. Semantic duplication check: avoid picking multiple clips making the same point
            if _is_similar_summary(candidate["hook_summary"], acc["hook_summary"]):
                log.info(
                    "Discarding clip [%.1fs - %.1fs] as semantically redundant with [%.1fs - %.1fs]: '%s'",
                    candidate["start_time"], candidate["end_time"], acc["start_time"], acc["end_time"], candidate["hook_summary"],
                )
                has_conflict = True
                break

        if not has_conflict:
            accepted.append(candidate)

        if len(accepted) >= max_clips:
            break

    # Sort chronologically by start_time
    accepted.sort(key=lambda x: x["start_time"])
    return accepted


def _fallback_clips(
    total_duration: float,
    min_clip_seconds: int,
    max_clip_seconds: int,
    max_clips: int,
) -> list:
    """Generates non-overlapping chronological chunks if AI detection fails."""
    clips = []
    chunk_len = float(max_clip_seconds)
    curr_start = 0.0

    while curr_start + min_clip_seconds <= total_duration and len(clips) < max_clips:
        curr_end = min(total_duration, curr_start + chunk_len)
        clips.append({
            "start_time": round(curr_start, 2),
            "end_time": round(curr_end, 2),
            "duration": round(curr_end - curr_start, 2),
            "hook_summary": f"Highlight segment starting at {int(curr_start)}s",
            "title_idea": f"Key Highlight Part {len(clips) + 1} #shorts",
            "punchline": "Watch Till The End 🔥",
        })
        curr_start = curr_end

    if not clips:
        clips.append({
            "start_time": 0.0,
            "end_time": round(min(total_duration, float(max_clip_seconds)), 2),
            "duration": round(min(total_duration, float(max_clip_seconds)), 2),
            "hook_summary": "Highlight segment",
            "title_idea": "Key Highlight #shorts",
            "punchline": "Watch Till The End 🔥",
        })

    return clips
