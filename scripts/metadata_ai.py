"""
Generate viral, authentic YouTube Shorts titles (3 ranked variants), captions, and tags
using Groq AI (Llama 3.3 / 3.1) or OpenAI grounded in the actual transcript content.
Strips download watermarks and enforces YouTube length limits without clickbait.
"""
import json
import logging
import os
import re
import time
from openai import OpenAI
import config
from clip_detection import _get_best_groq_model
import hook_generator

log = logging.getLogger("metadata_ai")


def clean_filename_fallback(filename: str) -> str:
    """Basic fallback cleaner if AI call fails."""
    name = filename
    name = re.sub(r"\.(mp4|mov|mkv|avi|webm)$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"(vidssave\.com|y2mate\.is|savefrom|ssyoutube|snaptik|ytshorts)", "", name, flags=re.IGNORECASE)
    name = re.sub(r"[_|•\-]+", " ", name)
    name = re.sub(r"\b(720p|1080p|480p|360p|320\s*kbps|128\s*kbps)\b", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\s+", " ", name).strip()
    return name or "Inspiring Short"


def extract_json_payload(content: str) -> dict:
    """
    Robustly extracts and parses a JSON object from LLM response text:
    - Handles raw JSON objects
    - Handles ```json ... ``` or ``` ... ``` markdown code fences
    - Handles surrounding conversational text before/after JSON
    - Handles unescaped newlines inside string values (strict=False)
    - Strips trailing commas
    """
    if not content or not isinstance(content, str):
        raise ValueError("Empty or non-string LLM response content")

    text = content.strip()

    # 1. Try stripping markdown code fences if wrapped
    fence_pattern = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
    match = fence_pattern.search(text)
    if match:
        fence_content = match.group(1).strip()
        try:
            parsed = json.loads(fence_content, strict=False)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    # 2. Try direct json.loads with strict=False
    try:
        parsed = json.loads(text, strict=False)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # 3. Try finding outermost JSON object brackets { ... }
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        brace_content = text[first_brace:last_brace + 1].strip()
        try:
            parsed = json.loads(brace_content, strict=False)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        # 4. Clean trailing commas e.g. { "a": 1, }
        cleaned = re.sub(r",\s*([\]}])", r"\1", brace_content)
        try:
            parsed = json.loads(cleaned, strict=False)
            if isinstance(parsed, dict):
                return parsed
        except Exception as e:
            raise ValueError(
                f"Could not parse JSON from braced block ({e}). Preview: {brace_content[:150]}"
            ) from e

    raise ValueError(f"No valid JSON object structure found in response. Preview: {text[:150]}")


def _get_api_key() -> str:
    """Dynamically resolves the active API key from config or environment."""
    return (
        getattr(config, "GROQ_API_KEY", "")
        or getattr(config, "OPENAI_API_KEY", "")
        or os.environ.get("GROQ_API_KEY", "")
        or os.environ.get("OPENAI_API_KEY", "")
    ).strip()


def generate_shorts_metadata(filename: str, transcript: str = "") -> dict:
    """
    Generates 3 ranked title variants, an engaging description, and tags
    strictly grounded in what is actually said in the transcript.

    Returns:
      {
        "title": title_1,
        "title_variants": [title_1, title_2, title_3],
        "punchline": punchline,
        "generated_hook": punchline,
        "description": description,
        "tags": tags,
      }
    """
    api_key = _get_api_key()
    if not api_key:
        log.warning(
            "[METADATA_MISSING_API_KEY] Neither GROQ_API_KEY nor OPENAI_API_KEY is configured. "
            "Using deterministic filename fallback for '%s'.",
            filename,
        )
        fallback = clean_filename_fallback(filename)
        t1 = f"{fallback[:75]} #shorts"
        return {
            "title": t1,
            "title_variants": [t1, f"Secret to {fallback[:60]} #shorts", f"Watch this: {fallback[:60]} #shorts"],
            "punchline": f"{fallback[:45]} ✨",
            "generated_hook": f"{fallback[:45]} ✨",
            "description": f"{fallback}\n\n#shorts #podcast #viral",
            "tags": ["shorts", "podcast", "viral", "clips"],
        }

    # Initialize client (Groq or OpenAI)
    groq_key = getattr(config, "GROQ_API_KEY", "") or os.environ.get("GROQ_API_KEY", "").strip()
    is_groq = api_key.startswith("gsk_") or bool(groq_key)
    chat_model = getattr(config, "GROQ_CHAT_MODEL", "") or os.environ.get("GROQ_CHAT_MODEL", "").strip()

    if is_groq:
        client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        model = chat_model or _get_best_groq_model(client)
    else:
        client = OpenAI(api_key=api_key)
        model = chat_model or "gpt-4o-mini"

    # Limit transcript context to avoid token bloat while keeping full semantic context
    context_transcript = transcript.strip()[:3500] if transcript else clean_filename_fallback(filename)

    prompt = f"""You are a top YouTube Shorts audience retention and CTR strategist.
Below is the video filename and the exact transcript spoken in the video:

Filename: "{filename}"
Transcript:
\"\"\"
{context_transcript}
\"\"\"

Task:
Generate YouTube Shorts metadata strictly grounded in what is ACTUALLY SAID in the transcript.
Avoid misleading clickbait (misleading titles destroy audience retention and algorithm reach).
Titles must be intriguing, punchy, and authentic to the core insight.

Requirements:
1. Generate 3 distinct title variants, ranked in order of predicted CTR:
   - Variant 1 (Primary): The strongest hook highlighting the core insight or punchline.
   - Variant 2 (Curiosity/Question): A curiosity-driven question or insight reflecting the conversation.
   - Variant 3 (Actionable/Takeaway): A direct, high-value takeaway or quote.
   Each title MUST be under 80 characters (maximum 90 chars total) and end with "#shorts".
2. Punchline / Top Hook Header:
   - A single, highly engaging, interactive 1-line hook or punchline (4 to 8 words) from or about this clip to display prominently at the top of the video (e.g. "Khan Sir with Raj Shamani 🥰", "Khan Sir's Advice for Youth 🔥", "Wait for the Reality Check 🤯", or a key punchline dialogue from the clip).
   - Must be punchy, attention-grabbing, easy to read in 2 seconds, and authentically reflect the clip. Can include 1 fitting emoji.
3. Description / Caption: 2-3 engaging sentences summarizing the clip's authentic insight, followed by 4-6 relevant hashtags.
4. Tags: 5-8 relevant search keyword tags.

Respond ONLY with valid JSON in this exact structure:
{{
  "title_1": "...",
  "title_2": "...",
  "title_3": "...",
  "punchline": "...",
  "description": "...",
  "tags": ["...", "..."]
}}
"""

    last_error = None
    data = None

    # Retry loop with exponential backoff for network resilience and transient rate limits
    for attempt in range(1, 4):
        try:
            log.info(
                "Requesting AI metadata generation (model=%s, attempt=%d/3, transcript_len=%d)...",
                model, attempt, len(context_transcript),
            )
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                max_tokens=1000,
                response_format={"type": "json_object"} if ("llama" in model.lower() or "gpt" in model.lower()) else None,
            )
            content = response.choices[0].message.content.strip()
            data = extract_json_payload(content)
            if data and isinstance(data, dict):
                log.info("Successfully received and parsed AI metadata response on attempt %d.", attempt)
                break
        except Exception as e:
            last_error = e
            log.warning(
                "[METADATA_API_ERROR] AI metadata attempt %d/3 failed (%s: %s).",
                attempt, type(e).__name__, str(e),
            )
            if attempt < 3:
                time.sleep(attempt * 1.5)

    if not data or not isinstance(data, dict):
        log.warning(
            "[METADATA_FALLBACK_USED] AI metadata generation failed after 3 attempts (%s). "
            "Using deterministic filename fallback for '%s'.",
            last_error, filename,
        )
        fb = clean_filename_fallback(filename)
        t1 = f"{fb[:75]} #shorts"
        detected_lang = "hi" if any(ord(c) >= 0x0900 and ord(c) <= 0x097F for c in transcript) else "en"
        fb_hook = hook_generator.get_fallback_hook(detected_lang=detected_lang)
        punchline = fb_hook["hook"]
        return {
            "title": t1,
            "title_variants": [t1, f"Secret to {fb[:60]} #shorts", f"Watch this: {fb[:60]} #shorts"],
            "punchline": punchline,
            "generated_hook": punchline,
            "description": f"{fb}\n\n#shorts #podcast #viral",
            "tags": ["shorts", "podcast", "viral"],
        }

    def sanitize_title(t_str, default_suffix):
        t = str(t_str or "").strip()
        if not t:
            t = default_suffix
        if not t.lower().endswith("#shorts"):
            t = f"{t[:78]} #shorts"
        return t[:95]

    fb = clean_filename_fallback(filename)
    title_1 = sanitize_title(data.get("title_1"), fb)
    title_2 = sanitize_title(data.get("title_2"), f"Insight: {fb}")
    title_3 = sanitize_title(data.get("title_3"), f"Must Watch: {fb}")

    # Validate or generate authentic short-form hook using Hook Upgrade #1
    raw_punchline = str(data.get("punchline", "")).strip()
    cand = {"hook": raw_punchline, "hook_type": "curiosity"}
    is_valid, _ = hook_generator.validate_hook(cand, transcript=transcript, filename=filename)
    if is_valid:
        punchline = cand["hook"]
    else:
        detected_lang = "hi" if any(ord(c) >= 0x0900 and ord(c) <= 0x097F for c in transcript) else "en"
        try:
            hook_res = hook_generator.generate_short_hook(
                transcript=transcript,
                hook_summary=raw_punchline,
                filename=filename,
                detected_lang=detected_lang,
            )
            punchline = hook_res["selected_hook"]
        except Exception as he:
            log.warning("Secondary hook generation fallback (%s). Using sanitized candidate.", he)
            punchline = hook_generator.clean_hook_text(raw_punchline) or f"{fb[:45]} ✨"

    # Ensure description is non-empty and transcript-grounded
    raw_desc = str(data.get("description", "")).strip()
    if raw_desc:
        description = raw_desc
    elif transcript.strip():
        first_part = " ".join(transcript.strip().split()[:25])
        description = f"{first_part}...\n\n#shorts #viral #podcast"
    else:
        description = f"{fb}\n\n#shorts #podcast #viral"

    raw_tags = data.get("tags")
    if isinstance(raw_tags, list) and raw_tags:
        tags = [str(t).strip() for t in raw_tags if str(t).strip()]
    else:
        tags = ["shorts", "podcast", "viral", "clips"]

    log.info("AI Generated Title 1 (Primary): %s", title_1)
    log.info("AI Generated Punchline Header: %s", punchline)
    log.info("AI Generated Description length: %d chars", len(description))

    return {
        "title": title_1,
        "title_variants": [title_1, title_2, title_3],
        "punchline": punchline,
        "generated_hook": punchline,
        "description": description,
        "tags": tags,
    }


def generate_punchline(filename: str, transcript: str = "", hook_summary: str = "", detected_lang: str = "en") -> str:
    """Convenience helper to extract or generate a validated short-form punchline hook for a clip."""
    hook_res = hook_generator.generate_short_hook(
        transcript=transcript,
        hook_summary=hook_summary,
        filename=filename,
        detected_lang=detected_lang,
    )
    return hook_res["selected_hook"]
