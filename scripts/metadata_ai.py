"""
Generate viral, authentic YouTube Shorts titles (3 ranked variants), captions, and tags
using Groq AI (Llama 3.3 / 3.1) grounded in the actual transcript content.
Strips download watermarks and enforces YouTube length limits without clickbait.
"""
import json
import logging
import re
from openai import OpenAI
from config import GROQ_API_KEY, OPENAI_API_KEY, GROQ_CHAT_MODEL
from clip_detection import _get_best_groq_model

log = logging.getLogger("metadata_ai")


def clean_filename_fallback(filename: str) -> str:
    """Basic fallback cleaner if AI call fails."""
    name = filename
    name = re.sub(r"\.(mp4|mov|mkv|avi|webm)$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"(vidssave\.com|y2mate\.is|savefrom|ssyoutube|snaptik|ytshorts)", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\b(720p|1080p|480p|360p|320\s*kbps|128\s*kbps)\b", "", name, flags=re.IGNORECASE)
    name = re.sub(r"[_|•\-]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name or "Inspiring Short"


def generate_shorts_metadata(filename: str, transcript: str = "") -> dict:
    """
    Generates 3 ranked title variants, an engaging description, and tags
    strictly grounded in what is actually said in the transcript.

    Returns:
      {
        "title": title_1,
        "title_variants": [title_1, title_2, title_3],
        "description": description,
        "tags": tags,
      }
    """
    api_key = GROQ_API_KEY or OPENAI_API_KEY
    if not api_key:
        fallback = clean_filename_fallback(filename)
        t1 = f"{fallback[:75]} #shorts"
        return {
            "title": t1,
            "title_variants": [t1, f"Secret to {fallback[:60]} #shorts", f"Watch this: {fallback[:60]} #shorts"],
            "punchline": f"{fallback[:45]} ✨",
            "description": f"{fallback}\n\n#shorts #podcast #viral",
            "tags": ["shorts", "podcast", "viral", "clips"],
        }

    # Initialize client (Groq or OpenAI)
    if api_key.startswith("gsk_") or GROQ_API_KEY:
        client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        model = GROQ_CHAT_MODEL or _get_best_groq_model(client)
    else:
        client = OpenAI(api_key=api_key)
        model = "gpt-4o-mini"

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

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=1000,
            response_format={"type": "json_object"} if ("llama" in model.lower() or "gpt" in model.lower()) else None,
        )
        content = response.choices[0].message.content.strip()
        content = re.sub(r"^```json\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
        data = json.loads(content)

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

        raw_punchline = str(data.get("punchline", "")).strip().strip('"').strip("'")
        raw_punchline = raw_punchline.replace("{", "").replace("}", "")
        punchline = raw_punchline[:60] if raw_punchline else f"{fb[:45]} ✨"

        description = str(data.get("description", "")).strip()
        tags = list(data.get("tags", ["shorts", "podcast", "viral"]))

        log.info("AI Generated Title 1 (Primary): %s", title_1)
        log.info("AI Generated Punchline Header: %s", punchline)

        return {
            "title": title_1,
            "title_variants": [title_1, title_2, title_3],
            "punchline": punchline,
            "description": description,
            "tags": tags,
        }
    except Exception as e:
        log.warning("AI metadata generation failed: %s. Using fallback.", e)
        fb = clean_filename_fallback(filename)
        t1 = f"{fb[:75]} #shorts"
        return {
            "title": t1,
            "title_variants": [t1, f"Secret to {fb[:60]} #shorts", f"Watch this: {fb[:60]} #shorts"],
            "punchline": f"{fb[:45]} ✨",
            "description": f"{fb}\n\n#shorts #podcast #viral",
            "tags": ["shorts", "podcast", "viral"],
        }


def generate_punchline(filename: str, transcript: str = "", hook_summary: str = "") -> str:
    """Convenience helper to extract or generate a punchline hook for a clip."""
    meta = generate_shorts_metadata(filename, transcript=transcript)
    return meta.get("punchline") or hook_summary or clean_filename_fallback(filename)

