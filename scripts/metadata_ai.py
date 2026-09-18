"""
Generate viral, clean YouTube Shorts titles, descriptions (captions), and tags
using Groq AI (Llama 3.3 / 3.1) from raw video filenames.
Automatically strips downloader watermarks (vidssave, y2mate), resolutions (720P), etc.
"""
import json
import logging
import re
from openai import OpenAI
from config import GROQ_API_KEY, OPENAI_API_KEY

log = logging.getLogger("metadata_ai")


def clean_filename_fallback(filename: str) -> str:
    """Basic fallback cleaner if AI call fails."""
    name = filename
    # Remove common video extensions
    name = re.sub(r"\.(mp4|mov|mkv|avi|webm)$", "", name, flags=re.IGNORECASE)
    # Remove website names and downloader tags
    name = re.sub(r"(vidssave\.com|y2mate\.is|savefrom|ssyoutube|snaptik|ytshorts)", "", name, flags=re.IGNORECASE)
    # Remove resolution/bitrate tags like 720P, 1080P, 320 Kbps
    name = re.sub(r"\b(720p|1080p|480p|360p|320\s*kbps|128\s*kbps)\b", "", name, flags=re.IGNORECASE)
    # Replace underscores, hyphens, extra punctuation with space
    name = re.sub(r"[_|•\-]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name or "Inspiring Short"


def generate_shorts_metadata(filename: str) -> dict:
    """
    Returns a dict with:
      - 'title': High-CTR Short title (<90 chars, includes #shorts)
      - 'description': Engaging caption with emojis and hashtags
      - 'tags': list of relevant SEO search tags
    """
    api_key = GROQ_API_KEY or OPENAI_API_KEY
    if not api_key:
        fallback = clean_filename_fallback(filename)
        return {
            "title": f"{fallback[:80]} #shorts",
            "description": f"{fallback}\n\n#shorts #podcast #viral",
            "tags": ["shorts", "podcast", "viral", "clips"],
        }

    # Initialize client (Groq or OpenAI)
    if api_key.startswith("gsk_") or GROQ_API_KEY:
        client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        model = "llama-3.3-70b-versatile"
    else:
        client = OpenAI(api_key=api_key)
        model = "gpt-4o-mini"

    prompt = f"""You are a YouTube Shorts growth expert.
Given this raw video filename: "{filename}"

Generate clean, viral YouTube Shorts metadata:
1. Strip all download watermarks (like vidssave.com, y2mate), resolutions (720P, 1080P), audio bitrates, and ugly characters.
2. Title: A compelling, high-CTR hook/title under 80 characters. Must end with "#shorts".
3. Description/Caption: An engaging 2-3 sentence caption/hook for viewers, followed by 4-6 relevant trending hashtags.
4. Tags: A list of 5-8 relevant search keyword tags.

Respond ONLY with valid JSON in this exact structure, with no extra text or markdown ticks:
{{
  "title": "...",
  "description": "...",
  "tags": ["...", "..."]
}}
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            response_format={"type": "json_object"} if "llama" in model or "gpt" in model else None,
        )
        content = response.choices[0].message.content.strip()
        # Clean potential markdown fences
        content = re.sub(r"^```json\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
        data = json.loads(content)

        title = str(data.get("title", "")).strip()
        if not title.lower().endswith("#shorts"):
            title = f"{title[:80]} #shorts"

        description = str(data.get("description", "")).strip()
        tags = list(data.get("tags", ["shorts", "podcast", "viral"]))

        log.info("AI Generated Title: %s", title)
        return {
            "title": title[:100],
            "description": description,
            "tags": tags,
        }
    except Exception as e:
        log.warning("AI metadata generation failed: %s. Using fallback.", e)
        fallback = clean_filename_fallback(filename)
        return {
            "title": f"{fallback[:80]} #shorts",
            "description": f"{fallback}\n\n#shorts #podcast #viral",
            "tags": ["shorts", "podcast", "viral"],
        }
