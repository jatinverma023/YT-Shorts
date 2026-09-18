"""
Hook Upgrade #1: AI-Generated Short-Form Hooks
Generates, validates, and ranks short, high-curiosity, context-grounded hooks
for YouTube Shorts top header overlays (HeaderPunchline ASS style).

Key Constraints:
- 3–8 words, hard maximum 42 characters
- Grounded strictly in the clip's actual transcript content (supported_by_clip == True)
- Single LLM call generates exactly 5 candidate hooks
- Deterministic Python validation and 6-dimension scoring
- Max 1 emoji allowed
- Natural Roman Hindi/Hinglish for Hindi clips, natural English for English clips
- Safe, non-hallucinatory fallback when AI fails
"""
import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

from openai import OpenAI
from config import GROQ_API_KEY, OPENAI_API_KEY, GROQ_CHAT_MODEL
from clip_detection import _get_best_groq_model

log = logging.getLogger("hook_generator")

# Maximum characters allowed for a Shorts top header hook
MAX_HOOK_CHARS = 42

# Allowed hook types
ALLOWED_HOOK_TYPES = {
    "curiosity",
    "question",
    "revelation",
    "unexpected_fact",
    "consequence",
    "debate",
    "explanation",
    "emotional",
    "insight",
}

# 6-Dimension Hook Scoring Weights (must sum to 1.0)
HOOK_SCORING_WEIGHTS = {
    "curiosity": 0.25,
    "relevance": 0.25,
    "payoff": 0.15,
    "clarity": 0.15,
    "brevity": 0.10,
    "impact": 0.10,
}

# Emoji detection pattern covering emoticons, flags, pictographs, symbols
EMOJI_PATTERN = re.compile(
    r"("
    r"[\U0001F1E6-\U0001F1FF]{2}|"  # flags (e.g. 🇮🇳)
    r"[\U0001F600-\U0001F64F]|"      # emoticons
    r"[\U0001F300-\U0001F5FF]|"      # misc symbols and pictographs
    r"[\U0001F680-\U0001F6FF]|"      # transport & maps
    r"[\U0001F700-\U0001F77F]|"      # alchemical symbols
    r"[\U0001F780-\U0001F7FF]|"      # geometric shapes extended
    r"[\U0001F800-\U0001F8FF]|"      # supplemental arrows-C
    r"[\U0001F900-\U0001F9FF]|"      # supplemental symbols & pictographs (🤯, etc)
    r"[\U0001FA00-\U0001FA6F]|"      # chess symbols
    r"[\U0001FA70-\U0001FAFF]|"      # symbols and pictographs extended-A
    r"[\u2600-\u26FF]|"              # misc symbols (⚠️, ⚡, etc)
    r"[\u2700-\u27BF]"               # dingbats (✨, etc)
    r")"
)

# Generic / lazy placeholders that should never be used as hooks
FORBIDDEN_PLACEHOLDERS = [
    "watch till the end",
    "watch till end",
    "watch to the end",
    "wait for it",
    "wait till the end",
    "inspiring short",
    "click here",
    "subscribe now",
    "must watch video",
    "viral video",
    "part 1",
    "part 2",
    "part 3",
]


def count_emojis(text: str) -> int:
    """Returns the total number of emojis present in the text."""
    return len(EMOJI_PATTERN.findall(text))


def clean_hook_text(text: str) -> str:
    """Sanitizes text by removing surrounding quotes, curly braces, and redundant whitespace."""
    if not text:
        return ""
    t = str(text).strip().strip('"\'`')
    t = t.replace("{", "").replace("}", "").replace("\n", " ").replace("\r", " ").replace("\t", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def is_filename_or_title_leak(hook_text: str, filename: str = "", source_title: str = "") -> bool:
    """Detects if hook lazily copies or leaks the raw video filename or source title."""
    hook_norm = hook_text.lower().strip()

    # Check against filename
    if filename:
        base = os.path.splitext(os.path.basename(filename))[0].lower()
        clean_base = re.sub(r"[_\-•\.]+", " ", base)
        clean_base = re.sub(r"\s+", " ", clean_base).strip()
        if len(clean_base) >= 6 and (clean_base in hook_norm or hook_norm in clean_base):
            return True
        if re.search(r"\.(mp4|mov|mkv|avi|webm|m4v)\b", hook_norm):
            return True

    # Check against source title
    if source_title:
        title_norm = re.sub(r"[_\-•\.]+", " ", source_title.lower()).strip()
        title_norm = re.sub(r"\s+", " ", title_norm)
        if len(title_norm) >= 6 and (title_norm in hook_norm or hook_norm in title_norm):
            return True

    return False


def validate_hook(
    candidate: dict,
    transcript: str = "",
    filename: str = "",
    source_title: str = "",
) -> Tuple[bool, str]:
    """
    Deterministically validates a single hook candidate.
    Returns (is_valid, reason).
    """
    # 1. Content Grounding: candidate must declare supported_by_clip == True
    if candidate.get("supported_by_clip") is False:
        return False, "Hook is not supported by the clip content"

    raw_hook = candidate.get("hook", "")
    hook = clean_hook_text(raw_hook)

    # 2. Non-empty check
    if not hook:
        return False, "Hook is empty or whitespace"

    # 3. Hard character limit (max 42 characters)
    if len(hook) > MAX_HOOK_CHARS:
        return False, f"Hook exceeds hard limit ({len(hook)} > {MAX_HOOK_CHARS} chars)"

    # 4. Word count check (target: 3–8 words, strictly max 8 words)
    no_emoji_text = EMOJI_PATTERN.sub("", hook).strip()
    words = [w for w in no_emoji_text.split() if w]
    if len(words) < 2:
        return False, f"Hook has too few words ({len(words)} < 2 words)"
    if len(words) > 8:
        return False, f"Hook exceeds maximum word count ({len(words)} > 8 words)"

    # 5. ASS subtitle override & control syntax check
    if "{" in raw_hook or "}" in raw_hook or "\\N" in raw_hook or "\\n" in raw_hook or "\n" in raw_hook or "\r" in raw_hook or "\t" in raw_hook:
        return False, "Hook contains raw ASS control syntax or newline/tab codes"

    # 6. Filename / source title leak check
    if is_filename_or_title_leak(hook, filename=filename, source_title=source_title):
        return False, "Hook copies source filename or source title"

    # 7. Truncation and dangling prepositions check
    if hook.endswith("...") or hook.endswith("…") or hook.endswith("-") or hook.endswith(":"):
        return False, "Hook appears truncated with trailing punctuation"
    last_word = words[-1].lower() if words else ""
    if last_word in {"with", "and", "or", "the", "a", "an", "is", "ki", "ka", "ke", "ko"}:
        return False, f"Hook ends prematurely with dangling conjunction/preposition ('{last_word}')"

    # 8. Generic placeholder rejection
    hook_lower = hook.lower()
    for ph in FORBIDDEN_PLACEHOLDERS:
        if ph in hook_lower:
            return False, f"Hook contains forbidden placeholder ('{ph}')"

    # 9. Emoji limit (max 1 emoji)
    n_emojis = count_emojis(hook)
    if n_emojis > 1:
        return False, f"Hook contains multiple emojis ({n_emojis} > 1 allowed)"

    # 10. Valid hook_type check
    hook_type = candidate.get("hook_type", "curiosity").lower()
    if hook_type not in ALLOWED_HOOK_TYPES:
        candidate["hook_type"] = "curiosity"

    return True, "Valid"


def score_hook(candidate: dict, transcript: str = "", hook_summary: str = "") -> float:
    """
    Calculates authoritative 0–100 score in Python using the exact 6-dimension formula:
    (curiosity * 0.25
    + relevance * 0.25
    + payoff * 0.15
    + clarity * 0.15
    + brevity * 0.10
    + impact * 0.10) * 10
    Each dimension is clamped to 0–10. Python is authoritative.
    """
    def get_dim(*aliases: str, default: float = 7.0) -> float:
        for alias in aliases:
            if alias in candidate:
                try:
                    return max(0.0, min(10.0, float(candidate[alias])))
                except (ValueError, TypeError):
                    pass
            dims = candidate.get("dimensions", {})
            if isinstance(dims, dict) and alias in dims:
                try:
                    return max(0.0, min(10.0, float(dims[alias])))
                except (ValueError, TypeError):
                    pass
        return default

    # Extract dimensions
    curiosity = get_dim("curiosity", default=8.0)
    relevance = get_dim("relevance", "clip_relevance", default=8.0)
    payoff = get_dim("payoff", "payoff_alignment", default=7.5)
    clarity = get_dim("clarity", default=8.5)
    brevity = get_dim("brevity", default=8.0)
    impact = get_dim("impact", default=7.5)

    # Exact weighted calculation
    final_score = (
        curiosity * HOOK_SCORING_WEIGHTS["curiosity"]
        + relevance * HOOK_SCORING_WEIGHTS["relevance"]
        + payoff * HOOK_SCORING_WEIGHTS["payoff"]
        + clarity * HOOK_SCORING_WEIGHTS["clarity"]
        + brevity * HOOK_SCORING_WEIGHTS["brevity"]
        + impact * HOOK_SCORING_WEIGHTS["impact"]
    ) * 10.0

    return max(0.0, min(100.0, round(final_score, 2)))


def get_fallback_hook(detected_lang: str = "en", hook_summary: str = "") -> dict:
    """
    Generates a deterministic, safe, short-form fallback hook appropriate to the language.
    Does NOT use source filename or title.
    """
    lang = (detected_lang or "en").lower()
    is_hindi_hinglish = lang.startswith("hi") or lang in {"hinglish", "mr", "ur"}

    if is_hindi_hinglish:
        fallbacks = [
            {"hook": "YE BAAT KYUN IMPORTANT HAI?", "hook_type": "question"},
            {"hook": "ASLI BAAT KYA HAI?", "hook_type": "curiosity"},
            {"hook": "ISKA MATLAB KYA HAI?", "hook_type": "insight"},
        ]
    else:
        fallbacks = [
            {"hook": "WHY DOES THIS MATTER?", "hook_type": "question"},
            {"hook": "WHAT DOES THIS MEAN?", "hook_type": "insight"},
            {"hook": "THE KEY POINT", "hook_type": "curiosity"},
        ]

    selected = fallbacks[0]
    return {
        "hook": selected["hook"],
        "hook_type": selected["hook_type"],
        "reason": "Deterministic safe fallback based on detected language",
        "score": 70.0,
        "source": "fallback",
        "supported_by_clip": True,
    }


def generate_hook_candidates(
    client: OpenAI,
    model: str,
    transcript: str,
    hook_summary: str = "",
    detected_lang: str = "en",
    filename: str = "",
    source_title: str = "",
) -> List[dict]:
    """
    Makes ONE single LLM call to generate exactly 5 candidate hooks.
    """
    clean_t = transcript.strip()[:3000] if transcript else ""
    lang_guidance = (
        "The audio language is Hindi/Hinglish. Hooks MUST be in natural Roman Hindi / Hinglish "
        "(e.g., 'YE BAAT KYUN IMPORTANT HAI?', 'ASLI BAAT KYA HAI?', 'YE KAISE POSSIBLE HAI?'). "
        "Preserve English technical/context terms naturally."
        if (detected_lang and (detected_lang.startswith("hi") or detected_lang == "hinglish"))
        else "The audio language is English. Hooks MUST be in natural, punchy English."
    )

    prompt = f"""You are an elite YouTube Shorts retention and top-hook specialist.
Your task is to generate exactly 5 distinct, high-impact top-overlay hook candidates for a YouTube Short based STRICTLY on the clip transcript below.

Clip Context:
- Language: {detected_lang}
- Hook Summary: "{hook_summary}"
- Transcript:
\"\"\"
{clean_t}
\"\"\"

CRITICAL CONSTRAINTS FOR EVERY HOOK:
1. Target length: 3 to 8 words. Maximum 8 words!
2. Hard maximum length: 42 characters total. NEVER exceed 42 characters!
3. Grounded in actual content: The hook must describe the actual selected clip and be supported by the clip (supported_by_clip: true).
4. No clickbait/hallucinations: Do not invent allegations, crime claims, secret plots, or facts not present in the clip.
5. Never use the source filename or video title as the hook.
6. Emoji: At most 1 relevant emoji (e.g. 🔥, 👀, ⚠️, 🤯, 🇮🇳). Zero emojis is also fine. Never use 2 or more emojis.
7. Language requirement: {lang_guidance}
8. Allowed hook types: curiosity, question, revelation, unexpected_fact, consequence, debate, explanation, emotional, insight.

Respond ONLY with valid JSON containing exactly 5 candidates in this exact schema:
{{
  "candidates": [
    {{
      "hook": "...",
      "hook_type": "curiosity",
      "curiosity": 8.5,
      "relevance": 9.0,
      "payoff": 8.0,
      "clarity": 9.0,
      "brevity": 8.5,
      "impact": 8.0,
      "supported_by_clip": true,
      "support_reason": "Explains why this hook aligns with the payoff in the transcript"
    }}
  ]
}}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.65,
        max_tokens=850,
        response_format={"type": "json_object"} if ("llama" in model.lower() or "gpt" in model.lower()) else None,
    )

    content = response.choices[0].message.content.strip()
    content = re.sub(r"^```json\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    data = json.loads(content)

    candidates = data.get("candidates", [])
    if not isinstance(candidates, list):
        return []

    return candidates


def generate_short_hook(
    transcript: str,
    hook_summary: str = "",
    punchline: str = "",
    filename: str = "",
    source_title: str = "",
    start_time: float = 0.0,
    end_time: float = 0.0,
    detected_lang: str = "en",
) -> dict:
    """
    Main entry point for Hook Upgrade #1:
    1. Generates 5 candidates in 1 single LLM call.
    2. Validates each candidate with deterministic Python rules.
    3. Scores validated candidates using the authoritative 6-dimension weighted formula.
    4. Deduplicates and selects the highest-scoring candidate.
    5. Falls back to a safe, language-appropriate short hook if AI fails.
    """
    api_key = GROQ_API_KEY or OPENAI_API_KEY

    # If no API key is present, return safe fallback immediately
    if not api_key:
        log.warning("No AI API key found. Using deterministic fallback hook.")
        fb = get_fallback_hook(detected_lang=detected_lang, hook_summary=hook_summary)
        return {
            "selected_hook": fb["hook"],
            "generated_hook": fb["hook"],
            "hook_type": fb["hook_type"],
            "score": fb["score"],
            "candidates": [fb],
            "source": "fallback",
        }

    # Initialize client
    try:
        if api_key.startswith("gsk_") or GROQ_API_KEY:
            client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
            model = GROQ_CHAT_MODEL or _get_best_groq_model(client)
        else:
            client = OpenAI(api_key=api_key)
            model = "gpt-4o-mini"

        raw_candidates = generate_hook_candidates(
            client=client,
            model=model,
            transcript=transcript,
            hook_summary=hook_summary,
            detected_lang=detected_lang,
            filename=filename,
            source_title=source_title,
        )
    except Exception as e:
        log.warning("LLM hook generation call failed (%s). Using fallback.", e)
        fb = get_fallback_hook(detected_lang=detected_lang, hook_summary=hook_summary)
        return {
            "selected_hook": fb["hook"],
            "generated_hook": fb["hook"],
            "hook_type": fb["hook_type"],
            "score": fb["score"],
            "candidates": [fb],
            "source": "fallback",
        }

    # Deterministic Python validation and scoring
    validated_candidates = []
    seen_hooks = set()

    for cand in raw_candidates:
        if not isinstance(cand, dict):
            continue

        raw_hook = clean_hook_text(cand.get("hook", ""))
        cand["hook"] = raw_hook

        # Deduplicate: normalize whitespace, lowercase
        norm_key = re.sub(r"\s+", " ", raw_hook.lower().strip())
        if not norm_key or norm_key in seen_hooks:
            continue

        is_valid, reason = validate_hook(
            cand,
            transcript=transcript,
            filename=filename,
            source_title=source_title,
        )
        if not is_valid:
            log.info("Rejected hook '%s': %s", raw_hook, reason)
            continue

        seen_hooks.add(norm_key)
        final_score = score_hook(cand, transcript=transcript, hook_summary=hook_summary)
        cand["score"] = final_score
        validated_candidates.append(cand)

    # Sort validated candidates by authoritative score descending
    validated_candidates.sort(key=lambda c: c.get("score", 0.0), reverse=True)

    if validated_candidates:
        best = validated_candidates[0]
        log.info(
            "Selected best AI hook: '%s' (type: %s, score: %.1f)",
            best["hook"], best.get("hook_type", "curiosity"), best["score"],
        )
        return {
            "selected_hook": best["hook"],
            "generated_hook": best["hook"],
            "hook_type": best.get("hook_type", "curiosity"),
            "score": best["score"],
            "candidates": validated_candidates,
            "source": "ai",
        }

    # If all candidates failed validation, use safe fallback
    log.warning("All LLM hook candidates failed validation. Using safe fallback.")
    fb = get_fallback_hook(detected_lang=detected_lang, hook_summary=hook_summary)
    return {
        "selected_hook": fb["hook"],
        "generated_hook": fb["hook"],
        "hook_type": fb["hook_type"],
        "score": fb["score"],
        "candidates": [fb],
        "source": "fallback",
    }
