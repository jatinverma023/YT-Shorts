"""
AI Hook Generation and Validation for YouTube Shorts.
Generates, validates, and ranks short, high-curiosity, context-grounded hooks
for YouTube Shorts top header overlays (HeaderPunchline ASS style).

CRITICAL "CLIP ONLY" REQUIREMENT:
The hook-generation model receives ONLY the selected clip transcript.
It never receives source filenames, video titles, surrounding full-video
transcripts, or external metadata.

Key Constraints:
- 3–8 words, hard maximum 42 characters
- Grounded strictly in the clip's actual transcript content (supported_by_clip == True)
- Short verbatim supporting_text excerpt from clip verified in Python
- Rejection of generic placeholders ("WAIT FOR THE TWIST", "THE TRUTH EXPOSED", etc.)
- 6-dimension scoring: Grounding (30%), Specificity (20%), Curiosity (20%),
  Relevance (15%), Clarity (10%), Brevity (5%)
- Max 1 emoji allowed
- Natural Roman Hindi/Hinglish for Hindi clips, natural English for English clips
- Safe, non-hallucinatory transcript-grounded fallback when AI fails
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

# Allowed hook types based on actual clip content
ALLOWED_HOOK_TYPES = {
    "question",
    "surprising_fact",
    "insight",
    "strong_claim",
    "myth_reality",
    "explanation",
    "cause_effect",
    "contradiction",
    "curiosity",
    "revelation",
    "unexpected_fact",
    "consequence",
    "debate",
    "emotional",
}

# Authoritative Grounded 6-Dimension Hook Scoring Weights (must sum to 1.0)
# Grounding / factual support: 30%
# Specificity: 20%
# Curiosity: 20%
# Relevance to clip payoff: 15%
# Clarity: 10%
# Brevity: 5%
HOOK_SCORING_WEIGHTS = {
    "grounding": 0.30,
    "specificity": 0.20,
    "curiosity": 0.20,
    "relevance": 0.15,
    "clarity": 0.10,
    "brevity": 0.05,
}

# Legacy scoring weights for backward compatibility with ungrounded test fixtures
LEGACY_HOOK_SCORING_WEIGHTS = {
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

# Patterns for generic curiosity templates without a concrete subject
FORBIDDEN_GENERIC_PATTERNS = [
    r"\bwait\s+for\s+(the\s+)?twist\b",
    r"\b(the\s+)?truth\s+(exposed|revealed)\b",
    r"\breality\s+check(\s+revealed)?\b",
    r"\bmust\s+watch(\s+insight)?\b",
    r"\byou\s+won'?t\s+believe\s+this\b",
    r"\byou\s+wont\s+believe\s+this\b",
    r"\bthis\s+changes\s+everything\b",
    r"\bshocking\s+truth\b",
    r"\b(the\s+)?secret\s+revealed\b",
    r"\bwhat\s+happens\s+next\b",
    r"\byou\s+need\s+to\s+know\s+this\b",
    r"\bthis\s+is\s+crazy\b",
    r"\bthe\s+answer\s+will\s+shock\s+you\b",
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


def normalize_text_for_matching(text: str) -> str:
    """Normalizes text for fuzzy/substring matching by stripping punctuation and extra whitespace."""
    if not text:
        return ""
    t = re.sub(r"[^\w\s]", " ", str(text).lower())
    return re.sub(r"\s+", " ", t).strip()


def is_supporting_text_in_transcript(supporting_text: str, transcript: str) -> bool:
    """
    Verifies that supporting_text actually exists verbatim or near-verbatim in the clip transcript.
    """
    if not supporting_text or not transcript:
        return False
    norm_support = normalize_text_for_matching(supporting_text)
    norm_transcript = normalize_text_for_matching(transcript)
    if not norm_support or not norm_transcript:
        return False

    # 1. Exact normalized substring match
    if norm_support in norm_transcript:
        return True

    # 2. Match with 3+ contiguous words window
    words = norm_support.split()
    if len(words) >= 3:
        for i in range(len(words) - 2):
            sub = " ".join(words[i : i + 3])
            if sub in norm_transcript:
                return True

    # 3. High word-overlap check for phrases with 4+ words (>= 80% words found in order)
    if len(words) >= 4:
        found_count = sum(1 for w in words if f" {w} " in f" {norm_transcript} ")
        if found_count / len(words) >= 0.8:
            return True

    return False


def is_forbidden_generic_hook(hook_text: str, transcript: str = "") -> bool:
    """
    Detects if a hook is a forbidden generic clickbait template without a concrete subject.
    Unless the actual clip explicitly contains that specific wording as a genuine topic,
    generic curiosity templates are strictly rejected.
    """
    clean_hook = EMOJI_PATTERN.sub("", hook_text).strip().lower()
    norm_hook = re.sub(r"[^\w\s]", " ", clean_hook)
    norm_hook = re.sub(r"\s+", " ", norm_hook).strip()
    norm_transcript = normalize_text_for_matching(transcript)

    for pattern in FORBIDDEN_GENERIC_PATTERNS:
        match = re.search(pattern, norm_hook, flags=re.IGNORECASE)
        if match:
            matched_phrase = match.group(0)
            # Only allow if the clip transcript itself explicitly mentions the exact phrase
            if norm_transcript and matched_phrase in norm_transcript:
                return False
            return True

    for ph in FORBIDDEN_PLACEHOLDERS:
        if ph in norm_hook:
            if norm_transcript and ph in norm_transcript:
                return False
            return True

    return False


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


def extract_grounded_fallback_hook(transcript: str, detected_lang: str = "en") -> Tuple[str, str]:
    """
    Extracts a clean, punchy, transcript-grounded hook directly from the clip transcript
    when LLM generation is unavailable or fails.
    Returns (hook_text, supporting_text).
    """
    if not transcript or not transcript.strip():
        return ("", "")

    # Split into candidate clauses by punctuation
    raw_clauses = re.split(r"[.!?,\n;]+", transcript)
    clean_clauses = [c.strip() for c in raw_clauses if c.strip()]

    # 1. Prefer questions if any exist in the clip transcript
    for c in clean_clauses:
        words = c.split()
        if 3 <= len(words) <= 8 and len(c) <= MAX_HOOK_CHARS:
            last = words[-1].lower()
            if last not in {"with", "and", "or", "the", "a", "an", "is", "ki", "ka", "ke", "ko"}:
                first = words[0].lower()
                if first in {"why", "how", "what", "is", "can", "kya", "kyun", "kaise"} or "?" in c:
                    hook = c.upper()
                    if not hook.endswith("?"):
                        hook += "?"
                    return hook, c

    # 2. Look for any standalone clause with 3-8 words and <= 42 chars
    for c in clean_clauses:
        words = c.split()
        if 3 <= len(words) <= 8 and len(c) <= MAX_HOOK_CHARS:
            last = words[-1].lower()
            if last not in {"with", "and", "or", "the", "a", "an", "is", "ki", "ka", "ke", "ko"}:
                return c.upper(), c

    # 3. Take the first clause and slice to 3-6 words <= 42 chars
    if clean_clauses:
        first_clause = clean_clauses[0]
        words = first_clause.split()
        for count in range(min(7, len(words)), 2, -1):
            sub = " ".join(words[:count])
            last = words[count - 1].lower()
            if len(sub) <= MAX_HOOK_CHARS and last not in {"with", "and", "or", "the", "a", "an", "is", "ki", "ka", "ke", "ko"}:
                return sub.upper(), sub

    return ("", "")


def validate_hook(
    candidate: dict,
    transcript: str = "",
    filename: str = "",
    source_title: str = "",
    require_supporting_text: bool = False,
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

    # 8. Forbidden generic placeholder & clickbait template rejection
    if is_forbidden_generic_hook(hook, transcript=transcript):
        return False, f"Hook is a forbidden generic clickbait template without concrete subject ('{hook}')"

    # 9. Emoji limit (max 1 emoji)
    n_emojis = count_emojis(hook)
    if n_emojis > 1:
        return False, f"Hook contains multiple emojis ({n_emojis} > 1 allowed)"

    # 10. Supporting text grounding verification
    support = candidate.get("supporting_text", "").strip()
    if support and transcript:
        if not is_supporting_text_in_transcript(support, transcript):
            return False, f"Supporting text was not found in the clip transcript: '{support[:40]}...'"
    elif require_supporting_text and transcript:
        return False, "Candidate is missing required supporting_text from clip transcript"

    # 11. Valid hook_type check
    hook_type = candidate.get("hook_type", "curiosity").lower()
    if hook_type not in ALLOWED_HOOK_TYPES:
        candidate["hook_type"] = "curiosity"

    return True, "Valid"


def score_hook(candidate: dict, transcript: str = "", hook_summary: str = "") -> float:
    """
    Calculates authoritative 0–100 score in Python using the exact weighted formula:
    Grounding / factual support: 30%
    Specificity: 20%
    Curiosity: 20%
    Relevance to clip payoff: 15%
    Clarity: 10%
    Brevity: 5%
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

    has_grounding_schema = any(
        k in candidate for k in ("grounding", "grounding_score", "specificity", "specificity_score", "supporting_text")
    )

    if has_grounding_schema:
        # Check supporting text grounding
        support = candidate.get("supporting_text", "")
        if support and transcript:
            if is_supporting_text_in_transcript(support, transcript):
                grounding = get_dim("grounding_score", "grounding", "factual_support", default=9.5)
            else:
                grounding = 1.0  # ungrounded penalty
        else:
            grounding = get_dim("grounding_score", "grounding", "factual_support", default=8.5)

        specificity = get_dim("specificity_score", "specificity", default=8.0)
        curiosity = get_dim("curiosity_score", "curiosity", default=8.0)
        relevance = get_dim("relevance_score", "relevance", "payoff_relevance", "payoff", default=8.0)
        clarity = get_dim("clarity_score", "clarity", default=8.5)
        brevity = get_dim("brevity_score", "brevity", default=8.5)

        final_score = (
            grounding * HOOK_SCORING_WEIGHTS["grounding"]
            + specificity * HOOK_SCORING_WEIGHTS["specificity"]
            + curiosity * HOOK_SCORING_WEIGHTS["curiosity"]
            + relevance * HOOK_SCORING_WEIGHTS["relevance"]
            + clarity * HOOK_SCORING_WEIGHTS["clarity"]
            + brevity * HOOK_SCORING_WEIGHTS["brevity"]
        ) * 10.0
    else:
        # Legacy formula for backward compatibility with ungrounded test fixtures:
        # (curiosity*0.25 + relevance*0.25 + payoff*0.15 + clarity*0.15 + brevity*0.10 + impact*0.10) * 10
        curiosity = get_dim("curiosity", default=8.0)
        relevance = get_dim("relevance", "clip_relevance", default=8.0)
        payoff = get_dim("payoff", "payoff_alignment", default=7.5)
        clarity = get_dim("clarity", default=8.5)
        brevity = get_dim("brevity", default=8.0)
        impact = get_dim("impact", default=7.5)

        final_score = (
            curiosity * LEGACY_HOOK_SCORING_WEIGHTS["curiosity"]
            + relevance * LEGACY_HOOK_SCORING_WEIGHTS["relevance"]
            + payoff * LEGACY_HOOK_SCORING_WEIGHTS["payoff"]
            + clarity * LEGACY_HOOK_SCORING_WEIGHTS["clarity"]
            + brevity * LEGACY_HOOK_SCORING_WEIGHTS["brevity"]
            + impact * LEGACY_HOOK_SCORING_WEIGHTS["impact"]
        ) * 10.0

    return max(0.0, min(100.0, round(final_score, 2)))


def get_fallback_hook(transcript: str = "", detected_lang: str = "en", hook_summary: str = "") -> dict:
    """
    Generates a deterministic, safe fallback hook grounded directly in the clip transcript.
    Never returns generic clickbait like 'WAIT FOR THE TWIST' or 'THE TRUTH EXPOSED'.
    """
    hook_text, supporting_text = extract_grounded_fallback_hook(transcript, detected_lang=detected_lang)
    if hook_text and len(transcript.strip().split()) >= 4:
        return {
            "hook": hook_text,
            "hook_type": "insight",
            "supporting_text": supporting_text,
            "reason": "Transcript-grounded deterministic fallback",
            "score": 75.0,
            "source": "fallback_transcript",
            "supported_by_clip": True,
        }

    lang = (detected_lang or "en").lower()
    is_hindi_hinglish = lang.startswith("hi") or lang in {"hinglish", "mr", "ur"}
    default_hook = "YE BAAT KYUN IMPORTANT HAI?" if is_hindi_hinglish else "WHY DOES THIS MATTER?"

    return {
        "hook": default_hook,
        "hook_type": "question",
        "supporting_text": "",
        "reason": "Neutral inquiry fallback",
        "score": 70.0,
        "source": "fallback",
        "supported_by_clip": True,
    }


def generate_hook_candidates(
    client: OpenAI,
    model: str,
    transcript: str,
    detected_lang: str = "en",
    batch_attempt: int = 1,
    **kwargs,
) -> List[dict]:
    """
    Makes ONE single LLM call to generate exactly 5 candidate hooks grounded ONLY in the clip transcript.
    CRITICAL: Never provides filenames, video titles, or outside context to the prompt.
    """
    clean_t = transcript.strip()[:3000] if transcript else ""
    is_hindi = bool(detected_lang and (detected_lang.startswith("hi") or detected_lang == "hinglish"))
    lang_instruction = (
        "LANGUAGE: The clip audio is in Hindi/Hinglish. Generate hooks in natural Roman Hindi / Hinglish "
        "(e.g., 'YE HABIT FERTILITY KO KAISE EFFECT KARTI HAI?', 'KYA SMOKING SE SPERM QUALITY GHAT TI HAI?'). "
        "Preserve English technical/medical/topic terms naturally."
        if is_hindi
        else "LANGUAGE: The clip audio is in English. Generate hooks in natural, punchy, conversational English."
    )

    batch_note = ""
    if batch_attempt > 1:
        batch_note = (
            "NOTE: Previous candidates were rejected for being too generic or failing strict transcript grounding. "
            "You MUST pick a specific, concrete fact, question, or claim verbatim from the transcript below!\n"
        )

    prompt = f"""You are an elite YouTube Shorts retention and top-hook specialist.
Your task is to analyze ONLY the provided clip transcript and generate exactly 5 distinct, highly compelling, strictly grounded top-overlay hooks for a vertical YouTube Short.

{batch_note}CRITICAL "CLIP ONLY" & GROUNDING RULES:
1. Use ONLY the ideas, facts, questions, and claims directly present in the clip transcript below.
2. Do NOT use outside knowledge, general assumptions, or external information.
3. NEVER produce generic clickbait hooks like "WAIT FOR THE TWIST", "THE TRUTH EXPOSED", "REALITY CHECK", "MUST WATCH", "YOU WON'T BELIEVE THIS", "THIS CHANGES EVERYTHING", or "SHOCKING TRUTH".
4. Every hook must feature a CONCRETE, SPECIFIC SUBJECT or CLAIM from the clip (e.g. fertility, smoking, throat cancer, salary negotiation, sleep cycle, etc.).
5. The `supporting_text` MUST be a short verbatim excerpt (3 to 15 words) copied directly from the transcript that directly supports the hook. Do NOT fabricate supporting text.
6. Length: 3 to 8 words. Absolute maximum: 42 characters.
7. Emojis: At most 1 relevant emoji (or 0 emojis). Never use 2 or more emojis.
8. {lang_instruction}

HOOK STYLES TO CHOOSE FROM (Pick the style that best matches what is actually said):
- Specific question (e.g. "CAN ORAL SEX CAUSE THROAT CANCER?")
- Surprising fact (e.g. "SMOKING CAN AFFECT FERTILITY")
- Counterintuitive insight (e.g. "THIS HABIT MAY HURT FERTILITY")
- Strong claim (e.g. "HPV CAN REACH THE THROAT")
- Myth/reality (e.g. "IS THIS FERTILITY MYTH TRUE?")
- Important explanation (e.g. "WHY SMOKING REDUCES FERTILITY")
- Cause/effect (e.g. "THIS CAN LOWER SPERM QUALITY")
- Contradiction (e.g. "THE FERTILITY MYTH IS WRONG")

CLIP TRANSCRIPT (THIS IS YOUR ONLY CONTEXT):
\"\"\"
{clean_t}
\"\"\"

Respond ONLY with a valid JSON object containing exactly 5 candidates in this schema:
{{
  "candidates": [
    {{
      "hook": "SPECIFIC GROUNDED HOOK (UPPERCASE)",
      "hook_type": "question | surprising_fact | insight | strong_claim | myth_reality | explanation | cause_effect | contradiction",
      "supporting_text": "exact verbatim excerpt from the transcript supporting this hook",
      "supported_by_clip": true,
      "grounding_reason": "brief explanation of how this hook reflects the clip's actual idea",
      "curiosity_score": 8.5,
      "specificity_score": 9.0,
      "relevance_score": 9.0,
      "clarity_score": 8.5,
      "grounding_score": 9.5
    }}
  ]
}}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.60,
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
    Main entry point for AI Hook Generation:
    1. Passes ONLY the clip transcript to the model (no filename, no title, no outside metadata).
    2. Generates 5 candidates in Batch 1.
    3. Validates each candidate with deterministic Python rules and transcript-grounding verification.
    4. If all candidates fail, runs a second grounded batch.
    5. Scores validated candidates using the 6-dimension weighted formula.
    6. Returns the highest-scoring candidate, or transcript-grounded fallback if AI fails.
    """
    api_key = GROQ_API_KEY or OPENAI_API_KEY

    # If no API key is present, return safe transcript-grounded fallback immediately
    if not api_key:
        log.warning("No AI API key found. Using transcript-grounded fallback hook.")
        fb = get_fallback_hook(transcript=transcript, detected_lang=detected_lang, hook_summary=hook_summary)
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
    except Exception as e:
        log.warning("Could not initialize AI client (%s). Using fallback.", e)
        fb = get_fallback_hook(transcript=transcript, detected_lang=detected_lang, hook_summary=hook_summary)
        return {
            "selected_hook": fb["hook"],
            "generated_hook": fb["hook"],
            "hook_type": fb["hook_type"],
            "score": fb["score"],
            "candidates": [fb],
            "source": "fallback",
        }

    def process_batch(candidates: List[dict]) -> List[dict]:
        validated = []
        seen = set()
        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            raw_hook = clean_hook_text(cand.get("hook", ""))
            cand["hook"] = raw_hook
            norm_key = re.sub(r"\s+", " ", raw_hook.lower().strip())
            if not norm_key or norm_key in seen:
                continue

            is_valid, reason = validate_hook(
                cand,
                transcript=transcript,
                filename=filename,
                source_title=source_title,
                require_supporting_text=False,
            )
            if not is_valid:
                log.info("Rejected hook candidate '%s': %s", raw_hook, reason)
                continue

            seen.add(norm_key)
            cand["score"] = score_hook(cand, transcript=transcript, hook_summary=hook_summary)
            validated.append(cand)
        return validated

    validated_candidates = []

    # Batch 1
    try:
        raw_candidates_1 = generate_hook_candidates(
            client=client,
            model=model,
            transcript=transcript,
            detected_lang=detected_lang,
            batch_attempt=1,
        )
        validated_candidates = process_batch(raw_candidates_1)
    except Exception as e:
        log.warning("LLM hook generation call Batch 1 failed (%s).", e)

    # Batch 2 if Batch 1 produced 0 valid candidates
    if not validated_candidates:
        log.info("Batch 1 produced no valid grounded hooks. Attempting second grounded batch...")
        try:
            raw_candidates_2 = generate_hook_candidates(
                client=client,
                model=model,
                transcript=transcript,
                detected_lang=detected_lang,
                batch_attempt=2,
            )
            validated_candidates = process_batch(raw_candidates_2)
        except Exception as e:
            log.warning("LLM hook generation call Batch 2 failed (%s).", e)

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

    # If all candidates failed validation across both batches, use safe transcript-grounded fallback
    log.warning("All LLM hook candidates failed validation. Using transcript-grounded fallback.")
    fb = get_fallback_hook(transcript=transcript, detected_lang=detected_lang, hook_summary=hook_summary)
    return {
        "selected_hook": fb["hook"],
        "generated_hook": fb["hook"],
        "hook_type": fb["hook_type"],
        "score": fb["score"],
        "candidates": [fb],
        "source": "fallback",
    }
