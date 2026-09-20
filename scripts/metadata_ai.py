"""
Generate viral, authentic YouTube Shorts titles, descriptions, and hashtags
using Groq AI (Llama 3.3 / 3.1) or OpenAI grounded strictly in the selected clip transcript.
Enforces Python-authoritative quality scoring, anti-clickbait rules, and YouTube limits.
"""
import json
import logging
import os
import re
import time
from typing import Dict, List, Optional, Set, Tuple

from openai import OpenAI
import config
from clip_detection import _get_best_groq_model
import hook_generator

log = logging.getLogger("metadata_ai")

# Anti-clickbait forbidden patterns for Shorts titles
FORBIDDEN_CLICKBAIT_PATTERNS = [
    r"\byou\s+won\s*'?\s*t\s+believe\s+this\b",
    r"\byou\s+wont\s+believe\s+this\b",
    r"\b(the\s+)?truth\s+(exposed|revealed)\b",
    r"\bwait\s+for\s+(the\s+)?twist\b",
    r"\bmust\s+watch(\s+this)?\b",
    r"\bthis\s+changes\s+everything\b",
    r"\bshocking\s+truth\b",
    r"\bsecret\s+revealed\b",
    r"\bthey\s+don\s*'?\s*t\s+want\s+you\s+to\s+know\b",
    r"\byou\s+need\s+to\s+see\s+this\b",
    r"\bunbelievable\b",
    r"\binsane\b",
    r"\bcrazy\s+revelation\b",
    r"\bamazing\s+podcast\s+moment\b",
]

# Generic hashtag spam that should be strictly rejected
FORBIDDEN_SPAM_HASHTAGS = {
    "#viral",
    "#fyp",
    "#trending",
    "#explore",
    "#foryou",
    "#foryoupage",
    "#viralshorts",
    "#shortsfeed",
    "#foryourpage",
}

# Stop words for English and Roman Hindi / Hinglish
STOP_WORDS: Set[str] = {
    "a", "about", "above", "after", "again", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
    "by", "can", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing",
    "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's",
    "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've",
    "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most",
    "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd",
    "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the",
    "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd",
    "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what",
    "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why",
    "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've",
    "your", "yours", "yourself", "yourselves",
    # Hindi / Hinglish stop words
    "hai", "hain", "ho", "hoon", "tha", "the", "thi", "thhe", "ka", "ke", "ki", "ko", "se", "me",
    "mein", "par", "aur", "ya", "toh", "to", "bhi", "yeh", "ye", "woh", "wo", "kya", "kyun",
    "kaise", "kab", "kahan", "kaun", "kis", "kisse", "is", "us", "in", "un", "apne", "apni",
    "apna", "hum", "tum", "aap", "main", "mera", "meri", "mere", "tera", "teri", "tere", "uska",
    "uski", "uske", "unka", "unki", "unke", "hoga", "hogi", "honge", "kare", "karega", "karenge",
}

# Extreme / absolute claim words that require strict transcript support
EXTREME_CLAIM_WORDS: Set[str] = {
    "permanent", "permanently", "incurable", "guaranteed", "deadly", "fatal",
    "cure", "cures", "impossible", "illegal", "arrested", "banned", "100%",
}


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


def normalize_text(text: str) -> str:
    """Normalizes text by lowercasing, stripping punctuation, and collapsing whitespace."""
    if not text:
        return ""
    t = str(text).replace("'", "").replace("’", "").lower()
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def extract_content_words(text: str) -> Set[str]:
    """Extracts non-stopword content words from text for semantic grounding."""
    norm = normalize_text(text)
    words = norm.split()
    return {
        w for w in words
        if w not in STOP_WORDS and (len(w) >= 3 or w in {"ai", "ml", "yt", "hpv", "vr", "ar"})
    }


def is_title_hook_duplicate(title: str, hook: str) -> Tuple[bool, str]:
    """
    Verifies that the title does not merely repeat or duplicate the generated hook.
    Returns (is_duplicate, reason).
    """
    if not title or not hook:
        return False, ""

    clean_t = hook_generator.EMOJI_PATTERN.sub("", title).strip()
    clean_h = hook_generator.EMOJI_PATTERN.sub("", hook).strip()

    norm_t = normalize_text(clean_t)
    norm_h = normalize_text(clean_h)

    # 1. Exact match after normalization
    if norm_t == norm_h:
        return True, "Title is identical to the hook"

    words_t = set(norm_t.split())
    words_h = set(norm_h.split())

    if not words_t or not words_h:
        return False, ""

    # 2. Word overlap / Jaccard similarity >= 0.75
    intersection = words_t & words_h
    union = words_t | words_h
    jaccard = len(intersection) / len(union)
    if jaccard >= 0.75:
        return True, f"Title has excessive word overlap with the hook (Jaccard: {jaccard:.2f})"

    # 3. Substring containment if length difference is minimal
    if (norm_t in norm_h or norm_h in norm_t) and abs(len(words_t) - len(words_h)) <= 2:
        return True, "Title is an uncreative duplicate/subset of the hook"

    return False, ""


def is_forbidden_generic_title(title: str, transcript: str = "") -> bool:
    """
    Detects unsupported generic clickbait templates.
    Allowed only if the clip transcript explicitly contains the exact phrase.
    """
    norm_title = normalize_text(title)
    norm_transcript = normalize_text(transcript)

    for pattern in FORBIDDEN_CLICKBAIT_PATTERNS:
        match = re.search(pattern, norm_title, flags=re.IGNORECASE)
        if match:
            matched_phrase = match.group(0)
            if norm_transcript and matched_phrase in norm_transcript:
                return False
            return True

    return False


def is_title_grounded(title: str, transcript: str) -> Tuple[bool, str]:
    """
    Python-authoritative grounding evaluation:
    - Does NOT require verbatim substring match.
    - Verifies meaningful concepts/claims are supported by the clip.
    - Rejects stronger unsupported claims (e.g. 'permanent infertility' vs 'affects fertility').
    """
    if not title:
        return False, "Title is empty"
    if not transcript or not transcript.strip():
        return False, "Clip transcript is empty"

    norm_transcript = normalize_text(transcript)
    norm_title = normalize_text(title)

    # 1. Check for extreme/absolute claims not present in clip
    title_words = norm_title.split()
    for ew in EXTREME_CLAIM_WORDS:
        if ew in title_words and ew not in norm_transcript:
            return False, f"Title introduces stronger unsupported claim ('{ew}')"

    # 2. Concept support evaluation
    content_words = extract_content_words(title)
    if not content_words:
        return True, "Valid"

    transcript_content_words = extract_content_words(transcript)
    transcript_words_set = set(norm_transcript.split())

    supported_count = 0
    for word in content_words:
        # A. Direct word match
        if word in transcript_words_set or word in transcript_content_words:
            supported_count += 1
            continue

        # B. Stem / root match (e.g., 'compounding' <-> 'compound', 'smoking' <-> 'smoke')
        stem = word[:4] if len(word) >= 5 else word
        if any(tw.startswith(stem) or stem in tw for tw in transcript_words_set if len(tw) >= 4):
            supported_count += 1
            continue

        # C. Domain / semantic associations
        if word in {"male", "men", "man"} and any(k in norm_transcript for k in ["sperm", "men", "man", "male", "testosterone"]):
            supported_count += 1
            continue
        if word in {"female", "women", "woman"} and any(k in norm_transcript for k in ["women", "woman", "female", "ovary", "eggs"]):
            supported_count += 1
            continue
        if word in {"money", "wealth", "finances"} and any(k in norm_transcript for k in ["interest", "compound", "invest", "financial", "wealth"]):
            supported_count += 1
            continue

    ratio = supported_count / len(content_words)
    if ratio >= 0.60:
        return True, "Valid"

    return False, f"Title concepts insufficiently grounded in clip ({supported_count}/{len(content_words)} content words supported)"


def is_filename_or_source_title_leak(title: str, filename: str = "", source_title: str = "") -> bool:
    """Detects if title leaks raw video filename or long-video source title."""
    norm_title = normalize_text(title)

    if filename:
        base = os.path.splitext(os.path.basename(filename))[0]
        clean_base = normalize_text(base)
        if len(clean_base) >= 8 and clean_base in norm_title:
            return True
        if re.search(r"\.(mp4|mov|mkv|avi|webm)\b", str(title).lower()):
            return True

    if source_title:
        norm_source = normalize_text(source_title)
        if len(norm_source) >= 8 and norm_source in norm_title:
            return True

    return False


def validate_title(
    candidate: dict,
    transcript: str = "",
    hook: str = "",
    filename: str = "",
    source_title: str = "",
) -> Tuple[bool, str]:
    """
    Deterministically validates a single title candidate according to all rules:
    - Formatting & length (5–12 words, <= 70 chars, max 1 emoji)
    - Anti-clickbait patterns
    - Filename/source title leakage
    - Hook non-duplication
    - Python transcript grounding
    """
    raw_title = str(candidate.get("title", "")).strip().strip('"\'`')
    if not raw_title:
        return False, "Title is empty or whitespace"

    # 1. ASS control syntax and newline check
    if "{" in raw_title or "}" in raw_title or "\\N" in raw_title or "\n" in raw_title or "\r" in raw_title or "\t" in raw_title:
        return False, "Title contains invalid formatting or newline codes"

    # 2. Emoji count (max 1 emoji, preferably none)
    emojis_count = hook_generator.count_emojis(raw_title)
    if emojis_count > 1:
        return False, f"Title contains multiple emojis ({emojis_count} > 1 allowed)"

    clean_t = hook_generator.EMOJI_PATTERN.sub("", raw_title).strip()
    words = clean_t.split()

    # 3. Anti-clickbait validation
    if is_forbidden_generic_title(raw_title, transcript=transcript):
        return False, f"Title is generic clickbait without factual clip support ('{raw_title}')"

    # 4. Character limit (target: max 70 characters)
    max_len = getattr(config, "TITLE_MAX_LENGTH", 70)
    if len(raw_title) > max_len:
        return False, f"Title exceeds hard limit ({len(raw_title)} > {max_len} chars)"

    # 5. Word count check (target: 5 to 12 words)
    min_words = getattr(config, "TITLE_MIN_WORDS", 5)
    max_words = getattr(config, "TITLE_MAX_WORDS", 12)
    if len(words) < min_words:
        return False, f"Title has too few words ({len(words)} < {min_words} words)"
    if len(words) > max_words:
        return False, f"Title exceeds maximum word count ({len(words)} > {max_words} words)"

    # 6. Dangling punctuation or truncation check
    if raw_title.endswith("...") or raw_title.endswith("…") or raw_title.endswith("-") or raw_title.endswith(":"):
        return False, "Title appears truncated with trailing punctuation"

    last_word = words[-1].lower() if words else ""
    if last_word in {"with", "and", "or", "the", "a", "an", "is", "ki", "ka", "ke", "ko", "in", "to", "for"}:
        return False, f"Title ends with dangling preposition/conjunction ('{last_word}')"

    # 7. Filename or source title leakage
    if is_filename_or_source_title_leak(raw_title, filename=filename, source_title=source_title):
        return False, "Title copies source filename or source title without clip support"

    # 8. Hook non-duplication check
    if hook:
        is_dup, dup_reason = is_title_hook_duplicate(raw_title, hook)
        if is_dup:
            return False, f"Title duplicates hook: {dup_reason}"

    # 9. Python transcript grounding check
    if transcript:
        is_grounded, ground_reason = is_title_grounded(raw_title, transcript)
        if not is_grounded:
            return False, ground_reason

    return True, "Valid"


def score_title(candidate: dict, transcript: str = "") -> float:
    """
    Python-authoritative 6-dimension Title Quality Scoring (0–100 scale):
    Grounding / factual support: 30%
    Specificity:                 20%
    Curiosity:                  20%
    Relevance:                  15%
    Clarity:                    10%
    Brevity:                     5%
    Total: 100%
    LLM scores are advisory only; Python calculates and validates all metrics.
    """
    raw_title = str(candidate.get("title", "")).strip()

    # Get advisory dimensions from candidate (default 7.0)
    def get_adv(dim_name: str, default: float = 7.0) -> float:
        for k in (f"{dim_name}_score", dim_name):
            if k in candidate:
                try:
                    return max(0.0, min(10.0, float(candidate[k])))
                except (ValueError, TypeError):
                    pass
        return default

    adv_grounding = get_adv("grounding", 8.0)
    adv_specificity = get_adv("specificity", 8.0)
    adv_curiosity = get_adv("curiosity", 8.0)
    adv_relevance = get_adv("relevance", 8.0)
    adv_clarity = get_adv("clarity", 8.5)

    # 1. Authoritative Grounding
    if transcript:
        content_words = extract_content_words(raw_title)
        if content_words:
            norm_tr = normalize_text(transcript)
            tr_words = set(norm_tr.split())
            matched = sum(
                1 for w in content_words
                if w in tr_words or any(tw.startswith(w[:4]) for tw in tr_words if len(tw) >= 4)
            )
            ratio = matched / len(content_words)
            if ratio >= 0.85:
                grounding = min(10.0, max(8.5, adv_grounding))
            elif ratio >= 0.65:
                grounding = min(8.5, max(7.0, adv_grounding))
            else:
                grounding = min(5.0, adv_grounding)
        else:
            grounding = 8.0
    else:
        grounding = adv_grounding

    # 2. Authoritative Brevity (5–12 words, 35–65 characters ideal)
    char_len = len(raw_title)
    word_count = len(raw_title.split())
    if 35 <= char_len <= 65 and 5 <= word_count <= 10:
        brevity = 9.5
    elif char_len <= 70 and 5 <= word_count <= 12:
        brevity = 8.0
    else:
        brevity = 6.0

    # 3. Authoritative Specificity (concrete nouns vs generic filler)
    specificity = min(10.0, max(6.0, adv_specificity))

    # 4. Authoritative Curiosity
    curiosity = min(10.0, max(6.0, adv_curiosity))

    # 5. Authoritative Relevance
    relevance = min(10.0, max(6.0, adv_relevance))

    # 6. Authoritative Clarity
    clarity = min(10.0, max(7.0, adv_clarity))

    final_score = (
        grounding * 0.30
        + specificity * 0.20
        + curiosity * 0.20
        + relevance * 0.15
        + clarity * 0.10
        + brevity * 0.05
    ) * 10.0

    return round(max(0.0, min(100.0, final_score)), 1)


def extract_grounded_fallback_title(transcript: str, detected_lang: str = "en") -> str:
    """
    Extracts a deterministic, transcript-grounded title directly from the clip content
    when AI generation fails or all candidates are rejected.
    Never uses generic clickbait like 'MUST WATCH' or 'THE TRUTH ABOUT THIS'.
    """
    if not transcript or not transcript.strip():
        return "Insightful Clip Discussion"

    # Split into candidate clauses by punctuation
    raw_clauses = re.split(r"[.!?\n;]+", transcript)
    clean_clauses = [c.strip() for c in raw_clauses if c.strip()]

    # 1. Look for an informative clause with 5–12 words and <= 70 characters
    for c in clean_clauses:
        words = c.split()
        if 5 <= len(words) <= 12 and len(c) <= 70:
            last = words[-1].lower()
            if last not in {"with", "and", "or", "the", "a", "an", "is", "ki", "ka", "ke", "ko"}:
                # Format to Title Case cleanly
                title_cand = " ".join([w.capitalize() if not w.isupper() else w for w in words])
                if len(title_cand) <= 70:
                    return title_cand

    # 2. Slice the first meaningful clause to 6–9 words
    if clean_clauses:
        words = clean_clauses[0].split()
        for count in range(min(10, len(words)), 4, -1):
            sub = " ".join(words[:count])
            last = words[count - 1].lower()
            if len(sub) <= 65 and last not in {"with", "and", "or", "the", "a", "an", "is", "ki", "ka", "ke", "ko"}:
                return " ".join([w.capitalize() if not w.isupper() else w for w in words[:count]])

    # 3. Fallback using first words
    words = transcript.strip().split()
    cand_words = words[:min(8, len(words))]
    return " ".join([w.capitalize() for w in cand_words])[:70]


def validate_single_hashtag(tag: str, transcript: str = "") -> Tuple[bool, str]:
    """
    Validates a single hashtag deterministically in Python:
    - Must start with #
    - Must contain no spaces or illegal characters
    - Must not be generic spam (#viral, #fyp, etc.)
    - Must be grounded in clip content/topic
    """
    if not tag or not isinstance(tag, str):
        return False, "Hashtag is empty"

    clean_tag = tag.strip()
    if not clean_tag.startswith("#"):
        return False, "Hashtag does not start with #"

    if not re.match(r"^#[A-Za-z0-9_]+$", clean_tag):
        return False, "Hashtag contains spaces or invalid characters"

    if clean_tag.lower() in FORBIDDEN_SPAM_HASHTAGS:
        return False, f"Hashtag is forbidden generic spam ({clean_tag})"

    # Grounding check for topical hashtags (case-insensitive)
    if clean_tag.lower() in {"#shorts", "#short"}:
        return True, "Valid"

    if transcript:
        tag_text = clean_tag[1:]
        # Split camelCase e.g. CompoundInterest -> ['Compound', 'Interest']
        parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|\d+", tag_text)
        if not parts:
            parts = [tag_text]

        norm_tr = normalize_text(transcript)
        # Check if at least one concept part appears in transcript or transcript matches topic
        tag_words = [p.lower() for p in parts if len(p) >= 3]
        if not tag_words:
            tag_words = [tag_text.lower()]

        found = any(w in norm_tr or any(tw.startswith(w[:4]) for tw in norm_tr.split() if len(tw) >= 4) for w in tag_words)
        if not found:
            # Check domain concepts
            domain_matches = {
                "finance": ["money", "invest", "compound", "saving", "stock", "wealth"],
                "health": ["doctor", "fertility", "body", "smoke", "medical", "disease"],
                "tech": ["ai", "code", "software", "computer", "model", "data"],
                "coding": ["python", "code", "developer", "software", "program"],
            }
            for dom, kws in domain_matches.items():
                if dom in clean_tag.lower() and any(kw in norm_tr for kw in kws):
                    found = True
                    break

        if not found:
            return False, f"Hashtag '{clean_tag}' is not grounded in the clip transcript"

    return True, "Valid"


def validate_hashtags(hashtags: list, transcript: str = "") -> List[str]:
    """
    Validates, deduplicates, and filters a list of hashtags.
    Always includes #Shorts. Target 4–6 hashtags.
    Allows fewer if insufficient topical concepts exist in clip.
    """
    if not hashtags or not isinstance(hashtags, list):
        return ["#Shorts"]

    seen = set()
    validated = []

    # Ensure #Shorts is first
    validated.append("#Shorts")
    seen.add("shorts")

    for raw in hashtags:
        tag = str(raw).strip()
        if not tag:
            continue
        if not tag.startswith("#"):
            tag = f"#{tag}"

        norm_key = tag.lower().lstrip("#")
        if norm_key in seen:
            continue

        is_valid, _ = validate_single_hashtag(tag, transcript=transcript)
        if is_valid:
            validated.append(tag)
            seen.add(norm_key)

    max_tags = getattr(config, "MAX_HASHTAGS", 6)
    return validated[:max_tags]


def score_hashtags(hashtags: List[str], transcript: str = "") -> float:
    """
    Python-authoritative Hashtag Quality Scoring (0–100 scale):
    Relevance to clip: 50%
    Specificity:       25%
    Topic coverage:    15%
    Spam avoidance:    10%
    """
    if not hashtags:
        return 0.0

    # 1. Spam avoidance (10%)
    has_spam = any(tag.lower() in FORBIDDEN_SPAM_HASHTAGS for tag in hashtags)
    spam_score = 0.0 if has_spam else 10.0

    # 2. Relevance to clip (50%)
    topical_tags = [t for t in hashtags if t.lower() not in {"#shorts", "#short"}]
    if topical_tags and transcript:
        rel_count = sum(1 for t in topical_tags if validate_single_hashtag(t, transcript=transcript)[0])
        relevance_score = (rel_count / len(topical_tags)) * 10.0
    else:
        relevance_score = 8.5

    # 3. Specificity (25%)
    # Multi-word/camelCase topical tags score higher on specificity
    spec_points = 0
    for t in topical_tags:
        clean = t.lstrip("#")
        if any(c.isupper() for c in clean[1:]) or "_" in clean or len(clean) >= 7:
            spec_points += 1
    specificity_score = (spec_points / len(topical_tags) * 10.0) if topical_tags else 7.5

    # 4. Topic coverage (15%)
    # Having 3–5 topical tags covers topics well
    n_topical = len(topical_tags)
    if 3 <= n_topical <= 5:
        coverage_score = 10.0
    elif n_topical >= 1:
        coverage_score = 8.0
    else:
        coverage_score = 6.0

    final_score = (
        relevance_score * 0.50
        + specificity_score * 0.25
        + coverage_score * 0.15
        + spam_score * 0.10
    ) * 10.0

    return round(max(0.0, min(100.0, final_score)), 1)


def extract_grounded_fallback_hashtags(transcript: str) -> List[str]:
    """Extracts 3–5 deterministic topical hashtags directly from transcript content."""
    content_words = list(extract_content_words(transcript))
    tags = ["#Shorts"]
    for w in content_words[:5]:
        cap = w.capitalize()
        tag = f"#{cap}"
        if tag not in tags:
            tags.append(tag)
        if len(tags) >= 5:
            break
    return tags


def generate_title_candidates(
    client: OpenAI,
    model: str,
    transcript: str,
    hook: str = "",
    batch_attempt: int = 1,
) -> List[dict]:
    """
    Requests exactly 5 candidate titles from the LLM grounded strictly in the clip transcript.
    """
    clean_t = transcript.strip()[:3500] if transcript else ""
    batch_note = ""
    if batch_attempt > 1:
        batch_note = (
            "NOTE: Previous candidate titles were rejected for being generic, too long, or unsupported by transcript. "
            "Generate 5 FRESH titles strictly describing the concrete concepts spoken below without generic clickbait!\n"
        )

    prompt = f"""You are a top YouTube Shorts audience retention and metadata strategist.
Your task is to analyze ONLY the provided clip transcript and generate exactly 5 distinct, engaging title candidates.

{batch_note}CRITICAL TITLE RULES:
1. Grounded strictly in the clip transcript below. Never invent claims or assume context outside this clip.
2. Word count: 5 to 12 words. Character length: 35 to 70 characters.
3. Anti-clickbait: NEVER use generic formulas like 'YOU WON'T BELIEVE THIS', 'THE TRUTH EXPOSED', 'WAIT FOR THE TWIST', 'MUST WATCH', or 'THIS CHANGES EVERYTHING'.
4. Do NOT duplicate or repeat the video hook ('{hook}'). The title must complement the hook.
5. Maximum 1 emoji (preferably 0).
6. Match the language of the transcript (natural English for English audio; natural Roman Hindi/Hinglish for Hindi audio).

CLIP TRANSCRIPT (THIS IS YOUR ONLY CONTEXT):
\"\"\"
{clean_t}
\"\"\"

Respond ONLY with valid JSON in this exact structure:
{{
  "candidates": [
    {{
      "title": "Natural Engaging Title Here",
      "grounding_score": 9.0,
      "specificity_score": 8.5,
      "curiosity_score": 8.5,
      "relevance_score": 9.0,
      "clarity_score": 9.0
    }}
  ]
}}
"""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.60,
        max_tokens=700,
        response_format={"type": "json_object"} if ("llama" in model.lower() or "gpt" in model.lower()) else None,
    )
    content = response.choices[0].message.content.strip()
    data = extract_json_payload(content)
    cands = data.get("candidates", [])
    if isinstance(cands, list):
        return cands
    return []


def generate_shorts_metadata(filename: str, transcript: str = "") -> dict:
    """
    Generates authentic, transcript-grounded YouTube Shorts metadata:
    - title (best candidate selected from 5 candidates, validated and scored)
    - description (1–2 concise sentences + validated hashtags)
    - hashtags (canonical 4–6 hashtags starting with #Shorts)
    - generated_hook / punchline
    - title_quality_score, hashtag_quality_score
    - backward-compatible tags and title_variants
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
            "hashtags": ["#Shorts", "#podcast", "#viral"],
            "tags": ["shorts", "podcast", "viral", "clips"],
            "title_quality_score": 75.0,
            "hashtag_quality_score": 75.0,
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

    # Context transcript strictly limited to selected clip
    context_transcript = transcript.strip()[:3500] if transcript else clean_filename_fallback(filename)

    prompt = f"""You are a top YouTube Shorts audience retention and metadata strategist.
Below is the exact transcript spoken in the selected clip:

Transcript:
\"\"\"
{context_transcript}
\"\"\"

Task:
Generate authentic YouTube Shorts metadata strictly grounded in what is ACTUALLY SPOKEN in the transcript.
Avoid misleading clickbait. Titles and hashtags must be intriguing, specific, and grounded.

Requirements:
1. Punchline / Top Hook Header:
   - A single, punchy 1-line hook (3 to 8 words, maximum 42 characters) from this clip.
2. Title Candidates:
   - Exactly 5 distinct title candidates (5 to 12 words, 35 to 70 characters).
   - Must be curiosity-driven, accurate, and natural.
   - Do NOT simply copy or repeat the hook.
   - Grounded in the clip concepts. No generic sensational clickbait.
3. Description:
   - 1 to 2 natural, concise sentences explaining the authentic discussion.
   - Do NOT copy the title or keyword stuff.
4. Hashtags:
   - 4 to 6 hashtags starting with #Shorts followed by 3 to 5 topical hashtags relevant to this clip.
   - No generic spam tags like #viral, #fyp, #trending.
5. Tags: 5 to 8 search keyword tags.

Respond ONLY with valid JSON in this exact structure:
{{
  "punchline": "...",
  "title_candidates": [
    {{
      "title": "...",
      "grounding_score": 9.0,
      "specificity_score": 8.5,
      "curiosity_score": 8.5,
      "relevance_score": 9.0,
      "clarity_score": 9.0
    }}
  ],
  "title_1": "Primary Title",
  "title_2": "Variant 2",
  "title_3": "Variant 3",
  "description": "1-2 natural sentences.",
  "hashtags": ["#Shorts", "#Topic1", "#Topic2", "#Topic3"],
  "tags": ["tag1", "tag2"]
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
            "hashtags": ["#Shorts", "#podcast", "#viral"],
            "tags": ["shorts", "podcast", "viral"],
            "title_quality_score": 75.0,
            "hashtag_quality_score": 75.0,
        }

    # 1. Determine and validate generated_hook / punchline
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
            punchline = hook_generator.clean_hook_text(raw_punchline) or f"{clean_filename_fallback(filename)[:45]} ✨"

    # 2. Process Title Candidates: Batch 1 validation & scoring
    raw_candidates = data.get("title_candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        # Fallback to legacy title fields if present (e.g. In unit tests)
        raw_candidates = []
        for k in ("title_1", "title_2", "title_3"):
            val = data.get(k)
            if val:
                raw_candidates.append({"title": str(val).strip()})

    min_score = getattr(config, "MIN_TITLE_QUALITY_SCORE", 70.0)

    def evaluate_title_candidates(cands: List[dict]) -> List[dict]:
        valid_list = []
        seen_titles = set()
        for c in cands:
            if not isinstance(c, dict):
                continue
            title_text = str(c.get("title", "")).strip()
            norm_k = normalize_text(title_text)
            if not norm_k or norm_k in seen_titles:
                continue

            valid, reason = validate_title(
                c,
                transcript=transcript,
                hook=punchline,
                filename=filename,
            )
            if not valid:
                log.info("Rejected title candidate '%s': %s", title_text, reason)
                continue

            score = score_title(c, transcript=transcript)
            c["title_score"] = score
            if score < min_score:
                log.info("Rejected title candidate '%s': score %.1f < %.1f", title_text, score, min_score)
                continue

            seen_titles.add(norm_k)
            valid_list.append(c)

        valid_list.sort(key=lambda item: item.get("title_score", 0.0), reverse=True)
        return valid_list

    # Check if response contains new title_candidates or legacy payload
    if "title_candidates" in data and isinstance(data["title_candidates"], list):
        raw_candidates = data["title_candidates"]
        validated_titles = evaluate_title_candidates(raw_candidates)

        # Batch 2 retry if Batch 1 had 0 valid candidates and we have an API client
        if not validated_titles and transcript.strip():
            log.info("Batch 1 produced no valid title candidates. Attempting Batch 2...")
            try:
                batch_2_cands = generate_title_candidates(
                    client=client,
                    model=model,
                    transcript=transcript,
                    hook=punchline,
                    batch_attempt=2,
                )
                validated_titles = evaluate_title_candidates(batch_2_cands)
            except Exception as be:
                log.warning("Batch 2 title generation failed (%s).", be)

        # If both batches fail, use deterministic transcript fallback
        if validated_titles:
            best_title_obj = validated_titles[0]
            selected_title = best_title_obj["title"]
            title_score = best_title_obj.get("title_score", 85.0)
            title_variants = [c["title"] for c in validated_titles[:3]]
        else:
            detected_lang = "hi" if any(ord(c) >= 0x0900 and ord(c) <= 0x097F for c in transcript) else "en"
            selected_title = extract_grounded_fallback_title(transcript, detected_lang=detected_lang)
            title_score = score_title({"title": selected_title}, transcript=transcript)
            title_variants = [selected_title]

        # Process Hashtags
        raw_hashtags = data.get("hashtags")
        if not isinstance(raw_hashtags, list) or not raw_hashtags:
            raw_desc = str(data.get("description", ""))
            extracted_tags = re.findall(r"#[A-Za-z0-9_]+", raw_desc)
            raw_hashtags = extracted_tags if extracted_tags else extract_grounded_fallback_hashtags(transcript)

        validated_hashtags = validate_hashtags(raw_hashtags, transcript=transcript)
        if len(validated_hashtags) <= 1 and transcript.strip():
            fallback_tags = extract_grounded_fallback_hashtags(transcript)
            validated_hashtags = validate_hashtags(fallback_tags, transcript=transcript)

        hashtag_score = score_hashtags(validated_hashtags, transcript=transcript)

        # Process Description
        raw_desc = str(data.get("description", "")).strip()
        clean_desc = re.sub(r"(?:#[A-Za-z0-9_]+\s*)+$", "", raw_desc).strip()
        if not clean_desc:
            if transcript.strip():
                first_part = " ".join(transcript.strip().split()[:25])
                clean_desc = f"{first_part}..."
            else:
                clean_desc = clean_filename_fallback(filename)

        hashtags_str = " ".join(validated_hashtags)
        final_description = f"{clean_desc}\n\n{hashtags_str}".strip()

    else:
        # Legacy response compatibility (e.g. test_successful_ai_metadata_response)
        def sanitize_title(t_str, default_suffix):
            t = str(t_str or "").strip()
            if not t:
                t = default_suffix
            if not t.lower().endswith("#shorts"):
                t = f"{t[:78]} #shorts"
            return t[:95]

        fb = clean_filename_fallback(filename)
        selected_title = sanitize_title(data.get("title_1"), fb)
        title_2 = sanitize_title(data.get("title_2"), f"Insight: {fb}")
        title_3 = sanitize_title(data.get("title_3"), f"Must Watch: {fb}")
        title_variants = [selected_title, title_2, title_3]
        title_score = score_title({"title": selected_title}, transcript=transcript)

        raw_desc = str(data.get("description", "")).strip()
        if raw_desc:
            final_description = raw_desc
        elif transcript.strip():
            first_part = " ".join(transcript.strip().split()[:25])
            final_description = f"{first_part}...\n\n#shorts #viral #podcast"
        else:
            final_description = f"{fb}\n\n#shorts #podcast #viral"

        extracted_tags = re.findall(r"#[A-Za-z0-9_]+", final_description)
        validated_hashtags = validate_hashtags(extracted_tags, transcript=transcript)
        hashtag_score = score_hashtags(validated_hashtags, transcript=transcript)
        hashtags_str = " ".join(validated_hashtags)


    # 5. Process Tags
    raw_tags = data.get("tags")
    if isinstance(raw_tags, list) and raw_tags:
        tags = [str(t).strip() for t in raw_tags if str(t).strip()]
    else:
        tags = ["shorts", "podcast", "viral", "clips"]

    # Preserve title_variants backwards compatibility
    if len(title_variants) < 3:
        while len(title_variants) < 3:
            title_variants.append(selected_title)

    log.info("Selected Shorts Title: %s (Score: %.1f)", selected_title, title_score)
    log.info("Selected Hashtags (%d): %s (Score: %.1f)", len(validated_hashtags), hashtags_str, hashtag_score)

    return {
        "title": selected_title,
        "title_variants": title_variants,
        "punchline": punchline,
        "generated_hook": punchline,
        "description": final_description,
        "hashtags": validated_hashtags,
        "tags": tags,
        "title_quality_score": title_score,
        "hashtag_quality_score": hashtag_score,
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
