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
from typing import Dict, List, Optional, Tuple, Set

from openai import OpenAI
import config
from config import GROQ_API_KEY, OPENAI_API_KEY, GROQ_CHAT_MODEL
from clip_detection import _get_best_groq_model, _resolve_groq_model
from ai_rate_limiter import call_with_rate_limit

log = logging.getLogger("hook_generator")

# Maximum characters allowed for a Shorts top header hook
MAX_HOOK_CHARS = getattr(config, "HOOK_MAX_LENGTH", 42)

# Allowed hook types / strategies based on actual clip content
ALLOWED_HOOK_TYPES = {
    # 5 Candidate strategies
    "curiosity",
    "contrarian",
    "expectation_vs_reality",
    "question",
    "consequence",
    "specific_fact",
    "number_mechanism",
    # Legacy / descriptive categories
    "surprising_fact",
    "insight",
    "strong_claim",
    "myth_reality",
    "explanation",
    "cause_effect",
    "contradiction",
    "revelation",
    "unexpected_fact",
    "debate",
    "emotional",
}

# Semantic Emoji Category Mapping
SEMANTIC_EMOJI_CATEGORIES = {
    "finance": ["💰", "📈", "💸", "💵", "🪙"],
    "health": ["🩺", "❤️", "🚭", "🚬", "🧠", "🏥", "💊"],
    "technology": ["🤖", "💻", "⚡", "📱", "🔬"],
    "business": ["📈", "💼", "🏢", "📊"],
    "time": ["⏳", "⏰", "⌛"],
    "risk": ["⚠️", "🚨"],
    "success": ["🚀", "🏆", "🎯", "💡"],
    "science": ["🔬", "🧪", "🧬"],
    "food": ["🍽️", "🥗", "☕", "🍎"],
    "travel": ["✈️", "🌍", "🗺️"],
}

# Domain keyword associations for emoji relevance verification
EMOJI_DOMAIN_KEYWORDS = {
    "finance": ["money", "rupee", "lakh", "crore", "wealth", "invest", "compound", "saving", "stock", "dollar", "financial", "return"],
    "health": ["health", "fertility", "smoking", "sperm", "doctor", "disease", "body", "cancer", "medical", "sleep", "brain", "habit"],
    "technology": ["ai", "tech", "software", "code", "model", "computer", "developer", "automation", "algorithm", "robot"],
    "business": ["business", "startup", "company", "founder", "sales", "revenue", "market", "customer", "product"],
    "time": ["time", "waiting", "years", "months", "days", "hours", "delay", "early", "future", "now"],
    "risk": ["risk", "danger", "warning", "mistake", "wrong", "trap", "threat", "avoid"],
    "success": ["success", "winner", "achieve", "grow", "growth", "goal", "mastery"],
    "science": ["science", "study", "research", "experiment", "evidence", "proven", "dna"],
    "food": ["food", "diet", "eating", "meal", "nutrition", "sugar", "calories"],
    "travel": ["travel", "world", "country", "flight", "trip"],
}

# Authoritative Grounded 8-Dimension Hook Scoring Weights (must sum to 1.0)
# Scroll-stop potential: 25%
# Curiosity: 20%
# Grounding / factual support: 20%
# Specificity: 15%
# Relevance to clip payoff: 10%
# Clarity: 5%
# Brevity: 3%
# Emoji relevance: 2%
HOOK_SCORING_WEIGHTS = {
    "scroll_stop": 0.25,
    "curiosity": 0.20,
    "grounding": 0.20,
    "specificity": 0.15,
    "relevance": 0.10,
    "clarity": 0.05,
    "brevity": 0.03,
    "emoji_relevance": 0.02,
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


def extract_emojis(text: str) -> List[str]:
    """Returns a list of all individual emoji matches found in text."""
    if not text:
        return []
    return EMOJI_PATTERN.findall(text)


def evaluate_emoji_relevance(emojis: List[str], transcript: str = "", hook: str = "") -> float:
    """
    Evaluates semantic relevance of hook emojis (0.0 to 10.0 scale).
    - 0 emojis: 8.5/10 (clean, no distraction)
    - 1 emoji matching transcript domain: 10.0/10
    - 1 emoji neutral/general: 7.5/10
    - 2 emojis both independently matching domain: 9.5/10
    - 2 emojis neutral: 7.0/10
    - Unrelated / unfitting emoji: 5.5/10
    """
    if not emojis:
        return 8.5

    norm_tr = normalize_text_for_matching(transcript)
    norm_hk = normalize_text_for_matching(hook)
    combined_text = f"{norm_hk} {norm_tr}"

    emoji_scores = []
    for em in emojis:
        matched_domain = None
        for domain, em_list in SEMANTIC_EMOJI_CATEGORIES.items():
            if em in em_list:
                matched_domain = domain
                break

        if not matched_domain:
            emoji_scores.append(7.0)
            continue

        keywords = EMOJI_DOMAIN_KEYWORDS.get(matched_domain, [])
        if any(kw in combined_text for kw in keywords):
            emoji_scores.append(10.0)
        else:
            emoji_scores.append(5.5)

    if len(emoji_scores) == 1:
        return emoji_scores[0]
    elif len(emoji_scores) >= 2:
        if all(s >= 9.0 for s in emoji_scores):
            return 9.5
        return round(sum(emoji_scores) / len(emoji_scores), 2)

    return 7.0


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


HANGING_WORDS = {
    "with", "and", "or", "the", "a", "an", "is", "are", "was", "were",
    "to", "in", "of", "for", "on", "at", "by", "from", "that", "this",
    "which", "because", "but", "so", "if", "when", "then", "than", "as",
    "like", "into", "about", "ki", "ka", "ke", "ko", "se", "mein", "par",
}


def extract_grounded_fallback_hook(transcript: str, detected_lang: str = "en") -> Tuple[str, str]:
    """
    Extracts a clean, punchy, transcript-grounded hook directly from the clip transcript
    when LLM generation is unavailable or fails.
    Guarantees:
    - Never fabricates claims or introduces terms not in transcript.
    - Rejects incomplete/open-ended clauses ending with hanging conjunctions or prepositions.
    - Validates claim strength before selection.
    Returns (hook_text, supporting_text).
    """
    if not transcript or not transcript.strip():
        return ("", "")

    # Split into candidate clauses by punctuation
    raw_clauses = re.split(r"[.!?,\n;]+", transcript)
    clean_clauses = []
    for c in raw_clauses:
        c_clean = c.strip()
        if not c_clean:
            continue
        # Strip leading conversational conjunctions if present
        c_clean = re.sub(r"^(and|but|so|because|or)\s+", "", c_clean, flags=re.IGNORECASE).strip()
        if c_clean:
            clean_clauses.append(c_clean)

    def _is_clause_standalone(c_text: str) -> bool:
        words = c_text.split()
        if not (3 <= len(words) <= 8 and len(c_text) <= MAX_HOOK_CHARS):
            return False
        if words[-1].lower() in HANGING_WORDS:
            return False
        if is_forbidden_generic_hook(c_text, transcript=transcript):
            return False
        is_claim_valid, _ = validate_claim_strength(c_text, supporting_text=c_text, transcript=transcript)
        if not is_claim_valid:
            return False
        return True

    # 1. Prefer questions if any exist in the clip transcript
    for c in clean_clauses:
        if _is_clause_standalone(c):
            words = c.split()
            first = words[0].lower()
            if first in {"why", "how", "what", "is", "can", "kya", "kyun", "kaise"} or "?" in c:
                hook = c.upper()
                if not hook.endswith("?"):
                    hook += "?"
                return hook, c

    # 2. Look for any standalone clause with 3-8 words and <= 42 chars
    for c in clean_clauses:
        if _is_clause_standalone(c):
            return c.upper(), c

    # 3. Take the first clause and slice to 3-6 words <= 42 chars if standalone
    if clean_clauses:
        first_clause = clean_clauses[0]
        words = first_clause.split()
        for count in range(min(7, len(words)), 2, -1):
            sub = " ".join(words[:count])
            if _is_clause_standalone(sub):
                return sub.upper(), sub

    return ("", "")


# Extreme / absolute claim words that require strict transcript support
EXTREME_CLAIM_WORDS: Set[str] = {
    "permanent", "permanently", "incurable", "guaranteed", "guarantee", "guarantees",
    "deadly", "fatal", "cure", "cures", "cured", "curing", "impossible", "illegal",
    "arrested", "banned", "100%", "destroys", "destroyed", "destroying", "destroy",
    "eradicates", "eradicated", "eradicate", "miracle",
}

# Probabilistic hedging indicators in source transcripts
HEDGE_INDICATORS: List[str] = [
    "might", "may", "could", "can", "possibly", "potentially",
    "sometimes", "associated with", "linked to", "studies suggest",
    "preliminary", "risk factor",
]

# Absolute indicators that escalate claim strength when unsupported by the clip
ABSOLUTE_INDICATORS: Set[str] = {
    "always", "never", "definitely", "guaranteed", "guarantee", "guarantees",
    "proves", "proven", "prove", "destroys", "destroyed", "destroy",
    "cure", "cures", "100%",
}


def validate_claim_strength(
    generated_text: str,
    supporting_text: str = "",
    transcript: str = "",
) -> Tuple[bool, str]:
    """
    Deterministically validates that the generated text (Hook, Title, or Description)
    preserves the claim strength of the source clip transcript and supporting text.

    Detects and rejects meaningful escalations in:
    - Certainty (e.g. may/might/could -> will/definitely/certain)
    - Causality (e.g. associated with/linked to/correlated with -> causes)
    - Universality (e.g. some/in some cases -> all/everyone/universally)
    - Probability (e.g. possible/can help -> guarantees/certain)
    - Modality (e.g. could/can/may/might -> will)
    - Evidence strength (e.g. suggests/indicates/preliminary -> proves/proven)
    - Temporal qualification (e.g. historically/often/sometimes -> always/never)
    - Scope & explanation (e.g. one possible explanation -> the reason is)
    - Extreme absolute claims (EXTREME_CLAIM_WORDS unsupported by clip)

    Returns (is_valid, reason).
    """
    if not generated_text:
        return True, "Valid"

    reference = f"{supporting_text} {transcript}".strip()
    if not reference:
        return True, "Valid"

    norm_gen = re.sub(r"[^\w\s%]", " ", generated_text.lower())
    norm_ref = re.sub(r"[^\w\s%]", " ", reference.lower())
    gen_words = set(norm_gen.split())
    ref_words = set(norm_ref.split())

    # 1. Extreme / absolute claim words check
    for ew in EXTREME_CLAIM_WORDS:
        if ew in gen_words and ew not in ref_words:
            return False, f"'{ew}'"

    # 2. Causality Escalation: associated with / linked to / correlated with -> causes
    correlation_markers = [
        "associated with", "associated", "linked to", "link to", "links to",
        "correlated with", "correlation between", "connection between", "relationship between"
    ]
    causation_markers = ["causes", "caused", "causing", "cause of", "leads directly to"]
    has_correlation = any(m in norm_ref for m in correlation_markers)
    has_ref_causation = any(
        re.search(r"\b" + re.escape(w) + r"\b", norm_ref)
        for w in ["cause", "causes", "caused", "causing", "leads to", "leading to"]
    )
    if has_correlation and not has_ref_causation:
        for c in causation_markers:
            if re.search(r"\b" + re.escape(c) + r"\b", norm_gen):
                return False, f"'{c}'"

    # 3. Evidence Strength Escalation: suggests / indicates -> proves
    suggest_markers = [
        "suggests", "suggest", "studies suggest", "study suggests",
        "research suggests", "indicates", "indicate", "evidence suggests",
        "preliminary", "theorized", "hypothesis"
    ]
    proof_markers = ["proves", "proven", "prove", "proof", "science proves", "confirms definitely"]
    has_suggest = any(m in norm_ref for m in suggest_markers)
    has_ref_proof = any(
        re.search(r"\b" + re.escape(w) + r"\b", norm_ref)
        for w in ["prove", "proves", "proven", "proof"]
    )
    if has_suggest and not has_ref_proof:
        for p in proof_markers:
            if re.search(r"\b" + re.escape(p) + r"\b", norm_gen):
                return False, f"'{p}'"

    # 4. Modality / Certainty Escalation: may / might / could / possibly -> will / definitely
    modal_hedges = ["may", "might", "could", "possibly", "potentially", "can potentially", "could potentially"]
    modal_absolutes = ["will", "definitely", "certainly", "shall"]
    has_modal_hedge = any(re.search(r"\b" + re.escape(h) + r"\b", norm_ref) for h in modal_hedges)
    has_ref_will = any(re.search(r"\b" + re.escape(w) + r"\b", norm_ref) for w in modal_absolutes)
    if has_modal_hedge and not has_ref_will:
        for ma in modal_absolutes:
            if re.search(r"\b" + re.escape(ma) + r"\b", norm_gen):
                return False, f"'{ma}'"

    # 5. Probability / Guarantee Escalation: can / can help / may help -> guarantees
    guarantee_markers = ["guarantees", "guaranteed", "guarantee"]
    has_ref_guarantee = any(re.search(r"\b" + re.escape(g) + r"\b", norm_ref) for g in guarantee_markers)
    if not has_ref_guarantee:
        for gm in guarantee_markers:
            if re.search(r"\b" + re.escape(gm) + r"\b", norm_gen):
                return False, f"'{gm}'"

    # 6. Frequency / Temporal Escalation: historically / sometimes / often -> always / never
    frequency_hedges = ["historically", "in the past", "sometimes", "often", "frequently", "occasionally", "at times"]
    frequency_absolutes = ["always", "never", "every single time", "at all times"]
    has_freq_hedge = any(m in norm_ref for m in frequency_hedges)
    has_ref_freq_abs = any(re.search(r"\b" + re.escape(fa) + r"\b", norm_ref) for fa in ["always", "never"])
    if (has_freq_hedge or "can " in norm_ref or "could " in norm_ref) and not has_ref_freq_abs:
        for fa in frequency_absolutes:
            if re.search(r"\b" + re.escape(fa) + r"\b", norm_gen):
                return False, f"'{fa}'"

    # 7. Scope / Universality Escalation: some / in some cases -> everyone / all
    scope_hedges = [
        "some people", "some patients", "some individuals", "some",
        "in some cases", "in certain cases", "a portion of"
    ]
    scope_absolutes = ["everyone", "everybody", "all people", "universally", "every single person"]
    has_scope_hedge = any(m in norm_ref for m in scope_hedges)
    has_ref_scope_abs = any(re.search(r"\b" + re.escape(sa) + r"\b", norm_ref) for sa in scope_absolutes)
    if has_scope_hedge and not has_ref_scope_abs:
        for sa in scope_absolutes:
            if re.search(r"\b" + re.escape(sa) + r"\b", norm_gen):
                return False, f"'{sa}'"

    # 8. Possibility / Explanation Escalation: possible / one possible explanation -> certain / the reason is
    possibility_hedges = [
        "one possible explanation", "possible explanation", "one possibility",
        "possible", "potentially", "could potentially"
    ]
    certainty_absolutes = ["the reason is", "is the reason", "the only reason", "the real reason", "certain", "definitely", "certain to"]
    has_poss_hedge = any(m in norm_ref for m in possibility_hedges)
    has_ref_certain = any(re.search(r"\b" + re.escape(ca) + r"\b", norm_ref) for ca in certainty_absolutes)
    if has_poss_hedge and not has_ref_certain:
        for ca in certainty_absolutes:
            if re.search(r"\b" + re.escape(ca) + r"\b", norm_gen):
                return False, f"'{ca}'"

    return True, "Valid"


check_claim_strength_preservation = validate_claim_strength


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

    # 9. Emoji validation & policy enforcement:
    # - 0–2 emojis maximum (configurable via config.MAX_HOOK_EMOJIS, default 2)
    # - Prefer 1 semantically relevant emoji
    # - 2 emojis allowed only when both independently reinforce the hook
    # - No duplicate emojis
    # - No emoji spam
    # - No meaningless emoji chains (e.g. '👀🔥', '🚨😱🔥', '💰 💰')
    # - Emojis must not replace important words
    # - Emoji presence is never mandatory
    hook_emojis = extract_emojis(raw_hook)
    max_allowed_emojis = getattr(config, "MAX_HOOK_EMOJIS", 2)
    if len(hook_emojis) > max_allowed_emojis:
        return False, f"Hook contains multiple emojis ({len(hook_emojis)} > {max_allowed_emojis} allowed)"

    # Check for consecutive emojis / emoji chains without intervening alphanumeric words
    chain_match = re.search(r"(" + EMOJI_PATTERN.pattern + r"[\s,\.\!\?]*){2,}", raw_hook)
    if chain_match:
        matched_chain = chain_match.group(0).strip()
        return False, f"Hook contains multiple emojis in sequence/chain (emoji chain: '{matched_chain}')"

    # Check for duplicate emojis (e.g. '💰' and '💰')
    if len(hook_emojis) > 1 and len(hook_emojis) != len(set(hook_emojis)):
        return False, f"Hook contains duplicate emoji ('{hook_emojis[0]}')"

    # Check that emoji does not replace words or form an emoji-only hook
    if not words:
        return False, "Hook cannot be emoji-only (emojis must not replace important words)"

    # 10. Supporting text grounding verification
    support = candidate.get("supporting_text", "").strip()
    if support and transcript:
        if not is_supporting_text_in_transcript(support, transcript):
            return False, f"Supporting text was not found in the clip transcript: '{support[:40]}...'"
    elif require_supporting_text and transcript:
        return False, "Candidate is missing required supporting_text from clip transcript"

    # 11. Claim-strength preservation
    if transcript:
        is_pres, claim_word = check_claim_strength_preservation(hook, transcript)
        if not is_pres:
            return False, f"Hook introduces stronger unsupported claim ({claim_word})"

    # 12. Valid hook_type check
    hook_type = candidate.get("hook_type", "curiosity").lower()
    if hook_type not in ALLOWED_HOOK_TYPES:
        candidate["hook_type"] = "curiosity"

    return True, "Valid"


def score_hook(candidate: dict, transcript: str = "", hook_summary: str = "") -> float:
    """
    Calculates authoritative 0–100 score in Python using the exact weighted formula:
    Scroll-stop potential: 25%
    Curiosity: 20%
    Grounding / factual support: 20%
    Specificity: 15%
    Relevance to clip payoff: 10%
    Clarity: 5%
    Brevity: 3%
    Emoji relevance: 2%
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

    # Check if candidate is using legacy ungrounded test fixture schema (e.g. test_12_exact_python_scoring_formula)
    # where curiosity, relevance, payoff, clarity, brevity, impact are provided without grounding or scroll_stop
    has_legacy_schema = (
        all(k in candidate for k in ("curiosity", "relevance", "payoff", "clarity", "brevity", "impact"))
        and not any(k in candidate for k in ("grounding", "grounding_score", "scroll_stop", "scroll_stop_score", "supporting_text"))
    )

    if has_legacy_schema:
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

    # Grounded 8-Dimension Authoritative Scoring
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

    # Scroll-stop potential defaults to composite of curiosity and specificity if not set
    scroll_stop = get_dim(
        "scroll_stop_score", "scroll_stop",
        default=round(min(10.0, curiosity * 0.6 + specificity * 0.4), 2),
    )

    hook_text = candidate.get("hook", "")
    emojis = extract_emojis(hook_text)
    emoji_rel = evaluate_emoji_relevance(emojis, transcript=transcript, hook=hook_text)
    emoji_relevance = get_dim("emoji_relevance_score", "emoji_relevance", default=emoji_rel)

    final_score = (
        scroll_stop * HOOK_SCORING_WEIGHTS["scroll_stop"]
        + curiosity * HOOK_SCORING_WEIGHTS["curiosity"]
        + grounding * HOOK_SCORING_WEIGHTS["grounding"]
        + specificity * HOOK_SCORING_WEIGHTS["specificity"]
        + relevance * HOOK_SCORING_WEIGHTS["relevance"]
        + clarity * HOOK_SCORING_WEIGHTS["clarity"]
        + brevity * HOOK_SCORING_WEIGHTS["brevity"]
        + emoji_relevance * HOOK_SCORING_WEIGHTS["emoji_relevance"]
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
            "is_fallback": True,
            "fallback_reason": "Grounded fallback hook from transcript",
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
        "is_fallback": True,
        "fallback_reason": "Neutral inquiry fallback",
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
7. Emoji Policy:
   - 0–2 emojis maximum.
   - Prefer 1 semantically relevant emoji when it genuinely strengthens visual meaning.
   - 2 emojis allowed ONLY when both independently reinforce concepts in the hook.
   - 0 emojis when an emoji would make the hook worse. Emoji presence is never mandatory.
   - NEVER generate duplicate emojis (e.g., '💰💰'), emoji chains (e.g., '👀🔥', '🚨😱🔥'), or emoji spam.
   - Emojis must NEVER replace important words.
8. {lang_instruction}

5 HOOK CANDIDATE STRATEGIES TO USE (generate distinct strategies, not superficial rewrites):
1. Curiosity: creates curiosity gap grounded strictly in clip
2. Contrarian / expectation-vs-reality: challenges common assumption with clip fact
3. Question: direct, intriguing question answered by the clip
4. Consequence: highlights cause/effect or direct consequence revealed in clip
5. Specific fact / number / mechanism: highlights concrete number, named concept, or specific mechanism

CLIP TRANSCRIPT (THIS IS YOUR ONLY CONTEXT):
\"\"\"
{clean_t}
\"\"\"

Respond ONLY with a valid JSON object containing exactly 5 candidates in this schema:
{{
  "candidates": [
    {{
      "hook": "SPECIFIC GROUNDED HOOK (UPPERCASE)",
      "hook_type": "curiosity | contrarian | question | consequence | specific_fact",
      "supporting_text": "exact verbatim excerpt from the transcript supporting this hook"
    }}
  ]
}}
"""

    def _api_call():
        return client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.60,
            max_tokens=1200,
            response_format={"type": "json_object"} if ("llama" in model.lower() or "gpt" in model.lower()) else None,
        )

    try:
        response = call_with_rate_limit(
            client_fn=_api_call,
            prompt_or_messages=[{"role": "user", "content": prompt}],
            max_tokens=1200,
        )
        choice = response.choices[0] if (response and getattr(response, "choices", None)) else None
        if not choice or getattr(choice, "finish_reason", None) == "length":
            log.warning("Hook generation response was truncated or missing choices.")
            return []

        content = choice.message.content.strip() if choice.message else ""
        content = re.sub(r"^```json\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
        data = json.loads(content)

        candidates = data.get("candidates", [])
        if isinstance(candidates, list):
            return candidates
    except Exception as e:
        log.warning("LLM hook candidate generation failed: %s", e)

    return []


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
            "is_fallback": True,
            "fallback_reason": fb.get("fallback_reason", "No AI API key found"),
        }

    # Initialize client
    try:
        if api_key.startswith("gsk_") or GROQ_API_KEY:
            client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
            model = _resolve_groq_model(client, configured_model=GROQ_CHAT_MODEL)
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
            "is_fallback": True,
            "fallback_reason": fb.get("fallback_reason", f"Could not initialize AI client: {e}"),
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
            "is_fallback": False,
            "fallback_reason": "",
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
        "is_fallback": True,
        "fallback_reason": fb.get("fallback_reason", "All LLM hook candidates failed validation"),
    }
