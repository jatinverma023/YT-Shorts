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
    GROQ_API_KEY, OPENAI_API_KEY, GROQ_CHAT_MODEL, DEFAULT_GROQ_CHAT_MODEL,
    MIN_CLIP_SECONDS, MAX_CLIP_SECONDS, MAX_CLIPS_PER_VIDEO,
    MAX_DISCOVERY_CANDIDATES,
    MIN_CLIP_QUALITY_SCORE, CLIP_OVERLAP_THRESHOLD,
    MIN_STANDALONE_SCORE,
    DISCOVERY_CHUNK_MAX_CHARS, DISCOVERY_CHUNK_OVERLAP_SECONDS,
)
from ai_rate_limiter import call_with_rate_limit, shared_rate_limiter

log = logging.getLogger("clip_detection")


class ClipDiscoveryError(RuntimeError):
    """Raised when AI multi-clip discovery fails at the API/LLM or infrastructure level."""
    pass


class ModelConfigurationError(ValueError):
    """Raised when an explicit model configuration is invalid or unavailable from the provider."""
    pass


# 8-dimension weights summing exactly to 1.00 (100%)
SCORING_WEIGHTS = {
    "hook_strength": 0.20,
    "standalone_clarity": 0.15,
    "payoff": 0.20,
    "curiosity": 0.10,
    "impact": 0.10,
    "retention": 0.10,
    "context_independence": 0.10,
    "punchline": 0.05,
}

# Suspicious opening phrases that often signal missing preceding context
SUSPICIOUS_OPENING_PATTERNS = [
    "and that's why", "and then", "so that's why", "as i said", "like i said",
    "as mentioned", "as we discussed", "as you know", "you already know",
    "the first one", "the second one", "this one", "that one", "the other one",
    "what happened next", "that's exactly", "that's why", "he did", "she did",
    "they did", "it was", "this is why", "yes absolutely", "yes, absolutely",
    "no because", "no, because",
]


def check_opening_risk(text: str) -> tuple:
    """
    Lightweight deterministic check for suspicious opening phrases that often
    indicate missing preceding context.
    IMPORTANT: This is ONLY an advisory risk signal, NOT an automatic rejection rule.
    """
    if not text:
        return False, ""
    clean = re.sub(r"[^\w\s']", " ", text.lower()).strip()
    clean = re.sub(r"\s+", " ", clean)

    for pattern in SUSPICIOUS_OPENING_PATTERNS:
        norm_pat = re.sub(r"[^\w\s']", " ", pattern.lower()).strip()
        norm_pat = re.sub(r"\s+", " ", norm_pat)
        if clean.startswith(norm_pat):
            return True, f"Opening phrase '{pattern}' often signals missing preceding context"
    return False, ""


def validate_standalone_context(
    candidate: dict,
    min_standalone_score: float = MIN_STANDALONE_SCORE,
) -> tuple:
    """
    Validates whether a candidate clip is understandable independently.
    Rejects clips with:
      - standalone_score < min_standalone_score (default 7.0)
      - missing_setup == True (setup exists outside candidate)
      - missing_payoff == True (payoff/climax cuts off after candidate)
      - critical_unresolved_reference == True (essential pronouns/references cannot be resolved)
      - risk opening combined with high context dependency or unresolved references
    Returns (is_valid: bool, reason: str).
    """
    if not isinstance(candidate, dict):
        return False, "Malformed candidate data"

    # 1. Parse standalone_score (defaults to scores.standalone_clarity if missing)
    raw_scores = candidate.get("quality_scores") or candidate.get("scores", {})
    if not isinstance(raw_scores, dict):
        raw_scores = {}

    try:
        if "standalone_score" in candidate and candidate["standalone_score"] is not None:
            standalone_score = float(candidate["standalone_score"])
        elif "standalone_clarity" in raw_scores:
            standalone_score = float(raw_scores["standalone_clarity"])
        else:
            standalone_score = 5.0
    except (ValueError, TypeError):
        standalone_score = 5.0

    standalone_score = max(0.0, min(10.0, standalone_score))

    # 2. Parse boolean flags
    missing_setup = bool(candidate.get("missing_setup", False))
    missing_payoff = bool(candidate.get("missing_payoff", False))
    critical_unresolved_ref = bool(candidate.get("critical_unresolved_reference", False))
    unresolved_refs = candidate.get("unresolved_references", [])
    if not isinstance(unresolved_refs, list):
        unresolved_refs = [str(unresolved_refs)] if unresolved_refs else []

    # 3. Deterministic opening risk check (used as supporting evidence only)
    opening_text = str(
        candidate.get("topic_summary")
        or candidate.get("opening_context")
        or candidate.get("hook_summary", "")
        or candidate.get("title_idea", "")
    )
    risk_detected, risk_reason = check_opening_risk(opening_text)

    # Check all rejection conditions and collect reasons
    rejection_reasons = []

    if missing_setup:
        rejection_reasons.append("Missing setup required from prior context")

    if missing_payoff:
        rejection_reasons.append("Cuts off before the essential resolution, punchline, or payoff")

    if critical_unresolved_ref:
        refs_str = f" ({', '.join(str(r) for r in unresolved_refs)})" if unresolved_refs else ""
        rejection_reasons.append(f"Critical unresolved reference(s){refs_str}")

    if standalone_score < min_standalone_score:
        rejection_reasons.append(f"Standalone score ({standalone_score:.1f}/10) is below threshold ({min_standalone_score:.1f})")

    # If an opening risk was detected AND there are unresolved references or high context dependency
    if "context_dependency" in raw_scores:
        try:
            context_dep = float(raw_scores["context_dependency"])
        except (ValueError, TypeError):
            context_dep = 0.0
    elif "context_independence" in raw_scores:
        try:
            context_dep = 10.0 - float(raw_scores["context_independence"])
        except (ValueError, TypeError):
            context_dep = 0.0
    else:
        context_dep = 0.0

    if risk_detected and (unresolved_refs or context_dep >= 7.0):
        refs_detail = f" with unresolved reference(s): {', '.join(str(r) for r in unresolved_refs)}" if unresolved_refs else ""
        rejection_reasons.append(f"Context-dependent opening{refs_detail} and high context dependency ({context_dep:.1f}/10)")

    if rejection_reasons:
        return False, "; ".join(rejection_reasons)

    return True, f"Candidate is self-contained (standalone score {standalone_score:.1f}/10)"




def calculate_quality_score(scores: dict) -> float:
    """
    Computes deterministic 0-100 composite quality score from 8 dimension scores (0-10):
      Hook Strength: 20%
      Standalone Clarity: 15%
      Payoff / Completion: 20%
      Curiosity / Intrigue: 10%
      Emotional / Intellectual Impact: 10%
      Retention Potential: 10%
      Context Independence: 10% (10 - context_dependency)
      Punchline / Memorable Moment: 5%
    Total = 100%.
    """
    if not isinstance(scores, dict):
        scores = {}

    def _get_dim(key: str, default: float = 5.0) -> float:
        try:
            val = float(scores.get(key, default))
            return max(0.0, min(10.0, val))
        except (ValueError, TypeError):
            return default

    hook = _get_dim("hook_strength")
    standalone = _get_dim("standalone_clarity")
    payoff = _get_dim("payoff")
    curiosity = _get_dim("curiosity")
    impact = _get_dim("impact")
    retention = _get_dim("retention")
    context_dep = _get_dim("context_dependency")
    punchline = _get_dim("punchline_score", default=_get_dim("punchline", default=5.0))

    # Convert context_dependency (negative factor) into context_independence
    context_indep = 10.0 - context_dep

    composite = (
        hook * SCORING_WEIGHTS["hook_strength"] +
        standalone * SCORING_WEIGHTS["standalone_clarity"] +
        payoff * SCORING_WEIGHTS["payoff"] +
        curiosity * SCORING_WEIGHTS["curiosity"] +
        impact * SCORING_WEIGHTS["impact"] +
        retention * SCORING_WEIGHTS["retention"] +
        context_indep * SCORING_WEIGHTS["context_independence"] +
        punchline * SCORING_WEIGHTS["punchline"]
    ) * 10.0

    return round(max(0.0, min(100.0, composite)), 1)


def _resolve_groq_model(client, configured_model: str = None) -> str:
    """
    Resolves and verifies the production Groq chat model.
    1. Uses explicit configured_model (or GROQ_CHAT_MODEL from config/environment).
    2. Falls back to documented DEFAULT_GROQ_CHAT_MODEL ('openai/gpt-oss-20b').
    3. Verifies that the model exists in the provider's active model list.
    4. If the model does not exist: fails fast with ModelConfigurationError.
       NEVER silently substitutes an arbitrary model.
    """
    model = (configured_model or GROQ_CHAT_MODEL or "").strip()
    if not model:
        model = DEFAULT_GROQ_CHAT_MODEL

    # Verify model availability against provider model-list API if client is available
    if client and hasattr(client, "models") and hasattr(client.models, "list"):
        try:
            available_models = [m.id for m in client.models.list().data]
            if available_models and model not in available_models:
                log.error(
                    "[MODEL_CONFIG_ERROR] Configured Groq model '%s' is not available. Available models: %s",
                    model, available_models,
                )
                raise ModelConfigurationError(
                    f"Configured Groq model '{model}' does not exist on provider. Available: {available_models}"
                )
        except ModelConfigurationError:
            raise
        except Exception as e:
            # Network issue or mock during testing without models.list response
            log.warning("Could not verify model '%s' with provider model list: %s", model, e)

    log.info("[MODEL_CONFIG] Using verified production Groq model: '%s'", model)
    return model


def _get_best_groq_model(client, configured_model: str = None) -> str:
    """Backward-compatible wrapper for _resolve_groq_model."""
    return _resolve_groq_model(client, configured_model=configured_model)


def _get_llm_client():
    api_key = GROQ_API_KEY or OPENAI_API_KEY
    if not api_key:
        return None, None
    if api_key.startswith("gsk_") or GROQ_API_KEY:
        client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        model = _resolve_groq_model(client, configured_model=GROQ_CHAT_MODEL)
        return client, model
    return OpenAI(api_key=api_key), "gpt-4o-mini"


def _format_segments_to_lines(segments: list) -> list:
    lines = []
    for s in segments:
        st = float(s.get("start", 0.0))
        et = float(s.get("end", 0.0))
        txt = str(s.get("text", "")).strip()
        if txt:
            lines.append((st, et, f"[{st:.1f}s - {et:.1f}s] {txt}"))
    return lines


def _split_lines_into_chunks(
    lines: list,
    max_chars: int = DISCOVERY_CHUNK_MAX_CHARS,
    overlap_seconds: float = DISCOVERY_CHUNK_OVERLAP_SECONDS,
) -> list:
    """
    Splits transcript timestamp lines into content-aware overlapping chunks.
    Preserves enough neighboring context across boundaries so complete talk segments
    and payoffs can be detected without context breaks.
    """
    if not lines:
        return []
    full_text = "\n".join(item[2] for item in lines)
    if len(full_text) <= max_chars:
        return [lines]

    chunks = []
    current_chunk = []
    current_len = 0
    i = 0
    while i < len(lines):
        st, et, line_str = lines[i]
        current_chunk.append(lines[i])
        current_len += len(line_str) + 1
        if current_len >= max_chars:
            chunks.append(current_chunk)
            chunk_end_time = et
            rollback_i = i
            while rollback_i > 0 and (chunk_end_time - lines[rollback_i][0]) < overlap_seconds:
                rollback_i -= 1
            if rollback_i == i:
                i += 1
            else:
                i = rollback_i + 1
            current_chunk = []
            current_len = 0
        else:
            i += 1
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def _detect_clips_llm(
    client,
    model: str,
    transcript_text: str,
    total_duration: float,
    min_clip_seconds: int,
    max_clip_seconds: int,
) -> list:
    """
    Prompts the LLM to identify meaningful, coherent talk/topic segments and return
    potential standalone Short candidates using a compact, rate-limit friendly schema.
    """
    prompt = f"""You are an elite YouTube Shorts editor and conversational analyst.
Analyze the timestamped transcript below from a video of total duration {total_duration:.1f} seconds.

DISCOVERY OBJECTIVE:
Identify at most 3 of the strongest coherent talk segments ({min_clip_seconds}s to {max_clip_seconds}s) with high standalone potential.
Each Short must be self-contained:
1. Setup: Clear context without needing prior context.
2. Core Insight: One clear argument, insight, explanation, or story.
3. Payoff: Satisfying conclusion, resolution, or takeaway.

AVOID: Greetings, sponsor reads, incomplete answers, setup without payoff, or clips requiring outside context.

Return at most 3 genuinely strong standalone moments (or an empty list if none meet standards).

Requirements:
1. Return AT MOST 3 candidate clips.
2. Each clip MUST be between {min_clip_seconds} and {max_clip_seconds} seconds long.
3. All start and end timestamps MUST be within 0.0 and {total_duration:.1f} seconds.
4. Standalone score (0-10): >=7 is self-contained; <7 requires outside context.

Transcript:
{transcript_text}

Respond ONLY with valid JSON in this exact structure:
{{
  "clips": [
    {{
      "start": 12.5,
      "end": 54.0,
      "topic": "Why most startups fail",
      "topic_summary": "Explains why morning screen time ruins focus and what to do instead.",
      "title_idea": "The Morning Habit Destroying Your Brain #shorts",
      "punchline": "Stop Ruining Your Mornings 🛑",
      "standalone": true,
      "standalone_score": 9.0,
      "missing_setup": false,
      "missing_payoff": false,
      "critical_unresolved_reference": false,
      "unresolved_references": [],
      "reason": "Clear standalone hook with setup and payoff."
    }}
  ]
}}
"""
    try:
        def _api_call():
            return client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=1500,
                response_format={"type": "json_object"} if ("llama" in model.lower() or "gpt" in model.lower()) else None,
            )

        response = call_with_rate_limit(
            client_fn=_api_call,
            prompt_or_messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
        )
        content = response.choices[0].message.content.strip()
        content = re.sub(r"^```json\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
        data = json.loads(content)
        clips = data.get("clips") or data.get("candidates", [])
        return clips[:3] if isinstance(clips, list) else []
    except Exception as e:
        log.warning("LLM detection call failed: %s", e)
        return None


def detect_clips_from_transcript(
    segments: list,
    total_duration: float,
    min_clip_seconds: int = MIN_CLIP_SECONDS,
    max_clip_seconds: int = MAX_CLIP_SECONDS,
    max_clips: int = None,
) -> list:
    """
    Analyzes full-length video transcripts using talk/topic segment detection.
    For long videos, splits the transcript into content-aware overlapping chunks,
    gathers potential candidates, and globally runs deterministic validation,
    standalone gating (>=7.0), quality scoring (>=70.0), and deduplication.
    max_clips=None yields all genuinely qualified clips dynamically without artificial limits.
    """
    total_duration = max(1.0, float(total_duration))

    # If video is already shorter than max_clip_seconds, return a single clip covering the whole video
    if total_duration <= max_clip_seconds:
        log.info("Video is %.1fs (<= %ds). Using single clip.", total_duration, max_clip_seconds)
        single_scores = {
            "hook_strength": 7.0,
            "standalone_clarity": 8.0,
            "payoff": 8.0,
            "curiosity": 7.0,
            "impact": 7.0,
            "retention": 7.0,
            "context_dependency": 2.0,
            "punchline_score": 7.0,
        }
        return [{
            "start_time": 0.0,
            "end_time": round(total_duration, 2),
            "duration": round(total_duration, 2),
            "clip_index": 1,
            "topic": "Full video",
            "topic_summary": "Full video clip",
            "hook_summary": "Full video clip",
            "title_idea": "Key Highlight #shorts",
            "punchline": "Watch Till The End 🔥",
            "scores": single_scores,
            "quality_scores": single_scores,
            "quality_score": calculate_quality_score(single_scores),
            "selection_reason": "Source video is already Short duration.",
            "standalone": True,
            "standalone_score": 8.0,
            "missing_setup": False,
            "missing_payoff": False,
            "critical_unresolved_reference": False,
            "unresolved_references": [],
            "standalone_reason": "Full video is already of Short duration.",
        }]

    client, model = _get_llm_client()
    if not client or not segments:
        log.warning("No LLM client or empty segments. AI discovery failed.")
        raise ClipDiscoveryError("AI clip discovery failed: No LLM client available or empty transcript segments.")

    lines = _format_segments_to_lines(segments)
    if not lines:
        log.warning("No formatted transcript lines. AI discovery failed.")
        raise ClipDiscoveryError("AI clip discovery failed: No formatted transcript lines available.")

    # Content-aware chunking for long-form video scale
    chunks = _split_lines_into_chunks(
        lines,
        max_chars=DISCOVERY_CHUNK_MAX_CHARS,
        overlap_seconds=DISCOVERY_CHUNK_OVERLAP_SECONDS,
    )
    all_raw_clips = []
    any_success = False

    for chunk in chunks:
        chunk_text = "\n".join(item[2] for item in chunk)
        raw_chunk_clips = _detect_clips_llm(
            client=client,
            model=model,
            transcript_text=chunk_text,
            total_duration=total_duration,
            min_clip_seconds=min_clip_seconds,
            max_clip_seconds=max_clip_seconds,
        )
        if raw_chunk_clips is not None:
            any_success = True
            all_raw_clips.extend(raw_chunk_clips)

    if not any_success:
        log.warning("AI clip detection failed across all transcript chunks. Returning zero valid clips.")
        raise ClipDiscoveryError("AI clip discovery failed across all transcript chunks (all LLM calls failed).")

    # Globally run quality filtering, standalone validation, and deduplication across all chunks
    validated = _validate_and_filter_clips(
        all_raw_clips,
        total_duration,
        min_clip_seconds,
        max_clip_seconds,
        max_clips=max_clips,
    )
    if validated:
        log.info("Successfully detected and scored %d high-retention clips.", len(validated))
        return validated
    elif all_raw_clips:
        log.info(
            "AI suggested %d candidate(s), but none met the quality/standalone threshold or survived deduplication. Returning empty list.",
            len(all_raw_clips),
        )
        return []

    return []


def _is_similar_summary(summary1: str, summary2: str, threshold: float = 0.60) -> bool:
    """
    Checks semantic word overlap between two hook summaries to prevent picking
    multiple clips that make the same core point.
    Uses Jaccard index on content words so that distinct topics sharing a single
    common word are not mistakenly discarded.
    """
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with",
        "about", "why", "how", "what", "is", "it", "this", "that", "of", "from",
        "explains", "details", "shows", "discusses", "talks", "reveals", "key", "moment",
        "topic", "candidate", "insight", "clip", "part", "video", "segment", "hook",
    }
    words1 = set(re.findall(r"\w+", summary1.lower())) - stop_words
    words2 = set(re.findall(r"\w+", summary2.lower())) - stop_words
    if not words1 or not words2:
        return False
    intersection = words1.intersection(words2)
    jaccard = len(intersection) / len(words1.union(words2))
    return jaccard >= threshold


def _validate_and_filter_clips(
    raw_clips: list,
    total_duration: float,
    min_clip_seconds: int,
    max_clip_seconds: int,
    max_clips: int = None,
    min_quality_score: float = MIN_CLIP_QUALITY_SCORE,
    overlap_threshold: float = CLIP_OVERLAP_THRESHOLD,
    min_standalone_score: float = MIN_STANDALONE_SCORE,
) -> list:
    """
    Validates candidate boundaries, enforces standalone/context validation gate,
    calculates deterministic Python quality scores, filters below quality threshold,
    deduplicates temporal & semantic overlaps, and returns top candidates.
    If max_clips is None, preserves all valid candidates up to MAX_DISCOVERY_CANDIDATES.
    Assigns stable clip_index in chronological order.
    """
    candidates = []

    for item in raw_clips:
        try:
            start = float(item.get("start") if item.get("start") is not None else item.get("start_time", 0.0))
            end = float(item.get("end") if item.get("end") is not None else item.get("end_time", 0.0))
            topic = str(item.get("topic") or "").strip()
            summary = str(
                item.get("topic_summary")
                or item.get("hook_summary")
                or item.get("summary")
                or topic
                or item.get("hook")
                or ""
            ).strip() or "Key moment from video"
            title = str(item.get("title_idea") or item.get("title") or "").strip() or (f"{topic} #shorts" if topic else "Must Watch Insight #shorts")
            punchline = str(item.get("punchline", "")).strip().strip('"').strip("'").replace("{", "").replace("}", "")[:60]
            if not punchline:
                clean_t = re.sub(r"#shorts", "", title, flags=re.IGNORECASE).strip()
                punchline = f"{clean_t[:45]} ✨" if clean_t else "Watch Till The End 🔥"
            selection_reason = str(item.get("selection_reason") or item.get("reason") or "").strip()
            raw_scores = item.get("quality_scores") or item.get("scores") or {}
            if not isinstance(raw_scores, dict):
                raw_scores = {}
        except (ValueError, TypeError):
            continue

        # Bound clamps
        start = max(0.0, start)
        end = min(total_duration, end)

        if end <= start:
            continue

        duration = end - start
        if duration < min_clip_seconds:
            if start + min_clip_seconds <= total_duration:
                end = start + min_clip_seconds
                duration = end - start
            else:
                continue

        if duration > max_clip_seconds:
            end = start + max_clip_seconds
            duration = max_clip_seconds

        # 1. Standalone / Context validation gate
        is_standalone, standalone_reason = validate_standalone_context(
            item, min_standalone_score=min_standalone_score
        )
        if not is_standalone:
            log.info(
                "Rejecting clip [%.1fs - %.1fs]: failed standalone validation - %s ('%s')",
                start, end, standalone_reason, summary,
            )
            continue

        # Standalone score baseline for deterministic Python quality scoring
        try:
            cand_standalone = float(item.get("standalone_score", 5.0))
        except (ValueError, TypeError):
            cand_standalone = 5.0
        cand_standalone = max(0.0, min(10.0, cand_standalone))

        missing_setup = bool(item.get("missing_setup", False))
        missing_payoff = bool(item.get("missing_payoff", False))
        crit_ref = bool(item.get("critical_unresolved_reference", False))

        # Context dependency vs independence mapping
        if "context_dependency" in raw_scores:
            try:
                c_dep = float(raw_scores["context_dependency"])
            except (ValueError, TypeError):
                c_dep = 5.0
        elif "context_independence" in raw_scores:
            try:
                c_dep = 10.0 - float(raw_scores["context_independence"])
            except (ValueError, TypeError):
                c_dep = 5.0
        else:
            if missing_setup or crit_ref:
                c_dep = 8.0
            else:
                c_dep = max(1.0, 10.0 - cand_standalone)

        default_payoff = 2.0 if missing_payoff else cand_standalone

        scores_cleaned = {
            "hook_strength": max(0.0, min(10.0, float(raw_scores.get("hook_strength", cand_standalone)))),
            "standalone_clarity": max(0.0, min(10.0, float(raw_scores.get("standalone_clarity", cand_standalone)))),
            "payoff": max(0.0, min(10.0, float(raw_scores.get("payoff_completion", raw_scores.get("payoff", default_payoff))))),
            "curiosity": max(0.0, min(10.0, float(raw_scores.get("curiosity", min(9.0, cand_standalone))))),
            "impact": max(0.0, min(10.0, float(raw_scores.get("emotional_intellectual_impact", raw_scores.get("impact", min(9.0, cand_standalone)))))),
            "retention": max(0.0, min(10.0, float(raw_scores.get("retention_potential", raw_scores.get("retention", min(9.0, cand_standalone)))))),
            "context_dependency": max(0.0, min(10.0, c_dep)),
            "punchline_score": max(0.0, min(10.0, float(raw_scores.get("punchline_memorable_moment", raw_scores.get("punchline_score", raw_scores.get("punchline", 8.0 if punchline else 5.0)))))),
        }

        # 2. Authoritative Python calculation for quality_score (LLM score is untrusted)
        calculated_quality = calculate_quality_score(scores_cleaned)

        # 3. Filter out candidates below the minimum quality score threshold
        if calculated_quality < min_quality_score:
            log.info(
                "Rejecting clip [%.1fs - %.1fs]: quality score %.1f is below threshold %.1f ('%s')",
                start, end, calculated_quality, min_quality_score, summary,
            )
            continue

        candidate = {
            "start_time": round(start, 2),
            "end_time": round(end, 2),
            "duration": round(duration, 2),
            "topic": topic,
            "topic_summary": summary,
            "hook_summary": summary,
            "title_idea": title,
            "punchline": punchline,
            "scores": scores_cleaned,
            "quality_scores": scores_cleaned,
            "quality_score": calculated_quality,
            "selection_reason": selection_reason or standalone_reason,
            "standalone": True,
            "standalone_score": float(item.get("standalone_score", scores_cleaned["standalone_clarity"])),
            "missing_setup": bool(item.get("missing_setup", False)),
            "missing_payoff": bool(item.get("missing_payoff", False)),
            "critical_unresolved_reference": bool(item.get("critical_unresolved_reference", False)),
            "standalone_reason": standalone_reason,
            "is_fallback": False,
            "discovery_status": "SUCCESS",
        }

        candidates.append(candidate)

    # Sort candidates by quality_score descending (highest quality first)
    candidates.sort(key=lambda x: x["quality_score"], reverse=True)

    # Deduplication: prefer higher quality candidate
    accepted = []
    for cand in candidates:
        has_conflict = False
        for acc in accepted:
            # 1. Temporal overlap ratio: overlap_duration / min(dur_cand, dur_acc)
            overlap_start = max(cand["start_time"], acc["start_time"])
            overlap_end = min(cand["end_time"], acc["end_time"])
            overlap_dur = max(0.0, overlap_end - overlap_start)
            shorter_dur = min(cand["duration"], acc["duration"])
            overlap_ratio = overlap_dur / shorter_dur if shorter_dur > 0 else 0.0

            if overlap_ratio >= overlap_threshold:
                log.info(
                    "Rejecting clip [%.1fs - %.1fs] (score %.1f): overlaps %.1f%% (>= %.1f%%) with higher-score clip [%.1fs - %.1fs] (score %.1f)",
                    cand["start_time"], cand["end_time"], cand["quality_score"],
                    overlap_ratio * 100, overlap_threshold * 100,
                    acc["start_time"], acc["end_time"], acc["quality_score"],
                )
                has_conflict = True
                break

            # 2. Semantic duplication check on hook summary
            if _is_similar_summary(cand["hook_summary"], acc["hook_summary"]):
                log.info(
                    "Discarding clip [%.1fs - %.1fs] as semantically redundant with higher-score clip [%.1fs - %.1fs]: '%s'",
                    cand["start_time"], cand["end_time"], acc["start_time"], acc["end_time"], cand["hook_summary"],
                )
                has_conflict = True
                break

        if not has_conflict:
            accepted.append(cand)

        if max_clips is not None and len(accepted) >= max_clips:
            break
        elif max_clips is None and len(accepted) >= MAX_DISCOVERY_CANDIDATES:
            break

    # Preserve stable discovery identity: chronological clip_index
    chrono_sorted = sorted(accepted, key=lambda x: (x["start_time"], x["end_time"]))
    for idx, cand in enumerate(chrono_sorted, start=1):
        cand["clip_index"] = idx

    return accepted


def _fallback_clips(
    total_duration: float,
    min_clip_seconds: int,
    max_clip_seconds: int,
    max_clips: int,
) -> list:
    """
    Generates non-overlapping chronological chunks if AI detection fails.
    Assigns conservative, non-passing quality_score (0.0) and non-passing
    standalone indicators so fallback candidates never artificially bypass quality
    or standalone thresholds.
    """
    FALLBACK_PUNCHLINES = [
        "Watch Till The End 🔥",
        "Wait For The Twist 🤯",
        "Reality Check Revealed ⚡",
        "The Truth Exposed ⚠️",
        "Must Watch Insight ✨",
    ]
    clips = []
    chunk_len = float(max_clip_seconds)
    curr_start = 0.0

    fallback_scores = {
        "hook_strength": 0.0,
        "standalone_clarity": 0.0,
        "payoff": 0.0,
        "curiosity": 0.0,
        "impact": 0.0,
        "retention": 0.0,
        "context_dependency": 10.0,
        "punchline_score": 0.0,
    }

    while curr_start + min_clip_seconds <= total_duration and len(clips) < max_clips:
        curr_end = min(total_duration, curr_start + chunk_len)
        idx = len(clips) + 1
        punchline = FALLBACK_PUNCHLINES[(idx - 1) % len(FALLBACK_PUNCHLINES)]
        clips.append({
            "start_time": round(curr_start, 2),
            "end_time": round(curr_end, 2),
            "duration": round(curr_end - curr_start, 2),
            "hook_summary": f"Highlight segment #{idx} ({int(curr_start)}s - {int(curr_end)}s)",
            "title_idea": f"Key Highlight Part {idx} #shorts",
            "punchline": punchline,
            "scores": fallback_scores.copy(),
            "quality_score": 0.0,
            "selection_reason": "Emergency fallback chunk (unscored by AI)",
            "standalone": False,
            "standalone_score": 0.0,
            "missing_setup": True,
            "missing_payoff": True,
            "critical_unresolved_reference": True,
            "unresolved_references": ["unscoped_chunk"],
            "standalone_reason": "Emergency fallback chunk (unscored by AI)",
        })
        curr_start = curr_end

    if not clips:
        clips.append({
            "start_time": 0.0,
            "end_time": round(min(total_duration, float(max_clip_seconds)), 2),
            "duration": round(min(total_duration, float(max_clip_seconds)), 2),
            "hook_summary": "Highlight segment #1",
            "title_idea": "Key Highlight #shorts",
            "punchline": "Watch Till The End 🔥",
            "scores": fallback_scores.copy(),
            "quality_score": 0.0,
            "selection_reason": "Emergency fallback chunk (unscored by AI)",
            "standalone": False,
            "standalone_score": 0.0,
            "missing_setup": True,
            "missing_payoff": True,
            "critical_unresolved_reference": True,
            "unresolved_references": ["unscoped_chunk"],
            "standalone_reason": "Emergency fallback chunk (unscored by AI)",
        })

    return clips

