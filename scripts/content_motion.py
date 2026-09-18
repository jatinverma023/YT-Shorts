"""
Visual Upgrade #1B: Content-Aware Dynamic Emphasis
Provides deterministic, transcript-grounded visual emphasis for YouTube Shorts.

Key Features:
- Timestamp-based: evaluated as time = in / fps, strictly preserving timing across 25 FPS and 30 FPS
- Smooth cosine ease-in (0.35s) -> hold peak -> cosine ease-out (0.45s)
- Strictly bounded: 1.00x <= zoom(t) <= 1.04x for every frame
- Non-overlapping emphasis regions with minimum 1.5s separation (target 1–3, max 4)
- Replaces sinusoidal micro-motion when valid emphasis points exist (never stacked)
- Fallback to existing #1A.1 sinusoidal motion when no valid emphasis points exist
- Spatial-only perspective corner inset on foreground layer: zero frame drops, zero frame duplications
"""
import logging
import math
import re
from typing import Dict, List, Optional, Tuple

try:
    from config import (
        CONTENT_MOTION_ENABLED,
        CONTENT_MOTION_MAX_ZOOM,
        CONTENT_MOTION_MIN_ZOOM,
        CONTENT_MOTION_IN_DURATION,
        CONTENT_MOTION_OUT_DURATION,
        CONTENT_MOTION_MAX_POINTS,
        VISUAL_MOTION_MAX_ZOOM,
    )
except ImportError:
    from scripts.config import (
        CONTENT_MOTION_ENABLED,
        CONTENT_MOTION_MAX_ZOOM,
        CONTENT_MOTION_MIN_ZOOM,
        CONTENT_MOTION_IN_DURATION,
        CONTENT_MOTION_OUT_DURATION,
        CONTENT_MOTION_MAX_POINTS,
        VISUAL_MOTION_MAX_ZOOM,
    )

log = logging.getLogger("content_motion")

# Allowed semantic emphasis types
ALLOWED_EMPHASIS_TYPES = {
    "hook",
    "buildup",
    "key_point",
    "payoff",
    "emotional",
    "conclusion",
}

# Linguistic contrast and focal patterns used as candidate signals
CONTRAST_MARKERS = {
    "but", "however", "although", "lekin", "par", "magar", "kintu", "parantu",
}

FOCAL_PHRASES = [
    # Hindi / Hinglish focal patterns
    "asli baat", "sabse bada", "sabse badi", "sabse important",
    "dhyan se", "reality yeh hai", "yeh samajhna", "main point",
    "secret yeh hai", "ek baat", "hamesha yaad",
    # English focal patterns
    "the key is", "the most important", "the real reason",
    "the truth is", "what actually", "here is the thing",
    "remember this", "the big secret", "bottom line",
]


def validate_emphasis_point(
    point: dict,
    clip_duration: float,
) -> Tuple[bool, str]:
    """
    Deterministically validates a single candidate emphasis point.
    Returns (is_valid, reason).
    """
    if not isinstance(point, dict):
        return False, "Point must be a dictionary"

    try:
        start = float(point.get("start", -1.0))
        end = float(point.get("end", -1.0))
        strength = float(point.get("strength", 0.0))
    except (ValueError, TypeError):
        return False, "Invalid numeric values for start, end, or strength"

    # Timestamp boundaries
    if start < 0.0 or end <= start:
        return False, f"Invalid timestamps: start ({start:.2f}) must be >= 0 and < end ({end:.2f})"

    if clip_duration > 0.0 and end > clip_duration:
        return False, f"End time ({end:.2f}s) exceeds clip duration ({clip_duration:.2f}s)"

    # Duration bounds for an emphasis region
    dur = end - start
    if dur < 0.6:
        return False, f"Emphasis duration too short ({dur:.2f}s < 0.6s)"
    if dur > 5.5:
        return False, f"Emphasis duration too long ({dur:.2f}s > 5.5s)"

    # Clamp strength
    point["strength"] = max(0.0, min(1.0, strength))

    # Type normalization
    ptype = str(point.get("type", "key_point")).lower()
    point["type"] = ptype if ptype in ALLOWED_EMPHASIS_TYPES else "key_point"

    return True, "Valid"


def filter_and_deduplicate_emphasis_points(
    points: List[dict],
    clip_duration: float,
    max_points: int = CONTENT_MOTION_MAX_POINTS,
    min_separation: float = 1.5,
) -> List[dict]:
    """
    Validates, sorts, resolves overlaps, enforces minimum separation (1.5s),
    and caps the list to max_points (hard maximum 4).
    """
    if not points:
        return []

    # 1. Validate individual points
    valid_points = []
    for p in points:
        is_val, reason = validate_emphasis_point(p, clip_duration=clip_duration)
        if is_val:
            valid_points.append(dict(p))
        else:
            log.debug("Rejected emphasis point %s: %s", p, reason)

    if not valid_points:
        return []

    # 2. Sort chronologically by start time
    valid_points.sort(key=lambda x: x["start"])

    # 3. Resolve overlaps and enforce minimum separation
    merged = []
    for current in valid_points:
        if not merged:
            merged.append(current)
            continue

        prev = merged[-1]
        # Check for overlap or violation of minimum separation
        if current["start"] < (prev["end"] + min_separation):
            # Conflict: keep the one with higher strength
            if current["strength"] > prev["strength"]:
                merged[-1] = current
        else:
            merged.append(current)

    # 4. Limit to max_points (hard maximum 4)
    effective_max = max(1, min(4, max_points))
    if len(merged) > effective_max:
        # Keep top points by strength, then restore chronological order
        ranked = sorted(merged, key=lambda x: x["strength"], reverse=True)[:effective_max]
        ranked.sort(key=lambda x: x["start"])
        merged = ranked

    return merged


def detect_emphasis_points(
    words: List[dict],
    clip_duration: float,
    hook_summary: str = "",
    punchline: str = "",
    max_points: int = CONTENT_MOTION_MAX_POINTS,
) -> List[dict]:
    """
    Deterministically detects candidate emphasis points from spoken word timestamps.
    Keyword markers alone NEVER trigger emphasis; candidates are evaluated from
    full phrase context, duration, and position in the clip.
    """
    if not words or clip_duration <= 3.0:
        return []

    candidates = []
    n_words = len(words)

    # 1. Scan for focal phrases (2-4 word sequences with strong focal meaning)
    full_text = " ".join(str(w.get("word", "")).lower() for w in words)

    for focal in FOCAL_PHRASES:
        if focal in full_text:
            focal_tokens = focal.split()
            focal_len = len(focal_tokens)
            for i in range(n_words - focal_len + 1):
                window = " ".join(str(words[i + k].get("word", "")).lower() for k in range(focal_len))
                if focal in window:
                    st = float(words[i].get("start", 0.0))
                    # Require at least 3 subsequent words after focal phrase
                    if i + focal_len + 2 >= n_words:
                        continue
                    end_idx = min(n_words - 1, i + focal_len + 4)
                    et = float(words[end_idx].get("end", st + 2.0))
                    dur = et - st
                    if 0.8 <= dur <= 4.5 and st >= 0.5 and et <= (clip_duration - 0.6):
                        strength = 0.75 + (0.10 if dur >= 1.5 else 0.0)
                        candidates.append({
                            "start": round(st, 2),
                            "end": round(et, 2),
                            "strength": round(strength, 2),
                            "type": "key_point",
                        })

    # 2. Scan for contrast transitions ("lekin" / "but" etc.) delivering a new thought.
    # Keyword presence alone NEVER triggers an emphasis point.
    # It must be accompanied by a substantive following phrase and proper timestamp context.
    for i in range(n_words - 4):
        w_text = re.sub(r"[^\w]", "", str(words[i].get("word", "")).lower())
        if w_text in CONTRAST_MARKERS:
            st = float(words[i].get("start", 0.0))
            # Timestamp context: must not occur immediately at video start or end
            if st < 0.8 or st > (clip_duration - 2.0):
                continue

            # Surrounding phrase context: require at least 4 subsequent meaningful words
            subsequent_words = [
                str(words[j].get("word", "")).strip().lower()
                for j in range(i + 1, min(n_words, i + 7))
            ]
            meaningful_words = [w for w in subsequent_words if len(w) > 1 and w not in {"um", "uh", "ah", "like"}]
            if len(meaningful_words) < 4:
                continue

            end_idx = min(n_words - 1, i + 5)
            et = float(words[end_idx].get("end", st + 2.2))
            dur = et - st
            if not (1.0 <= dur <= 4.0 and et <= (clip_duration - 0.8)):
                continue

            # Candidate strength dynamically evaluated based on phrase length, pacing, and position
            base_strength = 0.60
            if len(meaningful_words) >= 5:
                base_strength += 0.10
            if 1.5 <= dur <= 3.5:
                base_strength += 0.10
            # Central narrative bonus (between 25% and 75% of clip duration)
            if 0.25 * clip_duration <= st <= 0.75 * clip_duration:
                base_strength += 0.05

            candidates.append({
                "start": round(st, 2),
                "end": round(et, 2),
                "strength": min(0.90, round(base_strength, 2)),
                "type": "buildup",
            })

    # 3. Climax / payoff region near the conclusion (between 70% and 92% of clip duration)
    # Strictly derived from transcript word timestamps, NOT hook/punchline text
    climax_target = clip_duration * 0.75
    closest_word_idx = -1
    for idx, w in enumerate(words):
        w_st = float(w.get("start", 0.0))
        if w_st >= climax_target:
            closest_word_idx = idx
            break

    if closest_word_idx != -1 and closest_word_idx < n_words - 3:
        st = float(words[closest_word_idx].get("start", 0.0))
        end_idx = min(n_words - 1, closest_word_idx + 6)
        et = float(words[end_idx].get("end", st + 2.5))
        # Ensure it does not run into the final 0.8s of the clip
        if et <= clip_duration - 0.8 and (et - st) >= 1.0:
            candidates.append({
                "start": round(st, 2),
                "end": round(et, 2),
                "strength": 0.95,
                "type": "payoff",
            })

    # Filter, deduplicate, and enforce strict spacing
    return filter_and_deduplicate_emphasis_points(
        candidates,
        clip_duration=clip_duration,
        max_points=max_points,
    )


def evaluate_zoom_at_time(
    t: float,
    emphasis_points: List[dict],
    clip_duration: float,
    max_zoom: float = CONTENT_MOTION_MAX_ZOOM,
    in_dur: float = CONTENT_MOTION_IN_DURATION,
    out_dur: float = CONTENT_MOTION_OUT_DURATION,
) -> float:
    """
    Pure Python reference evaluator for the content-aware zoom curve at timestamp t (seconds).
    Guarantees: 1.00 <= zoom <= max_zoom for all t.
    """
    effective_max = max(1.00, min(1.04, float(max_zoom), float(VISUAL_MOTION_MAX_ZOOM)))
    if effective_max <= 1.00 or not emphasis_points:
        return 1.00

    current_alpha = 0.0

    for pt in emphasis_points:
        t_start = float(pt["start"])
        t_end = float(pt["end"])
        strength = max(0.0, min(1.0, float(pt.get("strength", 1.0))))

        pt_zoom = 1.00 + (effective_max - 1.00) * strength
        pt_alpha = (pt_zoom - 1.00) / (2.0 * pt_zoom)

        t0 = max(0.0, t_start - in_dur)
        din = max(0.05, t_start - t0)
        t1 = t_end
        t2 = min(clip_duration, t_end + out_dur)
        dout = max(0.05, t2 - t1)

        if t < t0:
            continue
        elif t0 <= t < t_start:
            # Ease in (smooth cosine curve from 0 to pt_alpha)
            progress = (t - t0) / din
            pulse = pt_alpha * 0.5 * (1.0 - math.cos(math.pi * progress))
            current_alpha = max(current_alpha, pulse)
        elif t_start <= t <= t_end:
            # Hold peak
            current_alpha = max(current_alpha, pt_alpha)
        elif t_end < t <= t2:
            # Ease out (smooth cosine curve from pt_alpha to 0)
            progress = (t - t_end) / dout
            pulse = pt_alpha * 0.5 * (1.0 + math.cos(math.pi * progress))
            current_alpha = max(current_alpha, pulse)

    # Convert alpha back to zoom: zoom = 1.0 / (1.0 - 2 * alpha)
    if current_alpha <= 0.0:
        return 1.00
    denom = max(0.001, 1.0 - 2.0 * current_alpha)
    return round(min(effective_max, 1.0 / denom), 5)


def build_content_motion_curve(
    emphasis_points: List[dict],
    fps: float,
    clip_duration: float,
    max_zoom: float = CONTENT_MOTION_MAX_ZOOM,
    in_dur: float = CONTENT_MOTION_IN_DURATION,
    out_dur: float = CONTENT_MOTION_OUT_DURATION,
) -> str:
    """
    Constructs a deterministic timestamp-based FFmpeg expression for corner insetting.
    Expression uses (in/fps) so timing is strictly preserved regardless of output FPS.
    """
    effective_max = max(1.001, min(1.04, float(max_zoom), float(VISUAL_MOTION_MAX_ZOOM)))
    fps_val = max(1.0, float(fps))

    if not emphasis_points:
        return "0"

    point_terms = []
    for pt in emphasis_points:
        t_start = float(pt["start"])
        t_end = float(pt["end"])
        strength = max(0.0, min(1.0, float(pt.get("strength", 1.0))))

        pt_zoom = 1.00 + (effective_max - 1.00) * strength
        alpha = round((pt_zoom - 1.00) / (2.0 * pt_zoom), 6)

        t0 = round(max(0.0, t_start - in_dur), 3)
        din = round(max(0.05, t_start - t0), 3)
        t_end_r = round(t_end, 3)
        t_start_r = round(t_start, 3)
        t2 = round(min(clip_duration, t_end + out_dur), 3)
        dout = round(max(0.05, t2 - t_end), 3)

        time_expr = f"(in/{fps_val:.3f})"

        # 1. Ease in
        term_in = (
            f"({alpha}*0.5*(1-cos(PI*({time_expr}-{t0})/{din}))*"
            f"gte({time_expr}\\,{t0})*lt({time_expr}\\,{t_start_r}))"
        )
        # 2. Hold
        term_hold = (
            f"({alpha}*gte({time_expr}\\,{t_start_r})*lte({time_expr}\\,{t_end_r}))"
        )
        # 3. Ease out
        term_out = (
            f"({alpha}*0.5*(1+cos(PI*({time_expr}-{t_end_r})/{dout}))*"
            f"gt({time_expr}\\,{t_end_r})*lte({time_expr}\\,{t2}))"
        )

        pt_expr = f"({term_in}+{term_hold}+{term_out})"
        point_terms.append(pt_expr)

    return "+".join(point_terms)


def build_content_motion_filter(
    emphasis_points: List[dict],
    fps: float,
    clip_duration: float,
    max_zoom: float = CONTENT_MOTION_MAX_ZOOM,
    in_dur: float = CONTENT_MOTION_IN_DURATION,
    out_dur: float = CONTENT_MOTION_OUT_DURATION,
) -> str:
    """
    Constructs the final FFmpeg perspective filter string for content-aware dynamic emphasis.
    """
    a_expr = build_content_motion_curve(
        emphasis_points=emphasis_points,
        fps=fps,
        clip_duration=clip_duration,
        max_zoom=max_zoom,
        in_dur=in_dur,
        out_dur=out_dur,
    )

    return (
        f"perspective="
        f"x0=W*({a_expr}):y0=H*({a_expr}):"
        f"x1=W*(1-({a_expr})):y1=H*({a_expr}):"
        f"x2=W*({a_expr}):y2=H*(1-({a_expr})):"
        f"x3=W*(1-({a_expr})):y3=H*(1-({a_expr})):"
        f"eval=frame:interpolation=linear"
    )
