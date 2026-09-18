"""
Video processing pipeline for YouTube Shorts:
- Auto-trims silences/dead air (>0.5s) using Whisper word timestamps
- Creates 9:16 vertical video (1080x1920) with blurred background & parallax slow zoom
- Enforces even width/height to avoid odd-pixel crashes
- Burns animated word-level ASS captions (karaoke/pop style)
- Applies audio loudness normalization (loudnorm filter)
- Encodes with -pix_fmt yuv420p and -movflags +faststart, with full stderr logging on failure
"""
import logging
import os
import subprocess

from config import (
    TARGET_WIDTH, TARGET_HEIGHT, ENABLE_ZOOM, BG_ZOOM_SPEED, FG_ZOOM_SPEED,
    ENABLE_SILENCE_REMOVAL, SILENCE_THRESHOLD_SECONDS, SILENCE_PADDING_SECONDS,
    ENABLE_LOUDNORM, LOUDNORM_TARGET_I,
    VISUAL_CONTRAST, VISUAL_SATURATION, VISUAL_BRIGHTNESS, VISUAL_SHARPEN_AMOUNT,
    VISUAL_VIGNETTE_ENABLED, VISUAL_VIGNETTE_STRENGTH,
    VISUAL_MOTION_ENABLED, VISUAL_MOTION_MAX_ZOOM,
    BG_BRIGHTNESS, BG_SATURATION, FOREGROUND_SCALE, TARGET_FPS,
)

log = logging.getLogger("video_process")


def get_video_dimensions(input_path: str):
    """Return (width, height) of the video using ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=s=x:p=0",
            input_path,
        ]
        out = subprocess.run(cmd, check=True, capture_output=True, text=True)
        parts = out.stdout.strip().split("x")
        return int(parts[0]), int(parts[1])
    except Exception as e:
        log.warning("Could not probe video dimensions: %s. Defaulting to 1920x1080.", e)
        return 1920, 1080


def get_duration_seconds(input_path: str) -> float:
    """Return video duration in seconds."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", input_path,
    ]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return float(out.stdout.strip())


def extract_clip_segment(source_path: str, start_time: float, end_time: float, output_path: str) -> str:
    """
    Extracts an exact time slice from the source video (start_time to end_time).
    Accurate timestamp-based FFmpeg extraction with re-encoding to guarantee
    accurate boundaries and avoid frozen audio/video frames at the cut point.
    """
    duration = max(1.0, end_time - start_time)
    cmd = [
        "ffmpeg", "-y",
        "-i", source_path,
        "-ss", f"{start_time:.3f}",
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-c:a", "aac", "-b:a", "128k",
        "-avoid_negative_ts", "make_zero",
        output_path,
    ]
    log.info("Extracting clip segment [%.1fs - %.1fs] (%.1fs) -> %s", start_time, end_time, duration, output_path)
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        log.error("FFmpeg extract_clip_segment failed with stderr:\n%s", e.stderr)
        raise
    return output_path


def trim_silences_from_words(input_path: str, words: list, output_path: str, max_duration=None):
    """
    Detects and selectively compresses/removes excessive pauses using Whisper word boundaries
    while preserving natural conversational pauses.
    """
    if not words or not ENABLE_SILENCE_REMOVAL:
        return input_path, words

    total_duration = get_duration_seconds(input_path)
    if max_duration:
        total_duration = min(total_duration, max_duration)

    pad = SILENCE_PADDING_SECONDS  # e.g. 0.12s
    keep_threshold = SILENCE_THRESHOLD_SECONDS  # default 0.70s: pauses <= this are preserved
    long_threshold = 1.20  # excessive dead-air threshold

    cuts = []  # list of (cut_start, cut_end)

    # Internal inter-word pause analysis & boundary-aware compression
    for i in range(len(words) - 1):
        w_curr = words[i]
        w_next = words[i + 1]

        w_curr_end = float(w_curr["end"])
        w_next_start = float(w_next["start"])

        # Timestamp safety: ensure gap is measured between valid forward-progressing boundaries
        if w_next_start <= w_curr_end:
            continue

        gap = w_next_start - w_curr_end

        # Tier 1: Very short (<0.35s) or normal conversational pause (<0.70s)
        # KEEP: Preserve natural breathing, dramatic/emotional timing, thinking pauses
        if gap <= keep_threshold:
            continue

        # Tier 2: Moderate dead-air (0.70s - 1.20s)
        # COMPRESS: Do not hard-cut to zero! Leave a natural ~0.28s conversational pause
        elif gap <= long_threshold:
            pad_start = max(pad, 0.14)
            pad_end = max(pad, 0.14)
            cut_start = w_curr_end + pad_start
            cut_end = w_next_start - pad_end
            if cut_end - cut_start >= 0.15:
                cuts.append((cut_start, cut_end))

        # Tier 3: Long dead-air (> 1.20s)
        # REMOVE / STRONGLY COMPRESS: Cut excessive dead air, leaving a clean ~0.20s breath
        else:
            pad_start = min(pad, 0.10)
            pad_end = min(pad, 0.10)
            cut_start = w_curr_end + pad_start
            cut_end = w_next_start - pad_end
            if cut_end - cut_start >= 0.15:
                cuts.append((cut_start, cut_end))

    # If no excessive internal dead-air pauses were identified to cut
    if not cuts:
        log.info("No dead air gaps > %.2fs found. Preserving natural conversational pauses.", keep_threshold)
        return input_path, words


    # 4. Build contiguous keep intervals from cuts
    keep_intervals = []
    current_pos = 0.0

    for (c_start, c_end) in cuts:
        if c_start > current_pos + 0.05:
            keep_intervals.append((current_pos, c_start))
        current_pos = c_end

    if current_pos < total_duration - 0.05:
        keep_intervals.append((current_pos, total_duration))

    if len(keep_intervals) <= 1 and not (cuts and keep_intervals):
        log.info("No valid keep segments constructed. Skipping silence cut.")
        return input_path, words

    log.info(
        "Boundary-aware pause compression: cutting %d excessive gap(s) while preserving natural conversational pauses. Keep segments: %d",
        len(cuts), len(keep_intervals)
    )

    # 5. Build FFmpeg filter complex to extract and concatenate keep intervals synchronously
    filter_parts = []
    concat_inputs = []
    for idx, (s, e) in enumerate(keep_intervals):
        filter_parts.append(f"[0:v]trim=start={s:.3f}:end={e:.3f},setpts=PTS-STARTPTS[v{idx}];")
        filter_parts.append(f"[0:a]atrim=start={s:.3f}:end={e:.3f},asetpts=PTS-STARTPTS[a{idx}];")
        concat_inputs.append(f"[v{idx}][a{idx}]")

    n_segs = len(keep_intervals)
    concat_str = "".join(concat_inputs) + f"concat=n={n_segs}:v=1:a=1[vcut][acut]"
    full_filter = "".join(filter_parts) + concat_str

    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-filter_complex", full_filter,
        "-map", "[vcut]", "-map", "[acut]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "19",
        "-c:a", "aac", "-b:a", "128k",
        output_path,
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        log.error("FFmpeg silence trimming failed with stderr:\n%s", e.stderr)
        return input_path, words

    # 6. Shift word timestamps so subtitles remain 100% in sync with the tightened cut
    shifted_words = []
    for w in words:
        orig_start = float(w["start"])
        orig_end = float(w["end"])

        sub_start = 0.0
        sub_end = 0.0
        for (c_start, c_end) in cuts:
            c_len = c_end - c_start
            if orig_start >= c_end:
                sub_start += c_len
            elif orig_start > c_start:
                sub_start += (orig_start - c_start)

            if orig_end >= c_end:
                sub_end += c_len
            elif orig_end > c_start:
                sub_end += (orig_end - c_start)

        shifted_words.append({
            "word": w["word"],
            "start": max(0.0, round(orig_start - sub_start, 3)),
            "end": max(0.05, round(orig_end - sub_end, 3)),
        })

    actual_out_dur = get_duration_seconds(output_path)
    log.info("Tightened video from %.2fs to %.2fs (removed %.2fs of dead air).", total_duration, actual_out_dur, total_duration - actual_out_dur)
    return output_path, shifted_words


def check_has_audio(input_path: str) -> bool:
    """Checks whether the input video file contains an audio stream."""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=codec_type",
            "-of", "csv=p=0",
            input_path,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        return bool(res.stdout.strip())
    except Exception as e:
        log.warning("ffprobe audio check failed (%s), assuming audio exists.", e)
        return True


def get_video_fps(input_path: str) -> float:
    """Return video frame rate (e.g. 30.0, 25.0) using ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            input_path,
        ]
        out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip()
        if "/" in out:
            num, den = out.split("/")
            return float(num) / float(den)
        return float(out)
    except Exception as e:
        log.warning("Could not probe video frame rate: %s. Defaulting to 30.0 fps.", e)
        return 30.0


def build_micro_motion_filter(fps: float = 30.0, max_zoom: float = VISUAL_MOTION_MAX_ZOOM) -> str:
    """
    Constructs a smooth, deterministic sinusoidal micro-zoom filter using FFmpeg's
    perspective filter evaluated per-frame on the input frame index ('in').
    Cycle duration: 10 seconds (10 * fps frames).
    Range: 1.00x -> max_zoom -> 1.00x.
    Guarantees:
    - Exactly 100% preservation of frame timing and frame counts
    - Zero frame drops or frame duplications
    - Exact frame dimension preservation
    - No audio or subtitle desync
    """
    cycle_frames = max(30, int(round(fps * 10)))
    clamped_zoom = max(1.001, min(1.10, float(max_zoom)))
    max_alpha = round((clamped_zoom - 1.0) / (2.0 * clamped_zoom), 6)
    a_expr = f"{max_alpha}*0.5*(1-cos(2*PI*in/{cycle_frames}))"

    return (
        f"perspective="
        f"x0=W*({a_expr}):y0=H*({a_expr}):"
        f"x1=W*(1-({a_expr})):y1=H*({a_expr}):"
        f"x2=W*({a_expr}):y2=H*(1-({a_expr})):"
        f"x3=W*(1-({a_expr})):y3=H*(1-({a_expr})):"
        f"eval=frame:interpolation=linear"
    )


def build_ffmpeg_filter(input_path: str, ass_path: str = None, has_audio: bool = True):
    """
    Constructs the FFmpeg filtergraph and stream mappings:
    - Vertical Short framing (blurred/dimmed backdrop for landscape, centered sharp foreground)
    - Deterministic foreground micro-motion (subtle breathing zoom, no frame rate alteration)
    - Professional color grading (contrast, saturation, brightness, unsharp)
    - Subtle 9:16 elliptical vignette on final composition
    - Burned-in ASS captions (karaoke/pop style, completely unaltered)
    - EBU R128 loudness normalization integrated into filtergraph
    """
    width, height = get_video_dimensions(input_path)
    fps = get_video_fps(input_path)
    is_portrait = (height > width) and ((height / width) >= 1.3)

    # 1. Professional color treatment (conservative, crisp, natural skin tones)
    enhance = (
        f"unsharp=5:5:{VISUAL_SHARPEN_AMOUNT:.2f}:5:5:0.0,"
        f"eq=contrast={VISUAL_CONTRAST:.2f}:saturation={VISUAL_SATURATION:.2f}:brightness={VISUAL_BRIGHTNESS:.2f}"
    )

    # 2. Subtle vignette on the 1080x1920 composition (applied before subtitles)
    if VISUAL_VIGNETTE_ENABLED:
        vignette_filter = "vignette=angle=PI/4:aspect=9/16"
    else:
        vignette_filter = None

    # 3. Micro-motion filter (applied strictly to the visual foreground layer)
    if VISUAL_MOTION_ENABLED and VISUAL_MOTION_MAX_ZOOM > 1.0:
        motion_filter = build_micro_motion_filter(fps=fps, max_zoom=VISUAL_MOTION_MAX_ZOOM)
    else:
        motion_filter = None

    # 4. Subtitle burning filter (applied LAST to visual canvas)
    if ass_path and os.path.isfile(ass_path):
        ass_escaped = ass_path.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
        sub_filter = f"subtitles='{ass_escaped}'"
    else:
        sub_filter = "null"

    if is_portrait:
        # Video is already vertical (e.g. 9:16)
        scale_crop = (
            f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=increase,"
            f"crop={TARGET_WIDTH}:{TARGET_HEIGHT}"
        )
        fg_filters = [scale_crop]
        if motion_filter:
            fg_filters.append(motion_filter)
        fg_filters.append(enhance)
        if vignette_filter:
            fg_filters.append(vignette_filter)
        fg_filters.append(sub_filter)
        video_chain = f"[0:v]{','.join(fg_filters)}[vout]"
    else:
        # Video is horizontal / landscape (e.g. 16:9)
        # Background: 1080x1920 blurred + dimmed + saturated
        bg_chain = (
            f"[0:v]scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=increase,"
            f"crop={TARGET_WIDTH}:{TARGET_HEIGHT},boxblur=25:5,"
            f"eq=brightness={BG_BRIGHTNESS:.2f}:saturation={BG_SATURATION:.2f}[bg]"
        )

        # Foreground: center original aspect ratio scaled with even dimensions + micro-motion
        if FOREGROUND_SCALE > 1.0:
            fg_w = int(round((TARGET_WIDTH * FOREGROUND_SCALE) / 2.0)) * 2
            fg_scale_filters = [
                f"scale={fg_w}:-2:force_original_aspect_ratio=decrease",
                "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                f"crop=min(iw\\,{TARGET_WIDTH}):ih",
            ]
        else:
            fg_scale_filters = [
                f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease",
                "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            ]

        fg_filters = list(fg_scale_filters)
        if motion_filter:
            fg_filters.append(motion_filter)
        fg_filters.append(enhance)
        fg_chain = f"[0:v]{','.join(fg_filters)}[fg]"

        # Merge foreground on background
        post_overlay = []
        if vignette_filter:
            post_overlay.append(vignette_filter)
        post_overlay.append(sub_filter)
        post_str = f",{','.join(post_overlay)}" if post_overlay else ""

        merge_chain = f"[bg][fg]overlay=(W-w)/2:(H-h)/2{post_str}[vout]"
        video_chain = f"{bg_chain};{fg_chain};{merge_chain}"

    # Audio stream handling
    if has_audio and ENABLE_LOUDNORM:
        audio_chain = f";[0:a]loudnorm=I={LOUDNORM_TARGET_I}:LRA=11:TP=-1.5[aout]"
        filter_complex = video_chain + audio_chain
        maps = ["-map", "[vout]", "-map", "[aout]"]
    elif has_audio:
        filter_complex = video_chain
        maps = ["-map", "[vout]", "-map", "0:a"]
    else:
        filter_complex = video_chain
        maps = ["-map", "[vout]"]

    return filter_complex, maps


def process_video(input_path: str, output_path: str, ass_path: str = None, max_seconds=None):
    """
    Renders the final vertical Short with:
    - Even-pixel scaling & visual enhancements
    - Animated ASS captions
    - Audio loudnorm normalization
    - -pix_fmt yuv420p & -movflags +faststart
    - Full stderr logging on failure
    """
    has_audio = check_has_audio(input_path)
    filt, maps = build_ffmpeg_filter(input_path, ass_path, has_audio=has_audio)

    cmd = ["ffmpeg", "-y", "-i", input_path]
    if max_seconds:
        cmd.extend(["-t", str(max_seconds)])

    cmd.extend(["-filter_complex", filt])
    cmd.extend(maps)

    # Output encoding parameters
    cmd.extend([
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p",
    ])
    if TARGET_FPS and TARGET_FPS > 0:
        cmd.extend(["-r", str(TARGET_FPS)])
    if has_audio:
        cmd.extend(["-c:a", "aac", "-b:a", "128k"])

    cmd.extend([
        "-movflags", "+faststart",
        output_path,
    ])

    log.info("Running ffmpeg render command...")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        log.error("FFmpeg render failed with exit code %d:\n%s", e.returncode, e.stderr)
        raise

    log.info("Rendered finished Short at: %s", output_path)
    return output_path
