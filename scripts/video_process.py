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


def trim_silences_from_words(input_path: str, words: list, output_path: str, max_duration=None):
    """
    Identifies dead air gaps (> SILENCE_THRESHOLD_SECONDS) from Whisper word timestamps,
    trims them from the video/audio, and returns (tightened_video_path, shifted_words).
    """
    if not words or not ENABLE_SILENCE_REMOVAL:
        return input_path, words

    total_duration = get_duration_seconds(input_path)
    if max_duration:
        total_duration = min(total_duration, max_duration)

    # Build keep intervals
    keep_intervals = []
    pad = SILENCE_PADDING_SECONDS

    # Initial start
    first_word_start = max(0.0, words[0]["start"] - pad)
    current_seg_start = first_word_start
    current_seg_end = words[0]["end"] + pad

    cuts = []  # list of (cut_start, cut_end)

    for i in range(len(words) - 1):
        w_curr = words[i]
        w_next = words[i + 1]

        gap = w_next["start"] - w_curr["end"]
        if gap > SILENCE_THRESHOLD_SECONDS:
            # End current segment
            cut_start = w_curr["end"] + pad
            cut_end = max(cut_start, w_next["start"] - pad)
            if cut_end > cut_start + 0.1:
                keep_intervals.append((current_seg_start, cut_start))
                cuts.append((cut_start, cut_end))
                current_seg_start = cut_end
        current_seg_end = min(total_duration, w_next["end"] + pad)

    keep_intervals.append((current_seg_start, min(total_duration, current_seg_end)))

    # If no significant silences found to cut
    if not cuts or len(keep_intervals) <= 1:
        log.info("No dead air gaps > %.2fs found. Skipping silence cut.", SILENCE_THRESHOLD_SECONDS)
        return input_path, words

    log.info("Cutting %d silence gaps to tighten pacing. Total keep segments: %d", len(cuts), len(keep_intervals))

    # Build ffmpeg filter to extract and concat keep intervals
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

    # Shift word timestamps so subtitles remain 100% in sync with the tightened cut
    shifted_words = []
    for w in words:
        orig_start = w["start"]
        orig_end = w["end"]
        # Subtract all cuts that happened before this word
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
            "start": max(0.0, orig_start - sub_start),
            "end": max(0.05, orig_end - sub_end),
        })

    log.info("Tightened video from %.1fs to %.1fs.", total_duration, get_duration_seconds(output_path))
    return output_path, shifted_words


def build_ffmpeg_filter(input_path: str, ass_path: str = None):
    """
    Constructs the video filter string:
    - Center uncropped foreground with even-pixel scaling
    - Blurred background with subtle parallax zoom motion
    - Sharpness and color grading
    - Burned-in ASS captions (karaoke/pop style)
    """
    width, height = get_video_dimensions(input_path)
    is_portrait = (height > width) and ((height / width) >= 1.3)

    # Visual enhancements for crispness and rich colors on mobile displays
    enhance = "unsharp=5:5:0.8:5:5:0.0,eq=contrast=1.06:saturation=1.18:brightness=0.01"

    # Motion / Zoom expressions (subtle slow zoom for depth)
    # Using dynamic crop expressions: zoom speed per second
    bg_zoom_rate = BG_ZOOM_SPEED * 25.0 if ENABLE_ZOOM else 0.0
    fg_zoom_rate = FG_ZOOM_SPEED * 25.0 if ENABLE_ZOOM else 0.0

    if ENABLE_ZOOM and bg_zoom_rate > 0:
        bg_zoom_filter = f",crop=w='in_w/(1+{bg_zoom_rate:.5f}*t)':h='in_h/(1+{bg_zoom_rate:.5f}*t)':x='(in_w-out_w)/2':y='(in_h-out_h)/2'"
        fg_zoom_filter = f",crop=w='in_w/(1+{fg_zoom_rate:.5f}*t)':h='in_h/(1+{fg_zoom_rate:.5f}*t)':x='(in_w-out_w)/2':y='(in_h-out_h)/2'"
    else:
        bg_zoom_filter = ""
        fg_zoom_filter = ""

    # Even-pixel scaling constraint to avoid odd-pixel crashes
    even_fg_scale = "scale='trunc(iw*min(1080/iw\,1920/ih)/2)*2':'trunc(ih*min(1080/iw\,1920/ih)/2)*2'"

    # Subtitle burning filter
    if ass_path and os.path.isfile(ass_path):
        ass_escaped = ass_path.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
        sub_filter = f"subtitles='{ass_escaped}'"
    else:
        sub_filter = "null"

    if is_portrait:
        # Video is already vertical (e.g. 9:16)
        scale_crop = (
            f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=increase,"
            f"crop={TARGET_WIDTH}:{TARGET_HEIGHT}"
        )
        return f"{scale_crop}{fg_zoom_filter},{enhance},{sub_filter}", False
    else:
        # Video is horizontal / landscape:
        # Background: 1080x1920 blurred + dimmed + slow zoom
        # Foreground: crisp center video with even dimensions + subtle slower zoom + unsharp
        filter_complex = (
            f"[0:v]scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=increase,"
            f"crop={TARGET_WIDTH}:{TARGET_HEIGHT}{bg_zoom_filter},boxblur=25:5,eq=brightness=-0.22:saturation=1.2[bg];"
            f"[0:v]{even_fg_scale}{fg_zoom_filter},{enhance}[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2[merged];"
            f"[merged]{sub_filter}[vout]"
        )
        return filter_complex, True


def process_video(input_path: str, output_path: str, ass_path: str = None, max_seconds=None):
    """
    Renders the final vertical Short with:
    - Parallax zoom & even-pixel scaling
    - Animated ASS captions
    - Audio loudnorm normalization
    - -pix_fmt yuv420p & -movflags +faststart
    - Full stderr logging on failure
    """
    filt, is_complex = build_ffmpeg_filter(input_path, ass_path)
    cmd = ["ffmpeg", "-y", "-i", input_path]
    if max_seconds:
        cmd.extend(["-t", str(max_seconds)])

    if is_complex:
        cmd.extend([
            "-filter_complex", filt,
            "-map", "[vout]",
            "-map", "0:a?",
        ])
    else:
        cmd.extend([
            "-vf", filt,
        ])

    # Audio loudness normalization (YouTube standard -14 LUFS)
    if ENABLE_LOUDNORM:
        cmd.extend(["-af", f"loudnorm=I={LOUDNORM_TARGET_I}:LRA=11:TP=-1.5"])

    # Output encoding parameters
    cmd.extend([
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
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
