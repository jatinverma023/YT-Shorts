"""
Turn a landscape/whatever-shaped source video into a 1080x1920 Shorts-ready
clip with burned-in subtitles.

Approach:
1. Scale the video so it COVERS 1080x1920 (crop overflow), centered.
   This avoids black bars — it crops the sides/top evenly. If your source
   is a talking-head podcast recorded in landscape, this keeps the speaker
   centered in most cases. For smarter face-tracking crop, see the note
   at the bottom of this file.
2. Burn the SRT as styled subtitles via the "subtitles" filter (libass),
   positioned in the lower-safe-zone so it doesn't collide with the
   Shorts UI (like/share/comment buttons on the right, caption area).
"""
import logging
import subprocess

from config import TARGET_WIDTH, TARGET_HEIGHT, SUBTITLE_FONT, SUBTITLE_FONT_SIZE

log = logging.getLogger("video_process")


def build_ffmpeg_filter(srt_path: str) -> str:
    # scale to cover, then crop to exact target, then burn subtitles
    scale_crop = (
        f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={TARGET_WIDTH}:{TARGET_HEIGHT}"
    )
    # libass style: white text, black outline, positioned ~15% from bottom
    # to stay clear of Shorts UI elements.
    style = (
        f"FontName={SUBTITLE_FONT},FontSize={SUBTITLE_FONT_SIZE},"
        "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
        "BorderStyle=1,Outline=2,Shadow=0,Alignment=2,MarginV=180"
    )
    # escape colons/backslashes for the filter path on all platforms
    srt_escaped = srt_path.replace("\\", "\\\\").replace(":", "\\:")
    subs = f"subtitles='{srt_escaped}':force_style='{style}'"
    return f"{scale_crop},{subs}"


def process_video(input_path: str, srt_path: str, output_path: str, max_seconds=None):
    vf = build_ffmpeg_filter(srt_path)
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
    ]
    if max_seconds:
        cmd.extend(["-t", str(max_seconds)])
    cmd.extend([
        "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k",
        output_path,
    ])
    log.info("Running ffmpeg: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path


def get_duration_seconds(input_path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", input_path,
    ]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return float(out.stdout.strip())


# NOTE on smarter cropping:
# For real face/speaker-tracking crop (instead of a fixed center crop),
# you'd add a face-detection pass (e.g. OpenCV/mediapipe) per-frame or on
# keyframes to compute a moving crop window, then feed dynamic crop
# coordinates into ffmpeg's `crop` filter with expressions, or pre-render
# per-segment crops. That's a solid v2 upgrade once the basic pipeline
# is stable — flag it if you want that added.
