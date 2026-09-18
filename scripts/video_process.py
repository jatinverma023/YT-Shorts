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


def build_ffmpeg_filter(input_path: str):
    width, height = get_video_dimensions(input_path)
    is_portrait = (height > width) and ((height / width) >= 1.3)

    # Visual enhancements:
    # 1. unsharp filter makes edges, faces, and details crisp on mobile screens
    # 2. eq filter enhances contrast and color saturation for a vibrant, professional look
    enhance_filter = "unsharp=5:5:0.8:5:5:0.0,eq=contrast=1.06:saturation=1.18:brightness=0.01"

    if is_portrait:
        # Video is already vertical (e.g. 9:16)
        scale_crop = (
            f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=increase,"
            f"crop={TARGET_WIDTH}:{TARGET_HEIGHT}"
        )
        return f"{scale_crop},{enhance_filter}", False
    else:
        # Video is horizontal / landscape:
        # Background: 1080x1920 with high blur & dimmed for a sleek cinematic background
        # Foreground: crisp original video in center with sharpening and color enhancement
        filter_complex = (
            f"[0:v]scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=increase,"
            f"crop={TARGET_WIDTH}:{TARGET_HEIGHT},boxblur=25:5,eq=brightness=-0.22:saturation=1.2[bg];"
            f"[0:v]scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease,{enhance_filter}[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2[vout]"
        )
        return filter_complex, True


def process_video(input_path: str, output_path: str, max_seconds=None):
    filt, is_complex = build_ffmpeg_filter(input_path)
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

    cmd.extend([
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
