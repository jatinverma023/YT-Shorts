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


def build_ffmpeg_filter(input_path: str, srt_path: str):
    width, height = get_video_dimensions(input_path)
    is_portrait = (height > width) and ((height / width) >= 1.3)

    # Subtitle styling with bold font, dark outline, clean shadow, and proper margins
    style = (
        f"FontName={SUBTITLE_FONT},FontSize={SUBTITLE_FONT_SIZE},Bold=1,"
        "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BackColour=&H80000000,"
        "BorderStyle=1,Outline=2.5,Shadow=1,Alignment=2,MarginV=180"
    )
    srt_escaped = srt_path.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    subs = f"subtitles='{srt_escaped}':force_style='{style}'"

    if is_portrait:
        # Video is already vertical (e.g. 9:16)
        scale_crop = (
            f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=increase,"
            f"crop={TARGET_WIDTH}:{TARGET_HEIGHT}"
        )
        return f"{scale_crop},{subs}", False
    else:
        # Video is horizontal / landscape:
        # 1. Background: scaled to 1080x1920 with high blur & dimmed
        # 2. Foreground: full original video uncropped in center (100% subject visible)
        # 3. Subtitles burned cleanly below the video
        filter_complex = (
            f"[0:v]scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=increase,"
            f"crop={TARGET_WIDTH}:{TARGET_HEIGHT},boxblur=25:5,eq=brightness=-0.15[bg];"
            f"[0:v]scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2[merged];"
            f"[merged]{subs}[vout]"
        )
        return filter_complex, True


def process_video(input_path: str, srt_path: str, output_path: str, max_seconds=None):
    filt, is_complex = build_ffmpeg_filter(input_path, srt_path)
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
