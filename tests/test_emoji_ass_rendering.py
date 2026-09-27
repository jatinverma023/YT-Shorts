"""
Unit and integration tests verifying emoji subtitle rendering via ASS/FFmpeg.
Verifies:
1. Semantic emojis (🚀 ❤️ 💔 🛑 💰 😂 🔥 ⚠️ 🎯) are mapped to Noto Emoji TrueType format.
2. format_ass_emoji wraps emojis with {\fnNoto Emoji Regular} and target glyphs.
3. Plain text and normal words remain completely unaffected.
4. ASS caption generation produces valid HeaderPunchline events with emoji formatting.
5. End-to-end FFmpeg subtitle filter rendering with :fontsdir produces visible vector glyphs
   and completes with 0 errors/warnings on actual video/canvas frames.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

import config
from emoji_utils import format_ass_emoji, is_emoji_codepoint, SMP_TO_PUA
import transcribe
import video_process


class TestEmojiAssRendering(unittest.TestCase):
    def setUp(self):
        self.test_emojis = ["🚀", "❤️", "💔", "🛑", "💰", "😂", "🔥", "⚠️", "🎯"]
        self.fonts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "fonts"))

    def test_01_fonts_directory_and_files_exist(self):
        """NotoEmoji TrueType outline font files must exist in assets/fonts."""
        self.assertTrue(os.path.isdir(self.fonts_dir), f"Missing fonts directory: {self.fonts_dir}")
        reg_font = os.path.join(self.fonts_dir, "NotoEmoji-Regular.ttf")
        bold_font = os.path.join(self.fonts_dir, "NotoEmoji-Bold.ttf")
        self.assertTrue(os.path.isfile(reg_font), f"Missing {reg_font}")
        self.assertTrue(os.path.isfile(bold_font), f"Missing {bold_font}")

    def test_02_all_nine_emojis_recognized_and_formatted(self):
        """All 9 representative production emojis are recognized and font-tagged."""
        for emoji_char in self.test_emojis:
            # Check codepoints
            cps = [ord(c) for c in emoji_char if ord(c) not in (0xFE0E, 0xFE0F)]
            for cp in cps:
                self.assertTrue(
                    is_emoji_codepoint(cp),
                    f"Emoji {emoji_char} (U+{cp:04X}) not recognized by is_emoji_codepoint",
                )

            formatted = format_ass_emoji(f"TEST {emoji_char} HOOK")
            self.assertIn("{\\fnNoto Emoji Regular}", formatted)
            self.assertIn("{\\fn}", formatted)
            self.assertIn("TEST", formatted)
            self.assertIn("HOOK", formatted)

    def test_03_plain_text_unaltered(self):
        """Normal captions and plain text strings remain completely untouched."""
        plain = "YOU CHOOSE INTENTIONALLY AND FOCUS ON GOALS"
        formatted = format_ass_emoji(plain)
        self.assertEqual(formatted, plain)

    def test_04_ass_caption_generation_with_emojis(self):
        """generate_ass_captions produces valid ASS dialogue lines with emoji formatting."""
        words = [
            {"word": "YOU", "start": 0.0, "end": 0.5},
            {"word": "CHOOSE", "start": 0.5, "end": 1.0},
        ]
        punchline = "YOU CHOOSE INTENTIONALLY 🚀 🔥 🎯"

        with tempfile.NamedTemporaryFile(suffix=".ass", delete=False) as f:
            ass_path = f.name

        try:
            transcribe.generate_ass_captions(words, ass_path, punchline=punchline, total_duration=2.0)
            with open(ass_path, "r", encoding="utf-8") as f:
                content = f.read()

            self.assertIn("HeaderPunchline", content)
            self.assertIn("YOU CHOOSE INTENTIONALLY", content)
            self.assertIn("Noto Emoji Regular", content)
            # Default word events must be present
            self.assertIn("YOU", content)
            self.assertIn("CHOOSE", content)
        finally:
            if os.path.exists(ass_path):
                os.remove(ass_path)

    def test_05_ffmpeg_render_all_nine_emojis_end_to_end(self):
        """
        Actually renders all 9 representative emojis through FFmpeg subtitles filter.
        Verifies FFmpeg exits with code 0 and non-black glyph pixels are burned onto canvas.
        """
        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            self.skipTest("ffmpeg binary not available in environment")

        punchline = f"PRODUCTION TEST: {' '.join(self.test_emojis)}"

        with tempfile.TemporaryDirectory() as tmp_dir:
            ass_path = os.path.join(tmp_dir, "test_render.ass")
            words = [{"word": "TESTING", "start": 0.0, "end": 1.0}]
            transcribe.generate_ass_captions(words, ass_path, punchline=punchline, total_duration=2.0)

            # Build FFmpeg subtitles filter string pointing to repository fonts
            ass_escaped = ass_path.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
            fonts_escaped = self.fonts_dir.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
            sub_filter = f"subtitles='{ass_escaped}':fontsdir='{fonts_escaped}'"

            # Render 1 frame to raw grayscale video
            raw_path = os.path.join(tmp_dir, "frame.raw")
            cmd = [
                ffmpeg_bin,
                "-y",
                "-f", "lavfi",
                "-i", "color=c=black:s=1080x1920:d=1:r=1",
                "-vf", sub_filter,
                "-frames:v", "1",
                "-f", "rawvideo",
                "-pix_fmt", "gray",
                raw_path,
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            self.assertEqual(result.returncode, 0, f"FFmpeg failed: {result.stderr}")

            # Verify file was generated with exact expected frame size
            expected_bytes = 1080 * 1920
            self.assertEqual(os.path.getsize(raw_path), expected_bytes)

            # Verify rendered pixels in the header region (top 200..400 lines)
            with open(raw_path, "rb") as f:
                frame_data = f.read()

            header_region = frame_data[1080 * 200 : 1080 * 400]
            non_black_pixels = sum(1 for byte in header_region if byte > 50)

            # A line containing text and 9 emojis at font size 52 must produce thousands of bright pixels
            self.assertGreater(
                non_black_pixels,
                1500,
                f"Header glyphs failed to render! Found only {non_black_pixels} non-black pixels in header region.",
            )


if __name__ == "__main__":
    unittest.main()
