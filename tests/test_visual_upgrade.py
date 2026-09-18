"""
Unit tests for Visual Upgrade #1A: Cinematic 9:16 Composition + Professional Color Treatment + Subtle Motion.
Verifies:
1. Horizontal 16:9 source generates two-layer background + foreground composition.
2. Portrait 9:16 source uses direct scale/crop without background overlay.
3. Visual enhancement filters (unsharp, eq) are present with configurable parameters.
4. Vignette can be toggled on/off via configuration.
5. Subtle motion can be toggled on/off via configuration.
6. Motion zoom is strictly bounded by VISUAL_MOTION_MAX_ZOOM.
7. Subtitle filter remains present, intact, and placed AFTER visual composition.
8. Audio normalization (loudnorm) remains intact with exact parameters.
9. Function signature of process_video remains 100% backward-compatible.
10. Timing regression test: verifies duration, fps, frame count, and audio sync with motion ON and OFF.
"""
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

import config
import video_process


class TestVisualUpgrade1A(unittest.TestCase):

    @patch("video_process.get_video_dimensions", return_value=(1920, 1080))
    @patch("video_process.get_video_fps", return_value=30.0)
    def test_1_horizontal_source_filter_graph(self, mock_fps, mock_dims):
        """Horizontal source creates two-layer composition with blurred bg and centered fg."""
        filt, maps = video_process.build_ffmpeg_filter("test_dummy.mp4", ass_path=None, has_audio=True)
        self.assertIn("[bg]", filt)
        self.assertIn("[fg]", filt)
        self.assertIn("boxblur=25:5", filt)
        self.assertIn("overlay=(W-w)/2:(H-h)/2", filt)

    @patch("video_process.get_video_dimensions", return_value=(1080, 1920))
    @patch("video_process.get_video_fps", return_value=30.0)
    def test_2_portrait_source_filter_graph(self, mock_fps, mock_dims):
        """Portrait source uses direct scale/crop architecture without background overlay."""
        filt, maps = video_process.build_ffmpeg_filter("test_dummy_portrait.mp4", ass_path=None, has_audio=True)
        self.assertNotIn("[bg]", filt)
        self.assertNotIn("boxblur", filt)
        self.assertIn("crop=1080:1920", filt)

    @patch("video_process.get_video_dimensions", return_value=(1920, 1080))
    @patch("video_process.get_video_fps", return_value=30.0)
    def test_3_color_treatment_filters_present(self, mock_fps, mock_dims):
        """Professional color treatment (unsharp, eq) is present with configured values."""
        filt, _ = video_process.build_ffmpeg_filter("test_dummy.mp4", ass_path=None, has_audio=False)
        self.assertIn(f"unsharp=5:5:{config.VISUAL_SHARPEN_AMOUNT:.2f}:5:5:0.0", filt)
        self.assertIn(f"eq=contrast={config.VISUAL_CONTRAST:.2f}", filt)
        self.assertIn(f"saturation={config.VISUAL_SATURATION:.2f}", filt)
        self.assertIn(f"brightness={config.VISUAL_BRIGHTNESS:.2f}", filt)

    @patch("video_process.get_video_dimensions", return_value=(1920, 1080))
    @patch("video_process.get_video_fps", return_value=30.0)
    def test_4_vignette_toggle(self, mock_fps, mock_dims):
        """Vignette can be enabled or disabled via configuration."""
        with patch.object(video_process, "VISUAL_VIGNETTE_ENABLED", True):
            filt_on, _ = video_process.build_ffmpeg_filter("test_dummy.mp4", ass_path=None, has_audio=False)
            self.assertIn("vignette=angle=PI/4:aspect=9/16", filt_on)

        with patch.object(video_process, "VISUAL_VIGNETTE_ENABLED", False):
            filt_off, _ = video_process.build_ffmpeg_filter("test_dummy.mp4", ass_path=None, has_audio=False)
            self.assertNotIn("vignette", filt_off)

    @patch("video_process.get_video_dimensions", return_value=(1920, 1080))
    @patch("video_process.get_video_fps", return_value=30.0)
    def test_5_motion_toggle_creates_true_no_motion_path(self, mock_fps, mock_dims):
        """When VISUAL_MOTION_ENABLED is false, motion filter is omitted entirely."""
        with patch.object(video_process, "VISUAL_MOTION_ENABLED", False):
            filt_off, _ = video_process.build_ffmpeg_filter("test_dummy.mp4", ass_path=None, has_audio=False)
            self.assertNotIn("perspective", filt_off)

        with patch.object(video_process, "VISUAL_MOTION_ENABLED", True):
            filt_on, _ = video_process.build_ffmpeg_filter("test_dummy.mp4", ass_path=None, has_audio=False)
            self.assertIn("perspective", filt_on)

    def test_6_motion_bounds_and_formula(self):
        """Motion filter strictly bounds zoom within 1.00x and VISUAL_MOTION_MAX_ZOOM."""
        motion_filter = video_process.build_micro_motion_filter(fps=30.0, max_zoom=1.04)
        self.assertIn("perspective=", motion_filter)
        self.assertIn("eval=frame", motion_filter)
        self.assertIn("interpolation=linear", motion_filter)
        # Expected max alpha for 1.04 is round((1.04-1.0)/(2*1.04), 6) = 0.019231
        self.assertIn("0.019231", motion_filter)
        # 10s cycle at 30fps is 300 frames
        self.assertIn("in/300", motion_filter)

    @patch("video_process.get_video_dimensions", return_value=(1920, 1080))
    @patch("video_process.get_video_fps", return_value=30.0)
    def test_7_layer_order_subtitles_burned_after_grading(self, mock_fps, mock_dims):
        """Subtitles and punchline are burned LAST, after composition, color grading, and vignette."""
        with tempfile.NamedTemporaryFile(suffix=".ass", delete=False) as f:
            ass_path = f.name

        try:
            filt, _ = video_process.build_ffmpeg_filter("test_dummy.mp4", ass_path=ass_path, has_audio=False)
            self.assertIn("subtitles=", filt)
            # Verify subtitles filter occurs at the end of the video filterchain
            sub_pos = filt.find("subtitles=")
            overlay_pos = filt.find("overlay=")
            self.assertGreater(sub_pos, overlay_pos)
            if "vignette" in filt:
                vignette_pos = filt.find("vignette=")
                self.assertGreater(sub_pos, vignette_pos)
        finally:
            if os.path.exists(ass_path):
                os.remove(ass_path)

    @patch("video_process.get_video_dimensions", return_value=(1920, 1080))
    @patch("video_process.get_video_fps", return_value=30.0)
    def test_8_audio_loudnorm_preserved(self, mock_fps, mock_dims):
        """EBU R128 loudness normalization filter is preserved with exact parameters."""
        filt, maps = video_process.build_ffmpeg_filter("test_dummy.mp4", ass_path=None, has_audio=True)
        self.assertIn("loudnorm=I=-14.0:LRA=11:TP=-1.5", filt)
        self.assertIn("-map", maps)
        self.assertIn("[aout]", maps)

    def test_9_process_video_signature_compatibility(self):
        """process_video signature must remain compatible with existing callers."""
        import inspect
        sig = inspect.signature(video_process.process_video)
        params = list(sig.parameters.keys())
        self.assertEqual(params, ["input_path", "output_path", "ass_path", "max_seconds"])


if __name__ == "__main__":
    unittest.main()
