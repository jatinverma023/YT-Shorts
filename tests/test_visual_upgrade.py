"""
Unit tests for Visual Upgrade #1A and #1A.1:
Full-Canvas Composition Correction + Cinematic Color Treatment + Subtle Motion.
Verifies:
1. Horizontal background fills the complete 1080x1920 canvas with clear, recognizable imagery.
2. Horizontal foreground is proportionally scaled with FOREGROUND_SCALE.
3. Fallback when FOREGROUND_SCALE <= 1.0 (standard fit, no crop).
4. Background and foreground remain separate branches merged via centered overlay.
5. Portrait 9:16 source uses direct scale/crop without background overlay.
6. Foreground color treatment (unsharp, eq) remains present.
7. Vignette can be toggled on/off via configuration.
8. Micro-motion remains present when enabled and omitted when disabled (true no-motion path).
9. Motion zoom is strictly bounded by VISUAL_MOTION_MAX_ZOOM.
10. ASS subtitles and top punchline remain the final overlay.
11. Audio normalization (loudnorm) remains intact with exact parameters.
12. Function signature of process_video remains 100% backward-compatible.
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
    def test_1_horizontal_background_fills_1080x1920_and_visible(self, mock_fps, mock_dims):
        """Horizontal background fills the complete 1080x1920 canvas and preserves recognizable imagery."""
        filt, _ = video_process.build_ffmpeg_filter("test_dummy.mp4", ass_path=None, has_audio=True)
        self.assertIn("[bg]", filt)
        self.assertIn("scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920", filt)
        self.assertIn("boxblur=25:5", filt)
        # Background brightness is conservative (-0.08) so background is visibly recognizable, not black
        self.assertIn(f"brightness={config.BG_BRIGHTNESS:.2f}", filt)
        self.assertIn(f"saturation={config.BG_SATURATION:.2f}", filt)

    @patch("video_process.get_video_dimensions", return_value=(1920, 1080))
    @patch("video_process.get_video_fps", return_value=30.0)
    def test_2_horizontal_foreground_proportionally_scaled(self, mock_fps, mock_dims):
        """Horizontal foreground is scaled larger (FOREGROUND_SCALE > 1.0) with centered crop."""
        with patch.object(video_process, "FOREGROUND_SCALE", 1.18):
            filt, _ = video_process.build_ffmpeg_filter("test_dummy.mp4", ass_path=None, has_audio=True)
            self.assertIn("[fg]", filt)
            # 1080 * 1.18 = 1274
            self.assertIn("scale=1274:-2:force_original_aspect_ratio=decrease", filt)
            self.assertIn("crop=min(iw\\,1080):ih", filt)

    @patch("video_process.get_video_dimensions", return_value=(1920, 1080))
    @patch("video_process.get_video_fps", return_value=30.0)
    def test_3_horizontal_foreground_scale_fallback(self, mock_fps, mock_dims):
        """When FOREGROUND_SCALE <= 1.0, falls back to standard decrease fit without crop."""
        with patch.object(video_process, "FOREGROUND_SCALE", 1.0):
            filt, _ = video_process.build_ffmpeg_filter("test_dummy.mp4", ass_path=None, has_audio=True)
            self.assertIn("scale=1080:1920:force_original_aspect_ratio=decrease", filt)
            self.assertNotIn("crop=min(iw\\,1080):ih", filt)

    @patch("video_process.get_video_dimensions", return_value=(1920, 1080))
    @patch("video_process.get_video_fps", return_value=30.0)
    def test_4_background_and_foreground_separate_branches(self, mock_fps, mock_dims):
        """Background and foreground remain separate filter branches merged via centered overlay."""
        filt, _ = video_process.build_ffmpeg_filter("test_dummy.mp4", ass_path=None, has_audio=True)
        self.assertIn("[0:v]scale=1080:1920", filt)  # bg branch
        self.assertIn("[bg][fg]overlay=(W-w)/2:(H-h)/2", filt)  # merge branch

    @patch("video_process.get_video_dimensions", return_value=(1080, 1920))
    @patch("video_process.get_video_fps", return_value=30.0)
    def test_5_portrait_source_filter_graph(self, mock_fps, mock_dims):
        """Portrait source uses direct scale/crop architecture without background overlay."""
        filt, maps = video_process.build_ffmpeg_filter("test_dummy_portrait.mp4", ass_path=None, has_audio=True)
        self.assertNotIn("[bg]", filt)
        self.assertNotIn("boxblur", filt)
        self.assertIn("crop=1080:1920", filt)

    @patch("video_process.get_video_dimensions", return_value=(1920, 1080))
    @patch("video_process.get_video_fps", return_value=30.0)
    def test_6_color_treatment_filters_present(self, mock_fps, mock_dims):
        """Professional color treatment (unsharp, eq) is present on foreground."""
        filt, _ = video_process.build_ffmpeg_filter("test_dummy.mp4", ass_path=None, has_audio=False)
        self.assertIn(f"unsharp=5:5:{config.VISUAL_SHARPEN_AMOUNT:.2f}:5:5:0.0", filt)
        self.assertIn(f"eq=contrast={config.VISUAL_CONTRAST:.2f}", filt)
        self.assertIn(f"saturation={config.VISUAL_SATURATION:.2f}", filt)
        self.assertIn(f"brightness={config.VISUAL_BRIGHTNESS:.2f}", filt)

    @patch("video_process.get_video_dimensions", return_value=(1920, 1080))
    @patch("video_process.get_video_fps", return_value=30.0)
    def test_7_vignette_toggle(self, mock_fps, mock_dims):
        """Vignette can be enabled or disabled via configuration."""
        with patch.object(video_process, "VISUAL_VIGNETTE_ENABLED", True):
            filt_on, _ = video_process.build_ffmpeg_filter("test_dummy.mp4", ass_path=None, has_audio=False)
            self.assertIn("vignette=angle=PI/4:aspect=9/16", filt_on)

        with patch.object(video_process, "VISUAL_VIGNETTE_ENABLED", False):
            filt_off, _ = video_process.build_ffmpeg_filter("test_dummy.mp4", ass_path=None, has_audio=False)
            self.assertNotIn("vignette", filt_off)

    @patch("video_process.get_video_dimensions", return_value=(1920, 1080))
    @patch("video_process.get_video_fps", return_value=30.0)
    def test_8_motion_toggle_creates_true_no_motion_path(self, mock_fps, mock_dims):
        """When VISUAL_MOTION_ENABLED is false, motion filter is omitted entirely."""
        with patch.object(video_process, "VISUAL_MOTION_ENABLED", False):
            filt_off, _ = video_process.build_ffmpeg_filter("test_dummy.mp4", ass_path=None, has_audio=False)
            self.assertNotIn("perspective", filt_off)

        with patch.object(video_process, "VISUAL_MOTION_ENABLED", True):
            filt_on, _ = video_process.build_ffmpeg_filter("test_dummy.mp4", ass_path=None, has_audio=False)
            self.assertIn("perspective", filt_on)

    def test_9_motion_bounds_and_formula(self):
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
    def test_10_layer_order_subtitles_burned_after_grading(self, mock_fps, mock_dims):
        """Subtitles and punchline are burned LAST, after composition, color grading, and vignette."""
        with tempfile.NamedTemporaryFile(suffix=".ass", delete=False) as f:
            ass_path = f.name

        try:
            filt, _ = video_process.build_ffmpeg_filter("test_dummy.mp4", ass_path=ass_path, has_audio=False)
            self.assertIn("subtitles=", filt)
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
    def test_11_audio_loudnorm_preserved(self, mock_fps, mock_dims):
        """EBU R128 loudness normalization filter is preserved with exact parameters."""
        filt, maps = video_process.build_ffmpeg_filter("test_dummy.mp4", ass_path=None, has_audio=True)
        self.assertIn("loudnorm=I=-14.0:LRA=11:TP=-1.5", filt)
        self.assertIn("-map", maps)
        self.assertIn("[aout]", maps)

    def test_12_process_video_signature_compatibility(self):
        """process_video signature must remain compatible with existing callers."""
        import inspect
        sig = inspect.signature(video_process.process_video)
        params = list(sig.parameters.keys())
        self.assertEqual(params, ["input_path", "output_path", "ass_path", "max_seconds"])

    @patch("subprocess.run")
    @patch("video_process.check_has_audio", return_value=True)
    @patch("video_process.build_ffmpeg_filter", return_value=("null", ["-map", "0:v", "-map", "0:a"]))
    def test_13_target_fps_command_generation(self, mock_filt, mock_audio, mock_run):
        """TARGET_FPS=0 preserves native source FPS (-r omitted); TARGET_FPS=30 adds -r 30."""
        # 1. Default / 0: -r must be omitted to preserve source FPS
        with patch.object(video_process, "TARGET_FPS", 0):
            video_process.process_video("in.mp4", "out.mp4")
            cmd = mock_run.call_args[0][0]
            self.assertNotIn("-r", cmd)

        # 2. Explicit 30: -r 30 must be present
        with patch.object(video_process, "TARGET_FPS", 30):
            video_process.process_video("in.mp4", "out.mp4")
            cmd = mock_run.call_args[0][0]
            self.assertIn("-r", cmd)
            r_idx = cmd.index("-r")
            self.assertEqual(cmd[r_idx + 1], "30")


if __name__ == "__main__":
    unittest.main()
