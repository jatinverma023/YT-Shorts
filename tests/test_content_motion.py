"""
Unit tests for Visual Upgrade #1B: Content-Aware Dynamic Emphasis.

Verifies:
1. Valid emphasis point accepted
2. Invalid timestamps rejected
3. Strength clamped to 0–1
4. Maximum emphasis points enforced (<= 4)
5. Overlapping points handled (higher strength retained)
6. Minimum separation enforced (>= 1.5s)
7. Maximum zoom never exceeded (<= 1.04x)
8. Minimum zoom respected (>= 1.00x)
9. Smooth motion curve generated (continuous)
10. Motion disabled produces no content-motion filter / true no-motion path
11. Empty transcript fallback to #1A.1 sinusoidal motion
12. Horizontal composition compatibility (foreground only)
13. Portrait composition compatibility
14. Captions remain after motion (stationary)
15. Top hook remains after motion (stationary)
16. Clip duration unchanged
17. TARGET_FPS behavior unchanged
18. Timestamp preservation across 25 FPS and 30 FPS
19. Keyword presence alone never creates an emphasis point
20. Single motion curve guarantee (never stacked)
"""
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Ensure scripts directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

import config
import content_motion
import video_process


class TestContentMotion(unittest.TestCase):
    """Test suite for Visual Upgrade #1B: Content-Aware Dynamic Emphasis."""

    def test_1_valid_emphasis_point_accepted(self):
        """Valid emphasis point within bounds is accepted."""
        pt = {
            "start": 2.5,
            "end": 4.5,
            "strength": 0.85,
            "type": "key_point",
        }
        is_val, reason = content_motion.validate_emphasis_point(pt, clip_duration=10.0)
        self.assertTrue(is_val, f"Expected valid, got: {reason}")
        self.assertEqual(pt["strength"], 0.85)
        self.assertEqual(pt["type"], "key_point")

    def test_2_invalid_timestamps_rejected(self):
        """Invalid, out-of-order, or out-of-bounds timestamps are strictly rejected."""
        clip_dur = 15.0

        # Negative start
        self.assertFalse(content_motion.validate_emphasis_point({"start": -1.0, "end": 3.0}, clip_dur)[0])
        # End <= start
        self.assertFalse(content_motion.validate_emphasis_point({"start": 5.0, "end": 5.0}, clip_dur)[0])
        self.assertFalse(content_motion.validate_emphasis_point({"start": 6.0, "end": 4.0}, clip_dur)[0])
        # Exceeds clip duration
        self.assertFalse(content_motion.validate_emphasis_point({"start": 12.0, "end": 16.0}, clip_dur)[0])
        self.assertFalse(content_motion.validate_emphasis_point({"start": 16.0, "end": 18.0}, clip_dur)[0])
        # Duration too short (< 0.6s)
        self.assertFalse(content_motion.validate_emphasis_point({"start": 2.0, "end": 2.4}, clip_dur)[0])
        # Duration too long (> 5.5s)
        self.assertFalse(content_motion.validate_emphasis_point({"start": 2.0, "end": 8.0}, clip_dur)[0])
        # Non-numeric
        self.assertFalse(content_motion.validate_emphasis_point({"start": "abc", "end": 3.0}, clip_dur)[0])

    def test_3_strength_clamped_to_0_1(self):
        """Strength is deterministically clamped to [0.0, 1.0]."""
        p_low = {"start": 2.0, "end": 4.0, "strength": -0.8}
        content_motion.validate_emphasis_point(p_low, 10.0)
        self.assertEqual(p_low["strength"], 0.0)

        p_high = {"start": 2.0, "end": 4.0, "strength": 2.5}
        content_motion.validate_emphasis_point(p_high, 10.0)
        self.assertEqual(p_high["strength"], 1.0)

    def test_4_maximum_emphasis_points_enforced(self):
        """Maximum points capped to max_points (hard max 4)."""
        pts = [
            {"start": 1.0, "end": 2.0, "strength": 0.5},
            {"start": 4.0, "end": 5.0, "strength": 0.9},
            {"start": 7.0, "end": 8.0, "strength": 0.8},
            {"start": 10.0, "end": 11.0, "strength": 0.7},
            {"start": 13.0, "end": 14.0, "strength": 0.95},
        ]
        res = content_motion.filter_and_deduplicate_emphasis_points(pts, clip_duration=16.0, max_points=3)
        self.assertLessEqual(len(res), 3)

        # With max_points=10, hard maximum 4 must be enforced
        res4 = content_motion.filter_and_deduplicate_emphasis_points(pts, clip_duration=16.0, max_points=10)
        self.assertLessEqual(len(res4), 4)

    def test_5_overlapping_points_handled(self):
        """Overlapping regions keep the point with higher strength."""
        pts = [
            {"start": 2.0, "end": 4.5, "strength": 0.60},
            {"start": 3.5, "end": 5.5, "strength": 0.90},  # overlaps with first, stronger
        ]
        res = content_motion.filter_and_deduplicate_emphasis_points(pts, clip_duration=10.0)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["strength"], 0.90)
        self.assertEqual(res[0]["start"], 3.5)

    def test_6_minimum_separation_enforced(self):
        """Points closer than min_separation (1.5s) are resolved to the stronger point."""
        pts = [
            {"start": 2.0, "end": 4.0, "strength": 0.95},
            {"start": 4.8, "end": 6.5, "strength": 0.70},  # gap is 0.8s < 1.5s, weaker
        ]
        res = content_motion.filter_and_deduplicate_emphasis_points(pts, clip_duration=10.0, min_separation=1.5)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["strength"], 0.95)

    def test_7_maximum_zoom_never_exceeded(self):
        """Zoom factor never exceeds configured maximum (<= 1.04x)."""
        pts = [
            {"start": 3.0, "end": 6.0, "strength": 1.0},
        ]
        clip_dur = 10.0
        # Sample every 0.05 seconds throughout the entire clip
        t = 0.0
        while t <= clip_dur:
            z = content_motion.evaluate_zoom_at_time(t, pts, clip_dur, max_zoom=1.04)
            self.assertLessEqual(z, 1.04001, f"Exceeded max zoom at t={t}: {z}")
            t += 0.05

    def test_8_minimum_zoom_respected(self):
        """Zoom factor is never below 1.00x at any point."""
        pts = [
            {"start": 3.0, "end": 6.0, "strength": 1.0},
        ]
        clip_dur = 10.0
        t = 0.0
        while t <= clip_dur:
            z = content_motion.evaluate_zoom_at_time(t, pts, clip_dur, max_zoom=1.04)
            self.assertGreaterEqual(z, 1.00000, f"Below min zoom at t={t}: {z}")
            t += 0.05

    def test_9_smooth_motion_curve_generated(self):
        """Zoom curve smoothly eases in, holds peak, and eases out with no abrupt jumps."""
        pts = [
            {"start": 4.0, "end": 6.0, "strength": 1.0},
        ]
        clip_dur = 10.0
        # t0 is max(0, 4.0 - 0.35) = 3.65s
        # Before ease-in (t=3.0s) -> 1.00
        self.assertEqual(content_motion.evaluate_zoom_at_time(3.0, pts, clip_dur), 1.00)

        # Mid ease-in (t=3.825s) -> between 1.00 and 1.04
        z_ease_in = content_motion.evaluate_zoom_at_time(3.825, pts, clip_dur)
        self.assertTrue(1.00 < z_ease_in < 1.04, f"Unexpected ease-in zoom: {z_ease_in}")

        # Peak (t=5.0s) -> 1.04
        self.assertEqual(content_motion.evaluate_zoom_at_time(5.0, pts, clip_dur), 1.04)

        # Mid ease-out (t=6.225s) -> between 1.00 and 1.04
        z_ease_out = content_motion.evaluate_zoom_at_time(6.225, pts, clip_dur)
        self.assertTrue(1.00 < z_ease_out < 1.04, f"Unexpected ease-out zoom: {z_ease_out}")

        # After ease-out (t=7.0s) -> 1.00
        self.assertEqual(content_motion.evaluate_zoom_at_time(7.0, pts, clip_dur), 1.00)

    @patch("video_process.get_video_dimensions", return_value=(1920, 1080))
    @patch("video_process.get_duration_seconds", return_value=12.0)
    @patch("video_process.get_video_fps", return_value=30.0)
    def test_10_motion_disabled_produces_no_content_motion_filter(self, mock_fps, mock_dur, mock_dims):
        """When CONTENT_MOTION_ENABLED is false, falls back to #1A.1 sinusoidal motion.
        When VISUAL_MOTION_ENABLED is false, produces a true no-motion path."""
        pts = [{"start": 3.0, "end": 6.0, "strength": 1.0}]

        # 1. Content motion disabled -> falls back to sinusoidal motion
        with patch.object(video_process, "CONTENT_MOTION_ENABLED", False):
            filt, _ = video_process.build_ffmpeg_filter("test.mp4", emphasis_points=pts)
            self.assertIn("cos(2*PI*in/", filt)  # Sinusoidal micro-motion formula

        # 2. Visual motion disabled entirely -> no motion filter in graph
        with patch.object(video_process, "VISUAL_MOTION_ENABLED", False):
            filt, _ = video_process.build_ffmpeg_filter("test.mp4", emphasis_points=pts)
            self.assertNotIn("perspective=", filt)

    @patch("video_process.get_video_dimensions", return_value=(1920, 1080))
    @patch("video_process.get_duration_seconds", return_value=12.0)
    @patch("video_process.get_video_fps", return_value=30.0)
    def test_11_empty_transcript_fallback_to_sinusoidal_motion(self, mock_fps, mock_dur, mock_dims):
        """When transcript is empty or no emphasis points are found, falls back to #1A.1 sinusoidal curve."""
        filt, _ = video_process.build_ffmpeg_filter("test.mp4", words=[], emphasis_points=[])
        self.assertIn("cos(2*PI*in/", filt)
        self.assertIn("perspective=", filt)

    @patch("video_process.get_video_dimensions", return_value=(1920, 1080))
    @patch("video_process.get_duration_seconds", return_value=12.0)
    @patch("video_process.get_video_fps", return_value=30.0)
    def test_12_horizontal_composition_compatibility(self, mock_fps, mock_dur, mock_dims):
        """Horizontal composition applies content-aware motion strictly to foreground; background is stationary."""
        pts = [{"start": 3.0, "end": 6.0, "strength": 1.0}]
        filt, _ = video_process.build_ffmpeg_filter("test.mp4", emphasis_points=pts)

        # Background chain must have blur and dimming, NO perspective filter
        bg_part = filt.split(";")[0]
        self.assertIn("boxblur=", bg_part)
        self.assertNotIn("perspective=", bg_part)

        # Foreground chain must contain the perspective motion filter
        fg_part = filt.split(";")[1]
        self.assertIn("perspective=", fg_part)

    @patch("video_process.get_video_dimensions", return_value=(1080, 1920))
    @patch("video_process.get_duration_seconds", return_value=12.0)
    @patch("video_process.get_video_fps", return_value=30.0)
    def test_13_portrait_composition_compatibility(self, mock_fps, mock_dur, mock_dims):
        """Portrait composition applies content-aware motion before color enhance and subtitles."""
        pts = [{"start": 3.0, "end": 6.0, "strength": 1.0}]
        filt, _ = video_process.build_ffmpeg_filter("test.mp4", emphasis_points=pts)
        self.assertIn("perspective=", filt)
        self.assertIn("scale=1080:1920", filt)

    @patch("video_process.get_video_dimensions", return_value=(1920, 1080))
    @patch("video_process.get_duration_seconds", return_value=12.0)
    @patch("video_process.get_video_fps", return_value=30.0)
    @patch("os.path.isfile", return_value=True)
    def test_14_captions_remain_after_motion(self, mock_isfile, mock_fps, mock_dur, mock_dims):
        """Subtitles filter is applied strictly AFTER motion filter in the final composition."""
        pts = [{"start": 3.0, "end": 6.0, "strength": 1.0}]
        filt, _ = video_process.build_ffmpeg_filter("test.mp4", ass_path="/tmp/test.ass", emphasis_points=pts)
        motion_idx = filt.index("perspective=")
        sub_idx = filt.index("subtitles=")
        self.assertLess(motion_idx, sub_idx, "Motion must be evaluated before subtitle burning")

    @patch("video_process.get_video_dimensions", return_value=(1920, 1080))
    @patch("video_process.get_duration_seconds", return_value=12.0)
    @patch("video_process.get_video_fps", return_value=30.0)
    def test_15_top_hook_remains_after_motion(self, mock_fps, mock_dur, mock_dims):
        """HeaderPunchline events in ASS captions are preserved and burned onto the stationary final canvas."""
        ass_content = (
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
            "Dialogue: 1,0:00:00.00,0:00:10.00,HeaderPunchline,,0,0,0,,Top Hook Text\n"
            "Dialogue: 0,0:00:02.00,0:00:05.00,Default,,0,0,0,,Spoken subtitle\n"
        )
        with patch("builtins.open", unittest.mock.mock_open(read_data=ass_content)):
            with patch("os.path.isfile", return_value=True):
                words = video_process.extract_words_from_ass("/tmp/dummy.ass")
                # HeaderPunchline must be ignored from words transcript
                self.assertEqual(len(words), 1)
                self.assertEqual(words[0]["word"], "Spoken subtitle")

    @patch("video_process.get_video_dimensions", return_value=(1920, 1080))
    @patch("video_process.get_duration_seconds", return_value=12.5)
    @patch("video_process.get_video_fps", return_value=30.0)
    def test_16_clip_duration_unchanged(self, mock_fps, mock_dur, mock_dims):
        """Content-aware motion filter preserves clip duration without trimming or time-dilation."""
        pts = [{"start": 2.0, "end": 5.0, "strength": 0.8}]
        filt, _ = video_process.build_ffmpeg_filter("test.mp4", emphasis_points=pts)
        # Perspective filter evaluates per-frame with linear interpolation and does not alter duration
        self.assertIn("eval=frame:interpolation=linear", filt)
        self.assertNotIn("setpts=", filt)
        self.assertNotIn("atempo=", filt)

    @patch("subprocess.run")
    @patch("video_process.check_has_audio", return_value=True)
    @patch("video_process.build_ffmpeg_filter", return_value=("null", ["-map", "0:v", "-map", "0:a"]))
    def test_17_target_fps_behavior_unchanged(self, mock_filt, mock_audio, mock_run):
        """TARGET_FPS behavior in process_video is unchanged."""
        with patch.object(video_process, "TARGET_FPS", 30):
            video_process.process_video("in.mp4", "out.mp4")
            cmd = mock_run.call_args[0][0]
            self.assertIn("-r", cmd)
            self.assertEqual(cmd[cmd.index("-r") + 1], "30")

    def test_18_timestamp_preservation_across_25_and_30_fps(self):
        """
        Critical Requirement 3 & 10:
        An emphasis point at 5.0 seconds occurs at the EXACT same 5.0-second position
        regardless of whether the stream is 25 FPS or 30 FPS.
        """
        pts = [{"start": 4.0, "end": 6.0, "strength": 1.0, "type": "key_point"}]
        clip_dur = 12.0

        fps_25 = 25.0
        fps_30 = 30.0

        # At timestamp t = 5.0s:
        # Frame index at 25 FPS = 5.0 * 25 = 125
        # Frame index at 30 FPS = 5.0 * 30 = 150
        frame_25_at_5 = int(round(5.0 * fps_25))
        frame_30_at_5 = int(round(5.0 * fps_30))

        t_25 = frame_25_at_5 / fps_25
        t_30 = frame_30_at_5 / fps_30

        self.assertEqual(t_25, 5.0)
        self.assertEqual(t_30, 5.0)

        # Verify zoom factor matches bit-for-bit at 5.0s
        z_25 = content_motion.evaluate_zoom_at_time(t_25, pts, clip_dur)
        z_30 = content_motion.evaluate_zoom_at_time(t_30, pts, clip_dur)
        self.assertEqual(z_25, z_30)
        self.assertEqual(z_25, 1.04)

        # Verify ease-in timestamp at t = 3.8s:
        # Frame index at 25 FPS = 3.8 * 25 = 95
        # Frame index at 30 FPS = 3.8 * 30 = 114
        frame_25_at_38 = int(round(3.8 * fps_25))
        frame_30_at_38 = int(round(3.8 * fps_30))

        t_25_38 = frame_25_at_38 / fps_25
        t_30_38 = frame_30_at_38 / fps_30
        self.assertAlmostEqual(t_25_38, 3.8, places=4)
        self.assertAlmostEqual(t_30_38, 3.8, places=4)

        z_25_38 = content_motion.evaluate_zoom_at_time(t_25_38, pts, clip_dur)
        z_30_38 = content_motion.evaluate_zoom_at_time(t_30_38, pts, clip_dur)
        self.assertEqual(z_25_38, z_30_38)

        # Verify generated FFmpeg expressions use (in/25.000) and (in/30.000)
        expr_25 = content_motion.build_content_motion_curve(pts, fps=25.0, clip_duration=clip_dur)
        expr_30 = content_motion.build_content_motion_curve(pts, fps=30.0, clip_duration=clip_dur)
        self.assertIn("(in/25.000)", expr_25)
        self.assertIn("(in/30.000)", expr_30)

    def test_19_keyword_presence_alone_never_creates_emphasis_point(self):
        """
        Critical Requirement 1:
        Keyword presence alone must NEVER create an emphasis point.
        An isolated 'but' or short filler without substantive proposition is rejected.
        """
        # Isolated single contrast word
        words_isolated = [
            {"word": "but", "start": 3.0, "end": 3.3},
        ]
        pts = content_motion.detect_emphasis_points(words_isolated, clip_duration=10.0)
        self.assertEqual(len(pts), 0, "Isolated keyword must NEVER create an emphasis point")

        # Contrast word with only 1 or 2 subsequent words (incomplete clause)
        words_fragment = [
            {"word": "par", "start": 2.5, "end": 2.8},
            {"word": "kyun", "start": 2.9, "end": 3.2},
        ]
        pts_frag = content_motion.detect_emphasis_points(words_fragment, clip_duration=10.0)
        self.assertEqual(len(pts_frag), 0, "Incomplete fragment after contrast marker must be rejected")

        # Substantive sentence with contrast marker and valid context
        words_substantive = [
            {"word": "humne", "start": 1.0, "end": 1.4},
            {"word": "koshish", "start": 1.5, "end": 1.9},
            {"word": "ki", "start": 2.0, "end": 2.3},
            {"word": "lekin", "start": 2.8, "end": 3.1},
            {"word": "asli", "start": 3.2, "end": 3.5},
            {"word": "raaz", "start": 3.6, "end": 3.9},
            {"word": "kisi", "start": 4.0, "end": 4.3},
            {"word": "ko", "start": 4.4, "end": 4.6},
            {"word": "nahi", "start": 4.7, "end": 5.0},
            {"word": "pata", "start": 5.1, "end": 5.4},
            {"word": "tha", "start": 5.5, "end": 5.8},
        ]
        pts_sub = content_motion.detect_emphasis_points(words_substantive, clip_duration=10.0)
        self.assertGreaterEqual(len(pts_sub), 1)

    @patch("video_process.get_video_dimensions", return_value=(1920, 1080))
    @patch("video_process.get_duration_seconds", return_value=12.0)
    @patch("video_process.get_video_fps", return_value=30.0)
    def test_20_single_motion_curve_guarantee_never_stacked(self, mock_fps, mock_dur, mock_dims):
        """
        Critical Requirement 4:
        Do not stack motion. Exactly ONE final foreground zoom curve.
        If content-aware emphasis exists, it replaces the sinusoidal curve.
        """
        pts = [{"start": 3.0, "end": 5.0, "strength": 0.85, "type": "key_point"}]
        filt, _ = video_process.build_ffmpeg_filter("test.mp4", emphasis_points=pts)

        # Must have exactly one perspective filter in the entire graph
        self.assertEqual(filt.count("perspective="), 1, "There must be exactly ONE motion filter in the graph")

        # Sinusoidal micro-motion formula (cos(2*PI*in/)) must NOT be present
        self.assertNotIn("cos(2*PI*in/", filt, "Sinusoidal curve must be replaced, NEVER stacked")


if __name__ == "__main__":
    unittest.main()
