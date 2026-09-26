"""
Comprehensive tests for production rendering and audio preservation audit:
1. Original source audio stream preservation & explicit mapping (0:a:0).
2. Final output resolution = 1080x1920 (no 720x1280 regression).
3. 9:16 aspect ratio & square pixel enforcement (SAR 1:1, DAR 9:16).
4. Background fills complete 1080x1920 canvas with moderate dimming (no black crushing).
5. No accidental audio replacement, dubbing, or TTS synthesis.
6. Real FFmpeg render verification with ffprobe stream inspection.
"""
import inspect
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

import config
import video_process


class TestRenderingAndAudioPreservationAudit(unittest.TestCase):

    def setUp(self):
        # Ensure static_ffmpeg is active if available
        try:
            import static_ffmpeg
            static_ffmpeg.add_paths()
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # 1. AUDIO PRESERVATION & EXPLICIT MAPPING
    # -------------------------------------------------------------------------
    @patch("video_process.get_video_dimensions", return_value=(1920, 1080))
    @patch("video_process.get_video_fps", return_value=30.0)
    def test_1a_final_render_explicitly_maps_primary_audio_stream_with_loudnorm(self, mock_fps, mock_dims):
        """Final FFmpeg filter must explicitly map [0:a:0] through loudnorm to [aout], and maps must include [aout]."""
        filt, maps = video_process.build_ffmpeg_filter("source.mp4", ass_path=None, has_audio=True)
        self.assertIn("[0:a:0]loudnorm=", filt)
        self.assertIn("-map", maps)
        self.assertIn("[aout]", maps)
        self.assertIn("[vout]", maps)
        # Verify loudnorm does not target vague [0:a]
        self.assertNotIn(";[0:a]loudnorm", filt)

    @patch("video_process.get_video_dimensions", return_value=(1920, 1080))
    @patch("video_process.get_video_fps", return_value=30.0)
    def test_1b_final_render_explicitly_maps_0_a_0_when_loudnorm_disabled(self, mock_fps, mock_dims):
        """When loudness normalization is disabled, audio must be explicitly mapped as 0:a:0 (not ambiguous 0:a)."""
        with patch.object(video_process, "ENABLE_LOUDNORM", False):
            filt, maps = video_process.build_ffmpeg_filter("source.mp4", ass_path=None, has_audio=True)
            self.assertIn("-map", maps)
            self.assertIn("0:a:0", maps)
            self.assertNotIn("loudnorm", filt)

    @patch("subprocess.run")
    def test_1c_extract_clip_segment_explicitly_maps_source_streams(self, mock_run):
        """extract_clip_segment must explicitly map video 0:v:0 and primary audio 0:a:0? to avoid wrong track selection."""
        video_process.extract_clip_segment("source.mp4", 10.0, 25.0, "out_slice.mp4")
        cmd = mock_run.call_args[0][0]
        # Check explicit stream mapping
        self.assertIn("-map", cmd)
        indices = [i for i, x in enumerate(cmd) if x == "-map"]
        mapped_streams = [cmd[i + 1] for i in indices]
        self.assertIn("0:v:0", mapped_streams)
        self.assertIn("0:a:0?", mapped_streams)
        # Check audio re-encoding parameters preserve quality without replacing audio
        self.assertIn("-c:a", cmd)
        c_a_idx = cmd.index("-c:a")
        self.assertEqual(cmd[c_a_idx + 1], "aac")

    @patch("subprocess.run")
    @patch("video_process.get_duration_seconds", return_value=30.0)
    def test_1d_silence_trimming_synchronously_slices_0_v_0_and_0_a_0(self, mock_dur, mock_run):
        """trim_silences_from_words must slice exact source streams [0:v:0] and [0:a:0] synchronously."""
        # Create dummy words with a >1.2s dead-air gap between word 1 and 2
        words = [
            {"word": "Hello", "start": 1.0, "end": 2.0},
            {"word": "world", "start": 4.5, "end": 5.5},
        ]
        video_process.trim_silences_from_words("in.mp4", words, "tightened.mp4")
        cmd = mock_run.call_args[0][0]
        self.assertIn("-filter_complex", cmd)
        filt_idx = cmd.index("-filter_complex")
        filt_str = cmd[filt_idx + 1]
        self.assertIn("[0:v:0]trim=", filt_str)
        self.assertIn("[0:a:0]atrim=", filt_str)
        self.assertIn("concat=n=", filt_str)
        self.assertIn("[vcut][acut]", filt_str)
        # Check stream maps
        vcut_idx = cmd.index("[vcut]")
        acut_idx = cmd.index("[acut]")
        self.assertEqual(cmd[vcut_idx - 1], "-map")
        self.assertEqual(cmd[acut_idx - 1], "-map")

    # -------------------------------------------------------------------------
    # 2. OUTPUT RESOLUTION (1080x1920)
    # -------------------------------------------------------------------------
    @patch("video_process.get_video_dimensions", return_value=(1920, 1080))
    @patch("video_process.get_video_fps", return_value=30.0)
    def test_2a_horizontal_video_filter_targets_exact_1080x1920(self, mock_fps, mock_dims):
        """Horizontal input composition filter must produce exactly 1080x1920 canvas."""
        filt, maps = video_process.build_ffmpeg_filter("horizontal.mp4", ass_path=None, has_audio=True)
        self.assertIn("scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920", filt)
        self.assertIn("setsar=1", filt)
        self.assertIn("[vout]", maps)

    @patch("video_process.get_video_dimensions", return_value=(720, 1280))
    @patch("video_process.get_video_fps", return_value=30.0)
    def test_2b_portrait_720x1280_video_upscaled_to_exact_1080x1920(self, mock_fps, mock_dims):
        """Portrait input (even if 720x1280) must be scaled to exactly 1080x1920 with setsar=1."""
        filt, maps = video_process.build_ffmpeg_filter("portrait_720.mp4", ass_path=None, has_audio=True)
        self.assertIn("scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1", filt)
        self.assertIn("[vout]", maps)

    def test_2c_config_defaults_guarantee_1080x1920(self):
        """Configuration defaults must guarantee TARGET_WIDTH=1080 and TARGET_HEIGHT=1920."""
        self.assertEqual(config.TARGET_WIDTH, 1080)
        self.assertEqual(config.TARGET_HEIGHT, 1920)

    # -------------------------------------------------------------------------
    # 3. 9:16 ASPECT RATIO & SQUARE PIXELS (SAR 1:1, DAR 9:16)
    # -------------------------------------------------------------------------
    @patch("video_process.get_video_dimensions", return_value=(1920, 1080))
    @patch("video_process.get_video_fps", return_value=30.0)
    def test_3a_setsar_enforced_on_all_composite_branches(self, mock_fps, mock_dims):
        """setsar=1 must be enforced on background, foreground, and merged overlay to ensure 1:1 square pixels."""
        filt, _ = video_process.build_ffmpeg_filter("test.mp4", ass_path=None, has_audio=False)
        # Background branch has setsar=1
        bg_part = filt.split(";")[0]
        self.assertIn("crop=1080:1920,setsar=1", bg_part)
        # Foreground branch has setsar=1
        fg_part = filt.split(";")[1]
        self.assertIn("setsar=1", fg_part)
        # Merge branch has setsar=1
        merge_part = filt.split(";")[2]
        self.assertIn("setsar=1[vout]", merge_part)

    @patch("video_process.get_video_dimensions", return_value=(1920, 1080))
    @patch("video_process.get_video_fps", return_value=30.0)
    def test_3b_foreground_preserves_aspect_ratio_without_stretching(self, mock_fps, mock_dims):
        """Foreground scaling must use force_original_aspect_ratio=decrease, never stretching pixels."""
        filt, _ = video_process.build_ffmpeg_filter("test.mp4", ass_path=None, has_audio=False)
        self.assertIn("force_original_aspect_ratio=decrease", filt)
        self.assertIn("crop=min(iw\\,1080):ih", filt)

    # -------------------------------------------------------------------------
    # 4. BACKGROUND TREATMENT (NO BLACK CRUSHING, VISIBLE EXTENSION)
    # -------------------------------------------------------------------------
    @patch("video_process.get_video_dimensions", return_value=(1920, 1080))
    @patch("video_process.get_video_fps", return_value=30.0)
    def test_4a_background_brightness_moderate_dimming_not_aggressive_crush(self, mock_fps, mock_dims):
        """Background brightness must use moderate dimming (-0.02) rather than aggressive darkening (-0.08 or -0.22)."""
        filt, _ = video_process.build_ffmpeg_filter("test.mp4", ass_path=None, has_audio=False)
        self.assertIn(f"brightness={config.BG_BRIGHTNESS:.2f}", filt)
        # Confirm default is -0.02 (moderate dimming)
        self.assertEqual(config.BG_BRIGHTNESS, -0.02)
        # Confirm it is not -0.08 or lower
        self.assertGreater(config.BG_BRIGHTNESS, -0.08)

    @patch("video_process.get_video_dimensions", return_value=(1920, 1080))
    @patch("video_process.get_video_fps", return_value=30.0)
    def test_4b_background_saturation_preserves_vibrant_colors(self, mock_fps, mock_dims):
        """Background saturation must be >= 1.05 to retain rich source video colors."""
        filt, _ = video_process.build_ffmpeg_filter("test.mp4", ass_path=None, has_audio=False)
        self.assertIn(f"saturation={config.BG_SATURATION:.2f}", filt)
        self.assertGreaterEqual(config.BG_SATURATION, 1.05)

    # -------------------------------------------------------------------------
    # 5. NO ACCIDENTAL AUDIO REPLACEMENT / DUBBING
    # -------------------------------------------------------------------------
    def test_5a_pipeline_has_no_tts_or_dubbing_imports(self):
        """Verify video_process and main do not import any TTS, dubbing, or speech synthesis libraries."""
        disallowed = [
            "gtts", "pyttsx", "pyttsx3", "elevenlabs", "edge_tts",
            "coqui", "bark", "soundfile",
        ]
        for mod_name in sys.modules:
            for dis in disallowed:
                self.assertNotIn(dis, mod_name.lower(), f"Disallowed TTS/dubbing library loaded: {mod_name}")

        # Check source code of video_process for any TTS references
        vproc_src = inspect.getsource(video_process)
        self.assertNotIn("text_to_speech", vproc_src.lower())
        self.assertNotIn("generate_speech", vproc_src.lower())
        self.assertNotIn("tts", vproc_src.lower())

    # -------------------------------------------------------------------------
    # 6. REAL FFmpeg & ffprobe END-TO-END RENDER TEST
    # -------------------------------------------------------------------------
    def test_6_real_render_verification_with_ffprobe(self):
        """
        Executes a real FFmpeg render on a synthetic 16:9 test clip with stereo audio
        and verifies with ffprobe that:
        - Output resolution is exactly 1080x1920.
        - SAR is 1:1 and DAR is 9:16.
        - Audio stream is preserved as AAC with stereo layout.
        - Background luminance confirms visible cinematic extension without black crushing.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            input_video = os.path.join(tmpdir, "synth_input.mp4")
            output_video = os.path.join(tmpdir, "rendered_short.mp4")

            # 1. Create a 1920x1080 synthetic input with stereo audio (left=440Hz, right=880Hz)
            create_cmd = [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", "smptebars=size=1920x1080:rate=30",
                "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
                "-f", "lavfi", "-i", "sine=frequency=880:sample_rate=48000",
                "-filter_complex", "[1:a][2:a]amerge=inputs=2[a]",
                "-map", "0:v", "-map", "[a]",
                "-t", "1.5",
                "-c:v", "libx264", "-preset", "ultrafast",
                "-c:a", "aac", "-b:a", "128k",
                input_video,
            ]
            try:
                subprocess.run(create_cmd, check=True, capture_output=True)
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                self.skipTest(f"FFmpeg not executable in test environment: {e}")

            # 2. Render through video_process.process_video
            video_process.process_video(input_video, output_video)
            self.assertTrue(os.path.isfile(output_video))

            # 3. Probe with ffprobe
            probe_cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "stream=codec_type,codec_name,width,height,sample_aspect_ratio,display_aspect_ratio,channels,channel_layout,r_frame_rate",
                "-of", "json",
                output_video,
            ]
            res = subprocess.run(probe_cmd, check=True, capture_output=True, text=True)
            import json
            data = json.loads(res.stdout)
            streams = data.get("streams", [])

            # Video stream assertions
            v_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
            self.assertIsNotNone(v_stream, "Rendered output must have a video stream")
            self.assertEqual(v_stream.get("width"), 1080, "Width must be exactly 1080")
            self.assertEqual(v_stream.get("height"), 1920, "Height must be exactly 1920")
            self.assertEqual(v_stream.get("sample_aspect_ratio"), "1:1", "SAR must be square (1:1)")
            self.assertEqual(v_stream.get("display_aspect_ratio"), "9:16", "DAR must be 9:16")

            # Audio stream assertions
            a_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
            self.assertIsNotNone(a_stream, "Rendered output must have an audio stream")
            self.assertEqual(a_stream.get("codec_name"), "aac", "Audio codec must be aac")
            self.assertEqual(a_stream.get("channels"), 2, "Stereo channel count (2) must be preserved")
            self.assertEqual(a_stream.get("channel_layout"), "stereo", "Stereo layout must be preserved")

            # 4. Extract first frame and verify background is not crushed to black
            ppm_cmd = [
                "ffmpeg", "-y", "-i", output_video,
                "-vframes", "1", "-f", "image2pipe", "-vcodec", "ppm", "-"
            ]
            ppm_res = subprocess.run(ppm_cmd, capture_output=True, check=True)
            ppm_data = ppm_res.stdout
            idx = ppm_data.find(b"255\n") + 4
            pixels = ppm_data[idx:]

            # Sample top background (y=100, x=540)
            offset = (100 * 1080 + 540) * 3
            r, g, b = pixels[offset], pixels[offset + 1], pixels[offset + 2]
            luma = 0.299 * r + 0.587 * g + 0.114 * b
            # Background with moderate dimming should be brightly visible (>40 luma for smpte bars green/cyan/white)
            self.assertGreater(luma, 35.0, f"Background must retain visible colors/luminance, got luma={luma:.1f}")


if __name__ == "__main__":
    unittest.main()
