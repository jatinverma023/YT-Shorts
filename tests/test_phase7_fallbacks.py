import unittest
from unittest.mock import MagicMock, patch
import os
import sys

# Ensure scripts dir is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

import config
import hook_generator
import metadata_ai
import clip_detection
from clip_detection import ClipDiscoveryError
from run_report import PipelineRunReport
import main as pipeline_main


class TestPhase7Fallbacks(unittest.TestCase):
    def setUp(self):
        import ai_rate_limiter
        ai_rate_limiter.shared_rate_limiter._history.clear()

    def test_hook_generation_fallback_reporting(self):
        """Content fallback (hook AI failure) returns deterministic grounded hook with is_fallback=True."""
        transcript = "Artificial intelligence is changing software development forever. The speed of iteration is 10 times faster."
        
        with patch.object(config, "GROQ_API_KEY", ""), \
             patch.object(config, "OPENAI_API_KEY", ""), \
             patch.dict(os.environ, {"GROQ_API_KEY": "", "OPENAI_API_KEY": ""}):
            res = hook_generator.generate_short_hook(
                transcript=transcript,
                detected_lang="en",
            )
            self.assertTrue(res.get("is_fallback"))
            self.assertIn("fallback_reason", res)
            self.assertTrue(len(res["selected_hook"]) > 0)
            self.assertEqual(res["source"], "fallback")

    def test_metadata_fallback_reporting(self):
        """Metadata AI failure returns deterministic fallback with is_fallback=True and fallback_reason."""
        transcript = "Artificial intelligence is changing software development forever."
        with patch.object(config, "GROQ_API_KEY", ""), \
             patch.object(config, "OPENAI_API_KEY", ""), \
             patch.dict(os.environ, {"GROQ_API_KEY": "", "OPENAI_API_KEY": ""}):
            res = metadata_ai.generate_shorts_metadata(
                filename="AI Future Part 1.mp4",
                transcript=transcript,
            )
            self.assertTrue(res.get("is_fallback"))
            self.assertIn("fallback_reason", res)
            self.assertEqual(res["fallback_reason"], "No AI API key configured")
            self.assertIn("#Shorts", res["hashtags"])

    @patch("notify.send")
    @patch("sheet_log.log_run")
    @patch("sheet_log.enqueue_clips")
    @patch("sheet_log.enqueue_zero_clips")
    @patch("clip_detection.detect_clips_from_transcript", side_effect=ClipDiscoveryError("All chunks failed"))
    @patch("transcribe.transcribe_audio", return_value={"segments": [{"start": 0.0, "end": 60.0}], "words": [], "text": "text"})
    @patch("transcribe.extract_audio")
    @patch("video_process.check_has_audio", return_value=True)
    @patch("video_process.get_duration_seconds", return_value=120.0)
    @patch("drive_utils.download_file")
    @patch("main.check_and_move_if_completed")
    def test_discovery_failure_safety_in_pipeline(
        self, mock_check_move, mock_download, mock_dur, mock_audio,
        mock_extract, mock_transcribe, mock_detect, mock_enqueue_zero,
        mock_enqueue_clips, mock_log_run, mock_notify
    ):
        """Discovery failure must raise ClipDiscoveryError, leave in Incoming, and set report to FAILED."""
        report = PipelineRunReport()
        mock_drive = MagicMock()
        file_info = {"id": "vid_fail", "name": "failed_discovery_vid.mp4"}

        with patch("main.run_report.get_current_report", return_value=report):
            stats = pipeline_main.discover_and_enqueue_video(mock_drive, file_info)

            # Must not move file to Processed
            mock_drive.move_file.assert_not_called()
            # Must not enqueue fallback clips
            mock_enqueue_clips.assert_not_called()
            mock_enqueue_zero.assert_not_called()
            # Report must reflect discovery failure
            self.assertEqual(report.discovery_status, "FAILED")
            self.assertEqual(report.summary_source_status, "Preserved in Incoming (AI discovery failure)")
            self.assertEqual(report.summary_total_clips, 0)
            self.assertEqual(report.determine_overall_status(), "FAILED 🔴")

    def test_run_report_status_differentiation(self):
        """Run report must distinguish SUCCESS, DEGRADED, and FAILED states."""
        # 1. Clean success
        rep_success = PipelineRunReport()
        rep_success.summary_total_clips = 1
        rep_success.summary_uploaded = 1
        self.assertEqual(rep_success.determine_overall_status(), "SUCCESS 🟢")

        # 2. Degraded (upload succeeded but content fallback used)
        rep_degraded = PipelineRunReport()
        rep_degraded.summary_total_clips = 1
        rep_degraded.summary_uploaded = 1
        rep_degraded.record_fallback("metadata", "Groq rate limit exceeded")
        self.assertTrue(rep_degraded.is_fallback)
        self.assertEqual(rep_degraded.fallback_count, 1)
        self.assertEqual(rep_degraded.determine_overall_status(), "DEGRADED ⚠️")

        # Telegram message includes fallback section
        tg_msg = rep_degraded.build_telegram_message()
        self.assertIn("DEGRADED ⚠️", tg_msg)
        self.assertIn("Fallbacks Used", tg_msg)
        self.assertIn("metadata: Groq rate limit exceeded", tg_msg)

        # 3. Failed (fatal error or discovery failure)
        rep_failed = PipelineRunReport()
        rep_failed.set_discovery_status("FAILED")
        self.assertEqual(rep_failed.determine_overall_status(), "FAILED 🔴")


if __name__ == "__main__":
    unittest.main()
