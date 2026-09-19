"""
Unit and regression tests for Centralized Telegram Run Report.
Verifies:
1. Status calculation: SUCCESS 🟢, PARTIAL FAILURE ⚠️, FAILED 🔴, NOTHING TO PROCESS 🔵
2. Full multi-clip (5 clips) and single-clip (1 clip) report formatting
3. Secret redaction for API keys and tokens in error messages
4. Telegram message length clamping under 4096 characters
5. Safe failover when Telegram notification raises an error
6. Guaranteed execution of run report in main() try...finally block
"""
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Ensure scripts directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

import run_report
from run_report import PipelineRunReport, sanitize_secrets, clean_concise_error
import main as pipeline_main


class TestPipelineRunReport(unittest.TestCase):
    """Test suite for PipelineRunReport data tracking and message generation."""

    def setUp(self):
        run_report.set_current_report(None)

    def tearDown(self):
        run_report.set_current_report(None)

    def test_secret_sanitization(self):
        """Ensures secrets (Groq, OpenAI, Google OAuth, Bearer tokens, Telegram bot tokens) are redacted."""
        raw_error = (
            "API error with key gsk_abc123XYZ456 and sk-proj-1234567890abcdef. "
            "OAuth token ya29.a0AfH6SMD_random_chars and Bearer secret-auth-token-123. "
            "Telegram bot bot123456789:ABCDefgh-token."
        )
        sanitized = sanitize_secrets(raw_error)
        self.assertNotIn("gsk_abc123XYZ456", sanitized)
        self.assertIn("[REDACTED_GROQ_KEY]", sanitized)
        self.assertNotIn("sk-proj-1234567890abcdef", sanitized)
        self.assertIn("[REDACTED_OPENAI_KEY]", sanitized)
        self.assertNotIn("ya29.a0AfH6SMD_random_chars", sanitized)
        self.assertIn("[REDACTED_OAUTH_TOKEN]", sanitized)
        self.assertNotIn("secret-auth-token-123", sanitized)
        self.assertIn("Bearer [REDACTED_TOKEN]", sanitized)
        self.assertNotIn("bot123456789:ABCDefgh-token", sanitized)
        self.assertIn("bot[REDACTED_BOT_TOKEN]", sanitized)

    def test_clean_concise_error(self):
        """Verifies multiline tracebacks are reduced to concise, sanitized one-liners."""
        traceback_str = (
            "Traceback (most recent call last):\n"
            "  File 'main.py', line 100, in run\n"
            "  File 'api.py', line 50, in call\n"
            "RuntimeError: Failed to connect with gsk_secretkey123"
        )
        concise = clean_concise_error(traceback_str)
        self.assertNotIn("Traceback", concise)
        self.assertNotIn("gsk_secretkey123", concise)
        self.assertIn("RuntimeError: Failed to connect with [REDACTED_GROQ_KEY]", concise)

    def test_status_calculation_success_five_clips(self):
        """Verifies 5-clip run with 5 uploads results in SUCCESS 🟢."""
        report = PipelineRunReport(trigger="schedule")
        report.set_source("test_podcast.mp4", duration=1200.0, destination="Moved to Processed")
        report.set_discovery(
            transcription_status="✅ Completed (Whisper)",
            candidate_count=20,
            qualified_count=5,
            filtering_results="5 clips met quality threshold",
        )
        for i in range(1, 6):
            report.add_or_update_clip(
                i,
                quality_score=8.5,
                hook=f"Amazing Hook #{i}",
                upload_status="Uploaded",
                youtube_url=f"https://youtube.com/shorts/test_{i}",
            )
        report.summary_total_clips = 5
        report.summary_uploaded = 5
        report.summary_failed = 0
        report.summary_completed = 5
        report.summary_source_status = "Moved to Processed"

        msg = report.build_telegram_message()
        self.assertEqual(report.overall_status, "SUCCESS 🟢")
        self.assertIn("Status: *SUCCESS 🟢*", msg)
        self.assertIn("• Video: `test_podcast.mp4`", msg)
        self.assertIn("• Destination: `Moved to Processed`", msg)
        self.assertIn("• Candidates: `20` | Qualified: `5`", msg)
        self.assertIn("Clip #1 (Score: 8.5): ✅ Uploaded", msg)
        self.assertIn("Clip #5 (Score: 8.5): ✅ Uploaded", msg)
        self.assertIn("• Clips Total: `5` | Uploaded: `5` | Failed: `0` | Pending: `0`", msg)

    def test_status_calculation_success_single_clip(self):
        """Verifies 1-clip queue execution results in SUCCESS 🟢."""
        report = PipelineRunReport(trigger="workflow_dispatch")
        report.set_source("interview.mp4", duration=300.0, destination="Preserved in Incoming")
        report.add_or_update_clip(
            1,
            quality_score=9.1,
            hook="Single punchline",
            upload_status="Uploaded",
            youtube_url="https://youtube.com/shorts/single_clip",
        )
        report.summary_total_clips = 1
        report.summary_uploaded = 1
        report.summary_failed = 0
        report.summary_completed = 1
        report.summary_source_status = "Preserved in Incoming"

        msg = report.build_telegram_message()
        self.assertEqual(report.overall_status, "SUCCESS 🟢")
        self.assertIn("Status: *SUCCESS 🟢*", msg)
        self.assertIn("Clip #1 (Score: 9.1): ✅ Uploaded", msg)
        self.assertIn("• Clips Total: `1` | Uploaded: `1` | Failed: `0` | Pending: `0`", msg)

    def test_status_calculation_partial_failure(self):
        """Verifies mixed upload and failure results in PARTIAL FAILURE ⚠️."""
        report = PipelineRunReport()
        report.set_source("partially_failing_video.mp4", destination="Preserved in Incoming")
        report.add_or_update_clip(1, quality_score=7.8, upload_status="Uploaded", youtube_url="https://yt.com/1")
        report.add_or_update_clip(2, quality_score=6.5, upload_status="Failed", error="FFmpeg encoding error")
        report.add_error("rendering", "FFmpegError", "FFmpeg exited with code 1", affected="partially_failing_video.mp4")

        report.summary_total_clips = 3
        report.summary_uploaded = 1
        report.summary_failed = 1
        report.summary_pending = 1
        report.summary_source_status = "Preserved in Incoming"

        msg = report.build_telegram_message()
        self.assertEqual(report.overall_status, "PARTIAL FAILURE ⚠️")
        self.assertIn("Status: *PARTIAL FAILURE ⚠️*", msg)
        self.assertIn("⚠️ *Error Report (1):*", msg)
        self.assertIn("• Clips Total: `3` | Uploaded: `1` | Failed: `1` | Pending: `1`", msg)

    def test_status_calculation_failed_discovery(self):
        """Verifies discovery failure (e.g. missing audio) results in FAILED 🔴."""
        report = PipelineRunReport()
        report.set_source("no_audio_video.mp4")
        report.update_stage("download", "✅ Completed")
        report.update_stage("transcription", "❌ No audio stream")
        report.add_error("audio_check", "ValueError", "Source video has no audio stream and cannot be transcribed.")

        msg = report.build_telegram_message()
        self.assertEqual(report.overall_status, "FAILED 🔴")
        self.assertIn("Status: *FAILED 🔴*", msg)
        self.assertIn("• Transcription: ❌ No audio stream", msg)
        self.assertIn("`[audio_check]` ValueError: Source video has no audio stream", msg)

    def test_status_calculation_nothing_to_process(self):
        """Verifies nothing to process produces NOTHING TO PROCESS 🔵."""
        report = PipelineRunReport()
        report.set_overall_status("NOTHING TO PROCESS 🔵")
        report.summary_source_status = "All Incoming videos already enqueued"

        msg = report.build_telegram_message()
        self.assertEqual(report.overall_status, "NOTHING TO PROCESS 🔵")
        self.assertIn("Status: *NOTHING TO PROCESS 🔵*", msg)
        self.assertIn("• Source Status: *All Incoming videos already enqueued*", msg)

    def test_message_clamping_telegram_limit(self):
        """Ensures messages over 4000 chars are cleanly clamped below 4096 characters."""
        report = PipelineRunReport()
        report.set_source("very_long_test.mp4")
        # Add 50 clips with long hooks to trigger length cap
        for i in range(50):
            report.add_or_update_clip(
                i,
                quality_score=8.0,
                hook=f"Very long hook description for clip number {i} " * 5,
                upload_status="Uploaded",
                youtube_url=f"https://youtube.com/shorts/long_url_{i}",
            )
        report.summary_total_clips = 50
        report.summary_uploaded = 50

        msg = report.build_telegram_message()
        self.assertLessEqual(len(msg), 4096)
        self.assertIn("... [Report truncated to fit Telegram limit]", msg)

    @patch("notify.send")
    def test_send_report_safe_failover(self, mock_notify_send):
        """Verifies that an exception during notify.send() does not raise or crash the pipeline."""
        mock_notify_send.side_effect = RuntimeError("Telegram connection reset / network error")
        report = PipelineRunReport()
        # Must not raise an exception
        try:
            report.send_report()
        except Exception as e:
            self.fail(f"send_report() raised an unhandled exception: {e}")

    @patch("run_report.PipelineRunReport.send_report")
    @patch("drive_utils.get_drive_service")
    def test_main_finally_always_sends_report_on_error(self, mock_get_drive, mock_send_report):
        """Verifies that pipeline_main.main() always triggers report.send_report() even on unhandled errors."""
        mock_get_drive.side_effect = RuntimeError("Fatal Drive initialization failure")

        with self.assertRaises(RuntimeError):
            pipeline_main.main()

        # The run report MUST be dispatched in finally
        mock_send_report.assert_called_once()


if __name__ == "__main__":
    unittest.main()
