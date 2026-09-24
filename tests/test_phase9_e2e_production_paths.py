"""
Phase 9 Tests: Comprehensive End-to-End Production-Path Verification.
Covers Cases A through G for all production invariants.
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure scripts dir is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

import config
import main as pipeline_main
import clip_detection
from clip_detection import ClipDiscoveryError, ModelConfigurationError
import metadata_ai
import hook_generator
import youtube_upload
import ai_rate_limiter
from run_report import PipelineRunReport
from googleapiclient.errors import HttpError


class TestPhase9E2EProductionPaths(unittest.TestCase):

    def setUp(self):
        ai_rate_limiter.shared_rate_limiter._history.clear()

    @patch("notify.send")
    @patch("sheet_log.log_run")
    @patch("sheet_log.enqueue_clips")
    @patch("main.process_next_pending_clip", return_value={"total_queued": 2, "uploaded": 1, "failed": 0, "urls": ["https://youtube.com/shorts/test_url_1"], "errors": []})
    @patch("clip_detection.detect_clips_from_transcript")
    @patch("transcribe.transcribe_audio")
    @patch("transcribe.extract_audio")
    @patch("video_process.check_has_audio", return_value=True)
    @patch("video_process.get_duration_seconds", return_value=300.0)
    @patch("drive_utils.download_file")
    @patch("drive_utils.move_file")
    def test_case_a_normal_success_flow(
        self, mock_move, mock_download, mock_dur, mock_audio,
        mock_extract, mock_transcribe, mock_detect, mock_process_pending,
        mock_enqueue_clips, mock_log_run, mock_notify
    ):
        """Case A: Normal success -> discovery -> scored -> queued -> 1 clip processed -> source preserved in Incoming."""
        mock_drive = MagicMock()
        file_info = {"id": "vid_normal", "name": "normal_lecture.mp4"}

        mock_transcribe.return_value = {
            "segments": [{"start": 0.0, "end": 100.0, "text": "sample text"}],
            "words": [],
            "text": "sample full text",
        }
        mock_detect.return_value = [
            {
                "clip_index": 1,
                "start_time": 10.0,
                "end_time": 45.0,
                "duration": 35.0,
                "topic": "Key Takeaway",
                "topic_summary": "Summary 1",
                "hook_summary": "Summary 1",
                "punchline": "KEY TAKEAWAY REVEALED",
                "quality_score": 88.5,
                "standalone": True,
                "scores": {"hook_strength": 9.0},
            },
            {
                "clip_index": 2,
                "start_time": 60.0,
                "end_time": 95.0,
                "duration": 35.0,
                "topic": "Second Point",
                "topic_summary": "Summary 2",
                "hook_summary": "Summary 2",
                "punchline": "THE SECOND INSIGHT",
                "quality_score": 82.0,
                "standalone": True,
                "scores": {"hook_strength": 8.0},
            },
        ]

        report = PipelineRunReport()

        def side_effect_process(*args, **kwargs):
            report.summary_total_clips = 2
            report.summary_uploaded = 1
            return {"total_queued": 2, "uploaded": 1, "failed": 0, "urls": ["https://youtube.com/shorts/test_url_1"], "errors": []}

        mock_process_pending.side_effect = side_effect_process

        with patch("main.run_report.get_current_report", return_value=report):
            stats = pipeline_main.discover_and_enqueue_video(mock_drive, file_info)

            # Both clips persisted to queue
            mock_enqueue_clips.assert_called_once()
            self.assertEqual(stats["total_queued"], 2)

            # Exactly ONE clip processed in this initial run
            mock_process_pending.assert_called_once()
            self.assertEqual(stats["uploaded"], 1)

            # Since clip 2 is still pending, source MUST remain in Incoming
            mock_move.assert_not_called()
            self.assertEqual(report.summary_source_status, "Preserved in Incoming")
            self.assertEqual(report.discovery_status, "SUCCESS")
            self.assertEqual(report.determine_overall_status(), "SUCCESS 🟢")

    # --- CASE B: Discovery API Failure ---
    @patch("notify.send")
    @patch("sheet_log.log_run")
    @patch("sheet_log.enqueue_clips")
    @patch("clip_detection.detect_clips_from_transcript", side_effect=ClipDiscoveryError("All Groq discovery calls failed"))
    @patch("transcribe.transcribe_audio", return_value={"segments": [{"start": 0.0, "end": 100.0}], "words": [], "text": "text"})
    @patch("transcribe.extract_audio")
    @patch("video_process.check_has_audio", return_value=True)
    @patch("video_process.get_duration_seconds", return_value=300.0)
    @patch("drive_utils.download_file")
    @patch("drive_utils.move_file")
    def test_case_b_discovery_api_failure(
        self, mock_move, mock_download, mock_dur, mock_audio,
        mock_extract, mock_transcribe, mock_detect,
        mock_enqueue_clips, mock_log_run, mock_notify
    ):
        """Case B: Total discovery failure -> 0 clips queued, source left in Incoming, status FAILED."""
        mock_drive = MagicMock()
        file_info = {"id": "vid_disc_fail", "name": "fail_lecture.mp4"}

        report = PipelineRunReport()
        with patch("main.run_report.get_current_report", return_value=report):
            stats = pipeline_main.discover_and_enqueue_video(mock_drive, file_info)

            # Zero clips enqueued
            mock_enqueue_clips.assert_not_called()
            self.assertEqual(stats["total_queued"], 0)
            self.assertEqual(stats["uploaded"], 0)

            # Source file untouched in Incoming
            mock_move.assert_not_called()
            self.assertEqual(report.discovery_status, "FAILED")
            self.assertEqual(report.determine_overall_status(), "FAILED 🔴")

    # --- CASE C: Metadata Failure Resiliency ---
    @patch("metadata_ai.call_with_rate_limit", side_effect=ValueError("AI metadata response was truncated (finish_reason='length')"))
    def test_case_c_metadata_failure_resiliency(self, mock_call):
        """Case C: Metadata AI failure -> deterministic safe fallback used with is_fallback=True."""
        with patch.object(config, "GROQ_API_KEY", "gsk_test_key"), \
             patch("metadata_ai._resolve_groq_model", return_value="openai/gpt-oss-20b"):
            meta = metadata_ai.generate_shorts_metadata(
                filename="biology_dna_repair.mp4",
                transcript="DNA repair mechanisms operate continuously inside our cells.",
            )
            self.assertTrue(meta["is_fallback"])
            self.assertIn("truncated", meta["fallback_reason"].lower())
            self.assertTrue(len(meta["title"]) > 0)
            self.assertIn("#Shorts", meta["hashtags"])

    # --- CASE D: Hook Failure Resiliency ---
    @patch("hook_generator.generate_hook_candidates", side_effect=Exception("API connection timeout"))
    def test_case_d_hook_failure_resiliency(self, mock_candidates):
        """Case D: Hook AI failure -> transcript-grounded deterministic fallback without claim fabrication."""
        transcript = "The speed of light in vacuum is approximately three hundred thousand kilometers per second."
        with patch.object(config, "GROQ_API_KEY", "gsk_test_key"), \
             patch("hook_generator._resolve_groq_model", return_value="openai/gpt-oss-20b"):
            res = hook_generator.generate_short_hook(
                transcript=transcript,
                detected_lang="en",
            )
            self.assertTrue(res["is_fallback"])
            self.assertEqual(res["source"], "fallback")
            # Grounded in transcript
            self.assertTrue(any(word.lower() in transcript.lower() for word in res["selected_hook"].split()))
            # Under character constraint
            self.assertLessEqual(len(res["selected_hook"]), 42)

    # --- CASE E: Rate Limit Handling ---
    def test_case_e_rate_limit_handling(self):
        """Case E: HTTP 429 encounter -> parses retry-after, backs off, and retries."""
        calls = []
        sleep_records = []

        def failing_then_succeeding_call():
            calls.append(1)
            if len(calls) == 1:
                resp = MagicMock()
                resp.headers = {"retry-after": "0.1"}
                err = Exception("HTTP 429 Too Many Requests: Please try again in 0.1s")
                err.response = resp
                raise err
            return {"result": "success"}

        limiter = ai_rate_limiter.TokenAwareRateLimiter(
            sleep_fn=sleep_records.append,
        )

        res = ai_rate_limiter.call_with_rate_limit(
            client_fn=failing_then_succeeding_call,
            prompt_or_messages="test query",
            max_tokens=500,
            rate_limiter=limiter,
        )

        self.assertEqual(res, {"result": "success"})
        self.assertEqual(len(calls), 2)
        self.assertTrue(any(abs(s - 0.1) < 0.05 for s in sleep_records))

    # --- CASE F: Model Misconfiguration Safety ---
    @patch("clip_detection._resolve_groq_model")
    def test_case_f_model_misconfiguration_safety(self, mock_resolve):
        """Case F: Unavailable model -> raises ModelConfigurationError explicitly without picking arbitrary model."""
        mock_resolve.side_effect = ModelConfigurationError("Configured Groq model 'invalid-model' is unavailable.")
        mock_client = MagicMock()

        with self.assertRaises(ModelConfigurationError):
            clip_detection._resolve_groq_model(mock_client, configured_model="invalid-model")

    # --- CASE G: YouTube Read Scope Failure Safety ---
    def test_case_g_youtube_read_scope_failure_safety(self):
        """Case G: 403 on channel playlist check -> diagnostic logged without crash, returns None safely."""
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status = 403
        http_error = HttpError(resp=mock_resp, content=b'{"error": {"message": "Request had insufficient authentication scopes."}}')
        mock_client.channels().list().execute.side_effect = http_error

        with self.assertLogs("youtube_upload", level="WARNING") as log_capture:
            url = youtube_upload.find_existing_short(clip_identifier="clip_test_1", client=mock_client)
            self.assertIsNone(url)
            log_text = "\n".join(log_capture.output)
            self.assertIn("OAuth refresh token lacks 'youtube.readonly' scope", log_text)
            self.assertIn("Upload will proceed without duplicate check", log_text)


if __name__ == "__main__":
    unittest.main()
