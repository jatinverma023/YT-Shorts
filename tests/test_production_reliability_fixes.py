"""
Unit tests for the four targeted production reliability fixes:
1. Groq discovery compact candidate schema & max-3 clip ceiling.
2. Rate limiter reservation refund on deterministic non-429 errors.
3. Google Sheets comma-separated numeric parsing.
4. Run summary accounting & degraded state tracking.
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

import ai_rate_limiter
from ai_rate_limiter import TokenAwareRateLimiter, call_with_rate_limit
import clip_detection
from clip_detection import _detect_clips_llm, _validate_and_filter_clips
import main as pipeline_main
import run_report
from run_report import PipelineRunReport
import sheet_log
from sheet_log import parse_sheet_float, parse_sheet_int


class TestGroqDiscoveryCompactFixes(unittest.TestCase):
    """Fix 1: Verifies compact schema prompt, <= 3 ceiling, and standalone fallback scoring."""

    def test_detect_clips_llm_enforces_max_3_clips(self):
        """Even if LLM returns 5 clips, _detect_clips_llm must truncate to at most 3."""
        fake_response = {
            "clips": [
                {"start_time": 10.0, "end_time": 40.0, "standalone_score": 9.0},
                {"start_time": 50.0, "end_time": 80.0, "standalone_score": 8.5},
                {"start_time": 90.0, "end_time": 120.0, "standalone_score": 8.0},
                {"start_time": 130.0, "end_time": 160.0, "standalone_score": 7.5},
                {"start_time": 170.0, "end_time": 200.0, "standalone_score": 7.0},
            ]
        }
        mock_client = MagicMock()
        mock_choice = MagicMock()
        import json
        mock_choice.message.content = json.dumps(fake_response)
        mock_client.chat.completions.create.return_value.choices = [mock_choice]

        with patch("clip_detection.call_with_rate_limit", side_effect=lambda *a, **kw: (a[0] if a else kw["client_fn"])()):
            clips = _detect_clips_llm(
                client=mock_client,
                model="openai/gpt-oss-20b",
                transcript_text="sample transcript",
                total_duration=200.0,
                min_clip_seconds=20,
                max_clip_seconds=60,
            )
            self.assertIsNotNone(clips)
            self.assertEqual(len(clips), 3)

    def test_validate_and_filter_clips_derives_quality_score_without_raw_scores(self):
        """When quality_scores dictionary is omitted from JSON, Python calculates valid quality score."""
        raw_candidates = [
            {
                "start_time": 15.0,
                "end_time": 50.0,
                "standalone_score": 9.0,
                "missing_setup": False,
                "missing_payoff": False,
                "critical_unresolved_reference": False,
                "punchline": "That is why persistence always wins.",
                "reason": "Complete thought with setup and punchline.",
            }
        ]
        filtered = _validate_and_filter_clips(
            raw_clips=raw_candidates,
            total_duration=100.0,
            min_clip_seconds=20,
            max_clip_seconds=60,
        )
        self.assertEqual(len(filtered), 1)
        clip = filtered[0]
        # Quality score must be computed deterministically and exceed passing threshold (70.0)
        self.assertGreaterEqual(clip["quality_score"], 70.0)
        self.assertIn("quality_scores", clip)
        self.assertEqual(clip["quality_scores"]["hook_strength"], 9.0)


class TestRateLimiterRefundFixes(unittest.TestCase):
    """Fix 2: Verifies reservation rollback/refund on deterministic non-429 errors."""

    def test_refund_decreases_rolling_usage(self):
        """refund() properly rolls back history reservations."""
        current_time = [1000.0]
        limiter = TokenAwareRateLimiter(
            tpm_limit=5000,
            request_margin=1.0,
            min_delay=0.0,
            time_fn=lambda: current_time[0],
            sleep_fn=lambda s: None,
        )

        limiter.acquire(2000)
        self.assertEqual(limiter.get_rolling_usage(1000.0), 2000)

        # Refund 1000 tokens
        limiter.refund(1000)
        self.assertEqual(limiter.get_rolling_usage(1000.0), 1000)

        # Refund remaining 1000 tokens
        limiter.refund(1000)
        self.assertEqual(limiter.get_rolling_usage(1000.0), 0)

        # Extra refund must not crash or make usage negative
        limiter.refund(500)
        self.assertEqual(limiter.get_rolling_usage(1000.0), 0)

    def test_call_with_rate_limit_refunds_on_http_400(self):
        """Deterministic non-429 client errors trigger token refund."""
        current_time = [1000.0]
        limiter = TokenAwareRateLimiter(
            tpm_limit=5000,
            request_margin=1.0,
            min_delay=0.0,
            time_fn=lambda: current_time[0],
            sleep_fn=lambda s: None,
        )

        def mock_bad_request():
            err = Exception("HTTP 400 json_validate_failed")
            err.status_code = 400
            raise err

        with self.assertRaises(Exception) as ctx:
            call_with_rate_limit(
                client_fn=mock_bad_request,
                prompt_or_messages="test prompt",
                max_tokens=1500,
                rate_limiter=limiter,
            )

        self.assertIn("400", str(ctx.exception))
        # Rolling usage must be 0 because tokens were refunded!
        self.assertEqual(limiter.get_rolling_usage(1000.0), 0)

    def test_call_with_rate_limit_does_not_refund_on_http_200(self):
        """Successful request retains token usage in rolling window."""
        current_time = [1000.0]
        limiter = TokenAwareRateLimiter(
            tpm_limit=5000,
            request_margin=1.0,
            min_delay=0.0,
            time_fn=lambda: current_time[0],
            sleep_fn=lambda s: None,
        )

        res = call_with_rate_limit(
            client_fn=lambda: {"status": "ok"},
            prompt_or_messages="test prompt",
            max_tokens=500,
            rate_limiter=limiter,
        )
        self.assertEqual(res, {"status": "ok"})
        self.assertGreater(limiter.get_rolling_usage(1000.0), 500)


class TestSheetLogNumericParsingFixes(unittest.TestCase):
    """Fix 3: Verifies robust parsing of comma-formatted numbers like '7,674.1'."""

    def test_parse_sheet_float_with_commas_and_whitespace(self):
        self.assertEqual(parse_sheet_float("7,674.1"), 7674.1)
        self.assertEqual(parse_sheet_float(" 1,234,567.89 "), 1234567.89)
        self.assertEqual(parse_sheet_float("45.0"), 45.0)
        self.assertEqual(parse_sheet_float(99.5), 99.5)
        self.assertEqual(parse_sheet_float(100), 100.0)
        self.assertEqual(parse_sheet_float("", default=0.0), 0.0)
        self.assertEqual(parse_sheet_float(None, default=5.0), 5.0)

        with self.assertRaises(ValueError):
            parse_sheet_float("invalid_number")
        with self.assertRaises(ValueError):
            parse_sheet_float("")

    def test_parse_sheet_int_with_commas_and_whitespace(self):
        self.assertEqual(parse_sheet_int("1,000"), 1000)
        self.assertEqual(parse_sheet_int(" 42 "), 42)
        self.assertEqual(parse_sheet_int(7), 7)
        self.assertEqual(parse_sheet_int("", default=1), 1)
        self.assertEqual(parse_sheet_int(None, default=1), 1)
        self.assertEqual(parse_sheet_int("not_a_num", default=1), 1)

    @patch("sheet_log.get_active_video_in_queue", return_value=("vid_123", "sample.mp4"))
    def test_get_next_pending_clip_handles_comma_separated_timestamps(self, mock_active):
        """Row 2 with '7,674.1' must be parsed successfully without being skipped."""
        mock_service = MagicMock()
        sheet_rows = [
            sheet_log.QUEUE_HEADERS,
            [
                "sample.mp4",
                "vid_123",
                "1",
                "7,674.1",  # formatted timestamp with comma!
                "7,714.5",  # formatted timestamp with comma!
                "Test hook",
                "pending",
                "",
                "",
                "2026-09-25T10:00:00",
                "2026-09-25T10:00:00",
                "Punchline",
                "85.5",
            ],
        ]
        mock_service.spreadsheets().values().get().execute.return_value = {"values": sheet_rows}

        result = sheet_log.get_next_pending_clip(service=mock_service)
        self.assertIsNotNone(result)
        row_num, clip_data = result
        self.assertEqual(row_num, 2)
        self.assertEqual(clip_data["start_time"], 7674.1)
        self.assertEqual(clip_data["end_time"], 7714.5)
        self.assertEqual(clip_data["quality_score"], 85.5)


class TestRunSummaryAccountingFixes(unittest.TestCase):
    """Fix 4: Verifies run summary accounting and degraded state reporting."""

    def test_run_report_degraded_when_clips_pending_but_zero_uploads(self):
        """When total > 0, uploaded == 0, failed == 0, and pending > 0 -> status is DEGRADED ⚠️."""
        report = PipelineRunReport()
        report.summary_total_clips = 2
        report.summary_uploaded = 0
        report.summary_failed = 0
        report.summary_pending = 2

        status = report.determine_overall_status()
        self.assertEqual(status, "DEGRADED ⚠️")

    @patch("main.shutil.rmtree")
    @patch("main.notify.send")
    @patch("main.sheet_log.enqueue_clips")
    @patch("main.process_next_pending_clip")
    @patch("main.clip_detection.detect_clips_from_transcript")
    @patch("main.transcribe.transcribe_audio")
    @patch("main.transcribe.extract_audio")
    @patch("main.video_process.check_has_audio", return_value=True)
    @patch("main.video_process.get_duration_seconds", return_value=120.0)
    @patch("main.drive_utils.download_file")
    def test_discover_and_enqueue_video_retains_total_queued(
        self,
        mock_down,
        mock_dur,
        mock_has_audio,
        mock_extract,
        mock_transcribe,
        mock_detect,
        mock_process_pending,
        mock_enqueue,
        mock_notify,
        mock_rmtree,
    ):
        """discover_and_enqueue_video stats['total_queued'] reflects discovered count."""
        mock_transcribe.return_value = {
            "segments": [{"start": 0.0, "end": 100.0, "text": "sample"}],
            "words": [],
            "text": "sample",
        }
        mock_detect.return_value = [
            {"clip_index": 1, "start_time": 10.0, "end_time": 40.0, "quality_score": 85.0},
            {"clip_index": 2, "start_time": 50.0, "end_time": 80.0, "quality_score": 82.0},
        ]
        # process_next_pending_clip processes ONE clip
        mock_process_pending.return_value = {
            "total_queued": 1,
            "uploaded": 1,
            "failed": 0,
            "urls": ["https://youtube.com/shorts/sample1"],
            "errors": [],
        }

        report = PipelineRunReport()
        with patch("main.run_report.get_current_report", return_value=report):
            stats = pipeline_main.discover_and_enqueue_video(
                drive_service=MagicMock(),
                file_info={"id": "vid_abc", "name": "sample_video.mp4"},
            )

            # stats['total_queued'] must equal 2 (discovered_count), NOT 1
            self.assertEqual(stats["total_queued"], 2)
            self.assertEqual(stats["enqueued"], 2)
            self.assertEqual(stats["uploaded"], 1)
            self.assertEqual(report.summary_total_clips, 2)


if __name__ == "__main__":
    unittest.main()
