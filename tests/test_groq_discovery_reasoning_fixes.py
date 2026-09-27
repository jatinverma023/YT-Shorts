"""
Unit and regression tests verifying Groq Discovery JSON / Reasoning fixes and Rate Limiter updates.
Verifies:
1. gpt-oss-20b discovery request includes response_format={"type": "json_object"}.
2. gpt-oss-20b discovery request includes extra_body={"reasoning_format": "hidden"}.
3. OpenAI client is initialized with max_retries=0.
4. HTTP 400 does NOT refund the full estimated request (retains prompt tokens in rolling history).
5. HTTP 429 follows the custom limiter retry path.
6. 429 retries do not bypass the limiter and do not double-reserve tokens.
7. Candidate maximum remains 3.
8. Compact discovery schema remains intact.
9. Existing Python quality scoring remains authoritative.
10. No mechanical/fake fallback is reintroduced.
11. Existing discovery failure still preserves source safety.
12. Existing production reliability tests continue passing.
"""
import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

import config
import clip_detection
from ai_rate_limiter import TokenAwareRateLimiter, call_with_rate_limit, estimate_request_tokens


class TestGroqDiscoveryReasoningFixes(unittest.TestCase):
    """Focused regression test suite for reasoning suppression, rate limiting, and client initialization."""

    def test_1_gpt_oss_20b_discovery_includes_response_format_json_object(self):
        """TEST 1: gpt-oss-20b discovery request explicitly includes response_format={'type': 'json_object'}."""
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content='{"clips": []}'))]
        mock_client.chat.completions.create.return_value = mock_resp

        with patch("clip_detection.call_with_rate_limit", side_effect=lambda client_fn, **kw: client_fn()):
            res = clip_detection._detect_clips_llm(
                client=mock_client,
                model="openai/gpt-oss-20b",
                transcript_text="[0.0s - 30.0s] This is a test talk segment.",
                total_duration=30.0,
                min_clip_seconds=20,
                max_clip_seconds=59,
            )

        self.assertIsNotNone(res)
        self.assertEqual(mock_client.chat.completions.create.call_count, 1)
        kwargs = mock_client.chat.completions.create.call_args[1]
        self.assertEqual(kwargs.get("response_format"), {"type": "json_object"})

    def test_2_gpt_oss_20b_discovery_includes_extra_body_reasoning_hidden(self):
        """TEST 2: gpt-oss-20b discovery request includes extra_body={'reasoning_format': 'hidden'}."""
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content='{"clips": []}'))]
        mock_client.chat.completions.create.return_value = mock_resp

        with patch("clip_detection.call_with_rate_limit", side_effect=lambda client_fn, **kw: client_fn()):
            clip_detection._detect_clips_llm(
                client=mock_client,
                model="openai/gpt-oss-20b",
                transcript_text="[0.0s - 30.0s] This is a test talk segment.",
                total_duration=30.0,
                min_clip_seconds=20,
                max_clip_seconds=59,
            )

        kwargs = mock_client.chat.completions.create.call_args[1]
        self.assertEqual(kwargs.get("extra_body"), {"reasoning_format": "hidden"})

    def test_1b_qwen_discovery_includes_response_format_json_object(self):
        """TEST 1b: qwen/qwen3.8-27b discovery request explicitly includes response_format={'type': 'json_object'}."""
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content='{"clips": []}'))]
        mock_client.chat.completions.create.return_value = mock_resp

        with patch("clip_detection.call_with_rate_limit", side_effect=lambda client_fn, **kw: client_fn()):
            res = clip_detection._detect_clips_llm(
                client=mock_client,
                model="qwen/qwen3.8-27b",
                transcript_text="[0.0s - 30.0s] This is a test talk segment.",
                total_duration=30.0,
                min_clip_seconds=20,
                max_clip_seconds=59,
            )

        self.assertIsNotNone(res)
        self.assertEqual(mock_client.chat.completions.create.call_count, 1)
        kwargs = mock_client.chat.completions.create.call_args[1]
        self.assertEqual(kwargs.get("response_format"), {"type": "json_object"})

    def test_2b_qwen_discovery_includes_extra_body_reasoning_hidden(self):
        """TEST 2b: qwen/qwen3.8-27b discovery request includes extra_body={'reasoning_format': 'hidden'}."""
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content='{"clips": []}'))]
        mock_client.chat.completions.create.return_value = mock_resp

        with patch("clip_detection.call_with_rate_limit", side_effect=lambda client_fn, **kw: client_fn()):
            clip_detection._detect_clips_llm(
                client=mock_client,
                model="qwen/qwen3.8-27b",
                transcript_text="[0.0s - 30.0s] This is a test talk segment.",
                total_duration=30.0,
                min_clip_seconds=20,
                max_clip_seconds=59,
            )

        kwargs = mock_client.chat.completions.create.call_args[1]
        self.assertEqual(kwargs.get("extra_body"), {"reasoning_format": "hidden"})

    @patch("clip_detection.OpenAI")
    def test_3_openai_client_initialized_with_max_retries_zero(self, mock_openai_cls):
        """TEST 3: OpenAI client is initialized with max_retries=0 to give limiter sole retry ownership."""
        with patch.object(clip_detection, "GROQ_API_KEY", "gsk_test_api_key_12345"), \
             patch("clip_detection._resolve_groq_model", return_value="openai/gpt-oss-20b"):
            client, model = clip_detection._get_llm_client()

        self.assertEqual(mock_openai_cls.call_count, 1)
        _, kwargs = mock_openai_cls.call_args
        self.assertEqual(kwargs.get("max_retries"), 0)
        self.assertEqual(kwargs.get("base_url"), "https://api.groq.com/openai/v1")

    @patch("clip_detection.OpenAI")
    def test_3b_openai_standard_client_initialized_with_max_retries_zero(self, mock_openai_cls):
        """TEST 3b: Standard OpenAI client fallback is also initialized with max_retries=0."""
        with patch.object(clip_detection, "GROQ_API_KEY", ""), \
             patch.object(clip_detection, "OPENAI_API_KEY", "sk_test_key"):
            client, model = clip_detection._get_llm_client()

        self.assertEqual(mock_openai_cls.call_count, 1)
        _, kwargs = mock_openai_cls.call_args
        self.assertEqual(kwargs.get("max_retries"), 0)

    def test_4_http_400_does_not_refund_full_estimated_request(self):
        """TEST 4: HTTP 400 does NOT refund full request; prompt tokens remain in rolling history."""
        current_time = [1000.0]
        limiter = TokenAwareRateLimiter(
            tpm_limit=6800,
            request_margin=1.0,
            min_delay=0.0,
            time_fn=lambda: current_time[0],
            sleep_fn=lambda s: None,
        )

        prompt_str = "Timestamped interview transcript chunk with several sentences."
        prompt_tokens = estimate_request_tokens(prompt_str, max_tokens=0)
        completion_budget = 1500
        total_estimated = prompt_tokens + completion_budget

        def mock_failing_call():
            err = Exception("HTTP 400 Bad Request: json_validate_failed")
            err.status_code = 400
            raise err

        with self.assertRaises(Exception) as ctx:
            call_with_rate_limit(
                client_fn=mock_failing_call,
                prompt_or_messages=prompt_str,
                max_tokens=completion_budget,
                rate_limiter=limiter,
            )

        self.assertIn("400", str(ctx.exception))
        usage_after_400 = limiter.get_rolling_usage(1000.0)
        # 1. Rolling usage must NOT be 0 (full refund prevented)
        self.assertGreater(usage_after_400, 0)
        # 2. Rolling usage must NOT be total_estimated (completion budget was refunded)
        self.assertLess(usage_after_400, total_estimated)
        # 3. Rolling usage must exactly match prompt_tokens
        self.assertEqual(usage_after_400, prompt_tokens)

    def test_5_http_429_follows_custom_limiter_retry_path(self):
        """TEST 5: HTTP 429 triggers bounded retries via custom TokenAwareRateLimiter."""
        current_time = [1000.0]
        sleep_durations = []

        limiter = TokenAwareRateLimiter(
            tpm_limit=10000,
            request_margin=1.0,
            min_delay=0.0,
            time_fn=lambda: current_time[0],
            sleep_fn=lambda s: sleep_durations.append(s),
        )

        attempts = [0]

        def call_with_transient_429():
            attempts[0] += 1
            if attempts[0] < 3:
                err = Exception("Rate limit reached. Please try again in 5.5s")
                err.status_code = 429
                raise err
            return {"status": "success"}

        res = call_with_rate_limit(
            client_fn=call_with_transient_429,
            prompt_or_messages="test prompt",
            max_tokens=500,
            rate_limiter=limiter,
            max_429_retries=3,
        )

        self.assertEqual(res, {"status": "success"})
        self.assertEqual(attempts[0], 3)
        self.assertEqual(len(sleep_durations), 2)
        self.assertAlmostEqual(sleep_durations[0], 5.5, places=1)
        self.assertAlmostEqual(sleep_durations[1], 5.5, places=1)

    def test_6_429_retries_do_not_bypass_limiter_or_double_reserve(self):
        """TEST 6: 429 retries re-acquire through limiter without double-counting tokens."""
        current_time = [1000.0]

        def fake_sleep(s):
            current_time[0] += s

        limiter = TokenAwareRateLimiter(
            tpm_limit=6800,
            request_margin=1.0,
            min_delay=0.5,
            time_fn=lambda: current_time[0],
            sleep_fn=fake_sleep,
        )

        prompt_str = "Test prompt text for retry accounting verification."
        prompt_tokens = estimate_request_tokens(prompt_str, max_tokens=0)
        max_toks = 500
        total_estimated = prompt_tokens + max_toks

        attempts = [0]

        def call_with_one_429():
            attempts[0] += 1
            if attempts[0] == 1:
                err = Exception("HTTP 429 Too Many Requests: try again in 2.0s")
                err.status_code = 429
                raise err
            return {"status": "ok"}

        res = call_with_rate_limit(
            client_fn=call_with_one_429,
            prompt_or_messages=prompt_str,
            max_tokens=max_toks,
            rate_limiter=limiter,
        )

        self.assertEqual(res, {"status": "ok"})
        self.assertEqual(attempts[0], 2)
        # Exactly ONE copy of total_estimated should be in rolling usage, NOT double (2x)
        self.assertEqual(limiter.get_rolling_usage(current_time[0]), total_estimated)

    def test_7_candidate_maximum_remains_three(self):
        """TEST 7: Candidate ceiling remains strictly at 3."""
        mock_client = MagicMock()
        four_clips_json = json.dumps({
            "clips": [
                {"start": 10.0, "end": 40.0, "topic": "Clip 1", "standalone_score": 8.0},
                {"start": 50.0, "end": 80.0, "topic": "Clip 2", "standalone_score": 8.5},
                {"start": 90.0, "end": 120.0, "topic": "Clip 3", "standalone_score": 9.0},
                {"start": 130.0, "end": 160.0, "topic": "Clip 4", "standalone_score": 7.5},
            ]
        })
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content=four_clips_json))]
        mock_client.chat.completions.create.return_value = mock_resp

        with patch("clip_detection.call_with_rate_limit", side_effect=lambda client_fn, **kw: client_fn()):
            res = clip_detection._detect_clips_llm(
                client=mock_client,
                model="openai/gpt-oss-20b",
                transcript_text="[0.0s - 180.0s] Long transcript text.",
                total_duration=180.0,
                min_clip_seconds=20,
                max_clip_seconds=59,
            )

        self.assertEqual(len(res), 3)

    def test_8_compact_discovery_schema_intact(self):
        """TEST 8: Compact schema structure does not include legacy 8-score quality_scores dict."""
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content='{"clips": []}'))]
        mock_client.chat.completions.create.return_value = mock_resp

        with patch("clip_detection.call_with_rate_limit", side_effect=lambda client_fn, **kw: client_fn()):
            clip_detection._detect_clips_llm(
                client=mock_client,
                model="openai/gpt-oss-20b",
                transcript_text="[0.0s - 30.0s] Some text.",
                total_duration=30.0,
                min_clip_seconds=20,
                max_clip_seconds=59,
            )

        prompt_sent = mock_client.chat.completions.create.call_args[1]["messages"][0]["content"]
        self.assertNotIn("quality_scores", prompt_sent)
        self.assertIn('"standalone_score": 9.0', prompt_sent)
        self.assertIn('"missing_setup": false', prompt_sent)

    def test_9_existing_python_quality_scoring_remains_authoritative(self):
        """TEST 9: Authoritative Python derivation computes composite quality score from candidate fields."""
        raw_candidate = [{
            "start": 10.0,
            "end": 45.0,
            "topic": "Mindset",
            "topic_summary": "Mindset shift required to overcome fear.",
            "title_idea": "The Secret to Overcoming Fear #shorts",
            "punchline": "Conquer Your Fear 🔥",
            "standalone": True,
            "standalone_score": 9.0,
            "missing_setup": False,
            "missing_payoff": False,
            "critical_unresolved_reference": False,
            "unresolved_references": [],
        }]

        validated = clip_detection._validate_and_filter_clips(
            raw_clips=raw_candidate,
            total_duration=100.0,
            min_clip_seconds=20,
            max_clip_seconds=59,
        )

        self.assertEqual(len(validated), 1)
        scored_clip = validated[0]
        self.assertIn("quality_score", scored_clip)
        # With standalone_score=9.0, Python derivation computes quality score ~89-91
        self.assertGreaterEqual(scored_clip["quality_score"], 80.0)
        self.assertEqual(scored_clip["quality_scores"]["standalone_clarity"], 9.0)

    def test_10_no_mechanical_fake_fallback_reintroduced(self):
        """TEST 10: Complete discovery failure raises ClipDiscoveryError; NO mechanical clips generated."""
        segments = [{"start": 0.0, "end": 100.0, "text": "Some dialogue across segments"}]
        mock_client = MagicMock()

        # All LLM calls return None (simulating API failure across all chunks)
        with patch.object(clip_detection, "_get_llm_client", return_value=(mock_client, "openai/gpt-oss-20b")), \
             patch.object(clip_detection, "_detect_clips_llm", return_value=None):
            with self.assertRaises(clip_detection.ClipDiscoveryError):
                clip_detection.detect_clips_from_transcript(
                    segments=segments,
                    total_duration=100.0,
                    min_clip_seconds=20,
                    max_clip_seconds=59,
                )

    def test_11_discovery_failure_still_preserves_source_safety(self):
        """TEST 11: main.py correctly catches ClipDiscoveryError and preserves video in Incoming."""
        import main as pipeline_main

        mock_drive_service = MagicMock()
        video = {"id": "incoming_vid_123", "name": "long_video.mp4"}

        with patch("main.drive_utils.download_file"), \
             patch("main.video_process.get_duration_seconds", return_value=120.0), \
             patch("main.video_process.check_has_audio", return_value=True), \
             patch("main.transcribe.extract_audio"), \
             patch("main.transcribe.transcribe_audio", return_value={"segments": [{"start": 0, "end": 100, "text": "text"}]}), \
             patch("main.clip_detection.detect_clips_from_transcript", side_effect=clip_detection.ClipDiscoveryError("All LLM calls failed")), \
             patch("main.drive_utils.move_file") as mock_move, \
             patch("main.sheet_log.log_run"), \
             patch("main.notify.send"):

            res = pipeline_main.discover_and_enqueue_video(
                drive_service=mock_drive_service,
                file_info=video,
            )

            # Move to Processed or Failed must NOT be called when discovery fails!
            mock_move.assert_not_called()
            self.assertEqual(res.get("total_queued"), 0)
            self.assertEqual(res.get("uploaded"), 0)
            self.assertTrue(res.get("discovery_failed"))


if __name__ == "__main__":
    unittest.main()
