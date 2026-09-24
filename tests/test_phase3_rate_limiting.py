"""
Phase 3 Tests: Fix Clip-Discovery Rate Limiting Properly.
Verifies that:
1. TEST 8: Two requests whose combined token usage would exceed the rolling limit -> second request is delayed.
2. TEST 9: Requests under the limit -> no unnecessary delay.
3. TEST 10: HTTP 429 with Retry-After -> retry respects server-provided delay.
4. TEST 11: Repeated 429s -> bounded retry behavior.
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

import ai_rate_limiter
from ai_rate_limiter import (
    TokenAwareRateLimiter,
    estimate_tokens,
    estimate_request_tokens,
    call_with_rate_limit,
)


class TestPhase3RateLimiting(unittest.TestCase):

    def test_estimate_tokens_basic(self):
        """Verifies conservative token estimation."""
        text = "Hello world this is a test transcript line."
        toks = estimate_tokens(text)
        self.assertGreater(toks, 5)
        # 10 words, approx 43 chars -> approx 13-18 tokens
        self.assertLess(toks, 30)

    def test_test8_two_requests_exceeding_rolling_limit_delays_second(self):
        """TEST 8: Two requests whose combined token usage would exceed limit -> second request is delayed."""
        # Simulated clock: starts at 1000.0
        current_time = [1000.0]
        sleep_calls = []

        def fake_time():
            return current_time[0]

        def fake_sleep(secs):
            sleep_calls.append(secs)
            current_time[0] += secs

        # Configure limiter with 5,000 TPM effective limit (6000 * 0.833)
        limiter = TokenAwareRateLimiter(
            tpm_limit=5000,
            request_margin=1.0,
            min_delay=0.1,
            window_seconds=60.0,
            time_fn=fake_time,
            sleep_fn=fake_sleep,
        )

        # Request 1: 3,500 tokens at t=1000
        limiter.acquire(3500)
        self.assertEqual(len(sleep_calls), 0)

        # Request 2: 3,500 tokens immediately (3500 + 3500 = 7000 > 5000 limit)
        # To drop under limit, Request 1 (at t=1000) must age out after 60s (at t=1060).
        # Wait time must be exactly 60.0 seconds!
        limiter.acquire(3500)
        self.assertEqual(len(sleep_calls), 1)
        self.assertAlmostEqual(sleep_calls[0], 60.0, places=1)
        self.assertAlmostEqual(current_time[0], 1060.0, places=1)

    def test_test9_requests_under_limit_no_unnecessary_delay(self):
        """TEST 9: Requests under the limit -> no unnecessary token delay."""
        current_time = [1000.0]
        sleep_calls = []

        def fake_time():
            return current_time[0]

        def fake_sleep(secs):
            sleep_calls.append(secs)
            current_time[0] += secs

        limiter = TokenAwareRateLimiter(
            tpm_limit=10000,
            request_margin=1.0,
            min_delay=0.0,  # disable min delay for clean test
            window_seconds=60.0,
            time_fn=fake_time,
            sleep_fn=fake_sleep,
        )

        # Request 1: 1,000 tokens
        limiter.acquire(1000)
        # Request 2: 1,000 tokens (1000 + 1000 = 2000 <= 10000)
        limiter.acquire(1000)
        # Request 3: 1,000 tokens (3000 <= 10000)
        limiter.acquire(1000)

        self.assertEqual(len(sleep_calls), 0)

    def test_test10_http_429_with_retry_after_respects_server_delay(self):
        """TEST 10: HTTP 429 with Retry-After -> retry respects server-provided delay."""
        current_time = [1000.0]
        sleep_calls = []

        def fake_time():
            return current_time[0]

        def fake_sleep(secs):
            sleep_calls.append(secs)
            current_time[0] += secs

        limiter = TokenAwareRateLimiter(
            tpm_limit=20000,
            request_margin=1.0,
            min_delay=0.0,
            time_fn=fake_time,
            sleep_fn=fake_sleep,
        )

        attempts = [0]

        def client_fn():
            attempts[0] += 1
            if attempts[0] == 1:
                # First attempt raises 429 with retry-after 23s
                mock_err = Exception("Rate limit reached. Please try again in 23.5s.")
                mock_err.status_code = 429
                raise mock_err
            return {"result": "success"}

        result = call_with_rate_limit(
            client_fn=client_fn,
            prompt_or_messages="test prompt",
            max_tokens=100,
            rate_limiter=limiter,
        )

        self.assertEqual(result, {"result": "success"})
        self.assertEqual(attempts[0], 2)
        # Verified: slept for server-provided 23.5 seconds!
        self.assertTrue(any(abs(s - 23.5) < 0.2 for s in sleep_calls))

    def test_test11_repeated_429s_bounded_retry_behavior(self):
        """TEST 11: Repeated 429s -> bounded retry behavior (fails after configured retries)."""
        current_time = [1000.0]
        sleep_calls = []

        def fake_time():
            return current_time[0]

        def fake_sleep(secs):
            sleep_calls.append(secs)
            current_time[0] += secs

        limiter = TokenAwareRateLimiter(
            tpm_limit=20000,
            request_margin=1.0,
            min_delay=0.0,
            time_fn=fake_time,
            sleep_fn=fake_sleep,
        )

        attempts = [0]

        def always_429():
            attempts[0] += 1
            mock_err = Exception("Rate limit reached: HTTP 429 Too Many Requests")
            mock_err.status_code = 429
            raise mock_err

        # max_429_retries=2
        with self.assertRaises(Exception) as ctx:
            call_with_rate_limit(
                client_fn=always_429,
                prompt_or_messages="test",
                max_tokens=100,
                rate_limiter=limiter,
                max_429_retries=2,
            )

        self.assertIn("429", str(ctx.exception))
        # Total attempts: 1 initial + 2 retries = 3 calls
        self.assertEqual(attempts[0], 3)
        self.assertEqual(len(sleep_calls), 2)


if __name__ == "__main__":
    unittest.main()
