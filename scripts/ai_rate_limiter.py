"""
Token-Aware Rolling-Window Rate Limiter for AI Provider Calls (Groq / OpenAI).
Prevents burst rate-limit exhaustion (HTTP 429) across sequential AI requests
by tracking estimated token usage (prompt + max_completion_tokens) over a rolling 60-second window.
"""
import logging
import os
import re
import time
from typing import Callable, List, Optional, Tuple

from config import (
    GROQ_TPM_LIMIT,
    GROQ_REQUEST_MARGIN,
    GROQ_MIN_REQUEST_DELAY,
    GROQ_MAX_429_RETRIES,
    GROQ_MAX_429_BACKOFF,
)

log = logging.getLogger("ai_rate_limiter")


def estimate_tokens(text: str) -> int:
    """
    Conservative, dependency-free token estimator.
    In English / mixed alphanumeric text, 1 token is roughly 3.5 to 4 characters.
    Using 3.2 chars/token ensures conservative estimation with headroom.
    """
    if not text:
        return 0
    # Base estimate from characters
    char_tokens = len(text) / 3.2
    # Count whitespace words for validation
    word_tokens = len(text.split()) * 1.3
    return int(max(char_tokens, word_tokens)) + 5


def estimate_request_tokens(prompt_or_messages, max_tokens: int = 0) -> int:
    """Estimates total token demand for an LLM chat completion request."""
    prompt_text = ""
    if isinstance(prompt_or_messages, str):
        prompt_text = prompt_or_messages
    elif isinstance(prompt_or_messages, list):
        for msg in prompt_or_messages:
            if isinstance(msg, dict):
                prompt_text += " " + str(msg.get("content", ""))

    prompt_toks = estimate_tokens(prompt_text)
    completion_budget = max(0, int(max_tokens or 0))
    return prompt_toks + completion_budget


def extract_retry_after(error: Exception) -> Optional[float]:
    """Extracts suggested retry-after delay in seconds from HTTP 429 exception/response."""
    # Check headers if available on error.response
    resp = getattr(error, "response", None)
    if resp and hasattr(resp, "headers"):
        ra = resp.headers.get("retry-after")
        if ra:
            try:
                return float(ra)
            except (ValueError, TypeError):
                pass

    # Check error message strings (e.g., "Please try again in 23s", "Retrying request in 23.000000 seconds")
    err_str = str(error)
    match = re.search(r"try again in ([\d\.]+)s", err_str, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass

    match2 = re.search(r"in ([\d\.]+) seconds", err_str, re.IGNORECASE)
    if match2:
        try:
            return float(match2.group(1))
        except ValueError:
            pass

    return None


class TokenAwareRateLimiter:
    """
    Rolling-window token-aware rate limiter.
    Maintains a rolling 60-second window of token consumption.
    Throttles outgoing requests BEFORE sending to prevent 429 TPM overages.
    """

    def __init__(
        self,
        tpm_limit: int = GROQ_TPM_LIMIT,
        request_margin: float = GROQ_REQUEST_MARGIN,
        min_delay: float = GROQ_MIN_REQUEST_DELAY,
        window_seconds: float = 60.0,
        time_fn: Callable[[], float] = time.time,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        self.tpm_limit = tpm_limit
        self.request_margin = request_margin
        self.min_delay = min_delay
        self.window_seconds = window_seconds
        self.time_fn = time_fn
        self.sleep_fn = sleep_fn
        self._history: List[Tuple[float, int]] = []
        self._last_request_time: float = 0.0

    @property
    def effective_limit(self) -> int:
        return int(self.tpm_limit * self.request_margin)

    def _purge_old_entries(self, current_time: float):
        cutoff = current_time - self.window_seconds
        self._history = [entry for entry in self._history if entry[0] > cutoff]

    def get_rolling_usage(self, current_time: float = None) -> int:
        now = self.time_fn() if current_time is None else current_time
        self._purge_old_entries(now)
        return sum(tokens for _, tokens in self._history)

    def calculate_wait_time(self, estimated_tokens: int, current_time: float = None) -> float:
        """Calculates seconds to wait before sending this request."""
        now = self.time_fn() if current_time is None else current_time
        self._purge_old_entries(now)

        rolling_usage = sum(tokens for _, tokens in self._history)
        limit = self.effective_limit
        wait_seconds = 0.0

        # Check token capacity
        if rolling_usage + estimated_tokens > limit:
            needed_reduction = (rolling_usage + estimated_tokens) - limit
            freed = 0
            for timestamp, tokens in sorted(self._history, key=lambda x: x[0]):
                freed += tokens
                candidate_wait = (timestamp + self.window_seconds) - now
                if candidate_wait > wait_seconds:
                    wait_seconds = candidate_wait
                if freed >= needed_reduction:
                    break

        # Check minimum inter-request delay
        if self._last_request_time > 0.0:
            elapsed_since_last = now - self._last_request_time
            if elapsed_since_last < self.min_delay:
                inter_delay = self.min_delay - elapsed_since_last
                wait_seconds = max(wait_seconds, inter_delay)

        return max(0.0, round(wait_seconds, 2))

    def acquire(self, estimated_tokens: int):
        """Waits if necessary to stay within rolling token limit, then records usage."""
        now = self.time_fn()
        wait_secs = self.calculate_wait_time(estimated_tokens, current_time=now)

        if wait_secs > 0:
            rolling = self.get_rolling_usage(now)
            log.info(
                "[AI_RATE_LIMIT] estimated_tokens=%d rolling_usage=%d limit=%d wait_seconds=%.2f",
                estimated_tokens, rolling, self.effective_limit, wait_secs,
            )
            if self.sleep_fn is not time.sleep or os.environ.get("RATE_LIMITER_NO_SLEEP") != "1":
                self.sleep_fn(wait_secs)
            now = self.time_fn()

        self._history.append((now, estimated_tokens))
        self._last_request_time = now

    def record_actual_completion(self, additional_tokens: int = 0):
        """Optional update when actual completion tokens are known."""
        if additional_tokens > 0 and self._history:
            t, est = self._history[-1]
            self._history[-1] = (t, est + additional_tokens)


# Shared global singleton for shared rate limiting across all AI subsystems
shared_rate_limiter = TokenAwareRateLimiter()


def call_with_rate_limit(
    client_fn: Callable[[], any],
    prompt_or_messages,
    max_tokens: int = 0,
    rate_limiter: TokenAwareRateLimiter = None,
    max_429_retries: int = GROQ_MAX_429_RETRIES,
    max_429_backoff: float = GROQ_MAX_429_BACKOFF,
):
    """
    Executes an LLM API call with token-aware proactive rate limiting
    and bounded HTTP 429 retry-after handling.
    """
    limiter = rate_limiter or shared_rate_limiter
    estimated = estimate_request_tokens(prompt_or_messages, max_tokens=max_tokens)

    retries_429 = 0
    while True:
        limiter.acquire(estimated)
        try:
            return client_fn()
        except Exception as e:
            err_str = str(e).lower()
            is_429 = "429" in err_str or "too many requests" in err_str or getattr(e, "status_code", None) == 429
            if is_429 and retries_429 < max_429_retries:
                retries_429 += 1
                server_delay = extract_retry_after(e)
                if server_delay is not None:
                    backoff = min(server_delay, max_429_backoff)
                else:
                    backoff = min(15.0 * (2 ** (retries_429 - 1)), max_429_backoff)

                log.warning(
                    "[AI_RATE_LIMIT] HTTP 429 encountered (retry %d/%d). Sleeping %.1fs before retrying...",
                    retries_429, max_429_retries, backoff,
                )
                if limiter.sleep_fn is not time.sleep or os.environ.get("RATE_LIMITER_NO_SLEEP") != "1":
                    limiter.sleep_fn(backoff)
                continue
            raise
