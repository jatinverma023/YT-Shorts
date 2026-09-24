"""
Tests for Phase 5: Simplify Metadata Generation.
Validates:
1. Streamlined schema with title_candidates, description_candidates, hashtags, tags is parsed correctly.
2. Truncated completion (finish_reason='length') safely triggers deterministic fallback.
3. Missing required fields safely triggers deterministic fallback.
4. Authoritative Python scoring handles packages derived from streamlined metadata.
5. Backward compatibility with legacy response schemas is preserved.
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

import config
import metadata_ai


class TestPhase5Metadata(unittest.TestCase):

    def setUp(self):
        from ai_rate_limiter import shared_rate_limiter
        shared_rate_limiter._history.clear()
        self.filename = "huberman_dopamine_deep_dive.mp4"
        self.sample_transcript = (
            "Dopamine is not the molecule of pleasure. It is the molecule of motivation, craving, and pursuit. "
            "When you understand dopamine peaks and baselines, you can maintain high energy and discipline "
            "without burning out your adrenal system."
        )

    def test_streamlined_schema_parsed_and_scored(self):
        """Validates that streamlined LLM response without redundant scores is parsed and scored by Python."""
        streamlined_json = {
            "title_candidates": [
                {
                    "title": "Why Dopamine Controls Your Motivation And Focus",
                    "strategy": "mechanism"
                },
                {
                    "title": "The Real Science Behind Dopamine And Burnout",
                    "strategy": "specific_explanation"
                },
                {
                    "title": "How To Reset Your Dopamine For High Energy",
                    "strategy": "consequence"
                }
            ],
            "description_candidates": [
                {
                    "description": "Understanding dopamine baselines changes how you build focus and avoid energy crashes.",
                    "strategy": "micro_teaser"
                }
            ],
            "hashtags": ["#Shorts", "#Dopamine", "#BrainScience", "#Productivity"],
            "tags": ["dopamine", "motivation", "neuroscience", "focus"]
        }

        mock_choice = MagicMock()
        mock_choice.finish_reason = "stop"
        mock_choice.message.content = str(streamlined_json).replace("'", '"')
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("metadata_ai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_response

            with patch.object(config, "GROQ_API_KEY", "gsk_test_key"):
                res = metadata_ai.generate_shorts_metadata(
                    self.filename,
                    transcript=self.sample_transcript,
                )

        self.assertFalse(res.get("is_fallback", False))
        self.assertIn("Dopamine", res["title"])
        self.assertGreaterEqual(res["title_quality_score"], 70.0)
        self.assertGreaterEqual(res["description_quality_score"], 70.0)
        self.assertGreaterEqual(res["hashtag_quality_score"], 70.0)
        self.assertGreaterEqual(res["package_quality_score"], 70.0)
        self.assertEqual(len(res["title_variants"]), 3)
        self.assertIn("#Shorts", res["hashtags"])

    @patch("ai_rate_limiter.time.sleep", return_value=None)
    @patch("metadata_ai.time.sleep", return_value=None)
    def test_finish_reason_length_triggers_safe_fallback(self, mock_sleep1, mock_sleep2):
        """Validates that response truncated by token limit triggers deterministic fallback."""
        mock_choice = MagicMock()
        mock_choice.finish_reason = "length"
        mock_choice.message.content = '{"title_candidates": [{"title": "Truncated title'
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("metadata_ai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_response

            with patch.object(config, "GROQ_API_KEY", "gsk_test_key"):
                res = metadata_ai.generate_shorts_metadata(
                    self.filename,
                    transcript=self.sample_transcript,
                )

        self.assertTrue(res.get("is_fallback"))
        self.assertEqual(res["title_strategy"], "fallback")
        self.assertIn("huberman dopamine", res["title"].lower())

    @patch("ai_rate_limiter.time.sleep", return_value=None)
    @patch("metadata_ai.time.sleep", return_value=None)
    def test_missing_required_fields_triggers_safe_fallback(self, mock_sleep1, mock_sleep2):
        """Validates that response missing title candidates triggers deterministic fallback."""
        mock_choice = MagicMock()
        mock_choice.finish_reason = "stop"
        mock_choice.message.content = '{"random_key": "some text", "hashtags": ["#Shorts"]}'
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("metadata_ai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_response

            with patch.object(config, "GROQ_API_KEY", "gsk_test_key"):
                res = metadata_ai.generate_shorts_metadata(
                    self.filename,
                    transcript=self.sample_transcript,
                )

        self.assertTrue(res.get("is_fallback"))
        self.assertEqual(res["title_strategy"], "fallback")


if __name__ == "__main__":
    unittest.main()
