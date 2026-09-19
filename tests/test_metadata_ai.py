"""
Unit tests for AI Metadata Generation and Robust Fallback Handling (scripts/metadata_ai.py).

Tests:
1. successful AI metadata response
2. valid JSON response
3. JSON wrapped in ```json fences (and ``` fences without language tag)
4. malformed JSON gracefully triggers fallback
5. API failure (e.g. RateLimitError/Timeout) triggers retry and falls back
6. missing API key uses fallback and logs clearly
7. description comes from transcript on successful AI generation
8. filename fallback is used only on genuine failure
9. fallback is clearly logged with [METADATA_FALLBACK_USED]
10. generated hook/title/description/tags remain compatible with the existing upload flow
"""
import json
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Ensure scripts directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

import config
import metadata_ai


class TestMetadataAI(unittest.TestCase):
    """Test suite for metadata generation, JSON extraction, and fallback mechanisms."""

    def test_extract_json_payload_valid_json(self):
        """Extracts JSON from clean, raw JSON string."""
        raw = '{"title_1": "Top Title #shorts", "description": "Engaging transcript caption.", "tags": ["viral"]}'
        result = metadata_ai.extract_json_payload(raw)
        self.assertEqual(result["title_1"], "Top Title #shorts")
        self.assertEqual(result["description"], "Engaging transcript caption.")

    def test_extract_json_payload_wrapped_in_json_code_fences(self):
        """Extracts JSON wrapped in ```json ... ``` markdown code fences."""
        raw = """```json
{
  "title_1": "Wrapped Title #shorts",
  "title_2": "Variant 2 #shorts",
  "title_3": "Variant 3 #shorts",
  "punchline": "Key punchline 🔥",
  "description": "This is a rich description grounded in the spoken words.",
  "tags": ["shorts", "podcast"]
}
```"""
        result = metadata_ai.extract_json_payload(raw)
        self.assertEqual(result["title_1"], "Wrapped Title #shorts")
        self.assertIn("rich description grounded", result["description"])

    def test_extract_json_payload_wrapped_in_plain_code_fences(self):
        """Extracts JSON wrapped in ``` ... ``` fences without 'json' language specifier."""
        raw = """```
{
  "title_1": "Plain Fenced Title #shorts",
  "description": "Fenced without json tag.",
  "tags": ["podcast"]
}
```"""
        result = metadata_ai.extract_json_payload(raw)
        self.assertEqual(result["title_1"], "Plain Fenced Title #shorts")

    def test_extract_json_payload_conversational_prefix_and_suffix(self):
        """Extracts JSON when model includes conversational text before and after."""
        raw = """Here is the metadata JSON you requested:
```json
{
  "title_1": "Surrounded Title #shorts",
  "description": "Insightful conversation excerpt."
}
```
Hope this helps increase your audience retention!"""
        result = metadata_ai.extract_json_payload(raw)
        self.assertEqual(result["title_1"], "Surrounded Title #shorts")

    def test_extract_json_payload_raw_newlines_in_string(self):
        """Handles unescaped newlines inside string values using strict=False."""
        raw = """{
  "title_1": "Newline Title #shorts",
  "description": "Line 1 insight summary.\n\n#shorts #podcast #mindset",
  "tags": ["shorts"]
}"""
        result = metadata_ai.extract_json_payload(raw)
        self.assertIn("Line 1 insight summary.", result["description"])
        self.assertIn("#shorts #podcast", result["description"])

    def test_extract_json_payload_trailing_commas(self):
        """Cleans trailing commas before closing braces."""
        raw = """{
  "title_1": "Trailing Comma Title #shorts",
  "description": "Description here.",
  "tags": ["tag1", "tag2",],
}"""
        result = metadata_ai.extract_json_payload(raw)
        self.assertEqual(result["title_1"], "Trailing Comma Title #shorts")

    @patch("metadata_ai.OpenAI")
    def test_successful_ai_metadata_response(self, mock_openai_cls):
        """Normal production behavior: returns AI title variants, description, tags, and punchline."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        ai_payload = {
            "title_1": "Discipline Beats Motivation Every Time #shorts",
            "title_2": "Why Motivation Is A Lie #shorts",
            "title_3": "How To Build Daily Habits #shorts",
            "punchline": "Discipline Beats Motivation 💡",
            "description": "In this clip, the speaker breaks down why relying on motivation fails.\n\n#discipline #habits #success #mindset",
            "tags": ["discipline", "habits", "success", "podcast"],
        }
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content=json.dumps(ai_payload)))]
        mock_client.chat.completions.create.return_value = mock_resp

        with patch.object(config, "GROQ_API_KEY", "gsk_test_key_123"):
            res = metadata_ai.generate_shorts_metadata(
                "podcast_episode_1 Part 1",
                transcript="Discipline is what keeps you going when motivation fades away.",
            )

        self.assertEqual(res["title"], "Discipline Beats Motivation Every Time #shorts")
        self.assertEqual(len(res["title_variants"]), 3)
        self.assertIn("In this clip, the speaker breaks down", res["description"])
        self.assertNotIn("podcast_episode_1", res["description"])
        self.assertEqual(res["tags"], ["discipline", "habits", "success", "podcast"])
        self.assertIn("punchline", res)

    @patch("metadata_ai.OpenAI")
    def test_description_comes_from_transcript_on_successful_ai_generation(self, mock_openai_cls):
        """Verifies that the description is derived from spoken transcript and NOT the source filename."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        ai_payload = {
            "title_1": "The Harsh Reality of Starting a Business #shorts",
            "title_2": "Are You Ready to Fail? #shorts",
            "title_3": "Business Truth Nobody Tells You #shorts",
            "punchline": "Starting A Business Is Hard 🔥",
            "description": "Founder reveals why 90% of startups fail within the first two years and how to survive.",
            "tags": ["startup", "business", "entrepreneurship"],
        }
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content=json.dumps(ai_payload)))]
        mock_client.chat.completions.create.return_value = mock_resp

        with patch.object(config, "GROQ_API_KEY", "gsk_test_key_123"):
            res = metadata_ai.generate_shorts_metadata(
                "my_raw_video_download_1080p Part 2",
                transcript="When we launched the company, we had zero revenue for eight months.",
            )

        # Description MUST contain the AI generated summary, NOT filename
        self.assertEqual(res["description"], "Founder reveals why 90% of startups fail within the first two years and how to survive.")
        self.assertNotIn("my_raw_video_download", res["description"])
        self.assertNotIn("Part 2", res["description"])

    @patch("metadata_ai.log.warning")
    def test_missing_api_key_fallback(self, mock_log_warn):
        """When neither GROQ_API_KEY nor OPENAI_API_KEY is configured, uses fallback and logs clearly."""
        with patch.object(config, "GROQ_API_KEY", ""), patch.object(config, "OPENAI_API_KEY", ""):
            res = metadata_ai.generate_shorts_metadata(
                "interview_clip_vidssave.com_720p.mp4",
                transcript="Some spoken words here.",
            )

        self.assertIn("interview clip", res["title"])
        self.assertEqual(res["description"], "interview clip\n\n#shorts #podcast #viral")
        # Ensure clear logging of missing API key
        mock_log_warn.assert_called()
        log_msg = mock_log_warn.call_args[0][0]
        self.assertIn("[METADATA_MISSING_API_KEY]", log_msg)

    @patch("metadata_ai.OpenAI")
    @patch("metadata_ai.log.warning")
    def test_malformed_json_fallback_and_logging(self, mock_log_warn, mock_openai_cls):
        """When the LLM returns invalid/malformed non-JSON text, retries, logs error, and uses fallback."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        # Returns non-JSON text
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content="I am sorry, but I cannot generate metadata."))]
        mock_client.chat.completions.create.return_value = mock_resp

        with patch.object(config, "GROQ_API_KEY", "gsk_test_key_123"):
            res = metadata_ai.generate_shorts_metadata(
                "finance_talk Part 1",
                transcript="Investing in index funds is historically the safest strategy.",
            )

        # Safe fallback returned
        self.assertIn("finance talk Part 1", res["title"])
        self.assertEqual(res["description"], "finance talk Part 1\n\n#shorts #podcast #viral")

        # Check that fallback and error were clearly logged
        warnings = [call[0][0] for call in mock_log_warn.call_args_list]
        self.assertTrue(any("[METADATA_FALLBACK_USED]" in w for w in warnings))

    @patch("metadata_ai.OpenAI")
    @patch("metadata_ai.log.warning")
    def test_api_failure_triggers_retry_and_fallback(self, mock_log_warn, mock_openai_cls):
        """When the API throws an exception (e.g. rate limit, 503, timeout), retries and falls back safely."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = RuntimeError("503 Service Unavailable")

        with patch.object(config, "GROQ_API_KEY", "gsk_test_key_123"), patch("time.sleep"):
            res = metadata_ai.generate_shorts_metadata(
                "tech_podcast Part 3",
                transcript="Quantum computing will revolutionize cryptography.",
            )

        # Retries attempted 3 times
        self.assertEqual(mock_client.chat.completions.create.call_count, 3)
        self.assertEqual(res["description"], "tech podcast Part 3\n\n#shorts #podcast #viral")

        warnings = [call[0][0] for call in mock_log_warn.call_args_list]
        self.assertTrue(any("[METADATA_API_ERROR]" in w for w in warnings))
        self.assertTrue(any("[METADATA_FALLBACK_USED]" in w for w in warnings))

    @patch("metadata_ai.OpenAI")
    def test_generated_metadata_compatible_with_upload_flow(self, mock_openai_cls):
        """Verifies that generated metadata matches the exact schema expected by youtube_upload.py."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        ai_payload = {
            "title_1": "AI Revolution In 2026 #shorts",
            "title_2": "How AI Changes Coding #shorts",
            "title_3": "Future of Software Engineering #shorts",
            "punchline": "AI Changes Everything 🤖",
            "description": "A deep dive into how AI automation reshapes full-stack development.\n\n#ai #coding #tech",
            "tags": ["ai", "coding", "software"],
        }
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content=json.dumps(ai_payload)))]
        mock_client.chat.completions.create.return_value = mock_resp

        with patch.object(config, "GROQ_API_KEY", "gsk_test_key_123"):
            res = metadata_ai.generate_shorts_metadata("future_code.mp4", transcript="AI is evolving faster than ever.")

        # Required fields for main.py and youtube_upload.py
        self.assertIn("title", res)
        self.assertIn("title_variants", res)
        self.assertIn("description", res)
        self.assertIn("tags", res)
        self.assertIn("punchline", res)
        self.assertIn("generated_hook", res)

        self.assertIsInstance(res["title"], str)
        self.assertIsInstance(res["title_variants"], list)
        self.assertIsInstance(res["description"], str)
        self.assertIsInstance(res["tags"], list)
        self.assertTrue(len(res["title"]) <= 100)
        self.assertTrue(res["title"].lower().endswith("#shorts"))


if __name__ == "__main__":
    unittest.main()
