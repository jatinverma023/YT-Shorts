"""
Tests for Phase 6: Simplify and Harden Hook Generation.
Validates:
1. Streamlined hook schema (hook, hook_type, supporting_text) is parsed and scored authoritatively in Python.
2. Deterministic fallback cannot fabricate claims (extreme claims not supported by transcript are rejected).
3. Incomplete/open-ended clauses ending in prepositions/conjunctions are not selected as fallback hooks.
4. Semantic emoji behavior (proper relevance, no chains, no duplicates).
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

import config
import hook_generator


class TestPhase6HookGeneration(unittest.TestCase):

    def setUp(self):
        from ai_rate_limiter import shared_rate_limiter
        shared_rate_limiter._history.clear()
        self.sample_transcript = (
            "Dopamine is not the molecule of pleasure. It is the molecule of motivation, craving, and pursuit. "
            "When you understand dopamine peaks and baselines, you can maintain high energy and discipline."
        )

    def test_streamlined_hook_schema_parsed_and_scored(self):
        """Validates that streamlined LLM hook schema without redundant score fields is scored in Python."""
        streamlined_candidates = {
            "candidates": [
                {
                    "hook": "WHY DOPAMINE CONTROLS MOTIVATION 🧠",
                    "hook_type": "specific_fact",
                    "supporting_text": "Dopamine is not the molecule of pleasure. It is the molecule of motivation"
                },
                {
                    "hook": "THE TRUTH ABOUT DOPAMINE PEAKS",
                    "hook_type": "curiosity",
                    "supporting_text": "understand dopamine peaks and baselines"
                }
            ]
        }

        mock_choice = MagicMock()
        mock_choice.finish_reason = "stop"
        mock_choice.message.content = str(streamlined_candidates).replace("'", '"')
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        cands = hook_generator.generate_hook_candidates(
            client=mock_client,
            model="openai/gpt-oss-20b",
            transcript=self.sample_transcript,
        )

        self.assertEqual(len(cands), 2)
        score1 = hook_generator.score_hook(cands[0], transcript=self.sample_transcript)
        self.assertGreaterEqual(score1, 70.0)

    def test_deterministic_fallback_cannot_fabricate_claims(self):
        """Proves that extract_grounded_fallback_hook will reject clauses containing unsupported extreme claims."""
        # Transcript mentions studying cancer, but does not claim guaranteed 100% cure
        transcript = (
            "Researchers are carefully studying cell biology. "
            "This therapy is definitely a guaranteed 100% cure for cancer. "
            "Early laboratory observations remain preliminary."
        )
        # Even though "This therapy is definitely a guaranteed 100% cure for cancer" is in the text,
        # it has extreme claim words without hedging or support
        hook, supporting = hook_generator.extract_grounded_fallback_hook(transcript)
        # Fallback hook must NOT be the fabricated claim
        self.assertNotIn("GUARANTEED", hook)
        self.assertNotIn("100%", hook)

    def test_incomplete_open_ended_clauses_rejected(self):
        """Proves that incomplete clauses ending in prepositions or conjunctions are rejected."""
        incomplete_transcript = (
            "And when we went with. Because they were in. "
            "Why does morning sunlight matter? It sets your circadian clock."
        )
        hook, supporting = hook_generator.extract_grounded_fallback_hook(incomplete_transcript)
        self.assertTrue(hook.endswith("?") or len(hook.split()) >= 3)
        self.assertFalse(hook.lower().endswith("with"))
        self.assertFalse(hook.lower().endswith("in"))
        self.assertIn("SUNLIGHT", hook)

    def test_semantic_emoji_rules(self):
        """Proves semantic emoji verification enforces relevance and rejects duplicate or chain emojis."""
        finance_transcript = "Investing ten thousand rupees monthly into mutual funds generates compound interest."
        # Valid emoji
        self.assertGreaterEqual(
            hook_generator.evaluate_emoji_relevance(["💰"], transcript=finance_transcript, hook="INVESTING FOR WEALTH 💰"),
            8.0
        )
        # Irrelevant emoji from food category scored lower than matching domain emoji
        self.assertLess(
            hook_generator.evaluate_emoji_relevance(["🥗"], transcript=finance_transcript, hook="INVESTING FOR WEALTH 🥗"),
            8.0
        )
        # Emoji count validation
        is_valid, _ = hook_generator.validate_hook(
            {"hook": "INVESTING FOR WEALTH 💰💰", "hook_type": "specific_fact", "supporting_text": "Investing ten thousand"},
            transcript=finance_transcript,
        )
        # Duplicate emoji is rejected
        self.assertFalse(is_valid)


if __name__ == "__main__":
    unittest.main()
