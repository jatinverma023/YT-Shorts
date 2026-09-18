"""
Unit tests for Hook Upgrade #1: AI-Generated Short-Form Hooks.
Verifies:
1. Valid short hook accepted (3–8 words, <= 42 chars)
2. >42 characters rejected
3. >8 words rejected
4. Duplicate removal (case-insensitive & whitespace normalized)
5. Empty / whitespace hook rejected
6. Emoji limit (0-1 accepted, >1 rejected)
7. Source filename rejected
8. Source title rejected
9. English hook preserved
10. Roman Hindi hook preserved
11. Hinglish preserved
12. Exact Python scoring formula calculation
13. Highest valid candidate selected
14. Fallback behavior when AI fails or all candidates are invalid
15. supported_by_clip requirement (must be true)
16. ASS-safe text (no braces, no newlines/tabs)
"""
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

import hook_generator


class TestHookGeneration(unittest.TestCase):

    def test_1_valid_short_hook(self):
        """Valid short hook (3–8 words, <= 42 chars) is accepted."""
        cand = {
            "hook": "THE REAL TRUTH REVEALED 🔥",
            "hook_type": "revelation",
            "supported_by_clip": True,
        }
        is_valid, reason = hook_generator.validate_hook(cand, transcript="Here is the real truth revealed.")
        self.assertTrue(is_valid, f"Expected valid, got: {reason}")
        self.assertLessEqual(len(cand["hook"]), 42)

    def test_2_greater_than_42_characters_rejected(self):
        """Hook exceeding 42 characters is strictly rejected."""
        cand = {
            "hook": "This is an extremely long hook exceeding forty two chars",
            "hook_type": "curiosity",
            "supported_by_clip": True,
        }
        self.assertGreater(len(cand["hook"]), 42)
        is_valid, reason = hook_generator.validate_hook(cand)
        self.assertFalse(is_valid)
        self.assertIn("exceeds hard limit", reason)

    def test_3_greater_than_8_words_rejected(self):
        """Hook with more than 8 words is strictly rejected even if <= 42 chars."""
        # 9 words: "one two three four five six seven eight nine" (44 chars) -> let's make short words:
        cand = {
            "hook": "a b c d e f g h i",  # 9 words, 17 chars
            "hook_type": "curiosity",
            "supported_by_clip": True,
        }
        is_valid, reason = hook_generator.validate_hook(cand)
        self.assertFalse(is_valid)
        self.assertIn("exceeds maximum word count", reason)

    def test_4_duplicate_removal(self):
        """Duplicate hooks differing only by casing/whitespace are deduplicated."""
        raw_candidates = [
            {"hook": "WHY DOES THIS MATTER?", "hook_type": "question", "curiosity": 8.0, "supported_by_clip": True},
            {"hook": "why does this matter?", "hook_type": "question", "curiosity": 7.0, "supported_by_clip": True},
            {"hook": "  Why  Does   This  Matter?  ", "hook_type": "question", "curiosity": 6.0, "supported_by_clip": True},
            {"hook": "THE REAL SECRET 🔥", "hook_type": "revelation", "curiosity": 9.0, "supported_by_clip": True},
        ]
        with patch("hook_generator.generate_hook_candidates", return_value=raw_candidates):
            with patch("hook_generator.GROQ_API_KEY", "gsk_test"):
                res = hook_generator.generate_short_hook("dummy transcript")
                self.assertEqual(len(res["candidates"]), 2)
                hooks = [c["hook"].lower() for c in res["candidates"]]
                self.assertIn("why does this matter?", hooks)
                self.assertIn("the real secret 🔥", hooks)

    def test_5_empty_hook_rejected(self):
        """Empty or whitespace-only hooks are rejected."""
        for empty_text in ["", "   ", None]:
            cand = {"hook": empty_text, "supported_by_clip": True}
            is_valid, reason = hook_generator.validate_hook(cand)
            self.assertFalse(is_valid)

    def test_6_emoji_limit(self):
        """0 or 1 emoji is accepted; 2 or more emojis are rejected."""
        # 0 emoji -> valid
        valid_0 = {"hook": "WHY DOES THIS MATTER?", "supported_by_clip": True}
        self.assertTrue(hook_generator.validate_hook(valid_0)[0])

        # 1 emoji -> valid
        valid_1 = {"hook": "WHY DOES THIS MATTER? 👀", "supported_by_clip": True}
        self.assertTrue(hook_generator.validate_hook(valid_1)[0])

        # 2 emojis -> rejected
        invalid_2 = {"hook": "WHY DOES THIS MATTER? 👀🔥", "supported_by_clip": True}
        is_valid_2, reason_2 = hook_generator.validate_hook(invalid_2)
        self.assertFalse(is_valid_2)
        self.assertIn("multiple emojis", reason_2)

    def test_7_source_filename_rejected(self):
        """Hooks containing or matching source filename are rejected."""
        cand = {
            "hook": "Khan Sir with Raj Shamani",
            "supported_by_clip": True,
        }
        is_valid, reason = hook_generator.validate_hook(
            cand,
            filename="Khan_Sir_with_Raj_Shamani.mp4",
        )
        self.assertFalse(is_valid)
        self.assertIn("copies source filename", reason)

    def test_8_source_title_rejected(self):
        """Hooks containing or matching source video title are rejected."""
        cand = {
            "hook": "Khan Sir with Raj Shamani",
            "supported_by_clip": True,
        }
        is_valid, reason = hook_generator.validate_hook(
            cand,
            source_title="Khan Sir with Raj Shamani - Full Interview",
        )
        self.assertFalse(is_valid)
        self.assertIn("copies source filename or source title", reason)

    def test_9_english_hook_preserved(self):
        """English clips produce natural English hooks without translation or Devanagari."""
        cand = {
            "hook": "HOW AI IS CHANGING JOBS 🔥",
            "hook_type": "insight",
            "supported_by_clip": True,
        }
        is_valid, _ = hook_generator.validate_hook(cand, transcript="Artificial intelligence is changing the future of jobs.")
        self.assertTrue(is_valid)
        self.assertEqual(cand["hook"], "HOW AI IS CHANGING JOBS 🔥")

    def test_10_roman_hindi_hook_preserved(self):
        """Hindi speech produces natural Roman Hindi hooks."""
        cand = {
            "hook": "YE BAAT KYUN IMPORTANT HAI? 👀",
            "hook_type": "question",
            "supported_by_clip": True,
        }
        is_valid, _ = hook_generator.validate_hook(cand, transcript="Ye baat samajhna sabke liye bahut zaroori hai.")
        self.assertTrue(is_valid)
        self.assertEqual(cand["hook"], "YE BAAT KYUN IMPORTANT HAI? 👀")

    def test_11_hinglish_preserved(self):
        """Hinglish hooks preserve natural English technical terms."""
        cand = {
            "hook": "STARTUP KA SECRET KYA HAI? 💡",
            "hook_type": "curiosity",
            "supported_by_clip": True,
        }
        is_valid, _ = hook_generator.validate_hook(cand, transcript="Startup grow karne ka sabse bada secret kya hai.")
        self.assertTrue(is_valid)
        self.assertIn("STARTUP", cand["hook"])

    def test_12_exact_python_scoring_formula(self):
        """Authoritative Python scoring computes exact weighted formula:
        (curiosity*0.25 + relevance*0.25 + payoff*0.15 + clarity*0.15 + brevity*0.10 + impact*0.10) * 10
        """
        cand = {
            "hook": "THE BIG QUESTION",
            "curiosity": 8.0,
            "relevance": 9.0,
            "payoff": 7.0,
            "clarity": 10.0,
            "brevity": 8.0,
            "impact": 6.0,
        }
        # Expected:
        # 8.0*0.25 = 2.0
        # 9.0*0.25 = 2.25
        # 7.0*0.15 = 1.05
        # 10.0*0.15 = 1.50
        # 8.0*0.10 = 0.80
        # 6.0*0.10 = 0.60
        # Sum = 8.20 -> * 10 = 82.0
        score = hook_generator.score_hook(cand)
        self.assertAlmostEqual(score, 82.0, places=2)

    def test_13_highest_valid_candidate_selected(self):
        """The highest-scoring valid candidate is selected."""
        candidates = [
            {"hook": "GOOD HOOK HERE", "hook_type": "curiosity", "curiosity": 6.0, "relevance": 6.0, "payoff": 6.0, "clarity": 6.0, "brevity": 6.0, "impact": 6.0, "supported_by_clip": True},
            {"hook": "BEST HOOK EVER 🔥", "hook_type": "revelation", "curiosity": 10.0, "relevance": 10.0, "payoff": 10.0, "clarity": 10.0, "brevity": 10.0, "impact": 10.0, "supported_by_clip": True},
            {"hook": "AVERAGE HOOK", "hook_type": "question", "curiosity": 7.0, "relevance": 7.0, "payoff": 7.0, "clarity": 7.0, "brevity": 7.0, "impact": 7.0, "supported_by_clip": True},
        ]
        with patch("hook_generator.generate_hook_candidates", return_value=candidates):
            with patch("hook_generator.GROQ_API_KEY", "gsk_test"):
                res = hook_generator.generate_short_hook("dummy transcript")
                self.assertEqual(res["selected_hook"], "BEST HOOK EVER 🔥")
                self.assertEqual(res["score"], 100.0)
                self.assertEqual(res["source"], "ai")

    def test_14_fallback_behavior(self):
        """When LLM call fails or returns no valid candidates, safe fallback is returned."""
        # Case A: No API key
        with patch("hook_generator.GROQ_API_KEY", ""):
            with patch("hook_generator.OPENAI_API_KEY", ""):
                res_en = hook_generator.generate_short_hook("dummy transcript", detected_lang="en")
                self.assertEqual(res_en["source"], "fallback")
                self.assertIn(res_en["selected_hook"], ["WHY DOES THIS MATTER?", "WHAT DOES THIS MEAN?", "THE KEY POINT"])

                res_hi = hook_generator.generate_short_hook("dummy transcript", detected_lang="hi")
                self.assertEqual(res_hi["source"], "fallback")
                self.assertIn(res_hi["selected_hook"], ["YE BAAT KYUN IMPORTANT HAI?", "ASLI BAAT KYA HAI?", "ISKA MATLAB KYA HAI?"])

        # Case B: All candidates invalid (>42 chars)
        invalid_candidates = [
            {"hook": "This hook is definitely way too long to ever fit in forty two chars", "supported_by_clip": True}
        ]
        with patch("hook_generator.generate_hook_candidates", return_value=invalid_candidates):
            with patch("hook_generator.GROQ_API_KEY", "gsk_test"):
                res_invalid = hook_generator.generate_short_hook("dummy transcript", detected_lang="en")
                self.assertEqual(res_invalid["source"], "fallback")

    def test_15_supported_by_clip_requirement(self):
        """Candidates with supported_by_clip == False are strictly rejected."""
        cand_unsupported = {
            "hook": "THE SECRET MONEY PLOT",
            "supported_by_clip": False,
            "support_reason": "Not discussed in this clip slice",
        }
        is_valid, reason = hook_generator.validate_hook(cand_unsupported, transcript="Talking about coding")
        self.assertFalse(is_valid)
        self.assertIn("not supported by the clip", reason)

    def test_16_ass_safe_text(self):
        """ASS control tags, curly braces, and newlines/tabs are strictly rejected."""
        unsafe_cases = [
            {"hook": "HELLO {\\b1}WORLD", "supported_by_clip": True},
            {"hook": "LINE ONE\\NLINE TWO", "supported_by_clip": True},
            {"hook": "LINE ONE\nLINE TWO", "supported_by_clip": True},
            {"hook": "TABBED\tTEXT", "supported_by_clip": True},
        ]
        for cand in unsafe_cases:
            is_valid, reason = hook_generator.validate_hook(cand)
            self.assertFalse(is_valid, f"Should reject ASS unsafe hook: {cand['hook']}")
            self.assertIn("ASS control syntax", reason)


if __name__ == "__main__":
    unittest.main()
