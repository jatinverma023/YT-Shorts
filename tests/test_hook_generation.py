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


class TestGroundedClipHookGeneration(unittest.TestCase):
    """
    Focused regression test suite for Clip-Only Grounded AI Hook Generation:
    1. Specific factual hook generated from clip transcript.
    2. Question hook grounded in clip.
    3. Generic "WAIT FOR THE TWIST" rejected.
    4. Generic "THE TRUTH EXPOSED" rejected.
    5. Source filename cannot influence hook.
    6. Source title cannot influence hook.
    7. Hook supported by exact clip text passes.
    8. Unsupported factual claim is rejected.
    9. Supporting text not present in clip is rejected.
    10. Outside/full-transcript information cannot be used.
    11. Hindi/Hinglish clip produces natural Roman Hindi/Hinglish hook.
    12. English clip produces English hook.
    13. Generic hook cannot outrank a specific grounded hook.
    14. Hook remains within existing length constraints.
    15. Existing generated_hook / punchline compatibility remains intact.
    """

    def test_1_specific_factual_hook_from_clip_transcript(self):
        """A specific factual hook supported by clip transcript passes validation."""
        clip_transcript = "Smoking is associated with lower fertility in both men and women. We see this in clinical studies."
        cand = {
            "hook": "SMOKING CAN AFFECT FERTILITY",
            "hook_type": "surprising_fact",
            "supporting_text": "Smoking is associated with lower fertility",
            "supported_by_clip": True,
            "specificity_score": 9.0,
            "grounding_score": 10.0,
        }
        is_valid, reason = hook_generator.validate_hook(cand, transcript=clip_transcript)
        self.assertTrue(is_valid, f"Expected valid, got: {reason}")
        score = hook_generator.score_hook(cand, transcript=clip_transcript)
        self.assertGreaterEqual(score, 80.0)

    def test_2_question_hook_grounded_in_clip(self):
        """A specific question hook grounded in the clip passes validation."""
        clip_transcript = "Can oral sex actually cause throat cancer? Doctors are now confirming the direct link with HPV."
        cand = {
            "hook": "CAN ORAL SEX CAUSE THROAT CANCER?",
            "hook_type": "question",
            "supporting_text": "Can oral sex actually cause throat cancer",
            "supported_by_clip": True,
            "specificity_score": 9.5,
            "grounding_score": 10.0,
        }
        is_valid, reason = hook_generator.validate_hook(cand, transcript=clip_transcript)
        self.assertTrue(is_valid, f"Expected valid question hook, got: {reason}")
        self.assertEqual(cand["hook_type"], "question")

    def test_3_generic_wait_for_the_twist_rejected(self):
        """Generic clickbait hook 'WAIT FOR THE TWIST' is rejected."""
        clip_transcript = "Studies on neuroplasticity demonstrate that daily meditation improves attention span."
        cand = {
            "hook": "WAIT FOR THE TWIST 🤯",
            "hook_type": "curiosity",
            "supported_by_clip": True,
        }
        is_valid, reason = hook_generator.validate_hook(cand, transcript=clip_transcript)
        self.assertFalse(is_valid)
        self.assertTrue("generic" in reason.lower() or "clickbait" in reason.lower())

    def test_4_generic_the_truth_exposed_rejected(self):
        """Generic clickbait hook 'THE TRUTH EXPOSED' is rejected."""
        clip_transcript = "Studies on neuroplasticity demonstrate that daily meditation improves attention span."
        cand = {
            "hook": "THE TRUTH EXPOSED ⚠️",
            "hook_type": "curiosity",
            "supported_by_clip": True,
        }
        is_valid, reason = hook_generator.validate_hook(cand, transcript=clip_transcript)
        self.assertFalse(is_valid)
        self.assertTrue("generic" in reason.lower() or "clickbait" in reason.lower())

    def test_5_source_filename_cannot_influence_hook(self):
        """Hooks copying or leaking the source video filename are strictly rejected."""
        cand = {
            "hook": "Dr Pal on Gut Health Ep12",
            "supported_by_clip": True,
        }
        is_valid, reason = hook_generator.validate_hook(
            cand,
            transcript="We discuss the gut microbiome and fiber intake.",
            filename="Dr_Pal_on_Gut_Health_Ep12.mp4",
        )
        self.assertFalse(is_valid)
        self.assertIn("copies source filename", reason)

    def test_6_source_title_cannot_influence_hook(self):
        """Hooks copying or leaking the source video title are strictly rejected."""
        cand = {
            "hook": "Deep Life Interview Series",
            "supported_by_clip": True,
        }
        is_valid, reason = hook_generator.validate_hook(
            cand,
            transcript="Today we talk about focus and time management.",
            source_title="Deep Life Interview Series - Full Conversation",
        )
        self.assertFalse(is_valid)
        self.assertIn("copies source filename or source title", reason)

    def test_7_hook_supported_by_exact_clip_text_passes(self):
        """Hook whose supporting text matches the exact clip transcript passes."""
        clip_transcript = "HPV virus can easily reach the throat tissues and cause cellular changes."
        cand = {
            "hook": "HPV CAN REACH THE THROAT",
            "hook_type": "strong_claim",
            "supporting_text": "HPV virus can easily reach the throat",
            "supported_by_clip": True,
        }
        is_valid, reason = hook_generator.validate_hook(cand, transcript=clip_transcript)
        self.assertTrue(is_valid, f"Expected valid, got: {reason}")

    def test_8_unsupported_factual_claim_rejected(self):
        """Hook making an unsupported factual claim absent from the clip is rejected."""
        clip_transcript = "Smoking is associated with lower fertility in both men and women."
        cand = {
            "hook": "SMOKING DESTROYS YOUR DNA",
            "hook_type": "strong_claim",
            "supporting_text": "smoking destroys your dna",
            "supported_by_clip": True,
        }
        is_valid, reason = hook_generator.validate_hook(cand, transcript=clip_transcript)
        self.assertFalse(is_valid)
        self.assertIn("Supporting text was not found", reason)

    def test_9_supporting_text_not_present_in_clip_rejected(self):
        """Candidate with invented supporting text not present in clip is rejected."""
        clip_transcript = "Artificial intelligence models are optimizing compiler execution."
        cand = {
            "hook": "ALIENS BUILT THE PYRAMIDS",
            "hook_type": "curiosity",
            "supporting_text": "aliens built the ancient pyramids",
            "supported_by_clip": True,
        }
        is_valid, reason = hook_generator.validate_hook(cand, transcript=clip_transcript)
        self.assertFalse(is_valid)
        self.assertIn("Supporting text was not found", reason)

    def test_10_outside_full_transcript_information_cannot_be_used(self):
        """
        Information from outside the selected clip slice (e.g. from elsewhere in the full video)
        cannot support the hook candidate.
        """
        # Selected clip is ONLY about deep sleep
        clip_transcript = "Deep sleep occurs primarily during the first third of the night."
        # Candidate referencing topic from outside the clip (e.g. real estate from later in video)
        cand = {
            "hook": "REAL ESTATE PRICES CRASHED",
            "hook_type": "revelation",
            "supporting_text": "real estate prices crashed in metro cities",
            "supported_by_clip": True,
        }
        is_valid, reason = hook_generator.validate_hook(cand, transcript=clip_transcript)
        self.assertFalse(is_valid)
        self.assertIn("Supporting text was not found", reason)

    def test_11_hindi_hinglish_clip_produces_natural_roman_hindi(self):
        """Hindi/Hinglish clip produces natural Roman Hindi/Hinglish hook."""
        clip_transcript = "Smoking se sperm quality par bahut bura asar padta hai aur fertility kam hoti hai."
        cand = {
            "hook": "KYA SMOKING SE FERTILITY GHAT TI HAI?",
            "hook_type": "question",
            "supporting_text": "Smoking se sperm quality par bahut bura asar padta hai",
            "supported_by_clip": True,
            "curiosity_score": 9.0,
            "specificity_score": 9.0,
            "grounding_score": 10.0,
        }
        is_valid, reason = hook_generator.validate_hook(cand, transcript=clip_transcript)
        self.assertTrue(is_valid, f"Expected valid Roman Hindi hook, got: {reason}")
        self.assertEqual(cand["hook"], "KYA SMOKING SE FERTILITY GHAT TI HAI?")

    def test_12_english_clip_produces_english_hook(self):
        """English clip produces natural English hook."""
        clip_transcript = "Nicotine constricts the blood vessels and significantly reduces ovarian blood flow."
        cand = {
            "hook": "WHY SMOKING REDUCES FERTILITY",
            "hook_type": "explanation",
            "supporting_text": "reduces ovarian blood flow",
            "supported_by_clip": True,
            "curiosity_score": 9.0,
            "specificity_score": 9.0,
            "grounding_score": 10.0,
        }
        is_valid, reason = hook_generator.validate_hook(cand, transcript=clip_transcript)
        self.assertTrue(is_valid, f"Expected valid English hook, got: {reason}")

    def test_13_generic_hook_cannot_outrank_specific_grounded_hook(self):
        """A generic dramatic hook cannot beat a specific grounded hook under the new scoring weights."""
        clip_transcript = "Clinical studies prove that smoking lowers sperm quality significantly."

        # Generic dramatic hook with high curiosity but low grounding & specificity
        generic_cand = {
            "hook": "YOU WON'T BELIEVE WHAT HAPPENED",
            "curiosity_score": 10.0,
            "grounding_score": 2.0,
            "specificity_score": 2.0,
            "relevance_score": 4.0,
            "clarity_score": 7.0,
            "brevity_score": 10.0,
        }

        # Specific grounded hook
        grounded_cand = {
            "hook": "THIS CAN LOWER SPERM QUALITY",
            "supporting_text": "smoking lowers sperm quality significantly",
            "curiosity_score": 8.5,
            "grounding_score": 10.0,
            "specificity_score": 9.0,
            "relevance_score": 9.0,
            "clarity_score": 9.5,
            "brevity_score": 9.0,
        }

        score_generic = hook_generator.score_hook(generic_cand, transcript=clip_transcript)
        score_grounded = hook_generator.score_hook(grounded_cand, transcript=clip_transcript)

        # Grounded candidate MUST outscore the generic dramatic candidate
        self.assertGreater(score_grounded, score_generic)
        self.assertGreaterEqual(score_grounded, 85.0)
        self.assertLess(score_generic, 60.0)

    def test_14_hook_remains_within_length_constraints(self):
        """Hook strictly satisfies word count (2-8 words) and character count (<= 42 chars)."""
        # Exactly 42 characters and 8 words: valid
        valid_edge = {
            "hook": "ONE TWO THREE FOUR FIVE SIX SEVEN EIGHT",  # 39 chars, 8 words
            "supported_by_clip": True,
        }
        self.assertTrue(hook_generator.validate_hook(valid_edge)[0])

        # 43 characters: rejected
        cand_43 = {
            "hook": "1234567890123456789012345678901234567890123",  # 43 chars
            "supported_by_clip": True,
        }
        is_val, reason = hook_generator.validate_hook(cand_43)
        self.assertFalse(is_val)
        self.assertIn("exceeds hard limit", reason)

        # 9 words: rejected (under 42 chars)
        cand_9_words = {
            "hook": "a b c d e f g h i",
            "supported_by_clip": True,
        }
        is_val, reason = hook_generator.validate_hook(cand_9_words)
        self.assertFalse(is_val)
        self.assertIn("exceeds maximum word count", reason)

    def test_15_compatibility_with_generated_hook_and_punchline(self):
        """Ensures generate_short_hook return schema maintains full backward compatibility."""
        clip_transcript = "Sleep deprivation directly increases cortisol production and impairs glucose metabolism."
        cand = {
            "hook": "HOW SLEEP DEPRIVATION RAISES CORTISOL",
            "hook_type": "explanation",
            "supporting_text": "increases cortisol production",
            "supported_by_clip": True,
            "curiosity_score": 8.5,
            "specificity_score": 9.0,
            "grounding_score": 9.5,
        }

        with patch("hook_generator.generate_hook_candidates", return_value=[cand]):
            with patch("hook_generator.GROQ_API_KEY", "gsk_mock"):
                res = hook_generator.generate_short_hook(
                    transcript=clip_transcript,
                    filename="health_podcast.mp4",
                    detected_lang="en",
                )
                self.assertIn("selected_hook", res)
                self.assertIn("generated_hook", res)
                self.assertEqual(res["selected_hook"], "HOW SLEEP DEPRIVATION RAISES CORTISOL")
                self.assertEqual(res["generated_hook"], res["selected_hook"])
                self.assertEqual(res["hook_type"], "explanation")
                self.assertEqual(res["source"], "ai")


if __name__ == "__main__":
    unittest.main()

