"""
Unit tests for Fix #4: Standalone / Context Validation.
"""
import os
import sys
import unittest

# Ensure scripts directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from clip_detection import (
    check_opening_risk,
    validate_standalone_context,
    _validate_and_filter_clips,
    _fallback_clips,
    calculate_quality_score,
)
import config


class TestStandaloneValidation(unittest.TestCase):

    def test_1_fully_standalone_clip(self):
        """Test 1: Complete explanation introducing its own subject is accepted (standalone_score >= 7)."""
        candidate = {
            "start_time": 10.0,
            "end_time": 45.0,
            "hook_summary": "OpenAI released a new model that can reason across long documents. The important part is that it can maintain context throughout the entire task.",
            "title_idea": "New AI Reasoning Breakthrough #shorts",
            "punchline": "Context Maintained Throughout 🚀",
            "standalone": True,
            "standalone_score": 9.0,
            "missing_setup": False,
            "missing_payoff": False,
            "critical_unresolved_reference": False,
            "unresolved_references": [],
            "scores": {
                "hook_strength": 9.0,
                "standalone_clarity": 9.0,
                "payoff": 9.0,
                "curiosity": 8.5,
                "impact": 8.0,
                "retention": 8.5,
                "context_dependency": 1.0,
                "punchline_score": 8.5,
            },
        }
        is_valid, reason = validate_standalone_context(candidate)
        self.assertTrue(is_valid, f"Expected valid, got: {reason}")
        results = _validate_and_filter_clips([candidate], total_duration=100.0, min_clip_seconds=20, max_clip_seconds=59, max_clips=5)
        self.assertEqual(len(results), 1)
        self.assertGreaterEqual(results[0]["standalone_score"], 7.0)

    def test_2_unexplained_pronoun(self):
        """Test 2: Unexplained pronoun ('And that's why he decided to leave') without identifying 'he' is rejected."""
        candidate = {
            "start_time": 20.0,
            "end_time": 50.0,
            "hook_summary": "And that's why he decided to leave.",
            "title_idea": "Why He Left #shorts",
            "punchline": "He Left 🚪",
            "standalone": False,
            "standalone_score": 4.0,
            "missing_setup": True,
            "missing_payoff": False,
            "critical_unresolved_reference": True,
            "unresolved_references": ["he"],
            "scores": {
                "hook_strength": 4.0,
                "standalone_clarity": 4.0,
                "payoff": 4.0,
                "curiosity": 4.0,
                "impact": 4.0,
                "retention": 4.0,
                "context_dependency": 8.5,
                "punchline_score": 4.0,
            },
        }
        is_valid, reason = validate_standalone_context(candidate)
        self.assertFalse(is_valid)
        self.assertIn("missing setup", reason.lower())

        results = _validate_and_filter_clips([candidate], total_duration=100.0, min_clip_seconds=20, max_clip_seconds=59, max_clips=5)
        self.assertEqual(len(results), 0)

    def test_3_missing_question_setup(self):
        """Test 3: Missing question/setup ('Yes, absolutely. I think that is the biggest reason.') is rejected."""
        candidate = {
            "start_time": 15.0,
            "end_time": 45.0,
            "hook_summary": "Yes, absolutely. I think that is the biggest reason.",
            "title_idea": "The Biggest Reason #shorts",
            "punchline": "The Real Reason Revealed 💡",
            "standalone": False,
            "standalone_score": 4.5,
            "missing_setup": True,
            "missing_payoff": False,
            "critical_unresolved_reference": False,
            "unresolved_references": [],
            "scores": {
                "hook_strength": 5.0,
                "standalone_clarity": 4.5,
                "payoff": 5.0,
                "curiosity": 5.0,
                "impact": 5.0,
                "retention": 5.0,
                "context_dependency": 7.0,
                "punchline_score": 5.0,
            },
        }
        is_valid, reason = validate_standalone_context(candidate)
        self.assertFalse(is_valid)
        results = _validate_and_filter_clips([candidate], total_duration=100.0, min_clip_seconds=20, max_clip_seconds=59, max_clips=5)
        self.assertEqual(len(results), 0)

    def test_4_missing_payoff(self):
        """Test 4: Missing payoff ('So when he finally opened the box...') cutting before climax is rejected."""
        candidate = {
            "start_time": 30.0,
            "end_time": 60.0,
            "hook_summary": "So when he finally opened the mystery box...",
            "title_idea": "Opening The Mystery Box #shorts",
            "punchline": "Wait for Part 2 📦",
            "standalone": False,
            "standalone_score": 5.0,
            "missing_setup": False,
            "missing_payoff": True,
            "critical_unresolved_reference": False,
            "unresolved_references": [],
            "scores": {
                "hook_strength": 8.0,
                "standalone_clarity": 5.0,
                "payoff": 2.0,
                "curiosity": 8.0,
                "impact": 5.0,
                "retention": 7.0,
                "context_dependency": 3.0,
                "punchline_score": 2.0,
            },
        }
        is_valid, reason = validate_standalone_context(candidate)
        self.assertFalse(is_valid)
        self.assertIn("payoff", reason.lower())
        results = _validate_and_filter_clips([candidate], total_duration=100.0, min_clip_seconds=20, max_clip_seconds=59, max_clips=5)
        self.assertEqual(len(results), 0)

    def test_5_pronoun_with_valid_context(self):
        """Test 5: Pronouns with valid antecedents in clip ('Rahul started the company... He expanded it') are accepted."""
        candidate = {
            "start_time": 10.0,
            "end_time": 45.0,
            "hook_summary": "Rahul started the company in 2019. He later expanded it into three major cities with immense success.",
            "title_idea": "How Rahul Built His Startup #shorts",
            "punchline": "Startup Success Story ✨",
            "standalone": True,
            "standalone_score": 9.0,
            "missing_setup": False,
            "missing_payoff": False,
            "critical_unresolved_reference": False,
            "unresolved_references": [],
            "scores": {
                "hook_strength": 8.5,
                "standalone_clarity": 9.0,
                "payoff": 8.5,
                "curiosity": 8.0,
                "impact": 8.0,
                "retention": 8.5,
                "context_dependency": 1.5,
                "punchline_score": 8.5,
            },
        }
        is_valid, reason = validate_standalone_context(candidate)
        self.assertTrue(is_valid, f"Expected valid, got: {reason}")
        results = _validate_and_filter_clips([candidate], total_duration=100.0, min_clip_seconds=20, max_clip_seconds=59, max_clips=5)
        self.assertEqual(len(results), 1)

    def test_6_strong_clip_with_conversational_wording(self):
        """Test 6: Phrase 'And that's why' alone does NOT cause rejection if candidate is self-contained."""
        candidate = {
            "start_time": 10.0,
            "end_time": 48.0,
            "hook_summary": "And that's why consistency matters more than motivation when you're learning to code.",
            "title_idea": "Consistency Beats Motivation #shorts",
            "punchline": "Consistency Over Motivation 💻",
            "standalone": True,
            "standalone_score": 8.5,
            "missing_setup": False,
            "missing_payoff": False,
            "critical_unresolved_reference": False,
            "unresolved_references": [],
            "scores": {
                "hook_strength": 9.0,
                "standalone_clarity": 8.5,
                "payoff": 8.5,
                "curiosity": 8.0,
                "impact": 8.5,
                "retention": 8.5,
                "context_dependency": 2.0,
                "punchline_score": 8.5,
            },
        }
        # Verify check_opening_risk flags the risk signal
        risk_detected, risk_reason = check_opening_risk(candidate["hook_summary"])
        self.assertTrue(risk_detected)

        # But validate_standalone_context accepts it because it is self-contained without missing setup
        is_valid, reason = validate_standalone_context(candidate)
        self.assertTrue(is_valid, f"Expected conversational opening to pass, but got: {reason}")
        results = _validate_and_filter_clips([candidate], total_duration=100.0, min_clip_seconds=20, max_clip_seconds=59, max_clips=5)
        self.assertEqual(len(results), 1)

    def test_7_context_score_threshold(self):
        """Test 7: Threshold behavior: standalone_score = 6 rejected; standalone_score = 7 accepted."""
        cand_score_6 = {
            "start_time": 10.0,
            "end_time": 40.0,
            "hook_summary": "Interesting point about system design",
            "title_idea": "System Design Tip #shorts",
            "standalone_score": 6.0,
            "missing_setup": False,
            "missing_payoff": False,
            "critical_unresolved_reference": False,
            "scores": {"hook_strength": 8.0, "standalone_clarity": 6.0, "payoff": 8.0, "curiosity": 7.0, "impact": 7.0, "retention": 7.0, "context_dependency": 3.0, "punchline_score": 7.0},
        }
        is_valid_6, _ = validate_standalone_context(cand_score_6, min_standalone_score=7.0)
        self.assertFalse(is_valid_6)

        cand_score_7 = {
            "start_time": 10.0,
            "end_time": 40.0,
            "hook_summary": "Clear principle of distributed systems",
            "title_idea": "Distributed Systems Rule #shorts",
            "standalone_score": 7.0,
            "missing_setup": False,
            "missing_payoff": False,
            "critical_unresolved_reference": False,
            "scores": {"hook_strength": 8.0, "standalone_clarity": 7.0, "payoff": 8.0, "curiosity": 7.0, "impact": 7.0, "retention": 7.0, "context_dependency": 3.0, "punchline_score": 7.0},
        }
        is_valid_7, _ = validate_standalone_context(cand_score_7, min_standalone_score=7.0)
        self.assertTrue(is_valid_7)

    def test_8_existing_fix3_compatibility(self):
        """Test 8: Verify all Fix #3 fields, scoring, and metadata are intact after standalone validation."""
        candidate = {
            "start_time": 15.0,
            "end_time": 50.0,
            "hook_summary": "Explaining the difference between concurrency and parallelism.",
            "title_idea": "Concurrency vs Parallelism #shorts",
            "punchline": "Concurrency Explained ⚡",
            "standalone": True,
            "standalone_score": 9.0,
            "missing_setup": False,
            "missing_payoff": False,
            "critical_unresolved_reference": False,
            "unresolved_references": [],
            "scores": {
                "hook_strength": 8.5,
                "standalone_clarity": 9.0,
                "payoff": 8.5,
                "curiosity": 8.0,
                "impact": 8.0,
                "retention": 8.0,
                "context_dependency": 1.5,
                "punchline_score": 8.0,
            },
        }
        results = _validate_and_filter_clips([candidate], total_duration=100.0, min_clip_seconds=20, max_clip_seconds=59, max_clips=5)
        self.assertEqual(len(results), 1)
        res = results[0]
        # Verify timestamps are NOT modified or expanded
        self.assertEqual(res["start_time"], 15.0)
        self.assertEqual(res["end_time"], 50.0)
        self.assertEqual(res["duration"], 35.0)
        # Verify core fields
        self.assertIn("hook_summary", res)
        self.assertIn("title_idea", res)
        self.assertIn("punchline", res)
        self.assertIn("scores", res)
        self.assertIn("quality_score", res)
        self.assertIn("selection_reason", res)
        # Verify Fix #4 metadata
        self.assertIn("standalone", res)
        self.assertIn("standalone_score", res)
        self.assertIn("missing_setup", res)
        self.assertIn("missing_payoff", res)
        self.assertIn("critical_unresolved_reference", res)

    def test_9_fallback_protection(self):
        """Test 9: Fallback candidates have standalone_score = 0.0 and cannot bypass standalone validation."""
        fallbacks = _fallback_clips(total_duration=120.0, min_clip_seconds=20, max_clip_seconds=59, max_clips=5)
        self.assertTrue(len(fallbacks) > 0)
        for fb in fallbacks:
            self.assertEqual(fb["standalone_score"], 0.0)
            self.assertTrue(fb["missing_setup"])
            self.assertTrue(fb["missing_payoff"])
            self.assertTrue(fb["critical_unresolved_reference"])
            is_valid, _ = validate_standalone_context(fb)
            self.assertFalse(is_valid)

        # Filtering fallbacks through _validate_and_filter_clips rejects them all
        accepted = _validate_and_filter_clips(fallbacks, total_duration=120.0, min_clip_seconds=20, max_clip_seconds=59, max_clips=5)
        self.assertEqual(len(accepted), 0)

    def test_10_pipeline_compatibility(self):
        """Test 10: Verify that accepted candidate structure fits downstream sheet_log and queue contracts."""
        candidate = {
            "start_time": 25.0,
            "end_time": 58.0,
            "hook_summary": "Understanding cache invalidation strategies.",
            "title_idea": "Cache Invalidation Rules #shorts",
            "punchline": "Cache Right 🔥",
            "standalone": True,
            "standalone_score": 8.5,
            "missing_setup": False,
            "missing_payoff": False,
            "critical_unresolved_reference": False,
            "scores": {
                "hook_strength": 8.0,
                "standalone_clarity": 8.5,
                "payoff": 8.0,
                "curiosity": 8.0,
                "impact": 7.5,
                "retention": 8.0,
                "context_dependency": 2.0,
                "punchline_score": 8.0,
            },
        }
        results = _validate_and_filter_clips([candidate], total_duration=100.0, min_clip_seconds=20, max_clip_seconds=59, max_clips=5)
        self.assertEqual(len(results), 1)
        res = results[0]
        # Verify fields match what enqueue_clips expects:
        # float(clip["start_time"]), float(clip["end_time"]), clip.get("hook_summary"), clip.get("punchline"), clip.get("quality_score")
        self.assertIsInstance(float(res["start_time"]), float)
        self.assertIsInstance(float(res["end_time"]), float)
        self.assertIsInstance(res.get("hook_summary"), str)
        self.assertIsInstance(res.get("punchline"), str)
        self.assertIsInstance(float(res.get("quality_score")), float)
        self.assertGreaterEqual(res["quality_score"], 70.0)


if __name__ == "__main__":
    unittest.main()
