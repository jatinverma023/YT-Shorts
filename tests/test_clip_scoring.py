"""
Unit tests for Fix #3: AI clip quality scoring, ranking, deduplication, and pipeline compatibility.
"""
import os
import sys
import unittest

# Ensure scripts directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from clip_detection import (
    calculate_quality_score,
    _validate_and_filter_clips,
    _fallback_clips,
    SCORING_WEIGHTS,
)


class TestClipQualityScoring(unittest.TestCase):

    def test_exact_scoring_formula(self):
        """Verify the exact 8-dimension weighted calculation normalized to 0-100."""
        # 8-dimensions:
        # hook: 20%, standalone: 15%, payoff: 20%, curiosity: 10%,
        # impact: 10%, retention: 10%, context_independence (10 - dep): 10%, punchline: 5%
        scores = {
            "hook_strength": 9.0,
            "standalone_clarity": 10.0,
            "payoff": 9.0,
            "curiosity": 8.0,
            "impact": 8.0,
            "retention": 9.0,
            "context_dependency": 2.0,  # context_indep = 8.0
            "punchline_score": 9.0,
        }
        # Expected:
        # (9.0*0.20 + 10.0*0.15 + 9.0*0.20 + 8.0*0.10 + 8.0*0.10 + 9.0*0.10 + 8.0*0.10 + 9.0*0.05) * 10
        # = (1.80 + 1.50 + 1.80 + 0.80 + 0.80 + 0.90 + 0.80 + 0.45) * 10
        # = 8.85 * 10 = 88.5
        calculated = calculate_quality_score(scores)
        self.assertEqual(calculated, 88.5)

    def test_llm_score_cannot_override_python(self):
        """Verify Python deterministically computes quality_score and overwrites any LLM-supplied score."""
        bogus_llm_candidate = [{
            "start_time": 10.0,
            "end_time": 45.0,
            "hook_summary": "Interesting moment",
            "title_idea": "Test Title #shorts",
            "punchline": "Punchline ✨",
            "scores": {
                "hook_strength": 5.0,
                "standalone_clarity": 5.0,
                "payoff": 5.0,
                "curiosity": 5.0,
                "impact": 5.0,
                "retention": 5.0,
                "context_dependency": 5.0,  # indep = 5.0
                "punchline_score": 5.0,
            },
            "quality_score": 99.9,  # Bogus score from LLM
            "standalone_score": 8.0,
            "standalone": True,
            "selection_reason": "LLM says 99.9",

        }]
        # Deterministic Python score for all 5s is 50.0.
        # With min_quality_score=40.0 so it passes filtering:
        results = _validate_and_filter_clips(
            bogus_llm_candidate,
            total_duration=100.0,
            min_clip_seconds=20,
            max_clip_seconds=59,
            max_clips=5,
            min_quality_score=40.0,
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["quality_score"], 50.0)
        self.assertNotEqual(results[0]["quality_score"], 99.9)

    def test_all_eight_dimensions_validated_and_clamped(self):
        """Verify that negative and overflowing scores are clamped to [0, 10]."""
        extreme_low = {
            "hook_strength": -5.0,
            "standalone_clarity": -10.0,
            "payoff": -1.0,
            "curiosity": 0.0,
            "impact": 0.0,
            "retention": 0.0,
            "context_dependency": 15.0,  # Clamped to 10.0 -> context_indep = 0.0
            "punchline_score": -3.0,
        }
        self.assertEqual(calculate_quality_score(extreme_low), 0.0)

        extreme_high = {
            "hook_strength": 15.0,
            "standalone_clarity": 20.0,
            "payoff": 11.0,
            "curiosity": 10.0,
            "impact": 10.0,
            "retention": 10.0,
            "context_dependency": -5.0,  # Clamped to 0.0 -> context_indep = 10.0
            "punchline_score": 100.0,
        }
        self.assertEqual(calculate_quality_score(extreme_high), 100.0)

    def test_fallback_does_not_bypass_threshold(self):
        """Verify unscored fallback candidates receive 0.0 quality_score and do not pass 70.0 threshold."""
        fallback_clips = _fallback_clips(total_duration=120.0, min_clip_seconds=20, max_clip_seconds=59, max_clips=5)
        self.assertTrue(len(fallback_clips) > 0)
        for clip in fallback_clips:
            self.assertEqual(clip["quality_score"], 0.0)
            self.assertEqual(clip["scores"]["context_dependency"], 10.0)

        # Passing fallback clips through filtering with 70.0 threshold rejects them
        filtered = _validate_and_filter_clips(
            fallback_clips,
            total_duration=120.0,
            min_clip_seconds=20,
            max_clip_seconds=59,
            max_clips=5,
            min_quality_score=70.0,
        )
        self.assertEqual(len(filtered), 0)

    def test_test1_strong_candidate(self):
        """Test 1: Strong candidate (high hook, standalone, payoff, retention) accepted."""
        strong_candidate = [{
            "start_time": 10.0,
            "end_time": 50.0,
            "hook_summary": "Reveals unexpected strategy that transformed everything.",
            "title_idea": "The Secret Strategy Revealed #shorts",
            "punchline": "Game Changer Revealed 🔥",
            "scores": {
                "hook_strength": 9.0,
                "standalone_clarity": 9.0,
                "payoff": 9.0,
                "curiosity": 8.5,
                "impact": 8.0,
                "retention": 8.5,
                "context_dependency": 1.0,  # indep = 9.0
                "punchline_score": 9.0,
            },
        }]
        results = _validate_and_filter_clips(
            strong_candidate,
            total_duration=100.0,
            min_clip_seconds=20,
            max_clip_seconds=59,
            max_clips=5,
            min_quality_score=70.0,
        )
        self.assertEqual(len(results), 1)
        self.assertGreaterEqual(results[0]["quality_score"], 70.0)

    def test_test2_weak_context_dependent_candidate(self):
        """Test 2: Weak context-dependent candidate ('And that's why he was wrong') rejected (< 70)."""
        weak_candidate = [{
            "start_time": 20.0,
            "end_time": 55.0,
            "hook_summary": "And that's why he was completely wrong.",
            "title_idea": "Why He Was Wrong #shorts",
            "punchline": "He was wrong 🛑",
            "scores": {
                "hook_strength": 4.0,
                "standalone_clarity": 2.0,
                "payoff": 4.0,
                "curiosity": 4.0,
                "impact": 3.0,
                "retention": 3.0,
                "context_dependency": 9.0,  # indep = 1.0 (heavily context dependent)
                "punchline_score": 3.0,
            },
        }]
        score = calculate_quality_score(weak_candidate[0]["scores"])
        self.assertLess(score, 70.0)

        results = _validate_and_filter_clips(
            weak_candidate,
            total_duration=100.0,
            min_clip_seconds=20,
            max_clip_seconds=59,
            max_clips=5,
            min_quality_score=70.0,
        )
        self.assertEqual(len(results), 0)

    def test_test3_strong_candidate_with_weak_ending(self):
        """Test 3: Candidate starts strongly but ends before payoff; does not outrank complete candidate."""
        weak_ending_candidate = {
            "start_time": 10.0,
            "end_time": 45.0,
            "hook_summary": "Astonishing problem that nobody solves.",
            "title_idea": "The Unsolved Problem #shorts",
            "punchline": "Wait for part 2",
            "scores": {
                "hook_strength": 9.0,
                "standalone_clarity": 7.0,
                "payoff": 2.0,  # Cuts off abruptly before payoff
                "curiosity": 8.0,
                "impact": 6.0,
                "retention": 7.0,
                "context_dependency": 3.0,
                "punchline_score": 2.0,
            },
        }
        complete_candidate = {
            "start_time": 60.0,
            "end_time": 95.0,
            "hook_summary": "How this paradox actually resolves.",
            "title_idea": "Paradox Explained #shorts",
            "punchline": "Mind Blown 🤯",
            "scores": {
                "hook_strength": 8.0,
                "standalone_clarity": 9.0,
                "payoff": 9.0,  # Strong payoff
                "curiosity": 8.0,
                "impact": 8.0,
                "retention": 8.0,
                "context_dependency": 2.0,
                "punchline_score": 8.0,
            },
        }
        results = _validate_and_filter_clips(
            [weak_ending_candidate, complete_candidate],
            total_duration=150.0,
            min_clip_seconds=20,
            max_clip_seconds=59,
            max_clips=5,
            min_quality_score=60.0,
        )
        # Complete candidate must outrank weak ending candidate
        self.assertGreater(results[0]["quality_score"], results[1]["quality_score"])
        self.assertEqual(results[0]["start_time"], 60.0)

    def test_test4_five_strong_candidates(self):
        """Test 4: 5 strong candidates accepted, ranked highest-to-lowest quality."""
        raw = []
        for i in range(5):
            raw.append({
                "start_time": float(i * 40),
                "end_time": float(i * 40 + 35),
                "hook_summary": f"Unique strong insight #{i}",
                "title_idea": f"Insight #{i} #shorts",
                "punchline": f"Insight #{i} ✨",
                "scores": {
                    "hook_strength": 7.0 + i * 0.5,
                    "standalone_clarity": 7.0 + i * 0.5,
                    "payoff": 7.0 + i * 0.5,
                    "curiosity": 7.0 + i * 0.5,
                    "impact": 7.0 + i * 0.5,
                    "retention": 7.0 + i * 0.5,
                    "context_dependency": 2.0,
                    "punchline_score": 7.0 + i * 0.5,
                },
            })

        results = _validate_and_filter_clips(
            raw,
            total_duration=300.0,
            min_clip_seconds=20,
            max_clip_seconds=59,
            max_clips=5,
            min_quality_score=70.0,
        )
        self.assertEqual(len(results), 5)
        # Verify ranked highest quality first
        scores = [r["quality_score"] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_test5_only_two_strong_candidates(self):
        """Test 5: [91, 86, 61, 54, 48] -> exactly 2 enter queue (does not force 5)."""
        raw = [
            {"start_time": 0.0, "end_time": 30.0, "scores": {"hook_strength": 9.2, "standalone_clarity": 9.2, "payoff": 9.2, "curiosity": 9.0, "impact": 9.0, "retention": 9.0, "context_dependency": 1.0, "punchline_score": 9.0}, "hook_summary": "Topic A"},
            {"start_time": 40.0, "end_time": 70.0, "scores": {"hook_strength": 8.7, "standalone_clarity": 8.7, "payoff": 8.7, "curiosity": 8.5, "impact": 8.5, "retention": 8.5, "context_dependency": 1.5, "punchline_score": 8.5}, "hook_summary": "Topic B"},
            {"start_time": 80.0, "end_time": 110.0, "scores": {"hook_strength": 6.0, "standalone_clarity": 6.0, "payoff": 6.0, "curiosity": 6.0, "impact": 6.0, "retention": 6.0, "context_dependency": 4.0, "punchline_score": 6.0}, "hook_summary": "Topic C"},
            {"start_time": 120.0, "end_time": 150.0, "scores": {"hook_strength": 5.5, "standalone_clarity": 5.5, "payoff": 5.5, "curiosity": 5.0, "impact": 5.0, "retention": 5.0, "context_dependency": 5.0, "punchline_score": 5.0}, "hook_summary": "Topic D"},
            {"start_time": 160.0, "end_time": 190.0, "scores": {"hook_strength": 4.8, "standalone_clarity": 4.8, "payoff": 4.8, "curiosity": 4.5, "impact": 4.5, "retention": 4.5, "context_dependency": 5.0, "punchline_score": 4.5}, "hook_summary": "Topic E"},
        ]
        results = _validate_and_filter_clips(
            raw,
            total_duration=300.0,
            min_clip_seconds=20,
            max_clip_seconds=59,
            max_clips=5,
            min_quality_score=70.0,
        )
        self.assertEqual(len(results), 2)
        self.assertGreaterEqual(results[0]["quality_score"], 70.0)
        self.assertGreaterEqual(results[1]["quality_score"], 70.0)

    def test_test6_overlapping_candidates(self):
        """
        Test 6: Overlapping candidate deduplication.
        Case A: 100-140 (score 90) vs 125-165 (score 82):
                Overlap is 15s out of 40s (37.5% >= 35%) -> Higher retained, lower rejected.
        Case B: 100-140 (score 90) vs 138-178 (score 85):
                Overlap is 2s out of 40s (5.0% < 35%) -> Both retained.
        """
        # Case A: Substantial overlap (15s / 40s = 37.5% >= 35%)
        candidate_a = {
            "start_time": 100.0,
            "end_time": 140.0,
            "hook_summary": "Candidate A hook",
            "scores": {"hook_strength": 9.0, "standalone_clarity": 9.0, "payoff": 9.0, "curiosity": 9.0, "impact": 9.0, "retention": 9.0, "context_dependency": 1.0, "punchline_score": 9.0},
        }
        candidate_b = {
            "start_time": 125.0,
            "end_time": 165.0,
            "hook_summary": "Candidate B hook",
            "scores": {"hook_strength": 8.2, "standalone_clarity": 8.2, "payoff": 8.2, "curiosity": 8.2, "impact": 8.2, "retention": 8.2, "context_dependency": 2.0, "punchline_score": 8.2},
        }
        res_a = _validate_and_filter_clips(
            [candidate_b, candidate_a],  # even if B is returned first by LLM
            total_duration=200.0,
            min_clip_seconds=20,
            max_clip_seconds=59,
            max_clips=5,
            min_quality_score=70.0,
            overlap_threshold=0.35,
        )
        self.assertEqual(len(res_a), 1)
        self.assertEqual(res_a[0]["start_time"], 100.0)
        self.assertEqual(res_a[0]["end_time"], 140.0)

        # Case B: Small overlap (2s / 40s = 5% < 35%)
        candidate_c = {
            "start_time": 138.0,
            "end_time": 178.0,
            "hook_summary": "Candidate C different hook",
            "scores": {"hook_strength": 8.5, "standalone_clarity": 8.5, "payoff": 8.5, "curiosity": 8.5, "impact": 8.5, "retention": 8.5, "context_dependency": 1.5, "punchline_score": 8.5},
        }
        res_b = _validate_and_filter_clips(
            [candidate_a, candidate_c],
            total_duration=200.0,
            min_clip_seconds=20,
            max_clip_seconds=59,
            max_clips=5,
            min_quality_score=70.0,
            overlap_threshold=0.35,
        )
        self.assertEqual(len(res_b), 2)
        self.assertEqual(res_b[0]["start_time"], 100.0)
        self.assertEqual(res_b[1]["start_time"], 138.0)

    def test_downstream_fields_compatibility(self):
        """Verify all existing fields required by downstream pipeline are present."""
        raw = [{
            "start_time": 15.0,
            "end_time": 50.0,
            "hook_summary": "Key insight from discussion",
            "title_idea": "The Key Insight #shorts",
            "punchline": "Insight ✨",
            "scores": {"hook_strength": 8.0, "standalone_clarity": 8.0, "payoff": 8.0, "curiosity": 8.0, "impact": 8.0, "retention": 8.0, "context_dependency": 2.0, "punchline_score": 8.0},
        }]
        results = _validate_and_filter_clips(
            raw,
            total_duration=100.0,
            min_clip_seconds=20,
            max_clip_seconds=59,
            max_clips=5,
            min_quality_score=70.0,
        )
        self.assertEqual(len(results), 1)
        clip = results[0]
        # Core pipeline fields
        self.assertIn("start_time", clip)
        self.assertIn("end_time", clip)
        self.assertIn("duration", clip)
        self.assertIn("hook_summary", clip)
        self.assertIn("title_idea", clip)
        self.assertIn("punchline", clip)
        # Fix #3 metadata fields
        self.assertIn("scores", clip)
        self.assertIn("quality_score", clip)
        self.assertIn("selection_reason", clip)
        self.assertEqual(clip["duration"], 35.0)

    def test_semantic_deduplication(self):
        """Verify that semantically redundant summaries are deduplicated, preferring higher quality."""
        # Candidate 1: Morning screen time focus
        cand1 = {
            "start_time": 10.0,
            "end_time": 45.0,
            "hook_summary": "Morning screen time ruins your daily focus and cognitive stamina",
            "scores": {"hook_strength": 9.0, "standalone_clarity": 9.0, "payoff": 9.0, "curiosity": 9.0, "impact": 9.0, "retention": 9.0, "context_dependency": 1.0, "punchline_score": 9.0},
        }
        # Candidate 2: Same topic at a later timestamp in video, lower score
        cand2 = {
            "start_time": 100.0,
            "end_time": 135.0,
            "hook_summary": "Why morning screen time ruins your daily focus completely",
            "scores": {"hook_strength": 7.5, "standalone_clarity": 7.5, "payoff": 7.5, "curiosity": 7.5, "impact": 7.5, "retention": 7.5, "context_dependency": 2.0, "punchline_score": 7.5},
        }
        results = _validate_and_filter_clips(
            [cand1, cand2],
            total_duration=200.0,
            min_clip_seconds=20,
            max_clip_seconds=59,
            max_clips=5,
            min_quality_score=70.0,
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["start_time"], 10.0)



if __name__ == "__main__":
    unittest.main()
