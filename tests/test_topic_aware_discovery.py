"""
Unit and integration tests for Topic-Aware Global Candidate Selection in clip discovery.
Verifies the production requirements:
1. 47 candidates -> max 5 final candidates.
2. 20 candidates belonging to the same topic -> only strongest representative(s).
3. Candidates across 5 distinct topics -> select one representative from each.
4. Candidates across 3 distinct topics -> return 3, not artificial 4.
5. Five distinct strong topics -> return 5.
6. Six or more distinct topics -> return maximum 5.
7. High-scoring duplicate topic does NOT beat slightly lower-scoring candidate from distinct topic.
8. Existing quality threshold remains >= 70.
9. Existing standalone threshold remains >= 7.0.
10. Temporal/semantic dedup remains active.
11. No mechanical fallback.
12. Selection is deterministic.
13. Existing production discovery tests continue to pass.
"""
import copy
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

import config
from clip_detection import (
    select_topic_diverse_clips,
    are_same_topic,
    extract_topic_stems,
    _validate_and_filter_clips,
    MIN_STANDALONE_SCORE,
    MIN_CLIP_QUALITY_SCORE,
)


def make_candidate(
    start: float,
    end: float,
    topic: str,
    topic_summary: str,
    quality_score: float = 85.0,
    standalone_score: float = 8.5,
    title: str = "Test Title",
    punchline: str = "Great Hook 🔥",
    reason: str = "High emotional payoff",
):
    return {
        "start": start,
        "end": end,
        "topic": topic,
        "topic_summary": topic_summary,
        "reason": reason,
        "title": title,
        "title_idea": title,
        "punchline": punchline,
        "quality_score": float(quality_score),
        "standalone_score": float(standalone_score),
        "confidence": 0.9,
        "transcript_evidence": f"Transcript for {topic}",
    }


class TestTopicAwareDiscovery(unittest.TestCase):
    def setUp(self):
        # Ensure default production constants are set
        self.target_clips = config.TARGET_CLIPS_PER_VIDEO
        self.max_clips = config.MAX_CLIPS_PER_VIDEO

    def test_01_forty_seven_candidates_to_max_five(self):
        """47 candidates across various topics -> maximum 5 final candidates."""
        candidates = []
        topics = [
            ("addiction", "struggles with addiction and recovery"),
            ("bollywood", "challenges of nepotism in bollywood cinema"),
            ("childhood", "memories from early childhood and school"),
            ("parents", "relationship with mother and father expectations"),
            ("relationships", "romantic relationship struggles and heartbreak"),
            ("success", "definition of career success and failure"),
            ("money", "wealth management and financial freedom"),
            ("fitness", "discipline in gym training and health"),
        ]
        # Generate 47 candidates across these 8 topics with varying timestamps
        for i in range(47):
            t_name, t_sum = topics[i % len(topics)]
            c = make_candidate(
                start=float(i * 70),
                end=float(i * 70 + 45),
                topic=f"{t_name} part {i}",
                topic_summary=f"{t_sum} discussion {i}",
                quality_score=75.0 + (i % 20),
                standalone_score=7.5 + (i % 3) * 0.5,
            )
            candidates.append(c)

        self.assertEqual(len(candidates), 47)
        final_clips = select_topic_diverse_clips(candidates, target_count=4, max_count=5)

        self.assertLessEqual(len(final_clips), 5)
        self.assertGreaterEqual(len(final_clips), 4)

    def test_02_twenty_candidates_same_topic_only_one_or_strongest(self):
        """20 candidates belonging to the same topic -> only strongest representative(s), not 20 final clips."""
        candidates = []
        for i in range(20):
            # All 20 candidates discuss parents/parenting
            c = make_candidate(
                start=float(i * 60),
                end=float(i * 60 + 40),
                topic="Parental expectations and parenting pressure",
                topic_summary=f"How parents influence children perspective {i}",
                quality_score=71.0 + (i * 1.2),  # Highest score is i=19 (93.8)
                standalone_score=7.2 + (i * 0.1),
            )
            candidates.append(c)

        final_clips = select_topic_diverse_clips(candidates, target_count=4, max_count=5)
        # Only 1 unique topic exists, so exactly 1 champion should be selected!
        self.assertEqual(len(final_clips), 1)
        self.assertAlmostEqual(final_clips[0]["quality_score"], 93.8, places=1)

    def test_03_candidates_across_five_distinct_topics_select_one_each(self):
        """Candidates across 5 distinct topics -> select one representative from each (5 total)."""
        topics = [
            ("Drug Addiction", "Why addiction feels like temporary freedom"),
            ("Bollywood Debut", "How nepotism works inside Bollywood"),
            ("Childhood Trauma", "Early memories growing up with anxiety"),
            ("Parenting Rules", "How strict parents shape discipline"),
            ("Romantic Heartbreak", "Dealing with toxic relationships and breakups"),
        ]
        candidates = []
        # Create 3 candidates per topic (15 total)
        for t_idx, (t_name, t_sum) in enumerate(topics):
            for sub in range(3):
                candidates.append(
                    make_candidate(
                        start=float(t_idx * 200 + sub * 60),
                        end=float(t_idx * 200 + sub * 60 + 45),
                        topic=f"{t_name} aspect {sub}",
                        topic_summary=f"{t_sum} subtopic {sub}",
                        quality_score=80.0 + t_idx + sub,
                        standalone_score=8.0,
                    )
                )

        final_clips = select_topic_diverse_clips(candidates, target_count=4, max_count=5)
        self.assertEqual(len(final_clips), 5)

        # Ensure all 5 topics are represented
        selected_topics = [c["topic"].lower() for c in final_clips]
        self.assertTrue(any("addiction" in t for t in selected_topics))
        self.assertTrue(any("bollywood" in t for t in selected_topics))
        self.assertTrue(any("childhood" in t for t in selected_topics))
        self.assertTrue(any("parent" in t for t in selected_topics))
        self.assertTrue(any("heartbreak" in t or "relationship" in t for t in selected_topics))

    def test_04_three_distinct_topics_returns_three_no_artificial_four(self):
        """Candidates across 3 distinct topics -> return 3, not artificial 4."""
        candidates = [
            make_candidate(0, 45, "Drug Addiction", "Struggles with substance abuse", quality_score=90),
            make_candidate(100, 145, "Parenting Rules", "Strict discipline by father", quality_score=88),
            make_candidate(200, 245, "Bollywood Stardom", "Acting journey in Bollywood", quality_score=86),
            # Add duplicates of the same 3 topics
            make_candidate(300, 345, "Substance Abuse", "Addiction patterns", quality_score=82),
            make_candidate(400, 445, "Strict Parents", "Mother and father expectations", quality_score=81),
            make_candidate(500, 545, "Cinema Industry", "Acting auditions in Bollywood", quality_score=80),
        ]

        final_clips = select_topic_diverse_clips(candidates, target_count=4, max_count=5)
        self.assertEqual(len(final_clips), 3)

    def test_05_five_distinct_strong_topics_returns_five(self):
        """Five distinct strong topics -> return 5."""
        candidates = [
            make_candidate(0, 40, "Addiction", "Recovering from addiction", quality_score=92),
            make_candidate(100, 140, "Childhood", "Growing up without siblings", quality_score=90),
            make_candidate(200, 240, "Parenting", "Father guidance lessons", quality_score=89),
            make_candidate(300, 340, "Bollywood", "Behind the scenes cinema", quality_score=88),
            make_candidate(400, 440, "Dating Heartbreak", "Lessons from past relationships", quality_score=87),
        ]

        final_clips = select_topic_diverse_clips(candidates, target_count=4, max_count=5)
        self.assertEqual(len(final_clips), 5)

    def test_06_six_or_more_distinct_topics_caps_at_five(self):
        """Seven distinct strong topics -> returns maximum 5."""
        candidates = [
            make_candidate(0, 40, "Addiction", "Substance abuse recovery", quality_score=95),
            make_candidate(100, 140, "Childhood", "Childhood anxiety and memory", quality_score=94),
            make_candidate(200, 240, "Parenting", "Parents emotional connection", quality_score=93),
            make_candidate(300, 340, "Bollywood", "Cinema industry nepotism debate", quality_score=92),
            make_candidate(400, 440, "Heartbreak", "Romantic heartbreak and breakup", quality_score=91),
            make_candidate(500, 540, "Wealth Creation", "Financial freedom and mindset", quality_score=90),
            make_candidate(600, 640, "Mental Health", "Depression and therapy benefits", quality_score=89),
        ]

        final_clips = select_topic_diverse_clips(candidates, target_count=4, max_count=5)
        self.assertEqual(len(final_clips), 5)

    def test_07_duplicate_topic_cannot_beat_distinct_topic(self):
        """High-scoring duplicate topic must NOT automatically beat a slightly lower-scoring candidate from a completely distinct topic."""
        candidates = [
            # Topic 1: Addiction has top 2 scores
            make_candidate(0, 40, "Substance Addiction", "Why addiction takes over", quality_score=96),
            make_candidate(50, 90, "Drug Addiction", "The craving of addiction", quality_score=94),
            # Topic 2: Parents
            make_candidate(150, 190, "Parents and Family", "Respecting father and mother", quality_score=89),
            # Topic 3: Bollywood
            make_candidate(250, 290, "Bollywood Acting", "Auditioning for movies", quality_score=88),
            # Topic 4: Childhood
            make_candidate(350, 390, "Childhood Memories", "School days nostalgia", quality_score=85),
        ]

        final_clips = select_topic_diverse_clips(candidates, target_count=4, max_count=5)
        self.assertEqual(len(final_clips), 4)

        # Must have only 1 addiction clip (the 96 one, NOT both 96 and 94)
        addiction_clips = [c for c in final_clips if "addict" in c["topic"].lower()]
        self.assertEqual(len(addiction_clips), 1)
        self.assertEqual(addiction_clips[0]["quality_score"], 96)

        # Childhood clip with score 85 MUST survive over the 94 addiction duplicate
        childhood_clips = [c for c in final_clips if "childhood" in c["topic"].lower()]
        self.assertEqual(len(childhood_clips), 1)

    def test_08_existing_quality_threshold_preserved(self):
        """Existing quality threshold remains >= 70 (candidates < 70 are rejected)."""
        candidates = [
            make_candidate(0, 40, "Topic A", "Summary A", quality_score=75.0, standalone_score=8.0),
            make_candidate(100, 140, "Topic B", "Summary B", quality_score=69.9, standalone_score=8.5),  # Under 70
            make_candidate(200, 240, "Topic C", "Summary C", quality_score=82.0, standalone_score=7.5),
        ]

        final_clips = select_topic_diverse_clips(candidates, target_count=4, max_count=5)
        # Only Topic A and Topic C should survive
        self.assertEqual(len(final_clips), 2)
        scores = [c["quality_score"] for c in final_clips]
        for s in scores:
            self.assertGreaterEqual(s, MIN_CLIP_QUALITY_SCORE)

    def test_09_existing_standalone_threshold_preserved(self):
        """Existing standalone threshold remains >= 7.0 (candidates < 7.0 are rejected)."""
        candidates = [
            make_candidate(0, 40, "Topic A", "Summary A", quality_score=85.0, standalone_score=8.0),
            make_candidate(100, 140, "Topic B", "Summary B", quality_score=88.0, standalone_score=6.9),  # Under 7.0
            make_candidate(200, 240, "Topic C", "Summary C", quality_score=82.0, standalone_score=7.0),
        ]

        final_clips = select_topic_diverse_clips(candidates, target_count=4, max_count=5)
        self.assertEqual(len(final_clips), 2)
        for c in final_clips:
            self.assertGreaterEqual(c["standalone_score"], MIN_STANDALONE_SCORE)

    def test_10_temporal_dedup_active(self):
        """Temporal overlap threshold is enforced (overlapping moments deduplicate to higher quality)."""
        candidates = [
            # Two clips overlapping in time (10.0-50.0 and 20.0-60.0, overlap = 30s > 10s max)
            make_candidate(10.0, 50.0, "Addiction Story", "Addiction reflection", quality_score=88.0),
            make_candidate(20.0, 60.0, "Substance Abuse", "Addiction reflection deep", quality_score=92.0),
            make_candidate(120.0, 160.0, "Bollywood Journey", "Cinema journey", quality_score=84.0),
        ]

        final_clips = select_topic_diverse_clips(candidates, target_count=4, max_count=5)
        self.assertEqual(len(final_clips), 2)
        # The 92.0 score candidate should be kept
        times = [(c["start"], c["end"]) for c in final_clips]
        self.assertIn((20.0, 60.0), times)
        self.assertNotIn((10.0, 50.0), times)

    def test_11_no_mechanical_fallback(self):
        """If 0 candidates qualify, returns empty list; does not manufacture fallback clips."""
        empty_clips = select_topic_diverse_clips([], target_count=4, max_count=5)
        self.assertEqual(empty_clips, [])

        unqualified = [
            make_candidate(0, 40, "Low Score A", "Summary A", quality_score=50.0, standalone_score=5.0),
            make_candidate(100, 140, "Low Score B", "Summary B", quality_score=60.0, standalone_score=6.5),
        ]
        filtered = select_topic_diverse_clips(unqualified, target_count=4, max_count=5)
        self.assertEqual(filtered, [])

    def test_12_selection_is_deterministic(self):
        """Selection is completely deterministic across repeated invocations."""
        candidates = [
            make_candidate(0, 40, "Topic A", "Summary A", quality_score=85),
            make_candidate(100, 140, "Topic B", "Summary B", quality_score=90),
            make_candidate(200, 240, "Topic C", "Summary C", quality_score=82),
            make_candidate(300, 340, "Topic D", "Summary D", quality_score=88),
            make_candidate(400, 440, "Topic E", "Summary E", quality_score=91),
            make_candidate(500, 540, "Topic F", "Summary F", quality_score=80),
        ]

        res1 = select_topic_diverse_clips(copy.deepcopy(candidates), target_count=4, max_count=5)
        res2 = select_topic_diverse_clips(copy.deepcopy(candidates), target_count=4, max_count=5)
        res3 = select_topic_diverse_clips(copy.deepcopy(candidates), target_count=4, max_count=5)

        self.assertEqual(len(res1), len(res2))
        self.assertEqual(len(res1), len(res3))
        for i in range(len(res1)):
            self.assertEqual(res1[i]["topic"], res2[i]["topic"])
            self.assertEqual(res1[i]["topic"], res3[i]["topic"])
            self.assertEqual(res1[i]["start"], res2[i]["start"])
            self.assertEqual(res1[i]["clip_index"], res2[i]["clip_index"])

    def test_13_topic_stems_and_canonicalization(self):
        """Topic stemming recognizes synonyms and morphological variants."""
        self.assertTrue(are_same_topic("My Mother and Father", "Parental Expectations"))
        self.assertTrue(are_same_topic("Drug Addiction in Youth", "Substance Abuse and Escapism"))
        self.assertTrue(are_same_topic("Bollywood Acting Auditions", "Working in Hindi Cinema"))
        self.assertFalse(are_same_topic("Drug Addiction", "Parental Expectations"))
        self.assertFalse(are_same_topic("Bollywood Cinema", "Childhood Memories"))


if __name__ == "__main__":
    unittest.main()
