"""
Tests for Phase 4: Reduce Clip-Discovery Request Size.
Validates:
1. Chunk boundary splitting respects character budget and preserves overlap.
2. Clips spanning boundaries are contained fully in overlapping chunks.
3. Streamlined discovery schema is correctly parsed and validated.
"""
import unittest
from unittest.mock import MagicMock, patch

from scripts.clip_detection import (
    _split_lines_into_chunks,
    _validate_and_filter_clips,
    _detect_clips_llm,
)
from scripts.config import (
    DISCOVERY_CHUNK_MAX_CHARS,
    DISCOVERY_CHUNK_OVERLAP_SECONDS,
)


class TestPhase4DiscoveryPayload(unittest.TestCase):

    def test_chunk_boundaries_and_overlap(self):
        """
        Verifies that transcript lines are split into chunks respecting max_chars,
        and adjacent chunks share at least overlap_seconds of time coverage.
        """
        # Create 100 lines spanning 500 seconds, total length ~12,000 chars
        lines = []
        for i in range(100):
            st = float(i * 5)
            et = float((i + 1) * 5)
            text = f"[{st:.1f}s - {et:.1f}s] This is sentence number {i} discussing deep concepts in physics."
            lines.append((st, et, text))

        max_chars = 3000
        overlap_seconds = 40.0
        chunks = _split_lines_into_chunks(lines, max_chars=max_chars, overlap_seconds=overlap_seconds)

        self.assertGreater(len(chunks), 1, "Should split into multiple chunks")

        for idx, chunk in enumerate(chunks):
            chunk_text = "\n".join(item[2] for item in chunk)
            self.assertLessEqual(
                len(chunk_text),
                max_chars + 300,  # Single line tolerance
                f"Chunk {idx} exceeds max_chars tolerance"
            )

        # Check overlap between chunk 0 and chunk 1
        for idx in range(len(chunks) - 1):
            c1 = chunks[idx]
            c2 = chunks[idx + 1]
            c1_end = c1[-1][1]
            c2_start = c2[0][0]
            overlap = c1_end - c2_start
            self.assertGreaterEqual(
                overlap,
                overlap_seconds - 5.0,  # Line granularity tolerance
                f"Overlap between chunk {idx} and {idx+1} is {overlap}s, expected at least {overlap_seconds}s"
            )

    def test_boundary_clip_discoverable_in_overlap(self):
        """
        Regression test: A 45-second talk segment straddling a chunk boundary
        is fully covered in the second overlapping chunk.
        """
        lines = []
        # 0s to 300s of speech
        for i in range(60):
            st = float(i * 5)
            et = float((i + 1) * 5)
            lines.append((st, et, f"[{st:.1f}s - {et:.1f}s] Continuous monologue segment {i} with valuable discussion."))

        # Boundary talk segment from 110.0s to 155.0s (45s duration)
        target_start = 110.0
        target_end = 155.0

        chunks = _split_lines_into_chunks(lines, max_chars=2500, overlap_seconds=65.0)
        self.assertGreater(len(chunks), 1)

        # Check if the target span [110, 155] is completely contained in at least one chunk
        found_in_chunk = False
        for c in chunks:
            c_start = c[0][0]
            c_end = c[-1][1]
            if c_start <= target_start and c_end >= target_end:
                found_in_chunk = True
                break

        self.assertTrue(
            found_in_chunk,
            f"Target clip [{target_start}, {target_end}] should be fully contained within at least one chunk"
        )

    def test_streamlined_discovery_schema_parsed_and_scored(self):
        """
        Verifies that candidates conforming to the streamlined discovery schema
        are successfully parsed, validated, and authoritatively scored by Python.
        """
        streamlined_raw_clips = [
            {
                "start": 10.0,
                "end": 50.0,
                "topic": "Why habits fail",
                "topic_summary": "Explains why willpower alone is never sufficient to sustain habits.",
                "title_idea": "Why Willpower Always Fails #shorts",
                "punchline": "System Over Willpower 🔥",
                "standalone": True,
                "standalone_score": 9.0,
                "missing_setup": False,
                "missing_payoff": False,
                "critical_unresolved_reference": False,
                "unresolved_references": [],
                "quality_scores": {
                    "hook_strength": 9.0,
                    "standalone_clarity": 9.0,
                    "payoff_completion": 9.0,
                    "curiosity": 8.0,
                    "emotional_intellectual_impact": 8.0,
                    "retention_potential": 8.0,
                    "context_independence": 9.0,
                    "punchline_memorable_moment": 8.0,
                },
                "reason": "Complete micro-narrative with crisp setup and conclusion."
            }
        ]

        validated = _validate_and_filter_clips(
            streamlined_raw_clips,
            total_duration=300.0,
            min_clip_seconds=20,
            max_clip_seconds=60,
        )

        self.assertEqual(len(validated), 1)
        clip = validated[0]
        self.assertEqual(clip["start_time"], 10.0)
        self.assertEqual(clip["end_time"], 50.0)
        self.assertEqual(clip["duration"], 40.0)
        self.assertEqual(clip["topic"], "Why habits fail")
        self.assertGreaterEqual(clip["quality_score"], 70.0)
        self.assertTrue(clip["standalone"])


if __name__ == "__main__":
    unittest.main()
