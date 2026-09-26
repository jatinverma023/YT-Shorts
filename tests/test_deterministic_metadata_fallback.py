"""
Focused unit tests for deterministic metadata fallback quality (Section 8).
Verifies:
1. Opening transcript should NOT automatically become fallback hook
2. Explicit question is preferred
3. Consequence is preferred
4. Strong insight is preferred
5. Fallback remains transcript-grounded
6. Generic topic-label rejection still works
7. Hook length limits still work (3-8 words, <=42 characters)
8. Semantic emoji rules still work
9. Filename is not blindly used as title
10. Transcript-derived title is preferred
11. Existing hashtag validation remains unchanged
12. Existing claim-strength validation remains unchanged
"""

import unittest
import re
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

import hook_generator
import metadata_ai


class TestDeterministicMetadataFallback(unittest.TestCase):

    def setUp(self):
        self.zakir_transcript = (
            "Zakir Khan Jab aap apne parents se baat karte hain toh unka perspective hamesha alag hota hai "
            "kyunki unhone duniya ko ek different angle se dekha hai aur hum apne generation ke lens se dekhte hain "
            "lekin pyaar aur care dono taraf barabar hota hai bas express karne ka tareeqa alag hai "
            "Aur yeh samajhna zindagi ka sabse bada sabak hai ki rishton mein communication hi sab kuch hota hai "
            "agar aap baat nahin karenge toh dooriyaan badhti jayengi isliye baat karna hamesha zaroori hai har relationship mein"
        )
        self.zakir_filename = "@ZakirKhan On Parents, Relationship, Bollywood, Success, Money Zakir Khan F.mp4"

    def test_01_opening_transcript_not_automatically_fallback_hook(self):
        """1. Opening transcript should NOT automatically become fallback hook."""
        hook_obj = hook_generator.get_fallback_hook(transcript=self.zakir_transcript)
        selected_hook = hook_obj["hook"]

        # The opening was "Zakir Khan Jab aap apne parents..."
        self.assertNotIn("ZAKIR KHAN JAB", selected_hook.upper())
        self.assertNotIn("JAB AAP APNE PARENTS", selected_hook.upper())
        # Instead, it should capture the core insight, consequence, or question
        norm_h = selected_hook.lower()
        self.assertTrue(
            any(k in norm_h for k in ["dooriyaan", "communication", "sabak", "zaroori"]),
            f"Hook should extract high-value clause, got: '{selected_hook}'"
        )

    def test_02_explicit_question_is_preferred(self):
        """2. Explicit question is preferred when present in transcript."""
        transcript_with_question = (
            "People always try to find the secret to great health. "
            "Why do we misunderstand sleep? "
            "Most people think sleep is passive, but your brain is actually active all night."
        )
        hook_text, _ = hook_generator.extract_grounded_fallback_hook(transcript_with_question)
        self.assertIn("?", hook_text)
        self.assertIn("SLEEP", hook_text.upper())
        self.assertIn("WHY", hook_text.upper())

    def test_03_consequence_is_preferred(self):
        """3. Consequence is preferred over weak neutral opening clauses."""
        transcript_with_consequence = (
            "We were discussing various habits yesterday morning. "
            "If you ignore chronic stress, your health breaks down completely."
        )
        hook_text, _ = hook_generator.extract_grounded_fallback_hook(transcript_with_consequence)
        # Should not pick "We were discussing various habits"
        self.assertNotIn("YESTERDAY", hook_text.upper())
        self.assertNotIn("DISCUSSING", hook_text.upper())
        # Should capture consequence
        norm_h = hook_text.lower()
        self.assertTrue(any(w in norm_h for w in ["health", "stress", "breaks", "down"]))

    def test_04_strong_insight_is_preferred(self):
        """4. Strong insight is preferred over neutral opening clauses."""
        transcript_with_insight = (
            "I was talking to a mentor last week about business. "
            "Discipline is everything when building a company. "
            "Motivation always fades quickly."
        )
        hook_text, _ = hook_generator.extract_grounded_fallback_hook(transcript_with_insight)
        self.assertNotIn("TALKING TO A MENTOR", hook_text.upper())
        self.assertNotIn("LAST WEEK", hook_text.upper())
        norm_h = hook_text.lower()
        self.assertTrue(any(w in norm_h for w in ["discipline", "everything", "company", "motivation"]))

    def test_05_fallback_remains_transcript_grounded(self):
        """5. Fallback hook remains strictly transcript-grounded."""
        hook_obj = hook_generator.get_fallback_hook(transcript=self.zakir_transcript)
        selected_hook = hook_obj["hook"]

        clean_hook = hook_generator.EMOJI_PATTERN.sub("", selected_hook).strip()
        words = [w.lower().rstrip("?,.!") for w in clean_hook.split()]
        norm_tr = self.zakir_transcript.lower()

        # At least 70% of words in fallback hook must appear in the transcript
        matching = [w for w in words if w in norm_tr]
        ratio = len(matching) / len(words)
        self.assertGreaterEqual(ratio, 0.70, f"Hook words not grounded in clip: {words}")

    def test_06_generic_topic_label_rejection_still_works(self):
        """6. Generic topic-label rejection still works."""
        bad_labels = [
            "Zakir Khan on Relationships",
            "Zakir Khan talks about parents",
            "Andrew Huberman discusses sleep",
            "Talking about mutual funds",
        ]
        for label in bad_labels:
            self.assertTrue(
                hook_generator.is_generic_topic_label_hook(label),
                f"Should detect generic topic label: {label}"
            )
            # Rejection in validate_hook
            is_val, reason = hook_generator.validate_hook(
                {"hook": label, "supported_by_clip": True},
                transcript=self.zakir_transcript
            )
            self.assertFalse(is_val)
            self.assertIn("generic topic label", reason.lower())

    def test_07_hook_length_limits_still_work(self):
        """7. Hook length limits still work (3-8 words, <= 42 characters)."""
        hook_obj = hook_generator.get_fallback_hook(transcript=self.zakir_transcript)
        selected_hook = hook_obj["hook"]

        self.assertLessEqual(len(selected_hook), hook_generator.MAX_HOOK_CHARS)
        no_em = hook_generator.EMOJI_PATTERN.sub("", selected_hook).strip()
        word_count = len(no_em.split())
        self.assertGreaterEqual(word_count, 3)
        self.assertLessEqual(word_count, 8)

    def test_08_semantic_emoji_rules_still_work(self):
        """8. Semantic emoji rules still work (0-2 emojis, trailing placement, valid domain)."""
        hook_obj = hook_generator.get_fallback_hook(transcript=self.zakir_transcript)
        selected_hook = hook_obj["hook"]

        emojis = hook_generator.extract_emojis(selected_hook)
        self.assertLessEqual(len(emojis), 2)
        if emojis:
            # Trailing emoji placement
            self.assertTrue(
                selected_hook.endswith(emojis[-1]),
                f"Emoji should be trailing, got: '{selected_hook}'"
            )
            # No emoji chains
            self.assertIsNone(re.search(r"(" + hook_generator.EMOJI_PATTERN.pattern + r"\s*){2,}", selected_hook))

    def test_09_filename_is_not_blindly_used_as_title(self):
        """9. Filename is not blindly used as title when transcript exists."""
        title = metadata_ai.extract_grounded_fallback_title(
            transcript=self.zakir_transcript,
            filename=self.zakir_filename,
        )
        self.assertNotIn("Bollywood", title)
        self.assertNotIn("Success", title)
        self.assertNotIn("Money", title)
        self.assertNotIn("Zakir Khan F", title)
        self.assertFalse(hook_generator.is_generic_topic_label_hook(title))

    def test_10_transcript_derived_title_is_preferred(self):
        """10. Transcript-derived title is preferred over filename."""
        title = metadata_ai.extract_grounded_fallback_title(
            transcript=self.zakir_transcript,
            filename=self.zakir_filename,
        )
        words = title.lower().replace("#shorts", "").split()
        norm_tr = self.zakir_transcript.lower()
        supported = [w for w in words if w in norm_tr]
        self.assertGreaterEqual(
            len(supported) / len(words),
            0.60,
            f"Title words insufficiently grounded in transcript: {title}"
        )
        self.assertTrue(title.lower().endswith("#shorts"))

    def test_11_existing_hashtag_validation_remains_unchanged(self):
        """11. Existing hashtag validation remains unchanged (4-6 hashtags, no #viral/#trending)."""
        hashtags = metadata_ai.extract_grounded_fallback_hashtags(
            transcript=self.zakir_transcript,
            filename=self.zakir_filename,
        )
        self.assertGreaterEqual(len(hashtags), 4)
        self.assertLessEqual(len(hashtags), 6)
        self.assertIn("#Shorts", hashtags)
        self.assertIn("#ZakirKhan", hashtags)

        norm_tags = [t.lower() for t in hashtags]
        self.assertNotIn("#viral", norm_tags)
        self.assertNotIn("#trending", norm_tags)
        self.assertNotIn("#fyp", norm_tags)

    def test_12_existing_claim_strength_validation_remains_unchanged(self):
        """12. Existing claim-strength validation remains unchanged."""
        transcript = (
            "Researchers are carefully studying cell biology. "
            "This therapy is definitely a guaranteed 100% cure for cancer. "
            "Early laboratory observations remain preliminary."
        )
        hook_text, _ = hook_generator.extract_grounded_fallback_hook(transcript)
        self.assertNotIn("GUARANTEED", hook_text.upper())
        self.assertNotIn("100%", hook_text.upper())


if __name__ == "__main__":
    unittest.main()
