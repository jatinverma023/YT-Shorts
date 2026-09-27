"""
Tests for targeted upgrade of Hook, Description & Hashtag Quality.

Verifies the 22 specific test criteria in PART 13 & PART 14:
1. Generic topic-label hook rejection.
2. Strong curiosity hook accepted.
3. Contrarian hook accepted when grounded.
4. Question hook accepted.
5. Consequence hook accepted.
6. Specific-insight hook accepted.
7. Hook remains within length limits.
8. Hook supports 0–2 semantic emojis.
9. Duplicate emojis rejected.
10. Emoji spam rejected.
11. Irrelevant emojis rejected.
12. Generic description rejected.
13. Description must add information beyond hook.
14. Description remains transcript-grounded.
15. Hashtag count is 4–6.
16. Duplicate hashtags rejected.
17. Generic-only hashtag sets rejected.
18. Clip-specific hashtags accepted.
19. Unsupported-topic hashtags rejected.
20. Metadata fallback produces useful hook/description/hashtags.
21. Existing claim-strength tests continue passing.
22. Existing package scoring tests continue passing.
"""
import unittest
from unittest.mock import patch
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

import hook_generator
import metadata_ai
import config
from ai_rate_limiter import shared_rate_limiter


class TestTargetedPackagingUpgrade(unittest.TestCase):

    def setUp(self):
        shared_rate_limiter._history.clear()
        self.relationship_transcript = (
            "Jab hum kisi se pyaar karte hain toh hum aksar samajh nahi paate ki saamne waala kya mehsoos kar raha hai. "
            "Relationships badalti hain kyunki hum sunna band kar dete hain aur sirf apni expectations thopte hain. "
            "Agar communicate karna seekh lein toh rishta bach sakta hai."
        )
        self.money_transcript = (
            "Most people think saving money is enough to build wealth, but inflation quietly destroys idle cash in bank accounts. "
            "If you do not invest in compounding assets early, you work for money your entire life instead of money working for you."
        )
        self.sleep_transcript = (
            "When you are having trouble falling asleep at night, your autonomic nervous system is stuck in high arousal. "
            "By doing two quick inhales through the nose and a long exhale through the mouth, you immediately drop your heart rate "
            "and turn off the mind to sleep."
        )

    # 1. Generic topic-label hook rejection
    def test_01_generic_topic_label_hook_rejected(self):
        weak_hooks = [
            "Zakir Khan on Relationships & Parents 🥰",
            "Zakir Khan Talks About Relationships",
            "Zakir Khan on Success",
            "Zakir Khan discusses money",
            "Zakir Khan shares his thoughts on life",
            "Talking about relationships",
            "Discussion about money",
            "Zakir Khan on Relationships",
        ]
        for hook_text in weak_hooks:
            self.assertTrue(
                hook_generator.is_generic_topic_label_hook(hook_text),
                f"Expected '{hook_text}' to be recognized as generic topic-label",
            )
            cand = {"hook": hook_text, "hook_type": "curiosity"}
            is_valid, reason = hook_generator.validate_hook(cand, transcript=self.relationship_transcript)
            self.assertFalse(is_valid, f"Expected '{hook_text}' to be rejected by validate_hook. Reason: {reason}")
            self.assertTrue("topic label" in reason.lower() or "topic-label" in reason.lower())

        # Ensure person-centered hooks that are NOT topic labels are preserved
        valid_person_hook = "Why Zakir Khan Changed His Mind ❤️"
        self.assertFalse(hook_generator.is_generic_topic_label_hook(valid_person_hook))

    # 2. Strong curiosity hook accepted
    def test_02_strong_curiosity_hook_accepted(self):
        cand = {
            "hook": "Why We Misunderstand Love ❤️",
            "hook_type": "curiosity",
            "supporting_text": "samajh nahi paate ki saamne waala kya mehsoos kar raha hai",
        }
        is_valid, reason = hook_generator.validate_hook(cand, transcript=self.relationship_transcript)
        self.assertTrue(is_valid, f"Hook should be accepted, got reason: {reason}")

    # 3. Contrarian hook accepted when grounded
    def test_03_contrarian_hook_accepted_when_grounded(self):
        cand = {
            "hook": "Saving Money Destroys Wealth 💸",
            "hook_type": "contrarian",
            "supporting_text": "saving money is enough to build wealth, but inflation quietly destroys",
        }
        is_valid, reason = hook_generator.validate_hook(cand, transcript=self.money_transcript)
        self.assertTrue(is_valid, f"Contrarian hook should be accepted, got reason: {reason}")

    # 4. Question hook accepted
    def test_04_question_hook_accepted(self):
        cand = {
            "hook": "Can't Sleep At Night? 💡",
            "hook_type": "question",
            "supporting_text": "When you are having trouble falling asleep at night",
        }
        is_valid, reason = hook_generator.validate_hook(cand, transcript=self.sleep_transcript)
        self.assertTrue(is_valid, f"Question hook should be accepted, got reason: {reason}")

    # 5. Consequence hook accepted
    def test_05_consequence_hook_accepted(self):
        cand = {
            "hook": "Stop Listening, Lose Relationships 💔",
            "hook_type": "consequence",
            "supporting_text": "Relationships badalti hain kyunki hum sunna band kar dete hain",
        }
        is_valid, reason = hook_generator.validate_hook(cand, transcript=self.relationship_transcript)
        self.assertTrue(is_valid, f"Consequence hook should be accepted, got reason: {reason}")

    # 6. Specific-insight hook accepted
    def test_06_specific_insight_hook_accepted(self):
        cand = {
            "hook": "Double Inhale Calms Heart Rate 🫁",
            "hook_type": "specific_fact",
            "supporting_text": "two quick inhales through the nose and a long exhale... drop your heart rate",
        }
        is_valid, reason = hook_generator.validate_hook(cand, transcript=self.sleep_transcript)
        self.assertTrue(is_valid, f"Specific insight hook should be accepted, got reason: {reason}")

    # 7. Hook remains within length limits
    def test_07_hook_length_limits(self):
        too_many_words = {"hook": "A B C D E F G H I"}  # 9 words, under 42 chars
        is_valid, reason = hook_generator.validate_hook(too_many_words, transcript=self.money_transcript)
        self.assertFalse(is_valid)
        self.assertIn("word", reason.lower())

        too_long_chars = {"hook": "This sentence is way too long for a short hook that fits under 42 characters"}
        is_valid, reason = hook_generator.validate_hook(too_long_chars, transcript=self.money_transcript)
        self.assertFalse(is_valid)
        self.assertIn("hard limit", reason.lower())

        perfect_hook = {"hook": "The Money Mistake Everyone Makes 💸", "hook_type": "curiosity"}
        is_valid, _ = hook_generator.validate_hook(perfect_hook, transcript=self.money_transcript)
        self.assertTrue(is_valid)

    # 8. Hook supports 0–2 semantic emojis
    def test_08_hook_supports_zero_to_two_semantic_emojis(self):
        # 0 emojis
        cand_0 = {"hook": "Why We Misunderstand Love", "hook_type": "curiosity"}
        valid_0, _ = hook_generator.validate_hook(cand_0, transcript=self.relationship_transcript)
        self.assertTrue(valid_0)

        # 1 emoji
        cand_1 = {"hook": "Why We Misunderstand Love ❤️", "hook_type": "curiosity"}
        valid_1, _ = hook_generator.validate_hook(cand_1, transcript=self.relationship_transcript)
        self.assertTrue(valid_1)

        # 2 distinct semantic emojis separated by words
        cand_2 = {"hook": "Why We 💔 Misunderstand Love ❤️", "hook_type": "curiosity"}
        valid_2, _ = hook_generator.validate_hook(cand_2, transcript=self.relationship_transcript)
        self.assertTrue(valid_2)

        # 3 emojis rejected
        cand_3 = {"hook": "Why We Misunderstand Love ❤️ 🫶 💔", "hook_type": "curiosity"}
        valid_3, reason = hook_generator.validate_hook(cand_3, transcript=self.relationship_transcript)
        self.assertFalse(valid_3)
        self.assertIn("emoji", reason.lower())

    # 9. Duplicate emojis rejected
    def test_09_duplicate_emojis_rejected(self):
        cand_dup = {"hook": "Why We Misunderstand Love ❤️ ❤️", "hook_type": "curiosity"}
        is_valid, reason = hook_generator.validate_hook(cand_dup, transcript=self.relationship_transcript)
        self.assertFalse(is_valid)
        self.assertIn("duplicate", reason.lower())

    # 10. Emoji spam rejected
    def test_10_emoji_spam_rejected(self):
        cand_spam = {"hook": "Why We Misunderstand Love 👀🔥💯🚀😂"}
        is_valid, reason = hook_generator.validate_hook(cand_spam, transcript=self.relationship_transcript)
        self.assertFalse(is_valid)

        cand_punct = {"hook": "WHY?! 😱🔥💯"}
        is_valid_punct, _ = hook_generator.validate_hook(cand_punct, transcript=self.relationship_transcript)
        self.assertFalse(is_valid_punct)

    # 11. Irrelevant emojis rejected
    def test_11_irrelevant_emojis_rejected(self):
        # Rocket on relationship clip
        cand_irrelevant = {"hook": "Why We Misunderstand Love 🚀", "hook_type": "curiosity"}
        is_valid, reason = hook_generator.validate_hook(cand_irrelevant, transcript=self.relationship_transcript)
        self.assertFalse(is_valid)
        self.assertIn("emoji", reason.lower())

    # 12. Generic description rejected
    def test_12_generic_description_rejected(self):
        generic_fillers = [
            "Watch this amazing clip to see what happens next in this video!",
            "You won't believe what happens in this clip, check it out!",
            "Don't forget to like and subscribe for more amazing daily videos!",
        ]
        hook = "Why We Misunderstand Love ❤️"
        title = "The Communication Mistake In Modern Relationships #shorts"
        for filler in generic_fillers:
            cand = {"description": filler}
            is_valid, reason = metadata_ai.validate_description(
                cand, transcript=self.relationship_transcript, hook=hook, title=title
            )
            self.assertFalse(is_valid, f"Expected '{filler}' to be rejected. Reason: {reason}")
            self.assertIn("generic filler", reason.lower())

    # 13. Description must add information beyond hook
    def test_13_description_must_add_information_beyond_hook(self):
        hook = "Why Saving Money Fails 💸"
        title = "Why Saving Money Alone Destroys Wealth #shorts"
        # Pure repetition of hook and title
        repetitive_desc = "In this video, why saving money alone destroys wealth."
        cand = {"description": repetitive_desc}
        is_valid, reason = metadata_ai.validate_description(
            cand, transcript=self.money_transcript, hook=hook, title=title
        )
        self.assertFalse(is_valid)
        self.assertTrue("restate" in reason.lower() or "repeat" in reason.lower())

        # Substantive description adding context + payoff grounded in transcript
        good_desc = (
            "Inflation quietly destroys idle cash in bank accounts over time. "
            "Investing in compounding assets early ensures your money works for you instead of working forever."
        )
        cand_good = {"description": good_desc}
        is_valid_good, reason_good = metadata_ai.validate_description(
            cand_good, transcript=self.money_transcript, hook=hook, title=title
        )
        self.assertTrue(is_valid_good, f"Expected good description to pass, got: {reason_good}")

    # 14. Description remains transcript-grounded
    def test_14_description_transcript_grounded(self):
        hook = "Double Inhale Calms Heart Rate 💡"
        title = "How Two Quick Inhales Drop Heart Rate #shorts"
        hallucinated_desc = (
            "Drinking two gallons of green tea every morning cures all sleep disorders permanently according to doctors."
        )
        cand = {"description": hallucinated_desc}
        is_valid, reason = metadata_ai.validate_description(
            cand, transcript=self.sleep_transcript, hook=hook, title=title
        )
        self.assertFalse(is_valid)
        self.assertTrue("unsupported" in reason.lower() or "ground" in reason.lower())

    # 15. Hashtag count is 4–6
    def test_15_hashtag_count_is_4_to_6(self):
        too_few = ["#Shorts", "#Love", "#Relationships"]  # 3
        is_val, _ = metadata_ai.is_valid_hashtag_set(too_few, self.relationship_transcript)
        self.assertFalse(is_val)

        too_many = ["#Shorts", "#Love", "#Relationships", "#LifeLessons", "#Podcast", "#HindiPodcast", "#Dating"]  # 7
        is_val_many, _ = metadata_ai.is_valid_hashtag_set(too_many, self.relationship_transcript)
        self.assertFalse(is_val_many)

        valid_4 = ["#Relationships", "#Love", "#LifeLessons", "#PodcastClips"]
        is_val_4, _ = metadata_ai.is_valid_hashtag_set(valid_4, self.relationship_transcript)
        self.assertTrue(is_val_4)

        valid_6 = ["#Relationships", "#Love", "#LifeLessons", "#HindiPodcast", "#PodcastClips", "#Shorts"]
        is_val_6, _ = metadata_ai.is_valid_hashtag_set(valid_6, self.relationship_transcript)
        self.assertTrue(is_val_6)

    # 16. Duplicate hashtags rejected
    def test_16_duplicate_hashtags_rejected(self):
        dups = ["#Relationships", "#Love", "#Love", "#LifeLessons", "#PodcastClips"]
        is_val, reason = metadata_ai.is_valid_hashtag_set(dups, self.relationship_transcript)
        self.assertFalse(is_val)
        self.assertIn("duplicate", reason.lower())

    # 17. Generic-only hashtag sets rejected
    def test_17_generic_only_hashtag_sets_rejected(self):
        generic_set = ["#viral", "#trending", "#fyp", "#explore", "#foryou"]
        is_val, reason = metadata_ai.is_valid_hashtag_set(generic_set, self.relationship_transcript)
        self.assertFalse(is_val)

    # 18. Clip-specific hashtags accepted
    def test_18_clip_specific_hashtags_accepted(self):
        specific_tags = ["#ZakirKhan", "#Relationships", "#Love", "#LifeLessons", "#PodcastClips"]
        is_val, reason = metadata_ai.is_valid_hashtag_set(
            specific_tags,
            transcript=self.relationship_transcript,
            filename="Zakir Khan Relationships Podcast.mp4",
        )
        self.assertTrue(is_val, f"Hashtags should be accepted, got: {reason}")

    # 19. Unsupported-topic hashtags rejected
    def test_19_unsupported_topic_hashtags_rejected(self):
        # Cryptocurrency / Bitcoin hashtag on relationship clip
        unsupported = ["#Relationships", "#Love", "#Crypto", "#Bitcoin", "#LifeLessons"]
        is_val, reason = metadata_ai.is_valid_hashtag_set(unsupported, self.relationship_transcript)
        self.assertFalse(is_val)
        self.assertIn("grounded", reason.lower())

    # 20. Metadata fallback produces useful hook/description/hashtags
    def test_20_metadata_fallback_produces_useful_metadata(self):
        with patch.object(config, "GROQ_API_KEY", ""), patch.object(config, "OPENAI_API_KEY", ""), patch.dict(os.environ, {"GROQ_API_KEY": "", "OPENAI_API_KEY": ""}):
            meta = metadata_ai.generate_shorts_metadata(
                filename="Zakir Khan Relationships Podcast FO123.mp4",
                transcript=self.relationship_transcript,
            )

        self.assertTrue(meta.get("is_fallback"))
        # Hook should NOT be generic topic-label "Zakir Khan on Relationships"
        hook = meta["punchline"]
        self.assertFalse(
            hook_generator.is_generic_topic_label_hook(hook),
            f"Fallback hook '{hook}' should not be a generic topic-label",
        )
        self.assertGreater(len(hook), 5)
        self.assertLessEqual(len(hook), 45)

        # Description should NOT be empty or generic filler
        desc = meta["description"]
        self.assertGreater(len(desc), 20)
        self.assertNotIn("watch this amazing clip", desc.lower())
        self.assertNotIn("like and subscribe", desc.lower())

        # Hashtags must be 4–6 and must NOT contain #viral
        tags = meta["hashtags"]
        self.assertTrue(4 <= len(tags) <= 6, f"Expected 4-6 hashtags, got: {tags}")
        self.assertNotIn("#viral", [t.lower() for t in tags])
        self.assertNotIn("#fyp", [t.lower() for t in tags])
        self.assertIn("#zakirkhan", [t.lower() for t in tags])
        self.assertIn("#relationships", [t.lower() for t in tags])

    # 21. Existing claim-strength tests continue passing
    def test_21_existing_claim_strength_preserved(self):
        supported_claim = "Two quick inhales and a long exhale immediately drop your heart rate"
        is_valid, _ = hook_generator.validate_claim_strength(supported_claim, self.sleep_transcript)
        self.assertTrue(is_valid)

        unsupported_absolute = "This will 100% cure all diseases instantly guaranteed"
        is_invalid, _ = hook_generator.validate_claim_strength(unsupported_absolute, self.sleep_transcript)
        self.assertFalse(is_invalid)

    # 22. Existing package scoring tests continue passing
    def test_22_package_scoring_continues_passing(self):
        hook_cand = {"hook": "Why We Misunderstand Love ❤️", "score": 9.2, "curiosity": 9.0}
        title_cand = {"title": "How Active Communication Protects Relationships #shorts", "title_score": 9.0, "curiosity": 8.5}
        desc_cand = {
            "description": "Zakir Khan explains why expecting perfection ruins relationships, and how communication resolves conflicts.",
            "desc_score": 8.8,
        }
        hashtags = ["#ZakirKhan", "#Relationships", "#Love", "#LifeLessons", "#PodcastClips"]

        pkg_score, dims = metadata_ai.score_package(
            hook_cand=hook_cand,
            title_cand=title_cand,
            desc_cand=desc_cand,
            hashtags=hashtags,
            transcript=self.relationship_transcript,
        )
        self.assertGreaterEqual(pkg_score, 70.0)
        self.assertIn("scroll_stop", dims)
        self.assertIn("curiosity", dims)


if __name__ == "__main__":
    unittest.main()
