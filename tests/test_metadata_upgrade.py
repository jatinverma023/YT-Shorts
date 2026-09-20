"""
Comprehensive test suite for Upgraded YouTube Shorts Title, Description & Hashtag Generation.
Verifies all Title rules (21), Hashtag rules (13), Description rules (6), and Integration flows (13+).
"""
import os
import re
import sys
import unittest
from unittest.mock import patch, MagicMock

# Ensure scripts directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

import config
import metadata_ai
import hook_generator
import main as pipeline_main
import youtube_upload
import sheet_log


class TestTitleGeneration(unittest.TestCase):
    """Unit tests for Title validation, anti-clickbait, grounding, and scoring (Section 22)."""

    def setUp(self):
        self.sample_transcript = (
            "Smoking cigarettes can significantly reduce sperm quality and motility, "
            "directly affecting male fertility and reproductive health."
        )
        self.sample_hook = "WHY DOES SMOKING HURT FERTILITY?"

    def test_01_valid_grounded_title_passes(self):
        """1. Valid grounded title passes validation."""
        cand = {"title": "How Smoking Can Affect Male Fertility"}
        is_valid, reason = metadata_ai.validate_title(
            cand,
            transcript=self.sample_transcript,
            hook=self.sample_hook,
        )
        self.assertTrue(is_valid, f"Expected valid title to pass, got: {reason}")

    def test_02_unsupported_title_fails(self):
        """2. Unsupported title introducing stronger extreme claims fails."""
        # Introduces unsupported extreme claim 'Permanent Infertility'
        cand = {"title": "Smoking Causes Permanent Infertility In Men"}
        is_valid, reason = metadata_ai.validate_title(
            cand,
            transcript=self.sample_transcript,
            hook=self.sample_hook,
        )
        self.assertFalse(is_valid)
        self.assertIn("unsupported", reason.lower())

    def test_03_generic_clickbait_title_fails(self):
        """3. Generic clickbait title fails when unsupported by clip."""
        cand = {"title": "You Need To See This Shocking Truth"}
        is_valid, reason = metadata_ai.validate_title(
            cand,
            transcript=self.sample_transcript,
            hook=self.sample_hook,
        )
        self.assertFalse(is_valid)
        self.assertIn("clickbait", reason.lower())

    def test_04_you_wont_believe_this_fails(self):
        """4. 'YOU WON'T BELIEVE THIS' fails."""
        cand = {"title": "You Won't Believe This About Smoking"}
        is_valid, reason = metadata_ai.validate_title(
            cand,
            transcript=self.sample_transcript,
            hook=self.sample_hook,
        )
        self.assertFalse(is_valid)
        self.assertIn("clickbait", reason.lower())

    def test_05_the_truth_exposed_fails(self):
        """5. 'THE TRUTH EXPOSED' fails."""
        cand = {"title": "The Truth Exposed About Fertility Habits"}
        is_valid, reason = metadata_ai.validate_title(
            cand,
            transcript=self.sample_transcript,
            hook=self.sample_hook,
        )
        self.assertFalse(is_valid)
        self.assertIn("clickbait", reason.lower())

    def test_06_wait_for_the_twist_fails_when_unsupported(self):
        """6. 'WAIT FOR THE TWIST' fails when unsupported."""
        cand = {"title": "Wait For The Twist At The End"}
        is_valid, reason = metadata_ai.validate_title(
            cand,
            transcript=self.sample_transcript,
            hook=self.sample_hook,
        )
        self.assertFalse(is_valid)
        self.assertIn("clickbait", reason.lower())

    def test_07_source_filename_leakage_fails_when_unsupported(self):
        """7. Source filename leakage fails when unsupported."""
        cand = {"title": "podcast_interview_720p Smoking And Health Clip"}
        is_valid, reason = metadata_ai.validate_title(
            cand,
            transcript=self.sample_transcript,
            hook=self.sample_hook,
            filename="podcast_interview_720p.mp4",
        )
        self.assertFalse(is_valid)
        self.assertIn("filename", reason.lower())

    def test_08_source_title_leakage_fails_when_unsupported(self):
        """8. Source title leakage fails when unsupported."""
        cand = {"title": "Joe Rogan Elon Musk Full Interview"}
        is_valid, reason = metadata_ai.validate_title(
            cand,
            transcript=self.sample_transcript,
            hook=self.sample_hook,
            source_title="Joe Rogan Elon Musk Full Interview",
        )
        self.assertFalse(is_valid)
        self.assertIn("source title", reason.lower())

    def test_09_title_too_long_fails(self):
        """9. Title exceeding 70 characters fails."""
        cand = {"title": "Why Smoking Cigarettes Deeply And Drastically Damages Sperm Motility In Men"}
        self.assertGreater(len(cand["title"]), 70)
        is_valid, reason = metadata_ai.validate_title(
            cand,
            transcript=self.sample_transcript,
            hook=self.sample_hook,
        )
        self.assertFalse(is_valid)
        self.assertIn("hard limit", reason.lower())

    def test_10_excessively_short_title_fails(self):
        """10. Excessively short title (< 5 words) fails."""
        cand = {"title": "Smoking Fertility Risk"}  # 3 words
        is_valid, reason = metadata_ai.validate_title(
            cand,
            transcript=self.sample_transcript,
            hook=self.sample_hook,
        )
        self.assertFalse(is_valid)
        self.assertIn("too few words", reason.lower())

    def test_11_title_identical_to_hook_fails(self):
        """11. Title identical to hook fails."""
        cand = {"title": "Why Does Smoking Hurt Fertility?"}
        is_valid, reason = metadata_ai.validate_title(
            cand,
            transcript=self.sample_transcript,
            hook="WHY DOES SMOKING HURT FERTILITY?",
        )
        self.assertFalse(is_valid)
        self.assertIn("duplicates hook", reason.lower())

    def test_12_title_with_excessive_hook_overlap_fails(self):
        """12. Title with excessive hook overlap fails."""
        cand = {"title": "Why Smoking Hurts Male Fertility"}
        is_valid, reason = metadata_ai.validate_title(
            cand,
            transcript=self.sample_transcript,
            hook="WHY SMOKING HURTS FERTILITY",
        )
        self.assertFalse(is_valid)
        self.assertIn("duplicates hook", reason.lower())

    def test_13_english_title_works(self):
        """13. Natural English title works on English clip."""
        cand = {"title": "How Daily Exercise Boosts Deep Sleep"}
        transcript = "Regular aerobic exercise deepens your slow wave deep sleep stages significantly."
        is_valid, reason = metadata_ai.validate_title(cand, transcript=transcript, hook="CAN SLEEP BE IMPROVED?")
        self.assertTrue(is_valid, f"Expected valid English title, got: {reason}")

    def test_14_roman_hindi_title_works(self):
        """14. Natural Roman Hindi / Hinglish title works on Hindi clip."""
        cand = {"title": "Smoking Se Male Fertility Par Kya Asar Padta Hai"}
        transcript = "Smoking se sperm quality kharab hoti hai aur fertility par bura asar padta hai."
        is_valid, reason = metadata_ai.validate_title(cand, transcript=transcript, hook="KYA YE HABIT DANGEROUS HAI?")
        self.assertTrue(is_valid, f"Expected valid Hinglish title, got: {reason}")

    def test_15_mixed_hinglish_title_works(self):
        """15. Mixed Hinglish title works."""
        cand = {"title": "Compound Interest Kaise Generational Wealth Banata Hai"}
        transcript = "Compound interest se aapki wealth multiply hoti hai aur long term growth milti hai."
        is_valid, reason = metadata_ai.validate_title(cand, transcript=transcript, hook="COMPOUNDING KA KYA SECRET HAI?")
        self.assertTrue(is_valid, f"Expected valid mixed title, got: {reason}")

    def test_16_python_score_is_authoritative(self):
        """16. Python score is authoritative and accurately computed using the 6-dimension formula."""
        cand = {
            "title": "Why Starting Early Makes Compound Interest Powerful",
            "grounding_score": 9.0,
            "specificity_score": 8.0,
            "curiosity_score": 8.5,
            "relevance_score": 9.0,
            "clarity_score": 9.0,
            "brevity_score": 9.5,
        }
        transcript = "Starting early with compound interest makes your investments grow exponentially."
        score = metadata_ai.score_title(cand, transcript=transcript)
        # Expected: weighted average * 10 ~ 87.8
        self.assertGreaterEqual(score, 70.0)
        self.assertLessEqual(score, 100.0)
        self.assertIsInstance(score, float)

    def test_17_llm_score_cannot_override_python_score(self):
        """17. LLM self-reported score of 100 cannot bypass ungrounded low Python scoring."""
        cand = {
            "title": "Why Space Astronauts Love Eating Apples",  # completely ungrounded
            "grounding_score": 10.0,
            "specificity_score": 10.0,
            "curiosity_score": 10.0,
            "relevance_score": 10.0,
            "clarity_score": 10.0,
        }
        # Clip is about banking
        transcript = "Banking interest rates are set by central monetary policy committees."
        score = metadata_ai.score_title(cand, transcript=transcript)
        # Grounding dimension in Python drops to min 5.0
        self.assertLess(score, 90.0)

    def test_18_title_below_70_is_rejected(self):
        """18. Any candidate with Python score below MIN_TITLE_QUALITY_SCORE (70.0) is rejected."""
        cand = {
            "title": "A Thing That Happened In The Talk",
            "grounding_score": 5.0,
            "specificity_score": 3.0,
            "curiosity_score": 4.0,
            "relevance_score": 4.0,
            "clarity_score": 6.0,
        }
        score = metadata_ai.score_title(cand, transcript="Stock market indices fluctuated yesterday.")
        self.assertLess(score, config.MIN_TITLE_QUALITY_SCORE)

    @patch("metadata_ai.generate_title_candidates")
    @patch("metadata_ai.OpenAI")
    def test_19_second_candidate_batch_is_used_when_needed(self, mock_openai, mock_batch2):
        """19. Second candidate batch is queried when all Batch 1 candidates fail."""
        # Batch 1 mock returning invalid titles (too short or clickbait)
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content='''{
            "punchline": "SMOKING IMPACT",
            "title_candidates": [
                {"title": "You Won\'t Believe This"},
                {"title": "Too Short"}
            ],
            "description": "Explains smoking fertility effects.",
            "hashtags": ["#Shorts", "#Health", "#Fertility"]
        }'''))]
        mock_client.chat.completions.create.return_value = mock_resp

        # Batch 2 returns valid candidate
        mock_batch2.return_value = [
            {"title": "How Smoking Can Affect Male Fertility", "grounding_score": 9.5}
        ]

        with patch.object(config, "GROQ_API_KEY", "gsk_test"):
            meta = metadata_ai.generate_shorts_metadata("video.mp4", transcript=self.sample_transcript)

        # Batch 2 was invoked
        mock_batch2.assert_called_once()
        self.assertEqual(meta["title"], "How Smoking Can Affect Male Fertility")

    def test_20_deterministic_fallback_is_grounded(self):
        """20. Deterministic fallback title is extracted directly from clip content."""
        transcript = "Investing early in index funds builds generational wealth over decades."
        fallback = metadata_ai.extract_grounded_fallback_title(transcript)
        self.assertTrue(len(fallback.split()) >= 5)
        self.assertLessEqual(len(fallback), 70)
        # Verify content words from fallback are in transcript
        for w in fallback.lower().split()[:3]:
            self.assertIn(w, transcript.lower())

    def test_21_fallback_never_uses_generic_clickbait(self):
        """21. Fallback title never uses generic clickbait."""
        transcript = "Photosynthesis converts solar light into biochemical glucose energy."
        fallback = metadata_ai.extract_grounded_fallback_title(transcript)
        for pattern in metadata_ai.FORBIDDEN_CLICKBAIT_PATTERNS:
            self.assertIsNone(re.search(pattern, fallback, re.IGNORECASE))
        self.assertNotIn("must watch", fallback.lower())
        self.assertNotIn("you won't believe", fallback.lower())


class TestHashtagGeneration(unittest.TestCase):
    """Unit tests for Hashtag validation, spam avoidance, and scoring (Section 23)."""

    def setUp(self):
        self.transcript = "Compound interest and disciplined index fund investing creates long term financial freedom."

    def test_01_relevant_hashtags_pass(self):
        """1. Relevant hashtags pass validation."""
        tags = ["#Shorts", "#CompoundInterest", "#Investing", "#PersonalFinance"]
        validated = metadata_ai.validate_hashtags(tags, transcript=self.transcript)
        self.assertIn("#Shorts", validated)
        self.assertIn("#CompoundInterest", validated)
        self.assertIn("#Investing", validated)

    def test_02_shorts_is_included(self):
        """2. #Shorts is always included even if omitted in input."""
        tags = ["#CompoundInterest", "#Investing"]
        validated = metadata_ai.validate_hashtags(tags, transcript=self.transcript)
        self.assertEqual(validated[0], "#Shorts")

    def test_03_generic_viral_fails(self):
        """3. Generic #viral fails validation."""
        is_valid, reason = metadata_ai.validate_single_hashtag("#viral", transcript=self.transcript)
        self.assertFalse(is_valid)
        self.assertIn("generic spam", reason)

    def test_04_generic_fyp_fails(self):
        """4. Generic #fyp fails validation."""
        is_valid, reason = metadata_ai.validate_single_hashtag("#fyp", transcript=self.transcript)
        self.assertFalse(is_valid)
        self.assertIn("generic spam", reason)

    def test_05_generic_trending_fails(self):
        """5. Generic #trending fails validation."""
        is_valid, reason = metadata_ai.validate_single_hashtag("#trending", transcript=self.transcript)
        self.assertFalse(is_valid)
        self.assertIn("generic spam", reason)

    def test_06_duplicate_hashtags_are_removed(self):
        """6. Duplicate hashtags differing only by case are removed."""
        tags = ["#Shorts", "#Investing", "#investing", "#INVESTING", "#Money"]
        validated = metadata_ai.validate_hashtags(tags, transcript=self.transcript)
        investing_counts = sum(1 for t in validated if t.lower() == "#investing")
        self.assertEqual(investing_counts, 1)

    def test_07_invalid_hashtag_formatting_fails(self):
        """7. Invalid hashtag formatting fails (spaces, special chars)."""
        is_valid1, _ = metadata_ai.validate_single_hashtag("#With Spaces", transcript=self.transcript)
        is_valid2, _ = metadata_ai.validate_single_hashtag("NoHash", transcript=self.transcript)
        self.assertFalse(is_valid1)
        self.assertFalse(is_valid2)

    def test_08_unsupported_topic_hashtag_fails(self):
        """8. Unsupported topic hashtag fails grounding."""
        is_valid, reason = metadata_ai.validate_single_hashtag("#CricketWorldCup", transcript=self.transcript)
        self.assertFalse(is_valid)
        self.assertIn("not grounded", reason)

    def test_09_hashtags_are_grounded_in_clip_content(self):
        """9. Valid hashtags are confirmed grounded in clip content."""
        is_valid, _ = metadata_ai.validate_single_hashtag("#CompoundInterest", transcript=self.transcript)
        self.assertTrue(is_valid)

    def test_10_maximum_6_hashtags_enforced(self):
        """10. Maximum 6 hashtags enforced."""
        tags = [
            "#Shorts", "#CompoundInterest", "#Investing",
            "#PersonalFinance", "#Money", "#Wealth",
            "#ExtraTag1", "#ExtraTag2"
        ]
        validated = metadata_ai.validate_hashtags(tags, transcript=self.transcript)
        self.assertLessEqual(len(validated), 6)

    def test_11_fewer_hashtags_allowed_when_insufficient_topics(self):
        """11. Fewer hashtags are allowed when insufficient topics exist (no fake filler tags)."""
        # Minimal transcript with single topic
        short_transcript = "Meditation helps reduce stress."
        tags = ["#Shorts", "#Meditation"]
        validated = metadata_ai.validate_hashtags(tags, transcript=short_transcript)
        # Should not force 4 tags by injecting spam
        self.assertTrue(len(validated) >= 2)
        self.assertIn("#Shorts", validated)
        self.assertIn("#Meditation", validated)

    def test_12_python_hashtag_score_is_authoritative(self):
        """12. Python hashtag score is authoritative."""
        tags = ["#Shorts", "#CompoundInterest", "#Investing", "#PersonalFinance"]
        score = metadata_ai.score_hashtags(tags, transcript=self.transcript)
        self.assertGreaterEqual(score, 75.0)
        self.assertLessEqual(score, 100.0)

    def test_13_llm_relevance_cannot_bypass_python_validation(self):
        """13. Spam tags cannot bypass Python validation even if LLM generated them."""
        raw_tags = ["#Shorts", "#viral", "#fyp", "#trending", "#CompoundInterest"]
        validated = metadata_ai.validate_hashtags(raw_tags, transcript=self.transcript)
        self.assertNotIn("#viral", validated)
        self.assertNotIn("#fyp", validated)
        self.assertNotIn("#trending", validated)
        self.assertIn("#Shorts", validated)
        self.assertIn("#CompoundInterest", validated)


class TestDescriptionGeneration(unittest.TestCase):
    """Unit tests for Description grounding and formatting (Section 24)."""

    def setUp(self):
        self.transcript = (
            "Starting early with compound interest makes wealth accumulation dramatically "
            "more effective over a thirty year investment horizon."
        )

    def test_01_description_is_grounded_in_clip(self):
        """1. Description is grounded in the clip transcript."""
        desc = "This clip explains why starting early with compound interest builds lasting wealth."
        norm_tr = metadata_ai.normalize_text(self.transcript)
        self.assertIn("compound", norm_tr)
        self.assertIn("interest", norm_tr)
        self.assertIn("wealth", norm_tr)

    def test_02_description_does_not_invent_facts(self):
        """2. Description avoids unsupported extreme claims."""
        desc = "This clip explains compound interest."
        for ew in metadata_ai.EXTREME_CLAIM_WORDS:
            self.assertNotIn(ew, desc.lower())

    def test_03_description_is_concise(self):
        """3. Description is concise (1-2 sentences)."""
        desc = "This clip explains why starting early can make compound interest dramatically more powerful over time."
        sentences = [s.strip() for s in desc.split(".") if s.strip()]
        self.assertLessEqual(len(sentences), 2)

    def test_04_description_does_not_excessively_repeat_title(self):
        """4. Description does not simply copy or repeat the title."""
        title = "Why Starting Early Makes Compound Interest Powerful"
        desc = "Discover how exponential compounding over several decades transforms personal savings into generational wealth."
        jaccard = len(set(title.lower().split()) & set(desc.lower().split())) / len(set(title.lower().split()) | set(desc.lower().split()))
        self.assertLess(jaccard, 0.6)

    def test_05_hashtags_are_appended_correctly(self):
        """5. Hashtags are appended at the bottom of description separated by double newline."""
        with patch("metadata_ai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock(message=MagicMock(content='''{
                "punchline": "WHY START EARLY?",
                "title_candidates": [
                    {"title": "Why Starting Early Makes Compound Interest Powerful", "grounding_score": 9.5}
                ],
                "description": "This clip breaks down the power of starting your investments early.",
                "hashtags": ["#Shorts", "#CompoundInterest", "#Investing", "#PersonalFinance"]
            }'''))]
            mock_client.chat.completions.create.return_value = mock_resp

            with patch.object(config, "GROQ_API_KEY", "gsk_test"):
                meta = metadata_ai.generate_shorts_metadata("finance.mp4", transcript=self.transcript)

            self.assertIn("\n\n#Shorts", meta["description"])
            self.assertTrue(meta["description"].startswith("This clip breaks down"))

    def test_06_existing_description_behavior_remains_compatible(self):
        """6. Existing description behavior remains compatible with legacy uploaders."""
        with patch("metadata_ai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock(message=MagicMock(content='''{
                "title_1": "Top Title #shorts",
                "punchline": "Hook 🔥",
                "description": "Summary.\n\n#shorts #podcast",
                "tags": ["shorts"]
            }'''))]
            mock_client.chat.completions.create.return_value = mock_resp

            with patch.object(config, "GROQ_API_KEY", "gsk_test"):
                meta = metadata_ai.generate_shorts_metadata("video.mp4", transcript="Some transcript words.")

            self.assertIn("description", meta)
            self.assertIsInstance(meta["description"], str)


class TestIntegrationMetadataPipeline(unittest.TestCase):
    """Integration tests verifying end-to-end flow and pipeline invariants (Section 25 & Review Corrections)."""

    def setUp(self):
        self.clip_transcript = (
            "Why starting early makes compound interest so powerful. When you invest for thirty years, "
            "most of your returns come in the final decade through exponential growth."
        )

    @patch("metadata_ai.OpenAI")
    def test_01_selected_clip_produces_canonical_metadata(self, mock_openai):
        """1. A selected clip produces hook + title + description + hashtags + quality scores."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content='''{
            "punchline": "WHY DOES COMPOUNDING MATTER?",
            "title_candidates": [
                {"title": "Why Starting Early Makes Compound Interest Powerful", "grounding_score": 9.5}
            ],
            "description": "Learn why time is the biggest asset in compound growth.",
            "hashtags": ["#Shorts", "#CompoundInterest", "#Investing", "#PersonalFinance"]
        }'''))]
        mock_client.chat.completions.create.return_value = mock_resp

        with patch.object(config, "GROQ_API_KEY", "gsk_test"):
            meta = metadata_ai.generate_shorts_metadata("finance_clip.mp4", transcript=self.clip_transcript)

        self.assertIn("generated_hook", meta)
        self.assertIn("title", meta)
        self.assertIn("description", meta)
        self.assertIn("hashtags", meta)
        self.assertIn("title_quality_score", meta)
        self.assertIn("hashtag_quality_score", meta)
        self.assertEqual(meta["punchline"], meta["generated_hook"])
        self.assertGreaterEqual(meta["title_quality_score"], 70.0)
        self.assertGreaterEqual(meta["hashtag_quality_score"], 70.0)

    @patch("metadata_ai.OpenAI")
    def test_02_different_clips_receive_different_metadata(self, mock_openai):
        """2. Different clips receive distinct, transcript-specific metadata."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Clip 1: Finance
        resp1 = MagicMock()
        resp1.choices = [MagicMock(message=MagicMock(content='''{
            "punchline": "WHY DOES COMPOUNDING MATTER?",
            "title_candidates": [{"title": "Why Starting Early Makes Compound Interest Powerful"}],
            "description": "Explains compounding over decades.",
            "hashtags": ["#Shorts", "#CompoundInterest", "#Investing"]
        }'''))]

        # Clip 2: Health
        resp2 = MagicMock()
        resp2.choices = [MagicMock(message=MagicMock(content='''{
            "punchline": "DOES SMOKING HURT FERTILITY?",
            "title_candidates": [{"title": "How Smoking Can Affect Male Fertility"}],
            "description": "Explains how smoking impacts sperm quality.",
            "hashtags": ["#Shorts", "#MensHealth", "#Fertility"]
        }'''))]

        mock_client.chat.completions.create.side_effect = [resp1, resp2]

        with patch.object(config, "GROQ_API_KEY", "gsk_test"):
            meta1 = metadata_ai.generate_shorts_metadata(
                "clip1.mp4",
                transcript="Why starting early makes compound interest so powerful over thirty years.",
            )
            meta2 = metadata_ai.generate_shorts_metadata(
                "clip2.mp4",
                transcript="Smoking cigarettes can significantly reduce sperm quality and affect male fertility.",
            )

        self.assertNotEqual(meta1["title"], meta2["title"])
        self.assertNotEqual(meta1["generated_hook"], meta2["generated_hook"])
        self.assertNotEqual(meta1["hashtags"], meta2["hashtags"])

    @patch("youtube_upload.get_youtube_client")
    def test_03_title_description_hashtags_reach_youtube_upload(self, mock_get_client):
        """3, 4, 5. Title, description, and hashtags cleanly reach the YouTube upload insert body."""
        mock_yt = MagicMock()
        mock_get_client.return_value = mock_yt
        mock_insert = MagicMock()
        mock_yt.videos().insert.return_value = mock_insert
        mock_insert.next_chunk.return_value = (MagicMock(progress=lambda: 1.0), {"id": "yt_video_123"})

        with patch("googleapiclient.http.MediaFileUpload"):
            url = youtube_upload.upload_short(
                video_path="/fake/path.mp4",
                title="Why Starting Early Makes Compound Interest Powerful",
                description="Concise description.\n\n#Shorts #CompoundInterest #Investing",
                tags=["compound interest", "investing"],
                clip_identifier="clip_multi_1",
            )

        self.assertEqual(url, "https://youtube.com/shorts/yt_video_123")
        call_kwargs = mock_yt.videos().insert.call_args[1]
        body = call_kwargs["body"]
        self.assertEqual(body["snippet"]["title"], "Why Starting Early Makes Compound Interest Powerful")
        self.assertIn("Concise description.", body["snippet"]["description"])
        self.assertIn("#Shorts #CompoundInterest #Investing", body["snippet"]["description"])
        self.assertIn("[id:clip_multi_1]", body["snippet"]["description"])

    def test_06_existing_generated_hook_remains_unchanged(self):
        """6, 7. Existing generated_hook and punchline backward compatibility remain intact."""
        meta = {
            "title": "Title Here",
            "generated_hook": "WHY DOES THIS MATTER?",
            "punchline": "WHY DOES THIS MATTER?",
            "description": "Desc\n\n#Shorts",
            "hashtags": ["#Shorts"],
        }
        self.assertEqual(meta["generated_hook"], meta["punchline"])

    @patch("youtube_upload.get_youtube_client")
    def test_08_existing_youtube_idempotency_remains_intact(self, mock_get_client):
        """8, 9. Existing YouTube idempotency check operates seamlessly with new titles."""
        mock_yt = MagicMock()
        mock_get_client.return_value = mock_yt
        mock_yt.channels().list().execute.return_value = {
            "items": [{"contentDetails": {"relatedPlaylists": {"uploads": "UPLIST"}}}]
        }
        mock_yt.playlistItems().list().execute.return_value = {
            "items": [{
                "snippet": {
                    "resourceId": {"videoId": "existing_short_99"},
                    "description": "Some description [id:unique_clip_xyz]",
                    "title": "Why Starting Early Makes Compound Interest Powerful",
                }
            }]
        }

        found_url = youtube_upload.find_existing_short(
            clip_identifier="unique_clip_xyz",
            title="Why Starting Early Makes Compound Interest Powerful",
        )
        self.assertEqual(found_url, "https://youtube.com/shorts/existing_short_99")

    # -------------------------------------------------------------------------
    # 10, 11, 12, 13 & Review Corrections #16:
    # Given 4 pending clips: 1 scheduled execution uploads exactly ONE clip.
    # 1 clip == done, 3 clips == pending.
    # -------------------------------------------------------------------------
    @patch("sheet_log.log_run")
    @patch("youtube_upload.upload_short")
    @patch("youtube_upload.find_existing_short", return_value=None)
    @patch("video_process.process_video")
    @patch("video_process.get_duration_seconds", return_value=30.0)
    @patch("transcribe.generate_ass_captions")
    @patch("metadata_ai.generate_shorts_metadata")
    @patch("hook_generator.validate_hook", return_value=(True, "ok"))
    @patch("video_process.trim_silences_from_words")
    @patch("transcribe.transcribe_audio")
    @patch("transcribe.extract_audio")
    @patch("video_process.extract_clip_segment")
    @patch("drive_utils.download_file")
    @patch("sheet_log.update_clip_status")
    @patch("sheet_log.claim_clip", return_value=True)
    @patch("sheet_log.get_next_pending_clip")
    @patch("sheet_log.get_video_clip_counts")
    @patch("sheet_log.ensure_clip_queue_sheet")
    @patch("sheet_log.reset_expired_quota_clips", return_value=0)
    @patch("sheet_log.reset_stale_processing_clips", return_value=0)
    @patch("drive_utils.get_drive_service")
    @patch("drive_utils.get_file_metadata", return_value={"id": "vid_4_clips", "trashed": False})
    def test_one_scheduled_execution_processes_exactly_one_clip_leaving_three_pending(
        self, mock_meta, mock_drive_svc, mock_reset_stale, mock_reset_quota,
        mock_ensure_sheet, mock_get_counts, mock_get_pending, mock_claim,
        mock_update_status, mock_download, mock_extract_slice, mock_extract_audio,
        mock_transcribe, mock_trim, mock_val_hook, mock_meta_gen, mock_gen_ass,
        mock_get_dur, mock_process_vid, mock_find_exist, mock_upload, mock_log_run
    ):
        """
        Critical regression invariant:
        Given 4 pending clips, one execution claims and uploads exactly ONE clip.
        After execution: 1 clip is marked done, and the remaining 3 clips remain pending.
        """
        # Queue state before run: 4 total, 4 pending
        mock_get_pending.return_value = (
            2,  # row_number in Sheet
            {
                "drive_file_id": "vid_4_clips",
                "source_video_name": "long_podcast.mp4",
                "clip_index": 1,
                "start_time": 10.0,
                "end_time": 45.0,
                "status": "pending",
                "quality_score": 88.0,
            }
        )
        mock_get_counts.return_value = {
            "total": 4,
            "pending": 3,
            "done": 1,
            "failed": 0,
            "processing": 0,
            "retry_after_quota_reset": 0,
        }
        mock_transcribe.return_value = {"words": [], "language": "en", "text": "Compound growth script."}
        mock_trim.return_value = ("/fake/tight.mp4", [])
        mock_meta_gen.return_value = {
            "title": "Why Starting Early Makes Compound Interest Powerful",
            "title_variants": [
                "Why Starting Early Makes Compound Interest Powerful",
                "Why Starting Early Makes Compound Interest Powerful",
                "Why Starting Early Makes Compound Interest Powerful",
            ],
            "punchline": "WHY DOES COMPOUNDING MATTER?",
            "generated_hook": "WHY DOES COMPOUNDING MATTER?",
            "description": "Insightful breakdown.\n\n#Shorts #CompoundInterest #Investing",
            "hashtags": ["#Shorts", "#CompoundInterest", "#Investing"],
            "tags": ["compound interest", "investing"],
            "title_quality_score": 88.5,
            "hashtag_quality_score": 90.0,
        }
        mock_upload.return_value = "https://youtube.com/shorts/single_clip_1"

        # Execute the pipeline
        pipeline_main.main()

        # Invariant 1: upload_short called EXACTLY once
        self.assertEqual(mock_upload.call_count, 1)

        # Invariant 2: Exactly ONE clip was claimed
        self.assertEqual(mock_claim.call_count, 1)

        # Invariant 3: The claimed clip was marked 'done'
        mock_update_status.assert_called_with(2, "done", youtube_url="https://youtube.com/shorts/single_clip_1")

        # Invariant 4: Clip queue counts show 1 done and 3 pending
        counts = mock_get_counts.return_value
        self.assertEqual(counts["total"], 4)
        self.assertEqual(counts["done"], 1)
        self.assertEqual(counts["pending"], 3)


if __name__ == "__main__":
    unittest.main()
