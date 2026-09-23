"""
Comprehensive production-path tests for YouTube Shorts Content Packaging:
TEST A: Valid packaging response reaches youtube_upload.upload_short (title, description, tags, hashtags).
TEST B: The hook burned into video captions matches the package hook (consistent packaging).
TEST C: Truncated (finish_reason='length') or malformed LLM responses are rejected and marked as fallback.
TEST D: Filename fallback is defensive-only and never overrides valid packaging metadata.
TEST E: Source-title / filename leakage remains strictly rejected by title validation.
TEST F: Claim-strength validation remains active.
"""
import unittest
from unittest.mock import MagicMock, patch
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

import config
import hook_generator
import metadata_ai
import youtube_upload
import main as pipeline_main


class TestProductionMetadataPipeline(unittest.TestCase):
    """Test suite verifying end-to-end wiring of the Content Packaging System into production."""

    def setUp(self):
        self.sample_transcript = (
            "When you are having trouble falling asleep at night, your autonomic nervous system "
            "is stuck in high arousal. By doing two quick inhales through the nose and a long exhale "
            "through the mouth, you immediately drop your heart rate and turn off the mind to sleep."
        )
        self.video_name = "Andrew Huberman Become Mentally Dangerous With These Daily Habits FO556 Raj Shamani.mp4"
        self.clip = {
            "drive_file_id": "vid_huberman_1",
            "source_video_name": self.video_name,
            "clip_index": 1,
            "start_time": 10.0,
            "end_time": 45.0,
            "hook_summary": "How to turn off your mind to sleep",
            "punchline": "TURN OFF YOUR MIND TO SLEEP",
            "youtube_url": "",
        }

    @patch("main.sheet_log.update_clip_status")
    @patch("main.sheet_log.log_run")
    @patch("main.notify.send")
    @patch("main.drive_utils.download_file")
    @patch("main.video_process.extract_clip_segment")
    @patch("main.transcribe.extract_audio")
    @patch("main.transcribe.transcribe_audio")
    @patch("main.video_process.trim_silences_from_words")
    @patch("main.video_process.get_duration_seconds", return_value=35.0)
    @patch("main.transcribe.generate_ass_captions")
    @patch("main.video_process.process_video")
    @patch("main.youtube_upload.find_existing_short", return_value=None)
    @patch("main.youtube_upload.upload_short")
    @patch("main.metadata_ai.generate_shorts_metadata")
    def test_a_valid_packaging_response_reaches_upload(
        self,
        mock_meta,
        mock_upload,
        mock_find_existing,
        mock_process_video,
        mock_gen_ass,
        mock_get_duration,
        mock_trim,
        mock_transcribe,
        mock_extract_audio,
        mock_extract_clip,
        mock_download,
        mock_notify,
        mock_log_run,
        mock_update_status,
    ):
        """TEST A: Valid packaging metadata (title, desc, hashtags, tags) reaches youtube_upload.upload_short."""
        expected_hook = "TURN OFF YOUR MIND TO SLEEP"
        expected_title = "How To Fall Asleep Fast By Calming Your Nervous System #shorts"
        expected_desc = (
            "Double inhales through your nose trigger rapid autonomic deceleration.\n\n"
            "#Shorts #SleepScience #HubermanLab #Mindset"
        )
        expected_hashtags = ["#Shorts", "#SleepScience", "#HubermanLab", "#Mindset"]
        expected_tags = ["sleep science", "huberman lab", "nervous system", "shorts"]

        mock_meta.return_value = {
            "title": expected_title,
            "title_variants": [expected_title, "Calm Your Nervous System For Deep Sleep #shorts", "Turn Off Brain Before Sleep #shorts"],
            "punchline": expected_hook,
            "generated_hook": expected_hook,
            "description": expected_desc,
            "hashtags": expected_hashtags,
            "tags": expected_tags,
            "title_quality_score": 92.0,
            "hashtag_quality_score": 90.0,
            "hook_quality_score": 94.0,
            "description_quality_score": 88.0,
            "package_quality_score": 91.5,
            "hook_strategy": "consequence",
            "title_strategy": "specific_explanation",
            "description_strategy": "micro_teaser",
            "is_fallback": False,
        }

        mock_transcribe.return_value = {
            "words": [{"word": "sleep", "start": 0.0, "end": 1.0}],
            "language": "en",
            "text": self.sample_transcript,
        }
        mock_trim.return_value = ("/fake/tight.mp4", [{"word": "sleep", "start": 0.0, "end": 1.0}])
        mock_upload.return_value = "https://youtube.com/shorts/test_packaging_123"

        mock_drive = MagicMock()
        success, url = pipeline_main.process_queue_clip(
            mock_drive, row_number=2, clip=self.clip, local_src_path="/fake/src.mp4"
        )

        self.assertTrue(success)
        self.assertEqual(url, "https://youtube.com/shorts/test_packaging_123")
        mock_upload.assert_called_once()

        # Verify exact parameters reaching upload_short
        _, kwargs = mock_upload.call_args
        args = mock_upload.call_args[0]
        call_title = kwargs.get("title") or (args[1] if len(args) > 1 else None)
        call_desc = kwargs.get("description") or (args[2] if len(args) > 2 else None)
        call_tags = kwargs.get("tags") or (args[3] if len(args) > 3 else None)
        call_hashtags = kwargs.get("hashtags")

        self.assertEqual(call_title, expected_title)
        self.assertIn("Double inhales", call_desc)
        self.assertEqual(call_tags, expected_tags)
        self.assertEqual(call_hashtags, expected_hashtags)

    @patch("main.sheet_log.update_clip_status")
    @patch("main.sheet_log.log_run")
    @patch("main.notify.send")
    @patch("main.drive_utils.download_file")
    @patch("main.video_process.extract_clip_segment")
    @patch("main.transcribe.extract_audio")
    @patch("main.transcribe.transcribe_audio")
    @patch("main.video_process.trim_silences_from_words")
    @patch("main.video_process.get_duration_seconds", return_value=35.0)
    @patch("main.transcribe.generate_ass_captions")
    @patch("main.video_process.process_video")
    @patch("main.youtube_upload.find_existing_short", return_value=None)
    @patch("main.youtube_upload.upload_short")
    @patch("main.metadata_ai.generate_shorts_metadata")
    def test_b_burned_hook_matches_package_hook(
        self,
        mock_meta,
        mock_upload,
        mock_find_existing,
        mock_process_video,
        mock_gen_ass,
        mock_get_duration,
        mock_trim,
        mock_transcribe,
        mock_extract_audio,
        mock_extract_clip,
        mock_download,
        mock_notify,
        mock_log_run,
        mock_update_status,
    ):
        """TEST B: The hook burned into the video captions matches the packaging system's validated hook."""
        package_hook = "DE-EXCITE YOUR BRAIN FOR SLEEP"
        mock_meta.return_value = {
            "title": "Calm Autonomic Arousal To Fall Asleep Fast #shorts",
            "title_variants": ["Calm Autonomic Arousal To Fall Asleep Fast #shorts"],
            "punchline": package_hook,
            "generated_hook": package_hook,
            "description": "How breathing cycles rapidly adjust heart rate.\n\n#Shorts #Sleep",
            "hashtags": ["#Shorts", "#Sleep"],
            "tags": ["sleep", "breathing"],
            "package_quality_score": 88.0,
            "hook_strategy": "curiosity",
            "is_fallback": False,
        }

        mock_transcribe.return_value = {
            "words": [{"word": "sleep", "start": 0.0, "end": 1.0}],
            "language": "en",
            "text": self.sample_transcript,
        }
        mock_trim.return_value = ("/fake/tight.mp4", [{"word": "sleep", "start": 0.0, "end": 1.0}])
        mock_upload.return_value = "https://youtube.com/shorts/test_b_url"

        mock_drive = MagicMock()
        # Even if durable queue had an older punchline, the active packaging hook must take precedence when valid
        clip_with_old_punchline = dict(self.clip)
        clip_with_old_punchline["punchline"] = "OLD DISCONNECTED HOOK"

        success, _ = pipeline_main.process_queue_clip(
            mock_drive, row_number=2, clip=clip_with_old_punchline, local_src_path="/fake/src.mp4"
        )

        self.assertTrue(success)
        mock_gen_ass.assert_called_once()
        burned_hook = mock_gen_ass.call_args.kwargs.get("punchline")
        self.assertEqual(burned_hook, package_hook)

    def test_c_truncated_or_invalid_llm_response_fails_defensively(self):
        """TEST C: Truncated response (finish_reason='length') or unparseable JSON is treated as fallback."""
        mock_choice = MagicMock()
        mock_choice.finish_reason = "length"
        mock_choice.message.content = '{"content_analysis": {"core_topic": "Sleep'  # truncated mid-JSON
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("metadata_ai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_response

            with patch.object(config, "GROQ_API_KEY", "gsk_test_key"):
                res = metadata_ai.generate_shorts_metadata("huberman_clip.mp4", transcript=self.sample_transcript)

        # Truncated response must trigger fallback, NOT partial corruption or crash
        self.assertTrue(res.get("is_fallback"))
        self.assertEqual(res["title_strategy"], "fallback")
        self.assertIn("huberman clip", res["title"].lower())

    def test_d_filename_fallback_defensive_only_not_used_when_packaging_available(self):
        """TEST D: Filename fallback is used ONLY when AI fails, never when packaging metadata succeeds."""
        # 1. Successful packaging metadata
        fertility_transcript = (
            "Smoking cigarettes introduces toxic chemicals that significantly reduce sperm count "
            "and motility, directly damaging male reproductive health and fertility over time. "
            "Quitting allows cellular recovery within ninety days."
        )
        valid_json = {
            "hook_candidates": [
                {
                    "hook": "WHY DOES SMOKING DAMAGE FERTILITY? 🚬",
                    "hook_type": "specific_fact",
                    "supporting_text": "smoking cigarettes introduces toxic chemicals",
                    "supported_by_clip": True,
                }
            ],
            "title_candidates": [
                {
                    "title": "How Smoking Reduces Sperm Count And Fertility",
                    "strategy": "mechanism",
                    "grounding_score": 9.5,
                    "specificity_score": 9.0,
                    "curiosity_score": 8.5,
                }
            ],
            "description_candidates": [
                {
                    "description": "Cigarette toxins significantly damage male reproductive health until quitting allows recovery.",
                    "strategy": "micro_teaser",
                }
            ],
            "hashtags": ["#Shorts", "#Health", "#Fertility"],
            "tags": ["health", "smoking", "fertility"],
        }

        mock_choice = MagicMock()
        mock_choice.finish_reason = "stop"
        mock_choice.message.content = import_json_dumps(valid_json)
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("metadata_ai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_response

            with patch.object(config, "GROQ_API_KEY", "gsk_test_key"):
                res = metadata_ai.generate_shorts_metadata("fertility_health_expert.mp4", transcript=fertility_transcript)

        # Successful packaging must NOT use filename fallback
        self.assertFalse(res.get("is_fallback"))
        self.assertEqual(res["title"], "How Smoking Reduces Sperm Count And Fertility")
        self.assertNotIn("fertility_health_expert", res["title"])

        # 2. Complete AI absence: Defensive fallback IS used
        with patch.object(config, "GROQ_API_KEY", ""), patch.object(config, "OPENAI_API_KEY", ""):
            with patch.dict(os.environ, {"GROQ_API_KEY": "", "OPENAI_API_KEY": ""}):
                fallback_res = metadata_ai.generate_shorts_metadata(self.video_name, transcript=self.sample_transcript)

        self.assertTrue(fallback_res.get("is_fallback"))
        self.assertIn("#shorts", fallback_res["title"].lower())

    def test_e_source_title_and_filename_leakage_rejected_by_title_validation(self):
        """TEST E: Title validation deterministically rejects raw filename and source-title copies."""
        cand_leak = {
            "title": "Andrew Huberman Become Mentally Dangerous With These Daily Habits FO556 Raj #shorts"
        }
        is_valid, reason = metadata_ai.validate_title(
            cand_leak,
            transcript=self.sample_transcript,
            filename=self.video_name,
        )
        self.assertFalse(is_valid)
        self.assertTrue("leak" in reason.lower() or "limit" in reason.lower() or "copies" in reason.lower())

    def test_f_claim_strength_validation_active(self):
        """TEST F: Claims exceeding transcript factual basis are flagged and rejected."""
        unsupported_claim = "THIS GUARANTEES 100% CURE FOR CHRONIC INSOMNIA FOREVER"
        is_valid, _ = hook_generator.validate_claim_strength(unsupported_claim, transcript=self.sample_transcript)
        self.assertFalse(is_valid)


def import_json_dumps(obj):
    import json
    return json.dumps(obj)


if __name__ == "__main__":
    unittest.main()
