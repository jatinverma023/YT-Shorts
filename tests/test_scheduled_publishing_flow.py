"""
Comprehensive unit and integration tests for the Redesigned Shorts Discovery & Scheduled Publishing Flow.

Verifies:
1. Production main() cannot invoke process_all_clips_for_video().
2. One scheduled execution cannot call YouTube upload more than once (One Scheduled Trigger = One Clip Upload).
3. max_clips=None does not truncate valid clips (no arbitrary 5-clip limit).
4. 12 valid candidates can produce 12 queue entries in Google Sheets queue.
5. clip_index remains stable after quality sorting (discovery identity vs publication priority).
6. Stale "processing" clip returns to retryable/pending state (>60 min), while recent remains processing.
7. Zero-valid-clip video is marked terminal, moved to Processed, and not rediscovered.
8. Active Video A blocks discovery/processing of Video B until Video A reaches terminal state.
9. Candidates discovered across transcript chunks are globally deduplicated without duplicates.
10. Claim state machine defends against race conditions (claim_clip returns False if not pending).
"""
import datetime
import os
import sys
import unittest
from unittest.mock import patch, MagicMock, call

# Ensure scripts directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

import config
import clip_detection
import sheet_log
import main as pipeline_main


class TestScheduledPublishingFlow(unittest.TestCase):
    """Test suite for the redesigned Shorts discovery and scheduled publishing flow."""

    def setUp(self):
        self.mock_service = MagicMock()

    # -------------------------------------------------------------------------
    # 1. Production main() cannot invoke process_all_clips_for_video()
    # -------------------------------------------------------------------------
    @patch("main.process_all_clips_for_video")
    @patch("main.process_next_pending_clip")
    @patch("sheet_log.get_next_pending_clip")
    @patch("sheet_log.ensure_clip_queue_sheet")
    @patch("sheet_log.reset_expired_quota_clips", return_value=0)
    @patch("sheet_log.reset_stale_processing_clips", return_value=0)
    @patch("drive_utils.get_drive_service")
    @patch("drive_utils.get_file_metadata", return_value={"id": "vid_1", "trashed": False})
    def test_production_main_cannot_invoke_process_all_clips_for_video(
        self, mock_meta, mock_drive_svc, mock_reset_stale, mock_reset_quota,
        mock_ensure_sheet, mock_get_pending, mock_process_next, mock_process_all
    ):
        """Verify main() only ever invokes process_next_pending_clip(), never process_all_clips_for_video()."""
        mock_get_pending.return_value = (
            2,
            {
                "drive_file_id": "vid_1",
                "source_video_name": "podcast.mp4",
                "clip_index": 1,
                "start_time": 10.0,
                "end_time": 40.0,
                "status": "pending",
                "quality_score": 85.0,
            }
        )
        mock_process_next.return_value = {
            "total_queued": 1, "uploaded": 1, "failed": 0, "urls": ["https://yt.com/s/1"], "errors": []
        }

        pipeline_main.main()

        mock_process_next.assert_called_once()
        mock_process_all.assert_not_called()

    # -------------------------------------------------------------------------
    # 2. One scheduled execution cannot call YouTube upload more than once
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
    @patch("drive_utils.get_file_metadata", return_value={"id": "vid_multi", "trashed": False})
    def test_one_scheduled_execution_cannot_call_youtube_upload_more_than_once(
        self, mock_meta, mock_drive_svc, mock_reset_stale, mock_reset_quota,
        mock_ensure_sheet, mock_get_counts, mock_get_pending, mock_claim,
        mock_update_status, mock_download, mock_extract_slice, mock_extract_audio,
        mock_transcribe, mock_trim, mock_val_hook, mock_meta_gen, mock_gen_ass,
        mock_get_dur, mock_process_vid, mock_find_exist, mock_upload, mock_log_run
    ):
        """Even if 5 clips are pending, ONE scheduled trigger only claims and uploads exactly ONE clip."""
        # Queue has pending clips
        mock_get_pending.return_value = (
            2,
            {
                "drive_file_id": "vid_multi",
                "source_video_name": "long_show.mp4",
                "clip_index": 1,
                "start_time": 0.0,
                "end_time": 30.0,
                "status": "pending",
                "quality_score": 90.0,
            }
        )
        mock_get_counts.return_value = {"total": 5, "pending": 4, "done": 1, "failed": 0, "processing": 0, "retry_after_quota_reset": 0}
        mock_transcribe.return_value = {"words": [], "language": "en", "text": "Clip script"}
        mock_trim.return_value = ("/fake/tight.mp4", [])
        mock_meta_gen.return_value = {"title": "Title #shorts", "description": "Desc", "tags": ["tag"], "punchline": "Punchline"}
        mock_upload.return_value = "https://youtube.com/shorts/only_one"

        pipeline_main.main()

        # Crucial invariant: upload_short must be called EXACTLY ONCE
        self.assertEqual(mock_upload.call_count, 1)

    # -------------------------------------------------------------------------
    # 3. max_clips=None does not truncate valid clips (no arbitrary 5-clip limit)
    # -------------------------------------------------------------------------
    def test_max_clips_none_does_not_truncate_valid_clips(self):
        """12 valid non-overlapping candidates are all accepted when max_clips=None."""
        topics = [
            ("Morning Sunlight", "Why morning sunlight sets your biological circadian clock"),
            ("Compound Growth", "How exponential compound interest creates generational wealth"),
            ("Deep Sleep", "The biological necessity of non-REM stage four recovery"),
            ("Roman Concrete", "Ancient volcanic ash chemistry producing ultra-durable underwater piers"),
            ("Quantum Physics", "Entangled particles demonstrating instantaneous non-local spin correlation"),
            ("HIIT Benefits", "High intensity sprinting maximizing mitochondrial energy density"),
            ("Printing Press", "Gutenberg movable type igniting European renaissance literacy revolution"),
            ("Cognitive Bias", "Confirmation fallacy causing irrational investment portfolio allocation"),
            ("Ocean Hydrothermal", "Deep abyssal volcanic fissures sustaining chemosynthetic extremophile life"),
            ("Turbofan Engines", "Bypass ratio thermodynamics generating high-altitude subsonic flight thrust"),
            ("Dopamine Neurochemistry", "Mesolimbic reward pathways driving compulsive smartphone notification checking"),
            ("Gothic Architecture", "Flying buttresses distributing vertical stone compression weight efficiently"),
        ]
        raw_candidates = []
        for i, (topic, summary) in enumerate(topics):
            raw_candidates.append({
                "start_time": float(i * 50),
                "end_time": float(i * 50 + 40),
                "duration": 40.0,
                "topic": topic,
                "topic_summary": summary,
                "title_idea": f"{topic} #shorts",
                "punchline": f"Insight on {topic} 🔥",
                "scores": {
                    "hook_strength": 8.0,
                    "standalone_clarity": 8.5,
                    "payoff": 8.0,
                    "curiosity": 8.0,
                    "impact": 8.0,
                    "retention": 8.0,
                    "context_dependency": 1.5,
                    "punchline_score": 8.0,
                },
                "standalone": True,
                "standalone_score": 8.5,
                "missing_setup": False,
                "missing_payoff": False,
                "critical_unresolved_reference": False,
            })

        results = clip_detection._validate_and_filter_clips(
            raw_candidates,
            total_duration=700.0,
            min_clip_seconds=20,
            max_clip_seconds=59,
            max_clips=None,  # Dynamic discovery: no arbitrary limit!
        )

        self.assertEqual(len(results), 12, "All 12 valid candidates must be accepted without truncation")

    # -------------------------------------------------------------------------
    # 4. 12 valid candidates produce 12 queue entries
    # -------------------------------------------------------------------------
    def test_12_valid_candidates_produce_12_queue_entries(self):
        """sheet_log.enqueue_clips enqueues all 12 clips into clip_queue."""
        clips = []
        for i in range(1, 13):
            clips.append({
                "clip_index": i,
                "start_time": float((i - 1) * 45),
                "end_time": float((i - 1) * 45 + 35),
                "hook_summary": f"Summary for clip {i}",
                "punchline": f"Hook {i}",
                "quality_score": 80.0 + (i % 5),
            })

        mock_svc = MagicMock()
        mock_append = mock_svc.spreadsheets().values().append
        mock_append.return_value.execute.return_value = {"updates": {"updatedRows": 12}}

        sheet_log.enqueue_clips("vid_12", "twelve_clips.mp4", clips, service=mock_svc)

        mock_append.assert_called_once()
        args, kwargs = mock_append.call_args
        values_inserted = kwargs.get("body", {}).get("values", [])
        self.assertEqual(len(values_inserted), 12)
        # Check first and last row
        self.assertEqual(values_inserted[0][2], 1)
        self.assertEqual(values_inserted[11][2], 12)

    # -------------------------------------------------------------------------
    # 5. clip_index remains stable after quality sorting
    # -------------------------------------------------------------------------
    def test_clip_index_remains_stable_after_quality_sorting(self):
        """
        Clip indices 1, 2, 3 reflect chronological discovery identity.
        Publishing order chooses quality_score DESC (e.g. Clip #2 publishes before Clip #1).
        Clip indices must NOT be mutated or renumbered to reflect publication order.
        """
        raw_candidates = [
            {
                "start_time": 10.0,
                "end_time": 40.0,
                "topic": "Early Context",
                "hook_summary": "First moment",
                "scores": {"hook_strength": 7.0, "standalone_clarity": 7.5, "payoff": 7.0, "curiosity": 7.0, "impact": 7.0, "retention": 7.0, "context_dependency": 2.0, "punchline_score": 7.0},
                "standalone": True,
                "standalone_score": 8.0,
            },
            {
                "start_time": 60.0,
                "end_time": 95.0,
                "topic": "Viral Twist",
                "hook_summary": "Second moment highest quality",
                "scores": {"hook_strength": 9.5, "standalone_clarity": 9.5, "payoff": 9.5, "curiosity": 9.0, "impact": 9.0, "retention": 9.0, "context_dependency": 1.0, "punchline_score": 9.0},
                "standalone": True,
                "standalone_score": 9.5,
            },
            {
                "start_time": 120.0,
                "end_time": 155.0,
                "topic": "Conclusion Point",
                "hook_summary": "Third moment medium quality",
                "scores": {"hook_strength": 8.0, "standalone_clarity": 8.0, "payoff": 8.0, "curiosity": 8.0, "impact": 8.0, "retention": 8.0, "context_dependency": 2.0, "punchline_score": 8.0},
                "standalone": True,
                "standalone_score": 8.0,
            },
        ]

        results = clip_detection._validate_and_filter_clips(
            raw_candidates,
            total_duration=200.0,
            min_clip_seconds=20,
            max_clip_seconds=59,
            max_clips=None,
        )

        # Chronological identities:
        # Clip at 10.0s must be clip_index 1
        # Clip at 60.0s must be clip_index 2
        # Clip at 120.0s must be clip_index 3
        by_start = {c["start_time"]: c for c in results}
        self.assertEqual(by_start[10.0]["clip_index"], 1)
        self.assertEqual(by_start[60.0]["clip_index"], 2)
        self.assertEqual(by_start[120.0]["clip_index"], 3)

        # Highest quality candidate is Clip #2 (quality ~93)
        self.assertEqual(results[0]["clip_index"], 2)
        self.assertGreater(results[0]["quality_score"], results[1]["quality_score"])

        # Now test sheet selection order:
        mock_svc = MagicMock()
        mock_svc.spreadsheets().values().get().execute.return_value = {
            "values": [
                ["source_video_name", "drive_file_id", "clip_index", "start_time", "end_time", "hook_summary", "status", "youtube_url", "error", "created_at", "updated_at", "punchline", "quality_score"],
                ["vid.mp4", "vid_active", "1", "10.0", "40.0", "First", "pending", "", "", "2026-09-19T10:00:00Z", "", "H1", "71.0"],
                ["vid.mp4", "vid_active", "2", "60.0", "95.0", "Second", "pending", "", "", "2026-09-19T10:00:00Z", "", "H2", "93.0"],
                ["vid.mp4", "vid_active", "3", "120.0", "155.0", "Third", "pending", "", "", "2026-09-19T10:00:00Z", "", "H3", "80.0"],
            ]
        }
        best = sheet_log.get_next_pending_clip(service=mock_svc, target_drive_file_id="vid_active")
        self.assertIsNotNone(best)
        row_num, clip_data = best
        # Must pick Clip #2 first (highest quality 93.0), but its clip_index remains 2!
        self.assertEqual(clip_data["clip_index"], 2)
        self.assertEqual(clip_data["quality_score"], 93.0)
        self.assertEqual(row_num, 3)

    # -------------------------------------------------------------------------
    # 6. Stale "processing" clip returns to retryable/pending state
    # -------------------------------------------------------------------------
    def test_stale_processing_clip_returns_to_pending_state(self):
        """Clips stuck in 'processing' longer than max_age_minutes (60 min) reset to 'pending'."""
        now = datetime.datetime.utcnow()
        stale_time = (now - datetime.timedelta(minutes=75)).isoformat() + "Z"
        recent_time = (now - datetime.timedelta(minutes=15)).isoformat() + "Z"

        mock_svc = MagicMock()
        mock_svc.spreadsheets().values().get().execute.return_value = {
            "values": [
                ["header"],
                # Row 2: Stale worker (75 minutes old) -> MUST reset to pending
                ["vid.mp4", "f1", "1", "0", "30", "Summary", "processing", "", "", stale_time, stale_time],
                # Row 3: Active worker (15 minutes old) -> MUST remain processing
                ["vid.mp4", "f1", "2", "30", "60", "Summary", "processing", "", "", recent_time, recent_time],
            ]
        }

        with patch("sheet_log.update_clip_status") as mock_update:
            reset_count = sheet_log.reset_stale_processing_clips(max_age_minutes=60.0, service=mock_svc)
            self.assertEqual(reset_count, 1)
            # Row 2 was reset to pending
            mock_update.assert_called_once_with(2, "pending", service=mock_svc)

    # -------------------------------------------------------------------------
    # 7. Zero-valid-clip video is marked terminal and not rediscovered
    # -------------------------------------------------------------------------
    @patch("notify.send")
    @patch("sheet_log.enqueue_zero_clips")
    @patch("clip_detection.detect_clips_from_transcript", return_value=[])
    @patch("transcribe.transcribe_audio", return_value={"segments": [{"start": 0.0, "end": 60.0}], "words": [], "text": "bad content"})
    @patch("transcribe.extract_audio")
    @patch("video_process.check_has_audio", return_value=True)
    @patch("video_process.get_duration_seconds", return_value=120.0)
    @patch("drive_utils.download_file")
    @patch("main.check_and_move_if_completed")
    def test_zero_valid_clip_video_marked_terminal_and_not_rediscovered(
        self, mock_check_move, mock_download, mock_dur, mock_audio,
        mock_extract, mock_transcribe, mock_detect, mock_enqueue_zero, mock_notify
    ):
        """When 0 clips pass quality/standalone gates, video is marked terminal and not stuck in Incoming."""
        mock_drive = MagicMock()
        file_info = {"id": "vid_zero", "name": "boring_video.mp4"}

        stats = pipeline_main.discover_and_enqueue_video(mock_drive, file_info)

        self.assertEqual(stats["uploaded"], 0)
        self.assertEqual(stats["total_queued"], 0)
        mock_enqueue_zero.assert_called_once_with("vid_zero", "boring_video.mp4", reason="no_valid_clips", candidate_count=1)
        mock_check_move.assert_called_once_with(mock_drive, "vid_zero", "boring_video.mp4")

        # Telegram message must explain clearly
        notify_calls = [c[0][0] for c in mock_notify.call_args_list]
        self.assertTrue(any("No publishable standalone clips passed quality requirements" in msg for msg in notify_calls))

    # -------------------------------------------------------------------------
    # 8. Active Video A blocks discovery/processing of Video B
    # -------------------------------------------------------------------------
    def test_active_video_a_blocks_discovery_and_processing_of_video_b(self):
        """
        If Video A has pending or quota-waiting clips, queue lock ensures:
        - Video B in Incoming is NOT discovered.
        - Video A's pending clip is selected.
        """
        mock_svc = MagicMock()
        mock_svc.spreadsheets().values().get().execute.return_value = {
            "values": [
                ["source_video_name", "drive_file_id", "clip_index", "start_time", "end_time", "hook_summary", "status"],
                # Video A has pending clips
                ["video_A.mp4", "vid_A", "1", "0.0", "30.0", "A1", "done"],
                ["video_A.mp4", "vid_A", "2", "30.0", "60.0", "A2", "pending"],
                # Video B has pending clips that were previously enqueued
                ["video_B.mp4", "vid_B", "1", "0.0", "30.0", "B1", "pending"],
            ]
        }

        # get_active_video_in_queue must return Video A as active
        active = sheet_log.get_active_video_in_queue(service=mock_svc)
        self.assertEqual(active, ("vid_A", "video_A.mp4"))

        # get_next_pending_clip must select Video A's clip, never Video B's
        best = sheet_log.get_next_pending_clip(service=mock_svc)
        self.assertIsNotNone(best)
        row_num, clip = best
        self.assertEqual(clip["drive_file_id"], "vid_A")
        self.assertEqual(clip["clip_index"], 2)

    # -------------------------------------------------------------------------
    # 9. Candidates discovered across transcript chunks globally deduplicated
    # -------------------------------------------------------------------------
    def test_candidates_discovered_across_transcript_chunks_globally_deduplicated(self):
        """Overlapping candidates from different transcript chunks are globally deduplicated."""
        # Candidate 1 from Chunk 1: [10s - 45s], Quality 88
        # Candidate 2 from Chunk 2 (overlap window): [15s - 48s], Quality 72 (redundant overlap)
        # Candidate 3 from Chunk 2: [60s - 95s], Quality 84 (independent)
        candidates = [
            {
                "start_time": 10.0,
                "end_time": 45.0,
                "duration": 35.0,
                "topic": "Chunk 1 Candidate",
                "hook_summary": "Deep insight on focus",
                "scores": {"hook_strength": 9.0, "standalone_clarity": 9.0, "payoff": 9.0, "curiosity": 8.5, "impact": 8.5, "retention": 8.5, "context_dependency": 1.0, "punchline_score": 9.0},
                "standalone": True,
                "standalone_score": 9.0,
            },
            {
                "start_time": 15.0,
                "end_time": 48.0,
                "duration": 33.0,
                "topic": "Chunk 2 Overlap Candidate",
                "hook_summary": "Insight on focus deep",
                "scores": {"hook_strength": 7.0, "standalone_clarity": 7.0, "payoff": 7.0, "curiosity": 7.0, "impact": 7.0, "retention": 7.0, "context_dependency": 3.0, "punchline_score": 7.0},
                "standalone": True,
                "standalone_score": 7.5,
            },
            {
                "start_time": 60.0,
                "end_time": 95.0,
                "duration": 35.0,
                "topic": "Chunk 2 Distinct Candidate",
                "hook_summary": "Completely different insight on sleep",
                "scores": {"hook_strength": 8.5, "standalone_clarity": 8.5, "payoff": 8.5, "curiosity": 8.0, "impact": 8.0, "retention": 8.0, "context_dependency": 1.5, "punchline_score": 8.0},
                "standalone": True,
                "standalone_score": 8.5,
            },
        ]

        results = clip_detection._validate_and_filter_clips(
            candidates,
            total_duration=120.0,
            min_clip_seconds=20,
            max_clip_seconds=59,
            max_clips=None,
        )

        # Candidate 2 must be eliminated by temporal + semantic deduplication against Candidate 1
        self.assertEqual(len(results), 2)
        start_times = [r["start_time"] for r in results]
        self.assertIn(10.0, start_times)
        self.assertIn(60.0, start_times)
        self.assertNotIn(15.0, start_times)

    # -------------------------------------------------------------------------
    # 10. Claim state machine defensive transition
    # -------------------------------------------------------------------------
    def test_claim_state_machine_defensive_transition(self):
        """claim_clip transitions status to 'processing' and returns True; rejects non-pending rows."""
        mock_svc = MagicMock()
        # Row 2 is pending
        mock_svc.spreadsheets().values().get().execute.return_value = {"values": [["pending"]]}

        claimed = sheet_log.claim_clip(2, service=mock_svc)
        self.assertTrue(claimed)

        # If row is already 'processing' (claimed by another concurrent worker)
        mock_svc.spreadsheets().values().get().execute.return_value = {"values": [["processing"]]}
        second_claim = sheet_log.claim_clip(2, service=mock_svc)
        self.assertFalse(second_claim, "Concurrent claim on already-processing clip must return False")


if __name__ == "__main__":
    unittest.main()
