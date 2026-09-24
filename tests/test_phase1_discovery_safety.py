"""
Phase 1 Tests: Critical Production Safety.
Verifies that:
1. Total AI discovery failure returns zero clips, enqueues nothing, and preserves source in Incoming.
2. Successful AI discovery with zero qualified candidates records terminal no_valid_clips and moves source to Processed.
3. Successful AI discovery with valid candidates enqueues clips normally.
4. One-clip-per-scheduled-run behavior is preserved.
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

import clip_detection
import main as pipeline_main


class TestPhase1DiscoverySafety(unittest.TestCase):

    @patch("clip_detection._get_llm_client", return_value=(None, None))
    def test_total_ai_discovery_failure_raises_clip_discovery_error(self, mock_client):
        """When LLM client is unavailable or all chunks fail, ClipDiscoveryError is raised."""
        segments = [{"start": 0.0, "end": 60.0, "text": "Sample text"}]
        with self.assertRaises(clip_detection.ClipDiscoveryError):
            clip_detection.detect_clips_from_transcript(segments, total_duration=120.0)

    @patch("clip_detection._get_llm_client")
    @patch("clip_detection._detect_clips_llm", return_value=None)
    def test_all_chunks_fail_raises_clip_discovery_error(self, mock_detect, mock_client):
        """When all LLM chunk calls return None (e.g. 400/429 errors), ClipDiscoveryError is raised."""
        mock_client.return_value = (MagicMock(), "test-model")
        segments = [{"start": 0.0, "end": 60.0, "text": "Sample text"}]
        with self.assertRaises(clip_detection.ClipDiscoveryError):
            clip_detection.detect_clips_from_transcript(segments, total_duration=120.0)

    @patch("notify.send")
    @patch("sheet_log.log_run")
    @patch("sheet_log.enqueue_clips")
    @patch("sheet_log.enqueue_zero_clips")
    @patch("clip_detection.detect_clips_from_transcript", side_effect=clip_detection.ClipDiscoveryError("All chunks failed"))
    @patch("transcribe.transcribe_audio", return_value={"segments": [{"start": 0.0, "end": 60.0}], "words": [], "text": "text"})
    @patch("transcribe.extract_audio")
    @patch("video_process.check_has_audio", return_value=True)
    @patch("video_process.get_duration_seconds", return_value=120.0)
    @patch("drive_utils.download_file")
    @patch("main.check_and_move_if_completed")
    def test_test1_ai_discovery_failure_zero_clips_no_queue_preserved_in_incoming(
        self, mock_check_move, mock_download, mock_dur, mock_audio,
        mock_extract, mock_transcribe, mock_detect, mock_enqueue_zero,
        mock_enqueue_clips, mock_log_run, mock_notify
    ):
        """TEST 1: All AI discovery chunks fail -> zero clips returned -> no queue rows created, preserved in Incoming."""
        mock_drive = MagicMock()
        file_info = {"id": "vid_fail", "name": "failed_discovery_vid.mp4"}

        stats = pipeline_main.discover_and_enqueue_video(mock_drive, file_info)

        self.assertEqual(stats["uploaded"], 0)
        self.assertEqual(stats["total_queued"], 0)
        self.assertEqual(stats["failed"], 1)
        self.assertTrue(stats.get("discovery_failed"))

        # Zero queue rows created!
        mock_enqueue_clips.assert_not_called()
        mock_enqueue_zero.assert_not_called()

        # Source preserved in Incoming: check_and_move_if_completed must NOT be called
        mock_check_move.assert_not_called()

        # Failure logged in sheet_log and notify
        mock_log_run.assert_called_once()
        self.assertIn("FAILED (AI Discovery)", mock_log_run.call_args[0][1])

    @patch("notify.send")
    @patch("sheet_log.enqueue_zero_clips")
    @patch("sheet_log.enqueue_clips")
    @patch("clip_detection.detect_clips_from_transcript", return_value=[])
    @patch("transcribe.transcribe_audio", return_value={"segments": [{"start": 0.0, "end": 60.0}], "words": [], "text": "bad content"})
    @patch("transcribe.extract_audio")
    @patch("video_process.check_has_audio", return_value=True)
    @patch("video_process.get_duration_seconds", return_value=120.0)
    @patch("drive_utils.download_file")
    @patch("main.check_and_move_if_completed")
    def test_test2_zero_valid_candidates_terminal_no_valid_clips(
        self, mock_check_move, mock_download, mock_dur, mock_audio,
        mock_extract, mock_transcribe, mock_detect, mock_enqueue_clips,
        mock_enqueue_zero, mock_notify
    ):
        """TEST 2: AI discovery returns candidates but all fail quality/standalone validation -> legitimate no-valid-clips."""
        mock_drive = MagicMock()
        file_info = {"id": "vid_zero", "name": "unqualified_content.mp4"}

        stats = pipeline_main.discover_and_enqueue_video(mock_drive, file_info)

        self.assertEqual(stats["uploaded"], 0)
        self.assertEqual(stats["total_queued"], 0)
        mock_enqueue_clips.assert_not_called()
        mock_enqueue_zero.assert_called_once_with("vid_zero", "unqualified_content.mp4", reason="no_valid_clips", candidate_count=1)
        mock_check_move.assert_called_once_with(mock_drive, "vid_zero", "unqualified_content.mp4")

    @patch("main.process_next_pending_clip", return_value={"uploaded": 1, "failed": 0, "urls": ["https://youtube.com/shorts/123"], "errors": []})
    @patch("notify.send")
    @patch("sheet_log.enqueue_clips")
    @patch("clip_detection.detect_clips_from_transcript")
    @patch("transcribe.transcribe_audio", return_value={"segments": [{"start": 0.0, "end": 60.0}], "words": [], "text": "valid"})
    @patch("transcribe.extract_audio")
    @patch("video_process.check_has_audio", return_value=True)
    @patch("video_process.get_duration_seconds", return_value=120.0)
    @patch("drive_utils.download_file")
    def test_test3_valid_ai_candidates_normal_queue(
        self, mock_download, mock_dur, mock_audio, mock_extract,
        mock_transcribe, mock_detect, mock_enqueue, mock_notify, mock_process_one
    ):
        """TEST 3: Existing valid AI candidates -> normal queue behavior unchanged."""
        valid_clips = [
            {"clip_index": 1, "start_time": 10.0, "end_time": 45.0, "quality_score": 85.0, "hook_summary": "Great hook"}
        ]
        mock_detect.return_value = valid_clips
        mock_drive = MagicMock()
        file_info = {"id": "vid_good", "name": "viral_talk.mp4"}

        stats = pipeline_main.discover_and_enqueue_video(mock_drive, file_info)

        mock_enqueue.assert_called_once_with("vid_good", "viral_talk.mp4", valid_clips)
        mock_process_one.assert_called_once()
        self.assertEqual(stats["uploaded"], 1)

    @patch("main.process_queue_clip", return_value=(True, "https://youtube.com/shorts/single"))
    @patch("sheet_log.claim_clip")
    @patch("sheet_log.get_next_pending_clip")
    @patch("main.check_and_move_if_completed")
    def test_test4_existing_one_clip_per_scheduled_run_unchanged(
        self, mock_check_move, mock_get_pending, mock_claim, mock_process_queue
    ):
        """TEST 4: Existing one-clip-per-scheduled-run behavior unchanged."""
        mock_get_pending.return_value = (
            5,
            {"clip_index": 2, "drive_file_id": "file_123", "source_video_name": "test.mp4", "start_time": 30.0, "end_time": 60.0}
        )
        mock_drive = MagicMock()

        stats = pipeline_main.process_next_pending_clip(mock_drive)

        mock_claim.assert_called_once_with(5)
        mock_process_queue.assert_called_once()
        self.assertEqual(stats["uploaded"], 1)
        self.assertEqual(stats["total_queued"], 1)


if __name__ == "__main__":
    unittest.main()
