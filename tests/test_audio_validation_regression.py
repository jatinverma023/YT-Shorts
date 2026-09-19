"""
Regression tests for Audio Stream Validation & Discovery Failure Propagation.

Verifies:
1. audio-present source continues normally (calls extract_audio and proceeds).
2. audio-missing source fails before extract_audio() (extract_audio is never called).
3. failed discovery produces non-zero process exit (main() exits with code 1).
4. failed source is not moved to Processed (stays in Incoming / never moved to Processed).
5. normal successful run does not produce non-zero exit (clean exit 0).
"""
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Ensure scripts directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

import config
import drive_utils
import main as pipeline_main
import sheet_log
import transcribe
import video_process


class TestAudioValidationRegression(unittest.TestCase):
    """Regression test suite for missing audio validation and failure exit propagation."""

    @patch("main.process_all_clips_for_video")
    @patch("notify.send")
    @patch("sheet_log.enqueue_clips")
    @patch("clip_detection.detect_clips_from_transcript")
    @patch("transcribe.transcribe_audio")
    @patch("transcribe.extract_audio")
    @patch("video_process.check_has_audio", return_value=True)
    @patch("video_process.get_duration_seconds", return_value=60.0)
    @patch("drive_utils.download_file")
    def test_audio_present_source_continues_normally(
        self, mock_download, mock_get_dur, mock_has_audio,
        mock_extract_audio, mock_transcribe, mock_detect, mock_enqueue,
        mock_notify, mock_process_clips,
    ):
        """When an audio stream is present, extract_audio is called and the normal discovery flow proceeds."""
        mock_drive = MagicMock()
        file_info = {"id": "vid_valid_audio", "name": "podcast_with_audio.mp4"}

        mock_transcribe.return_value = {"segments": [{"id": 1, "start": 0.0, "end": 30.0}], "words": [], "text": "hello"}
        mock_detect.return_value = [{"start_time": 0.0, "end_time": 30.0, "hook_summary": "Intro"}]
        mock_process_clips.return_value = {
            "total_queued": 1, "uploaded": 1, "failed": 0,
            "urls": ["https://youtube.com/shorts/test"], "errors": [],
        }

        stats = pipeline_main.discover_and_enqueue_video(mock_drive, file_info)

        # check_has_audio must be called to validate audio presence
        mock_has_audio.assert_called_once()
        # extract_audio MUST be called when audio stream exists
        mock_extract_audio.assert_called_once()
        # Full discovery & processing flow proceeds
        mock_transcribe.assert_called_once()
        mock_detect.assert_called_once()
        mock_enqueue.assert_called_once()
        mock_process_clips.assert_called_once()
        self.assertEqual(stats["uploaded"], 1)
        self.assertEqual(stats["failed"], 0)

    @patch("main.process_all_clips_for_video")
    @patch("clip_detection.detect_clips_from_transcript")
    @patch("transcribe.extract_audio")
    @patch("video_process.check_has_audio", return_value=False)
    @patch("video_process.get_duration_seconds", return_value=60.0)
    @patch("drive_utils.download_file")
    @patch("sheet_log.log_run")
    @patch("notify.send")
    def test_audio_missing_source_fails_before_extract_audio(
        self, mock_notify, mock_log_run, mock_download, mock_get_dur,
        mock_has_audio, mock_extract_audio, mock_detect, mock_process_clips,
    ):
        """When audio stream is missing, fails immediately before extract_audio is ever called."""
        mock_drive = MagicMock()
        file_info = {"id": "vid_no_audio", "name": "videoplayback.mp4"}

        stats = pipeline_main.discover_and_enqueue_video(mock_drive, file_info)

        # check_has_audio was checked
        mock_has_audio.assert_called_once()
        # CRITICAL: extract_audio must NOT be called
        mock_extract_audio.assert_not_called()
        # Must not attempt clip detection or uploads
        mock_detect.assert_not_called()
        mock_process_clips.assert_not_called()

        # Stats must report failure
        self.assertEqual(stats["failed"], 1)
        self.assertEqual(stats["total_queued"], 0)
        self.assertTrue(any("no audio stream" in err.lower() for err in stats["errors"]))

        # Failure notification sent to Telegram
        mock_notify.assert_called_once()
        self.assertIn("Failed analyzing clips", mock_notify.call_args[0][0])
        self.assertIn("no audio stream", mock_notify.call_args[0][0].lower())

        # Failure recorded in Google Sheets
        mock_log_run.assert_called_once()
        self.assertEqual(mock_log_run.call_args[0][1], "FAILED (Discovery)")
        self.assertIn("no audio stream", mock_log_run.call_args[1]["error"].lower())

    @patch("sheet_log.reset_expired_quota_clips", return_value=0)
    @patch("sheet_log.get_next_pending_clip", return_value=None)
    @patch("sheet_log.has_active_video_in_queue", return_value=False)
    @patch("sheet_log.ensure_clip_queue_sheet")
    @patch("sheet_log.get_enqueued_video_ids", return_value=set())
    @patch("drive_utils.get_drive_service")
    @patch("drive_utils.list_new_videos")
    @patch("drive_utils.download_file")
    @patch("video_process.get_duration_seconds", return_value=60.0)
    @patch("video_process.check_has_audio", return_value=False)
    @patch("transcribe.extract_audio")
    @patch("sheet_log.log_run")
    @patch("notify.send")
    def test_failed_discovery_produces_non_zero_process_exit(
        self, mock_notify, mock_log_run, mock_extract_audio, mock_has_audio,
        mock_get_dur, mock_download, mock_list_new, mock_get_drive,
        mock_get_enqueued, mock_ensure_sheet, mock_has_active, mock_get_pending,
        mock_reset_quota,
    ):
        """A fatal discovery failure causes python main.py to exit with non-zero status (code 1)."""
        mock_list_new.return_value = [{"id": "vid_bad", "name": "videoplayback.mp4"}]

        with self.assertRaises(SystemExit) as cm:
            pipeline_main.main()

        self.assertEqual(cm.exception.code, 1)
        mock_extract_audio.assert_not_called()

    @patch("drive_utils.move_file")
    @patch("transcribe.extract_audio")
    @patch("video_process.check_has_audio", return_value=False)
    @patch("video_process.get_duration_seconds", return_value=60.0)
    @patch("drive_utils.download_file")
    @patch("sheet_log.log_run")
    @patch("notify.send")
    def test_failed_source_is_not_moved_to_processed(
        self, mock_notify, mock_log_run, mock_download, mock_get_dur,
        mock_has_audio, mock_extract_audio, mock_move_file,
    ):
        """When discovery fails on a video with missing audio, it is NEVER moved to the Processed folder."""
        mock_drive = MagicMock()
        file_info = {"id": "vid_no_audio", "name": "videoplayback.mp4"}

        with patch.object(config, "DRIVE_PROCESSED_FOLDER_ID", "processed_folder_xyz"):
            pipeline_main.discover_and_enqueue_video(mock_drive, file_info)

            # Ensure move_file was NEVER called with Processed folder ID
            for call in mock_move_file.call_args_list:
                to_folder = call[0][3] if len(call[0]) > 3 else call[1].get("to_folder_id")
                self.assertNotEqual(to_folder, "processed_folder_xyz", "Failed source video must NOT be moved to Processed!")

    @patch("sheet_log.reset_expired_quota_clips", return_value=0)
    @patch("sheet_log.get_next_pending_clip", return_value=None)
    @patch("sheet_log.has_active_video_in_queue", return_value=False)
    @patch("sheet_log.ensure_clip_queue_sheet")
    @patch("drive_utils.get_drive_service")
    @patch("drive_utils.list_new_videos", return_value=[])
    def test_successful_clean_run_does_not_exit_nonzero(
        self, mock_list_new, mock_get_drive, mock_ensure_sheet,
        mock_has_active, mock_get_pending, mock_reset_quota,
    ):
        """Expected clean run (empty queue and empty incoming) exits normally with code 0 without SystemExit."""
        try:
            pipeline_main.main()
        except SystemExit as e:
            self.fail(f"main() raised SystemExit({e.code}) unexpectedly on a normal clean run")


if __name__ == "__main__":
    unittest.main()
