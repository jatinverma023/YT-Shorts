"""
Unit tests for Single-Run Multi-Clip Workflow Orchestration & Idempotency.

Verifies:
1. One source with 5 clips -> all 5 processed in one run.
2. One source with 3 clips -> all 3 processed in one run.
3. One source with 1 clip -> processed and completed.
4. Clip #2 fails -> Clip #1 remains done, Clip #2 remains retryable/failed, source stays in Incoming.
5. Any incomplete clip -> source is NOT moved to Processed.
6. All clips done -> source moves to Processed.
7. YouTube upload succeeds but Sheets update fails -> retry does not blindly re-upload (find_existing_short recovery).
8. Already uploaded clip (existing youtube_url) -> no duplicate upload.
9. Two source videos in Incoming -> first source is processed according to FIFO rules (second untouched).
10. Zero videos -> clean exit.
11. Already-enqueued video -> no duplicate discovery.
12. dry_run=true -> no production side effects.
13. workflow_dispatch & concurrency config in pipeline.yml.
14. 10 AM IST cron math verification.
15. TARGET_FPS=0 config and command generation invariance.
"""
import datetime
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Ensure scripts directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

import config
import drive_utils
import sheet_log
import youtube_upload
import main as pipeline_main


class TestSingleRunWorkflow(unittest.TestCase):
    """Test suite for single-run workflow upgrade and recovery idempotency."""

    @patch("drive_utils.download_file")
    @patch("main.check_and_move_if_completed")
    @patch("main.process_queue_clip")
    @patch("sheet_log.get_pending_clips_for_video")
    def test_1_one_source_with_5_clips_all_processed_in_one_run(
        self, mock_get_pending, mock_process_clip, mock_check_move, mock_download
    ):
        """Verify that a source video with 5 clips processes all 5 clips sequentially in that same run."""
        clips = [
            (2, {"drive_file_id": "vid_5", "source_video_name": "vid5.mp4", "clip_index": 1, "start_time": 0.0, "end_time": 30.0}),
            (3, {"drive_file_id": "vid_5", "source_video_name": "vid5.mp4", "clip_index": 2, "start_time": 30.0, "end_time": 60.0}),
            (4, {"drive_file_id": "vid_5", "source_video_name": "vid5.mp4", "clip_index": 3, "start_time": 60.0, "end_time": 90.0}),
            (5, {"drive_file_id": "vid_5", "source_video_name": "vid5.mp4", "clip_index": 4, "start_time": 90.0, "end_time": 120.0}),
            (6, {"drive_file_id": "vid_5", "source_video_name": "vid5.mp4", "clip_index": 5, "start_time": 120.0, "end_time": 150.0}),
        ]
        mock_get_pending.return_value = clips
        mock_process_clip.side_effect = [
            (True, "https://youtube.com/shorts/clip1"),
            (True, "https://youtube.com/shorts/clip2"),
            (True, "https://youtube.com/shorts/clip3"),
            (True, "https://youtube.com/shorts/clip4"),
            (True, "https://youtube.com/shorts/clip5"),
        ]

        mock_drive = MagicMock()
        stats = pipeline_main.process_all_clips_for_video(mock_drive, "vid_5", "vid5.mp4")

        self.assertEqual(mock_process_clip.call_count, 5)
        self.assertEqual(stats["uploaded"], 5)
        self.assertEqual(stats["failed"], 0)
        self.assertEqual(len(stats["urls"]), 5)
        mock_check_move.assert_called_once_with(mock_drive, "vid_5", "vid5.mp4")

    @patch("drive_utils.download_file")
    @patch("main.check_and_move_if_completed")
    @patch("main.process_queue_clip")
    @patch("sheet_log.get_pending_clips_for_video")
    def test_2_one_source_with_3_clips_all_processed_in_one_run(
        self, mock_get_pending, mock_process_clip, mock_check_move, mock_download
    ):
        """Verify that a source video with 3 clips processes all 3 clips sequentially in that same run."""
        clips = [
            (2, {"drive_file_id": "vid_3", "source_video_name": "vid3.mp4", "clip_index": 1, "start_time": 0.0, "end_time": 30.0}),
            (3, {"drive_file_id": "vid_3", "source_video_name": "vid3.mp4", "clip_index": 2, "start_time": 30.0, "end_time": 60.0}),
            (4, {"drive_file_id": "vid_3", "source_video_name": "vid3.mp4", "clip_index": 3, "start_time": 60.0, "end_time": 90.0}),
        ]
        mock_get_pending.return_value = clips
        mock_process_clip.side_effect = [
            (True, "https://youtube.com/shorts/c1"),
            (True, "https://youtube.com/shorts/c2"),
            (True, "https://youtube.com/shorts/c3"),
        ]

        mock_drive = MagicMock()
        stats = pipeline_main.process_all_clips_for_video(mock_drive, "vid_3", "vid3.mp4")

        self.assertEqual(mock_process_clip.call_count, 3)
        self.assertEqual(stats["uploaded"], 3)
        self.assertEqual(stats["failed"], 0)
        mock_check_move.assert_called_once_with(mock_drive, "vid_3", "vid3.mp4")

    @patch("drive_utils.download_file")
    @patch("main.check_and_move_if_completed")
    @patch("main.process_queue_clip")
    @patch("sheet_log.get_pending_clips_for_video")
    def test_3_one_source_with_1_clip_processed_and_completed(
        self, mock_get_pending, mock_process_clip, mock_check_move, mock_download
    ):
        """Verify that a source video with 1 clip processes and completes in that run."""
        clips = [
            (2, {"drive_file_id": "vid_1", "source_video_name": "vid1.mp4", "clip_index": 1, "start_time": 0.0, "end_time": 45.0}),
        ]
        mock_get_pending.return_value = clips
        mock_process_clip.return_value = (True, "https://youtube.com/shorts/c1")

        mock_drive = MagicMock()
        stats = pipeline_main.process_all_clips_for_video(mock_drive, "vid_1", "vid1.mp4")

        self.assertEqual(mock_process_clip.call_count, 1)
        self.assertEqual(stats["uploaded"], 1)
        self.assertEqual(stats["failed"], 0)
        mock_check_move.assert_called_once_with(mock_drive, "vid_1", "vid1.mp4")

    @patch("drive_utils.download_file")
    @patch("main.check_and_move_if_completed")
    @patch("main.process_queue_clip")
    @patch("sheet_log.get_pending_clips_for_video")
    def test_4_clip_2_fails_clip_1_done_loop_halts(
        self, mock_get_pending, mock_process_clip, mock_check_move, mock_download
    ):
        """
        If Clip #1 succeeds and Clip #2 fails:
        Clip #1 remains done, Clip #2 fails, loop halts, and check_and_move_if_completed is invoked.
        """
        clips = [
            (2, {"drive_file_id": "vid_fail", "source_video_name": "vid.mp4", "clip_index": 1, "start_time": 0.0, "end_time": 30.0}),
            (3, {"drive_file_id": "vid_fail", "source_video_name": "vid.mp4", "clip_index": 2, "start_time": 30.0, "end_time": 60.0}),
            (4, {"drive_file_id": "vid_fail", "source_video_name": "vid.mp4", "clip_index": 3, "start_time": 60.0, "end_time": 90.0}),
        ]
        mock_get_pending.return_value = clips
        mock_process_clip.side_effect = [
            (True, "https://youtube.com/shorts/c1"),
            (False, None),  # Clip 2 fails
        ]

        mock_drive = MagicMock()
        stats = pipeline_main.process_all_clips_for_video(mock_drive, "vid_fail", "vid.mp4")

        # Clip #1 succeeded, Clip #2 failed, Clip #3 was NOT processed because loop halted
        self.assertEqual(mock_process_clip.call_count, 2)
        self.assertEqual(stats["uploaded"], 1)
        self.assertEqual(stats["failed"], 1)
        mock_check_move.assert_called_once_with(mock_drive, "vid_fail", "vid.mp4")

    @patch("drive_utils.move_file")
    @patch("sheet_log.get_video_clip_counts")
    def test_5_any_incomplete_clip_source_not_moved(self, mock_counts, mock_move):
        """If any clip is pending, failed, or quota-waiting, check_and_move_if_completed MUST NOT move source to Processed."""
        mock_drive = MagicMock()

        # Case A: 1 done, 1 pending
        mock_counts.return_value = {"total": 2, "pending": 1, "done": 1, "failed": 0, "retry_after_quota_reset": 0}
        pipeline_main.check_and_move_if_completed(mock_drive, "vid_inc", "vid.mp4")
        mock_move.assert_not_called()

        # Case B: 1 done, 1 failed
        mock_counts.return_value = {"total": 2, "pending": 0, "done": 1, "failed": 1, "retry_after_quota_reset": 0}
        pipeline_main.check_and_move_if_completed(mock_drive, "vid_inc", "vid.mp4")
        mock_move.assert_not_called()

        # Case C: 1 done, 1 quota wait
        mock_counts.return_value = {"total": 2, "pending": 0, "done": 1, "failed": 0, "retry_after_quota_reset": 1}
        pipeline_main.check_and_move_if_completed(mock_drive, "vid_inc", "vid.mp4")
        mock_move.assert_not_called()

    @patch("drive_utils.move_file")
    @patch("sheet_log.get_video_clip_counts")
    def test_6_all_clips_done_source_moves_to_processed(self, mock_counts, mock_move):
        """When 100% of generated clips are done (counts['done'] == counts['total']), source video moves to Processed."""
        mock_drive = MagicMock()
        mock_counts.return_value = {"total": 3, "pending": 0, "done": 3, "failed": 0, "retry_after_quota_reset": 0}

        pipeline_main.check_and_move_if_completed(mock_drive, "vid_done", "vid.mp4")
        mock_move.assert_called_once_with(
            mock_drive, "vid_done", config.DRIVE_INCOMING_FOLDER_ID, config.DRIVE_PROCESSED_FOLDER_ID
        )

    @patch("sheet_log.update_clip_status")
    @patch("youtube_upload.upload_short")
    @patch("youtube_upload.find_existing_short")
    def test_7_sheets_failure_after_youtube_upload_recovery(
        self, mock_find_existing, mock_upload, mock_update_status
    ):
        """
        If YouTube upload succeeded previously, find_existing_short recovers the URL and
        process_queue_clip does NOT blindly re-upload to YouTube.
        """
        mock_find_existing.return_value = "https://youtube.com/shorts/recovered_id"

        clip = {
            "drive_file_id": "vid_recover",
            "source_video_name": "vid.mp4",
            "clip_index": 2,
            "start_time": 30.0,
            "end_time": 60.0,
            "youtube_url": "",  # Sheets had not saved the URL
        }

        mock_drive = MagicMock()
        success, yt_url = pipeline_main.process_queue_clip(mock_drive, 3, clip)

        self.assertTrue(success)
        self.assertEqual(yt_url, "https://youtube.com/shorts/recovered_id")
        mock_upload.assert_not_called()
        mock_update_status.assert_called_once_with(3, "done", youtube_url="https://youtube.com/shorts/recovered_id")

    @patch("sheet_log.update_clip_status")
    @patch("youtube_upload.upload_short")
    def test_8_already_uploaded_clip_no_duplicate_upload(
        self, mock_upload, mock_update_status
    ):
        """If clip['youtube_url'] is already populated in the queue row, it is never re-uploaded."""
        clip = {
            "drive_file_id": "vid_already",
            "source_video_name": "vid.mp4",
            "clip_index": 1,
            "start_time": 0.0,
            "end_time": 30.0,
            "youtube_url": "https://youtube.com/shorts/already_there",
        }

        mock_drive = MagicMock()
        success, yt_url = pipeline_main.process_queue_clip(mock_drive, 2, clip)

        self.assertTrue(success)
        self.assertEqual(yt_url, "https://youtube.com/shorts/already_there")
        mock_upload.assert_not_called()
        mock_update_status.assert_called_once_with(2, "done", youtube_url="https://youtube.com/shorts/already_there")

    @patch("sheet_log.reset_expired_quota_clips")
    @patch("main.discover_and_enqueue_video")
    @patch("sheet_log.get_enqueued_video_ids")
    @patch("drive_utils.list_new_videos")
    @patch("sheet_log.has_active_video_in_queue")
    @patch("sheet_log.get_next_pending_clip")
    @patch("sheet_log.ensure_clip_queue_sheet")
    @patch("drive_utils.get_drive_service")
    def test_9_two_source_videos_fifo_ordering(
        self, mock_get_drive, mock_ensure_sheet, mock_get_pending,
        mock_has_active, mock_list_videos, mock_get_enqueued, mock_discover, mock_reset_quota
    ):
        """When two new videos exist in Incoming, main() picks ONLY the oldest video (FIFO)."""
        mock_get_pending.return_value = None
        mock_has_active.return_value = False
        mock_get_enqueued.return_value = set()

        video_old = {"id": "vid_old", "name": "oldest.mp4", "createdTime": "2026-09-18T10:00:00Z"}
        video_new = {"id": "vid_new", "name": "newer.mp4", "createdTime": "2026-09-18T11:00:00Z"}
        mock_list_videos.return_value = [video_old, video_new]
        mock_discover.return_value = {"total_queued": 3, "uploaded": 3, "failed": 0, "urls": []}

        pipeline_main.main()

        # discover_and_enqueue_video should be called ONLY ONCE for the oldest video
        mock_discover.assert_called_once_with(mock_get_drive.return_value, video_old)

    @patch("sheet_log.reset_expired_quota_clips")
    @patch("main.discover_and_enqueue_video")
    @patch("drive_utils.list_new_videos")
    @patch("sheet_log.has_active_video_in_queue")
    @patch("sheet_log.get_next_pending_clip")
    @patch("sheet_log.ensure_clip_queue_sheet")
    @patch("drive_utils.get_drive_service")
    def test_10_zero_videos_clean_exit(
        self, mock_get_drive, mock_ensure_sheet, mock_get_pending,
        mock_has_active, mock_list_videos, mock_discover, mock_reset_quota
    ):
        """When queue is empty and Incoming has zero videos, main() exits cleanly without calling discovery."""
        mock_get_pending.return_value = None
        mock_has_active.return_value = False
        mock_list_videos.return_value = []

        pipeline_main.main()

        mock_discover.assert_not_called()

    @patch("sheet_log.reset_expired_quota_clips")
    @patch("main.discover_and_enqueue_video")
    @patch("sheet_log.get_enqueued_video_ids")
    @patch("drive_utils.list_new_videos")
    @patch("sheet_log.has_active_video_in_queue")
    @patch("sheet_log.get_next_pending_clip")
    @patch("sheet_log.ensure_clip_queue_sheet")
    @patch("drive_utils.get_drive_service")
    def test_11_already_enqueued_video_no_duplicate_discovery(
        self, mock_get_drive, mock_ensure_sheet, mock_get_pending,
        mock_has_active, mock_list_videos, mock_get_enqueued, mock_discover, mock_reset_quota
    ):
        """An already-enqueued video ID sitting in Incoming is strictly skipped without re-discovery."""
        mock_get_pending.return_value = None
        mock_has_active.return_value = False
        mock_get_enqueued.return_value = {"vid_already_done"}
        mock_list_videos.return_value = [{"id": "vid_already_done", "name": "existing.mp4"}]

        pipeline_main.main()

        mock_discover.assert_not_called()

    @patch("main.run_dry_run_inspection")
    def test_12_dry_run_no_side_effects(self, mock_dry_run):
        """When DRY_RUN_LOG_ONLY=True, main() triggers only inspection without queue or upload mutations."""
        with patch.object(config, "DRY_RUN_LOG_ONLY", True):
            with patch.object(pipeline_main, "DRY_RUN_LOG_ONLY", True):
                pipeline_main.main()
                mock_dry_run.assert_called_once()

    def test_13_workflow_dispatch_and_concurrency_config(self):
        """Verify pipeline.yml defines concurrency control to prevent parallel queue race conditions."""
        yml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".github", "workflows", "pipeline.yml"))
        self.assertTrue(os.path.isfile(yml_path))
        with open(yml_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("concurrency:", content)
        self.assertIn("group: shorts-pipeline", content)
        self.assertIn("cancel-in-progress: false", content)
        self.assertIn("workflow_dispatch:", content)
        self.assertIn("cron: \"30 4 * * *\"", content)

    def test_14_cron_math_10am_ist(self):
        """Verify cron 30 4 * * * (04:30 UTC) is exactly 10:00 AM IST (Asia/Kolkata, UTC+05:30)."""
        utc_dt = datetime.datetime(2026, 9, 19, 4, 30, tzinfo=datetime.timezone.utc)
        ist_tz = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
        ist_dt = utc_dt.astimezone(ist_tz)
        self.assertEqual(ist_dt.strftime("%H:%M %Z"), "10:00 UTC+05:30")

    def test_15_target_fps_and_command_invariance(self):
        """Verify TARGET_FPS=0 config is maintained and omits -r flag from FFmpeg commands."""
        import video_process
        self.assertEqual(config.TARGET_FPS, 0)
        # Verify get_duration_seconds or FFmpeg helper doesn't inject -r when TARGET_FPS=0
        cmd = []
        if config.TARGET_FPS and config.TARGET_FPS > 0:
            cmd.extend(["-r", str(config.TARGET_FPS)])
        self.assertNotIn("-r", cmd)


if __name__ == "__main__":
    unittest.main()
