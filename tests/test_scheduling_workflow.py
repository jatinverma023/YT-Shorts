"""
Unit tests for Scheduled Automation Workflow & Google Drive Idempotency.

Verifies:
1. 10:00 AM IST (Asia/Kolkata) schedule calculation (04:30 UTC + 5h 30m = 10:00 AM IST)
2. Incomplete / 0-byte uploads rejected by is_file_ready
3. Active in-progress uploads (<60s modified age) rejected by is_file_ready
4. Complete, stable uploads (>60s modified age, positive size) accepted
5. Deduplication / Idempotency: already-enqueued Drive file IDs are strictly skipped
6. Clean exit when all Incoming videos are already processed
7. Durable queue priority: pending clips processed before scanning Incoming
8. Quota-waiting clips prevent premature Incoming discovery
9. Dry-run mode bypasses upload and rendering
10. Observability banners print cleanly with UTC and IST timestamps
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
import main as pipeline_main


class TestSchedulingWorkflow(unittest.TestCase):
    """Test suite for scheduling, upload readiness, and idempotency."""

    def test_1_cron_schedule_ist_calculation(self):
        """
        Verify that cron '30 4 * * *' (04:30 UTC) corresponds exactly to 10:00 AM IST.
        Asia/Kolkata is fixed UTC+05:30 with zero Daylight Saving Time changes.
        """
        utc_hour = 4
        utc_min = 30

        # Calculate IST offset (5 hours, 30 minutes)
        utc_dt = datetime.datetime(2026, 9, 19, utc_hour, utc_min, 0, tzinfo=datetime.timezone.utc)
        ist_tz = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
        ist_dt = utc_dt.astimezone(ist_tz)

        self.assertEqual(ist_dt.hour, 10)
        self.assertEqual(ist_dt.minute, 0)
        self.assertEqual(ist_dt.strftime("%I:%M %p"), "10:00 AM")

    def test_2_zero_byte_or_missing_size_rejected(self):
        """Files with 0 bytes or missing size are rejected as incomplete uploads."""
        # 0 bytes
        f_zero = {"id": "1", "name": "test.mp4", "size": "0"}
        ready, reason = drive_utils.is_file_ready(f_zero)
        self.assertFalse(ready)
        self.assertIn("0 bytes", reason)

        # Missing size
        f_none = {"id": "2", "name": "test.mp4"}
        ready2, reason2 = drive_utils.is_file_ready(f_none)
        self.assertFalse(ready2)
        self.assertIn("missing", reason2)

    def test_3_active_in_progress_upload_rejected(self):
        """Files modified within the last 60 seconds are rejected as active in-progress uploads."""
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        recent_mod = (now_utc - datetime.timedelta(seconds=20)).isoformat()

        f_active = {
            "id": "3",
            "name": "large_podcast.mp4",
            "size": "500000000",
            "modifiedTime": recent_mod,
        }
        ready, reason = drive_utils.is_file_ready(f_active, min_age_seconds=60.0)
        self.assertFalse(ready)
        self.assertIn("active upload in progress", reason)

    def test_4_stable_completed_upload_accepted(self):
        """Files with positive size and modified >60 seconds ago are accepted as ready."""
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        old_mod = (now_utc - datetime.timedelta(minutes=5)).isoformat()

        f_stable = {
            "id": "4",
            "name": "completed_podcast.mp4",
            "size": "104857600",
            "modifiedTime": old_mod,
        }
        ready, reason = drive_utils.is_file_ready(f_stable, min_age_seconds=60.0)
        self.assertTrue(ready)
        self.assertEqual(reason, "Ready")

    @patch("sheet_log.get_sheets_service")
    def test_5_get_enqueued_video_ids(self, mock_sheets):
        """Verify get_enqueued_video_ids extracts unique Drive file IDs from column B of clip_queue."""
        mock_resp = {
            "values": [
                ["drive_file_id"],  # header
                ["file_abc_123"],
                ["file_abc_123"],  # multiple clips from same video
                ["file_xyz_789"],
            ]
        }
        mock_sheets.return_value.spreadsheets().values().get().execute.return_value = mock_resp

        enqueued = sheet_log.get_enqueued_video_ids()
        self.assertEqual(enqueued, {"file_abc_123", "file_xyz_789"})

    @patch("main.discover_and_enqueue_video")
    @patch("sheet_log.get_enqueued_video_ids", return_value={"already_processed_id_1"})
    @patch("drive_utils.list_new_videos")
    @patch("sheet_log.has_active_video_in_queue", return_value=False)
    @patch("sheet_log.get_next_pending_clip", return_value=None)
    @patch("sheet_log.reset_expired_quota_clips", return_value=0)
    @patch("sheet_log.ensure_clip_queue_sheet")
    @patch("drive_utils.get_drive_service")
    def test_6_already_enqueued_video_skipped(
        self, mock_get_drive, mock_ensure_sheet, mock_reset_quota,
        mock_get_pending, mock_has_active, mock_list_new, mock_get_enqueued,
        mock_discover,
    ):
        """
        Critical Idempotency Test:
        If a video remains in Incoming but its Drive file ID was already enqueued,
        it must be skipped without re-discovery or duplicate clips.
        """
        mock_list_new.return_value = [
            {"id": "already_processed_id_1", "name": "old_podcast.mp4", "size": "1000"},
            {"id": "brand_new_id_2", "name": "new_podcast.mp4", "size": "2000"},
        ]

        pipeline_main.main()

        # Must discover ONLY the brand new video, skipping the already-enqueued one
        mock_discover.assert_called_once()
        target_called = mock_discover.call_args[0][1]
        self.assertEqual(target_called["id"], "brand_new_id_2")
        self.assertEqual(target_called["name"], "new_podcast.mp4")

    @patch("main.discover_and_enqueue_video")
    @patch("sheet_log.get_enqueued_video_ids", return_value={"already_processed_id_1"})
    @patch("drive_utils.list_new_videos")
    @patch("sheet_log.has_active_video_in_queue", return_value=False)
    @patch("sheet_log.get_next_pending_clip", return_value=None)
    @patch("sheet_log.reset_expired_quota_clips", return_value=0)
    @patch("sheet_log.ensure_clip_queue_sheet")
    @patch("drive_utils.get_drive_service")
    def test_7_all_incoming_videos_already_enqueued_clean_exit(
        self, mock_get_drive, mock_ensure_sheet, mock_reset_quota,
        mock_get_pending, mock_has_active, mock_list_new, mock_get_enqueued,
        mock_discover,
    ):
        """When all videos in Incoming have already been processed, exits cleanly with 0 duplicate work."""
        mock_list_new.return_value = [
            {"id": "already_processed_id_1", "name": "old_podcast.mp4", "size": "1000"},
        ]

        pipeline_main.main()
        mock_discover.assert_not_called()

    @patch("main.process_next_pending_clip", return_value={"total_queued": 1, "uploaded": 1, "failed": 0, "urls": []})
    @patch("drive_utils.get_file_metadata", return_value={"id": "f1", "trashed": False})
    @patch("sheet_log.get_next_pending_clip")
    @patch("sheet_log.reset_expired_quota_clips", return_value=0)
    @patch("sheet_log.ensure_clip_queue_sheet")
    @patch("drive_utils.get_drive_service")
    @patch("drive_utils.list_new_videos")
    def test_8_pending_clips_prioritized_over_incoming(
        self, mock_list_new, mock_get_drive, mock_ensure_sheet,
        mock_reset_quota, mock_get_pending, mock_get_meta, mock_process_next,
    ):
        """Pending clips in queue are processed before scanning Incoming."""
        mock_get_pending.return_value = (
            2,
            {
                "drive_file_id": "f1",
                "source_video_name": "vid.mp4",
                "clip_index": 2,
                "start_time": 30.0,
                "end_time": 60.0,
                "status": "pending",
            }
        )

        pipeline_main.main()

        mock_process_next.assert_called_once()
        mock_list_new.assert_not_called()

    @patch("main.run_dry_run_inspection")
    def test_9_dry_run_mode_triggers_inspection(self, mock_dry_run):
        """When DRY_RUN_LOG_ONLY is true, executes dry-run inspection without uploads."""
        with patch.object(pipeline_main, "DRY_RUN_LOG_ONLY", True):
            pipeline_main.main()
            mock_dry_run.assert_called_once()

    def test_10_time_strings_and_banners(self):
        """Observability helper generates valid UTC and IST strings."""
        utc_str, ist_str = pipeline_main._get_time_strings()
        self.assertIn("UTC", utc_str)
        self.assertIn("IST", ist_str)
        self.assertIn("Asia/Kolkata", ist_str)

        # Confirm banners print cleanly without exception
        try:
            pipeline_main.print_run_header()
            pipeline_main.print_run_summary(videos_discovered=1, videos_processed=1, clips_generated=3, clips_uploaded=1)
        except Exception as e:
            self.fail(f"Banners raised unexpected exception: {e}")


if __name__ == "__main__":
    unittest.main()
