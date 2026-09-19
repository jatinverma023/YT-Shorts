"""
Focused regression tests for Google Sheets enqueued-video-ID lookup.
Verifies:
1. Successful retrieval of enqueued Drive IDs (filtering header, stripping whitespace, deduplicating)
2. Empty sheet handling (returns empty set without error)
3. Sheets API failure handling (explicitly raises error instead of silently returning empty set)
4. Correct service/client method usage:
   - Proper Sheets service usage (calling spreadsheets().values().get())
   - Safety fallback when a Drive Resource or non-Sheets service is passed by mistake
5. Integration in main.py: safe abort with non-zero exit when Sheets lookup fails, avoiding duplicate processing.
"""
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Ensure scripts directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

import sheet_log
import main as pipeline_main


class TestSheetLogEnqueuedIds(unittest.TestCase):
    """Regression test suite for get_enqueued_video_ids and Sheets client initialization."""

    def test_successful_retrieval_of_enqueued_drive_ids(self):
        """Verify unique Drive file IDs are correctly parsed, stripped, and deduplicated from column B."""
        mock_service = MagicMock()
        mock_get = MagicMock()
        mock_service.spreadsheets().values().get = mock_get
        mock_resp = {
            "values": [
                ["drive_file_id"],  # header row
                ["drive_id_101"],
                ["  drive_id_102  "],  # whitespace padded
                ["drive_id_101"],  # duplicate clip from same video
                [""],  # empty row
                ["drive_id_103"],
            ]
        }
        mock_get.return_value.execute.return_value = mock_resp

        result = sheet_log.get_enqueued_video_ids(service=mock_service)
        self.assertEqual(result, {"drive_id_101", "drive_id_102", "drive_id_103"})

        # Verify exact API call parameters
        mock_get.assert_called_once_with(
            spreadsheetId=sheet_log.LOG_SHEET_ID,
            range=f"{sheet_log.CLIP_QUEUE_TAB}!B:B",
        )

    def test_empty_sheet_returns_empty_set(self):
        """Verify an empty sheet tab or a sheet with only the header row returns an empty set."""
        mock_service = MagicMock()

        # Case A: Only header row
        mock_service.spreadsheets().values().get().execute.return_value = {"values": [["drive_file_id"]]}
        result_header_only = sheet_log.get_enqueued_video_ids(service=mock_service)
        self.assertEqual(result_header_only, set())

        # Case B: Completely empty sheet (values key is empty or missing)
        mock_service.spreadsheets().values().get().execute.return_value = {"values": []}
        result_empty_values = sheet_log.get_enqueued_video_ids(service=mock_service)
        self.assertEqual(result_empty_values, set())

        mock_service.spreadsheets().values().get().execute.return_value = {}
        result_missing_key = sheet_log.get_enqueued_video_ids(service=mock_service)
        self.assertEqual(result_missing_key, set())

    def test_sheets_api_failure_raises_explicit_error(self):
        """Verify Sheets API failure raises RuntimeError rather than silently returning an empty set."""
        mock_service = MagicMock()
        mock_service.spreadsheets().values().get().execute.side_effect = RuntimeError("503 Service Unavailable")

        with self.assertRaises(RuntimeError) as ctx:
            sheet_log.get_enqueued_video_ids(service=mock_service)

        self.assertIn("Could not fetch enqueued video IDs from clip_queue", str(ctx.exception))

    @patch("sheet_log.get_sheets_service")
    def test_drive_service_fallback_safety(self, mock_get_sheets):
        """
        Verify that mistakenly passing a Google Drive Resource object (which lacks .spreadsheets)
        does not crash with AttributeError, but safely falls back to get_sheets_service().
        """
        # Simulate a Google Drive Resource object that has .files() but NO .spreadsheets
        mock_drive_resource = MagicMock(spec=["files", "permissions", "changes"])
        self.assertFalse(hasattr(mock_drive_resource, "spreadsheets"))

        # Configure the fallback Sheets service returned by get_sheets_service()
        fallback_sheets = MagicMock()
        mock_get = MagicMock()
        fallback_sheets.spreadsheets().values().get = mock_get
        mock_get.return_value.execute.return_value = {
            "values": [["drive_file_id"], ["recovered_id_456"]]
        }
        mock_get_sheets.return_value = fallback_sheets

        # Calling with mock_drive_resource should trigger safe fallback
        result = sheet_log.get_enqueued_video_ids(service=mock_drive_resource)

        mock_get_sheets.assert_called_once()
        mock_get.assert_called_once_with(
            spreadsheetId=sheet_log.LOG_SHEET_ID,
            range=f"{sheet_log.CLIP_QUEUE_TAB}!B:B",
        )
        self.assertEqual(result, {"recovered_id_456"})

    @patch("sheet_log.get_sheets_service")
    def test_default_none_service_initialization(self, mock_get_sheets):
        """Verify get_enqueued_video_ids() called with service=None initializes Sheets service."""
        mock_sheets = MagicMock()
        mock_sheets.spreadsheets().values().get().execute.return_value = {
            "values": [["drive_file_id"], ["id_default"]]
        }
        mock_get_sheets.return_value = mock_sheets

        result = sheet_log.get_enqueued_video_ids()
        mock_get_sheets.assert_called_once()
        self.assertEqual(result, {"id_default"})

    @patch("notify.send")
    @patch("sheet_log.reset_expired_quota_clips")
    @patch("main.discover_and_enqueue_video")
    @patch("sheet_log.get_enqueued_video_ids")
    @patch("drive_utils.list_new_videos")
    @patch("sheet_log.has_active_video_in_queue", return_value=False)
    @patch("sheet_log.get_next_pending_clip", return_value=None)
    @patch("sheet_log.ensure_clip_queue_sheet")
    @patch("drive_utils.get_drive_service")
    def test_main_aborts_safely_when_sheets_lookup_fails(
        self, mock_get_drive, mock_ensure_sheet, mock_get_pending,
        mock_has_active, mock_list_videos, mock_get_enqueued, mock_discover,
        mock_reset_quota, mock_notify
    ):
        """
        Verify that if get_enqueued_video_ids() fails during main(), discovery is aborted,
        no videos are processed as duplicates, and the process exits with status 1.
        """
        mock_list_videos.return_value = [{"id": "vid_incoming_1", "name": "source.mp4"}]
        mock_get_enqueued.side_effect = RuntimeError("Sheets API network failure")

        with self.assertRaises(SystemExit) as cm:
            pipeline_main.main()

        # Must exit with non-zero code
        self.assertEqual(cm.exception.code, 1)
        # Must NEVER attempt to discover/process video when idempotency check fails
        mock_discover.assert_not_called()


if __name__ == "__main__":
    unittest.main()
