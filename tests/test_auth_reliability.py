"""
Unit tests for Google authentication reliability, startup configuration validation,
explicit OAuth invalid_grant handling, and queue row failure protection.
"""
import json
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Ensure scripts dir is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

import config
import youtube_upload
import sheet_log
import main as pipeline_main
from google.auth.exceptions import RefreshError


class TestAuthReliability(unittest.TestCase):
    """Test suite covering startup config validation and explicit OAuth invalid_grant handling."""

    # -------------------------------------------------------------------------
    # 1. Startup configuration validation
    # -------------------------------------------------------------------------
    def test_validate_startup_config_detects_missing_variables(self):
        """validate_startup_config detects missing environment variables without exposing values."""
        with patch.object(config, "GOOGLE_SERVICE_ACCOUNT_JSON", ""), \
             patch.object(config, "LOG_SHEET_ID", ""), \
             patch.object(config, "DRIVE_INCOMING_FOLDER_ID", ""), \
             patch.object(config, "YT_CLIENT_ID", ""), \
             patch.object(config, "YT_CLIENT_SECRET", ""), \
             patch.object(config, "YT_REFRESH_TOKEN", ""):
            is_valid, missing = config.validate_startup_config(check_youtube=True)
            self.assertFalse(is_valid)
            self.assertIn("GOOGLE_SERVICE_ACCOUNT_JSON", missing)
            self.assertIn("LOG_SHEET_ID", missing)
            self.assertIn("DRIVE_INCOMING_FOLDER_ID", missing)
            self.assertIn("YT_CLIENT_ID", missing)
            self.assertIn("YT_CLIENT_SECRET", missing)
            self.assertIn("YT_REFRESH_TOKEN", missing)

    def test_validate_startup_config_detects_malformed_service_account_json(self):
        """validate_startup_config detects malformed JSON in GOOGLE_SERVICE_ACCOUNT_JSON."""
        with patch.object(config, "GOOGLE_SERVICE_ACCOUNT_JSON", "not-a-json-string"), \
             patch.object(config, "LOG_SHEET_ID", "mock_sheet"), \
             patch.object(config, "DRIVE_INCOMING_FOLDER_ID", "mock_folder"):
            is_valid, missing = config.validate_startup_config(check_youtube=False)
            self.assertFalse(is_valid)
            self.assertTrue(any("malformed" in m.lower() for m in missing))

    def test_validate_startup_config_detects_missing_service_account_internal_keys(self):
        """validate_startup_config verifies required service account keys (client_email, private_key)."""
        invalid_sa = json.dumps({"type": "service_account", "project_id": "test"})
        with patch.object(config, "GOOGLE_SERVICE_ACCOUNT_JSON", invalid_sa), \
             patch.object(config, "LOG_SHEET_ID", "mock_sheet"), \
             patch.object(config, "DRIVE_INCOMING_FOLDER_ID", "mock_folder"):
            is_valid, missing = config.validate_startup_config(check_youtube=False)
            self.assertFalse(is_valid)
            self.assertTrue(any("missing" in m.lower() for m in missing))

    def test_validate_startup_config_succeeds_when_valid(self):
        """validate_startup_config succeeds when all required keys and valid SA JSON are present."""
        valid_sa = json.dumps({
            "type": "service_account",
            "client_email": "sa@proj.iam.gserviceaccount.com",
            "private_key": "-----BEGIN PRIVATE KEY-----\nMOCK\n-----END PRIVATE KEY-----",
        })
        with patch.object(config, "GOOGLE_SERVICE_ACCOUNT_JSON", valid_sa), \
             patch.object(config, "LOG_SHEET_ID", "mock_sheet"), \
             patch.object(config, "DRIVE_INCOMING_FOLDER_ID", "mock_folder"), \
             patch.object(config, "YT_CLIENT_ID", "mock_client"), \
             patch.object(config, "YT_CLIENT_SECRET", "mock_secret"), \
             patch.object(config, "YT_REFRESH_TOKEN", "mock_refresh"):
            is_valid, missing = config.validate_startup_config(check_youtube=True)
            self.assertTrue(is_valid)
            self.assertEqual(missing, [])

    # -------------------------------------------------------------------------
    # 2. YouTube OAuth client initialization and credential validation
    # -------------------------------------------------------------------------
    def test_get_youtube_credentials_raises_when_env_missing(self):
        """get_youtube_credentials raises YouTubeAuthenticationError when secrets are missing."""
        with patch.object(youtube_upload, "YT_REFRESH_TOKEN", ""):
            with self.assertRaises(youtube_upload.YouTubeAuthenticationError) as ctx:
                youtube_upload.get_youtube_credentials()
            self.assertIn("YT_REFRESH_TOKEN", str(ctx.exception))

    @patch("youtube_upload.google.auth.transport.requests.Request")
    def test_validate_youtube_credentials_handles_invalid_grant_explicitly(self, mock_req):
        """validate_youtube_credentials translates RefreshError(invalid_grant) into YouTubeAuthenticationError."""
        mock_creds = MagicMock()
        mock_creds.refresh.side_effect = RefreshError(
            "invalid_grant: Token has been expired or revoked.",
            {"error": "invalid_grant", "error_description": "Token has been expired or revoked."}
        )

        with self.assertRaises(youtube_upload.YouTubeAuthenticationError) as ctx:
            youtube_upload.validate_youtube_credentials(creds=mock_creds)

        err_msg = str(ctx.exception)
        self.assertIn("invalid_grant", err_msg)
        self.assertIn("get_refresh_token.py", err_msg)
        self.assertIn("YT_REFRESH_TOKEN", err_msg)

    @patch("youtube_upload.google.auth.transport.requests.Request")
    def test_validate_youtube_credentials_succeeds_on_valid_token(self, mock_req):
        """validate_youtube_credentials returns True when refresh succeeds."""
        mock_creds = MagicMock()
        mock_creds.refresh.return_value = None

        result = youtube_upload.validate_youtube_credentials(creds=mock_creds)
        self.assertTrue(result)
        mock_creds.refresh.assert_called_once()

    # -------------------------------------------------------------------------
    # 3. find_existing_short and upload_short error escalation
    # -------------------------------------------------------------------------
    def test_find_existing_short_raises_auth_error_on_invalid_grant(self):
        """find_existing_short does NOT swallow invalid_grant RefreshError."""
        mock_client = MagicMock()
        mock_client.channels().list().execute.side_effect = RefreshError(
            "invalid_grant: Token has been expired or revoked."
        )

        with self.assertRaises(youtube_upload.YouTubeAuthenticationError) as ctx:
            youtube_upload.find_existing_short(clip_identifier="clip_1", client=mock_client)

        self.assertIn("invalid_grant", str(ctx.exception))

    @patch("youtube_upload.googleapiclient.http.MediaFileUpload")
    @patch("youtube_upload.get_youtube_client")
    def test_upload_short_raises_auth_error_on_invalid_grant(self, mock_get_client, mock_media):
        """upload_short escalates RefreshError(invalid_grant) as YouTubeAuthenticationError."""
        mock_client = MagicMock()
        mock_request = MagicMock()
        mock_request.next_chunk.side_effect = RefreshError(
            "invalid_grant: Token has been expired or revoked."
        )
        mock_client.videos().insert.return_value = mock_request
        mock_get_client.return_value = mock_client

        with self.assertRaises(youtube_upload.YouTubeAuthenticationError) as ctx:
            youtube_upload.upload_short("/mock/path.mp4", "Title", "Description")

        self.assertIn("invalid_grant", str(ctx.exception))

    # -------------------------------------------------------------------------
    # 4. Queue protection: Clips are NOT marked 'failed' on auth error
    # -------------------------------------------------------------------------
    @patch("main.sheet_log.log_run")
    @patch("main.sheet_log.update_clip_status_only")
    @patch("main.sheet_log.update_clip_status")
    @patch("main.youtube_upload.find_existing_short")
    def test_process_queue_clip_preserves_row_on_auth_error(
        self, mock_find, mock_update_status, mock_update_only, mock_log_run
    ):
        """
        When YouTubeAuthenticationError occurs during clip processing,
        process_queue_clip does NOT mark the clip as 'failed'. It preserves it as 'pending'
        via update_clip_status_only without overwriting columns H & I.
        """
        mock_find.side_effect = youtube_upload.YouTubeAuthenticationError(
            "YouTube OAuth refresh token expired or revoked (invalid_grant)."
        )

        clip = {
            "drive_file_id": "test_vid_1",
            "source_video_name": "podcast.mp4",
            "clip_index": 1,
            "start_time": 10.0,
            "end_time": 40.0,
        }

        with self.assertRaises(youtube_upload.YouTubeAuthenticationError):
            pipeline_main.process_queue_clip(MagicMock(), row_number=5, clip=clip)

        # CRITICAL ASSERTIONS:
        # update_clip_status(..., "failed") must NOT be called!
        for c in mock_update_status.call_args_list:
            self.assertNotEqual(c[0][1], "failed")

        # update_clip_status_only must be called with "pending"
        mock_update_only.assert_called_once_with(5, "pending")

    # -------------------------------------------------------------------------
    # 5. Pipeline halts immediately on auth error without failing subsequent clips
    # -------------------------------------------------------------------------
    @patch("main.sheet_log.reset_expired_quota_clips")
    @patch("main.sheet_log.reset_stale_processing_clips")
    @patch("main.drive_utils.get_file_metadata", return_value={"id": "vid_1", "trashed": False})
    @patch("main.sheet_log.log_run")
    @patch("main.youtube_upload.validate_youtube_credentials", return_value=True)
    @patch("main.config.validate_startup_config", return_value=(True, []))
    @patch("main.process_next_pending_clip")
    @patch("main.sheet_log.get_next_pending_clip")
    @patch("main.sheet_log.ensure_clip_queue_sheet")
    @patch("main.drive_utils.get_drive_service")
    def test_main_halts_immediately_on_auth_error(
        self, mock_get_drive, mock_ensure_sheet, mock_get_pending, mock_process_next,
        mock_validate_config, mock_validate_yt, mock_log_run, mock_get_meta,
        mock_reset_stale, mock_reset_quota
    ):
        """When an auth error occurs on a clip, main() halts immediately and does not process subsequent clips."""
        mock_get_pending.return_value = (
            2,
            {
                "drive_file_id": "vid_1",
                "source_video_name": "podcast.mp4",
                "clip_index": 1,
                "start_time": 10.0,
                "end_time": 40.0,
                "status": "pending",
            }
        )
        mock_process_next.side_effect = youtube_upload.YouTubeAuthenticationError(
            "YouTube OAuth refresh token expired or revoked (invalid_grant)."
        )

        with patch.object(pipeline_main, "DRY_RUN_LOG_ONLY", False):
            pipeline_main.main()

        # Called once for the first clip, then halted immediately
        mock_process_next.assert_called_once()

    @patch("main.sheet_log.reset_expired_quota_clips")
    @patch("main.sheet_log.reset_stale_processing_clips")
    @patch("main.notify.send")
    @patch("main.sheet_log.log_run")
    @patch("main.youtube_upload.validate_youtube_credentials")
    @patch("main.config.validate_startup_config", return_value=(True, []))
    @patch("main.process_next_pending_clip")
    @patch("main.sheet_log.get_next_pending_clip")
    @patch("main.sheet_log.ensure_clip_queue_sheet")
    @patch("main.drive_utils.get_drive_service")
    def test_main_halts_on_preflight_auth_error(
        self, mock_get_drive, mock_ensure_sheet, mock_get_pending, mock_process_next,
        mock_validate_config, mock_validate_yt, mock_log_run, mock_notify,
        mock_reset_stale, mock_reset_quota
    ):
        """When preflight auth check fails, main() halts before rendering/downloading and does NOT process clips."""
        mock_get_pending.return_value = (
            2,
            {
                "drive_file_id": "vid_1",
                "source_video_name": "podcast.mp4",
                "clip_index": 1,
                "start_time": 10.0,
                "end_time": 40.0,
                "status": "pending",
            }
        )
        mock_validate_yt.side_effect = youtube_upload.YouTubeAuthenticationError(
            "YouTube OAuth refresh token expired or revoked (invalid_grant)."
        )

        with patch.object(pipeline_main, "DRY_RUN_LOG_ONLY", False):
            pipeline_main.main()

        # Must never attempt to process the clip if preflight validation fails
        mock_process_next.assert_not_called()
        self.assertTrue(any("Pre-flight Check Failed" in str(c) for c in mock_notify.call_args_list))


if __name__ == "__main__":
    unittest.main()
