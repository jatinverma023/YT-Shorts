import unittest
from unittest.mock import MagicMock, patch
import os
import sys

# Ensure scripts dir is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

import youtube_upload
import get_refresh_token
from googleapiclient.errors import HttpError
import httpx


class TestPhase8YouTubeOAuth(unittest.TestCase):
    def test_refresh_token_scopes_include_readonly(self):
        """get_refresh_token.py must request both upload and readonly scopes."""
        self.assertIn("https://www.googleapis.com/auth/youtube.upload", get_refresh_token.SCOPES)
        self.assertIn("https://www.googleapis.com/auth/youtube.readonly", get_refresh_token.SCOPES)

    @patch("youtube_upload.googleapiclient.discovery.build")
    @patch("youtube_upload.google.oauth2.credentials.Credentials")
    def test_get_youtube_client_includes_both_scopes(self, mock_creds_cls, mock_build):
        """get_youtube_client must request both upload and readonly scopes."""
        with patch.object(youtube_upload, "YT_REFRESH_TOKEN", "mock_refresh"), \
             patch.object(youtube_upload, "YT_CLIENT_ID", "mock_id"), \
             patch.object(youtube_upload, "YT_CLIENT_SECRET", "mock_secret"):
            youtube_upload.get_youtube_client()
            mock_creds_cls.assert_called_once()
            called_kwargs = mock_creds_cls.call_args[1]
            scopes = called_kwargs.get("scopes", [])
            self.assertIn("https://www.googleapis.com/auth/youtube.upload", scopes)
            self.assertIn("https://www.googleapis.com/auth/youtube.readonly", scopes)

    def test_find_existing_short_handles_403_insufficient_scope_defensively(self):
        """find_existing_short catches HTTP 403 insufficientPermissions without raising or leaking credentials."""
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status = 403
        mock_resp.reason = "Forbidden"
        http_error = HttpError(resp=mock_resp, content=b'{"error": {"message": "Request had insufficient authentication scopes.", "status": "PERMISSION_DENIED"}}')
        
        mock_client.channels().list().execute.side_effect = http_error

        with self.assertLogs("youtube_upload", level="WARNING") as log_capture:
            result = youtube_upload.find_existing_short(clip_identifier="clip_123_1", client=mock_client)

            # Must return None so upload proceeds
            self.assertIsNone(result)
            
            # Log must contain diagnostic guidance
            combined_logs = "\n".join(log_capture.output)
            self.assertIn("OAuth refresh token lacks 'youtube.readonly' scope", combined_logs)
            self.assertIn("Upload will proceed without duplicate check", combined_logs)
            self.assertIn("get_refresh_token.py", combined_logs)
            # Ensure no credentials leaked
            self.assertNotIn("mock_secret", combined_logs)
            self.assertNotIn("mock_refresh", combined_logs)

    def test_find_existing_short_finds_match(self):
        """find_existing_short returns URL when duplicate exists."""
        mock_client = MagicMock()
        mock_client.channels().list().execute.return_value = {
            "items": [{"contentDetails": {"relatedPlaylists": {"uploads": "UU12345"}}}]
        }
        mock_client.playlistItems().list().execute.return_value = {
            "items": [
                {
                    "snippet": {
                        "resourceId": {"videoId": "test_video_abc"},
                        "title": "Existing Short Title",
                        "description": "Some description [id:clip_123_1] more text",
                    }
                }
            ]
        }

        url = youtube_upload.find_existing_short(clip_identifier="clip_123_1", client=mock_client)
        self.assertEqual(url, "https://youtube.com/shorts/test_video_abc")


if __name__ == "__main__":
    unittest.main()
