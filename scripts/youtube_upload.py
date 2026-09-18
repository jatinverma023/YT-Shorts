"""
Upload the finished vertical video to YouTube as a Short.

YouTube uploads on behalf of a real channel require OAuth2 user credentials
(a service account cannot upload to a personal channel). One-time setup:

1. In Google Cloud Console, enable "YouTube Data API v3".
2. Create an OAuth 2.0 Client ID (type: Desktop app).
3. Run a one-time local script (see get_refresh_token.py) to log in with
   your YouTube-owning Google account and obtain a REFRESH TOKEN.
4. Store client_id, client_secret, refresh_token as secrets. After that,
   this script needs no browser/device — it silently refreshes the
   access token every run.
"""
import logging

import google.oauth2.credentials
import googleapiclient.discovery
import googleapiclient.http

from googleapiclient.errors import HttpError
from config import (
    YT_CLIENT_ID, YT_CLIENT_SECRET, YT_REFRESH_TOKEN,
    DEFAULT_TAGS, DEFAULT_CATEGORY_ID, UPLOAD_AS_PRIVATE_FIRST,
)

log = logging.getLogger("youtube_upload")


class YouTubeQuotaExceededError(Exception):
    """Raised specifically when YouTube Data API daily upload quota is exhausted."""
    pass


def is_quota_exceeded_error(error: Exception) -> bool:
    """Detects if an HttpError corresponds to YouTube API quota exhaustion."""
    if not isinstance(error, HttpError):
        return False
    status_code = getattr(error.resp, "status", None)
    if status_code in (403, 429):
        content = ""
        try:
            content = str(error.content.decode("utf-8") if isinstance(error.content, bytes) else error.content).lower()
        except Exception:
            content = str(error).lower()
        quota_indicators = [
            "quotaexceeded",
            "dailylimitexceeded",
            "userratelimitexceeded",
            "ratelimitexceeded",
            "exceeded your quota",
            "quota",
        ]
        return any(ind in content for ind in quota_indicators)
    return False


def get_youtube_client():
    creds = google.oauth2.credentials.Credentials(
        token=None,
        refresh_token=YT_REFRESH_TOKEN,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=YT_CLIENT_ID,
        client_secret=YT_CLIENT_SECRET,
        scopes=["https://www.googleapis.com/auth/youtube.upload"],
    )
    return googleapiclient.discovery.build("youtube", "v3", credentials=creds)


def upload_short(video_path: str, title: str, description: str, tags=None):
    youtube = get_youtube_client()

    body = {
        "snippet": {
            "title": title[:100],
            "description": description,
            "tags": tags or DEFAULT_TAGS,
            "categoryId": DEFAULT_CATEGORY_ID,
        },
        "status": {
            "privacyStatus": "private" if UPLOAD_AS_PRIVATE_FIRST else "public",
            "selfDeclaredMadeForKids": False,
        },
    }

    media = googleapiclient.http.MediaFileUpload(
        video_path, chunksize=-1, resumable=True, mimetype="video/mp4"
    )

    request = youtube.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media,
    )

    response = None
    try:
        while response is None:
            status, response = request.next_chunk()
            if status:
                log.info("Upload progress: %d%%", int(status.progress() * 100))
    except HttpError as e:
        if is_quota_exceeded_error(e):
            log.error("YouTube Data API quota exceeded (status %s): %s", e.resp.status, e)
            raise YouTubeQuotaExceededError("YouTube Data API quota exceeded (10,000 units/day limit reached).") from e
        log.error("YouTube API HTTP error: %s", e)
        raise

    video_id = response["id"]
    log.info("Uploaded. Video ID: %s", video_id)
    return f"https://youtube.com/shorts/{video_id}"
