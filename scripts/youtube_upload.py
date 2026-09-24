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
        scopes=[
            "https://www.googleapis.com/auth/youtube.upload",
            "https://www.googleapis.com/auth/youtube.readonly",
        ],
    )
    return googleapiclient.discovery.build("youtube", "v3", credentials=creds)


def find_existing_short(clip_identifier: str = None, title: str = None, client=None) -> str:
    """
    Checks if a Short matching clip_identifier or exact title was already uploaded
    to the authenticated channel. Uses channel uploads playlist (costs only 1-2 quota units,
    NOT expensive search.list). Returns full shorts URL or None.
    """
    if not clip_identifier and not title:
        return None

    try:
        youtube = client or get_youtube_client()
        # 1. Fetch channel's uploads playlist ID (1 quota unit)
        ch_resp = youtube.channels().list(part="contentDetails", mine=True).execute()
        items = ch_resp.get("items", [])
        if not items:
            return None
        uploads_id = items[0].get("contentDetails", {}).get("relatedPlaylists", {}).get("uploads")
        if not uploads_id:
            return None

        # 2. Fetch the 25 most recent uploads (1 quota unit)
        pl_resp = youtube.playlistItems().list(
            part="snippet",
            playlistId=uploads_id,
            maxResults=25,
        ).execute()

        for item in pl_resp.get("items", []):
            snippet = item.get("snippet", {})
            vid_id = snippet.get("resourceId", {}).get("videoId")
            if not vid_id:
                continue
            desc = snippet.get("description", "")
            item_title = snippet.get("title", "")

            # Check for unique clip identifier embedded in description
            if clip_identifier and f"[id:{clip_identifier}]" in desc:
                log.info("Found existing YouTube Short for identifier '%s': videoId=%s", clip_identifier, vid_id)
                return f"https://youtube.com/shorts/{vid_id}"

            # Secondary fallback: exact title match
            if title and item_title.strip().lower() == title.strip()[:100].lower():
                log.info("Found existing YouTube Short matching title '%s': videoId=%s", item_title, vid_id)
                return f"https://youtube.com/shorts/{vid_id}"

    except Exception as e:
        err_str = str(e)
        if "insufficient" in err_str.lower() or "403" in err_str:
            log.warning(
                "YouTube duplicate check skipped: OAuth refresh token lacks 'youtube.readonly' scope. "
                "Upload will proceed without duplicate check. To enable duplicate detection, re-run "
                "'get_refresh_token.py' to generate a token with 'https://www.googleapis.com/auth/youtube.readonly' scope."
            )
        else:
            log.warning("Could not check for existing YouTube upload: %s", e)

    return None


def upload_short(video_path: str, title: str, description: str, tags=None, clip_identifier: str = None, hashtags=None):
    youtube = get_youtube_client()

    final_desc = description
    if hashtags:
        for ht in hashtags:
            ht_str = str(ht).strip()
            if ht_str and ht_str not in final_desc:
                final_desc = f"{final_desc} {ht_str}".strip()

    if clip_identifier and f"[id:{clip_identifier}]" not in final_desc:
        final_desc = f"{final_desc}\n\n[id:{clip_identifier}]".strip()

    final_tags = list(tags or DEFAULT_TAGS)
    if clip_identifier and clip_identifier not in final_tags:
        final_tags.append(clip_identifier[:500])

    body = {
        "snippet": {
            "title": title[:100],
            "description": final_desc,
            "tags": final_tags,
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

