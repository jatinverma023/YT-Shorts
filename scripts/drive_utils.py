"""
Google Drive helpers using a SERVICE ACCOUNT.

Setup required (one-time, done by you in Google Cloud Console):
1. Create a GCP project -> enable "Google Drive API".
2. Create a Service Account -> generate a JSON key.
3. Share your "Incoming" and "Processed" Drive folders with the service
   account's email (looks like xxx@yyy.iam.gserviceaccount.com) as Editor.
4. Put the full JSON key content into the GOOGLE_SERVICE_ACCOUNT_JSON secret.
"""
import datetime
import io
import json
import logging
from typing import Tuple

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from config import GOOGLE_SERVICE_ACCOUNT_JSON, DRIVE_INCOMING_FOLDER_ID

SCOPES = ["https://www.googleapis.com/auth/drive"]

log = logging.getLogger("drive_utils")


def is_file_ready(file_info: dict, min_age_seconds: float = 60.0) -> Tuple[bool, str]:
    """
    Verifies whether a Google Drive file has finished uploading and is stable for download.
    Checks:
    1. File size is present and > 0 bytes (rejects 0-byte or null-size uploads)
    2. File modification age is >= min_age_seconds (guards against in-progress chunked uploads)
    """
    if not isinstance(file_info, dict):
        return False, "File info must be a dictionary"

    # 1. Check size
    raw_size = file_info.get("size")
    if raw_size is None:
        return False, "File size is missing (possible incomplete upload)"
    try:
        size_bytes = int(raw_size)
        if size_bytes <= 0:
            return False, f"File size is {size_bytes} bytes (empty/incomplete upload)"
    except (ValueError, TypeError):
        return False, f"Invalid file size value: {raw_size}"

    # 2. Check modification age if modifiedTime is present
    mod_time_str = file_info.get("modifiedTime")
    if mod_time_str and min_age_seconds > 0.0:
        try:
            ts_clean = mod_time_str[:-1] + "+00:00" if mod_time_str.endswith("Z") else mod_time_str
            mod_dt = datetime.datetime.fromisoformat(ts_clean)
            if mod_dt.tzinfo is not None:
                now_dt = datetime.datetime.now(datetime.timezone.utc)
            else:
                now_dt = datetime.datetime.utcnow()
            age_sec = (now_dt - mod_dt).total_seconds()
            if age_sec < min_age_seconds:
                return False, f"File modified {age_sec:.1f}s ago (< {min_age_seconds}s threshold) — active upload in progress"
        except Exception as err:
            log.warning("Could not parse modifiedTime '%s': %s", mod_time_str, err)

    return True, "Ready"


def get_drive_service():
    info = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
    creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
    return build("drive", "v3", credentials=creds)


def list_new_videos(service, folder_id=DRIVE_INCOMING_FOLDER_ID, require_ready: bool = True):
    """
    Return list of {id, name, mimeType, size, createdTime, modifiedTime} for video files sitting in Incoming.
    When require_ready=True, skips files that are 0-byte or currently in the middle of active upload.
    """
    clean_id = folder_id.strip()
    query = (
        f"'{clean_id}' in parents "
        "and trashed = false "
        "and (mimeType contains 'video/')"
    )
    results = []
    page_token = None
    while True:
        resp = service.files().list(
            q=query,
            orderBy="createdTime asc",
            fields="nextPageToken, files(id, name, mimeType, size, createdTime, modifiedTime)",
            pageToken=page_token,
        ).execute()
        raw_files = resp.get("files", [])

        for f in raw_files:
            if require_ready:
                ready, reason = is_file_ready(f)
                if not ready:
                    log.warning(
                        "Incoming video '%s' (ID: %s) is not ready: %s. Skipping this run.",
                        f.get("name"), f.get("id"), reason,
                    )
                    continue
            results.append(f)

        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    log.info("Found %d ready video(s) in Incoming.", len(results))
    return results


def download_file(service, file_id, dest_path):
    request = service.files().get_media(fileId=file_id)
    with io.FileIO(dest_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                log.info("Download %d%%", int(status.progress() * 100))
    return dest_path


def move_file(service, file_id, from_folder_id, to_folder_id):
    """Move a file between folders (removes old parent, adds new one)."""
    service.files().update(
        fileId=file_id,
        addParents=to_folder_id.strip(),
        removeParents=from_folder_id.strip(),
        fields="id, parents",
    ).execute()
    log.info("Moved file %s -> folder %s", file_id, to_folder_id)


def get_file_metadata(service, file_id: str):
    """Fetches file metadata including whether it is trashed or still exists in Google Drive."""
    try:
        return service.files().get(
            fileId=file_id,
            fields="id, name, trashed, parents",
            supportsAllDrives=True,
        ).execute()
    except Exception as e:
        log.warning("Could not fetch metadata for file %s: %s", file_id, e)
        return None
