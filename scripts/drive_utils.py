"""
Google Drive helpers using a SERVICE ACCOUNT.

Setup required (one-time, done by you in Google Cloud Console):
1. Create a GCP project -> enable "Google Drive API".
2. Create a Service Account -> generate a JSON key.
3. Share your "Incoming" and "Processed" Drive folders with the service
   account's email (looks like xxx@yyy.iam.gserviceaccount.com) as Editor.
4. Put the full JSON key content into the GOOGLE_SERVICE_ACCOUNT_JSON secret.
"""
import io
import json
import logging

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from config import GOOGLE_SERVICE_ACCOUNT_JSON, DRIVE_INCOMING_FOLDER_ID

SCOPES = ["https://www.googleapis.com/auth/drive"]

log = logging.getLogger("drive_utils")


def get_drive_service():
    info = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
    creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
    return build("drive", "v3", credentials=creds)


def list_new_videos(service, folder_id=DRIVE_INCOMING_FOLDER_ID):
    """Return list of {id, name, mimeType} for video files sitting in Incoming."""
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
            fields="nextPageToken, files(id, name, mimeType, size)",
            pageToken=page_token,
        ).execute()
        results.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    log.info("Found %d new video(s) in Incoming.", len(results))
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
