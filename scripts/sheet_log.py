"""
Append one row per processed video to a Google Sheet, using the same
service account as Drive (share the Sheet with it as Editor too).
Columns: timestamp | source_filename | status | detected_lang | youtube_url | error
"""
import datetime
import json
import logging

from google.oauth2 import service_account
from googleapiclient.discovery import build

from config import GOOGLE_SERVICE_ACCOUNT_JSON, LOG_SHEET_ID, LOG_SHEET_TAB, CLIP_QUEUE_TAB

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
log = logging.getLogger("sheet_log")

QUEUE_HEADERS = [
    "source_video_name",
    "drive_file_id",
    "clip_index",
    "start_time",
    "end_time",
    "hook_summary",
    "status",
    "youtube_url",
    "error",
    "created_at",
    "updated_at",
    "punchline",
    "quality_score",
]


def get_sheets_service():
    info = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
    creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
    return build("sheets", "v4", credentials=creds)


def ensure_clip_queue_sheet(service=None):
    """Ensures the clip_queue sheet tab and headers exist in the spreadsheet."""
    service = service or get_sheets_service()
    try:
        sheet_meta = service.spreadsheets().get(spreadsheetId=LOG_SHEET_ID).execute()
        existing_tabs = [s["properties"]["title"] for s in sheet_meta.get("sheets", [])]

        if CLIP_QUEUE_TAB not in existing_tabs:
            log.info("Creating '%s' sheet tab in Google Sheets...", CLIP_QUEUE_TAB)
            service.spreadsheets().batchUpdate(
                spreadsheetId=LOG_SHEET_ID,
                body={
                    "requests": [{
                        "addSheet": {
                            "properties": {"title": CLIP_QUEUE_TAB}
                        }
                    }]
                },
            ).execute()

        # Check if headers exist and are complete
        header_resp = service.spreadsheets().values().get(
            spreadsheetId=LOG_SHEET_ID,
            range=f"{CLIP_QUEUE_TAB}!A1:M1",
        ).execute()
        rows = header_resp.get("values", [])
        if not rows or not rows[0] or len(rows[0]) < len(QUEUE_HEADERS):
            log.info("Updating headers in '%s' tab...", CLIP_QUEUE_TAB)
            service.spreadsheets().values().update(
                spreadsheetId=LOG_SHEET_ID,
                range=f"{CLIP_QUEUE_TAB}!A1:M1",
                valueInputOption="RAW",
                body={"values": [QUEUE_HEADERS]},
            ).execute()
    except Exception as e:
        log.warning("Could not ensure clip_queue sheet: %s", e)


def enqueue_clips(drive_file_id: str, video_name: str, clips: list, service=None):
    """Appends detected clips as pending rows to the durable clip_queue."""
    service = service or get_sheets_service()
    now = datetime.datetime.utcnow().isoformat()
    rows = []

    for idx, clip in enumerate(clips, start=1):
        rows.append([
            video_name,
            drive_file_id,
            idx,
            float(clip["start_time"]),
            float(clip["end_time"]),
            str(clip.get("hook_summary", "")),
            "pending",
            "",  # youtube_url
            "",  # error
            now,  # created_at
            now,  # updated_at
            str(clip.get("punchline", "")),
            float(clip.get("quality_score", 0.0)),
        ])

    if not rows:
        return

    service.spreadsheets().values().append(
        spreadsheetId=LOG_SHEET_ID,
        range=f"{CLIP_QUEUE_TAB}!A:M",
        valueInputOption="RAW",
        insertDataOption="INSERT_ROWS",
        body={"values": rows},
    ).execute()
    log.info("Enqueued %d clips for video '%s' in %s", len(rows), video_name, CLIP_QUEUE_TAB)


def get_next_pending_clip(service=None):
    """
    Finds the first pending clip in the durable queue.
    Returns (row_number, clip_dict) or None if no pending clips exist.
    """
    service = service or get_sheets_service()
    try:
        resp = service.spreadsheets().values().get(
            spreadsheetId=LOG_SHEET_ID,
            range=f"{CLIP_QUEUE_TAB}!A:M",
        ).execute()
    except Exception as e:
        log.warning("Failed to fetch clip queue from sheet: %s", e)
        return None

    rows = resp.get("values", [])
    if len(rows) <= 1:
        return None

    for idx, row in enumerate(rows[1:], start=2):  # Row 1 is header, 1-indexed for Sheets
        # Pad row to at least 13 columns for backwards compatibility
        padded = row + [""] * (13 - len(row))
        status = padded[6].strip().lower()

        if status == "pending":
            try:
                clip_data = {
                    "source_video_name": padded[0],
                    "drive_file_id": padded[1],
                    "clip_index": int(padded[2]) if padded[2] else 1,
                    "start_time": float(padded[3]),
                    "end_time": float(padded[4]),
                    "hook_summary": padded[5],
                    "status": "pending",
                    "youtube_url": padded[7],
                    "error": padded[8],
                    "punchline": padded[11].strip() if len(padded) > 11 else "",
                    "quality_score": float(padded[12]) if len(padded) > 12 and padded[12] else 0.0,
                }
                return idx, clip_data
            except (ValueError, IndexError) as err:
                log.warning("Skipping malformed queue row %d: %s", idx, err)
                continue

    return None


def update_clip_status(row_number: int, status: str, youtube_url: str = "", error: str = "", service=None):
    """Updates the status and updated_at timestamp of a specific row in the clip_queue."""
    service = service or get_sheets_service()
    now = datetime.datetime.utcnow().isoformat()
    # Update columns G:I (status, youtube_url, error)
    service.spreadsheets().values().update(
        spreadsheetId=LOG_SHEET_ID,
        range=f"{CLIP_QUEUE_TAB}!G{row_number}:I{row_number}",
        valueInputOption="RAW",
        body={"values": [[status, youtube_url, error]]},
    ).execute()
    # Update column K (updated_at)
    service.spreadsheets().values().update(
        spreadsheetId=LOG_SHEET_ID,
        range=f"{CLIP_QUEUE_TAB}!K{row_number}",
        valueInputOption="RAW",
        body={"values": [[now]]},
    ).execute()
    log.info("Updated queue row %d -> status: %s", row_number, status)


def reset_expired_quota_clips(min_age_hours: float = 20.0, service=None) -> int:
    """
    Finds any clips in clip_queue with status 'retry_after_quota_reset' whose
    updated_at timestamp is older than min_age_hours (safely past YouTube's midnight PT quota reset),
    and resets their status back to 'pending' so they are processed normally in the current run.
    Returns the number of clips reset.
    """
    service = service or get_sheets_service()
    try:
        resp = service.spreadsheets().values().get(
            spreadsheetId=LOG_SHEET_ID,
            range=f"{CLIP_QUEUE_TAB}!A:K",
        ).execute()
    except Exception as e:
        log.warning("Failed to fetch clip queue for quota reset check: %s", e)
        return 0

    rows = resp.get("values", [])
    if len(rows) <= 1:
        return 0

    now = datetime.datetime.utcnow()
    reset_count = 0

    for idx, row in enumerate(rows[1:], start=2):
        padded = row + [""] * (11 - len(row))
        status = padded[6].strip().lower()

        if status == "retry_after_quota_reset":
            time_str = padded[10].strip() or padded[9].strip()
            should_reset = False

            if not time_str:
                should_reset = True
            else:
                try:
                    ts_clean = time_str[:-1] + "+00:00" if time_str.endswith("Z") else time_str
                    updated_time = datetime.datetime.fromisoformat(ts_clean)
                    if updated_time.tzinfo is not None:
                        now_ts = datetime.datetime.now(datetime.timezone.utc)
                    else:
                        now_ts = datetime.datetime.utcnow()
                    age_seconds = (now_ts - updated_time).total_seconds()
                    if age_seconds >= min_age_hours * 3600:
                        should_reset = True
                        log.info(
                            "Clip row %d was marked quota_exceeded %.1f hours ago (>= %.1f hours). Resetting to pending.",
                            idx, age_seconds / 3600, min_age_hours,
                        )
                    else:
                        log.info(
                            "Clip row %d marked quota_exceeded %.1f hours ago (< %.1f hours). Still waiting for quota reset.",
                            idx, age_seconds / 3600, min_age_hours,
                        )
                except Exception as err:
                    log.warning("Could not parse timestamp '%s' on row %d: %s. Defaulting to reset.", time_str, idx, err)
                    should_reset = True

            if should_reset:
                update_clip_status(
                    idx,
                    "pending",
                    youtube_url="",
                    error="",
                    service=service,
                )
                reset_count += 1

    if reset_count > 0:
        log.info("Reset %d quota-exhausted clip(s) back to 'pending'.", reset_count)
    return reset_count


def get_clip_by_status(target_status: str, service=None):
    """Finds the first clip in the durable queue with a specific status."""
    service = service or get_sheets_service()
    try:
        resp = service.spreadsheets().values().get(
            spreadsheetId=LOG_SHEET_ID,
            range=f"{CLIP_QUEUE_TAB}!A:L",
        ).execute()
    except Exception as e:
        log.warning("Failed to fetch clip queue from sheet: %s", e)
        return None

    rows = resp.get("values", [])
    if len(rows) <= 1:
        return None

    for idx, row in enumerate(rows[1:], start=2):
        padded = row + [""] * (12 - len(row))
        status = padded[6].strip().lower()
        if status == target_status.lower():
            try:
                return idx, {
                    "source_video_name": padded[0],
                    "drive_file_id": padded[1],
                    "clip_index": int(padded[2]) if padded[2] else 1,
                    "start_time": float(padded[3]),
                    "end_time": float(padded[4]),
                    "hook_summary": padded[5],
                    "status": status,
                    "youtube_url": padded[7],
                    "error": padded[8],
                    "punchline": padded[11].strip() if len(padded) > 11 else "",
                }
            except (ValueError, IndexError) as err:
                log.warning("Skipping malformed row %d: %s", idx, err)
                continue
    return None


def cancel_clips_for_video(drive_file_id: str, reason: str = "source_video_deleted", service=None) -> int:
    """Marks all pending or retry clips for a given drive_file_id as 'cancelled'."""
    service = service or get_sheets_service()
    try:
        resp = service.spreadsheets().values().get(
            spreadsheetId=LOG_SHEET_ID,
            range=f"{CLIP_QUEUE_TAB}!A:L",
        ).execute()
    except Exception as e:
        log.warning("Failed to fetch clip queue for cancellation: %s", e)
        return 0

    rows = resp.get("values", [])
    if len(rows) <= 1:
        return 0

    cancelled_count = 0
    for idx, row in enumerate(rows[1:], start=2):
        if len(row) > 1 and row[1].strip() == drive_file_id:
            status = row[6].strip().lower() if len(row) > 6 else ""
            if status in ("pending", "retry_after_quota_reset"):
                update_clip_status(idx, "cancelled", error=reason, service=service)
                cancelled_count += 1

    if cancelled_count > 0:
        log.info("Cancelled %d queued clip(s) for removed video (ID: %s)", cancelled_count, drive_file_id)
    return cancelled_count


def get_video_clip_counts(drive_file_id: str, service=None) -> dict:
    """Returns counts of total, pending, done, failed, and retry_after_quota_reset clips for a given drive_file_id."""
    service = service or get_sheets_service()
    counts = {"total": 0, "pending": 0, "done": 0, "failed": 0, "retry_after_quota_reset": 0}
    try:
        resp = service.spreadsheets().values().get(
            spreadsheetId=LOG_SHEET_ID,
            range=f"{CLIP_QUEUE_TAB}!B:G",
        ).execute()
    except Exception as e:
        log.warning("Failed to check video clip counts: %s", e)
        return counts

    rows = resp.get("values", [])
    if len(rows) <= 1:
        return counts

    for row in rows[1:]:
        if len(row) < 1:
            continue
        file_id = row[0].strip()
        if file_id == drive_file_id:
            counts["total"] += 1
            status = row[5].strip().lower() if len(row) > 5 else "pending"
            if status in counts:
                counts[status] += 1
            else:
                counts["pending"] += 1

    return counts


def has_active_video_in_queue(service=None) -> bool:
    """Returns True if there is ANY clip in the queue with status 'pending' or 'retry_after_quota_reset'."""
    service = service or get_sheets_service()
    try:
        resp = service.spreadsheets().values().get(
            spreadsheetId=LOG_SHEET_ID,
            range=f"{CLIP_QUEUE_TAB}!G:G",
        ).execute()
        rows = resp.get("values", [])
        for row in rows[1:]:
            if row:
                st = row[0].strip().lower()
                if st in ("pending", "retry_after_quota_reset"):
                    return True
    except Exception as e:
        log.warning("Could not check active clips in queue: %s", e)
    return False


def log_run(filename, status, detected_lang="", youtube_url="", title_1="", title_2="", title_3="", error=""):
    service = get_sheets_service()
    row = [[
        datetime.datetime.utcnow().isoformat(),
        filename,
        status,
        detected_lang,
        title_1,
        title_2,
        title_3,
        youtube_url,
        error,
    ]]
    service.spreadsheets().values().append(
        spreadsheetId=LOG_SHEET_ID,
        range=f"{LOG_SHEET_TAB}!A:I",
        valueInputOption="RAW",
        insertDataOption="INSERT_ROWS",
        body={"values": row},
    ).execute()
    log.info("Logged run for %s: %s (Title: %s)", filename, status, title_1)
