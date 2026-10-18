"""
Append one row per processed video to a Google Sheet, using the same
service account as Drive (share the Sheet with it as Editor too).
Columns: timestamp | source_filename | status | detected_lang | youtube_url | error
"""
import datetime
import json
import logging
import time
from typing import Optional

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


def parse_sheet_float(val, default: Optional[float] = None) -> float:
    """
    Safely converts Google Sheets cell value to float.
    Handles numeric strings with commas (e.g. '7,674.1'), spaces, empty strings,
    or native numbers.
    """
    if val is None or val == "":
        if default is not None:
            return default
        raise ValueError("Empty value cannot be converted to float")
    if isinstance(val, (int, float)):
        return float(val)
    cleaned = str(val).strip().replace(",", "")
    if not cleaned:
        if default is not None:
            return default
        raise ValueError("Empty string cannot be converted to float")
    return float(cleaned)


def parse_sheet_int(val, default: int = 1) -> int:
    """
    Safely converts Google Sheets cell value to int.
    Handles numeric strings with commas (e.g. '1,000'), spaces, empty strings,
    or native numbers.
    """
    if val is None or val == "":
        return default
    if isinstance(val, int):
        return val
    cleaned = str(val).strip().replace(",", "")
    if not cleaned:
        return default
    try:
        return int(float(cleaned))
    except (ValueError, TypeError):
        return default


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
        clip_idx = int(clip.get("clip_index", idx))
        rows.append([
            video_name,
            drive_file_id,
            clip_idx,
            float(clip["start_time"]),
            float(clip["end_time"]),
            str(clip.get("topic_summary") or clip.get("topic") or clip.get("hook_summary", "")),
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


def enqueue_zero_clips(drive_file_id: str, video_name: str, reason: str = "no_valid_clips", candidate_count: int = 0, service=None):
    """
    Appends a terminal marker row for a video where 0 candidates met quality/standalone criteria,
    ensuring it is recorded as completed and never re-analyzed by get_enqueued_video_ids().
    """
    service = service or get_sheets_service()
    now = datetime.datetime.utcnow().isoformat()
    row = [[
        video_name,
        drive_file_id,
        0,
        0.0,
        0.0,
        f"Analyzed {candidate_count} candidate(s); 0 met quality/standalone criteria",
        "no_valid_clips",
        "",
        reason,
        now,
        now,
        "",
        0.0,
    ]]
    service.spreadsheets().values().append(
        spreadsheetId=LOG_SHEET_ID,
        range=f"{CLIP_QUEUE_TAB}!A:M",
        valueInputOption="RAW",
        insertDataOption="INSERT_ROWS",
        body={"values": row},
    ).execute()
    log.info("Recorded zero-valid-clips marker for video '%s' (ID: %s) in %s", video_name, drive_file_id, CLIP_QUEUE_TAB)


def get_active_video_in_queue(service=None):
    """
    Returns (drive_file_id, source_video_name) of the active source video in the queue,
    following FIFO order of queue entries.
    A video is active if it has at least one clip with status in ('pending', 'processing', 'retry_after_quota_reset').
    Returns None if no active video exists.
    """
    service = service or get_sheets_service()
    try:
        resp = service.spreadsheets().values().get(
            spreadsheetId=LOG_SHEET_ID,
            range=f"{CLIP_QUEUE_TAB}!A:G",
        ).execute()
    except Exception as e:
        log.warning("Failed to fetch clip queue for active video lookup: %s", e)
        return None

    rows = resp.get("values", [])
    if len(rows) <= 1:
        return None

    for row in rows[1:]:
        if len(row) < 2:
            continue
        v_name = row[0].strip()
        v_id = row[1].strip()
        status = row[6].strip().lower() if len(row) > 6 else ""
        if status in ("pending", "processing", "retry_after_quota_reset"):
            return v_id, v_name

    return None


def claim_clip(row_number: int, service=None) -> bool:
    """
    Defensively marks a pending clip as 'processing' to prevent overlapping runs
    from picking up the same clip. Verifies that the row's current status is 'pending'.
    Returns True if successfully claimed, or False if already claimed or no longer pending.
    """
    try:
        service = service or get_sheets_service()
        resp = service.spreadsheets().values().get(
            spreadsheetId=LOG_SHEET_ID,
            range=f"{CLIP_QUEUE_TAB}!G{row_number}",
        ).execute()
        vals = resp.get("values", [])
        current_status = vals[0][0].strip().lower() if vals and vals[0] else ""
        if current_status != "pending":
            log.warning("Cannot claim row %d: current status is '%s', not 'pending'.", row_number, current_status)
            return False
    except Exception as e:
        log.warning("Could not verify status before claiming row %d: %s. Attempting update anyway.", row_number, e)

    update_clip_status(row_number, "processing", service=service)
    return True


def get_next_pending_clip(service=None, target_drive_file_id: Optional[str] = None):
    """
    Finds the deterministic next pending clip to publish for the active video.
    Ordering:
      1. quality_score DESC
      2. row_number ASC (earlier discovery order)
      3. clip_index ASC
    If target_drive_file_id is not specified, identifies the active video first
    to preserve active video lock.
    Returns (row_number, clip_dict) or None.
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

    if not target_drive_file_id:
        active = get_active_video_in_queue(service=service)
        if not active:
            return None
        target_drive_file_id = active[0]

    if not target_drive_file_id:
        return None

    target_id = target_drive_file_id.strip()

    pending_candidates = []
    for idx, row in enumerate(rows[1:], start=2):  # Row 1 is header, 1-indexed for Sheets
        padded = row + [""] * (13 - len(row))
        if padded[1].strip() == target_id:
            status = padded[6].strip().lower()
            if status == "pending":
                try:
                    clip_data = {
                        "source_video_name": padded[0],
                        "drive_file_id": padded[1],
                        "clip_index": parse_sheet_int(padded[2], default=1),
                        "start_time": parse_sheet_float(padded[3]),
                        "end_time": parse_sheet_float(padded[4]),
                        "hook_summary": padded[5],
                        "status": "pending",
                        "youtube_url": padded[7],
                        "error": padded[8],
                        "punchline": padded[11].strip() if len(padded) > 11 else "",
                        "quality_score": parse_sheet_float(padded[12], default=0.0) if len(padded) > 12 and padded[12] else 0.0,
                    }
                    pending_candidates.append((idx, clip_data))
                except (ValueError, IndexError) as err:
                    log.warning("Skipping malformed queue row %d: %s", idx, err)
                    continue

    if not pending_candidates:
        return None

    # Deterministic selection:
    # 1. quality_score DESC
    # 2. row_number ASC (earlier discovery)
    # 3. clip_index ASC
    pending_candidates.sort(
        key=lambda item: (
            -parse_sheet_float(item[1].get("quality_score", 0.0), default=0.0),
            item[0],
            parse_sheet_int(item[1].get("clip_index", 1), default=1),
        )
    )
    return pending_candidates[0]


def get_pending_clips_for_video(drive_file_id: str, service=None) -> list:
    """
    Finds all pending clips in the durable queue for a specific source video (by drive_file_id).
    Returns list of (row_number, clip_dict) ordered by clip_index ascending.
    """
    service = service or get_sheets_service()
    try:
        resp = service.spreadsheets().values().get(
            spreadsheetId=LOG_SHEET_ID,
            range=f"{CLIP_QUEUE_TAB}!A:M",
        ).execute()
    except Exception as e:
        log.warning("Failed to fetch clip queue for video %s: %s", drive_file_id, e)
        return []

    rows = resp.get("values", [])
    if len(rows) <= 1:
        return []

    pending = []
    for idx, row in enumerate(rows[1:], start=2):
        padded = row + [""] * (13 - len(row))
        if padded[1].strip() == drive_file_id.strip():
            status = padded[6].strip().lower()
            if status == "pending":
                try:
                    clip_data = {
                        "source_video_name": padded[0],
                        "drive_file_id": padded[1],
                        "clip_index": parse_sheet_int(padded[2], default=1),
                        "start_time": parse_sheet_float(padded[3]),
                        "end_time": parse_sheet_float(padded[4]),
                        "hook_summary": padded[5],
                        "status": "pending",
                        "youtube_url": padded[7],
                        "error": padded[8],
                        "punchline": padded[11].strip() if len(padded) > 11 else "",
                        "quality_score": parse_sheet_float(padded[12], default=0.0) if len(padded) > 12 and padded[12] else 0.0,
                    }
                    pending.append((idx, clip_data))
                except (ValueError, IndexError) as err:
                    log.warning("Skipping malformed queue row %d for video %s: %s", idx, drive_file_id, err)
                    continue

    pending.sort(key=lambda x: parse_sheet_int(x[1].get("clip_index", 1), default=1))
    return pending


def update_clip_status(row_number: int, status: str, youtube_url: str = "", error: str = "", service=None, max_retries: int = 3):
    """Updates the status and updated_at timestamp of a specific row in the clip_queue with retry logic."""
    service = service or get_sheets_service()
    now = datetime.datetime.utcnow().isoformat()
    last_err = None

    for attempt in range(1, max_retries + 1):
        try:
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
            log.info("Updated queue row %d -> status: %s (attempt %d)", row_number, status, attempt)
            return
        except Exception as err:
            last_err = err
            log.warning("Attempt %d/%d to update queue row %d failed: %s", attempt, max_retries, row_number, err)
            if attempt < max_retries:
                time.sleep(1.0 * attempt)

    log.error("Failed to update queue row %d after %d attempts: %s", row_number, max_retries, last_err)
    if last_err is not None:
        raise last_err
    raise RuntimeError(f"Failed to update queue row {row_number}")


def update_clip_status_only(row_number: int, status: str, service=None, max_retries: int = 3):
    """
    Updates ONLY the status (column G) and updated_at timestamp (column K) of a specific row,
    without touching or overwriting column H (youtube_url) or column I (error).
    Used for safe re-discovery marker transitions to preserve historical metadata.
    """
    service = service or get_sheets_service()
    now = datetime.datetime.utcnow().isoformat()
    last_err = None

    for attempt in range(1, max_retries + 1):
        try:
            # Update strictly column G (status)
            service.spreadsheets().values().update(
                spreadsheetId=LOG_SHEET_ID,
                range=f"{CLIP_QUEUE_TAB}!G{row_number}",
                valueInputOption="RAW",
                body={"values": [[status]]},
            ).execute()
            # Update strictly column K (updated_at)
            service.spreadsheets().values().update(
                spreadsheetId=LOG_SHEET_ID,
                range=f"{CLIP_QUEUE_TAB}!K{row_number}",
                valueInputOption="RAW",
                body={"values": [[now]]},
            ).execute()
            log.info("Updated queue row %d status only -> %s (attempt %d)", row_number, status, attempt)
            return
        except Exception as err:
            last_err = err
            log.warning("Attempt %d/%d to update queue row %d status failed: %s", attempt, max_retries, row_number, err)
            if attempt < max_retries:
                time.sleep(1.0 * attempt)

    log.error("Failed to update queue row %d status after %d attempts: %s", row_number, max_retries, last_err)
    if last_err is not None:
        raise last_err
    raise RuntimeError(f"Failed to update queue row {row_number} status")



def reset_expired_quota_clips(min_age_hours: float = 20.0, service=None) -> int:
    """
    Finds any clips in clip_queue with status 'retry_after_quota_reset' whose
    updated_at timestamp is older than min_age_hours (safely past YouTube's midnight PT quota reset),
    and resets their status back to 'pending' so they are processed normally in the current run.
    Returns the number of clips reset.
    """
    try:
        service = service or get_sheets_service()
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


def reset_stale_processing_clips(max_age_minutes: float = 60.0, service=None) -> int:
    """
    Finds any clips in clip_queue with status 'processing' whose updated_at timestamp
    is older than max_age_minutes (indicates a crashed or timed out worker),
    and resets their status back to 'pending' so they can be retried cleanly.
    Returns the number of clips reset.
    """
    try:
        service = service or get_sheets_service()
        resp = service.spreadsheets().values().get(
            spreadsheetId=LOG_SHEET_ID,
            range=f"{CLIP_QUEUE_TAB}!A:K",
        ).execute()
    except Exception as e:
        log.warning("Failed to fetch clip queue for stale processing reset check: %s", e)
        return 0

    rows = resp.get("values", [])
    if len(rows) <= 1:
        return 0

    now = datetime.datetime.utcnow()
    reset_count = 0

    for idx, row in enumerate(rows[1:], start=2):
        padded = row + [""] * (11 - len(row))
        status = padded[6].strip().lower()

        if status in ("processing", "rediscover_processing"):
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
                    if age_seconds >= max_age_minutes * 60:
                        should_reset = True
                        log.info(
                            "Clip row %d was marked '%s' %.1f minutes ago (>= %.1f min). Resetting to %s.",
                            idx, status, age_seconds / 60, max_age_minutes,
                            "pending" if status == "processing" else "rediscover",
                        )
                except Exception as err:
                    log.warning("Could not parse timestamp '%s' on row %d: %s. Defaulting to reset.", time_str, idx, err)
                    should_reset = True

            if should_reset:
                target_status = "pending" if status == "processing" else "rediscover"
                if status == "rediscover_processing":
                    update_clip_status_only(idx, target_status, service=service)
                else:
                    update_clip_status(idx, target_status, service=service)
                reset_count += 1

    if reset_count > 0:
        log.info("Reset %d stale processing/rediscovery clip(s).", reset_count)
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
                    "clip_index": parse_sheet_int(padded[2], default=1),
                    "start_time": parse_sheet_float(padded[3]),
                    "end_time": parse_sheet_float(padded[4]),
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
            if status in ("pending", "processing", "retry_after_quota_reset"):
                update_clip_status(idx, "cancelled", error=reason, service=service)
                cancelled_count += 1

    if cancelled_count > 0:
        log.info("Cancelled %d queued clip(s) for removed video (ID: %s)", cancelled_count, drive_file_id)
    return cancelled_count


def get_video_clip_counts(drive_file_id: str, service=None) -> dict:
    """Returns counts of total, pending, processing, done, failed, retry_after_quota_reset, and no_valid_clips clips for a given drive_file_id."""
    service = service or get_sheets_service()
    counts = {
        "total": 0,
        "pending": 0,
        "processing": 0,
        "done": 0,
        "failed": 0,
        "retry_after_quota_reset": 0,
        "no_valid_clips": 0,
    }
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

    # Administrative/lifecycle markers that do NOT represent actionable clip items
    NON_CLIP_MARKERS = {"rediscovered", "rediscover", "rediscover_processing", "cancelled"}

    for row in rows[1:]:
        if len(row) < 1:
            continue
        file_id = row[0].strip()
        if file_id == drive_file_id:
            status = row[5].strip().lower() if len(row) > 5 and row[5].strip() else "unknown"
            if status == "no_valid_clips":
                counts["no_valid_clips"] += 1
            elif status in NON_CLIP_MARKERS:
                # Track marker presence without inflating actionable clip total or pending count
                counts[status] = counts.get(status, 0) + 1
            elif status in counts:
                counts["total"] += 1
                counts[status] += 1
            else:
                # Unknown / corrupted status: safely treat as an incomplete clip so video is not prematurely moved
                counts["total"] += 1
                counts["pending"] += 1

    return counts


def has_active_video_in_queue(service=None) -> bool:
    """Returns True if there is ANY clip in the queue with status 'pending', 'processing', or 'retry_after_quota_reset'."""
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
                if st in ("pending", "processing", "retry_after_quota_reset"):
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


def get_enqueued_video_ids(service=None) -> set:
    """
    Returns the set of Google Drive file IDs that should be blocked from re-discovery.

    A video is blocked if:
    - It has any rows with active statuses (pending, processing, retry_after_quota_reset, rediscover_processing), OR
    - It has only terminal rows (done, failed, no_valid_clips, cancelled, rediscovered) AND none
      of those rows have been explicitly marked 'rediscover' by an operator, OR
    - It has any row with an unknown or missing status (fail-safe protection).

    This ensures:
    - Completed historical videos remain deduplicated by default
    - Videos with pending/active work are never re-discovered mid-processing
    - An operator can authorize one-time re-discovery by setting status='rediscover' on any row
      (consumed after successful enqueueing via consume_rediscovery_flag)
    """
    # Defensive guard: if a Google Drive Resource or other non-Sheets service is passed by mistake,
    # safely discard it and obtain a valid Google Sheets service.
    if service is not None and not hasattr(service, "spreadsheets"):
        log.warning(
            "get_enqueued_video_ids received an invalid service object without 'spreadsheets' attribute (%s). "
            "Falling back to default Google Sheets service.",
            type(service),
        )
        service = None

    service = service or get_sheets_service()

    ACTIVE_STATUSES = {"pending", "processing", "retry_after_quota_reset", "rediscover_processing"}
    TERMINAL_STATUSES = {"done", "failed", "no_valid_clips", "cancelled", "rediscovered"}
    REDISCOVER_STATUS = "rediscover"

    active_ids = set()
    terminal_ids = set()
    rediscover_ids = set()

    try:
        resp = service.spreadsheets().values().get(
            spreadsheetId=LOG_SHEET_ID,
            range=f"{CLIP_QUEUE_TAB}!B:G",
        ).execute()
        rows = resp.get("values", [])
        for row in rows[1:]:  # skip header row
            if not row or not row[0].strip():
                continue
            drive_id = row[0].strip()
            status = row[5].strip().lower() if len(row) > 5 and row[5].strip() else ""
            if status in ACTIVE_STATUSES:
                active_ids.add(drive_id)
            elif status == REDISCOVER_STATUS:
                rediscover_ids.add(drive_id)
            elif status in TERMINAL_STATUSES:
                terminal_ids.add(drive_id)
            else:
                # Row with unknown or missing status — fail safely by treating as active (blocked)
                active_ids.add(drive_id)
    except Exception as e:
        log.error("Could not fetch enqueued video IDs from clip_queue: %s", e)
        raise RuntimeError(f"Could not fetch enqueued video IDs from clip_queue: {e}") from e

    # Active work ALWAYS takes precedence.
    # Blocked = all active IDs + terminal IDs that haven't been authorized for rediscovery.
    blocked_ids = active_ids | (terminal_ids - rediscover_ids)

    if rediscover_ids:
        authorized = rediscover_ids - active_ids
        if authorized:
            log.info(
                "Rediscovery authorized for %d video(s): %s",
                len(authorized), ", ".join(sorted(authorized)),
            )

    return blocked_ids


def mark_video_for_rediscovery(drive_file_id: str, service=None) -> bool:
    """
    Marks a historical video for re-discovery by setting the status of its
    first terminal row to 'rediscover'. This signals get_enqueued_video_ids()
    to exclude this video from the block list, allowing the next scheduled run
    to re-analyze it from Incoming.

    Active work ALWAYS takes precedence: if any row is 'pending', 'processing',
    'retry_after_quota_reset', or 'rediscover_processing', re-discovery authorization is rejected.

    Does NOT delete, modify YouTube URLs, or alter any other historical rows.
    Returns True if a row was marked, False if active work exists or no terminal row was found.
    """
    try:
        service = service or get_sheets_service()
        resp = service.spreadsheets().values().get(
            spreadsheetId=LOG_SHEET_ID,
            range=f"{CLIP_QUEUE_TAB}!B:G",
        ).execute()
    except Exception as e:
        log.warning("Failed to fetch clip queue for rediscovery marking: %s", e)
        return False

    rows = resp.get("values", [])
    active_statuses = {"pending", "processing", "retry_after_quota_reset", "rediscover_processing"}
    terminal_statuses = {"done", "failed", "no_valid_clips", "cancelled", "rediscovered"}

    # First verify no active row exists for this video
    for row in rows[1:]:
        if not row or not row[0].strip():
            continue
        file_id = row[0].strip()
        status = row[5].strip().lower() if len(row) > 5 else ""
        if file_id == drive_file_id and status in active_statuses:
            log.warning(
                "Cannot mark drive_file_id=%s for rediscovery: active work exists (status='%s').",
                drive_file_id, status,
            )
            return False

    marked_count = 0
    for idx, row in enumerate(rows[1:], start=2):
        if not row or not row[0].strip():
            continue
        file_id = row[0].strip()
        status = row[5].strip().lower() if len(row) > 5 else ""
        if file_id == drive_file_id and status in terminal_statuses:
            update_clip_status_only(idx, "rediscover", service=service)
            marked_count += 1
            log.info(
                "Marked row %d (drive_file_id=%s) as 'rediscover'. "
                "Next scheduled run will re-analyze this video from Incoming.",
                idx, drive_file_id,
            )

    if marked_count > 0:
        return True

    log.warning(
        "No terminal row found for drive_file_id=%s. Cannot mark for rediscovery.",
        drive_file_id,
    )
    return False


def claim_video_for_rediscovery(drive_file_id: str, service=None) -> bool:
    """
    Atomically claims a video marked for rediscovery by transitioning its
    'rediscover' row(s) to 'rediscover_processing'.
    This prevents concurrent runs from both discovering the same video.

    Returns:
    - True: If video is brand-new (no rows in clip_queue), or successfully claimed ('rediscover' -> 'rediscover_processing').
    - False: If active work exists ('pending', 'processing', 'retry_after_quota_reset', 'rediscover_processing'),
      or if the video has historical rows but none are 'rediscover'.
    """
    try:
        if not LOG_SHEET_ID:
            log.error("LOG_SHEET_ID is not configured. Cannot claim video for rediscovery.")
            return False
        service = service or get_sheets_service()
        resp = service.spreadsheets().values().get(
            spreadsheetId=LOG_SHEET_ID,
            range=f"{CLIP_QUEUE_TAB}!B:G",
        ).execute()
    except Exception as e:
        log.error("Failed to fetch clip queue for rediscovery claim: %s", e)
        return False

    rows = resp.get("values", [])
    active_statuses = {"pending", "processing", "retry_after_quota_reset", "rediscover_processing"}

    has_video_rows = False
    rediscover_row_indices = []

    for idx, row in enumerate(rows[1:], start=2):
        if not row or not row[0].strip():
            continue
        file_id = row[0].strip()
        if file_id == drive_file_id:
            has_video_rows = True
            status = row[5].strip().lower() if len(row) > 5 else ""
            if status in active_statuses:
                log.warning("Cannot claim %s for rediscovery: active work exists (status='%s').", drive_file_id, status)
                return False
            if status == "rediscover":
                rediscover_row_indices.append(idx)

    # If brand-new video (not previously in queue), claim succeeds as no-op
    if not has_video_rows:
        return True

    # If historical video but no rediscover row, cannot claim
    if not rediscover_row_indices:
        log.warning("Cannot claim %s for rediscovery: no 'rediscover' authorization row found.", drive_file_id)
        return False

    # Transition all rediscover rows to rediscover_processing
    for idx in rediscover_row_indices:
        update_clip_status_only(idx, "rediscover_processing", service=service)
        log.info("Claimed row %d for rediscovery of %s ('rediscover_processing').", idx, drive_file_id)

    return True


def revert_rediscovery_claim(drive_file_id: str, service=None) -> int:
    """
    Reverts 'rediscover_processing' rows back to 'rediscover' upon discovery failure,
    preserving the authorization state for a safe retry on subsequent runs.
    """
    try:
        service = service or get_sheets_service()
        resp = service.spreadsheets().values().get(
            spreadsheetId=LOG_SHEET_ID,
            range=f"{CLIP_QUEUE_TAB}!B:G",
        ).execute()
    except Exception as e:
        log.warning("Failed to fetch clip queue to revert rediscovery claim: %s", e)
        return 0

    rows = resp.get("values", [])
    reverted = 0
    for idx, row in enumerate(rows[1:], start=2):
        if not row or not row[0].strip():
            continue
        file_id = row[0].strip()
        status = row[5].strip().lower() if len(row) > 5 else ""
        if file_id == drive_file_id and status == "rediscover_processing":
            update_clip_status_only(idx, "rediscover", service=service)
            reverted += 1
            log.info("Reverted row %d for %s back to 'rediscover' after discovery failure.", idx, drive_file_id)

    return reverted


def consume_rediscovery_flag(drive_file_id: str, service=None) -> int:
    """
    Transitions all 'rediscover' and 'rediscover_processing' rows for a given drive_file_id to 'rediscovered',
    consuming the one-time re-discovery authorization after successful enqueueing.

    This ensures the rediscovery flag is not permanently active:
    - Next scheduled run will see 'rediscovered' (a terminal status) instead of 'rediscover'
    - Combined with newly enqueued 'pending' rows, the video is fully blocked again
    Returns the number of rows consumed. Returns 0 if no 'rediscover' rows exist (no-op for new videos).
    """
    try:
        service = service or get_sheets_service()
        resp = service.spreadsheets().values().get(
            spreadsheetId=LOG_SHEET_ID,
            range=f"{CLIP_QUEUE_TAB}!B:G",
        ).execute()
    except Exception as e:
        log.warning("Failed to fetch clip queue for rediscovery consumption: %s", e)
        return 0

    rows = resp.get("values", [])
    consumed = 0

    for idx, row in enumerate(rows[1:], start=2):
        if not row or not row[0].strip():
            continue
        file_id = row[0].strip()
        status = row[5].strip().lower() if len(row) > 5 else ""
        if file_id == drive_file_id and status in ("rediscover", "rediscover_processing"):
            update_clip_status_only(idx, "rediscovered", service=service)
            consumed += 1
            log.info(
                "Consumed rediscovery flag for row %d (drive_file_id=%s) -> 'rediscovered'.",
                idx, drive_file_id,
            )

    return consumed


def get_max_clip_index_for_video(drive_file_id: str, service=None) -> int:
    """
    Returns the maximum clip_index recorded in clip_queue for a given drive_file_id.
    Returns 0 if no historical clips exist or if previous entries had index 0 (e.g. no_valid_clips).
    """
    try:
        service = service or get_sheets_service()
        resp = service.spreadsheets().values().get(
            spreadsheetId=LOG_SHEET_ID,
            range=f"{CLIP_QUEUE_TAB}!B:C",
        ).execute()
        rows = resp.get("values", [])
        if not rows or len(rows) <= 1:
            return 0
        max_idx = 0
        for row in rows[1:]:
            if not row or not row[0].strip():
                continue
            if row[0].strip() == drive_file_id and len(row) > 1:
                idx = parse_sheet_int(row[1], default=0)
                if idx > max_idx:
                    max_idx = idx
        return max_idx
    except Exception as e:
        log.warning("Could not fetch max clip_index for video %s: %s", drive_file_id, e)
        return 0
