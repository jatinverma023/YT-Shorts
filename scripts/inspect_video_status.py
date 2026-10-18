#!/usr/bin/env python3
"""
Read-only inspection script for inspecting Google Sheets clip_queue rows.
Safe to run in any environment with Sheets credentials configured.
DOES NOT MODIFY any Google Sheets, Google Drive, or YouTube data.

Usage:
    python scripts/inspect_video_status.py [--drive-id DRIVE_FILE_ID]

Defaults to inspecting the Rashmika-Vijay video:
    1Zp82g0NzcpGGWHbBqVm_mgC2AA0b1Liy
"""
import argparse
import os
import sys

# Ensure scripts dir is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import sheet_log


DEFAULT_DRIVE_ID = "1Zp82g0NzcpGGWHbBqVm_mgC2AA0b1Liy"


def inspect_video_queue_status(drive_file_id: str):
    print("=" * 80)
    print(f"🔍 READ-ONLY QUEUE STATUS INSPECTION")
    print(f"Target Google Drive File ID: {drive_file_id}")
    print(f"Log Sheet ID:               {config.LOG_SHEET_ID or '[NOT CONFIGURED]'}")
    print(f"Clip Queue Tab:             {config.CLIP_QUEUE_TAB}")
    print("=" * 80)

    if not config.GOOGLE_SERVICE_ACCOUNT_JSON or not config.LOG_SHEET_ID:
        print("\n❌ Google Sheets credentials are not configured in this environment.")
        print("To run this inspection, configure GOOGLE_SERVICE_ACCOUNT_JSON and LOG_SHEET_ID in .env or environment.")
        return

    try:
        service = sheet_log.get_sheets_service()
        resp = service.spreadsheets().values().get(
            spreadsheetId=config.LOG_SHEET_ID,
            range=f"{config.CLIP_QUEUE_TAB}!A:M",
        ).execute()
    except Exception as e:
        print(f"\n❌ Failed to query Google Sheets API: {e}")
        return

    rows = resp.get("values", [])
    if not rows or len(rows) <= 1:
        print("\nℹ️ clip_queue sheet is empty (no data rows).")
        return

    headers = rows[0]
    matching_rows = []

    for idx, row in enumerate(rows[1:], start=2):
        padded = row + [""] * (13 - len(row))
        if padded[1].strip() == drive_file_id:
            matching_rows.append((idx, padded))

    if not matching_rows:
        print(f"\nℹ️ No rows found in clip_queue for drive_file_id='{drive_file_id}'.")
        print("This source has never been enqueued in this sheet, or was enqueued under a different ID.")
        return

    print(f"\nFound {len(matching_rows)} row(s) for drive_file_id='{drive_file_id}':\n")
    print(f"{'Row':<6} {'Idx':<5} {'Status':<22} {'Start-End':<14} {'Hook / Error / URL'}")
    print("-" * 80)

    active_statuses = {"pending", "processing", "retry_after_quota_reset", "rediscover_processing"}
    has_active = False
    has_rediscover = False
    all_terminal = True

    for row_num, r in matching_rows:
        clip_idx = r[2].strip() or "0"
        status = r[6].strip() or "[EMPTY]"
        start_t = r[3].strip() or "0"
        end_t = r[4].strip() or "0"
        hook = r[5].strip()
        yt_url = r[7].strip()
        err = r[8].strip()

        extra = yt_url or err or hook[:40]
        print(f"{row_num:<6} {clip_idx:<5} {status:<22} {start_t}-{end_t:<10} {extra}")

        st_lower = status.lower()
        if st_lower in active_statuses:
            has_active = True
            all_terminal = False
        elif st_lower == "rediscover":
            has_rediscover = True
            all_terminal = False
        elif st_lower in ("done", "failed", "no_valid_clips", "cancelled", "rediscovered"):
            pass
        else:
            all_terminal = False

    print("-" * 80)
    print("\n📊 Deduplication / Rediscovery Status Assessment:")
    if has_active:
        print("  Status: ACTIVE WORK EXISTS (pending / processing / retry_after_quota_reset)")
        print("  Action: Automatic rediscovery is BLOCKED. Active clips must finish processing first.")
    elif has_rediscover:
        print("  Status: EXPLICITLY AUTHORIZED FOR ONE-TIME REDISCOVERY ('rediscover')")
        print("  Action: The next scheduled run will discover and enqueue new clips for this video.")
    elif all_terminal:
        print("  Status: HISTORICAL TERMINAL ROWS ONLY (done / failed / no_valid_clips / cancelled)")
        print("  Action: Automatic rediscovery is BLOCKED by default (duplicate prevention).")
        print("  To authorize one-time re-discovery, an operator can run:")
        print(f"      python scripts/inspect_video_status.py --drive-id {drive_file_id} --authorize-rediscovery")
    else:
        print("  Status: UNKNOWN OR MIXED STATUSES")
        print("  Action: BLOCKED by default for safety.")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect or authorize clip_queue rows for a Google Drive video ID.")
    parser.add_argument("--drive-id", default=DEFAULT_DRIVE_ID, help="Google Drive file ID to inspect")
    parser.add_argument(
        "--authorize-rediscovery",
        action="store_true",
        help="Explicitly authorize one-time re-discovery for this video in Google Sheets",
    )
    args = parser.parse_args()

    if args.authorize_rediscovery:
        print("=" * 80)
        print(f"⚡ AUTHORIZING ONE-TIME REDISCOVERY FOR: {args.drive_id}")
        print("=" * 80)
        if not config.GOOGLE_SERVICE_ACCOUNT_JSON or not config.LOG_SHEET_ID:
            print("\n❌ Google Sheets credentials are not configured in this environment.")
            print("Configure GOOGLE_SERVICE_ACCOUNT_JSON and LOG_SHEET_ID to perform this update.")
            sys.exit(1)
        success = sheet_log.mark_video_for_rediscovery(args.drive_id)
        if success:
            print("\n✅ Successfully marked video for one-time re-discovery in clip_queue!")
            print("The next scheduled run will re-analyze this video from Incoming.")
        else:
            print("\n❌ Could not authorize re-discovery.")
            print("Either active work exists (pending/processing) or no terminal row was found.")
            sys.exit(1)
    else:
        inspect_video_queue_status(args.drive_id)
