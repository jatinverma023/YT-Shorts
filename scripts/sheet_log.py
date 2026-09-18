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

from config import GOOGLE_SERVICE_ACCOUNT_JSON, LOG_SHEET_ID, LOG_SHEET_TAB

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
log = logging.getLogger("sheet_log")


def get_sheets_service():
    info = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
    creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
    return build("sheets", "v4", credentials=creds)


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
