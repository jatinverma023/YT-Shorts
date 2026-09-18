"""
Central configuration. Everything is read from environment variables so that
in GitHub Actions these come from Encrypted Secrets, and locally (for testing)
you can use a .env file (never commit it).
"""
import os

# --- Google Drive folder IDs (get these from the folder URL in Drive) ---
DRIVE_INCOMING_FOLDER_ID = os.environ["DRIVE_INCOMING_FOLDER_ID"].strip()
DRIVE_PROCESSED_FOLDER_ID = os.environ["DRIVE_PROCESSED_FOLDER_ID"].strip()
DRIVE_FAILED_FOLDER_ID = os.environ.get("DRIVE_FAILED_FOLDER_ID", "").strip()

# --- Google service account (for Drive + Sheets) ---
# Store the FULL JSON key content (not a path) as a GitHub secret.
GOOGLE_SERVICE_ACCOUNT_JSON = os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"].strip()

# --- Google Sheet used as a run log ---
LOG_SHEET_ID = os.environ["LOG_SHEET_ID"].strip()
LOG_SHEET_TAB = os.environ.get("LOG_SHEET_TAB", "Log").strip()

# --- Transcription (Groq Whisper-large-v3 or OpenAI Whisper) ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
OPENAI_API_KEY = (os.environ.get("OPENAI_API_KEY", "") or GROQ_API_KEY).strip()

# --- YouTube upload (OAuth2, not service account — YT upload needs a real user) ---
YT_CLIENT_ID = os.environ["YT_CLIENT_ID"].strip()
YT_CLIENT_SECRET = os.environ["YT_CLIENT_SECRET"].strip()
YT_REFRESH_TOKEN = os.environ["YT_REFRESH_TOKEN"].strip()

# --- Notifications (pick one, both optional) ---
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "").strip()

# --- Video / caption styling ---
TARGET_WIDTH = 1080
TARGET_HEIGHT = 1920
MAX_SHORT_SECONDS = 60          # trim/reject if longer, or split into parts later
SUBTITLE_FONT = "DejaVu Sans"
SUBTITLE_FONT_SIZE = 14         # in libass "scale" terms, tuned in video_process.py

# --- YouTube upload defaults ---
DEFAULT_TAGS = ["shorts", "podcast", "clips"]
DEFAULT_CATEGORY_ID = "24"      # "Entertainment" — change if needed
UPLOAD_AS_PRIVATE_FIRST = False  # set True if you want to review before public

WORKDIR = os.environ.get("WORKDIR", "/tmp/shorts_pipeline")
