"""
Central configuration. Everything is read from environment variables so that
in GitHub Actions these come from Encrypted Secrets, and locally (for testing)
you can use a .env file (never commit it).
"""
import os

try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
except Exception:
    pass

try:
    from dotenv import load_dotenv
    load_dotenv()
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except Exception:
    pass

# --- Google Drive folder IDs (get these from the folder URL in Drive) ---
DRIVE_INCOMING_FOLDER_ID = os.environ.get("DRIVE_INCOMING_FOLDER_ID", "").strip()
DRIVE_PROCESSED_FOLDER_ID = os.environ.get("DRIVE_PROCESSED_FOLDER_ID", "").strip()
DRIVE_FAILED_FOLDER_ID = os.environ.get("DRIVE_FAILED_FOLDER_ID", "").strip()

# --- Google service account (for Drive + Sheets) ---
# Store the FULL JSON key content (not a path) as a GitHub secret.
GOOGLE_SERVICE_ACCOUNT_JSON = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()

# --- Google Sheet used as a run log and durable clip queue ---
LOG_SHEET_ID = os.environ.get("LOG_SHEET_ID", "").strip()
LOG_SHEET_TAB = os.environ.get("LOG_SHEET_TAB", "Log").strip()
CLIP_QUEUE_TAB = os.environ.get("CLIP_QUEUE_TAB", "clip_queue").strip()

# --- Multi-Clip Detection & Splitting ---
MIN_CLIP_SECONDS = int(os.environ.get("MIN_CLIP_SECONDS", "20"))
MAX_CLIP_SECONDS = int(os.environ.get("MAX_CLIP_SECONDS", "59"))
MAX_CLIPS_PER_VIDEO = int(os.environ.get("MAX_CLIPS_PER_VIDEO", "5"))

# --- Transcription & LLM (Groq Whisper-large-v3 or OpenAI Whisper) ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
OPENAI_API_KEY = (os.environ.get("OPENAI_API_KEY", "") or GROQ_API_KEY).strip()
GROQ_CHAT_MODEL = os.environ.get("GROQ_CHAT_MODEL", "").strip()

# --- YouTube upload (OAuth2, not service account — YT upload needs a real user) ---
YT_CLIENT_ID = os.environ.get("YT_CLIENT_ID", "").strip()
YT_CLIENT_SECRET = os.environ.get("YT_CLIENT_SECRET", "").strip()
YT_REFRESH_TOKEN = os.environ.get("YT_REFRESH_TOKEN", "").strip()

# --- Notifications (pick one, both optional) ---
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "").strip()

# --- Video / caption styling ---
TARGET_WIDTH = 1080
TARGET_HEIGHT = 1920
MAX_SHORT_SECONDS = int(os.environ.get("MAX_SHORT_SECONDS", "59"))
SUBTITLE_FONT = os.environ.get("SUBTITLE_FONT", "Noto Sans Devanagari")
SUBTITLE_FONT_SIZE = int(os.environ.get("SUBTITLE_FONT_SIZE", "52"))  # ASS 1080x1920 PlayRes scale
CAPTION_HIGHLIGHT_COLOR = os.environ.get("CAPTION_HIGHLIGHT_COLOR", "&H0000FFFF&")  # Yellow in ASS hex
CAPTION_BASE_COLOR = os.environ.get("CAPTION_BASE_COLOR", "&H00FFFFFF&")            # White
CAPTION_OUTLINE_COLOR = os.environ.get("CAPTION_OUTLINE_COLOR", "&H00000000&")      # Black
WORDS_PER_PHRASE = int(os.environ.get("WORDS_PER_PHRASE", "4"))

# --- Top Punchline / Hook Header styling (e.g. 'Khan Sir with Raj Shamani 🥰') ---
ENABLE_TOP_PUNCHLINE = os.environ.get("ENABLE_TOP_PUNCHLINE", "true").lower() == "true"
PUNCHLINE_FONT = os.environ.get("PUNCHLINE_FONT", SUBTITLE_FONT)
PUNCHLINE_FONT_SIZE = int(os.environ.get("PUNCHLINE_FONT_SIZE", "52"))
PUNCHLINE_MARGIN_TOP = int(os.environ.get("PUNCHLINE_MARGIN_TOP", "280"))  # Vertical distance from top of 1920 canvas
PUNCHLINE_COLOR = os.environ.get("PUNCHLINE_COLOR", "&H00FFFFFF&")         # Crisp white text
PUNCHLINE_OUTLINE_COLOR = os.environ.get("PUNCHLINE_OUTLINE_COLOR", "&H00000000&")  # Solid dark outline


# --- Motion / Zoom (Parallax Depth) ---
ENABLE_ZOOM = os.environ.get("ENABLE_ZOOM", "false").lower() == "true"
BG_ZOOM_SPEED = float(os.environ.get("BG_ZOOM_SPEED", "0.0008"))
FG_ZOOM_SPEED = float(os.environ.get("FG_ZOOM_SPEED", "0.0004"))

# --- Audio & Pacing Optimization ---
ENABLE_SILENCE_REMOVAL = os.environ.get("ENABLE_SILENCE_REMOVAL", "true").lower() == "true"
SILENCE_THRESHOLD_SECONDS = float(os.environ.get("SILENCE_THRESHOLD_SECONDS", "0.5"))
SILENCE_PADDING_SECONDS = float(os.environ.get("SILENCE_PADDING_SECONDS", "0.12"))
ENABLE_LOUDNORM = os.environ.get("ENABLE_LOUDNORM", "true").lower() == "true"
LOUDNORM_TARGET_I = float(os.environ.get("LOUDNORM_TARGET_I", "-14.0"))

# --- YouTube upload defaults ---
DEFAULT_TAGS = ["shorts", "podcast", "clips"]
DEFAULT_CATEGORY_ID = "24"      # "Entertainment" — change if needed
UPLOAD_AS_PRIVATE_FIRST = False  # set True if you want to review before public

# --- Verification / Debug Mode ---
# When true, runs clip detection and metadata generation and logs the full plan
# without rendering video, uploading to YouTube, or consuming quota.
DRY_RUN_LOG_ONLY = os.environ.get("DRY_RUN_LOG_ONLY", "false").strip().lower() == "true"

WORKDIR = os.environ.get("WORKDIR", "/tmp/shorts_pipeline")

