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
MAX_CLIPS_PER_VIDEO = int(os.environ.get("MAX_CLIPS_PER_VIDEO", "5"))  # Legacy default
MAX_DISCOVERY_CANDIDATES = int(os.environ.get("MAX_DISCOVERY_CANDIDATES", "100"))  # Technical safety ceiling against malformed LLM output; NOT a target or desired limit
MIN_CLIP_QUALITY_SCORE = float(os.environ.get("MIN_CLIP_QUALITY_SCORE", "70.0"))
CLIP_OVERLAP_THRESHOLD = float(os.environ.get("CLIP_OVERLAP_THRESHOLD", "0.35"))
MIN_STANDALONE_SCORE = float(os.environ.get("MIN_STANDALONE_SCORE", "7.0"))


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
SUBTITLE_FONT = os.environ.get("SUBTITLE_FONT", "Arial")
SUBTITLE_FONT_SIZE = int(os.environ.get("SUBTITLE_FONT_SIZE", "52"))  # ASS 1080x1920 PlayRes scale
CAPTION_HIGHLIGHT_COLOR = os.environ.get("CAPTION_HIGHLIGHT_COLOR", "&H0000FFFF&")  # Yellow in ASS hex
CAPTION_BASE_COLOR = os.environ.get("CAPTION_BASE_COLOR", "&H00FFFFFF&")            # White
CAPTION_OUTLINE_COLOR = os.environ.get("CAPTION_OUTLINE_COLOR", "&H00000000&")      # Black
WORDS_PER_PHRASE = int(os.environ.get("WORDS_PER_PHRASE", "4"))
CAPTION_LANGUAGE_MODE = os.environ.get("CAPTION_LANGUAGE_MODE", "romanized").strip().lower()
CAPTION_UPPERCASE = os.environ.get("CAPTION_UPPERCASE", "true").strip().lower() == "true"
CAPTION_MAX_LINES = int(os.environ.get("CAPTION_MAX_LINES", "2"))
CAPTION_MAX_CHARS_PER_LINE = int(os.environ.get("CAPTION_MAX_CHARS_PER_LINE", "30"))
CAPTION_MAX_WORDS = int(os.environ.get("CAPTION_MAX_WORDS", "8"))
CAPTION_MARGIN_BOTTOM = int(os.environ.get("CAPTION_MARGIN_BOTTOM", "280"))

# --- Top Punchline / Hook Header styling (e.g. 'Khan Sir with Raj Shamani 🥰') ---
ENABLE_TOP_PUNCHLINE = os.environ.get("ENABLE_TOP_PUNCHLINE", "true").lower() == "true"
PUNCHLINE_FONT = os.environ.get("PUNCHLINE_FONT", SUBTITLE_FONT)
PUNCHLINE_FONT_SIZE = int(os.environ.get("PUNCHLINE_FONT_SIZE", "52"))
PUNCHLINE_MARGIN_TOP = int(os.environ.get("PUNCHLINE_MARGIN_TOP", "280"))  # Vertical distance from top of 1920 canvas
PUNCHLINE_COLOR = os.environ.get("PUNCHLINE_COLOR", "&H00FFFFFF&")         # Crisp white text
PUNCHLINE_OUTLINE_COLOR = os.environ.get("PUNCHLINE_OUTLINE_COLOR", "&H00000000&")  # Solid dark outline


# --- Visual Upgrade #1A: Cinematic Composition, Color Treatment, Vignette & Micro-Motion ---
VISUAL_CONTRAST = float(os.environ.get("VISUAL_CONTRAST", "1.05"))
VISUAL_SATURATION = float(os.environ.get("VISUAL_SATURATION", "1.10"))
VISUAL_BRIGHTNESS = float(os.environ.get("VISUAL_BRIGHTNESS", "0.01"))
VISUAL_SHARPEN_AMOUNT = float(os.environ.get("VISUAL_SHARPEN_AMOUNT", "0.60"))
VISUAL_VIGNETTE_ENABLED = os.environ.get("VISUAL_VIGNETTE_ENABLED", "true").lower() == "true"
VISUAL_VIGNETTE_STRENGTH = float(os.environ.get("VISUAL_VIGNETTE_STRENGTH", "0.25"))
VISUAL_MOTION_ENABLED = os.environ.get("VISUAL_MOTION_ENABLED", "true").lower() == "true"
VISUAL_MOTION_MAX_ZOOM = float(os.environ.get("VISUAL_MOTION_MAX_ZOOM", "1.04"))
BG_BRIGHTNESS = float(os.environ.get("BG_BRIGHTNESS", "-0.08"))
BG_SATURATION = float(os.environ.get("BG_SATURATION", "1.05"))
FOREGROUND_SCALE = float(os.environ.get("FOREGROUND_SCALE", "1.18"))
TARGET_FPS = int(os.environ.get("TARGET_FPS", "0"))
ENABLE_ZOOM = VISUAL_MOTION_ENABLED
BG_ZOOM_SPEED = float(os.environ.get("BG_ZOOM_SPEED", "0.0008"))
FG_ZOOM_SPEED = float(os.environ.get("FG_ZOOM_SPEED", "0.0004"))

# --- Visual Upgrade #1B: Content-Aware Dynamic Emphasis ---
CONTENT_MOTION_ENABLED = os.environ.get("CONTENT_MOTION_ENABLED", "true").strip().lower() == "true"
CONTENT_MOTION_MAX_ZOOM = float(os.environ.get("CONTENT_MOTION_MAX_ZOOM", "1.04"))
CONTENT_MOTION_MIN_ZOOM = float(os.environ.get("CONTENT_MOTION_MIN_ZOOM", "1.00"))
CONTENT_MOTION_IN_DURATION = float(os.environ.get("CONTENT_MOTION_IN_DURATION", "0.35"))
CONTENT_MOTION_OUT_DURATION = float(os.environ.get("CONTENT_MOTION_OUT_DURATION", "0.45"))
CONTENT_MOTION_MAX_POINTS = int(os.environ.get("CONTENT_MOTION_MAX_POINTS", "3"))

# --- Audio & Pacing Optimization ---
ENABLE_SILENCE_REMOVAL = os.environ.get("ENABLE_SILENCE_REMOVAL", "true").lower() == "true"
SILENCE_THRESHOLD_SECONDS = float(os.environ.get("SILENCE_THRESHOLD_SECONDS", "0.7"))
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

# --- YouTube Metadata Generation (Title, Description, Hashtags, Hook & Packaging) ---
MIN_TITLE_QUALITY_SCORE = float(os.environ.get("MIN_TITLE_QUALITY_SCORE", "70.0"))
TITLE_MIN_WORDS = int(os.environ.get("TITLE_MIN_WORDS", "5"))
TITLE_MAX_WORDS = int(os.environ.get("TITLE_MAX_WORDS", "12"))
TITLE_MAX_LENGTH = int(os.environ.get("TITLE_MAX_LENGTH", "70"))
MAX_TITLE_CANDIDATES = int(os.environ.get("MAX_TITLE_CANDIDATES", "5"))
MIN_HASHTAGS = int(os.environ.get("MIN_HASHTAGS", "4"))
MAX_HASHTAGS = int(os.environ.get("MAX_HASHTAGS", "6"))

# --- Hook & Content Packaging Configuration ---
MIN_HOOK_QUALITY_SCORE = float(os.environ.get("MIN_HOOK_QUALITY_SCORE", "70.0"))
HOOK_MIN_WORDS = int(os.environ.get("HOOK_MIN_WORDS", "3"))
HOOK_MAX_WORDS = int(os.environ.get("HOOK_MAX_WORDS", "8"))
HOOK_MAX_LENGTH = int(os.environ.get("HOOK_MAX_LENGTH", "42"))
MAX_HOOK_CANDIDATES = int(os.environ.get("MAX_HOOK_CANDIDATES", "5"))
MAX_HOOK_EMOJIS = int(os.environ.get("MAX_HOOK_EMOJIS", "2"))
PREFERRED_HOOK_EMOJIS = int(os.environ.get("PREFERRED_HOOK_EMOJIS", "1"))
MIN_DESCRIPTION_QUALITY_SCORE = float(os.environ.get("MIN_DESCRIPTION_QUALITY_SCORE", "70.0"))
MAX_DESCRIPTION_CANDIDATES = int(os.environ.get("MAX_DESCRIPTION_CANDIDATES", "5"))
MIN_PACKAGE_QUALITY_SCORE = float(os.environ.get("MIN_PACKAGE_QUALITY_SCORE", "70.0"))



