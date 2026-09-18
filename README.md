# YouTube Shorts Automation Pipeline

Fully cloud-hosted pipeline: drop a video in a Google Drive folder → it gets
transcribed, subtitled (English or Hindi, auto-detected), resized to
1080x1920 for Shorts, and uploaded to your YouTube channel — automatically,
every 15 minutes, forever, with **no computer/device of yours needing to be
on**. It all runs on GitHub's cloud servers (GitHub Actions).

## How it works

```
Google Drive "Incoming" folder
        │
        ▼  (GitHub Actions cron job checks every 15 min)
GitHub Actions runner (ubuntu-latest, spins up fresh each run)
        │
        ├─ Download video (Google Drive API)
        ├─ Extract audio + transcribe (OpenAI Whisper API, auto EN/HI)
        ├─ Generate SRT captions
        ├─ FFmpeg: burn subtitles + resize/crop to 1080x1920
        ├─ Upload to YouTube as a Short (YouTube Data API v3)
        ├─ Move source file: Incoming → Processed (Drive API)
        └─ Log the run to a Google Sheet
        ▼
Telegram/Slack notification: success or failure
```

## One-time setup (~30–45 minutes, then it's hands-off forever)

### 1. Google Cloud project + APIs
1. Go to https://console.cloud.google.com → create a new project.
2. Enable these APIs: **Google Drive API**, **Google Sheets API**,
   **YouTube Data API v3**.

### 2. Service account (for Drive + Sheets access)
1. IAM & Admin → Service Accounts → Create.
2. Create a JSON key for it, download it.
3. In Google Drive, create two folders: `Incoming` and `Processed`
   (optionally a third, `Failed`). Share **both** with the service
   account's email (ends in `iam.gserviceaccount.com`) as **Editor**.
4. Create a Google Sheet for logging, with header row:
   `timestamp | source_filename | status | detected_lang | youtube_url | error`.
   Share it with the same service account email as **Editor**.
5. Copy the folder IDs (from each folder's URL) and the Sheet ID
   (from its URL) — you'll need these as secrets.

### 3. YouTube OAuth (uploads need a real user, not a service account)
1. APIs & Services → Credentials → Create OAuth Client ID → type **Desktop app**.
   Download the `client_secret.json`.
2. On your own computer (one-time only), run:
   ```
   pip install google-auth-oauthlib google-api-python-client
   python scripts/get_refresh_token.py
   ```
3. Log in via the browser popup with the Google account that owns your
   YouTube channel. Copy the printed `client_id`, `client_secret`, and
   `refresh_token`.

### 4. OpenAI API key (for Whisper transcription)
Get one at https://platform.openai.com/api-keys.

### 5. (Optional) Notifications
- **Telegram**: create a bot via @BotFather, get the bot token, and get
  your chat ID (message the bot, then check
  `https://api.telegram.org/bot<token>/getUpdates`).
- **Slack**: create an Incoming Webhook URL for a channel.

### 6. Push this project to GitHub
```
git init
git add .
git commit -m "Shorts automation pipeline"
git remote add origin <your-repo-url>
git push -u origin main
```

### 7. Add GitHub Secrets
Repo → Settings → Secrets and variables → Actions → New repository secret.
Add all of these:

| Secret name | Value |
|---|---|
| `DRIVE_INCOMING_FOLDER_ID` | Incoming folder ID |
| `DRIVE_PROCESSED_FOLDER_ID` | Processed folder ID |
| `DRIVE_FAILED_FOLDER_ID` | (optional) Failed folder ID |
| `GOOGLE_SERVICE_ACCOUNT_JSON` | full contents of the service account JSON key |
| `LOG_SHEET_ID` | Google Sheet ID |
| `OPENAI_API_KEY` | your OpenAI key |
| `YT_CLIENT_ID` | from step 3 |
| `YT_CLIENT_SECRET` | from step 3 |
| `YT_REFRESH_TOKEN` | from step 3 |
| `TELEGRAM_BOT_TOKEN` | (optional) |
| `TELEGRAM_CHAT_ID` | (optional) |
| `SLACK_WEBHOOK_URL` | (optional) |

### 8. Done
The workflow in `.github/workflows/pipeline.yml` runs automatically every
15 minutes. You can also trigger it manually from the **Actions** tab
("Run workflow" button) to test it immediately after setup — just drop a
short test video into the `Incoming` folder first.

## Notes / things to tune later
- **Video length**: YouTube Shorts limits currently run up to ~3 minutes,
  but this pipeline defaults `MAX_SHORT_SECONDS = 60` in `scripts/config.py`
  — change it, or verify the current limit before relying on a number here.
- **Cropping**: the current version does a centered crop to 9:16. If your
  source podcasts have the speaker off-center, look at the note in
  `scripts/video_process.py` about adding face-tracking crop as a v2.
- **Rights**: since you're reposting other creators' podcast clips, make
  sure you have permission or are adding enough transformative editorial
  value (commentary, curation, edits) to fit YouTube's reused-content
  policy — pure re-upload + subtitles risks takedowns or demonetization.
- **Cost**: GitHub Actions free tier gives you generous minutes for public
  repos (and a monthly allotment for private repos); Whisper API is billed
  per minute of audio — both are usage-based, no fixed server cost.
