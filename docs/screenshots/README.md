# Production Screenshot Guidelines

This directory stores visual assets and production screenshots for the YouTube Shorts Automation repository documentation.

## Required Screenshots

To maintain a professional, production-grade presentation without compromising security, please follow these guidelines when adding screenshots:

| File | Target View | Description |
|---|---|---|
| [`github-actions.png`](github-actions.png) | GitHub Actions workflow runs tab | Demonstrates automated 3-slot cron executions and manual dispatch runs. |
| [`google-sheets-queue.png`](google-sheets-queue.png) | Google Sheets `clip_queue` tab | Illustrates the durable state machine across `pending`, `processing`, and `done`. |
| [`google-drive.png`](google-drive.png) | Google Drive folders view | Displays source video ingestion in `Incoming` and completed video archival in `Processed`. |

## Privacy & Security Checklist

Before committing any screenshot:
- [ ] No API keys (`gsk_...`, `sk-...`, `AIza...`) visible
- [ ] No OAuth tokens or Google Service Account emails visible
- [ ] No personal emails or phone numbers visible
- [ ] No private folder IDs or Google Sheet IDs visible in the browser address bar
- [ ] No Telegram bot tokens or chat IDs visible
