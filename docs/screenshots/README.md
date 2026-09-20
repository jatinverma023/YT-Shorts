# Production Screenshot Guidelines

This directory stores visual assets and production screenshots for the YouTube Shorts Automation repository documentation.

## Required Screenshots

To maintain a professional, production-grade presentation without compromising security, please follow these guidelines when adding screenshots:

| File | Target View | Sanitization Instructions |
|---|---|---|
| `github-actions.png` | GitHub Actions workflow runs tab | Ensure repository secrets and private account IDs are hidden. |
| `google-sheets-queue.png` | Google Sheets `clip_queue` tab showing state transitions | Hide spreadsheet ID, channel email, and sensitive sheet URLs. |
| `google-drive.png` | Google Drive folder view (`Incoming` / `Processed`) | Mask folder ID in the URL bar and any private personal file names. |
| `pipeline-logs.png` | Pipeline terminal / execution logs showing run banners | Mask API keys (Groq, OpenAI, Google) and OAuth tokens. |
| `youtube-short.png` | Final rendered YouTube Short playback | Clean preview of the rendered vertical layout on mobile or desktop. |
| `metadata.png` | YouTube Studio video details page | Showcase the AI-generated Title, Description, and Hashtags. |
| `caption-rendering.png` | Close-up of animated karaoke ASS captions & Top Hook | Illustrate the dual-overlay layout and Romanized captions. |
| `before-after.png` | Split comparison: landscape 16:9 source vs. vertical 9:16 Short | Highlights full-canvas composition and foreground scaling. |

## Privacy & Security Checklist

Before committing any screenshot:
- [ ] No API keys (`gsk_...`, `sk-...`, `AIza...`) visible
- [ ] No OAuth tokens or Google Service Account emails visible
- [ ] No personal emails or phone numbers visible
- [ ] No private folder IDs or Google Sheet IDs visible in the browser address bar
- [ ] No Telegram bot tokens or chat IDs visible
