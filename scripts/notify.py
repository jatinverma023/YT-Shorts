"""Send a success/failure ping to Telegram and/or Slack (whichever is configured)."""
import logging
import requests

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, SLACK_WEBHOOK_URL

log = logging.getLogger("notify")


def send(message: str):
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            resp = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=15)
            if resp.status_code == 200:
                log.info("Telegram notification sent successfully.")
            else:
                log.warning("Telegram notification failed (HTTP %d): %s", resp.status_code, resp.text)
        except Exception as e:
            log.warning("Telegram notify failed: %s", e)

    if SLACK_WEBHOOK_URL:
        try:
            resp = requests.post(SLACK_WEBHOOK_URL, json={"text": message}, timeout=15)
            if resp.status_code == 200:
                log.info("Slack notification sent successfully.")
            else:
                log.warning("Slack notification failed (HTTP %d): %s", resp.status_code, resp.text)
        except Exception as e:
            log.warning("Slack notify failed: %s", e)

    if not (TELEGRAM_BOT_TOKEN or SLACK_WEBHOOK_URL):
        log.info("No notification channel configured. Message was: %s", message)


def send_quota_warning(message: str):
    """Sends a distinct high-visibility notification for YouTube API quota exhaustion."""
    prefix = "⚠️ [YOUTUBE QUOTA ALERT] ⚠️\n"
    send(f"{prefix}{message}")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    test_msg = sys.argv[1] if len(sys.argv) > 1 else "🤖 Test notification from YouTube Shorts Automation pipeline: Telegram alerts are working properly! 🎉"
    log.info("Running manual notify test...")
    send(test_msg)


