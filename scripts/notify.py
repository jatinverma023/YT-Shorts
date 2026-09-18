"""Send a success/failure ping to Telegram and/or Slack (whichever is configured)."""
import logging
import requests

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, SLACK_WEBHOOK_URL

log = logging.getLogger("notify")


def send(message: str):
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=15)
        except Exception as e:
            log.warning("Telegram notify failed: %s", e)

    if SLACK_WEBHOOK_URL:
        try:
            requests.post(SLACK_WEBHOOK_URL, json={"text": message}, timeout=15)
        except Exception as e:
            log.warning("Slack notify failed: %s", e)

    if not (TELEGRAM_BOT_TOKEN or SLACK_WEBHOOK_URL):
        log.info("No notification channel configured. Message was: %s", message)
