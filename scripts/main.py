"""
Orchestrator: run on a schedule (via GitHub Actions cron — see
.github/workflows/pipeline.yml). Each run:
  1. Lists new videos in Drive "Incoming"
  2. For each: download -> transcribe -> caption -> resize -> upload -> log -> move
  3. Notifies you of the outcome
Fully unattended — no device of yours needs to be on.
"""
import logging
import os
import shutil

from config import (
    WORKDIR, DRIVE_INCOMING_FOLDER_ID, DRIVE_PROCESSED_FOLDER_ID,
    DRIVE_FAILED_FOLDER_ID, MAX_SHORT_SECONDS,
)
import drive_utils
import transcribe
import video_process
import youtube_upload
import sheet_log
import notify

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("main")


def process_one(service, file_info):
    file_id = file_info["id"]
    name = file_info["name"]
    base = os.path.splitext(name)[0]

    run_dir = os.path.join(WORKDIR, file_id)
    os.makedirs(run_dir, exist_ok=True)

    src_path = os.path.join(run_dir, name)
    audio_path = os.path.join(run_dir, "audio.wav")
    srt_path = os.path.join(run_dir, "captions.srt")
    out_path = os.path.join(run_dir, f"{base}_short.mp4")

    try:
        log.info("Processing %s", name)
        drive_utils.download_file(service, file_id, src_path)

        duration = video_process.get_duration_seconds(src_path)
        if duration > MAX_SHORT_SECONDS:
            log.warning(
                "%s is %.1fs (> %ds). Uploading trimmed/first %ds only. "
                "Consider pre-splitting long episodes into clips before this stage.",
                name, duration, MAX_SHORT_SECONDS, MAX_SHORT_SECONDS,
            )

        transcribe.extract_audio(src_path, audio_path)
        _, detected_lang = transcribe.transcribe_to_srt(audio_path, srt_path)

        video_process.process_video(src_path, srt_path, out_path)

        title = base.replace("_", " ").replace("-", " ").strip()[:90] + " #shorts"
        description = (
            f"{title}\n\nClip auto-captioned in "
            f"{'Hindi' if detected_lang == 'hi' else 'English'}. "
            "#shorts #podcast"
        )
        youtube_url = youtube_upload.upload_short(out_path, title, description)

        drive_utils.move_file(service, file_id, DRIVE_INCOMING_FOLDER_ID, DRIVE_PROCESSED_FOLDER_ID)
        sheet_log.log_run(name, "SUCCESS", detected_lang, youtube_url)
        notify.send(f"✅ Uploaded: {name}\n{youtube_url}")

    except Exception as e:
        log.exception("Failed processing %s", name)
        if DRIVE_FAILED_FOLDER_ID:
            try:
                drive_utils.move_file(service, file_id, DRIVE_INCOMING_FOLDER_ID, DRIVE_FAILED_FOLDER_ID)
            except Exception:
                log.warning("Could not move failed file to Failed folder.")
        sheet_log.log_run(name, "FAILED", error=str(e))
        notify.send(f"❌ Failed: {name}\nError: {e}")

    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def main():
    os.makedirs(WORKDIR, exist_ok=True)
    service = drive_utils.get_drive_service()
    videos = drive_utils.list_new_videos(service)

    if not videos:
        log.info("No new videos found. Nothing to do.")
        return

    for file_info in videos:
        process_one(service, file_info)


if __name__ == "__main__":
    main()
