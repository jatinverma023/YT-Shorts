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
import metadata_ai

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
        clip_duration = min(duration, MAX_SHORT_SECONDS)
        if duration > MAX_SHORT_SECONDS:
            log.warning(
                "%s is %.1fs (> %ds). Trimming to first %ds for YouTube Shorts format.",
                name, duration, MAX_SHORT_SECONDS, MAX_SHORT_SECONDS,
            )

        # 1. Extract audio & transcribe with word-level timestamps
        log.info("Extracting audio and generating word-level transcription...")
        transcribe.extract_audio(src_path, audio_path, max_seconds=clip_duration)
        trans_res = transcribe.transcribe_audio(audio_path)
        words = trans_res["words"]
        detected_lang = trans_res["language"]
        transcript_text = trans_res["text"]

        # 2. Auto-trim silences/dead air (>0.5s) using Whisper word timestamps
        tightened_path = os.path.join(run_dir, "tightened.mp4")
        active_video, shifted_words = video_process.trim_silences_from_words(
            src_path, words, tightened_path, max_duration=clip_duration
        )

        # 3. Generate animated, pop/karaoke word-level ASS captions
        ass_path = os.path.join(run_dir, "captions.ass")
        transcribe.generate_ass_captions(shifted_words, ass_path)

        # 4. Render video with parallax zoom, loudnorm, even-pixel fix & burned ASS captions
        video_process.process_video(active_video, out_path, ass_path=ass_path, max_seconds=clip_duration)

        # 5. Generate 3 ranked title variants, description, and tags grounded in transcript
        meta = metadata_ai.generate_shorts_metadata(name, transcript=transcript_text)
        title = meta["title"]
        variants = meta.get("title_variants", [title, title, title])
        description = meta["description"]
        tags = meta["tags"]

        youtube_url = youtube_upload.upload_short(out_path, title, description, tags=tags)

        drive_utils.move_file(service, file_id, DRIVE_INCOMING_FOLDER_ID, DRIVE_PROCESSED_FOLDER_ID)
        sheet_log.log_run(
            name, "SUCCESS", detected_lang=detected_lang, youtube_url=youtube_url,
            title_1=variants[0], title_2=variants[1], title_3=variants[2],
        )
        notify.send(f"✅ Uploaded: {title}\n{youtube_url}")

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
