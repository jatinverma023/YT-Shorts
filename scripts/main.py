"""
Orchestrator: run on a schedule (via GitHub Actions cron — see .github/workflows/pipeline.yml).
Durable multi-clip queue architecture:
  1. Checks Google Sheets durable queue (clip_queue tab) for any pending clips.
  2. If found: downloads source video, renders + uploads exactly ONE clip, marks done.
  3. If queue empty: checks Drive "Incoming" for new video, transcribes full video,
     detects N high-retention clips, enqueues them in clip_queue, and processes clip #1.
  4. Only moves the source video from Incoming to Processed once all its clips are done.
Fully unattended — survives ephemeral GitHub Actions runners without local state.
"""
import logging
import os
import shutil
import sys

from config import (
    WORKDIR, DRIVE_INCOMING_FOLDER_ID, DRIVE_PROCESSED_FOLDER_ID,
    DRIVE_FAILED_FOLDER_ID, DRY_RUN_LOG_ONLY,
)
import drive_utils
import transcribe
import video_process
import youtube_upload
import sheet_log
import notify
import metadata_ai
import clip_detection

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("main")


def check_and_move_if_completed(drive_service, drive_file_id: str, video_name: str):
    """
    Checks if all clips for this source video are finished (no pending or quota-waiting clips remain).
    If completed, moves the source video from Incoming to Processed in Google Drive.
    """
    counts = sheet_log.get_video_clip_counts(drive_file_id, service=None)
    log.info(
        "Clip status for '%s' (ID: %s): Total: %d, Pending: %d, Done: %d, Failed: %d, QuotaWait: %d",
        video_name, drive_file_id, counts["total"], counts["pending"], counts["done"], counts["failed"], counts.get("retry_after_quota_reset", 0),
    )

    # Only move when ALL clips are settled (pending == 0 and retry_after_quota_reset == 0)
    if counts["total"] > 0 and counts["pending"] == 0 and counts.get("retry_after_quota_reset", 0) == 0:
        if counts["done"] > 0:
            log.info("All clips completed for '%s'. Moving source video to Processed folder.", video_name)
            try:
                drive_utils.move_file(
                    drive_service, drive_file_id, DRIVE_INCOMING_FOLDER_ID, DRIVE_PROCESSED_FOLDER_ID
                )
            except Exception as e:
                log.warning("Could not move completed video to Processed folder: %s", e)
        else:
            log.warning("All clips failed for '%s'. Moving to Failed folder if configured.", video_name)
            if DRIVE_FAILED_FOLDER_ID:
                try:
                    drive_utils.move_file(
                        drive_service, drive_file_id, DRIVE_INCOMING_FOLDER_ID, DRIVE_FAILED_FOLDER_ID
                    )
                except Exception as e:
                    log.warning("Could not move failed video to Failed folder: %s", e)


def process_queue_clip(drive_service, row_number: int, clip: dict, local_src_path: str = None):
    """
    Processes and uploads a single clip from the durable queue:
    1. Downloads source video if not already available locally
    2. Slices exact start_time to end_time
    3. Transcribes clip slice & auto-trims dead air
    4. Burns animated ASS captions
    5. Renders vertical Short with parallax zoom, loudnorm, and mobile enhancements
    6. Generates transcript-grounded title variants, description, and tags
    7. Uploads to YouTube and updates queue status to 'done' (or 'failed' / 'retry_after_quota_reset')
    8. Moves source video to Processed if all its clips are done
    """
    drive_file_id = clip["drive_file_id"]
    video_name = clip["source_video_name"]
    clip_index = clip["clip_index"]
    start_time = clip["start_time"]
    end_time = clip["end_time"]
    base_name = os.path.splitext(video_name)[0]

    run_dir = os.path.join(WORKDIR, f"{drive_file_id}_clip{clip_index}")
    os.makedirs(run_dir, exist_ok=True)

    src_path = local_src_path
    cleanup_src = False

    try:
        log.info(
            "--- Processing Clip #%s for '%s' [%.1fs - %.1fs] ---",
            clip_index, video_name, start_time, end_time,
        )

        # 1. Ensure source video is downloaded
        if not src_path or not os.path.isfile(src_path):
            src_path = os.path.join(run_dir, video_name)
            log.info("Downloading source video from Google Drive (ID: %s)...", drive_file_id)
            drive_utils.download_file(drive_service, drive_file_id, src_path)
            cleanup_src = True

        # 2. Extract the exact clip segment
        slice_path = os.path.join(run_dir, f"{base_name}_clip{clip_index}_raw.mp4")
        video_process.extract_clip_segment(src_path, start_time, end_time, slice_path)

        # 3. Transcribe this specific clip segment for frame-accurate word timestamps
        clip_audio_path = os.path.join(run_dir, "clip_audio.mp3")
        transcribe.extract_audio(slice_path, clip_audio_path)
        trans_res = transcribe.transcribe_audio(clip_audio_path)
        words = trans_res["words"]
        detected_lang = trans_res["language"]
        clip_transcript = trans_res["text"]

        # 4. Auto-trim dead air (>0.5s) using Whisper word timestamps
        tightened_path = os.path.join(run_dir, "tightened.mp4")
        active_video, shifted_words = video_process.trim_silences_from_words(
            slice_path, words, tightened_path
        )

        # 5. Generate metadata & determine punchline hook grounded in this clip's transcript
        meta = metadata_ai.generate_shorts_metadata(
            f"{base_name} Part {clip_index}",
            transcript=clip_transcript,
        )
        title = meta["title"]
        variants = meta.get("title_variants", [title, title, title])
        description = meta["description"]
        tags = meta["tags"]
        punchline = clip.get("punchline") or meta.get("punchline") or clip.get("hook_summary", "")

        # 6. Generate animated, pop/karaoke word-level ASS captions with top punchline hook
        clip_duration = video_process.get_duration_seconds(active_video)
        ass_path = os.path.join(run_dir, "captions.ass")
        transcribe.generate_ass_captions(
            shifted_words,
            ass_path,
            punchline=punchline,
            total_duration=clip_duration,
        )

        # 7. Render final vertical Short with even-pixel scaling, loudnorm, top punchline, and burned captions
        out_path = os.path.join(run_dir, f"{base_name}_clip{clip_index}_short.mp4")
        video_process.process_video(active_video, out_path, ass_path=ass_path)

        # 8. Upload to YouTube Shorts
        youtube_url = youtube_upload.upload_short(out_path, title, description, tags=tags)

        # 9. Mark clip as done in durable queue and log run
        sheet_log.update_clip_status(row_number, "done", youtube_url=youtube_url)
        sheet_log.log_run(
            video_name, "SUCCESS", detected_lang=detected_lang, youtube_url=youtube_url,
            title_1=variants[0], title_2=variants[1], title_3=variants[2],
        )
        notify.send(
            f"✅ Uploaded Short ({video_name} Clip #{clip_index}):\n{title}\n💬 Hook: {punchline}\n{youtube_url}"
        )

        # 10. Check if all clips for this video are finished -> Move source to Processed
        check_and_move_if_completed(drive_service, drive_file_id, video_name)

    except youtube_upload.YouTubeQuotaExceededError as qe:
        log.warning("YouTube quota exceeded for clip #%s of %s: %s", clip_index, video_name, qe)
        sheet_log.update_clip_status(
            row_number,
            "retry_after_quota_reset",
            error="quota_exceeded — retry after quota reset",
        )
        sheet_log.log_run(
            video_name,
            f"QUOTA_EXCEEDED (Clip #{clip_index})",
            error="quota_exceeded — retry after quota reset",
        )
        notify.send_quota_warning(
            f"YouTube quota limit reached while uploading '{video_name}' (Clip #{clip_index}).\n"
            f"Status set to 'retry_after_quota_reset'. The source video remains safely in Incoming."
        )
        # Explicitly DO NOT call check_and_move_if_completed! File must stay in Incoming.

    except Exception as e:
        log.exception("Failed processing clip #%s for %s", clip_index, video_name)
        sheet_log.update_clip_status(row_number, "failed", error=str(e))
        sheet_log.log_run(video_name, f"FAILED (Clip #{clip_index})", error=str(e))
        notify.send(f"❌ Failed processing Clip #{clip_index} of {video_name}:\nError: {e}")
        check_and_move_if_completed(drive_service, drive_file_id, video_name)

    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def discover_and_enqueue_video(drive_service, file_info: dict):
    """
    Downloads a new long-form video, transcribes full audio, detects multiple
    high-engagement clips, persists them to the durable Google Sheets queue, and
    processes clip #1 immediately in the current run.
    """
    file_id = file_info["id"]
    name = file_info["name"]

    run_dir = os.path.join(WORKDIR, f"discovery_{file_id}")
    os.makedirs(run_dir, exist_ok=True)
    src_path = os.path.join(run_dir, name)
    full_audio_path = os.path.join(run_dir, "full_audio.mp3")

    try:
        log.info("Discovered new source video in Incoming: %s", name)
        log.info("Downloading full video for multi-clip analysis...")
        drive_utils.download_file(drive_service, file_id, src_path)

        total_duration = video_process.get_duration_seconds(src_path)
        log.info("Source video duration: %.1fs (%d min)", total_duration, int(total_duration / 60))

        # 1. Extract full audio and transcribe
        log.info("Extracting full audio for multi-clip transcription...")
        transcribe.extract_audio(src_path, full_audio_path)
        trans_res = transcribe.transcribe_audio(full_audio_path)
        segments = trans_res.get("segments", [])

        # 2. Run AI multi-clip detection
        log.info("Detecting viral, high-retention clips from full transcript...")
        clips = clip_detection.detect_clips_from_transcript(segments, total_duration)

        # 3. Persist detected clips into durable Google Sheets queue
        sheet_log.enqueue_clips(file_id, name, clips)
        notify.send(
            f"🎬 Discovered {len(clips)} clip(s) for '{name}' (Duration: {int(total_duration)}s). Enqueued in queue."
        )

        # 4. Immediately process the first clip in the same run (reusing downloaded file)
        pending = sheet_log.get_next_pending_clip()
        if pending:
            first_row_num, first_clip = pending
            process_queue_clip(drive_service, first_row_num, first_clip, local_src_path=src_path)

    except Exception as e:
        log.exception("Failed during clip discovery and enqueuing for %s", name)
        sheet_log.log_run(name, "FAILED (Discovery)", error=str(e))
        notify.send(f"❌ Failed analyzing clips for {name}:\nError: {e}")
        if DRIVE_FAILED_FOLDER_ID:
            try:
                drive_utils.move_file(drive_service, file_id, DRIVE_INCOMING_FOLDER_ID, DRIVE_FAILED_FOLDER_ID)
            except Exception:
                log.warning("Could not move failed video to Failed folder.")

    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def run_dry_run_inspection(drive_service=None, target_video_path: str = None):
    """
    Lightweight verification/debug mode (DRY_RUN_LOG_ONLY=true or --dry-run).
    Runs clip detection and metadata generation for a video without rendering video,
    without uploading to YouTube, without consuming quota, and without modifying Drive/Sheets.
    """
    log.info("================================================================================")
    log.info("🔍 DRY-RUN INSPECTION MODE ACTIVE (DRY_RUN_LOG_ONLY=True)")
    log.info("No video rendering, no YouTube uploads, and no quota will be consumed.")
    log.info("================================================================================")

    src_path = target_video_path
    cleanup_temp = False

    if not src_path:
        for arg in sys.argv[1:]:
            if arg != "--dry-run" and not arg.startswith("-") and os.path.isfile(arg):
                src_path = arg
                break

    if not src_path:
        drive_service = drive_service or drive_utils.get_drive_service()
        incoming = drive_utils.list_new_videos(drive_service)
        if not incoming:
            log.info("Dry-run: No videos found in Google Drive Incoming folder.")
            return []
        target = incoming[0]
        src_path = os.path.join(WORKDIR, f"dryrun_{target['name']}")
        os.makedirs(WORKDIR, exist_ok=True)
        log.info("Dry-run: Downloading incoming video '%s' (ID: %s)...", target["name"], target["id"])
        drive_utils.download_file(drive_service, target["id"], src_path)
        cleanup_temp = True

    try:
        video_name = os.path.basename(src_path)
        base_name = os.path.splitext(video_name)[0]
        total_duration = video_process.get_duration_seconds(src_path)
        log.info("Analyzing video '%s' (Duration: %.1fs / %d min)", video_name, total_duration, int(total_duration / 60))

        # 1. Extract audio & transcribe
        temp_audio = os.path.join(WORKDIR, f"dryrun_audio_{base_name}.mp3")
        transcribe.extract_audio(src_path, temp_audio)
        trans_res = transcribe.transcribe_audio(temp_audio)
        segments = trans_res.get("segments", [])
        words = trans_res.get("words", [])

        # 2. Detect clips
        clips = clip_detection.detect_clips_from_transcript(segments, total_duration)
        log.info("Detected %d candidate clip(s) for '%s':", len(clips), video_name)

        report = []
        # 3. Generate metadata for each clip slice
        for idx, clip in enumerate(clips, start=1):
            st = clip["start_time"]
            et = clip["end_time"]
            clip_words = [w["word"] for w in words if w.get("start", 0.0) >= st and w.get("end", 0.0) <= et]
            clip_text = " ".join(clip_words) if clip_words else clip.get("hook_summary", "")

            meta = metadata_ai.generate_shorts_metadata(
                f"{base_name} Part {idx}",
                transcript=clip_text,
            )
            title = meta["title"]
            variants = meta.get("title_variants", [title, title, title])

            punchline = clip.get("punchline") or meta.get("punchline", "")

            clip_report = {
                "clip_index": idx,
                "start_time": st,
                "end_time": et,
                "duration": round(et - st, 2),
                "hook_summary": clip.get("hook_summary", ""),
                "punchline": punchline,
                "primary_title": title,
                "title_variants": variants,
                "description": meta.get("description", ""),
                "tags": meta.get("tags", []),
                "transcript_preview": clip_text[:150] + ("..." if len(clip_text) > 150 else ""),
            }
            report.append(clip_report)

            print(f"\n--------------------------------------------------------------------------------")
            print(f"🎬 CLIP #{idx} [{st:.1f}s - {et:.1f}s] (Duration: {et - st:.1f}s)")
            print(f"  Punchline Hook: {punchline}")
            print(f"  Hook Summary:   {clip_report['hook_summary']}")
            print(f"  Primary Title:  {title}")
            print(f"  Variant 2:      {variants[1] if len(variants) > 1 else ''}")
            print(f"  Variant 3:      {variants[2] if len(variants) > 2 else ''}")
            print(f"  Transcript:     \"{clip_report['transcript_preview']}\"")
            print(f"  Tags:           {', '.join(clip_report['tags'][:5])}")
            print(f"--------------------------------------------------------------------------------")

        print(f"\n================================================================================")
        print(f"✅ DRY-RUN COMPLETE: {len(report)} clip(s) inspected. No quota used, no files uploaded.")
        print(f"================================================================================\n")
        return report

    finally:
        if cleanup_temp and os.path.exists(src_path):
            os.remove(src_path)


def main():
    os.makedirs(WORKDIR, exist_ok=True)

    if DRY_RUN_LOG_ONLY or "--dry-run" in sys.argv:
        run_dry_run_inspection()
        return

    drive_service = drive_utils.get_drive_service()
    sheet_log.ensure_clip_queue_sheet()

    # Step 0: Auto-reset quota-exhausted clips older than 20 hours (past YouTube midnight PT reset)
    sheet_log.reset_expired_quota_clips(min_age_hours=20.0)

    # Step A: Check durable queue for pending clips from previous runs/batches
    while True:
        pending = sheet_log.get_next_pending_clip()
        if not pending:
            break

        row_number, clip = pending
        drive_file_id = clip["drive_file_id"]
        video_name = clip["source_video_name"]

        # Verify source file still exists and is not trashed in Google Drive
        file_meta = drive_utils.get_file_metadata(drive_service, drive_file_id)
        if not file_meta or file_meta.get("trashed", False):
            log.warning(
                "Source video '%s' (ID: %s) was deleted or moved to trash in Google Drive. "
                "Cancelling remaining queued clips for this video.",
                video_name, drive_file_id,
            )
            sheet_log.cancel_clips_for_video(
                drive_file_id, reason="source_video_deleted_or_trashed", service=None
            )
            continue

        log.info(
            "Found pending clip in queue: row %d (Video: '%s', Clip #%s: %.1fs - %.1fs)",
            row_number, video_name, clip["clip_index"], clip["start_time"], clip["end_time"],
        )
        process_queue_clip(drive_service, row_number, clip)
        return

    # Step B: Check if any clip is waiting for quota reset
    if sheet_log.has_active_video_in_queue():
        log.info(
            "Queue has active clip(s) waiting for quota reset ('retry_after_quota_reset'). "
            "Skipping Incoming folder until current video is completely processed."
        )
        return

    # Step C: Queue is empty -> Check Drive "Incoming" for new videos
    log.info("Queue is completely empty. Checking Drive 'Incoming' folder for new videos...")
    incoming_videos = drive_utils.list_new_videos(drive_service)
    if not incoming_videos:
        log.info("No pending clips in queue and no new videos in Incoming folder. Run finished.")
        return

    # Pick ONLY the single oldest video (FIFO sequential ordering)
    target_video = incoming_videos[0]
    log.info(
        "Found %d new video(s) in Incoming. Starting with the single oldest: '%s' (ID: %s)",
        len(incoming_videos), target_video["name"], target_video["id"],
    )
    discover_and_enqueue_video(drive_service, target_video)


if __name__ == "__main__":
    main()

