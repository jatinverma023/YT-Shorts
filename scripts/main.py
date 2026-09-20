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
import datetime
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
import hook_generator
import run_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("main")


def check_and_move_if_completed(drive_service, drive_file_id: str, video_name: str):
    """
    Checks if all clips for this source video are finished.
    Moves the source video from Incoming to Processed ONLY when ALL generated clips are 'done' (100% success)
    or if the video has a terminal 'no_valid_clips' status.
    If any clip is pending, processing, failed, or quota-waiting, the source video MUST remain in Incoming.
    """
    counts = sheet_log.get_video_clip_counts(drive_file_id, service=None)
    log.info(
        "Clip status for '%s' (ID: %s): Total: %d, Pending: %d, Processing: %d, Done: %d, Failed: %d, QuotaWait: %d, NoValid: %d",
        video_name, drive_file_id, counts["total"], counts["pending"], counts.get("processing", 0),
        counts["done"], counts["failed"], counts.get("retry_after_quota_reset", 0), counts.get("no_valid_clips", 0),
    )

    report = run_report.get_current_report()

    # Case 0: Terminal zero-valid-clips video (Correction #6)
    if counts.get("no_valid_clips", 0) > 0 and counts["pending"] == 0 and counts.get("processing", 0) == 0:
        log.info("Source video '%s' had zero valid clips. Moving to Processed folder.", video_name)
        if report:
            report.set_source(video_name, file_id=drive_file_id, destination="Moved to Processed (no valid clips)")
            report.summary_source_status = "Moved to Processed (no valid clips)"
            report.update_stage("drive_movement", "✅ Moved to Processed (no valid clips)")
        try:
            drive_utils.move_file(
                drive_service, drive_file_id, DRIVE_INCOMING_FOLDER_ID, DRIVE_PROCESSED_FOLDER_ID
            )
        except Exception as e:
            log.warning("Could not move zero-valid-clips video to Processed folder: %s", e)
        return

    # Strictly require ALL generated clips to be completed successfully before moving to Processed
    if counts["total"] > 0 and counts["done"] == counts["total"]:
        log.info("All %d clip(s) completed successfully for '%s'. Moving source video to Processed folder.", counts["total"], video_name)
        if report:
            report.set_source(video_name, file_id=drive_file_id, destination="Moved to Processed")
            report.summary_source_status = "Moved to Processed"
            report.update_stage("drive_movement", "✅ Moved to Processed")
        try:
            drive_utils.move_file(
                drive_service, drive_file_id, DRIVE_INCOMING_FOLDER_ID, DRIVE_PROCESSED_FOLDER_ID
            )
        except Exception as e:
            log.warning("Could not move completed video to Processed folder: %s", e)
    elif counts["total"] > 0 and counts["pending"] == 0 and counts.get("processing", 0) == 0 and counts.get("retry_after_quota_reset", 0) == 0 and counts["failed"] == counts["total"]:
        log.warning("All clips failed for '%s'. Moving to Failed folder if configured.", video_name)
        if report:
            report.set_source(video_name, file_id=drive_file_id, destination="Moved to Failed")
            report.summary_source_status = "Moved to Failed"
            report.update_stage("drive_movement", "❌ Moved to Failed")
        if DRIVE_FAILED_FOLDER_ID:
            try:
                drive_utils.move_file(
                    drive_service, drive_file_id, DRIVE_INCOMING_FOLDER_ID, DRIVE_FAILED_FOLDER_ID
                )
            except Exception as e:
                log.warning("Could not move failed video to Failed folder: %s", e)
    else:
        log.info(
            "Source video '%s' has incomplete clips (Total: %d, Pending: %d, Processing: %d, Done: %d, Failed: %d). Preserving in Incoming.",
            video_name, counts["total"], counts["pending"], counts.get("processing", 0), counts["done"], counts["failed"]
        )
        if report:
            report.set_source(video_name, file_id=drive_file_id, destination="Preserved in Incoming")
            report.summary_source_status = "Preserved in Incoming"
            report.update_stage("drive_movement", "⏸️ Preserved in Incoming")


def process_queue_clip(drive_service, row_number: int, clip: dict, local_src_path: str = None) -> tuple:
    """
    Processes and uploads a single clip from the durable queue:
    1. Checks if clip was already uploaded to YouTube (idempotency recovery)
    2. Downloads source video if not already available locally
    3. Slices exact start_time to end_time
    4. Transcribes clip slice & auto-trims dead air (>0.5s)
    5. Burns animated ASS captions + top punchline hook
    6. Renders vertical Short with parallax zoom, loudnorm, and mobile enhancements
    7. Generates transcript-grounded metadata
    8. Uploads to YouTube and updates queue status to 'done' (or 'failed' / 'retry_after_quota_reset')
    Returns (success: bool, youtube_url: str).
    """
    drive_file_id = clip["drive_file_id"]
    video_name = clip["source_video_name"]
    clip_index = clip["clip_index"]
    start_time = clip["start_time"]
    end_time = clip["end_time"]
    base_name = os.path.splitext(video_name)[0]
    clip_identifier = f"clip_{drive_file_id}_{clip_index}"

    run_dir = os.path.join(WORKDIR, f"{drive_file_id}_clip{clip_index}")
    os.makedirs(run_dir, exist_ok=True)

    src_path = local_src_path
    cleanup_src = False
    youtube_url = None

    try:
        log.info(
            "--- Processing Clip #%s for '%s' [%.1fs - %.1fs] ---",
            clip_index, video_name, start_time, end_time,
        )

        # 1. Idempotency Check: Was this clip already uploaded?
        existing_url = clip.get("youtube_url")
        if not existing_url:
            existing_url = youtube_upload.find_existing_short(clip_identifier=clip_identifier)

        if existing_url:
            log.info(
                "Clip #%s for '%s' was already uploaded to YouTube (found URL: %s). Re-using existing upload without duplicate.",
                clip_index, video_name, existing_url,
            )
            youtube_url = existing_url
            sheet_log.update_clip_status(row_number, "done", youtube_url=youtube_url)
            return True, youtube_url

        # 2. Ensure source video is downloaded
        if not src_path or not os.path.isfile(src_path):
            src_path = os.path.join(run_dir, video_name)
            log.info("Downloading source video from Google Drive (ID: %s)...", drive_file_id)
            drive_utils.download_file(drive_service, drive_file_id, src_path)
            cleanup_src = True

        # 3. Extract the exact clip segment
        slice_path = os.path.join(run_dir, f"{base_name}_clip{clip_index}_raw.mp4")
        video_process.extract_clip_segment(src_path, start_time, end_time, slice_path)

        # 4. Transcribe this specific clip segment for frame-accurate word timestamps
        clip_audio_path = os.path.join(run_dir, "clip_audio.mp3")
        transcribe.extract_audio(slice_path, clip_audio_path)
        trans_res = transcribe.transcribe_audio(clip_audio_path)
        words = trans_res["words"]
        detected_lang = trans_res["language"]
        clip_transcript = trans_res["text"]

        # 5. Auto-trim dead air (>0.5s) using Whisper word timestamps
        tightened_path = os.path.join(run_dir, "tightened.mp4")
        active_video, shifted_words = video_process.trim_silences_from_words(
            slice_path, words, tightened_path
        )

        # 6. Generate metadata & determine punchline hook grounded in this clip's transcript
        meta = metadata_ai.generate_shorts_metadata(
            f"{base_name} Part {clip_index}",
            transcript=clip_transcript,
        )
        title = meta["title"]
        variants = meta.get("title_variants", [title, title, title])
        description = meta["description"]
        tags = meta["tags"]

        # Hook Upgrade #1: Ensure top hook is a dedicated, validated short-form hook grounded in clip transcript
        candidate_punchline = clip.get("punchline") or meta.get("punchline") or clip.get("hook_summary", "")
        cand_dict = {"hook": candidate_punchline, "hook_type": "curiosity"}
        is_valid, _ = hook_generator.validate_hook(cand_dict, transcript=clip_transcript, filename=video_name)
        if is_valid:
            punchline = cand_dict["hook"]
        else:
            hook_res = hook_generator.generate_short_hook(
                transcript=clip_transcript,
                hook_summary=clip.get("hook_summary", ""),
                filename=video_name,
                detected_lang=detected_lang,
            )
            punchline = hook_res["selected_hook"]

        # 7. Generate animated, pop/karaoke word-level ASS captions with top punchline hook
        clip_duration = video_process.get_duration_seconds(active_video)
        ass_path = os.path.join(run_dir, "captions.ass")
        transcribe.generate_ass_captions(
            shifted_words,
            ass_path,
            punchline=punchline,
            total_duration=clip_duration,
        )

        # 8. Render final vertical Short with even-pixel scaling, loudnorm, top punchline, and burned captions
        out_path = os.path.join(run_dir, f"{base_name}_clip{clip_index}_short.mp4")
        video_process.process_video(active_video, out_path, ass_path=ass_path)

        # Secondary check before uploading
        existing_url = youtube_upload.find_existing_short(clip_identifier=clip_identifier, title=title)
        if existing_url:
            log.info("Found existing upload for '%s' prior to upload call: %s", title, existing_url)
            youtube_url = existing_url
        else:
            # 9. Upload to YouTube Shorts with tracking clip_identifier
            youtube_url = youtube_upload.upload_short(
                out_path, title, description, tags=tags, clip_identifier=clip_identifier
            )

        # 10. Mark clip as done in durable queue and log run
        sheet_log.update_clip_status(row_number, "done", youtube_url=youtube_url)
        sheet_log.log_run(
            video_name, "SUCCESS", detected_lang=detected_lang, youtube_url=youtube_url,
            title_1=variants[0], title_2=variants[1], title_3=variants[2],
        )
        notify.send(
            f"✅ Uploaded Short ({video_name} Clip #{clip_index}):\n{title}\n💬 Hook: {punchline}\n{youtube_url}"
        )
        report = run_report.get_current_report()
        if report:
            report.update_stage("metadata_generation", "✅ Completed")
            report.update_stage("silence_processing", "✅ Completed")
            report.update_stage("rendering", "✅ Rendered (1080x1920)")
            report.update_stage("youtube_upload", "✅ Uploaded")
            report.update_stage("sheets_update", "✅ Synced")
            report.add_or_update_clip(
                clip_index,
                hook=punchline,
                hook_status="Validated",
                metadata_status="AI Generated",
                rendering_status="Rendered (1080x1920)",
                upload_status="Uploaded",
                youtube_url=youtube_url,
            )
        return True, youtube_url

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
        report = run_report.get_current_report()
        if report:
            report.update_stage("youtube_upload", "⚠️ Quota Exceeded")
            report.add_or_update_clip(clip_index, upload_status="Quota Limit", error=str(qe))
            report.add_error("youtube_upload", "QuotaExceeded", str(qe), affected=f"{video_name} Clip #{clip_index}")
        return False, None

    except Exception as e:
        log.exception("Failed processing clip #%s for %s", clip_index, video_name)
        status_to_write = "done" if youtube_url else "failed"
        sheet_log.update_clip_status(row_number, status_to_write, youtube_url=youtube_url or "", error=str(e))
        sheet_log.log_run(video_name, f"{status_to_write.upper()} (Clip #{clip_index})", error=str(e), youtube_url=youtube_url or "")
        notify.send(f"❌ Failed processing Clip #{clip_index} of {video_name}:\nError: {e}")
        report = run_report.get_current_report()
        if report:
            report.update_stage("rendering" if not youtube_url else "youtube_upload", "❌ Failed")
            report.add_or_update_clip(clip_index, upload_status="Failed", error=str(e))
            report.add_error("clip_processing", type(e).__name__, str(e), affected=f"{video_name} Clip #{clip_index}")
        return (True, youtube_url) if youtube_url else (False, None)

    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
        if cleanup_src and src_path and os.path.exists(src_path):
            try:
                os.remove(src_path)
            except Exception:
                pass


def process_next_pending_clip(
    drive_service,
    row_number: int = None,
    clip: dict = None,
    drive_file_id: str = None,
    video_name: str = None,
    local_src_path: str = None,
) -> dict:
    """
    STRICT ONE-CLIP PRODUCTION EXECUTION:
    Selects, claims, and processes exactly ONE pending clip from the persistent queue:
    1. If clip/row_number not provided, fetches the deterministic top pending clip for the active source.
    2. Atomically/defensively claims the clip by transitioning state to 'processing'.
    3. Executes process_queue_clip (extract, silence trim, hook gen, captions, 9:16 render, YouTube upload).
    4. Updates final state in Google Sheets ('done' on success, 'failed' or 'retry_after_quota_reset' on error).
    5. Checks completion of source video: moves to Processed ONLY if 100% of discovered clips are done.
    6. Updates execution run report and returns single-clip stats.
    STRUCTURALLY INCAPABLE of processing or uploading multiple clips in one execution.
    """
    if row_number is None or clip is None:
        pending = sheet_log.get_next_pending_clip(target_drive_file_id=drive_file_id)
        if not pending:
            log.info("No eligible pending clips found to process.")
            if drive_file_id and video_name:
                check_and_move_if_completed(drive_service, drive_file_id, video_name)
            return {"total_queued": 0, "uploaded": 0, "failed": 0, "urls": [], "errors": []}
        row_number, clip = pending

    drive_file_id = clip["drive_file_id"]
    video_name = clip["source_video_name"]
    clip_idx = clip["clip_index"]

    # Defensive claim: transition state from 'pending' to 'processing'
    sheet_log.claim_clip(row_number)
    log.info("Claimed clip #%s for video '%s' (row %d) -> state: processing", clip_idx, video_name, row_number)

    stats = {
        "total_queued": 1,
        "uploaded": 0,
        "failed": 0,
        "urls": [],
        "errors": [],
    }

    try:
        success, yt_url = process_queue_clip(drive_service, row_number, clip, local_src_path=local_src_path)
        if success:
            stats["uploaded"] = 1
            if yt_url:
                stats["urls"].append(yt_url)
        else:
            stats["failed"] = 1
            stats["errors"].append(f"Clip #{clip_idx} failed")
    except Exception as e:
        stats["failed"] = 1
        stats["errors"].append(str(e))
        log.exception("Unexpected exception in process_next_pending_clip for clip #%s: %s", clip_idx, e)

    # Check completion: Moves source video to Processed ONLY if 100% of discovered clips are done
    check_and_move_if_completed(drive_service, drive_file_id, video_name)

    report = run_report.get_current_report()
    if report:
        report.summary_total_clips = 1
        report.summary_uploaded = stats["uploaded"]
        report.summary_failed = stats["failed"]
        report.summary_completed = stats["uploaded"]
        counts = sheet_log.get_video_clip_counts(drive_file_id, service=None)
        report.summary_pending = counts.get("pending", 0)

    return stats


def process_all_clips_for_video(drive_service, drive_file_id: str, video_name: str, local_src_path: str = None) -> dict:
    """
    Processes ALL pending clips belonging to the specified source video sequentially.
    Reuses the downloaded source video locally across all clips.
    Halts on unrecoverable failure (e.g. quota limit reached or render error), leaving
    subsequent clips pending/retryable without touching other videos.
    Only moves the source video to Processed if 100% of its clips are done.
    PRESERVED ONLY FOR BACKWARD COMPATIBILITY / HISTORICAL UNIT TESTS.
    UNREACHABLE FROM PRODUCTION ORCHESTRATION.
    """
    pending_clips = sheet_log.get_pending_clips_for_video(drive_file_id, service=None)
    log.info("Found %d pending clip(s) for video '%s' (ID: %s)", len(pending_clips), video_name, drive_file_id)

    stats = {
        "total_queued": len(pending_clips),
        "uploaded": 0,
        "failed": 0,
        "urls": [],
        "errors": [],
    }

    if not pending_clips:
        check_and_move_if_completed(drive_service, drive_file_id, video_name)
        return stats

    run_dir = None
    src_path = local_src_path
    cleanup_src = False

    # If source video not provided locally, download once for all clips
    if not src_path or not os.path.isfile(src_path):
        run_dir = os.path.join(WORKDIR, f"batch_{drive_file_id}")
        os.makedirs(run_dir, exist_ok=True)
        src_path = os.path.join(run_dir, video_name)
        log.info("Downloading source video once for batch clip processing (ID: %s)...", drive_file_id)
        drive_utils.download_file(drive_service, drive_file_id, src_path)
        cleanup_src = True

    try:
        for row_number, clip in pending_clips:
            clip_idx = clip["clip_index"]
            log.info("Processing clip %d of %d for '%s'...", clip_idx, len(pending_clips), video_name)
            success, yt_url = process_queue_clip(drive_service, row_number, clip, local_src_path=src_path)
            if success:
                stats["uploaded"] += 1
                if yt_url:
                    stats["urls"].append(yt_url)
            else:
                stats["failed"] += 1
                stats["errors"].append(f"Clip #{clip_idx} failed")
                log.warning(
                    "Clip #%s failed for '%s'. Halting batch processing of remaining clips for this run.",
                    clip_idx, video_name,
                )
                break
    finally:
        if cleanup_src and run_dir:
            shutil.rmtree(run_dir, ignore_errors=True)

    # Check completion: Moves source video to Processed ONLY if all clips are done
    check_and_move_if_completed(drive_service, drive_file_id, video_name)
    report = run_report.get_current_report()
    if report:
        report.summary_total_clips = stats.get("total_queued", len(pending_clips))
        report.summary_uploaded = stats.get("uploaded", 0)
        report.summary_failed = stats.get("failed", 0)
        report.summary_completed = stats.get("uploaded", 0)
        report.summary_pending = max(0, report.summary_total_clips - stats.get("uploaded", 0) - stats.get("failed", 0))
    return stats


def discover_and_enqueue_video(drive_service, file_info: dict) -> dict:
    """
    Downloads a new long-form video, transcribes full audio, detects multiple
    high-engagement talk segment clips (dynamic yield, no 5-clip cap), persists all
    valid clips to the durable Google Sheets queue, and processes ONLY ONE clip in this initial run.
    Remaining clips persist in Google Sheets as pending for subsequent scheduled triggers.
    """
    file_id = file_info["id"]
    name = file_info["name"]

    run_dir = os.path.join(WORKDIR, f"discovery_{file_id}")
    os.makedirs(run_dir, exist_ok=True)
    src_path = os.path.join(run_dir, name)
    full_audio_path = os.path.join(run_dir, "full_audio.mp3")

    report = run_report.get_current_report()
    if report:
        report.set_source(name, file_id=file_id)

    try:
        log.info("Discovered new source video in Incoming: %s", name)
        log.info("Downloading full video for multi-clip analysis...")
        drive_utils.download_file(drive_service, file_id, src_path)
        if report:
            report.update_stage("download", "✅ Success")

        total_duration = video_process.get_duration_seconds(src_path)
        log.info("Source video duration: %.1fs (%d min)", total_duration, int(total_duration / 60))
        if report:
            report.set_source(name, duration=total_duration)

        # Check whether the video contains an audio stream before transcription
        if not video_process.check_has_audio(src_path):
            error_msg = f"Source video '{name}' has no audio stream and cannot be transcribed."
            log.error(error_msg)
            if report:
                report.update_stage("transcription", "❌ No audio stream")
                report.add_error("audio_check", "ValueError", error_msg, affected=name)
                report.set_overall_status("FAILED 🔴")
            raise ValueError(error_msg)

        # 1. Extract full audio and transcribe
        log.info("Extracting full audio for multi-clip transcription...")
        transcribe.extract_audio(src_path, full_audio_path)
        trans_res = transcribe.transcribe_audio(full_audio_path)
        segments = trans_res.get("segments", [])
        if report:
            report.update_stage("transcription", "✅ Completed (Whisper)")
            report.set_discovery(transcription_status="✅ Completed (Whisper)")

        # 2. Run AI multi-clip detection (dynamic yield: max_clips=None)
        log.info("Detecting viral, high-retention talk segment clips from full transcript...")
        clips = clip_detection.detect_clips_from_transcript(segments, total_duration, max_clips=None)

        if not clips:
            # ZERO VALID CLIPS TERMINAL HANDLING (Correction #6):
            log.info("0 valid clips met quality/standalone criteria for '%s'. Marking terminal state.", name)
            sheet_log.enqueue_zero_clips(file_id, name, reason="no_valid_clips", candidate_count=len(segments))
            if report:
                report.update_stage("ai_analysis", "✅ Completed")
                report.update_stage("quality_scoring", "✅ Completed (0 qualified)")
                report.update_stage("standalone_validation", "✅ Completed")
                report.set_discovery(
                    candidate_count=len(segments),
                    qualified_count=0,
                    filtering_results="0 clip(s) met quality/standalone criteria",
                )
                report.set_overall_status("NOTHING TO PROCESS 🔵")
                report.summary_source_status = "Moved to Processed (no valid clips)"
            notify.send(
                f"ℹ️ Video analyzed successfully for '{name}'. No publishable standalone clips passed quality requirements."
            )
            check_and_move_if_completed(drive_service, file_id, name)
            return {"total_queued": 0, "uploaded": 0, "failed": 0, "urls": [], "errors": []}

        if report:
            report.update_stage("ai_analysis", "✅ Completed")
            report.update_stage("quality_scoring", "✅ Completed")
            report.update_stage("standalone_validation", "✅ Completed")
            report.set_discovery(
                candidate_count=len(segments),
                qualified_count=len(clips),
                filtering_results=f"{len(clips)} clip(s) met quality threshold",
            )
            for c in clips:
                report.add_or_update_clip(
                    c.get("clip_index", 1),
                    quality_score=c.get("quality_score"),
                    hook=c.get("punchline") or c.get("hook_summary", ""),
                    upload_status="Queued",
                )

        # 3. Persist ALL detected clips into durable Google Sheets queue
        sheet_log.enqueue_clips(file_id, name, clips)
        if report:
            report.update_stage("sheets_update", "✅ Enqueued")
        notify.send(
            f"🎬 Discovered {len(clips)} clip(s) for '{name}' (Duration: {int(total_duration)}s). Enqueued in queue."
        )

        # 4. STRICT ONE-CLIP PRODUCTION: Process ONLY ONE clip in this initial run!
        # Reuses the downloaded source video locally. Remaining clips stay pending for future triggers.
        stats = process_next_pending_clip(drive_service, drive_file_id=file_id, video_name=name, local_src_path=src_path)
        return stats

    except Exception as e:
        log.exception("Failed during clip discovery and enqueuing for %s", name)
        sheet_log.log_run(name, "FAILED (Discovery)", error=str(e))
        notify.send(f"❌ Failed analyzing clips for {name}:\nError: {e}")
        if report:
            report.add_error("discovery", type(e).__name__, str(e), affected=name)
            report.set_overall_status("FAILED 🔴")
        if DRIVE_FAILED_FOLDER_ID:
            try:
                drive_utils.move_file(drive_service, file_id, DRIVE_INCOMING_FOLDER_ID, DRIVE_FAILED_FOLDER_ID)
            except Exception:
                log.warning("Could not move failed video to Failed folder.")
        return {"total_queued": 0, "uploaded": 0, "failed": 1, "urls": [], "errors": [str(e)]}

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


def _get_time_strings():
    """Returns (utc_str, ist_str) formatted strings for logging."""
    now_utc = datetime.datetime.utcnow()
    # IST is strictly UTC+05:30 (Asia/Kolkata) with no daylight saving
    now_ist = now_utc + datetime.timedelta(hours=5, minutes=30)
    utc_str = now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
    ist_str = now_ist.strftime("%Y-%m-%d %I:%M:%S %p IST (Asia/Kolkata)")
    return utc_str, ist_str


def print_run_header(trigger: str = None, incoming_folder: str = None):
    """Prints clear observability start banner."""
    utc_str, ist_str = _get_time_strings()
    trig_str = trigger or os.environ.get("GITHUB_TRIGGER", "manual / local")
    folder_str = incoming_folder or DRIVE_INCOMING_FOLDER_ID or "Not configured"

    print("\n========================================")
    print("SHORTS AUTOMATION RUN")
    print("========================================")
    print(f"Current UTC time:         {utc_str}")
    print(f"Current IST time:         {ist_str}")
    print(f"Scheduled/manual trigger: {trig_str}")
    print(f"Incoming folder ID:       {folder_str}")
    print("========================================\n", flush=True)


def print_run_summary(
    videos_discovered: int = 0,
    videos_processed: int = 0,
    clips_generated: int = 0,
    clips_uploaded: int = 0,
    clips_failed: int = 0,
    clips_skipped: int = 0,
    youtube_urls: list = None,
    errors: list = None,
):
    """Prints clear observability summary banner."""
    urls = [u for u in (youtube_urls or []) if u]
    errs = [e for e in (errors or []) if e]

    print("\n========================================")
    print("RUN SUMMARY")
    print("========================================")
    print(f"Videos discovered: {videos_discovered}")
    print(f"Videos processed:  {videos_processed}")
    print(f"Clips generated:   {clips_generated}")
    print(f"Clips uploaded:    {clips_uploaded}")
    print(f"Clips failed:      {clips_failed}")
    print(f"Clips skipped:     {clips_skipped}")
    print(f"YouTube URLs:      {', '.join(urls) if urls else 'None'}")
    print(f"Errors:            {', '.join(errs) if errs else 'None'}")
    print("========================================\n", flush=True)


def main():
    os.makedirs(WORKDIR, exist_ok=True)
    print_run_header()

    report = run_report.PipelineRunReport()
    run_report.set_current_report(report)

    try:
        if DRY_RUN_LOG_ONLY or "--dry-run" in sys.argv:
            report.set_overall_status("NOTHING TO PROCESS 🔵")
            run_dry_run_inspection()
            print_run_summary(videos_discovered=0, videos_processed=0, clips_generated=0, clips_uploaded=0, clips_failed=0, clips_skipped=0)
            return

        drive_service = drive_utils.get_drive_service()
        sheet_log.ensure_clip_queue_sheet()

        # Step 0: Auto-reset quota-exhausted clips older than 20 hours and stale processing workers older than 60 minutes
        sheet_log.reset_expired_quota_clips(min_age_hours=20.0)
        sheet_log.reset_stale_processing_clips(max_age_minutes=60.0)

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
                "Found active video with pending clips: '%s' (ID: %s). Processing ONE pending clip...",
                video_name, drive_file_id,
            )
            stats = process_next_pending_clip(
                drive_service,
                row_number=row_number,
                clip=clip,
                drive_file_id=drive_file_id,
                video_name=video_name,
            )
            print_run_summary(
                videos_discovered=0,
                videos_processed=1,
                clips_generated=stats.get("total_queued", 0),
                clips_uploaded=stats.get("uploaded", 0),
                clips_failed=stats.get("failed", 0),
                clips_skipped=0,
                youtube_urls=stats.get("urls", []),
                errors=stats.get("errors", []),
            )
            return

        # Step B: Check if any clip is waiting for quota reset
        if sheet_log.has_active_video_in_queue():
            log.info(
                "Queue has active clip(s) waiting for quota reset ('retry_after_quota_reset'). "
                "Skipping Incoming folder until current video is completely processed."
            )
            report.set_overall_status("NOTHING TO PROCESS 🔵")
            report.summary_source_status = "Waiting for quota reset"
            print_run_summary(
                videos_discovered=0,
                videos_processed=0,
                clips_generated=0,
                clips_uploaded=0,
                clips_failed=0,
                clips_skipped=1,
                errors=["queue waiting for quota reset"],
            )
            return

        # Step C: Queue is empty -> Check Drive "Incoming" for new videos
        log.info("Queue is completely empty. Checking Drive 'Incoming' folder for new videos...")
        incoming_videos = drive_utils.list_new_videos(drive_service, require_ready=True)
        if not incoming_videos:
            log.info("No pending clips in queue and no new videos in Incoming folder. Run finished.")
            report.set_overall_status("NOTHING TO PROCESS 🔵")
            report.summary_source_status = "No new videos in Incoming"
            print_run_summary(
                videos_discovered=0,
                videos_processed=0,
                clips_generated=0,
                clips_uploaded=0,
                clips_failed=0,
                clips_skipped=0,
            )
            return

        # Idempotency check: Filter incoming videos against all previously enqueued Drive file IDs
        try:
            already_enqueued_ids = sheet_log.get_enqueued_video_ids()
        except Exception as e:
            log.error(
                "Could not fetch enqueued video IDs from clip_queue: %s. "
                "Aborting discovery to prevent duplicate video processing.",
                e,
            )
            if report:
                report.add_error("sheets_check", type(e).__name__, str(e))
                report.set_overall_status("FAILED 🔴")
            sys.exit(1)
        unprocessed_videos = []
        for vid in incoming_videos:
            v_id = vid["id"]
            v_name = vid["name"]
            if v_id in already_enqueued_ids:
                log.info(
                    "Incoming video '%s' (ID: %s) is ALREADY PROCESSED/ENQUEUED in clip_queue. "
                    "Preserving in Incoming without re-discovery.",
                    v_name, v_id,
                )
            else:
                unprocessed_videos.append(vid)

        if not unprocessed_videos:
            log.info(
                "All %d video(s) in Incoming have already been processed/enqueued. "
                "Zero duplicate work. Clean exit.",
                len(incoming_videos),
            )
            report.set_overall_status("NOTHING TO PROCESS 🔵")
            report.summary_source_status = "All Incoming videos already enqueued"
            print_run_summary(
                videos_discovered=len(incoming_videos),
                videos_processed=0,
                clips_generated=0,
                clips_uploaded=0,
                clips_failed=0,
                clips_skipped=len(incoming_videos),
            )
            return

        # Pick ONLY the single oldest unprocessed video (FIFO sequential ordering)
        target_video = unprocessed_videos[0]
        log.info(
            "Found %d new unprocessed video(s) in Incoming. Starting with the single oldest: '%s' (ID: %s)",
            len(unprocessed_videos), target_video["name"], target_video["id"],
        )
        stats = discover_and_enqueue_video(drive_service, target_video)
        print_run_summary(
            videos_discovered=len(unprocessed_videos),
            videos_processed=1,
            clips_generated=stats.get("total_queued", 0),
            clips_uploaded=stats.get("uploaded", 0),
            clips_failed=stats.get("failed", 0),
            clips_skipped=len(unprocessed_videos) - 1,
            youtube_urls=stats.get("urls", []),
            errors=stats.get("errors", []),
        )
        if isinstance(stats, dict) and stats.get("failed", 0) > 0 and stats.get("total_queued", 0) == 0:
            sys.exit(1)

    except SystemExit:
        raise
    except Exception as exc:
        report.add_error("pipeline", type(exc).__name__, str(exc))
        report.set_overall_status("FAILED 🔴")
        raise
    finally:
        try:
            report.send_report()
        except Exception as te:
            log.warning("Could not dispatch Telegram run report: %s", te)
        run_report.set_current_report(None)


if __name__ == "__main__":
    main()

