"""
Centralized Run Report and Observability State for YouTube Shorts Automation.
Tracks end-to-end execution state across all pipeline stages and sends
a single, unified, structured summary to Telegram at the end of every run.
"""
import datetime
import logging
import os
import re
import time
from typing import Dict, List, Optional

import notify
from config import (
    DRIVE_INCOMING_FOLDER_ID,
    DRIVE_PROCESSED_FOLDER_ID,
    DRIVE_FAILED_FOLDER_ID,
)

log = logging.getLogger("run_report")

_CURRENT_REPORT: Optional["PipelineRunReport"] = None


def get_current_report() -> Optional["PipelineRunReport"]:
    """Returns the active run report for this execution if initialized."""
    return _CURRENT_REPORT


def set_current_report(report: Optional["PipelineRunReport"]):
    """Sets or clears the active run report singleton."""
    global _CURRENT_REPORT
    _CURRENT_REPORT = report


def sanitize_secrets(text: str) -> str:
    """Redacts API keys, OAuth tokens, and bearer credentials from messages."""
    if not text or not isinstance(text, str):
        return ""
    # Redact common key prefixes and formats
    text = re.sub(r"gsk_[a-zA-Z0-9_-]+", "[REDACTED_GROQ_KEY]", text)
    text = re.sub(r"sk-[a-zA-Z0-9_-]+", "[REDACTED_OPENAI_KEY]", text)
    text = re.sub(r"ya29\.[a-zA-Z0-9_-]+", "[REDACTED_OAUTH_TOKEN]", text)
    text = re.sub(r"Bearer\s+[a-zA-Z0-9_\.-]+", "Bearer [REDACTED_TOKEN]", text, flags=re.IGNORECASE)
    text = re.sub(r"bot\d+:[a-zA-Z0-9_-]+", "bot[REDACTED_BOT_TOKEN]", text)
    return text


def clean_concise_error(msg: str, max_len: int = 220) -> str:
    """Extracts a concise single-line error description without massive tracebacks."""
    if not msg:
        return "Unknown error"
    clean = sanitize_secrets(str(msg)).strip()
    # Take the last relevant line of exception if multiline traceback
    lines = [line.strip() for line in clean.splitlines() if line.strip()]
    summary = lines[-1] if lines else clean
    if len(summary) > max_len:
        summary = summary[:max_len - 3] + "..."
    return summary


class PipelineRunReport:
    """
    Centralized execution state object for the entire YouTube Shorts pipeline.
    Accumulates status across discovery, clipping, rendering, uploading, and drive movement.
    """

    def __init__(self, trigger: str = None):
        self.start_time = datetime.datetime.utcnow()
        self.end_time: Optional[datetime.datetime] = None
        self.trigger = trigger or os.environ.get("GITHUB_TRIGGER", "manual / local")

        # 1. Overall Status: SUCCESS 🟢 | PARTIAL FAILURE ⚠️ | FAILED 🔴 | NOTHING TO PROCESS 🔵
        self.overall_status = "NOTHING TO PROCESS 🔵"
        self._manual_status: Optional[str] = None

        # 2. Source Information
        self.source_filename = "N/A"
        self.source_file_id = "N/A"
        self.source_duration: Optional[float] = None
        self.drive_source_folder = DRIVE_INCOMING_FOLDER_ID or "Incoming"
        self.drive_final_destination = "Preserved in Incoming"

        # 3. Discovery Information
        self.transcription_status = "⏭️ Skipped / N/A"
        self.candidate_clips_count = 0
        self.qualified_clips_count = 0
        self.filtering_results = "N/A"

        # 4. Per-Clip Status: list of dicts
        self.clips: List[Dict] = []

        # 5. Overall Stage Status
        self.stages = {
            "download": "⏭️ Skipped",
            "transcription": "⏭️ Skipped",
            "ai_analysis": "⏭️ Skipped",
            "quality_scoring": "⏭️ Skipped",
            "standalone_validation": "⏭️ Skipped",
            "metadata_generation": "⏭️ Skipped",
            "silence_processing": "⏭️ Skipped",
            "rendering": "⏭️ Skipped",
            "youtube_upload": "⏭️ Skipped",
            "sheets_update": "⏭️ Skipped",
            "drive_movement": "⏭️ Skipped",
        }

        # 6. Errors: list of {stage, error_type, message, affected}
        self.errors: List[Dict] = []

        # 7. Final Summary
        self.summary_total_clips = 0
        self.summary_completed = 0
        self.summary_uploaded = 0
        self.summary_failed = 0
        self.summary_pending = 0
        self.summary_source_status = "Preserved in Incoming"

    def finish(self):
        """Marks the end of pipeline execution and records total runtime."""
        if not self.end_time:
            self.end_time = datetime.datetime.utcnow()

    def set_source(
        self,
        filename: str,
        file_id: str = None,
        duration: float = None,
        destination: str = None,
    ):
        """Sets source video information."""
        if filename:
            self.source_filename = filename
        if file_id:
            self.source_file_id = file_id
        if duration is not None:
            self.source_duration = duration
        if destination:
            self.drive_final_destination = destination

    def set_discovery(
        self,
        transcription_status: str = None,
        candidate_count: int = None,
        qualified_count: int = None,
        filtering_results: str = None,
    ):
        """Sets discovery and candidate filtering results."""
        if transcription_status:
            self.transcription_status = transcription_status
        if candidate_count is not None:
            self.candidate_clips_count = candidate_count
        if qualified_count is not None:
            self.qualified_clips_count = qualified_count
        if filtering_results:
            self.filtering_results = filtering_results

    def update_stage(self, stage: str, status: str):
        """Updates the status of an overall pipeline stage."""
        if stage in self.stages:
            self.stages[stage] = status
            if stage == "transcription":
                self.transcription_status = status

    def add_or_update_clip(self, clip_index: int, **kwargs):
        """Adds or updates status fields for a specific clip."""
        for clip in self.clips:
            if clip.get("clip_index") == clip_index:
                clip.update(kwargs)
                return
        # If clip not found, add new entry
        new_clip = {
            "clip_index": clip_index,
            "quality_score": kwargs.get("quality_score"),
            "hook": kwargs.get("hook", ""),
            "hook_status": kwargs.get("hook_status", "N/A"),
            "metadata_status": kwargs.get("metadata_status", "N/A"),
            "rendering_status": kwargs.get("rendering_status", "Pending"),
            "upload_status": kwargs.get("upload_status", "Pending"),
            "youtube_url": kwargs.get("youtube_url"),
            "error": kwargs.get("error"),
        }
        self.clips.append(new_clip)

    def add_error(self, stage: str, error_type: str, message: str, affected: str = None):
        """Records a structured error sanitized of all secrets and formatted concisely."""
        clean_msg = clean_concise_error(message)
        self.errors.append({
            "stage": stage,
            "error_type": error_type,
            "message": clean_msg,
            "affected": affected or self.source_filename,
        })
        log.warning("[REPORT_ERROR] Stage: %s | Type: %s | Msg: %s", stage, error_type, clean_msg)

    def set_overall_status(self, status: str):
        """Explicitly sets the overall status override."""
        self._manual_status = status
        self.overall_status = status

    def determine_overall_status(self):
        """Calculates the overall status based on clips, errors, and discovery state."""
        if self._manual_status:
            # If manual status is set, only override if errors were recorded that make it FAILED
            if self.errors and "FAILED" not in self._manual_status and "PARTIAL" not in self._manual_status:
                self.overall_status = "FAILED 🔴"
            else:
                self.overall_status = self._manual_status
            return

        # 1. Fatal discovery or pipeline errors with 0 uploads
        if any(e.get("stage") in ("discovery", "audio_check", "download", "pipeline") for e in self.errors) and self.summary_uploaded == 0:
            self.overall_status = "FAILED 🔴"
            return

        # 2. Clips were processed
        total = self.summary_total_clips or len(self.clips)
        uploaded = self.summary_uploaded
        failed = self.summary_failed

        if total > 0:
            if uploaded > 0 and failed == 0:
                self.overall_status = "SUCCESS 🟢"
            elif uploaded > 0 and failed > 0:
                self.overall_status = "PARTIAL FAILURE ⚠️"
            elif uploaded == 0 and failed > 0:
                self.overall_status = "FAILED 🔴"
            else:
                self.overall_status = "PARTIAL FAILURE ⚠️"
        elif self.errors:
            self.overall_status = "FAILED 🔴"
        else:
            self.overall_status = "NOTHING TO PROCESS 🔵"

    def get_github_run_url(self) -> str:
        """Constructs GitHub Actions run URL if running inside GitHub Actions CI."""
        server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
        repo = os.environ.get("GITHUB_REPOSITORY")
        run_id = os.environ.get("GITHUB_RUN_ID")
        if repo and run_id:
            return f"{server}/{repo}/actions/runs/{run_id}"
        return "Local / Manual Execution"

    def build_telegram_message(self) -> str:
        """Builds the comprehensive, structured, emoji-rich Telegram run report."""
        self.finish()
        self.determine_overall_status()

        # Format timestamps
        utc_str = self.start_time.strftime("%Y-%m-%d %H:%M:%S UTC")
        ist_time = self.start_time + datetime.timedelta(hours=5, minutes=30)
        ist_str = ist_time.strftime("%Y-%m-%d %I:%M:%S %p IST")

        # Total runtime
        runtime_sec = max(0.0, (self.end_time - self.start_time).total_seconds())
        if runtime_sec >= 60:
            m, s = divmod(int(runtime_sec), 60)
            runtime_str = f"{m}m {s}s"
        else:
            runtime_str = f"{runtime_sec:.1f}s"

        # Duration string
        dur_str = f"{self.source_duration:.1f}s ({int(self.source_duration // 60)}m {int(self.source_duration % 60)}s)" if self.source_duration else "N/A"

        gh_url = self.get_github_run_url()

        # Build message sections
        lines = [
            "📊 *YOUTUBE SHORTS RUN REPORT*",
            f"Status: *{self.overall_status}*",
            "",
            "⏱️ *Run Information:*",
            f"• Time: `{utc_str}`",
            f"• IST: `{ist_str}`",
            f"• Runtime: `{runtime_str}`",
            f"• Trigger: `{self.trigger}`",
            f"• Workflow: {gh_url}",
            "",
            "📁 *Source Information:*",
            f"• Video: `{self.source_filename}`",
            f"• Duration: `{dur_str}`",
            f"• Source Folder: `{self.drive_source_folder}`",
            f"• Destination: `{self.drive_final_destination}`",
            "",
            "🔍 *Discovery & Filtering:*",
            f"• Transcription: {self.transcription_status}",
            f"• Candidates: `{self.candidate_clips_count}` | Qualified: `{self.qualified_clips_count}`",
            f"• Filtering: {self.filtering_results}",
        ]

        # Per-clip details
        if self.clips:
            lines.append("")
            lines.append("🎬 *Per-Clip Status:*")
            for c in self.clips:
                idx = c.get("clip_index", "?")
                score = f"Score: {c['quality_score']:.1f}" if c.get("quality_score") is not None else ""
                up_status = c.get("upload_status", "Pending")
                icon = "✅" if "uploaded" in up_status.lower() or "done" in up_status.lower() else ("❌" if "fail" in up_status.lower() else "⏳")
                score_part = f" ({score})" if score else ""
                lines.append(f"• Clip #{idx}{score_part}: {icon} {up_status}")

                if c.get("hook"):
                    lines.append(f"  💬 Hook: {c['hook'][:60]}")
                if c.get("youtube_url"):
                    lines.append(f"  🔗 {c['youtube_url']}")
                if c.get("error"):
                    lines.append(f"  ⚠️ Error: {clean_concise_error(c['error'], max_len=100)}")

        # Stages
        lines.append("")
        lines.append("⚙️ *Overall Stages:*")
        s = self.stages
        lines.append(f"• Download: {s['download']} | Transcribe: {s['transcription']} | AI Analysis: {s['ai_analysis']}")
        lines.append(f"• Metadata: {s['metadata_generation']} | Silence: {s['silence_processing']} | Render: {s['rendering']}")
        lines.append(f"• Upload: {s['youtube_upload']} | Sheets: {s['sheets_update']} | Drive: {s['drive_movement']}")

        # Errors section
        if self.errors:
            lines.append("")
            lines.append(f"⚠️ *Error Report ({len(self.errors)}):*")
            for err in self.errors[:5]:  # Limit to first 5 concise errors to prevent message bloat
                lines.append(f"• `[{err['stage']}]` {err['error_type']}: {err['message']} (affected: {err['affected']})")
            if len(self.errors) > 5:
                lines.append(f"• ... and {len(self.errors) - 5} more error(s)")

        # Summary section
        lines.append("")
        lines.append("📈 *Final Summary:*")
        lines.append(
            f"• Clips Total: `{self.summary_total_clips}` | Uploaded: `{self.summary_uploaded}` | Failed: `{self.summary_failed}` | Pending: `{self.summary_pending}`"
        )
        lines.append(f"• Source Status: *{self.summary_source_status}*")

        full_message = "\n".join(lines)

        # Enforce Telegram 4096 char limit
        if len(full_message) > 4000:
            full_message = full_message[:3950] + "\n\n... [Report truncated to fit Telegram limit]"

        return full_message

    def send_report(self):
        """Sends the compiled report to Telegram via notify.send(), safely catching any notification errors."""
        try:
            msg = self.build_telegram_message()
            log.info("Sending centralized Telegram run report (%d chars)...", len(msg))
            notify.send(msg)
            log.info("Centralized Telegram run report dispatched successfully.")
        except Exception as e:
            log.warning("Could not dispatch Telegram run report (safe failover): %s", e)
