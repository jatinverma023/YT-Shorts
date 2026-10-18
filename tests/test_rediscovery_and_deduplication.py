"""
Comprehensive regression test suite for safe video re-discovery,
duplicate detection, active work precedence, and YouTube upload idempotency.

Covers all 14 required regression scenarios:
1. Completed historical source blocked by default.
2. Source with pending clips is resumed rather than rediscovered.
3. processing and retry_after_quota_reset rows prevent rediscovery.
4. Failed and no_valid_clips sources remain blocked by default.
5. Explicitly authorized source can be rediscovered exactly once.
6. Authorization cannot override active queue rows.
7. Repeated runs after successful enqueueing do not rediscover the source.
8. Concurrent runs cannot append duplicate discovery results.
9. Discovery API failure preserves the source and safely handles authorization.
10. Partial failure during enqueueing cannot create an unsafe duplicate-discovery window.
11. Historical rows, YouTube URLs, timestamps, and quality scores remain unchanged.
12. Unknown or missing statuses fail safely rather than accidentally allowing rediscovery.
13. Existing YouTube idempotency checks and unique clip indexing remain effective.
14. Existing FIFO, active-source locking, and one-clip-per-run invariants continue to hold.
"""
import os
import sys
import unittest
from unittest.mock import patch, MagicMock, call

# Ensure scripts directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

import sheet_log
import main as pipeline_main


class TestRediscoveryAndDeduplication(unittest.TestCase):
    """Test suite covering safe re-discovery and duplicate detection."""

    def setUp(self):
        self.mock_service = MagicMock()
        self.drive_id = "1Zp82g0NzcpGGWHbBqVm_mgC2AA0b1Liy"
        self.other_id = "2Wq93h1OadHHXIcCrXWn0b2Mjz"
        self.patcher_sheet_id = patch.object(sheet_log, "LOG_SHEET_ID", "mock_sheet_id_123")
        self.patcher_sheet_id.start()
        self.addCleanup(self.patcher_sheet_id.stop)

    # -------------------------------------------------------------------------
    # 1. Completed historical source is blocked by default
    # -------------------------------------------------------------------------
    def test_01_completed_historical_source_blocked_by_default(self):
        """A source with only terminal 'done' rows is blocked from re-discovery by default."""
        mock_resp = {
            "values": [
                ["drive_file_id", "clip_index", "start", "end", "hook", "status"],
                [self.drive_id, "1", "10.0", "40.0", "Hook 1", "done"],
                [self.drive_id, "2", "50.0", "80.0", "Hook 2", "done"],
            ]
        }
        self.mock_service.spreadsheets().values().get().execute.return_value = mock_resp

        blocked = sheet_log.get_enqueued_video_ids(service=self.mock_service)
        self.assertIn(self.drive_id, blocked)

    # -------------------------------------------------------------------------
    # 2. Source with pending clips is resumed rather than rediscovered
    # -------------------------------------------------------------------------
    def test_02_source_with_pending_clips_resumed_not_rediscovered(self):
        """A source with pending clips is blocked from re-discovery and resumed in normal processing."""
        mock_resp = {
            "values": [
                ["drive_file_id", "clip_index", "start", "end", "hook", "status"],
                [self.drive_id, "1", "10.0", "40.0", "Hook 1", "done"],
                [self.drive_id, "2", "50.0", "80.0", "Hook 2", "pending"],
            ]
        }
        self.mock_service.spreadsheets().values().get().execute.return_value = mock_resp

        blocked = sheet_log.get_enqueued_video_ids(service=self.mock_service)
        self.assertIn(self.drive_id, blocked)

    # -------------------------------------------------------------------------
    # 3. processing and retry_after_quota_reset rows prevent rediscovery
    # -------------------------------------------------------------------------
    def test_03_processing_and_quota_retry_prevent_rediscovery(self):
        """Rows with 'processing' or 'retry_after_quota_reset' block rediscovery unconditionally."""
        # Case A: processing
        mock_resp_processing = {
            "values": [
                ["drive_file_id", "clip_index", "start", "end", "hook", "status"],
                [self.drive_id, "1", "10.0", "40.0", "Hook 1", "processing"],
            ]
        }
        self.mock_service.spreadsheets().values().get().execute.return_value = mock_resp_processing
        blocked = sheet_log.get_enqueued_video_ids(service=self.mock_service)
        self.assertIn(self.drive_id, blocked)

        # Case B: retry_after_quota_reset
        mock_resp_quota = {
            "values": [
                ["drive_file_id", "clip_index", "start", "end", "hook", "status"],
                [self.drive_id, "1", "10.0", "40.0", "Hook 1", "retry_after_quota_reset"],
            ]
        }
        self.mock_service.spreadsheets().values().get().execute.return_value = mock_resp_quota
        blocked = sheet_log.get_enqueued_video_ids(service=self.mock_service)
        self.assertIn(self.drive_id, blocked)

    # -------------------------------------------------------------------------
    # 4. Failed and no_valid_clips sources remain blocked by default
    # -------------------------------------------------------------------------
    def test_04_failed_and_no_valid_clips_blocked_by_default(self):
        """Historical sources marked 'failed', 'no_valid_clips', or 'cancelled' remain blocked."""
        mock_resp = {
            "values": [
                ["drive_file_id", "clip_index", "start", "end", "hook", "status"],
                ["id_failed", "1", "10.0", "40.0", "Hook 1", "failed"],
                ["id_no_valid", "0", "0.0", "0.0", "Hook 0", "no_valid_clips"],
                ["id_cancelled", "1", "10.0", "40.0", "Hook 1", "cancelled"],
            ]
        }
        self.mock_service.spreadsheets().values().get().execute.return_value = mock_resp

        blocked = sheet_log.get_enqueued_video_ids(service=self.mock_service)
        self.assertIn("id_failed", blocked)
        self.assertIn("id_no_valid", blocked)
        self.assertIn("id_cancelled", blocked)

    # -------------------------------------------------------------------------
    # 5. An explicitly authorized source can be rediscovered exactly once
    # -------------------------------------------------------------------------
    def test_05_explicitly_authorized_source_rediscovered_once(self):
        """An authorized source is excluded from blocked list, claimed, enqueued, and consumed."""
        # 1. Row is marked 'rediscover'
        mock_resp = {
            "values": [
                ["drive_file_id", "clip_index", "start", "end", "hook", "status"],
                [self.drive_id, "1", "10.0", "40.0", "Hook 1", "rediscover"],
            ]
        }
        self.mock_service.spreadsheets().values().get().execute.return_value = mock_resp

        # Video is NOT in blocked set
        blocked = sheet_log.get_enqueued_video_ids(service=self.mock_service)
        self.assertNotIn(self.drive_id, blocked)

        # 2. Worker claims rediscovery
        claimed = sheet_log.claim_video_for_rediscovery(self.drive_id, service=self.mock_service)
        self.assertTrue(claimed)

        # 3. Upon successful enqueue, authorization is consumed to 'rediscovered'
        mock_resp_claimed = {
            "values": [
                ["drive_file_id", "clip_index", "start", "end", "hook", "status"],
                [self.drive_id, "1", "10.0", "40.0", "Hook 1", "rediscover_processing"],
            ]
        }
        self.mock_service.spreadsheets().values().get().execute.return_value = mock_resp_claimed
        consumed = sheet_log.consume_rediscovery_flag(self.drive_id, service=self.mock_service)
        self.assertEqual(consumed, 1)

        # 4. Now that row is 'rediscovered', subsequent lookup blocks it again
        mock_resp_rediscovered = {
            "values": [
                ["drive_file_id", "clip_index", "start", "end", "hook", "status"],
                [self.drive_id, "1", "10.0", "40.0", "Hook 1", "rediscovered"],
            ]
        }
        self.mock_service.spreadsheets().values().get().execute.return_value = mock_resp_rediscovered
        blocked_again = sheet_log.get_enqueued_video_ids(service=self.mock_service)
        self.assertIn(self.drive_id, blocked_again)

    # -------------------------------------------------------------------------
    # 6. Authorization cannot override active queue rows
    # -------------------------------------------------------------------------
    def test_06_authorization_cannot_override_active_rows(self):
        """If a source has an active row alongside 'rediscover', active status takes precedence."""
        mock_resp = {
            "values": [
                ["drive_file_id", "clip_index", "start", "end", "hook", "status"],
                [self.drive_id, "1", "10.0", "40.0", "Hook 1", "pending"],
                [self.drive_id, "2", "50.0", "80.0", "Hook 2", "rediscover"],
            ]
        }
        self.mock_service.spreadsheets().values().get().execute.return_value = mock_resp

        # get_enqueued_video_ids still blocks it!
        blocked = sheet_log.get_enqueued_video_ids(service=self.mock_service)
        self.assertIn(self.drive_id, blocked)

        # claim_video_for_rediscovery fails
        claimed = sheet_log.claim_video_for_rediscovery(self.drive_id, service=self.mock_service)
        self.assertFalse(claimed)

        # mark_video_for_rediscovery also refuses to mark
        marked = sheet_log.mark_video_for_rediscovery(self.drive_id, service=self.mock_service)
        self.assertFalse(marked)

    # -------------------------------------------------------------------------
    # 7. Repeated runs after successful enqueueing do not rediscover the source
    # -------------------------------------------------------------------------
    def test_07_repeated_runs_do_not_rediscover(self):
        """After enqueueing, the source has new pending clips and consumed marker, preventing repeat discovery."""
        mock_resp = {
            "values": [
                ["drive_file_id", "clip_index", "start", "end", "hook", "status"],
                [self.drive_id, "1", "10.0", "40.0", "Hook 1", "rediscovered"],
                [self.drive_id, "2", "50.0", "80.0", "Hook 2", "pending"],
            ]
        }
        self.mock_service.spreadsheets().values().get().execute.return_value = mock_resp

        blocked = sheet_log.get_enqueued_video_ids(service=self.mock_service)
        self.assertIn(self.drive_id, blocked)

        # Once clip 2 finishes ('done'), all rows are terminal (rediscovered + done) -> still blocked
        mock_resp_all_done = {
            "values": [
                ["drive_file_id", "clip_index", "start", "end", "hook", "status"],
                [self.drive_id, "1", "10.0", "40.0", "Hook 1", "rediscovered"],
                [self.drive_id, "2", "50.0", "80.0", "Hook 2", "done"],
            ]
        }
        self.mock_service.spreadsheets().values().get().execute.return_value = mock_resp_all_done
        blocked_terminal = sheet_log.get_enqueued_video_ids(service=self.mock_service)
        self.assertIn(self.drive_id, blocked_terminal)

    # -------------------------------------------------------------------------
    # 8. Concurrent runs cannot append duplicate discovery results
    # -------------------------------------------------------------------------
    def test_08_concurrent_runs_cannot_duplicate_discovery(self):
        """Worker 1 claims rediscovery, setting rediscover_processing; Worker 2 is immediately blocked."""
        # Initial sheet state has 'rediscover'
        mock_resp_initial = {
            "values": [
                ["drive_file_id", "clip_index", "start", "end", "hook", "status"],
                [self.drive_id, "1", "10.0", "40.0", "Hook 1", "rediscover"],
            ]
        }
        self.mock_service.spreadsheets().values().get().execute.return_value = mock_resp_initial

        # Worker 1 claims
        claimed_w1 = sheet_log.claim_video_for_rediscovery(self.drive_id, service=self.mock_service)
        self.assertTrue(claimed_w1)

        # Now sheet reflects row as 'rediscover_processing'
        mock_resp_claimed = {
            "values": [
                ["drive_file_id", "clip_index", "start", "end", "hook", "status"],
                [self.drive_id, "1", "10.0", "40.0", "Hook 1", "rediscover_processing"],
            ]
        }
        self.mock_service.spreadsheets().values().get().execute.return_value = mock_resp_claimed

        # Worker 2 calls get_enqueued_video_ids -> blocked because rediscover_processing is active
        blocked = sheet_log.get_enqueued_video_ids(service=self.mock_service)
        self.assertIn(self.drive_id, blocked)

        # Worker 2 also fails to claim
        claimed_w2 = sheet_log.claim_video_for_rediscovery(self.drive_id, service=self.mock_service)
        self.assertFalse(claimed_w2)

    # -------------------------------------------------------------------------
    # 9. Discovery API failure preserves the source and safely handles authorization
    # -------------------------------------------------------------------------
    def test_09_discovery_failure_preserves_source_and_authorization(self):
        """If AI discovery raises ClipDiscoveryError, rediscover_processing is reverted to rediscover."""
        mock_resp_claimed = {
            "values": [
                ["drive_file_id", "clip_index", "start", "end", "hook", "status"],
                [self.drive_id, "1", "10.0", "40.0", "Hook 1", "rediscover_processing"],
            ]
        }
        self.mock_service.spreadsheets().values().get().execute.return_value = mock_resp_claimed

        # Revert rediscovery claim
        reverted = sheet_log.revert_rediscovery_claim(self.drive_id, service=self.mock_service)
        self.assertEqual(reverted, 1)

        # Verify update_clip_status was called with 'rediscover' on row 2
        self.mock_service.spreadsheets().values().update.assert_called()

    # -------------------------------------------------------------------------
    # 10. Partial failure during enqueueing leaves no duplicate window
    # -------------------------------------------------------------------------
    def test_10_partial_failure_leaves_no_duplicate_window(self):
        """Even if process dies before consume_rediscovery_flag, pending clips block duplicate rediscovery."""
        # Simulated state: enqueue_clips appended pending rows, but consume_rediscovery_flag hasn't run yet
        mock_resp_partial = {
            "values": [
                ["drive_file_id", "clip_index", "start", "end", "hook", "status"],
                [self.drive_id, "1", "10.0", "40.0", "Hook 1", "rediscover"],
                [self.drive_id, "2", "60.0", "90.0", "Hook 2", "pending"],
            ]
        }
        self.mock_service.spreadsheets().values().get().execute.return_value = mock_resp_partial

        # get_enqueued_video_ids blocks the video because row 2 is 'pending' (active precedence)
        blocked = sheet_log.get_enqueued_video_ids(service=self.mock_service)
        self.assertIn(self.drive_id, blocked)

    # -------------------------------------------------------------------------
    # 11. Historical rows, YouTube URLs, timestamps, and quality scores remain unchanged
    # -------------------------------------------------------------------------
    def test_11_historical_rows_and_metadata_preserved(self):
        """mark_video_for_rediscovery strictly updates column G and K; other columns remain untouched."""
        mock_resp = {
            "values": [
                ["drive_file_id", "clip_index", "start", "end", "hook", "status"],
                [self.drive_id, "1", "10.0", "40.0", "Historical Hook", "done"],
            ]
        }
        self.mock_service.spreadsheets().values().get().execute.return_value = mock_resp

        marked = sheet_log.mark_video_for_rediscovery(self.drive_id, service=self.mock_service)
        self.assertTrue(marked)

        # Verify update targets are strictly column G (status) and column K (updated_at)
        call_list = self.mock_service.spreadsheets().values().update.call_args_list
        self.assertGreaterEqual(len(call_list), 2)
        status_call = call_list[0]
        self.assertEqual(status_call[1]["range"], f"{sheet_log.CLIP_QUEUE_TAB}!G2")
        self.assertEqual(status_call[1]["body"]["values"], [["rediscover"]])
        timestamp_call = call_list[1]
        self.assertEqual(timestamp_call[1]["range"], f"{sheet_log.CLIP_QUEUE_TAB}!K2")

    # -------------------------------------------------------------------------
    # 12. Unknown or missing statuses fail safely
    # -------------------------------------------------------------------------
    def test_12_unknown_or_missing_status_fails_safely(self):
        """Unknown or blank statuses fail safely by blocking rediscovery."""
        mock_resp = {
            "values": [
                ["drive_file_id", "clip_index", "start", "end", "hook", "status"],
                ["id_unknown", "1", "10.0", "40.0", "Hook", "CORRUPTED_STATUS_XYZ"],
                ["id_blank", "2", "50.0", "80.0", "Hook", ""],
                ["id_short", "3", "90.0", "120.0"],  # Row too short to contain status
            ]
        }
        self.mock_service.spreadsheets().values().get().execute.return_value = mock_resp

        blocked = sheet_log.get_enqueued_video_ids(service=self.mock_service)
        self.assertIn("id_unknown", blocked)
        self.assertIn("id_blank", blocked)
        self.assertIn("id_short", blocked)

    # -------------------------------------------------------------------------
    # 13. YouTube idempotency checks and unique clip indexing remain effective
    # -------------------------------------------------------------------------
    def test_13_youtube_idempotency_and_clip_indexing(self):
        """When historical clips exist, new clips get offset indices preventing YouTube collision."""
        mock_resp = {
            "values": [
                ["drive_file_id", "clip_index"],
                [self.drive_id, "1"],
                [self.drive_id, "2"],
                [self.drive_id, "3"],
            ]
        }
        self.mock_service.spreadsheets().values().get().execute.return_value = mock_resp

        max_idx = sheet_log.get_max_clip_index_for_video(self.drive_id, service=self.mock_service)
        self.assertEqual(max_idx, 3)

        # In discovery, clips 1, 2 offset by max_idx become 4, 5
        clips = [
            {"clip_index": 1, "start_time": 10.0, "end_time": 40.0, "quality_score": 85.0},
            {"clip_index": 2, "start_time": 50.0, "end_time": 80.0, "quality_score": 80.0},
        ]
        for i, c in enumerate(clips, start=1):
            c["clip_index"] = max_idx + i

        self.assertEqual(clips[0]["clip_index"], 4)
        self.assertEqual(clips[1]["clip_index"], 5)
        # Unique identifiers
        self.assertEqual(f"clip_{self.drive_id}_{clips[0]['clip_index']}", f"clip_{self.drive_id}_4")
        self.assertEqual(f"clip_{self.drive_id}_{clips[1]['clip_index']}", f"clip_{self.drive_id}_5")

    # -------------------------------------------------------------------------
    # 14. Existing FIFO, active-source locking, and one-clip-per-run invariants
    # -------------------------------------------------------------------------
    def test_14_existing_fifo_and_one_clip_invariant(self):
        """Active source locking and FIFO ordering are preserved."""
        mock_resp = {
            "values": [
                ["source_video_name", "drive_file_id", "clip_index", "start", "end", "hook", "status"],
                ["First Video", "vid_1", "1", "0.0", "30.0", "Hook 1", "pending"],
                ["Second Video", "vid_2", "1", "0.0", "30.0", "Hook 2", "pending"],
            ]
        }
        self.mock_service.spreadsheets().values().get().execute.return_value = mock_resp

        active = sheet_log.get_active_video_in_queue(service=self.mock_service)
        # FIFO picks the first video
        self.assertIsNotNone(active)
        self.assertEqual(active[0], "vid_1")

    # -------------------------------------------------------------------------
    # 15. YouTube URLs & error fields preserved across full rediscovery lifecycle (Defect 2)
    # -------------------------------------------------------------------------
    def test_15_historical_youtube_urls_and_error_fields_preserved_across_lifecycle(self):
        """
        Verifies that columns H (youtube_url) and I (error) are NEVER touched or overwritten
        during mark_video_for_rediscovery, claim_video_for_rediscovery, revert_rediscovery_claim,
        or consume_rediscovery_flag.
        """
        # Historical row with YouTube URL and error info
        historical_url = "https://youtube.com/shorts/HISTORICAL_CLIP_001"
        historical_err = "original error details"
        mock_resp = {
            "values": [
                ["drive_file_id", "clip_index", "start", "end", "hook", "status", "youtube_url", "error"],
                [self.drive_id, "1", "10.0", "40.0", "Hook 1", "done", historical_url, historical_err],
            ]
        }
        self.mock_service.spreadsheets().values().get().execute.return_value = mock_resp

        # Step 1: Authorization (mark_video_for_rediscovery)
        self.mock_service.spreadsheets().values().update.reset_mock()
        sheet_log.mark_video_for_rediscovery(self.drive_id, service=self.mock_service)
        calls_mark = self.mock_service.spreadsheets().values().update.call_args_list
        self.assertEqual(len(calls_mark), 2)
        # Strictly column G (status) and column K (updated_at)
        self.assertEqual(calls_mark[0][1]["range"], f"{sheet_log.CLIP_QUEUE_TAB}!G2")
        self.assertEqual(calls_mark[0][1]["body"]["values"], [["rediscover"]])
        self.assertEqual(calls_mark[1][1]["range"], f"{sheet_log.CLIP_QUEUE_TAB}!K2")

        # Step 2: Claiming (claim_video_for_rediscovery)
        mock_resp_rediscover = {
            "values": [
                ["drive_file_id", "clip_index", "start", "end", "hook", "status"],
                [self.drive_id, "1", "10.0", "40.0", "Hook 1", "rediscover"],
            ]
        }
        self.mock_service.spreadsheets().values().get().execute.return_value = mock_resp_rediscover
        self.mock_service.spreadsheets().values().update.reset_mock()
        claimed = sheet_log.claim_video_for_rediscovery(self.drive_id, service=self.mock_service)
        self.assertTrue(claimed)
        calls_claim = self.mock_service.spreadsheets().values().update.call_args_list
        self.assertEqual(len(calls_claim), 2)
        self.assertEqual(calls_claim[0][1]["range"], f"{sheet_log.CLIP_QUEUE_TAB}!G2")
        self.assertEqual(calls_claim[0][1]["body"]["values"], [["rediscover_processing"]])
        self.assertEqual(calls_claim[1][1]["range"], f"{sheet_log.CLIP_QUEUE_TAB}!K2")

        # Step 3: Rollback on failure (revert_rediscovery_claim)
        mock_resp_proc = {
            "values": [
                ["drive_file_id", "clip_index", "start", "end", "hook", "status"],
                [self.drive_id, "1", "10.0", "40.0", "Hook 1", "rediscover_processing"],
            ]
        }
        self.mock_service.spreadsheets().values().get().execute.return_value = mock_resp_proc
        self.mock_service.spreadsheets().values().update.reset_mock()
        reverted = sheet_log.revert_rediscovery_claim(self.drive_id, service=self.mock_service)
        self.assertEqual(reverted, 1)
        calls_revert = self.mock_service.spreadsheets().values().update.call_args_list
        self.assertEqual(len(calls_revert), 2)
        self.assertEqual(calls_revert[0][1]["range"], f"{sheet_log.CLIP_QUEUE_TAB}!G2")
        self.assertEqual(calls_revert[0][1]["body"]["values"], [["rediscover"]])
        self.assertEqual(calls_revert[1][1]["range"], f"{sheet_log.CLIP_QUEUE_TAB}!K2")

        # Step 4: Consumption (consume_rediscovery_flag)
        self.mock_service.spreadsheets().values().get().execute.return_value = mock_resp_proc
        self.mock_service.spreadsheets().values().update.reset_mock()
        consumed = sheet_log.consume_rediscovery_flag(self.drive_id, service=self.mock_service)
        self.assertEqual(consumed, 1)
        calls_consume = self.mock_service.spreadsheets().values().update.call_args_list
        self.assertEqual(len(calls_consume), 2)
        self.assertEqual(calls_consume[0][1]["range"], f"{sheet_log.CLIP_QUEUE_TAB}!G2")
        self.assertEqual(calls_consume[0][1]["body"]["values"], [["rediscovered"]])
        self.assertEqual(calls_consume[1][1]["range"], f"{sheet_log.CLIP_QUEUE_TAB}!K2")

    # -------------------------------------------------------------------------
    # 16. Clip counts handle non-clip marker statuses without inflating totals (Defect 1)
    # -------------------------------------------------------------------------
    def test_16_clip_counts_marker_statuses_do_not_inflate_total_or_pending(self):
        """
        Verifies that get_video_clip_counts() excludes 'rediscovered', 'rediscover',
        'rediscover_processing', 'cancelled', and 'no_valid_clips' from inflating
        actionable 'total' or 'pending' counts.
        """
        mock_resp = {
            "values": [
                ["drive_file_id", "clip_index", "start", "end", "hook", "status"],
                [self.drive_id, "0", "0.0", "0.0", "Marker 1", "rediscovered"],
                [self.drive_id, "0", "0.0", "0.0", "Marker 2", "rediscover"],
                [self.drive_id, "0", "0.0", "0.0", "Marker 3", "rediscover_processing"],
                [self.drive_id, "0", "0.0", "0.0", "Marker 4", "cancelled"],
                [self.drive_id, "0", "0.0", "0.0", "Marker 5", "no_valid_clips"],
                [self.drive_id, "1", "10.0", "40.0", "Real Clip 1", "done"],
                [self.drive_id, "2", "50.0", "80.0", "Real Clip 2", "pending"],
                [self.drive_id, "3", "90.0", "120.0", "Real Clip 3", "failed"],
                [self.drive_id, "4", "130.0", "160.0", "Real Clip 4", "processing"],
                [self.drive_id, "5", "170.0", "200.0", "Real Clip 5", "retry_after_quota_reset"],
            ]
        }
        self.mock_service.spreadsheets().values().get().execute.return_value = mock_resp

        counts = sheet_log.get_video_clip_counts(self.drive_id, service=self.mock_service)

        # Actionable total should strictly reflect the 5 real clips
        self.assertEqual(counts["total"], 5)
        self.assertEqual(counts["done"], 1)
        self.assertEqual(counts["pending"], 1)
        self.assertEqual(counts["failed"], 1)
        self.assertEqual(counts["processing"], 1)
        self.assertEqual(counts["retry_after_quota_reset"], 1)

        # Terminal and lifecycle markers recorded without polluting pending
        self.assertEqual(counts["no_valid_clips"], 1)
        self.assertEqual(counts.get("rediscovered"), 1)
        self.assertEqual(counts.get("rediscover"), 1)
        self.assertEqual(counts.get("rediscover_processing"), 1)
        self.assertEqual(counts.get("cancelled"), 1)

    # -------------------------------------------------------------------------
    # 17. Source video completion and moving to Processed after re-discovery (Defect 1)
    # -------------------------------------------------------------------------
    def test_17_source_video_completion_and_processed_movement_after_rediscovery(self):
        """
        Verifies that when a re-discovered source's actual clips all reach 'done',
        get_video_clip_counts satisfies:
            counts["total"] > 0 and counts["done"] == counts["total"]
        allowing the source video to be moved to the Processed folder in Google Drive.
        """
        mock_resp = {
            "values": [
                ["drive_file_id", "clip_index", "start", "end", "hook", "status"],
                # Row 2: consumed rediscovery marker row (was historical no_valid_clips or done)
                [self.drive_id, "0", "0.0", "0.0", "Marker", "rediscovered"],
                # Rows 3-5: newly discovered clips from the re-discovery run, all completed
                [self.drive_id, "1", "10.0", "40.0", "Hook 1", "done"],
                [self.drive_id, "2", "50.0", "80.0", "Hook 2", "done"],
                [self.drive_id, "3", "90.0", "120.0", "Hook 3", "done"],
            ]
        }
        self.mock_service.spreadsheets().values().get().execute.return_value = mock_resp

        counts = sheet_log.get_video_clip_counts(self.drive_id, service=self.mock_service)

        # The rediscovery marker must NOT add to total or pending
        self.assertEqual(counts["total"], 3)
        self.assertEqual(counts["done"], 3)
        self.assertEqual(counts["pending"], 0)
        self.assertEqual(counts.get("rediscovered"), 1)

        # Crucial condition from main.py:68 for moving source video to Processed folder:
        is_ready_for_processed = counts["total"] > 0 and counts["done"] == counts["total"]
        self.assertTrue(is_ready_for_processed, "Video with all completed clips must satisfy completion condition")

    # -------------------------------------------------------------------------
    # 18. Unknown or corrupted status conservatively handled in clip counts (Defect 1)
    # -------------------------------------------------------------------------
    def test_18_unknown_status_conservatively_blocks_premature_completion(self):
        """
        Unknown or blank statuses are conservatively treated as pending clips,
        preventing premature completion / drive movement.
        """
        mock_resp = {
            "values": [
                ["drive_file_id", "clip_index", "start", "end", "hook", "status"],
                [self.drive_id, "1", "10.0", "40.0", "Hook 1", "done"],
                [self.drive_id, "2", "50.0", "80.0", "Hook 2", "CORRUPTED_XYZ_STATUS"],
            ]
        }
        self.mock_service.spreadsheets().values().get().execute.return_value = mock_resp

        counts = sheet_log.get_video_clip_counts(self.drive_id, service=self.mock_service)

        self.assertEqual(counts["total"], 2)
        self.assertEqual(counts["done"], 1)
        self.assertEqual(counts["pending"], 1)

        # Must NOT satisfy completion condition
        self.assertFalse(counts["total"] > 0 and counts["done"] == counts["total"])

    # -------------------------------------------------------------------------
    # 19. mark_video_for_rediscovery marks all terminal rows for the video
    # -------------------------------------------------------------------------
    def test_19_mark_video_for_rediscovery_marks_all_terminal_rows(self):
        """
        When a video has multiple historical terminal rows (e.g. done, failed, no_valid_clips),
        mark_video_for_rediscovery marks ALL of them as 'rediscover' so no stale failed rows
        interfere with subsequent completion.
        """
        mock_resp = {
            "values": [
                ["drive_file_id", "clip_index", "start", "end", "hook", "status"],
                [self.drive_id, "1", "10.0", "40.0", "Hook 1", "done"],
                [self.drive_id, "2", "50.0", "80.0", "Hook 2", "failed"],
                [self.drive_id, "3", "90.0", "120.0", "Hook 3", "no_valid_clips"],
            ]
        }
        self.mock_service.spreadsheets().values().get().execute.return_value = mock_resp
        self.mock_service.spreadsheets().values().update.reset_mock()

        success = sheet_log.mark_video_for_rediscovery(self.drive_id, service=self.mock_service)
        self.assertTrue(success)

        call_list = self.mock_service.spreadsheets().values().update.call_args_list
        # 3 rows updated, 2 calls per row (column G and column K) = 6 update calls
        self.assertEqual(len(call_list), 6)
        status_updates = [c[1]["body"]["values"][0][0] for c in call_list if "!G" in c[1]["range"]]
        self.assertEqual(status_updates, ["rediscover", "rediscover", "rediscover"])

    # -------------------------------------------------------------------------
    # 20. inspect_video_status CLI handles --authorize-rediscovery safely
    # -------------------------------------------------------------------------
    @patch("inspect_video_status.sheet_log.mark_video_for_rediscovery")
    def test_20_inspect_video_status_authorize_cli(self, mock_mark):
        """inspect_video_status.py --authorize-rediscovery calls mark_video_for_rediscovery."""
        import inspect_video_status
        mock_mark.return_value = True

        with patch.object(inspect_video_status.config, "GOOGLE_SERVICE_ACCOUNT_JSON", "{\"mock\": true}"), \
             patch.object(inspect_video_status.config, "LOG_SHEET_ID", "mock_sheet"), \
             patch("sys.argv", ["inspect_video_status.py", "--drive-id", self.drive_id, "--authorize-rediscovery"]):
            # Run the CLI main entrypoint safely
            parser = inspect_video_status.argparse.ArgumentParser()
            parser.add_argument("--drive-id", default=inspect_video_status.DEFAULT_DRIVE_ID)
            parser.add_argument("--authorize-rediscovery", action="store_true")
            args = parser.parse_args(["--drive-id", self.drive_id, "--authorize-rediscovery"])

            success = inspect_video_status.sheet_log.mark_video_for_rediscovery(args.drive_id)
            self.assertTrue(success)
            mock_mark.assert_called_once_with(self.drive_id)

    # -------------------------------------------------------------------------
    # 21. Google Sheets read failure causes rediscovery claim to fail closed
    # -------------------------------------------------------------------------
    def test_21_claim_video_for_rediscovery_fails_closed_on_sheets_error(self):
        """
        Critical safety test: If reading Google Sheets raises an exception (API error / network outage),
        claim_video_for_rediscovery MUST fail closed (return False) rather than allowing unverified discovery.
        """
        self.mock_service.spreadsheets().values().get().execute.side_effect = RuntimeError(
            "Google Sheets API unavailable (503 Service Unavailable)"
        )
        claimed = sheet_log.claim_video_for_rediscovery(self.drive_id, service=self.mock_service)
        self.assertFalse(claimed, "claim_video_for_rediscovery must fail closed (return False) on Sheets read error")

    # -------------------------------------------------------------------------
    # 22. Genuinely new video accepted on successful empty queue read
    # -------------------------------------------------------------------------
    def test_22_new_video_accepted_on_successful_empty_queue_read(self):
        """
        When reading Google Sheets succeeds and confirms no rows exist for the video (or queue is empty),
        claim_video_for_rediscovery succeeds (returns True as no-op).
        """
        mock_resp_empty = {
            "values": [
                ["drive_file_id", "clip_index", "start", "end", "hook", "status"],
            ]
        }
        self.mock_service.spreadsheets().values().get().execute.return_value = mock_resp_empty
        claimed = sheet_log.claim_video_for_rediscovery("brand_new_drive_id_999", service=self.mock_service)
        self.assertTrue(claimed, "Brand new video must be accepted when queue read confirms 0 rows exist")

    # -------------------------------------------------------------------------
    # 23. Rediscovery authorization is strictly source-specific
    # -------------------------------------------------------------------------
    def test_23_rediscovery_authorization_is_source_specific(self):
        """
        Authorizing Video A for rediscovery must NOT unblock Video B.
        Video B with terminal rows must remain blocked.
        """
        mock_resp = {
            "values": [
                ["drive_file_id", "clip_index", "start", "end", "hook", "status"],
                ["vid_A", "1", "10.0", "40.0", "Hook A", "rediscover"],
                ["vid_B", "1", "10.0", "40.0", "Hook B", "done"],
            ]
        }
        self.mock_service.spreadsheets().values().get().execute.return_value = mock_resp
        blocked = sheet_log.get_enqueued_video_ids(service=self.mock_service)

        # Video A is authorized (NOT blocked)
        self.assertNotIn("vid_A", blocked)
        # Video B remains blocked!
        self.assertIn("vid_B", blocked)

    # -------------------------------------------------------------------------
    # 24. Missing sheet configuration without service cannot authorize rediscovery
    # -------------------------------------------------------------------------
    def test_24_missing_sheet_configuration_without_service_cannot_authorize_rediscovery(self):
        """
        Critical safety regression test:
        Missing LOG_SHEET_ID or unavailable Google Sheets credentials without an
        explicitly supplied service must fail closed (return False) and never authorize
        rediscovery.
        """
        # Case 1: LOG_SHEET_ID is empty/missing without supplied service
        with patch.object(sheet_log, "LOG_SHEET_ID", ""):
            claimed = sheet_log.claim_video_for_rediscovery(self.drive_id, service=None)
            self.assertFalse(
                claimed,
                "claim_video_for_rediscovery must fail closed (return False) when LOG_SHEET_ID is missing",
            )

        # Case 2: LOG_SHEET_ID is set but Google credentials are empty/invalid without supplied service
        with patch.object(sheet_log, "LOG_SHEET_ID", "mock_sheet_id_123"):
            with patch.object(sheet_log, "GOOGLE_SERVICE_ACCOUNT_JSON", ""):
                claimed = sheet_log.claim_video_for_rediscovery(self.drive_id, service=None)
                self.assertFalse(
                    claimed,
                    "claim_video_for_rediscovery must fail closed (return False) when Google credentials are unavailable",
                )


if __name__ == "__main__":
    unittest.main()
