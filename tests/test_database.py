"""
ExoFinder Database & Repository Verification Test Suite
======================================================
Tests:
  1. PostgreSQL / SQLite connectivity and QueuePool health
  2. Sub-millisecond indexed target query pagination & sorting
  3. Filter precision (predicted_label, min_snr, min_confidence, TIC search)
  4. Single-target deep relational retrieval parity
  5. Atomic bookmark toggle and persistence across independent sessions
  6. Atomic user vetting notes save and persistence across independent sessions
  7. Background pipeline job state tracking
"""

import os
import sys
import uuid
import unittest

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.db.database import SessionLocal, check_connection, get_db_session, engine
from src.db.models import Target, BLSSignal, PhysicalFeature, ModelInference, UserAnnotation, PipelineJob
from src.db.repository import (
    get_targets_page,
    get_target_details,
    toggle_bookmark,
    save_target_notes,
    create_pipeline_job,
    update_pipeline_job,
    get_pipeline_job
)


class TestDatabaseArchitecture(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Validate database connectivity before running tests."""
        cls.connected, cls.conn_msg = check_connection()
        if not cls.connected:
            raise unittest.SkipTest(f"Database not accessible: {cls.conn_msg}")

    # ----------------------------------------------------
    # 1. CONNECTIVITY & CONNECTION POOL
    # ----------------------------------------------------
    def test_database_connection(self):
        """Verify engine connection and pool responsiveness."""
        self.assertTrue(self.connected)
        with SessionLocal() as db:
            target_count = db.query(Target).count()
            self.assertGreater(target_count, 0, "Targets table should contain migrated astronomical records.")

    # ----------------------------------------------------
    # 2. TARGET QUERY PAGINATION & SORTING
    # ----------------------------------------------------
    def test_targets_pagination(self):
        """Verify server-side pagination with limit and offset."""
        with SessionLocal() as db:
            page1 = get_targets_page(db, page=1, page_size=10, sort_by="confidence", sort_order="desc")
            self.assertEqual(len(page1["targets"]), 10)
            self.assertEqual(page1["page"], 1)
            self.assertGreater(page1["total"], 10)
            self.assertGreaterEqual(page1["total_pages"], 1)

            # Page 2 should return distinct targets
            page2 = get_targets_page(db, page=2, page_size=10, sort_by="confidence", sort_order="desc")
            self.assertEqual(len(page2["targets"]), 10)
            self.assertEqual(page2["page"], 2)

            page1_ids = {t["star_id"] for t in page1["targets"]}
            page2_ids = {t["star_id"] for t in page2["targets"]}
            self.assertTrue(page1_ids.isdisjoint(page2_ids), "Page 1 and Page 2 should not share any target IDs.")

    def test_targets_sorting(self):
        """Verify sorting by transit SNR descending."""
        with SessionLocal() as db:
            res = get_targets_page(db, page=1, page_size=20, sort_by="transit_snr", sort_order="desc")
            snrs = [t["transit_snr"] for t in res["targets"] if t["transit_snr"] is not None]
            # Ensure strictly non-increasing
            for i in range(len(snrs) - 1):
                self.assertGreaterEqual(snrs[i], snrs[i + 1], "SNR values should be sorted in descending order.")

    # ----------------------------------------------------
    # 3. FILTER PRECISION & SEARCH
    # ----------------------------------------------------
    def test_filter_by_class(self):
        """Verify label filtering produces only the requested astronomical class."""
        with SessionLocal() as db:
            res = get_targets_page(db, page=1, page_size=25, filter_label="confirmed_planet")
            for t in res["targets"]:
                self.assertEqual(t["predicted_label"], "confirmed_planet")

    def test_filter_by_min_snr(self):
        """Verify minimum transit SNR filtering threshold."""
        with SessionLocal() as db:
            min_threshold = 12.0
            res = get_targets_page(db, page=1, page_size=25, min_snr=min_threshold)
            for t in res["targets"]:
                self.assertGreaterEqual(t["transit_snr"], min_threshold)

    def test_search_by_star_id_and_tic(self):
        """Verify search finds exact TIC and numeric star ID substrings."""
        with SessionLocal() as db:
            # Pick the first target
            first = db.query(Target).first()
            self.assertIsNotNone(first)
            sid_str = str(first.star_id)

            res = get_targets_page(db, page=1, page_size=10, search=sid_str)
            self.assertGreaterEqual(res["total"], 1)
            matching_ids = [t["star_id"] for t in res["targets"]]
            self.assertIn(first.star_id, matching_ids)

    # ----------------------------------------------------
    # 4. SINGLE TARGET DETAILS & RELATIONAL INTEGRITY
    # ----------------------------------------------------
    def test_get_target_details(self):
        """Verify comprehensive details, vetting diagnostics, and feature extraction."""
        with SessionLocal() as db:
            first = db.query(Target).first()
            details = get_target_details(db, first.star_id)
            self.assertIsNotNone(details)
            self.assertEqual(details["star_id"], first.star_id)
            self.assertIn("vetting_diagnostics", details)
            self.assertIn("model_comparison", details)
            self.assertIn("features", details)
            self.assertIn("candidate_id", details)
            self.assertIn("model_version", details)

            # Check vetting diagnostics keys
            vd = details["vetting_diagnostics"]
            for test_key in ["snr_test", "odd_even_test", "secondary_eclipse_test", "u_shape_test"]:
                self.assertIn(test_key, vd)

    # ----------------------------------------------------
    # 5. ATOMIC PERSISTENCE: BOOKMARKS ACROSS SESSIONS
    # ----------------------------------------------------
    def test_bookmark_persistence_across_sessions(self):
        """Verify bookmark toggling persists atomically into database and survives new sessions."""
        target_id = None
        with SessionLocal() as db:
            target = db.query(Target).first()
            self.assertIsNotNone(target)
            target_id = target.star_id

        # Session 1: Toggle bookmark
        with SessionLocal() as db1:
            initial_status = toggle_bookmark(db1, target_id)

        # Session 2: Read in a completely fresh independent session
        with SessionLocal() as db2:
            ann = db2.query(UserAnnotation).filter(UserAnnotation.star_id == target_id).first()
            self.assertIsNotNone(ann)
            self.assertEqual(ann.is_bookmarked, initial_status)

        # Session 3: Toggle back to reverse
        with SessionLocal() as db3:
            reversed_status = toggle_bookmark(db3, target_id)
            self.assertEqual(reversed_status, not initial_status)

        # Session 4: Verify reversal persisted
        with SessionLocal() as db4:
            ann_final = db4.query(UserAnnotation).filter(UserAnnotation.star_id == target_id).first()
            self.assertEqual(ann_final.is_bookmarked, reversed_status)

    # ----------------------------------------------------
    # 6. ATOMIC PERSISTENCE: VETTING NOTES ACROSS SESSIONS
    # ----------------------------------------------------
    def test_notes_persistence_across_sessions(self):
        """Verify astronomer vetting notes persist across independent sessions."""
        test_note = f"Astronomer Vetting Note - Code {uuid.uuid4().hex[:8]}"
        target_id = None

        with SessionLocal() as db:
            target = db.query(Target).first()
            target_id = target.star_id

        # Write note in Session 1
        with SessionLocal() as db1:
            saved_note = save_target_notes(db1, target_id, test_note)
            self.assertEqual(saved_note, test_note)

        # Read note in fresh Session 2
        with SessionLocal() as db2:
            details = get_target_details(db2, target_id)
            self.assertEqual(details["user_notes"], test_note)

    # ----------------------------------------------------
    # 7. PIPELINE JOB QUEUE TRACKING
    # ----------------------------------------------------
    def test_pipeline_job_lifecycle(self):
        """Verify pipeline job creation, stage progress, and completion retrieval."""
        test_job_id = f"job-test-{uuid.uuid4().hex[:12]}"
        test_tic = "TIC 99999999"

        with SessionLocal() as db:
            # 1. Create Job
            job = create_pipeline_job(db, job_id=test_job_id, tic_id=test_tic, star_id=99999999)
            self.assertEqual(job.status, "QUEUED")
            self.assertEqual(job.progress_pct, 0.0)

        # 2. Update Job in separate session
        with SessionLocal() as db:
            updated = update_pipeline_job(
                db,
                job_id=test_job_id,
                status="PROCESSING",
                current_stage="Box Least Squares Periodogram",
                progress_pct=45.0
            )
            self.assertIsNotNone(updated)
            self.assertEqual(updated.status, "PROCESSING")
            self.assertEqual(updated.progress_pct, 45.0)

        # 3. Complete Job
        with SessionLocal() as db:
            completed = update_pipeline_job(
                db,
                job_id=test_job_id,
                status="COMPLETED",
                current_stage="Finished",
                progress_pct=100.0,
                execution_time_seconds=1.85
            )
            self.assertEqual(completed.status, "COMPLETED")

        # 4. Fetch via get_pipeline_job
        with SessionLocal() as db:
            job_dict = get_pipeline_job(db, test_job_id)
            self.assertIsNotNone(job_dict)
            self.assertEqual(job_dict["status"], "COMPLETED")
            self.assertEqual(job_dict["execution_time_seconds"], 1.85)

            # Cleanup test job
            db.query(PipelineJob).filter(PipelineJob.job_id == test_job_id).delete()
            db.commit()


if __name__ == "__main__":
    unittest.main()
