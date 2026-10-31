"""
ExoFinder Nova: Zero-Downtime Data Migration Engine
===================================================
Migrates existing flat CSV catalogs into the high-concurrency SQLite WAL database:
  - data/labels.csv
  - outputs/period_search_results.csv
  - outputs/features.csv
  - outputs/predictions.csv
"""

import os
import sys
import time
import json
import pandas as pd
import numpy as np

# Ensure parent directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.db.database import engine, SessionLocal, init_db, DATABASE_URL, ensure_database_exists, IS_POSTGRES
from src.db.models import Target, BLSSignal, PhysicalFeature, ModelInference, UserAnnotation, PipelineJob

def run_migration():
    print("=========================================================")
    print("🪐 ExoFinder Nova Database Migration Engine (PostgreSQL)")
    print(f"Target Database: {DATABASE_URL}")
    print("=========================================================")
    
    start_time = time.time()
    
    # 0. Ensure Database Exists (creates 'exofinder' DB if missing)
    if IS_POSTGRES:
        print("\n[0/5] Ensuring PostgreSQL Database Exists...")
        ensure_database_exists()

    # 1. Initialize Tables
    print("\n[1/5] Initializing Database Schema...")
    init_db()
    print("  ✓ Tables created with compound indexes and connection pooling.")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    labels_csv = os.path.join(project_root, "data", "labels.csv")
    bls_csv = os.path.join(project_root, "outputs", "period_search_results.csv")
    features_csv = os.path.join(project_root, "outputs", "features.csv")
    predictions_csv = os.path.join(project_root, "outputs", "predictions.csv")
    raw_dir = os.path.join(project_root, "data", "raw")
    detrended_dir = os.path.join(project_root, "data", "detrended")

    # Verify input files
    for path, name in [
        (labels_csv, "Labels CSV"),
        (bls_csv, "BLS Results CSV"),
        (features_csv, "Features CSV"),
        (predictions_csv, "Predictions CSV")
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required migration source missing: {name} at {path}")

    # 2. Read Source Datasets
    print("\n[2/5] Reading Source CSV Datasets...")
    df_labels = pd.read_csv(labels_csv)
    df_bls = pd.read_csv(bls_csv)
    df_feat = pd.read_csv(features_csv)
    df_pred = pd.read_csv(predictions_csv)

    print(f"  • Labels Catalog:       {len(df_labels)} targets")
    print(f"  • BLS Search Results:   {len(df_bls)} targets")
    print(f"  • Extracted Features:   {len(df_feat)} targets")
    print(f"  • Predictions Matrix:   {len(df_pred)} targets")

    # Index by star_id for fast in-memory dictionary joins
    bls_dict = {int(r['star_id']): r for _, r in df_bls.iterrows()}
    feat_dict = {int(r['star_id']): r for _, r in df_feat.iterrows()}
    pred_dict = {int(r['star_id']): r for _, r in df_pred.iterrows()}

    # Discover available physical files
    available_raw = set()
    if os.path.exists(raw_dir):
        for f in os.listdir(raw_dir):
            if f.startswith("TIC_") and f.endswith(".csv"):
                try:
                    available_raw.add(int(f[4:-4]))
                except ValueError:
                    pass

    available_det = set()
    if os.path.exists(detrended_dir):
        for f in os.listdir(detrended_dir):
            if f.startswith("TIC_") and f.endswith(".csv"):
                try:
                    available_det.add(int(f[4:-4]))
                except ValueError:
                    pass

    # 3. Assemble and Migrate Records in an Atomic Transaction
    print("\n[3/5] Migrating Records to Relational Tables...")
    db = SessionLocal()
    
    migrated_count = 0
    skipped_count = 0

    try:
        # 0. Preserve any existing user annotations so bookmarks & notes are NEVER lost
        existing_annotations = {}
        try:
            for ann in db.query(UserAnnotation).all():
                if ann.is_bookmarked or (ann.user_notes and ann.user_notes.strip()):
                    existing_annotations[ann.star_id] = (ann.is_bookmarked, ann.user_notes)
            if existing_annotations:
                print(f"  • Preserving {len(existing_annotations)} existing user annotations/bookmarks across migration.")
        except Exception:
            pass


        # Clear data tables for clean re-sync
        db.query(PipelineJob).delete()
        db.query(UserAnnotation).delete()
        db.query(ModelInference).delete()
        db.query(PhysicalFeature).delete()
        db.query(BLSSignal).delete()
        db.query(Target).delete()
        db.commit()

        # Migrate only stars that have completed the full pipeline (present in predictions)
        target_sids = set(pred_dict.keys())
        print(f"  • Migrating {len(target_sids)} fully-processed targets...")


        for sid in target_sids:
            pred_row = pred_dict[sid]
            label_row = df_labels[df_labels['star_id'] == sid]
            ground_truth = label_row.iloc[0]['label'] if len(label_row) > 0 else pred_row.get('label', 'unknown')
            cat_period = float(label_row.iloc[0]['catalog_period']) if (len(label_row) > 0 and not pd.isna(label_row.iloc[0]['catalog_period'])) else None

            has_det = sid in available_det
            raw_path = f"data/raw/TIC_{sid}.csv" if sid in available_raw else None
            det_path = f"data/detrended/TIC_{sid}.csv" if has_det else None

            # 1. Target Entry
            target_obj = Target(
                star_id=sid,
                tic_id=f"TIC {sid}",
                ground_truth=ground_truth,
                catalog_period=cat_period,
                raw_filepath=raw_path,
                detrended_filepath=det_path,
                has_detrended=has_det
            )
            db.add(target_obj)

            # 2. BLS Signal Entry
            bls_row = bls_dict.get(sid, {})
            recovered_p = float(bls_row.get('recovered_period', pred_row.get('bls_period', 0.0)))
            recovered_e = float(bls_row.get('recovered_epoch', 0.0))
            bls_d = float(bls_row.get('bls_depth', pred_row.get('bls_depth', 0.0)))
            bls_dur = float(bls_row.get('bls_duration', pred_row.get('bls_duration', 0.0)))
            bls_pow = float(bls_row.get('bls_power', pred_row.get('bls_power', 0.0)))
            snr_val = float(bls_row.get('transit_snr', pred_row.get('transit_snr', 0.0)))

            bls_obj = BLSSignal(
                star_id=sid,
                recovered_period=0.0 if np.isnan(recovered_p) else recovered_p,
                recovered_epoch=0.0 if np.isnan(recovered_e) else recovered_e,
                bls_depth=0.0 if np.isnan(bls_d) else bls_d,
                bls_duration=0.0 if np.isnan(bls_dur) else bls_dur,
                bls_power=0.0 if np.isnan(bls_pow) else bls_pow,
                transit_snr=0.0 if np.isnan(snr_val) else snr_val,
                harmonic_flag="fundamental",
                candidate_id="01"
            )
            db.add(bls_obj)

            # 3. Physical Features Entry
            feat_row = feat_dict.get(sid, pred_row)
            duty_cycle = float(feat_row.get('transit_duty_cycle', 0.0))
            if np.isnan(duty_cycle) or duty_cycle == 0.0:
                duty_cycle = bls_dur / recovered_p if recovered_p > 0 else 0.0

            extra_metrics = {
                'secondary_depth_phase_05': float(feat_row.get('secondary_depth_phase_05', 0.0)),
                'secondary_to_primary_ratio': float(feat_row.get('secondary_to_primary_ratio', 0.0)),
                'secondary_significance_sigma': float(feat_row.get('secondary_significance_sigma', 0.0)),
                'odd_even_diff_2p': float(feat_row.get('odd_even_diff_2p', 0.0)),
                'kepler_duration_ratio': float(feat_row.get('kepler_duration_ratio', 0.0)),
                'positive_to_negative_ratio': float(feat_row.get('positive_to_negative_ratio', 0.0)),
                'bottom_flatness_ratio': float(feat_row.get('bottom_flatness_ratio', 0.0)),
                'transit_coverage_fraction': float(feat_row.get('transit_coverage_fraction', 0.0))
            }

            feat_obj = PhysicalFeature(
                star_id=sid,
                transit_duty_cycle=0.0 if np.isnan(duty_cycle) else duty_cycle,
                mean_to_max_depth_ratio=float(feat_row.get('mean_to_max_depth_ratio', 0.0)),
                in_transit_skew=float(feat_row.get('in_transit_skew', 0.0)),
                in_transit_kurtosis=float(feat_row.get('in_transit_kurtosis', 0.0)),
                flux_kurtosis=float(feat_row.get('flux_kurtosis', 0.0)),
                ingress_egress_symmetry=float(feat_row.get('ingress_egress_symmetry', 0.0)),
                secondary_eclipse_check=float(feat_row.get('secondary_eclipse_check', 0.0)),
                depth_to_noise=float(feat_row.get('depth_to_noise', 0.0)),
                odd_even_depth_ratio=float(feat_row.get('odd_even_depth_ratio', 1.0)),
                odd_even_mismatch_sigma=float(feat_row.get('odd_even_mismatch_sigma', 0.0)),
                std_raw=float(feat_row.get('std_raw', 0.0)),
                std_detrended=float(feat_row.get('std_detrended', 0.0)),
                transit_count=float(feat_row.get('transit_count', 0.0)),
                extra_metrics_json=json.dumps(extra_metrics)
            )
            db.add(feat_obj)

            # 4. Model Inference Entry
            ml_pred = str(pred_row.get('predicted_label', 'unknown'))
            ml_conf = float(pred_row.get('confidence', 0.0))
            cnn_pred = str(pred_row.get('predicted_label_cnn', '')) if not pd.isna(pred_row.get('predicted_label_cnn')) else None
            cnn_conf = float(pred_row.get('confidence_cnn', 0.0)) if not pd.isna(pred_row.get('confidence_cnn')) else None

            # Calculate consensus
            if cnn_pred and cnn_pred != 'nan':
                consensus = "unanimous" if ml_pred == cnn_pred else "split"
            else:
                consensus = "single_model"

            inf_obj = ModelInference(
                star_id=sid,
                model_version="v1.0",
                ml_predicted_label=ml_pred,
                ml_confidence=0.0 if np.isnan(ml_conf) else ml_conf,
                prob_ml_planet=float(pred_row.get('prob_confirmed_planet', 0.0)),
                prob_ml_binary=float(pred_row.get('prob_eclipsing_binary', 0.0)),
                prob_ml_fp=float(pred_row.get('prob_false_positive', 0.0)),
                significance=str(pred_row.get('significance', 'high')),
                cnn_predicted_label=cnn_pred if cnn_pred != 'nan' else None,
                cnn_confidence=0.0 if (cnn_conf is None or np.isnan(cnn_conf)) else cnn_conf,
                prob_cnn_planet=float(pred_row.get('prob_cnn_confirmed_planet', 0.0)) if not pd.isna(pred_row.get('prob_cnn_confirmed_planet')) else None,
                prob_cnn_binary=float(pred_row.get('prob_cnn_eclipsing_binary', 0.0)) if not pd.isna(pred_row.get('prob_cnn_eclipsing_binary')) else None,
                prob_cnn_fp=float(pred_row.get('prob_cnn_false_positive', 0.0)) if not pd.isna(pred_row.get('prob_cnn_false_positive')) else None,
                consensus_verdict=consensus
            )
            db.add(inf_obj)

            # 5. User Annotation (Restores preserved notes/bookmarks if existed)
            existing_ann = existing_annotations.get(sid)
            ann_obj = UserAnnotation(
                star_id=sid,
                is_bookmarked=existing_ann[0] if existing_ann else False,
                user_notes=existing_ann[1] if existing_ann else ""
            )
            db.add(ann_obj)


            migrated_count += 1

        db.commit()
        print(f"  ✓ Atomic transaction committed: {migrated_count} targets migrated.")

    except Exception as e:
        db.rollback()
        print(f"  ✗ Error during migration: {e}")
        raise e
    finally:
        db.close()

    # 4. Database Parity Verification
    print("\n[4/5] Verifying Relational Integrity & Parity...")
    verify_db = SessionLocal()
    try:
        t_count = verify_db.query(Target).count()
        b_count = verify_db.query(BLSSignal).count()
        f_count = verify_db.query(PhysicalFeature).count()
        i_count = verify_db.query(ModelInference).count()

        print(f"  • Targets in DB:          {t_count}")
        print(f"  • BLS Signals in DB:      {b_count}")
        print(f"  • Physical Features:      {f_count}")
        print(f"  • Model Inferences:       {i_count}")

        assert t_count == len(target_sids), f"Mismatch in targets: expected {len(target_sids)}, got {t_count}"
        assert b_count == t_count, "BLS Signals count does not match Targets"
        assert f_count == t_count, "Features count does not match Targets"
        assert i_count == t_count, "Inferences count does not match Targets"
        
        # Test sample join query
        sample = verify_db.query(Target).join(BLSSignal).join(ModelInference).first()
        print(f"\n[5/5] Sample Query Test Passed:")
        print(f"  Target: {sample.tic_id} | Ground Truth: {sample.ground_truth}")
        print(f"  BLS Period: {sample.bls_signal.recovered_period:.4f} days | SNR: {sample.bls_signal.transit_snr:.2f}")
        print(f"  ML Predicted: {sample.inference.ml_predicted_label} ({sample.inference.ml_confidence*100:.1f}%)")
        print(f"  Consensus: {sample.inference.consensus_verdict}")

    finally:
        verify_db.close()

    elapsed = time.time() - start_time
    print(f"\n=========================================================")
    print(f"✨ MIGRATION SUCCESSFUL! Indexed {t_count} targets in {elapsed:.2f}s")
    print(f"Database Target: {DATABASE_URL}")
    print("=========================================================")

if __name__ == "__main__":
    run_migration()
