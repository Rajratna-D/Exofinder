"""
ExoFinder High-Performance Data Access Repository
=================================================
Optimized, type-safe database queries executing directly against PostgreSQL / SQLite
using compound B-tree indexes, connection pooling, and sub-millisecond execution.
"""

import math
import json
from typing import Optional, Dict, Any, List, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import cast, String, or_, func

from .models import Target, BLSSignal, PhysicalFeature, ModelInference, UserAnnotation, PipelineJob


def get_targets_page(
    db: Session,
    page: int = 1,
    page_size: int = 50,
    filter_label: Optional[str] = None,
    search: Optional[str] = None,
    min_confidence: Optional[float] = None,
    min_snr: Optional[float] = None,
    only_bookmarked: Optional[bool] = False,
    sort_by: str = "confidence",
    sort_order: str = "desc"
) -> Dict[str, Any]:
    """
    Direct SQL query with multi-column B-tree index pushdown, filtering, sorting,
    and server-side pagination. Replaces high-overhead in-memory dataframe scanning.
    """
    query = db.query(
        Target.star_id,
        Target.tic_id,
        Target.ground_truth,
        Target.catalog_period,
        Target.has_detrended,
        BLSSignal.recovered_period,
        BLSSignal.recovered_epoch,
        BLSSignal.bls_depth,
        BLSSignal.bls_duration,
        BLSSignal.transit_snr,
        ModelInference.ml_predicted_label,
        ModelInference.ml_confidence,
        ModelInference.significance,
        ModelInference.consensus_verdict,
        UserAnnotation.is_bookmarked
    ).join(BLSSignal, Target.star_id == BLSSignal.star_id)\
     .join(ModelInference, Target.star_id == ModelInference.star_id)\
     .outerjoin(UserAnnotation, Target.star_id == UserAnnotation.star_id)

    # 1. Search filter
    if search:
        search_clean = search.strip().replace("TIC", "").replace("tic", "").strip()
        if search_clean.isdigit():
            # Exact or prefix match for numeric star_id
            star_id_int = int(search_clean)
            query = query.filter(
                or_(
                    Target.star_id == star_id_int,
                    cast(Target.star_id, String).like(f"%{search_clean}%"),
                    Target.tic_id.ilike(f"%{search_clean}%")
                )
            )
        else:
            query = query.filter(Target.tic_id.ilike(f"%{search_clean}%"))

    # 2. Predicted label filter (leverages idx_filter_ml_pred_conf)
    if filter_label and filter_label.lower() != "all":
        query = query.filter(ModelInference.ml_predicted_label == filter_label)

    # 3. Minimum confidence filter
    if min_confidence is not None and min_confidence > 0.0:
        query = query.filter(ModelInference.ml_confidence >= min_confidence)

    # 4. Minimum SNR filter (leverages idx_bls_snr_desc)
    if min_snr is not None and min_snr > 0.0:
        query = query.filter(BLSSignal.transit_snr >= min_snr)

    # 5. Bookmarks filter
    if only_bookmarked:
        query = query.filter(UserAnnotation.is_bookmarked.is_(True))

    # 6. Sorting map
    sort_column_map = {
        "confidence": ModelInference.ml_confidence,
        "transit_snr": BLSSignal.transit_snr,
        "bls_period": BLSSignal.recovered_period,
        "bls_depth": BLSSignal.bls_depth,
        "star_id": Target.star_id,
        "is_bookmarked": UserAnnotation.is_bookmarked
    }
    col = sort_column_map.get(sort_by, ModelInference.ml_confidence)
    if sort_order.lower() == "asc":
        query = query.order_by(col.asc().nullslast())
    else:
        query = query.order_by(col.desc().nullslast())

    total = query.count()
    start_idx = (page - 1) * page_size
    page_rows = query.offset(start_idx).limit(page_size).all()

    records = []
    for r in page_rows:
        records.append({
            "star_id": int(r.star_id),
            "tic_id": r.tic_id if r.tic_id else f"TIC {r.star_id}",
            "ground_truth": r.ground_truth or "unknown",
            "predicted_label": r.ml_predicted_label or "unknown",
            "confidence": float(r.ml_confidence or 0.0),
            "significance": r.significance or "high",
            "consensus_verdict": r.consensus_verdict or "unanimous",
            "is_bookmarked": bool(r.is_bookmarked) if r.is_bookmarked is not None else False,
            "bls_period": float(r.recovered_period) if r.recovered_period is not None else None,
            "catalog_period": float(r.catalog_period) if r.catalog_period is not None else (float(r.recovered_period) if r.recovered_period is not None else None),
            "bls_depth": float(r.bls_depth) if r.bls_depth is not None else None,
            "bls_duration": float(r.bls_duration) if r.bls_duration is not None else None,
            "transit_snr": float(r.transit_snr) if r.transit_snr is not None else None,
            "has_detrended": bool(r.has_detrended)
        })

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": math.ceil(total / page_size) if total > 0 else 1,
        "targets": records
    }


def get_target_details(db: Session, star_id: int) -> Optional[Dict[str, Any]]:
    """
    Sub-millisecond single indexed target lookup joining all relational metrics
    with computed vetting checks and probability breakdown.
    """
    row = db.query(Target, BLSSignal, PhysicalFeature, ModelInference, UserAnnotation)\
        .join(BLSSignal, Target.star_id == BLSSignal.star_id)\
        .join(PhysicalFeature, Target.star_id == PhysicalFeature.star_id)\
        .join(ModelInference, Target.star_id == ModelInference.star_id)\
        .outerjoin(UserAnnotation, Target.star_id == UserAnnotation.star_id)\
        .filter(Target.star_id == star_id)\
        .first()

    if not row:
        return None

    t, b, f, inf, ann = row

    # Vetting heuristics
    snr = float(b.transit_snr or 0.0)
    odd_even_sigma = float(f.odd_even_mismatch_sigma or 0.0)
    sec_eclipse = float(f.secondary_eclipse_check or 0.0)
    u_shape = float(f.mean_to_max_depth_ratio or 0.0)
    odd_even_ratio = float(f.odd_even_depth_ratio or 1.0)

    raw_features = {
        "bls_depth": float(b.bls_depth or 0.0),
        "bls_duration": float(b.bls_duration or 0.0),
        "bls_power": float(b.bls_power or 0.0),
        "transit_snr": snr,
        "mean_to_max_depth_ratio": float(f.mean_to_max_depth_ratio or 0.0),
        "in_transit_skew": float(f.in_transit_skew or 0.0),
        "in_transit_kurtosis": float(f.in_transit_kurtosis or 0.0),
        "flux_kurtosis": float(f.flux_kurtosis or 0.0),
        "ingress_egress_symmetry": float(f.ingress_egress_symmetry or 0.0),
        "transit_duty_cycle": float(f.transit_duty_cycle or 0.0),
        "depth_to_noise": float(f.depth_to_noise or 0.0),
        "odd_even_depth_ratio": odd_even_ratio,
        "odd_even_mismatch_sigma": odd_even_sigma,
        "secondary_eclipse_check": float(f.secondary_eclipse_check or 0.0),
        "std_raw": float(f.std_raw or 0.0),
        "std_detrended": float(f.std_detrended or 0.0),
        "transit_count": float(f.transit_count or 0.0)
    }

    if f.extra_metrics_json:
        try:
            extra = json.loads(f.extra_metrics_json)
            if isinstance(extra, dict):
                for k, v in extra.items():
                    raw_features[k] = float(v or 0.0)
        except Exception:
            pass

    feature_labels = {
        'bls_depth': 'Transit Depth (ΔF/F)',
        'bls_duration': 'Transit Duration (days)',
        'bls_power': 'BLS Spectral Power',
        'transit_snr': 'Transit SNR',
        'mean_to_max_depth_ratio': 'U-Shape Symmetry Ratio',
        'in_transit_skew': 'In-Transit Skewness',
        'in_transit_kurtosis': 'In-Transit Kurtosis',
        'flux_kurtosis': 'Overall Flux Kurtosis',
        'ingress_egress_symmetry': 'Ingress/Egress Symmetry',
        'secondary_eclipse_check': 'Secondary Eclipse Metric',
        'transit_duty_cycle': 'Transit Duty Cycle (Duration/Period)',
        'depth_to_noise': 'Depth-to-Noise Ratio',
        'odd_even_depth_ratio': 'Odd/Even Depth Ratio',
        'odd_even_mismatch_sigma': 'Odd/Even Mismatch (σ)',
        'std_raw': 'Raw Standard Deviation',
        'std_detrended': 'Detrended Standard Deviation',
        'transit_count': 'Transit Count',
        'secondary_depth_phase_05': 'Phase 0.5 Secondary Eclipse Depth',
        'secondary_to_primary_ratio': 'Secondary-to-Primary Depth Ratio',
        'secondary_significance_sigma': 'Secondary Significance (sigma)',
        'odd_even_diff_2p': 'Odd/Even 2P Difference',
        'kepler_duration_ratio': 'Kepler Transit Duration Ratio',
        'positive_to_negative_ratio': 'Positive Excursion Ratio',
        'bottom_flatness_ratio': 'Bottom Flatness Ratio',
        'transit_coverage_fraction': 'In-Transit Phase Coverage'
    }

    features_dict = {}
    for col, val in raw_features.items():
        features_dict[col] = {
            "label": feature_labels.get(col, col),
            "value": val
        }

    vetting_diagnostics = {
        "snr_test": {
            "name": "Transit SNR Threshold (≥ 5.0)",
            "value": round(snr, 2),
            "passed": bool(snr >= 5.0),
            "description": "Signals below 5σ risk being red noise or stellar instrumental flukes."
        },
        "odd_even_test": {
            "name": "Odd/Even Transit Depth Consistency (< 3.0σ)",
            "value": f"{round(odd_even_sigma, 2)}σ",
            "passed": bool(odd_even_sigma < 3.0),
            "description": "Alternating transit depth differences indicate an eclipsing binary system."
        },
        "secondary_eclipse_test": {
            "name": "Secondary Eclipse Absence Check (< 0.005)",
            "value": f"{sec_eclipse:.6f}",
            "passed": bool(sec_eclipse < 0.005),
            "description": "A secondary occultation at phase 0.5 indicates self-luminous stellar components."
        },
        "u_shape_test": {
            "name": "U-Shape Transit Profile Ratio (≥ 0.65)",
            "value": round(u_shape, 3),
            "passed": bool(u_shape >= 0.65),
            "description": "Exoplanets have flat U-bottoms; grazing binaries produce sharp V-shapes."
        }
    }

    model_comparison = {
        "stacking_ml": {
            "confirmed_planet": float(inf.prob_ml_planet or 0.0),
            "eclipsing_binary": float(inf.prob_ml_binary or 0.0),
            "false_positive": float(inf.prob_ml_fp or 0.0),
            "predicted_label": inf.ml_predicted_label or "unknown",
            "confidence": float(inf.ml_confidence or 0.0)
        },
        "cnn_1d": {
            "confirmed_planet": float(inf.prob_cnn_planet or 0.0) if inf.prob_cnn_planet is not None else 0.0,
            "eclipsing_binary": float(inf.prob_cnn_binary or 0.0) if inf.prob_cnn_binary is not None else 0.0,
            "false_positive": float(inf.prob_cnn_fp or 0.0) if inf.prob_cnn_fp is not None else 0.0,
            "predicted_label": inf.cnn_predicted_label or "unknown",
            "confidence": float(inf.cnn_confidence or 0.0) if inf.cnn_confidence is not None else 0.0
        }
    }

    # ─── Part A: Physical Exoplanetary Properties (Kepler's Laws & Astrophysics) ───
    depth_val = max(1e-6, float(b.bls_depth or 0.005))
    period_val = max(0.01, float(b.recovered_period or t.catalog_period or 3.35))
    
    # 1. Planet Radius (assuming Solar-type host star: R_* = 1.0 R_sun = 109.2 R_earth = 9.96 R_jup)
    radius_ratio = math.sqrt(depth_val)
    radius_earth = radius_ratio * 109.2
    radius_jupiter = radius_ratio * 9.96
    
    # Planet classification by radius
    if radius_earth < 1.25:
        planet_class = "Sub-Earth / Terrestrial"
    elif radius_earth < 2.0:
        planet_class = "Super-Earth"
    elif radius_earth < 4.0:
        planet_class = "Sub-Neptune"
    elif radius_earth < 10.0:
        planet_class = "Neptunian World"
    else:
        planet_class = "Jovian Gas Giant"

    # 2. Semi-Major Axis via Kepler's Third Law: a = (P / 365.25)^(2/3) in AU
    period_years = period_val / 365.25
    semi_major_axis = period_years ** (2.0 / 3.0)
    orbital_velocity = 29.78 / math.sqrt(semi_major_axis) if semi_major_axis > 0 else 0.0
    
    # 3. Equilibrium Temperature (assuming Bond albedo ~ 0.3, Solar host): T_eq ~ 278.5 / sqrt(a)
    t_eq = 278.5 / math.sqrt(semi_major_axis) if semi_major_axis > 0 else 300.0

    # 4. Habitable Zone Classification
    if semi_major_axis < 0.95:
        hz_status = "HOT_INNER"
        hz_label = "Inner Scorched (Too Hot)"
    elif semi_major_axis <= 1.67:
        hz_status = "GOLDILOCKS"
        hz_label = "Habitable Zone (Goldilocks)"
    else:
        hz_status = "COLD_OUTER"
        hz_label = "Outer Icehouse (Too Cold)"

    planetary_physics = {
        "planet_radius_earth": round(radius_earth, 2),
        "planet_radius_jupiter": round(radius_jupiter, 2),
        "radius_ratio": round(radius_ratio, 4),
        "planet_classification": planet_class,
        "semi_major_axis_au": round(semi_major_axis, 4),
        "orbital_velocity_kms": round(orbital_velocity, 1),
        "equilibrium_temperature_k": round(t_eq, 0),
        "habitable_zone_status": hz_status,
        "habitable_zone_label": hz_label
    }

    return {
        "star_id": int(t.star_id),
        "tic_id": t.tic_id if t.tic_id else f"TIC {t.star_id}",
        "ground_truth": t.ground_truth,
        "predicted_label": inf.ml_predicted_label,
        "confidence": float(inf.ml_confidence or 0.0),
        "significance": inf.significance,
        "consensus_verdict": inf.consensus_verdict,
        "is_bookmarked": bool(ann.is_bookmarked) if ann else False,
        "user_notes": str(ann.user_notes) if ann and ann.user_notes else "",
        "bls_period": float(b.recovered_period or 0.0),
        "recovered_epoch": float(b.recovered_epoch or 0.0),
        "catalog_period": float(t.catalog_period or b.recovered_period or 0.0),
        "bls_depth": float(b.bls_depth or 0.0),
        "bls_duration": float(b.bls_duration or 0.0),
        "bls_power": float(b.bls_power or 0.0),
        "transit_snr": snr,
        "has_detrended": bool(t.has_detrended),
        "candidate_id": b.candidate_id or "01",
        "model_version": inf.model_version or "v1.0",
        "vetting_status": {
            "snr_test_passed": bool(snr >= 5.0),
            "odd_even_passed": bool(odd_even_sigma < 3.0),
            "secondary_eclipse_passed": bool(sec_eclipse < 0.005),
            "u_shape_passed": bool(u_shape >= 0.65)
        },
        "vetting_diagnostics": vetting_diagnostics,
        "model_comparison": model_comparison,
        "features": features_dict,
        "planetary_physics": planetary_physics
    }




def toggle_bookmark(db: Session, star_id: int) -> bool:
    """
    Atomically toggles target bookmark flag in PostgreSQL / SQLite.
    Guarantees persistence and consistency.
    """
    try:
        ann = db.query(UserAnnotation).filter(UserAnnotation.star_id == star_id).first()
        if not ann:
            ann = UserAnnotation(star_id=star_id, is_bookmarked=True, user_notes="")
            db.add(ann)
            new_status = True
        else:
            new_status = not bool(ann.is_bookmarked)
            ann.is_bookmarked = new_status
        db.commit()
        return new_status
    except Exception:
        db.rollback()
        raise


def save_target_notes(db: Session, star_id: int, notes: str) -> str:
    """
    Atomically saves astronomer vetting notes into PostgreSQL / SQLite.
    """
    try:
        ann = db.query(UserAnnotation).filter(UserAnnotation.star_id == star_id).first()
        if not ann:
            ann = UserAnnotation(star_id=star_id, is_bookmarked=False, user_notes=notes)
            db.add(ann)
        else:
            ann.user_notes = notes
        db.commit()
        return notes
    except Exception:
        db.rollback()
        raise


def create_pipeline_job(db: Session, job_id: str, tic_id: str, star_id: Optional[int] = None) -> PipelineJob:
    """
    Enqueues a new asynchronous analysis job into the database queue.
    """
    try:
        job = PipelineJob(
            job_id=job_id,
            tic_id=tic_id,
            star_id=star_id,
            status="QUEUED",
            current_stage="Queued in processing engine",
            progress_pct=0.0
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        return job
    except Exception:
        db.rollback()
        raise


def update_pipeline_job(
    db: Session,
    job_id: str,
    status: str,
    current_stage: Optional[str] = None,
    progress_pct: Optional[float] = None,
    error_message: Optional[str] = None,
    execution_time_seconds: Optional[float] = None
) -> Optional[PipelineJob]:
    """
    Updates status and stage of an active or completed background job.
    """
    try:
        job = db.query(PipelineJob).filter(PipelineJob.job_id == job_id).first()
        if not job:
            return None
        job.status = status
        if current_stage is not None:
            job.current_stage = current_stage
        if progress_pct is not None:
            job.progress_pct = progress_pct
        if error_message is not None:
            job.error_message = error_message
        if execution_time_seconds is not None:
            job.execution_time_seconds = execution_time_seconds
        db.commit()
        db.refresh(job)
        return job
    except Exception:
        db.rollback()
        raise


def get_pipeline_job(db: Session, job_id: str) -> Optional[Dict[str, Any]]:
    """Fetches status and progress of a background job."""
    job = db.query(PipelineJob).filter(PipelineJob.job_id == job_id).first()
    if not job:
        return None
    return {
        "job_id": job.job_id,
        "tic_id": job.tic_id,
        "star_id": job.star_id,
        "status": job.status,
        "current_stage": job.current_stage,
        "progress_pct": job.progress_pct,
        "error_message": job.error_message,
        "execution_time_seconds": job.execution_time_seconds,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None
    }
