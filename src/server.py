import os
import math
import pickle
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

import io
import scipy.io.wavfile as wavfile
from fastapi import FastAPI, HTTPException, Query, status, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel, Field

# PostgreSQL High-Performance Database Integration
try:
    from src.db.database import SessionLocal, get_db, DATABASE_URL, check_connection
    from src.db.models import Target, BLSSignal, PhysicalFeature, ModelInference, UserAnnotation, PipelineJob
    from src.db.migrate import run_migration
    from src.db.repository import (
        get_targets_page,
        get_target_details as repo_get_target_details,
        toggle_bookmark as repo_toggle_bookmark,
        save_target_notes as repo_save_target_notes,
        create_pipeline_job,
        update_pipeline_job,
        get_pipeline_job
    )
    HAS_DB = True
except Exception as e:
    HAS_DB = False
    DATABASE_URL = None
    def get_db():
        yield None
    print(f"[ExoFinder Nova Engine] Warning: Could not load src.db: {e}")


# Determine project paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
DATA_DIR = os.path.join(BASE_DIR, "data")
DETRENDED_DIR = os.path.join(DATA_DIR, "detrended")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PRECOMPUTED_DIR = os.path.join(DATA_DIR, "precomputed")
os.makedirs(PRECOMPUTED_DIR, exist_ok=True)
_LC_RAM_CACHE: Dict[int, bytes] = {}


PREDICTIONS_PATH = os.path.join(OUTPUTS_DIR, "predictions.csv")
PERIOD_RESULTS_PATH = os.path.join(OUTPUTS_DIR, "period_search_results.csv")
FEATURES_PATH = os.path.join(OUTPUTS_DIR, "features.csv")
MODEL_PATH = os.path.join(OUTPUTS_DIR, "model.pkl")
CNN_MODEL_PATH = os.path.join(OUTPUTS_DIR, "cnn_model.pt")

# Global Cache State
state: Dict[str, Any] = {
    "targets_df": None,
    "target_dict": {},
    "model_bundle": None,
    "cnn_model": None,
    "available_detrended_ids": set(),
}

FEATURE_COLS = [
    'bls_depth', 'bls_duration', 'bls_power', 
    'transit_snr', 'mean_to_max_depth_ratio', 'in_transit_skew', 
    'in_transit_kurtosis', 'flux_kurtosis',
    'ingress_egress_symmetry', 'transit_duty_cycle', 'depth_to_noise',
    'odd_even_depth_ratio', 'odd_even_mismatch_sigma',
    'secondary_depth_phase_05', 'secondary_to_primary_ratio', 'secondary_significance_sigma',
    'odd_even_diff_2p', 'kepler_duration_ratio', 'positive_to_negative_ratio',
    'bottom_flatness_ratio', 'transit_coverage_fraction', 'bls_period'
]

FEATURE_LABELS = {
    'bls_depth': 'Transit Depth (ΔF/F)',
    'bls_duration': 'Transit Duration (days)',
    'bls_power': 'BLS Spectral Power',
    'transit_snr': 'Integrated Transit SNR (σ)',
    'mean_to_max_depth_ratio': 'U-Shape Profile Ratio',
    'in_transit_skew': 'In-Transit Skewness',
    'in_transit_kurtosis': 'In-Transit Kurtosis',
    'flux_kurtosis': 'Overall Flux Kurtosis',
    'ingress_egress_symmetry': 'Ingress/Egress Symmetry',
    'transit_duty_cycle': 'Transit Duty Cycle (Duration/Period)',
    'depth_to_noise': 'Depth-to-Noise Ratio',
    'odd_even_depth_ratio': 'Odd/Even Depth Ratio (1P)',
    'odd_even_mismatch_sigma': 'Odd/Even Mismatch (1P σ)',
    'secondary_depth_phase_05': 'Phase 0.5 Secondary Eclipse Depth',
    'secondary_to_primary_ratio': 'Secondary/Primary Depth Ratio',
    'secondary_significance_sigma': 'Secondary Eclipse Significance (σ)',
    'odd_even_diff_2p': 'Odd/Even Depth Mismatch (2P Harmonic)',
    'kepler_duration_ratio': 'Keplerian Duration Bound (T_dur / P^(1/3))',
    'positive_to_negative_ratio': 'Positive/Negative Excursion Ratio',
    'bottom_flatness_ratio': 'Transit Bottom Flatness Ratio',
    'transit_coverage_fraction': 'Orbital Phase Coverage Fraction',
    'bls_period': 'Recovered Orbital Period (days)'
}

def load_data_and_models():
    """Initializes and caches targets, features, models into memory from SQLite WAL database."""
    # 1. Discover available detrended light curve files (CSV or Parquet)
    if os.path.exists(DETRENDED_DIR):
        for fname in os.listdir(DETRENDED_DIR):
            if fname.startswith("TIC_") and fname.endswith(".csv"):
                try:
                    sid = int(fname[4:-4])
                    state["available_detrended_ids"].add(sid)
                except ValueError:
                    pass

    pq_dir = os.path.join(DATA_DIR, "detrended_parquet")
    if os.path.exists(pq_dir):
        try:
            import pyarrow.parquet as pq
            ds = pq.ParquetDataset(pq_dir)
            t = ds.read(columns=['star_id'])
            sids = set(t['star_id'].to_pylist())
            state["available_detrended_ids"].update(sids)
        except Exception:
            pass

    # 2. Check if PostgreSQL database exists and connect
    db_loaded = False
    if HAS_DB:
        try:
            connected, msg = check_connection()
            if connected:
                db = SessionLocal()
                try:
                    target_count = db.query(Target).count()
                    if target_count == 0:
                        print("[ExoFinder Nova Engine] PostgreSQL database empty. Running initial migration...")
                        try:
                            run_migration()
                        except Exception as me:
                            print(f"[ExoFinder Nova Engine] Migration error: {me}")
                    
                    records = db.query(Target, BLSSignal, PhysicalFeature, ModelInference, UserAnnotation)\
                        .join(BLSSignal, Target.star_id == BLSSignal.star_id)\
                        .join(PhysicalFeature, Target.star_id == PhysicalFeature.star_id)\
                        .join(ModelInference, Target.star_id == ModelInference.star_id)\
                        .outerjoin(UserAnnotation, Target.star_id == UserAnnotation.star_id)\
                        .all()

                    target_dict = {}
                    records_list = []
                    for t, b, f, inf, ann in records:
                        sid = int(t.star_id)
                        row_dict = {
                            "star_id": sid,
                            "tic_id": t.tic_id,
                            "ground_truth": t.ground_truth,
                            "label": t.ground_truth,
                            "catalog_period": t.catalog_period or b.recovered_period,
                            "bls_period": b.recovered_period,
                            "recovered_epoch": b.recovered_epoch,
                            "bls_depth": b.bls_depth,
                            "bls_duration": b.bls_duration,
                            "bls_power": b.bls_power,
                            "transit_snr": b.transit_snr,
                            "predicted_label": inf.ml_predicted_label,
                            "confidence": inf.ml_confidence,
                            "significance": inf.significance,
                            "prob_confirmed_planet": inf.prob_ml_planet,
                            "prob_eclipsing_binary": inf.prob_ml_binary,
                            "prob_false_positive": inf.prob_ml_fp,
                            "predicted_label_cnn": inf.cnn_predicted_label,
                            "confidence_cnn": inf.cnn_confidence,
                            "prob_cnn_confirmed_planet": inf.prob_cnn_planet,
                            "prob_cnn_eclipsing_binary": inf.prob_cnn_binary,
                            "prob_cnn_false_positive": inf.prob_cnn_fp,
                            "consensus_verdict": inf.consensus_verdict,
                            "is_bookmarked": ann.is_bookmarked if ann else False,
                            "user_notes": ann.user_notes if ann else "",
                            "has_detrended": sid in state["available_detrended_ids"],
                            # Features
                            "transit_duty_cycle": f.transit_duty_cycle,
                            "mean_to_max_depth_ratio": f.mean_to_max_depth_ratio,
                            "in_transit_skew": f.in_transit_skew,
                            "in_transit_kurtosis": f.in_transit_kurtosis,
                            "flux_kurtosis": f.flux_kurtosis,
                            "ingress_egress_symmetry": f.ingress_egress_symmetry,
                            "secondary_eclipse_check": f.secondary_eclipse_check,
                            "depth_to_noise": f.depth_to_noise,
                            "odd_even_depth_ratio": f.odd_even_depth_ratio,
                            "odd_even_mismatch_sigma": f.odd_even_mismatch_sigma,
                            "std_raw": f.std_raw,
                            "std_detrended": f.std_detrended,
                            "transit_count": f.transit_count
                        }
                        target_dict[sid] = row_dict
                        records_list.append(row_dict)

                    state["target_dict"] = target_dict
                    state["targets_df"] = pd.DataFrame(records_list)
                    db_loaded = True
                    print(f"[ExoFinder Nova Engine] Indexed {len(target_dict)} targets from PostgreSQL ({DATABASE_URL}).")
                finally:
                    db.close()
        except Exception as e:
            print(f"[ExoFinder Nova Engine] PostgreSQL loading exception: {e}")

    # Fallback to CSV if DB loading failed
    if not db_loaded and os.path.exists(PREDICTIONS_PATH):
        df = pd.read_csv(PREDICTIONS_PATH)
        if os.path.exists(PERIOD_RESULTS_PATH):
            res_df = pd.read_csv(PERIOD_RESULTS_PATH)
            if 'recovered_epoch' in res_df.columns and 'star_id' in res_df.columns:
                df = pd.merge(df, res_df[['star_id', 'recovered_epoch']], on='star_id', how='left')
        if 'significance' not in df.columns:
            df['significance'] = 'high'
        if 'confidence' not in df.columns:
            df['confidence'] = 0.5
        if 'recovered_epoch' not in df.columns:
            df['recovered_epoch'] = 0.0
        df['has_detrended'] = df['star_id'].isin(state["available_detrended_ids"])
        df['recovered_epoch'] = df['recovered_epoch'].fillna(0.0)
        df['catalog_period'] = df['catalog_period'].fillna(df['bls_period'])
        df['is_bookmarked'] = False
        df['user_notes'] = ""
        df['consensus_verdict'] = "unanimous"
        state["targets_df"] = df
        state["target_dict"] = {int(row['star_id']): row.to_dict() for _, row in df.iterrows()}
        print(f"[ExoFinder Nova Engine] Fallback: Indexed {len(state['target_dict'])} targets from CSVs.")

    # 3. Load Trained Stacking ML Ensemble
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                state["model_bundle"] = pickle.load(f)
            print("[ExoFinder Nova Engine] Calibrated Stacking ML Ensemble loaded successfully.")
        except Exception as e:
            print(f"[ExoFinder Nova Engine] Warning: Failed to load model.pkl: {e}")

    # 4. Attempt loading 1D-CNN PyTorch model
    if os.path.exists(CNN_MODEL_PATH):
        try:
            import torch
            from src.deep_learning import ExoFinderNet
            cnn = ExoFinderNet(num_tabular_features=len(FEATURE_COLS), num_classes=3)
            checkpoint = torch.load(CNN_MODEL_PATH, map_location='cpu', weights_only=False)
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    cnn.load_state_dict(checkpoint['state_dict'])
                elif 'model_state_dict' in checkpoint:
                    cnn.load_state_dict(checkpoint['model_state_dict'])
                else:
                    cnn.load_state_dict(checkpoint)
            else:
                cnn.load_state_dict(checkpoint)
            cnn.eval()
            state["cnn_model"] = cnn
            print("[ExoFinder Nova Engine] NASA Astronet 1D-CNN model loaded on CPU.")
        except Exception as e:
            print(f"[ExoFinder Nova Engine] 1D-CNN initialization skipped or CPU fallback: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_data_and_models()
    yield

app = FastAPI(
    title="ExoFinder Nova Engine API",
    description="High-Speed Astronomical Photometry & Exoplanet Classification Engine",
    version="2.0.0",
    lifespan=lifespan
)


# Enable CORS for local Vite development and remote proxies
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────

def fold_lightcurve(time: np.ndarray, flux: np.ndarray, period: float, epoch: float):
    """Folds time-series into [-0.5, 0.5] phase space and sorts by phase."""
    if len(time) == 0 or np.isnan(period) or period <= 0:
        return np.array([]), np.array([])
    phase = ((time - epoch) % period) / period
    phase = np.where(phase > 0.5, phase - 1.0, phase)
    sort_idx = np.argsort(phase)
    return phase[sort_idx], flux[sort_idx]

def compute_binned_profile(phase: np.ndarray, flux: np.ndarray, num_bins: int = 80, window: float = 0.2):
    """Computes median binned flux profile within [-window, window]."""
    bins = np.linspace(-window, window, num_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    binned_flux = []
    for i in range(len(bins) - 1):
        mask = (phase >= bins[i]) & (phase < bins[i + 1])
        if np.any(mask):
            binned_flux.append(float(np.nanmedian(flux[mask])))
        else:
            binned_flux.append(1.0)
    return bin_centers.tolist(), binned_flux

def downsample_lttb(time: np.ndarray, flux: np.ndarray, max_points: int = 2000):
    """
    Uniform decimation / stride downsampling that preserves endpoints and dips.
    Ensures 60 FPS lightweight transfer over network for large lightcurves.
    """
    n = len(time)
    if n <= max_points:
        return time.tolist(), flux.tolist()
    
    step = int(math.ceil(n / max_points))
    # Select regularly spaced cadences, always including the minimum flux point (deepest transit)
    indices = np.arange(0, n, step)
    min_idx = int(np.nanargmin(flux))
    if min_idx not in indices:
        indices = np.sort(np.append(indices, min_idx))
    
    return time[indices].tolist(), flux[indices].tolist()

def generate_analytical_transit(phase_arr: np.ndarray, depth: float, duration_phase: float):
    """
    Calculates a smooth limb-darkened Mandel & Agol transit curve approximation
    given depth (ΔF/F) and duration in orbital phase.
    """
    transit_flux = np.ones_like(phase_arr, dtype=float)
    half_dur = duration_phase / 2.0
    if half_dur <= 0:
        return transit_flux.tolist()
    
    in_transit = np.abs(phase_arr) <= half_dur
    # Smooth quadratic limb-darkening ingress/bottom/egress shape
    x = np.abs(phase_arr[in_transit]) / half_dur
    # Shape factor gives flat-bottomed or rounded trough
    transit_flux[in_transit] = 1.0 - depth * (1.0 - 0.2 * (x ** 2))
    return transit_flux.tolist()

# ─────────────────────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────────────────────

@app.get("/api/health")
def health_check():
    """Health status and telemetry of the Nova Backend Engine and PostgreSQL / SQLite database."""
    target_count = len(state["target_dict"])
    db_connected = False
    engine_name = "In-Memory / CSV Fallback"
    if HAS_DB:
        try:
            connected, _ = check_connection()
            if connected:
                db_connected = True
                engine_name = "PostgreSQL (Enterprise Connection Pool)" if (DATABASE_URL and DATABASE_URL.startswith("postgresql")) else "SQLite 3 (WAL Mode)"
        except Exception:
            pass
    return {
        "status": "online",
        "service": "ExoFinder Nova Engine",
        "version": "2.1.0",
        "database_engine": engine_name,
        "database_connected": db_connected,
        "targets_indexed": target_count,
        "detrended_curves_count": len(state["available_detrended_ids"]),
        "stacking_ensemble_active": state["model_bundle"] is not None,
        "cnn_deep_learning_active": state["cnn_model"] is not None
    }


@app.get("/api/targets")
def list_targets(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=200, description="Items per page"),
    filter_label: Optional[str] = Query(None, description="Filter by class: confirmed_planet, eclipsing_binary, false_positive"),
    search: Optional[str] = Query(None, description="Search TIC ID substring"),
    min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum confidence filter"),
    min_snr: Optional[float] = Query(None, ge=0.0, description="Minimum SNR filter"),
    only_bookmarked: Optional[bool] = Query(False, description="Filter bookmarked targets only"),
    sort_by: str = Query("confidence", description="Sort by: confidence, transit_snr, bls_period, bls_depth, star_id, is_bookmarked"),
    sort_order: str = Query("desc", description="Sort order: asc or desc"),
    db: Optional[Any] = Depends(get_db)
):
    # Normalize parameters (supports both FastAPI dependency injection and direct python calls)
    page_int = page if isinstance(page, int) else 1
    page_size_int = page_size if isinstance(page_size, int) else 50
    filter_label_str = filter_label if isinstance(filter_label, str) else None
    search_str = search if isinstance(search, str) else None
    min_conf_flt = min_confidence if isinstance(min_confidence, (int, float)) else None
    min_snr_flt = min_snr if isinstance(min_snr, (int, float)) else None
    only_bk_bool = bool(only_bookmarked) if isinstance(only_bookmarked, bool) else False
    sort_by_str = sort_by if isinstance(sort_by, str) else "confidence"
    sort_order_str = sort_order if isinstance(sort_order, str) else "desc"

    if HAS_DB and db is not None:
        try:
            return get_targets_page(
                db=db,
                page=page_int,
                page_size=page_size_int,
                filter_label=filter_label_str,
                search=search_str,
                min_confidence=min_conf_flt,
                min_snr=min_snr_flt,
                only_bookmarked=only_bk_bool,
                sort_by=sort_by_str,
                sort_order=sort_order_str
            )
        except Exception as e:
            print(f"[ExoFinder API] Database query error, using in-memory cache: {e}")

    df = state["targets_df"]
    if df is None:
        raise HTTPException(status_code=503, detail="Targets database not yet initialized.")
        
    filtered = df.copy()
    
    if search_str:
        search_clean = search_str.strip().replace("TIC", "").replace("tic", "").strip()
        filtered = filtered[filtered['star_id'].astype(str).str.contains(search_clean, na=False)]
        
    if filter_label_str and filter_label_str != "all":
        filtered = filtered[filtered['predicted_label'] == filter_label_str]
        
    if min_conf_flt is not None:
        filtered = filtered[filtered['confidence'] >= min_conf_flt]
        
    if min_snr_flt is not None and 'transit_snr' in filtered.columns:
        filtered = filtered[filtered['transit_snr'] >= min_snr_flt]
        
    if only_bk_bool and 'is_bookmarked' in filtered.columns:
        filtered = filtered[filtered['is_bookmarked'] == True]
        
    # Sort
    valid_sorts = {
        "confidence": "confidence",
        "transit_snr": "transit_snr",
        "bls_period": "bls_period",
        "bls_depth": "bls_depth",
        "star_id": "star_id",
        "is_bookmarked": "is_bookmarked"
    }
    col = valid_sorts.get(sort_by_str, "confidence")
    ascending = (sort_order_str.lower() == "asc")
    
    if col in filtered.columns:
        filtered = filtered.sort_values(by=col, ascending=ascending, na_position="last")
        
    total = len(filtered)
    start_idx = (page_int - 1) * page_size_int
    end_idx = start_idx + page_size_int
    page_data = filtered.iloc[start_idx:end_idx]

    
    records = []
    for _, r in page_data.iterrows():
        records.append({
            "star_id": int(r['star_id']),
            "tic_id": f"TIC {int(r['star_id'])}",
            "ground_truth": r.get('label', 'unknown'),
            "predicted_label": r.get('predicted_label', 'unknown'),
            "confidence": float(r.get('confidence', 0.0)),
            "significance": r.get('significance', 'high'),
            "consensus_verdict": r.get('consensus_verdict', 'unanimous'),
            "is_bookmarked": bool(r.get('is_bookmarked', False)),
            "bls_period": float(r.get('bls_period', 0.0)) if not pd.isna(r.get('bls_period')) else None,
            "catalog_period": float(r.get('catalog_period', 0.0)) if not pd.isna(r.get('catalog_period')) else None,
            "bls_depth": float(r.get('bls_depth', 0.0)) if not pd.isna(r.get('bls_depth')) else None,
            "bls_duration": float(r.get('bls_duration', 0.0)) if not pd.isna(r.get('bls_duration')) else None,
            "transit_snr": float(r.get('transit_snr', 0.0)) if not pd.isna(r.get('transit_snr')) else None,
            "has_detrended": bool(r.get('has_detrended', False))
        })
        
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": math.ceil(total / page_size) if total > 0 else 1,
        "targets": records
    }

@app.get("/api/target/{star_id}")
def get_target_details(star_id: int, db: Optional[Any] = Depends(get_db)):
    """Fetches comprehensive astronomical metadata, model predictions, and vetting diagnostics."""
    if HAS_DB and db is not None:
        try:
            db_target = repo_get_target_details(db, star_id)
            if db_target:
                db_target["has_detrended"] = bool(star_id in state["available_detrended_ids"])
                return db_target
        except Exception as e:
            print(f"[ExoFinder API] Database lookup error, using in-memory cache: {e}")

    target = state["target_dict"].get(star_id)
    if not target:
        raise HTTPException(status_code=404, detail=f"Target TIC {star_id} not found.")

        
    # Vetting Checks evaluation
    snr = target.get('transit_snr', 0.0)
    snr = 0.0 if pd.isna(snr) else float(snr)
    
    odd_even_sigma = target.get('odd_even_mismatch_sigma', 0.0)
    odd_even_sigma = 0.0 if pd.isna(odd_even_sigma) else float(odd_even_sigma)
    
    sec_eclipse = target.get('secondary_eclipse_check', 0.0)
    sec_eclipse = 0.0 if pd.isna(sec_eclipse) else float(sec_eclipse)
    
    u_shape = target.get('mean_to_max_depth_ratio', 0.0)
    u_shape = 0.0 if pd.isna(u_shape) else float(u_shape)

    odd_even_ratio = target.get('odd_even_depth_ratio', 1.0)
    odd_even_ratio = 1.0 if pd.isna(odd_even_ratio) else float(odd_even_ratio)

    vetting = {
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

    # Model comparisons
    probabilities = {
        "stacking_ml": {
            "confirmed_planet": float(target.get('prob_confirmed_planet', 0.0)),
            "eclipsing_binary": float(target.get('prob_eclipsing_binary', 0.0)),
            "false_positive": float(target.get('prob_false_positive', 0.0)),
            "predicted_label": target.get('predicted_label', 'unknown'),
            "confidence": float(target.get('confidence', 0.0))
        },
        "cnn_1d": {
            "confirmed_planet": float(target.get('prob_cnn_confirmed_planet', 0.0)),
            "eclipsing_binary": float(target.get('prob_cnn_eclipsing_binary', 0.0)),
            "false_positive": float(target.get('prob_cnn_false_positive', 0.0)),
            "predicted_label": target.get('predicted_label_cnn', 'unknown'),
            "confidence": float(target.get('confidence_cnn', 0.0))
        }
    }

    # Cleaned feature dict
    features = {}
    for col in FEATURE_COLS:
        val = target.get(col, 0.0)
        features[col] = {
            "label": FEATURE_LABELS.get(col, col),
            "value": float(val) if not pd.isna(val) else 0.0
        }

    # Planetary Physics Derivation (Roadmap Item A)
    depth_val = max(1e-6, float(target.get('bls_depth', 0.005) or 0.005))
    period_val = max(0.01, float(target.get('bls_period', 3.35) or 3.35))
    radius_ratio = math.sqrt(depth_val)
    radius_earth = radius_ratio * 109.2
    radius_jupiter = radius_ratio * 9.96

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

    period_years = period_val / 365.25
    semi_major_axis = period_years ** (2.0 / 3.0)
    orbital_velocity = 29.78 / math.sqrt(semi_major_axis) if semi_major_axis > 0 else 0.0
    t_eq = 278.5 / math.sqrt(semi_major_axis) if semi_major_axis > 0 else 300.0

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
        "star_id": star_id,
        "tic_id": f"TIC {star_id}",
        "ground_truth": target.get('label', 'unknown'),
        "is_bookmarked": bool(target.get('is_bookmarked', False)),
        "user_notes": str(target.get('user_notes', '')),
        "consensus_verdict": target.get('consensus_verdict', 'unanimous'),
        "bls_period": float(target.get('bls_period', 0.0)) if not pd.isna(target.get('bls_period')) else 0.0,
        "catalog_period": float(target.get('catalog_period', 0.0)) if not pd.isna(target.get('catalog_period')) else 0.0,
        "recovered_epoch": float(target.get('recovered_epoch', 0.0)),
        "bls_depth": float(target.get('bls_depth', 0.0)),
        "bls_duration": float(target.get('bls_duration', 0.0)),
        "bls_power": float(target.get('bls_power', 0.0)),
        "transit_snr": snr,
        "significance": target.get('significance', 'high'),
        "has_detrended": bool(star_id in state["available_detrended_ids"]),
        "vetting_diagnostics": vetting,
        "model_comparison": probabilities,
        "features": features,
        "planetary_physics": planetary_physics
    }

@app.post("/api/target/{star_id}/bookmark")
def toggle_bookmark(star_id: int, db: Optional[Any] = Depends(get_db)):
    """Toggles bookmark status of a target with atomic PostgreSQL persistence."""
    target = state["target_dict"].get(star_id)
    new_status = None

    if HAS_DB and db is not None:
        try:
            new_status = repo_toggle_bookmark(db, star_id)
        except Exception as e:
            print(f"[ExoFinder API] Database bookmark persistence error: {e}")

    if new_status is None:
        if not target:
            raise HTTPException(status_code=404, detail=f"Target TIC {star_id} not found.")
        new_status = not bool(target.get("is_bookmarked", False))

    if target:
        target["is_bookmarked"] = new_status
    if state["targets_df"] is not None:
        state["targets_df"].loc[state["targets_df"]['star_id'] == star_id, 'is_bookmarked'] = new_status

    return {"star_id": star_id, "is_bookmarked": new_status}

class NotesPayload(BaseModel):
    notes: str

@app.post("/api/target/{star_id}/notes")
def save_target_notes(star_id: int, payload: NotesPayload, db: Optional[Any] = Depends(get_db)):
    """Saves astronomer vetting notes for a target with atomic PostgreSQL persistence."""
    target = state["target_dict"].get(star_id)

    if HAS_DB and db is not None:
        try:
            repo_save_target_notes(db, star_id, payload.notes)
        except Exception as e:
            print(f"[ExoFinder API] Database notes persistence error: {e}")

    if target:
        target["user_notes"] = payload.notes
    if state["targets_df"] is not None:
        state["targets_df"].loc[state["targets_df"]['star_id'] == star_id, 'user_notes'] = payload.notes

    return {"star_id": star_id, "user_notes": payload.notes, "status": "saved"}

@app.get("/api/job/{job_id}")
def check_job_status(job_id: str, db: Optional[Any] = Depends(get_db)):
    """Checks progress and status of an astronomical pipeline job from the database."""
    if HAS_DB and db is not None:
        job = get_pipeline_job(db, job_id)
        if job:
            return job
    raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")


@app.get("/api/lightcurve/{star_id}")
def get_lightcurve(
    star_id: int,
    max_points: int = Query(2500, ge=500, le=10000, description="Max cadences to return for 60fps rendering"),
    phase_window: float = Query(0.2, ge=0.05, le=0.5, description="Phase window for zoom")
):
    """
    Returns precomputed light curves with sub-millisecond static serving.
    Fails over to write-through computation if not yet cached.
    """
    # 1. Check in-memory RAM cache (< 0.1ms)
    if star_id in _LC_RAM_CACHE:
        return Response(
            content=_LC_RAM_CACHE[star_id],
            media_type="application/json",
            headers={"Cache-Control": "public, max-age=86400"}
        )

    # 2. Check disk precomputed cache (< 2ms)
    json_path = os.path.join(PRECOMPUTED_DIR, f"{star_id}.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "rb") as f:
                raw_bytes = f.read()
            _LC_RAM_CACHE[star_id] = raw_bytes
            return Response(
                content=raw_bytes,
                media_type="application/json",
                headers={"Cache-Control": "public, max-age=86400"}
            )
        except Exception:
            pass

    # 3. Dynamic Calculation Fallback with Write-Through Caching
    det_file = os.path.join(DETRENDED_DIR, f"TIC_{star_id}.csv")
    if os.path.exists(det_file):
        try:
            df_det = pd.read_csv(det_file)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read light curve CSV: {e}")
    else:
        try:
            from src.data_loader import load_detrended_star
            df_det = load_detrended_star(star_id)
        except Exception:
            raise HTTPException(status_code=404, detail=f"Light curve data for TIC {star_id} not found in detrended archive.")

    # Extract target metadata for period & epoch
    target = state["target_dict"].get(star_id, {})
    period = target.get('bls_period', 0.0)
    if pd.isna(period) or period <= 0:
        period = target.get('catalog_period', 3.0)
    if pd.isna(period) or period <= 0:
        period = 3.0
    period = float(period)

    epoch = float(target.get('recovered_epoch', 0.0))
    depth = float(target.get('bls_depth', 0.005))
    duration = float(target.get('bls_duration', 0.08))

    time_arr = df_det['time'].values
    detrended_flux = df_det['detrended_flux'].values
    trend_arr = df_det['trend'].values if 'trend' in df_det.columns else np.ones_like(time_arr)
    raw_flux = df_det['flux'].values if 'flux' in df_det.columns else detrended_flux

    # Normalize raw flux for comparison
    raw_med = np.nanmedian(raw_flux)
    raw_norm = (raw_flux / raw_med) if raw_med > 0 else raw_flux

    # Downsampled time series for fast 60fps rendering
    ds_time, ds_detrended = downsample_lttb(time_arr, detrended_flux, max_points=max_points)
    _, ds_raw = downsample_lttb(time_arr, raw_norm, max_points=max_points)

    # Fold phase
    phase_all, flux_folded_all = fold_lightcurve(time_arr, detrended_flux, period, epoch)
    
    # Downsample phase-folded curve around the zoom window
    mask_zoom = (phase_all >= -phase_window) & (phase_all <= phase_window)
    phase_zoom = phase_all[mask_zoom]
    flux_zoom = flux_folded_all[mask_zoom]

    ds_phase, ds_flux_folded = downsample_lttb(phase_zoom, flux_zoom, max_points=1200)

    # Compute median binned profile
    bin_centers, binned_flux = compute_binned_profile(phase_all, flux_folded_all, num_bins=80, window=phase_window)

    # Theoretical transit fit curve over binned phase centers
    duration_phase = duration / period if period > 0 else 0.02
    theoretical_model = generate_analytical_transit(np.array(bin_centers), depth=depth, duration_phase=duration_phase)

    # Generate 1D-CNN Views
    from src.deep_learning import generate_binned_views
    g_view, l_view = generate_binned_views(time_arr, detrended_flux, period, epoch)

    payload = {
        "star_id": star_id,
        "tic_id": f"TIC {star_id}",
        "period": period,
        "epoch": epoch,
        "depth": depth,
        "duration": duration,
        "time_series": {
            "time": ds_time,
            "detrended_flux": ds_detrended,
            "raw_flux": ds_raw
        },
        "phase_folded": {
            "phase": ds_phase,
            "flux": ds_flux_folded,
            "bin_centers": bin_centers,
            "binned_flux": binned_flux,
            "theoretical_curve": theoretical_model
        },
        "cnn_views": {
            "global_view": g_view.tolist(),
            "local_view": l_view.tolist(),
            "global_phase": np.linspace(-0.5, 0.5, len(g_view)).tolist(),
            "local_phase": np.linspace(-0.08, 0.08, len(l_view)).tolist()
        }
    }

    # Write-through to disk and memory cache
    try:
        import orjson
        raw_bytes = orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY)
        _LC_RAM_CACHE[star_id] = raw_bytes
        with open(json_path, "wb") as f:
            f.write(raw_bytes)
        return Response(
            content=raw_bytes,
            media_type="application/json",
            headers={"Cache-Control": "public, max-age=86400"}
        )
    except Exception:
        return payload


class ClassificationInput(BaseModel):
    star_id: Optional[int] = None
    bls_depth: Optional[float] = Field(None, description="Transit depth ΔF/F")
    bls_duration: Optional[float] = Field(None, description="Transit duration in days")
    bls_power: Optional[float] = Field(None, description="BLS periodogram peak power")
    transit_snr: Optional[float] = Field(None, description="Transit signal-to-noise ratio")
    mean_to_max_depth_ratio: Optional[float] = Field(None, description="U-shape vs V-shape depth ratio (0 to 1)")
    in_transit_skew: Optional[float] = Field(None, description="In-transit skewness")
    in_transit_kurtosis: Optional[float] = Field(None, description="In-transit kurtosis")
    flux_kurtosis: Optional[float] = Field(None, description="Global flux kurtosis")
    ingress_egress_symmetry: Optional[float] = Field(None, description="Ingress vs Egress symmetry")
    secondary_eclipse_check: Optional[float] = Field(None, description="Secondary eclipse dip magnitude")
    transit_duty_cycle: Optional[float] = Field(None, description="Transit duty cycle (duration / period)")
    depth_to_noise: Optional[float] = Field(None, description="Depth / Detrended noise ratio")
    odd_even_depth_ratio: Optional[float] = Field(None, description="Odd transit depth / Even transit depth")
    odd_even_mismatch_sigma: Optional[float] = Field(None, description="Statistical significance of depth difference")

@app.post("/api/classify")
def run_classification(input_data: ClassificationInput):
    """
    Executes real-time inference using the Calibrated Stacking ML Ensemble.
    Accepts full feature sets or partial overrides based on an existing star_id.
    """
    bundle = state["model_bundle"]
    if not bundle:
        raise HTTPException(status_code=503, detail="Classification model not available.")

    model = bundle['model']
    scaler = bundle['scaler']
    le = bundle['label_encoder']

    # Base feature values from star_id if provided
    base_values = {}
    if input_data.star_id and input_data.star_id in state["target_dict"]:
        base_values = state["target_dict"][input_data.star_id]

    # Build input feature vector
    input_dict = input_data.model_dump(exclude_unset=True)
    row_features = {}
    for col in FEATURE_COLS:
        if col in input_dict and input_dict[col] is not None:
            row_features[col] = float(input_dict[col])
        elif col in base_values:
            val = base_values[col]
            row_features[col] = float(val) if not pd.isna(val) else 0.0
        else:
            # Default median placeholder
            row_features[col] = 0.0

    X_df = pd.DataFrame([row_features], columns=FEATURE_COLS)
    X_scaled = pd.DataFrame(scaler.transform(X_df), columns=FEATURE_COLS)

    probs = model.predict_proba(X_scaled)[0]
    pred_idx = np.argmax(probs)
    pred_label = le.inverse_transform([pred_idx])[0]
    confidence = float(probs[pred_idx])

    # Class probability breakdown
    prob_dict = {}
    for idx, class_name in enumerate(le.classes_):
        prob_dict[class_name] = float(probs[idx])

    # Significance & Vetting
    snr = row_features.get('transit_snr', 0.0)
    odd_even_sigma = row_features.get('odd_even_mismatch_sigma', 0.0)
    sec_eclipse = row_features.get('secondary_eclipse_check', 0.0)
    u_shape = row_features.get('mean_to_max_depth_ratio', 0.0)

    significance = "high" if snr >= 5.0 else "low_significance"

    # Feature contribution proxies (scaled deviation from population mean)
    contributions = []
    for col in FEATURE_COLS:
        scaled_val = float(X_scaled[col].iloc[0])
        contributions.append({
            "feature": col,
            "label": FEATURE_LABELS.get(col, col),
            "raw_value": row_features[col],
            "scaled_deviation": round(scaled_val, 3),
            "impact": "supports_planet" if (scaled_val > 0 and col in ['transit_snr', 'mean_to_max_depth_ratio', 'bls_power']) else "neutral"
        })

    contributions = sorted(contributions, key=lambda x: abs(x['scaled_deviation']), reverse=True)

    return {
        "predicted_label": pred_label,
        "confidence": confidence,
        "significance": significance,
        "probabilities": prob_dict,
        "vetting_status": {
            "snr_test_passed": bool(snr >= 5.0),
            "odd_even_passed": bool(odd_even_sigma < 3.0),
            "secondary_eclipse_passed": bool(sec_eclipse < 0.005),
            "u_shape_passed": bool(u_shape >= 0.65)
        },
        "feature_contributions": contributions[:6]
    }

@app.get("/api/transit-model")
def get_theoretical_transit_model(
    period: float = Query(3.5, ge=0.1, le=100.0, description="Orbital period in days"),
    radius_ratio: float = Query(0.08, ge=0.001, le=0.5, description="Planetary-to-stellar radius ratio Rp/Rs"),
    semi_major_axis_ratio: float = Query(12.0, ge=1.5, le=100.0, description="Semi-major axis ratio a/Rs"),
    inclination: float = Query(88.5, ge=60.0, le=90.0, description="Orbital inclination in degrees"),
    u1: float = Query(0.3, ge=0.0, le=1.0, description="Linear limb-darkening coefficient"),
    u2: float = Query(0.2, ge=0.0, le=1.0, description="Quadratic limb-darkening coefficient"),
    num_points: int = Query(200, ge=50, le=1000, description="Number of phase curve points")
):
    """
    Calculates an analytical limb-darkened Mandel & Agol transit curve
    for the interactive 3D Keplerian simulator.
    """
    depth = radius_ratio ** 2
    inc_rad = math.radians(inclination)
    # Impact parameter b = (a/Rs) * cos(i)
    b = semi_major_axis_ratio * math.cos(inc_rad)
    
    # Calculate transit duration in phase space
    if b < (1.0 + radius_ratio):
        duration_phase = (1.0 / (math.pi * semi_major_axis_ratio)) * math.sqrt(max(0.0, (1.0 + radius_ratio)**2 - b**2))
    else:
        duration_phase = 0.0

    phase_arr = np.linspace(-0.15, 0.15, num_points)
    flux_arr = np.ones_like(phase_arr)

    if duration_phase > 0:
        half_dur = duration_phase / 2.0
        in_transit = np.abs(phase_arr) <= half_dur
        # Limb darkening profile
        x = np.abs(phase_arr[in_transit]) / half_dur
        mu = np.sqrt(np.maximum(0.0, 1.0 - (x * (1.0 - radius_ratio))**2))
        limb_factor = 1.0 - u1 * (1.0 - mu) - u2 * ((1.0 - mu)**2)
        flux_arr[in_transit] = 1.0 - depth * limb_factor

    return {
        "period": period,
        "radius_ratio": radius_ratio,
        "transit_depth_ppm": round(depth * 1e6, 1),
        "impact_parameter_b": round(b, 3),
        "duration_hours": round(duration_phase * period * 24.0, 2),
        "phase": phase_arr.tolist(),
        "flux": flux_arr.tolist()
    }

@app.get("/api/star/{star_id}/audio")
def get_star_sonification_audio(
    star_id: int,
    base_freq: float = Query(440.0, ge=150.0, le=1200.0, description="Carrier baseline pitch in Hz"),
    duration: float = Query(4.0, ge=1.5, le=15.0, description="Sonified audio duration in seconds"),
    contrast: float = Query(1.0, ge=0.5, le=3.0, description="Pitch dip sensitivity multiplier"),
    format: str = Query("wav", description="'wav' for raw audio file, 'json' for telemetry metadata")
):
    """
    Synthesizes an acoustic transit sonification WAV file from real photometric data.
    The baseline flux is rendered as a clean harmonic carrier tone; as the planet
    eclipses the stellar disk, the frequency dips in proportion to transit depth,
    panning spatially from left (ingress) to right (egress).
    """
    det_file = os.path.join(DETRENDED_DIR, f"TIC_{star_id}.csv")
    if os.path.exists(det_file):
        try:
            df_det = pd.read_csv(det_file)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read light curve CSV: {e}")
    else:
        try:
            from src.data_loader import load_detrended_star
            df_det = load_detrended_star(star_id)
        except Exception:
            raise HTTPException(status_code=404, detail=f"Light curve data for TIC {star_id} not found.")

    target = state["target_dict"].get(star_id, {})
    period = target.get('bls_period', 0.0)
    if pd.isna(period) or period <= 0:
        period = target.get('catalog_period', 3.0)
    if pd.isna(period) or period <= 0:
        period = 3.0
    period = float(period)

    epoch = float(target.get('recovered_epoch', 0.0))
    time_arr = df_det['time'].values
    detrended_flux = df_det['detrended_flux'].values

    # Fold phase and extract transit window [-0.15, 0.15]
    phase_all, flux_folded_all = fold_lightcurve(time_arr, detrended_flux, period, epoch)
    window = 0.15
    mask = (phase_all >= -window) & (phase_all <= window)
    phase_sub = phase_all[mask]
    flux_sub = flux_folded_all[mask]

    # Compute median binned profile for smooth audio curve
    bin_centers, binned_flux = compute_binned_profile(phase_sub, flux_sub, num_bins=120, window=window)
    bin_centers = np.array(bin_centers)
    binned_flux = np.array(binned_flux)

    # Audio synthesis parameters
    sample_rate = 44100
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples, endpoint=False)

    # Map time t -> phase space [-window, window]
    t_phase = np.linspace(-window, window, num_samples)
    interpolated_flux = np.interp(t_phase, bin_centers, binned_flux)

    # Compute pitch dip profile
    baseline_flux = 1.0
    dip = np.maximum(0.0, baseline_flux - interpolated_flux)
    max_dip = float(np.max(dip)) if len(dip) > 0 else 0.005
    if max_dip < 1e-5:
        max_dip = 0.005

    relative_dip = dip / max_dip
    # Smooth frequency curve with harmonic modulation
    pitch_drop_ratio = min(0.40, 0.28 * contrast)
    freq_curve = base_freq * (1.0 - pitch_drop_ratio * relative_dip)

    # Continuous phase accumulator (avoids clicks from frequency jumps)
    audio_phase = 2.0 * np.pi * np.cumsum(freq_curve) / sample_rate

    # Rich harmonic timbre: Sine fundamental (82%) + Warm 2nd harmonic (14%) + 3rd harmonic (4%)
    waveform = (
        0.82 * np.sin(audio_phase) +
        0.14 * np.sin(2.0 * audio_phase) +
        0.04 * np.sin(3.0 * audio_phase)
    )

    # Smooth attack and release envelope (60ms fade at boundaries)
    fade_len = int(0.06 * sample_rate)
    envelope = np.ones(num_samples)
    if fade_len > 0 and 2 * fade_len < num_samples:
        fade_in = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, fade_len)))
        fade_out = 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, fade_len)))
        envelope[:fade_len] = fade_in
        envelope[-fade_len:] = fade_out

    # Spatial Stereo Panning (Left -> Center -> Right across transit)
    pan = np.linspace(0.0, 1.0, num_samples)
    gain_l = np.cos(pan * (np.pi / 2.0))
    gain_r = np.sin(pan * (np.pi / 2.0))

    scale = 32767.0 * 0.75
    left_channel = (waveform * envelope * gain_l * scale).astype(np.int16)
    right_channel = (waveform * envelope * gain_r * scale).astype(np.int16)
    stereo_data = np.column_stack((left_channel, right_channel))

    min_freq = float(np.min(freq_curve))
    max_drop = float(base_freq - min_freq)

    if format.lower() == "json":
        return {
            "star_id": star_id,
            "tic_id": f"TIC {star_id}",
            "period": period,
            "epoch": epoch,
            "base_frequency_hz": base_freq,
            "minimum_frequency_hz": round(min_freq, 1),
            "max_pitch_drop_hz": round(max_drop, 1),
            "duration_seconds": duration,
            "transit_depth_ppm": round(max_dip * 1e6, 1),
            "audio_download_url": f"/api/star/{star_id}/audio?format=wav&base_freq={base_freq}&duration={duration}&contrast={contrast}"
        }

    # Generate WAV buffer
    wav_io = io.BytesIO()
    wavfile.write(wav_io, sample_rate, stereo_data)
    wav_bytes = wav_io.getvalue()

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "Content-Disposition": f'attachment; filename="TIC_{star_id}_transit_sonification.wav"',
            "Content-Length": str(len(wav_bytes)),
            "Cache-Control": "public, max-age=3600"
        }
    )

