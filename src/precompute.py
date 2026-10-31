"""
ExoFinder High-Performance Batch Precomputation Engine
======================================================
Pre-calculates downsampled light curves, folded profiles, theoretical overlays,
and 1D-CNN views for stars and stores them into ready-to-serve JSON payloads in data/precomputed/.
Enables < 2ms zero-computation runtime responses.
"""

import os
import sys
import orjson
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.deep_learning import generate_binned_views

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
DETRENDED_DIR = os.path.join(DATA_DIR, "detrended")
PRECOMPUTED_DIR = os.path.join(DATA_DIR, "precomputed")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")


def downsample_lttb(time, flux, max_points=2500):
    """Largest-Triangle-Three-Buckets (LTTB) downsampling."""
    n = len(time)
    if n <= max_points or max_points < 3:
        return time.tolist(), flux.tolist()

    sampled_time = [time[0]]
    sampled_flux = [flux[0]]

    bucket_size = (n - 2) / (max_points - 2)
    a = 0

    for i in range(max_points - 2):
        range_start = int((i + 1) * bucket_size) + 1
        range_end = int((i + 2) * bucket_size) + 1
        range_end = min(range_end, n)

        next_bucket_time = np.mean(time[range_start:range_end])
        next_bucket_flux = np.mean(flux[range_start:range_end])

        cur_start = int(i * bucket_size) + 1
        cur_end = int((i + 1) * bucket_size) + 1
        cur_end = min(cur_end, n)

        t_a, f_a = time[a], flux[a]

        c_times = time[cur_start:cur_end]
        c_fluxes = flux[cur_start:cur_end]

        areas = 0.5 * np.abs(
            (t_a - next_bucket_time) * (c_fluxes - f_a) -
            (t_a - c_times) * (next_bucket_flux - f_a)
        )

        max_idx = np.argmax(areas)
        best_point_idx = cur_start + max_idx

        sampled_time.append(time[best_point_idx])
        sampled_flux.append(flux[best_point_idx])
        a = best_point_idx

    sampled_time.append(time[-1])
    sampled_flux.append(flux[-1])

    return sampled_time, sampled_flux


def fold_lightcurve(time, flux, period, epoch):
    phase = ((time - epoch) % period) / period
    phase[phase > 0.5] -= 1.0
    sort_idx = np.argsort(phase)
    return phase[sort_idx], flux[sort_idx]


def compute_binned_profile(phase, flux, num_bins=80, window=0.2):
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


def generate_analytical_transit(phase, depth=0.01, duration_phase=0.03):
    model = np.ones_like(phase, dtype=float)
    in_transit = np.abs(phase) < (duration_phase / 2.0)
    model[in_transit] = 1.0 - depth
    return model.tolist()


def precompute_star(star_id: int, period: float, epoch: float, depth: float, duration: float, max_points: int = 2500, overwrite: bool = False) -> bool:
    """Precomputes and writes static binary JSON payload for a single star."""
    det_file = os.path.join(DETRENDED_DIR, f"TIC_{star_id}.csv")
    if not os.path.exists(det_file):
        return False

    out_path = os.path.join(PRECOMPUTED_DIR, f"{star_id}.json")
    if not overwrite and os.path.exists(out_path) and os.path.getsize(out_path) > 100:
        return True

    try:
        df_det = pd.read_csv(det_file)
        time_arr = df_det['time'].values
        detrended_flux = df_det['detrended_flux'].values
        raw_flux = df_det['flux'].values if 'flux' in df_det.columns else detrended_flux

        raw_med = np.nanmedian(raw_flux)
        raw_norm = (raw_flux / raw_med) if raw_med > 0 else raw_flux

        # Downsample
        ds_time, ds_detrended = downsample_lttb(time_arr, detrended_flux, max_points=max_points)
        _, ds_raw = downsample_lttb(time_arr, raw_norm, max_points=max_points)

        # Fold
        phase_all, flux_folded_all = fold_lightcurve(time_arr, detrended_flux, period, epoch)
        mask_zoom = (phase_all >= -0.2) & (phase_all <= 0.2)
        phase_zoom = phase_all[mask_zoom]
        flux_zoom = flux_folded_all[mask_zoom]
        ds_phase, ds_flux_folded = downsample_lttb(phase_zoom, flux_zoom, max_points=1200)

        # Binned profile
        bin_centers, binned_flux = compute_binned_profile(phase_all, flux_folded_all, num_bins=80, window=0.2)

        # Theoretical model
        duration_phase = duration / period if period > 0 else 0.02
        theoretical_model = generate_analytical_transit(np.array(bin_centers), depth=depth, duration_phase=duration_phase)

        # 1D-CNN views
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

        os.makedirs(PRECOMPUTED_DIR, exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY))
        return True


    except Exception as e:
        print(f"Error precomputing TIC {star_id}: {e}")
        return False


def precompute_all_stars(limit: int = None, overwrite: bool = False):
    """Precomputes catalog stars for instant 0ms responses."""
    print("=" * 60)
    print("🪐 ExoFinder: Batch Precomputation Engine Starting...")
    print("=" * 60)

    predictions_path = os.path.join(OUTPUTS_DIR, "predictions.csv")
    if not os.path.exists(predictions_path):
        print(f"Predictions catalog not found at {predictions_path}")
        return

    df_pred = pd.read_csv(predictions_path)
    bls_path = os.path.join(OUTPUTS_DIR, "period_search_results.csv")
    if os.path.exists(bls_path):
        df_bls = pd.read_csv(bls_path)
        if 'recovered_epoch' in df_bls.columns:
            epoch_map = dict(zip(df_bls['star_id'], df_bls['recovered_epoch']))
            df_pred['recovered_epoch'] = df_pred['star_id'].map(epoch_map).fillna(0.0)

    # Prioritize confirmed planets and high SNR targets
    if 'transit_snr' in df_pred.columns:
        df_pred = df_pred.sort_values(by='transit_snr', ascending=False)

    total_stars = len(df_pred) if limit is None else min(limit, len(df_pred))
    target_subset = df_pred.head(total_stars)
    print(f"Precomputing {total_stars} targets (Catalog total: {len(df_pred)})...")

    success = 0
    for idx, (_, row) in enumerate(target_subset.iterrows()):
        sid = int(row['star_id'])
        period = float(row.get('bls_period', row.get('catalog_period', 3.35)))
        epoch = float(row.get('recovered_epoch', 0.0))
        depth = float(row.get('bls_depth', 0.005))
        duration = float(row.get('bls_duration', 0.09))

        ok = precompute_star(sid, period, epoch, depth, duration, overwrite=overwrite)
        if ok:
            success += 1
            if success % 100 == 0 or success == total_stars:
                print(f"  ✓ [{success}/{total_stars}] Precomputed TIC {sid}")

    print(f"\n🎉 Successfully precomputed {success} star light curves in {PRECOMPUTED_DIR}!")


if __name__ == "__main__":
    precompute_all_stars()
