import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from astropy.timeseries import BoxLeastSquares

def get_binned_profile(phase, flux, num_bins=80):
    """
    Bins the folded light curve between phases -0.15 and 0.15.
    
    Parameters:
    -----------
    phase : np.ndarray
        Folded phases (typically between -0.5 and 0.5).
    flux : np.ndarray
        Normalized flux values.
    num_bins : int
        Number of bins.
        
    Returns:
    --------
    bin_centers : np.ndarray
        Centers of the bins.
    binned_flux : np.ndarray
        Median flux in each bin.
    """
    bins = np.linspace(-0.15, 0.15, num_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    binned_flux = []
    for i in range(len(bins)-1):
        mask = (phase >= bins[i]) & (phase < bins[i+1])
        binned_flux.append(np.median(flux[mask]) if np.any(mask) else 1.0)
    return bin_centers, np.array(binned_flux)

def extract_features_for_star(star_id, label, cat_period, bls_row, detrended_df, raw_df):
    """
    Extracts a feature vector for a single star based on its detrended light curve
    and BoxLeastSquares (BLS) period search results.
    
    Parameters:
    -----------
    star_id : int
        TIC ID of the star.
    label : str
        The catalog class label.
    cat_period : float
        The expected catalog period (from labels.csv).
    bls_row : dict/pd.Series
        Row containing recovered BLS parameters (period, epoch, depth, duration, power).
    detrended_df : pd.DataFrame
        DataFrame of detrended light curve with columns ['time', 'detrended_flux'].
    raw_df : pd.DataFrame
        DataFrame of raw light curve with columns ['time', 'flux'].
        
    Returns:
    --------
    dict
        Feature dictionary for the star.
    """
    time = detrended_df['time'].values
    flux = detrended_df['detrended_flux'].values
    
    raw_flux = raw_df['flux'].values
    
    # Retrieve BLS parameters
    period = float(bls_row['recovered_period'])
    epoch = float(bls_row['recovered_epoch'])
    depth = float(bls_row['recovered_depth'])
    duration = float(bls_row['recovered_duration'])
    bls_power = float(bls_row['bls_power'])
    
    # If BLS failed, fill features with defaults
    if np.isnan(period) or period <= 0:
        return {
            'star_id': star_id,
            'label': label,
            'catalog_period': cat_period,
            'bls_period': 0.0,
            'bls_depth': 0.0,
            'bls_duration': 0.0,
            'bls_power': 0.0,
            'transit_snr': 0.0,
            'secondary_depth_phase_05': 0.0,
            'secondary_to_primary_ratio': 0.0,
            'secondary_significance_sigma': 0.0,
            'odd_even_depth_ratio': 1.0,
            'odd_even_mismatch_sigma': 0.0,
            'odd_even_diff_2p': 0.0,
            'kepler_duration_ratio': 0.0,
            'positive_to_negative_ratio': 1.0,
            'bottom_flatness_ratio': 1.0,
            'mean_to_max_depth_ratio': 0.0,
            'in_transit_skew': 0.0,
            'in_transit_kurtosis': 0.0,
            'flux_kurtosis': float(kurtosis(flux)) if len(flux) > 3 and np.var(flux) > 1e-12 else 0.0,
            'ingress_egress_symmetry': 0.0,
            'transit_duty_cycle': 0.0,
            'depth_to_noise': 0.0,
            'transit_coverage_fraction': 0.0,
            'std_raw': np.std(raw_flux) if len(raw_flux) > 0 else 0.0,
            'std_detrended': np.std(flux) if len(flux) > 0 else 0.0,
            'transit_count': 0.0
        }
        
    # Phase fold into [-0.5, 0.5] range
    phase = ((time - epoch) % period) / period
    phase = np.where(phase > 0.5, phase - 1.0, phase)
    
    half_dur_phase = max(0.003, min(0.15, (duration / 2.0) / period))
    in_transit_mask = np.abs(phase) <= half_dur_phase
    out_transit_mask = (np.abs(phase) >= 1.5 * half_dur_phase) & (np.abs(phase) <= 0.45)
    sec_transit_mask = np.abs(np.abs(phase) - 0.5) <= half_dur_phase
    
    # 1. Out-Of-Transit (OOT) scatter
    flux_oot = flux[out_transit_mask]
    if len(flux_oot) > 10:
        std_oot = float(np.std(flux_oot))
    else:
        std_oot = float(np.std(flux)) if len(flux) > 0 else 1e-5
    std_oot = max(std_oot, 1e-6)
    
    # 2. True Integrated Transit SNR
    n_in_transit = int(np.sum(in_transit_mask))
    prim_drop = max(0.0, 1.0 - float(np.median(flux[in_transit_mask]))) if n_in_transit > 2 else depth
    transit_snr = float((prim_drop / std_oot) * np.sqrt(max(1, n_in_transit)))
    
    # 3. Secondary Eclipse Analysis at Phase 0.5 (Gold Standard for EB Identification)
    n_in_sec = int(np.sum(sec_transit_mask))
    sec_drop = max(0.0, 1.0 - float(np.median(flux[sec_transit_mask]))) if n_in_sec > 2 else 0.0
    sec_to_prim = float(sec_drop / max(prim_drop, 1e-5))
    sec_sig_sigma = float((sec_drop / std_oot) * np.sqrt(max(1, n_in_sec)))
    
    # 4. Odd vs Even Transit Depth Analysis at 1P
    cycle = np.round((time - epoch) / period).astype(int)
    even_transit_mask = in_transit_mask & (cycle % 2 == 0)
    odd_transit_mask = in_transit_mask & (cycle % 2 != 0)
    
    even_flux = flux[even_transit_mask]
    odd_flux = flux[odd_transit_mask]
    
    if len(even_flux) >= 3:
        depth_even = max(0.0, 1.0 - float(np.median(even_flux)))
        std_even = float(np.std(even_flux)) if len(even_flux) > 1 else std_oot
    else:
        depth_even = prim_drop
        std_even = std_oot
        
    if len(odd_flux) >= 3:
        depth_odd = max(0.0, 1.0 - float(np.median(odd_flux)))
        std_odd = float(np.std(odd_flux)) if len(odd_flux) > 1 else std_oot
    else:
        depth_odd = prim_drop
        std_odd = std_oot
        
    max_d = max(depth_odd, depth_even)
    min_d = min(depth_odd, depth_even)
    odd_even_depth_ratio = float((min_d / max_d) if max_d > 1e-6 else 1.0)
    
    err_denom = np.sqrt((std_odd**2 / max(1, len(odd_flux))) + (std_even**2 / max(1, len(even_flux))))
    odd_even_mismatch_sigma = float(np.abs(depth_odd - depth_even) / max(err_denom, 1e-6))
    
    # 5. Odd vs Even Analysis at 2P (Exposes Half-Period Alias EBs)
    p2 = 2.0 * period
    phase2 = ((time - epoch) % p2) / p2
    phase2 = np.where(phase2 > 0.5, phase2 - 1.0, phase2)
    half_dur2 = max(0.003, min(0.1, (duration / 2.0) / p2))
    
    mask_ev1 = np.abs(phase2) <= half_dur2
    mask_ev2 = np.abs(np.abs(phase2) - 0.5) <= half_dur2
    d1 = max(0.0, 1.0 - float(np.median(flux[mask_ev1]))) if np.sum(mask_ev1) > 2 else prim_drop
    d2 = max(0.0, 1.0 - float(np.median(flux[mask_ev2]))) if np.sum(mask_ev2) > 2 else 0.0
    odd_even_diff_2p = float(abs(d1 - d2) / max(1e-5, max(d1, d2)))
    
    # 6. Keplerian Duration Scaling & Duty Cycle
    kepler_duration_ratio = float(duration / (period ** (1.0 / 3.0))) if period > 0 else 0.0
    transit_duty_cycle = float(duration / period) if period > 0 else 0.0
    
    # 7. Positive vs Negative Asymmetry (Stellar Flares / Systematics vs Transits)
    max_pos = max(0.0, float(np.max(flux) - 1.0)) if len(flux) > 0 else 0.0
    max_neg = max(0.0, float(1.0 - np.min(flux))) if len(flux) > 0 else 1e-6
    positive_to_negative_ratio = float(max_pos / max(max_neg, 1e-6))
    
    # 8. Bottom Flatness Ratio (U-Shape vs V-Shape)
    deep_mask = in_transit_mask & ((1.0 - flux) >= 0.5 * prim_drop)
    bottom_std = float(np.std(flux[deep_mask])) if np.sum(deep_mask) > 3 else std_oot
    bottom_flatness_ratio = float(bottom_std / std_oot)
    
    # 9. Extract Binned Profile Shape Metrics
    bin_centers, binned_flux = get_binned_profile(phase, flux, num_bins=80)
    binned_in_transit_mask = np.abs(bin_centers) <= half_dur_phase
    binned_flux_in = binned_flux[binned_in_transit_mask]
    
    if len(binned_flux_in) > 3 and np.var(binned_flux_in) > 1e-12:
        depths = 1.0 - binned_flux_in
        max_depth = np.max(depths)
        mean_depth = np.mean(depths)
        mean_to_max_ratio = float(mean_depth / max_depth if max_depth > 0 else 0.0)
        in_transit_skew = float(skew(binned_flux_in))
        in_transit_kurt = float(kurtosis(binned_flux_in))
        if np.isnan(mean_to_max_ratio): mean_to_max_ratio = 0.0
        if np.isnan(in_transit_skew): in_transit_skew = 0.0
        if np.isnan(in_transit_kurt): in_transit_kurt = 0.0
    else:
        mean_to_max_ratio = 0.0
        in_transit_skew = 0.0
        in_transit_kurt = 0.0
        
    # 10. Ingress/Egress Symmetry
    left_half_flux = flux[(phase < 0) & in_transit_mask]
    right_half_flux = flux[(phase > 0) & in_transit_mask]
    if len(left_half_flux) < 3 or len(right_half_flux) < 3 or np.var(left_half_flux) < 1e-12 or np.var(right_half_flux) < 1e-12:
        ingress_egress_symmetry = 0.0
    else:
        skew_left = skew(left_half_flux)
        skew_right = skew(right_half_flux)
        if np.isnan(skew_left) or np.isnan(skew_right):
            ingress_egress_symmetry = 0.0
        else:
            ingress_egress_symmetry = float(abs(skew_left) - abs(skew_right))
            
    # 11. Transit Counts & Coverage Fraction
    total_time = float(np.max(time) - np.min(time)) if len(time) > 0 else 0.0
    expected_transits = total_time / period if period > 0 else 1.0
    
    # Count how many individual transit epochs have >= 2 observed points
    observed_epochs = 0
    cycle_min = int(np.floor((np.min(time) - epoch) / period))
    cycle_max = int(np.ceil((np.max(time) - epoch) / period))
    for c in range(cycle_min, cycle_max + 1):
        t_center = epoch + c * period
        pts_in_epoch = np.sum(np.abs(time - t_center) <= (duration / 2.0))
        if pts_in_epoch >= 2:
            observed_epochs += 1
    transit_coverage_fraction = float(observed_epochs / max(1.0, expected_transits))
    
    std_raw = float(np.std(raw_flux)) if len(raw_flux) > 0 else 0.0
    std_detrend = float(np.std(flux)) if len(flux) > 0 else 0.0
    flux_kurt = float(kurtosis(flux)) if len(flux) > 3 and np.var(flux) > 1e-12 else 0.0
    if np.isnan(flux_kurt): flux_kurt = 0.0

    return {
        'star_id': star_id,
        'label': label,
        'catalog_period': cat_period,
        'bls_period': period,
        'bls_depth': depth,
        'bls_duration': duration,
        'bls_power': bls_power,
        'transit_snr': transit_snr,
        'secondary_depth_phase_05': sec_drop,
        'secondary_to_primary_ratio': sec_to_prim,
        'secondary_significance_sigma': sec_sig_sigma,
        'odd_even_depth_ratio': odd_even_depth_ratio,
        'odd_even_mismatch_sigma': odd_even_mismatch_sigma,
        'odd_even_diff_2p': odd_even_diff_2p,
        'kepler_duration_ratio': kepler_duration_ratio,
        'positive_to_negative_ratio': positive_to_negative_ratio,
        'bottom_flatness_ratio': bottom_flatness_ratio,
        'mean_to_max_depth_ratio': mean_to_max_ratio,
        'in_transit_skew': in_transit_skew,
        'in_transit_kurtosis': in_transit_kurt,
        'flux_kurtosis': flux_kurt,
        'ingress_egress_symmetry': ingress_egress_symmetry,
        'transit_duty_cycle': transit_duty_cycle,
        'depth_to_noise': depth / max(std_detrend, 1e-6),
        'transit_coverage_fraction': transit_coverage_fraction,
        'std_raw': std_raw,
        'std_detrended': std_detrend,
        'transit_count': float(observed_epochs)
    }


def extract_features_all(labels_path="data/labels.csv", bls_results_path="outputs/period_search_results.csv",
                         raw_dir="data/raw", detrended_dir="data/detrended", output_csv="outputs/features.csv"):
    """
    Loads all datasets, extracts features for each star, and saves them to a CSV file.
    """
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    if not os.path.exists(bls_results_path):
        raise FileNotFoundError(f"BLS search results file not found: {bls_results_path}")
        
    labels_df = pd.read_csv(labels_path)
    bls_df = pd.read_csv(bls_results_path)
    
    features_list = []
    failed_list = []
    
    print(f"Extracting features for {len(labels_df)} targets...")
    
    for idx, row in labels_df.iterrows():
        star_id = int(row['star_id'])
        label = row['label']
        cat_period = float(row['catalog_period'])
        
        # Load BLS row
        bls_sub = bls_df[bls_df['star_id'] == star_id]
        if len(bls_sub) == 0:
            print(f"Warning: No BLS results for TIC {star_id}")
            failed_list.append({'star_id': star_id, 'reason': 'No BLS results'})
            continue
        bls_row = bls_sub.iloc[0]
        
        # Load light curve files
        raw_file = os.path.join(raw_dir, f"TIC_{star_id}.csv")
        det_file = os.path.join(detrended_dir, f"TIC_{star_id}.csv")
        
        if not (os.path.exists(raw_file) and os.path.exists(det_file)):
            print(f"Warning: Missing files for TIC {star_id}")
            failed_list.append({'star_id': star_id, 'reason': 'Missing raw/detrended CSV files'})
            continue
            
        try:
            raw_df = pd.read_csv(raw_file)
            det_df = pd.read_csv(det_file)
            
            feat = extract_features_for_star(
                star_id=star_id,
                label=label,
                cat_period=cat_period,
                bls_row=bls_row,
                detrended_df=det_df,
                raw_df=raw_df
            )
            features_list.append(feat)
            
        except Exception as e:
            print(f"Error extracting features for TIC {star_id}: {e}")
            failed_list.append({'star_id': star_id, 'reason': str(e)})
            
    # Save features
    features_df = pd.DataFrame(features_list)
    
    # Sanity check: Ensure no missing values (NaNs) in the feature matrix
    # Loop through all columns that have NaN values; if numeric, impute with median, else fallback to 0.0
    nan_cols = features_df.columns[features_df.isna().any()].tolist()
    if nan_cols:
        print(f"Warning: Found NaN values in columns: {nan_cols}. Imputing with median/fallback.")
        for col in nan_cols:
            if pd.api.types.is_numeric_dtype(features_df[col]):
                col_median = features_df[col].median()
                if pd.isna(col_median):
                    features_df[col] = features_df[col].fillna(0.0)
                else:
                    features_df[col] = features_df[col].fillna(col_median)
            else:
                features_df[col] = features_df[col].fillna(0.0)
        
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    features_df.to_csv(output_csv, index=False)
    print(f"Features extracted successfully! Saved to {output_csv} (Rows: {len(features_df)})")
    
    # Save failures log
    if failed_list:
        failed_df = pd.DataFrame(failed_list)
        failed_csv = os.path.join(os.path.dirname(output_csv), "feature_failures.csv")
        failed_df.to_csv(failed_csv, index=False)
        print(f"Logged {len(failed_list)} failures to {failed_csv}")
        
    return features_df
